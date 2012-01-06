/**
 * Copyright (c) 2011, CESNET z.s.p.o
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, git OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "view.h"
#include "image.h"
#include "util.h"
#include "../../libgpujpeg/gpujpeg.h"
#include <pthread.h>
#include <cuda.h>

//#define TEST_OPENGL_INTEROP_MULTI_THREAD

/**
 * Transfer type
 */
enum transfer_type {
    TRANSFER_HOST = 0,
    TRANSFER_DEVICE = 1
};

/**
 * Application structure that hold all common variables
 */
struct application {
    // Size
    int width;
    int height;
    // View structure
    struct view* view;
    // Image structure
    struct image* image;
    // Mutex
    pthread_mutex_t mutex;
    // CUDA context
    volatile CUcontext cuda_context;
    // Tranfer type
    enum transfer_type transfer_type;
    // Flag if image thread should quit
    int quit;
    
    // OpenGL parameters
    unsigned int texture_id;
    unsigned int pbo_id;
    // CUDA parameters
    struct cudaGraphicsResource* pbo_res;
    
    // JPEG
    struct gpujpeg_encoder* encoder;
    struct gpujpeg_decoder* decoder;
    struct gpujpeg_decoder_output decoder_output;
};

/**
 * Thread that shows window and in loop renders current image.
 * Before rendering it calls on_render callback.
 */
void*
thread_view_run(void* arg)
{
    struct application* app = (struct application*)arg;
    
    // Run through GLX
    view_glx(app->view);
    
    // Run through GLUT
    //view_glut(app->view);
    
    // Quit image thread
    app->quit = 1;
    
    return 0;
}

/**
 * On init callback for view.
 */
void
view_on_init(void* param)
{
    struct application* app = (struct application*)param;
    
    // Create image (image should be created after OpenGL is initialized)
    app->image = image_create(app->width, app->height);
    assert(app->image != NULL);
    
    // Create PBO
    glGenBuffers(1, &app->pbo_id);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, app->pbo_id);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, app->width * app->height * 3 * sizeof(uint8_t), NULL, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    
    // Create CUDA PBO Resource
    cudaGraphicsGLRegisterBuffer(&app->pbo_res, app->pbo_id, cudaGraphicsMapFlagsNone);
    cudaCheckError(); 
    
    // Create texture
    glGenTextures(1, &app->texture_id);
    glBindTexture(GL_TEXTURE_2D, app->texture_id);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    
    // Init JPEG params
    struct gpujpeg_parameters param_coder;
    gpujpeg_set_default_parameters(&param_coder);
    struct gpujpeg_image_parameters param_image;
    gpujpeg_image_set_default_parameters(&param_image);
    param_image.width = app->width;
    param_image.height = app->height;
    
    // Init JPEG encoder and decoder
    app->encoder = gpujpeg_encoder_create(&param_coder, &param_image);
    assert(app->encoder != NULL);
    app->decoder = gpujpeg_decoder_create();
    assert(app->decoder != NULL);
    assert(gpujpeg_decoder_init(app->decoder, &param_coder, &param_image) == 0);
    
    // Init JPEG decoder output
    if ( app->transfer_type == TRANSFER_DEVICE ) {
        app->decoder_output.type = GPUJPEG_DECODER_OUTPUT_OPENGL_TEXTURE;
        app->decoder_output.texture_pbo_resource = app->pbo_res;
    } else {
        gpujpeg_decoder_output_set_default(&app->decoder_output);
    }
}

/**
 * Generate new image, encode it with JPEG and decode it into OpenGL texture
 * 
 * @param app
 * @return void
 */
void
image_generate(struct application* app)
{
    static int max = 100;
    static int change = 10;
    
    printf("Image: ImageRender Started\n");
        
    TIMER_INIT();
    TIMER_START();
    
    // Render new image
    max += change;
    if ( max < 0 || max > 255 ) {
        change = -change;
        max += change;
    }
    image_render(app->image, max);
        
    TIMER_STOP_PRINT("Image: ImageRendered");
    TIMER_START();
    
    // Encode image
    uint8_t* image_compressed = NULL;
    int image_compressed_size = 0;
    // Copy data to host memory
    cudaMemcpy(app->image->data, app->image->d_data, app->width * app->height * 3 * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    assert(gpujpeg_encoder_encode(app->encoder, app->image->data, &image_compressed, &image_compressed_size) == 0);
    
    TIMER_STOP_PRINT("Image: ImageEncode");
    TIMER_START();
    
    // Decode image
    gpujpeg_decoder_decode(app->decoder, image_compressed, image_compressed_size, &app->decoder_output);
    
    TIMER_STOP_PRINT("Image: ImageDecode");
}

/**
 * On render callback for view. Check if cuda_context is available (means new 
 * image is available) and if it is, load new image to view.
 */
void
view_on_render(void* param)
{    
    struct application* app = (struct application*)param;
    
#ifdef TEST_OPENGL_INTEROP_MULTI_THREAD
    // If CUDA context is not filled, it means new image is not ready
    if ( app->cuda_context == NULL )
        return;
#endif
    TIMER_INIT();
    TIMER_START();
        
    pthread_mutex_lock(&app->mutex);
        
#ifdef TEST_OPENGL_INTEROP_MULTI_THREAD
    assert(cuCtxPushCurrent(app->cuda_context) == CUDA_SUCCESS);
#else
    image_generate(app);
#endif

    if ( app->decoder_output.type == GPUJPEG_DECODER_OUTPUT_OPENGL_TEXTURE ) {
        // Set data to texture from pbo
        glBindTexture(GL_TEXTURE_2D, app->texture_id);
        glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, app->pbo_id);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, app->width, app->height, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
        glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
    } else {
        // Set texture data from host memory
        glBindTexture(GL_TEXTURE_2D, app->texture_id);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, app->width, app->height, 0, GL_RGB, GL_UNSIGNED_BYTE, app->decoder_output.data);
    }
    glFinish();
    
    view_set_texture(app->view, app->texture_id);
        
#ifdef TEST_OPENGL_INTEROP_MULTI_THREAD
    assert(cuCtxPopCurrent((CUcontext*)&app->cuda_context) == CUDA_SUCCESS);
#endif
        
    app->cuda_context = NULL;
    
    pthread_mutex_unlock(&app->mutex);
        
    TIMER_STOP_PRINT("View: ImageLoad");
}

/**
 * Thread that in loop generates images. For every generated image it detaches
 * CUDA context and wait until view thread load that image by detached CUDA context
 */
void*
thread_image_run(void* arg)
{
    struct application* app = (struct application*)arg;

    // Wait until work thread is ready to render image
    while ( app->image == NULL ) {
        usleep(1000);
        continue;
    }
    
    // Generated image in loop until view thread quit
    while ( app->quit == 0 ) {
        usleep(30000);
        
        image_generate(app);
        
        pthread_mutex_lock(&app->mutex);
        
        CUcontext cuda_context;
        assert(cuCtxPopCurrent(&cuda_context) == CUDA_SUCCESS);
        app->cuda_context = cuda_context;
        
        pthread_mutex_unlock(&app->mutex);
        
        while ( app->quit == 0 ) {
            if ( app->cuda_context == NULL )
                break;
            usleep(1000);
        }
        
        assert(cuCtxPushCurrent(cuda_context) == CUDA_SUCCESS);
    }
    
    return 0;
}

int
main(int argc, char **argv)
{    
    // Create application
    struct application app;
    app.width = 1920;
    app.height = 1080;
    app.view = view_create(app.width, app.height, 1280, 720);
    assert(app.view != NULL);
    app.image = NULL;
    assert(pthread_mutex_init(&app.mutex, NULL) == 0);
    app.cuda_context = NULL;
    app.transfer_type = TRANSFER_DEVICE;
    app.quit = 0;
    
    // Set view callbacks
    view_set_on_init(app.view, &view_on_init, (void*)&app);
    view_set_on_render(app.view, &view_on_render, (void*)&app);
    
    // Create threads
    pthread_t thread_view;
    pthread_create(&thread_view, NULL, thread_view_run, (void*)&app);
#ifdef TEST_OPENGL_INTEROP_MULTI_THREAD
    pthread_t thread_image;
    pthread_create(&thread_image, NULL, thread_image_run, (void*)&app);
#endif
    
    // Wait for threads to exit and check result status
    void* result;
    pthread_join(thread_view, &result);
    assert(result == 0);
#ifdef TEST_OPENGL_INTEROP_MULTI_THREAD
    pthread_join(thread_image, &result);
    assert(result == 0);
#endif
    
    // Destroy application
    image_destroy(app.image);
    view_destroy(app.view);
    pthread_mutex_destroy(&app.mutex);
    gpujpeg_encoder_destroy(app.encoder);
    gpujpeg_decoder_destroy(app.decoder);
    
    return 0;
}
