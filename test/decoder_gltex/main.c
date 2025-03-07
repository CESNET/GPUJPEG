#include <libgpujpeg/gpujpeg.h>
#include <stdio.h>
#include "gpujpeg_reformat.h"
#include <GL/glew.h>
#include <GL/glut.h>

int g_texture_id;
int g_width;
int g_height;

void glutOnDisplay(void);
void glutOnIdle(void);
void glutOnKeyboard(unsigned char key, int x, int y);
void glutOnReshape(int width, int height);

int main(int argc, char *argv[])
{
    if ( argc < 2 ) {
        fprintf(stderr, "Please supply image filename!\n");
        return -1;
    }
    const char * input_filename = argv[1];

    // Init OpenGL
    glutInit(&argc, argv);
    glutInitWindowSize(640, 480);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
    glutCreateWindow(input_filename);
    glutDisplayFunc(glutOnDisplay);
    glutIdleFunc(glutOnIdle);
    glutKeyboardFunc(glutOnKeyboard);
    glutReshapeFunc(glutOnReshape);
    GLenum result = glewInit();
    if (GLEW_OK != result) {
        fprintf(stderr, "Failed to initialize GLEW: %s\n", glewGetErrorString(result));
        return -1;
    }

    // Init CUDA device
    int device_id = 0;
    int flags = GPUJPEG_OPENGL_INTEROPERABILITY;
    if (gpujpeg_init_device(device_id, flags) != 0) {
        return -1;
    }

    // Create decoder
    struct gpujpeg_decoder * decoder = gpujpeg_decoder_create(NULL);
    if (decoder == NULL) {
        fprintf(stderr, "Failed to create decoder!\n");
        return -1;
    }

    // Load image from file
    size_t image_size = 0;
    uint8_t * image = NULL;
    if (0 != gpujpeg_image_load_from_file(input_filename, &image, &image_size)) {
        fprintf(stderr, "Failed to load image [%s]!\n", input_filename);
        return -1;
    }

    // Add segment info headers into JPEG stream
    uint8_t * image_old = image;
    size_t image_old_size = image_size;
    double duration = gpujpeg_get_time();
    gpujpeg_reformat(image, image_size, &image, &image_size);
    duration = gpujpeg_get_time() - duration;
    printf("Rewritten JPEG stream in %0.2f ms (from %zd bytes to %zd bytes.\n", duration * 1000.0, image_old_size, image_size);
    gpujpeg_image_destroy(image_old);

    // Get image size and check number of color components
    struct gpujpeg_parameters param;
    struct gpujpeg_image_parameters param_image;
    if (0 != gpujpeg_decoder_get_image_info(image, image_size, &param_image, &param, 0 /* verbose */)) {
        fprintf(stderr, "Failed to read image size from file [%s]!\n", input_filename);
        return -1;
    }
    if (param.comp_count != 3) {
        fprintf(stderr, "Only JPEG images with 3 color components can be decoded into OpenGL texture!\n");
        return -1;
    }
    glutReshapeWindow(param_image.width, param_image.height);

    // Prepare decoder output to OpenGL texture
    struct gpujpeg_opengl_texture * texture = NULL;
    int texture_id = gpujpeg_opengl_texture_create(param_image.width, param_image.height, NULL);
    if (texture_id == 0) {
        fprintf(stderr, "Failed to create OpenGL texture!\n");
        return -1;
    }
    texture = gpujpeg_opengl_texture_register(texture_id, GPUJPEG_OPENGL_TEXTURE_WRITE);
    if (texture == NULL) {
        fprintf(stderr, "Failed to register OpenGL texture!\n");
        return -1;
    }
    struct gpujpeg_decoder_output decoder_output;
    gpujpeg_decoder_output_set_texture(&decoder_output, texture);

    // Decode image
    duration = gpujpeg_get_time();
    const int iterationCount = 20;
    for (int iteration = 0; iteration < iterationCount; iteration++)
    {
        double duration = gpujpeg_get_time();
        printf("Decoding Image %s\n", input_filename);
        if ( gpujpeg_decoder_decode(decoder, image, image_size, &decoder_output) != 0 ) {
            fprintf(stderr, "Failed to decode image [%s]!\n", input_filename);
            return -1;
        }
        duration = gpujpeg_get_time() - duration;
        struct gpujpeg_duration_stats stats;
        gpujpeg_decoder_get_stats(decoder, &stats);
        printf("    Stream Reader:     %10.2f ms\n", stats.duration_stream);
        printf("    Copy To Device:    %10.2f ms\n", stats.duration_memory_to);
        printf("    Huffman Decoder:   %10.2f ms\n", stats.duration_huffman_coder);
        printf("    DCT & Quantization:%10.2f ms\n", stats.duration_dct_quantization);
        printf("    Postprocessing:    %10.2f ms\n", stats.duration_preprocessor);
        printf("    Copy To Texture:   %10.2f ms\n", stats.duration_memory_from);
        printf("Image decoded OK in %0.2f ms (%dx%d)\n", duration * 1000.0, param_image.width, param_image.height);
    }
    duration = gpujpeg_get_time() - duration;
    printf("FPS: %0.2f\n", ((double) iterationCount) / duration);

    // Get data from OpenGL texture
    uint8_t* data = NULL;
    size_t data_size = 0;
    data = malloc(param_image.width * param_image.height * param.comp_count);
    gpujpeg_opengl_texture_get_data(texture->texture_id, data, &data_size);

    // Save image
    char output_filename[255];
    sprintf(output_filename, "%s.raw", input_filename);
    printf("Saving Image %s (%zu bytes)\n", output_filename, data_size);
    if ( gpujpeg_image_save_to_file(output_filename, data, data_size, &param_image) != 0 ) {
        fprintf(stderr, "Failed to save image [%s]!\n", output_filename);
        return -1;
    }

    // Clean up decoder
    gpujpeg_image_destroy(image);
    gpujpeg_decoder_destroy(decoder);

    // Show texture
    g_texture_id = texture_id;
    glutMainLoop();

    // Destroy OpenGL texture
    gpujpeg_opengl_texture_unregister(texture);
    gpujpeg_opengl_texture_destroy(texture_id);

    return 0;
}

void glutOnDisplay(void)
{
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, g_texture_id);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    gluOrtho2D(0, g_width, 0, g_height);
    glScalef(1, -1, 1);
    glTranslatef(0, -g_height, 0);
    glMatrixMode(GL_MODELVIEW);

    glBegin(GL_QUADS);
        glTexCoord2d(0.0,0.0); glVertex2f(0, 0);
        glTexCoord2d(1.0,0.0); glVertex2f(g_width, 0);
        glTexCoord2d(1.0,1.0); glVertex2f(g_width, g_height);
        glTexCoord2d(0.0,1.0); glVertex2f(0, g_height);
    glEnd();

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);

    glutSwapBuffers();
}

void glutOnIdle(void)
{
    glutPostRedisplay();
}

void glutOnKeyboard(unsigned char key, int x, int y)
{
    (void) x, (void) y;
    switch(key)
    {
        case 27: // ESCAPE key
            exit(0);
            break;
    }
}

void glutOnReshape(int width, int height)
{
    g_width = width;
    g_height = height;
    glViewport(0, 0, width, height);
}
/* vi: set expandtab sw=4 : */
