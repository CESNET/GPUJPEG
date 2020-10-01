/**
 * @file
 * Copyright (c) 2011-2020, CESNET z.s.p.o
 * Copyright (c) 2011, Silicon Genome, LLC.
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
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef GPUJPEG_COMMON_H
#define GPUJPEG_COMMON_H

#include <stddef.h> // size_t
#include <stdint.h>
#include <libgpujpeg/gpujpeg_type.h>

/**
 * @todo
 * Remove the redefinition and instead of cudaStream_t use a typedef to (void *).
 */
#ifndef __DRIVER_TYPES_H__
struct CUstream_st;
typedef struct CUstream_st *cudaStream_t;
#endif

#ifdef __cplusplus
extern "C" {
#endif

#if (defined(_MSC_VER) || defined(__MINGW32__)) && !defined(GPUJPEG_STATIC)
    #ifdef GPUJPEG_EXPORTS
        #define GPUJPEG_API __declspec(dllexport)
    #else
        #define GPUJPEG_API __declspec(dllimport)
    #endif
#else
    #define GPUJPEG_API
#endif

/** @return current time in seconds */
GPUJPEG_API double
gpujpeg_get_time();

/** Marker used as segment info */
#define GPUJPEG_MARKER_SEGMENT_INFO GPUJPEG_MARKER_APP13

/** Maximum number of devices for get device info */
#define GPUJPEG_MAX_DEVICE_COUNT 10

#define GPUJPEG_IDCT_BLOCK_X	8
#define GPUJPEG_IDCT_BLOCK_Y 	8
#define GPUJPEG_IDCT_BLOCK_Z 	2

/** Device info for one device */
struct gpujpeg_device_info
{
    /// Device id
    int id;
    /// Device name
    char name[255];
    /// Compute capability major version
    int cc_major;
    /// Compute capability minor version
    int cc_minor;
    /// Amount of global memory
    size_t global_memory;
    /// Amount of constant memory
    size_t constant_memory;
    /// Amount of shared memory
    size_t shared_memory;
    /// Number of registers per block
    int register_count;
    /// Number of multiprocessors
    int multiprocessor_count;
};

/** Device info for all devices */
struct gpujpeg_devices_info
{
    /// Number of devices
    int device_count;
    /// Device info for each
    struct gpujpeg_device_info device[GPUJPEG_MAX_DEVICE_COUNT];
};

/**
 * Get information about available devices
 *
 * @return devices info
 */
GPUJPEG_API struct gpujpeg_devices_info
gpujpeg_get_devices_info();

/**
 * Print information about available devices
 *
 * @return void
 */
GPUJPEG_API void
gpujpeg_print_devices_info();

/**
 * Init CUDA device
 *
 * @param device_id  CUDA device id (starting at 0)
 * @param flags  Flags, e.g. if device info should be printed out (GPUJPEG_VERBOSE) or
 *               enable OpenGL interoperability (GPUJPEG_OPENGL_INTEROPERABILITY)
 * @return 0 if succeeds, otherwise nonzero
 */
GPUJPEG_API int
gpujpeg_init_device(int device_id, int flags);

/**
 * JPEG parameters. This structure should not be initialized only be hand,
 * but at first gpujpeg_set_default_parameters should be call and then
 * some parameters could be changed.
 */
struct gpujpeg_parameters
{
    /// Verbosity level - show more information, collects duration of each phase, etc.
    /// 0 - normal, 1 - verbose, 2 - debug
    int verbose;

    /// Encoder quality level (0-100)
    int quality;

    /// Restart interval (0 means that restart interval is disabled and CPU huffman coder is used)
    int restart_interval;

    /// Flag which determines if interleaved format of JPEG stream should be used, "1" = only
    /// one scan which includes all color components (e.g. Y Cb Cr Y Cb Cr ...),
    /// or "0" = one scan for each color component (e.g. Y Y Y ..., Cb Cb Cb ..., Cr Cr Cr ...)
    int interleaved;

    /// Use segment info in stream for fast decoding. The segment info is placed into special
    /// application headers and contains indexes to beginnings of segments in the stream, so
    /// the decoder don't have to parse the stream byte-by-byte but he can only read the
    /// segment info and start decoding. The segment info is presented for each scan, and thus
    /// the best result is achieved when it is used in combination with "interleaved = 1" settings.
    int segment_info;

    /// Sampling factors for each color component inside JPEG stream.
    struct gpujpeg_component_sampling_factor sampling_factor[GPUJPEG_MAX_COMPONENT_COUNT];

    /// Color space that is used inside JPEG stream = that is carried in JPEG format = to
    /// which are input data converted (default value is JPEG YCbCr)
    enum gpujpeg_color_space color_space_internal;
};

/**
 * Set default parameters for JPEG coder
 *
 * @param param  Parameters for JPEG coder
 * @return void
 */
GPUJPEG_API void
gpujpeg_set_default_parameters(struct gpujpeg_parameters* param);

/**
 * Set parameters for using 4:2:2 chroma subsampling
 *
 * @param param  Parameters for coder
 * @return void
 */
GPUJPEG_API void
gpujpeg_parameters_chroma_subsampling_422(struct gpujpeg_parameters* param);

/**
 * Set parameters for using 4:2:0 chroma subsampling
 *
 * @param param  Parameters for coder
 * @return void
 */
GPUJPEG_API void
gpujpeg_parameters_chroma_subsampling_420(struct gpujpeg_parameters* param);

/**
 * Image parameters. This structure should not be initialized only be hand,
 * but at first gpujpeg_image_set_default_parameters should be call and then
 * some parameters could be changed.
 */
struct gpujpeg_image_parameters {
    /// Image data width
    int width;
    /// Image data height
    int height;
    /// Image data component count
    int comp_count;
    /// Image data color space
    enum gpujpeg_color_space color_space;
    /// Image data sampling factor
    enum gpujpeg_pixel_format pixel_format;
};

/**
 * Set default parameters for JPEG image
 *
 * @param param  Parameters for image
 * @return void
 */
GPUJPEG_API void
gpujpeg_image_set_default_parameters(struct gpujpeg_image_parameters* param);

/**
 * Image file formats
 *
 * If running out numbers, the representation may be more dense, eg. lower 2 bits for RAW/JPEG
 * and then GRAY -> 1|12, then 1|16, 1|20, 1|24 etc.
 */
enum gpujpeg_image_file_format {
    /// Unknown image file format
    GPUJPEG_IMAGE_FILE_UNKNOWN = 0,
    /// Raw file format
    GPUJPEG_IMAGE_FILE_RAW = 1,
    /// JPEG file format
    GPUJPEG_IMAGE_FILE_JPEG = 2,
    /// RGB file format, simple data format without header [R G B] [R G B] ...
    GPUJPEG_IMAGE_FILE_RGB = 1 | 4,
    /// YUV file format, simple data format without header [Y U V] [Y U V] ...
    GPUJPEG_IMAGE_FILE_YUV = 1 | 8,
    /// Gray file format
    GPUJPEG_IMAGE_FILE_GRAY = 1 | 16,
    /// RGBX file format, simple data format without header [R G B X] [R G B X] ...
    GPUJPEG_IMAGE_FILE_RGBA = 1 | 32,
    /// i420 file format
    GPUJPEG_IMAGE_FILE_I420 = 1 | 64,
    /// PNM file format
    GPUJPEG_IMAGE_FILE_PNM = 1 | 128,
    /// PAM file format
    GPUJPEG_IMAGE_FILE_PAM = 1 | 256,
};

/**
 * Encoder/decoder fine-grained statistics with duration of individual steps
 * of JPEG compression/decompression. All values are in milliseconds.
 *
 * @note
 * The values are only informative and for debugging only and thus this is
 * not considered as a part of a public API.
 */
struct gpujpeg_duration_stats {
    double duration_memory_to;
    double duration_memory_from;
    double duration_memory_map;
    double duration_memory_unmap;
    double duration_preprocessor;
    double duration_dct_quantization;
    double duration_huffman_coder;
    double duration_stream;
    double duration_in_gpu;
};

/**
 * Get image file format from filename
 *
 * @param filename  Filename of image file
 * @return image_file_format or GPUJPEG_IMAGE_FILE_UNKNOWN if type cannot be determined
 */
GPUJPEG_API enum gpujpeg_image_file_format
gpujpeg_image_get_file_format(const char* filename);

/**
 * Sets cuda device.
 *
 * @param index  Index of the CUDA device to be activated.
 */
GPUJPEG_API void gpujpeg_set_device(int index);

/**
 * Calculate size for image by parameters
 *
 * @param param  Image parameters
 * @return calculate size
 */
GPUJPEG_API int
gpujpeg_image_calculate_size(struct gpujpeg_image_parameters* param);

/**
 * Load RGB image from file
 *
 * @param         filaname    Image filename
 * @param[out]    image       Image data buffer allocated as CUDA host buffer
 * @param[in,out] image_size  Image data buffer size (can be specified for verification or 0 for retrieval)
 * @return 0 if succeeds, otherwise nonzero
 */
GPUJPEG_API int
gpujpeg_image_load_from_file(const char* filename, uint8_t** image, int* image_size);

/**
 * Save RGB image to file
 *
 * @param filaname  Image filename
 * @param image  Image data buffer
 * @param image_size  Image data buffer size
 * @param param_image Image properties (may be NULL)
 * @return 0 if succeeds, otherwise nonzero
 */
GPUJPEG_API int
gpujpeg_image_save_to_file(const char* filename, uint8_t* image, int image_size, const struct gpujpeg_image_parameters *param_image);

/**
 * Reads/obtains properties from uncompressed file (PNM etc.)
 *
 * May also return some with a null value - eg. when the file doesn't exist
 * but color space may be deduced from extension.
 */
GPUJPEG_API int
gpujpeg_image_get_properties(const char *filename, struct gpujpeg_image_parameters *param_image, int file_exists);

/**
 * Destroy DXT image
 *
 * @param image  Image data buffer
 * @return 0 if succeeds, otherwise nonzero
 */
GPUJPEG_API int
gpujpeg_image_destroy(uint8_t* image);

/**
 * Print range info for image samples
 *
 * @param filename
 * @param width
 * @param height
 * @param sampling_factor
 */
GPUJPEG_API void
gpujpeg_image_range_info(const char* filename, int width, int height, enum gpujpeg_pixel_format sampling_factor);

/**
 * Convert image
 *
 * @param input
 * @param filename
 * @param param_image_from
 * @param param_image_to
 */
GPUJPEG_API void
gpujpeg_image_convert(const char* input, const char* output, struct gpujpeg_image_parameters param_image_from,
        struct gpujpeg_image_parameters param_image_to);

/**
 * Init OpenGL context
 *
 * @return 0 if succeeds, otherwise nonzero
 */
GPUJPEG_API int
gpujpeg_opengl_init();

/**
 * Create OpenGL texture
 *
 * @param width
 * @param height
 * @param data
 * @return nonzero texture id if succeeds, otherwise 0
 */
GPUJPEG_API int
gpujpeg_opengl_texture_create(int width, int height, uint8_t* data);

/**
 * Set data to OpenGL texture
 *
 * @param texture_id
 * @param data
 * @return 0 if succeeds, otherwise nonzero
 */
GPUJPEG_API int
gpujpeg_opengl_texture_set_data(int texture_id, uint8_t* data);

/**
 * Get data from OpenGL texture
 *
 * @param texture_id
 * @param data
 * @param data_size
 * @return 0 data if succeeds, otherwise nonzero
 */
GPUJPEG_API int
gpujpeg_opengl_texture_get_data(int texture_id, uint8_t* data, int* data_size);

/**
 * Destroy OpenGL texture
 *
 * @param texture_id
 */
GPUJPEG_API void
gpujpeg_opengl_texture_destroy(int texture_id);

/**
 * Registered OpenGL texture type
 */
enum gpujpeg_opengl_texture_type
{
    GPUJPEG_OPENGL_TEXTURE_READ = 1,
    GPUJPEG_OPENGL_TEXTURE_WRITE = 2
};

/**
 * Represents OpenGL texture that is registered to CUDA,
 * thus the device pointer can be acquired.
 */
struct gpujpeg_opengl_texture
{
    /// Texture id
    int texture_id;
    /// Texture type
    enum gpujpeg_opengl_texture_type texture_type;
    /// Texture width
    int texture_width;
    /// Texture height
    int texture_height;
    /// Texture pixel buffer object type
    int texture_pbo_type;
    /// Texture pixel buffer object id
    int texture_pbo_id;
    /// Texture PBO resource for CUDA
    struct cudaGraphicsResource* texture_pbo_resource;

    /// Texture callbacks parameter
    void * texture_callback_param;
    /// Texture callback for attaching OpenGL context (by default not used)
    void (*texture_callback_attach_opengl)(void* param);
    /// Texture callback for detaching OpenGL context (by default not used)
    void (*texture_callback_detach_opengl)(void* param);
    /// If you develop multi-threaded application where one thread use CUDA
    /// for JPEG decoding and other thread use OpenGL for displaying results
    /// from JPEG decoder, when an image is decoded you must detach OpenGL context
    /// from displaying thread and attach it to compressing thread (inside
    /// code of texture_callback_attach_opengl which is automatically invoked
    /// by decoder), decoder then is able to copy data from GPU memory used
    /// for compressing to GPU memory used by OpenGL texture for displaying,
    /// then decoder call the second callback and you have to detach OpenGL context
    /// from compressing thread and attach it to displaying thread (inside code of
    /// texture_callback_detach_opengl).
    ///
    /// If you develop single-thread application where the only thread use CUDA
    /// for compressing and OpenGL for displaying you don't have to implement
    /// these callbacks because OpenGL context is already attached to thread
    /// that use CUDA for JPEG decoding.
};

/**
 * Register OpenGL texture to CUDA
 *
 * @param texture_id
 * @return allocated registred texture structure
 */
GPUJPEG_API struct gpujpeg_opengl_texture*
gpujpeg_opengl_texture_register(int texture_id, enum gpujpeg_opengl_texture_type texture_type);

/**
 * Unregister OpenGL texture from CUDA. Deallocated given
 * structure.
 *
 * @param texture
 */
GPUJPEG_API void
gpujpeg_opengl_texture_unregister(struct gpujpeg_opengl_texture* texture);

/**
 * Map registered OpenGL texture to CUDA and return
 * device pointer to the texture data
 *
 * @param texture
 * @param data_size  Data size in returned buffer
 * @param copy_from_texture  Specifies whether memory copy from texture
 *                           should be performed
 */
GPUJPEG_API uint8_t*
gpujpeg_opengl_texture_map(struct gpujpeg_opengl_texture* texture, int* data_size);

/**
 * Unmap registered OpenGL texture from CUDA and the device
 * pointer is no longer useable.
 *
 * @param texture
 * @param copy_to_texture  Specifies whether memoryc copy to texture
 *                         should be performed
 */
GPUJPEG_API void
gpujpeg_opengl_texture_unmap(struct gpujpeg_opengl_texture* texture);

/**
 * Get color space name
 *
 * @param color_space
 */
GPUJPEG_API const char*
gpujpeg_color_space_get_name(enum gpujpeg_color_space color_space);

/** Returns pixel format by string name */
GPUJPEG_API enum gpujpeg_pixel_format
gpujpeg_pixel_format_by_name(const char *name);

/** Returns number of color components in pixel format */
GPUJPEG_API int
gpujpeg_pixel_format_get_comp_count(enum gpujpeg_pixel_format pixel_format);

/** Returns name of the pixel format */
GPUJPEG_API const char*
gpujpeg_pixel_format_get_name(enum gpujpeg_pixel_format pixel_format);

/** Returns true if a pixel format is planar */
GPUJPEG_API int
gpujpeg_pixel_format_is_planar(enum gpujpeg_pixel_format pixel_format);

/** Returns true if a pixel format is subsampled */
GPUJPEG_API int
gpujpeg_pixel_format_is_subsampled(enum gpujpeg_pixel_format pixel_format);

#ifdef __cplusplus
}
#endif

#endif // GPUJPEG_COMMON_H
