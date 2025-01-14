/**
 * @file
 * Copyright (c) 2011-2024, CESNET
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

#ifdef __cplusplus
#include <cstddef> // size_t
#include <cstdint>
#else
#include <stdbool.h>
#include <stddef.h> // size_t
#include <stdint.h>
#endif

#include "gpujpeg_type.h"

/**
 * @todo
 * Remove the redefinition and instead of cudaStream_t use a typedef to (void *).
 */
#ifndef __DRIVER_TYPES_H__
struct CUstream_st;
typedef struct CUstream_st *cudaStream_t;
#endif

#if __cplusplus >= 201402L || __STDC_VERSION__ >= 202311L
#define GPUJPEG_DEPRECATED [[deprecated]]
#elif defined _MSC_VER
#define GPUJPEG_DEPRECATED __declspec(deprecated)
#else
#define GPUJPEG_DEPRECATED __attribute__((deprecated))
#endif

#ifdef __cplusplus
extern "C" {
#endif

// CMake defines implicitly gpujpeg_EXPORTS in Windows (lower-case target name)
#ifdef gpujpeg_EXPORTS
#define GPUJPEG_EXPORTS
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

/** @return runtime version of the GPUJPEG library */
GPUJPEG_API int gpujpeg_version(void);
/** @return text representation of GPUJPEG version number */
GPUJPEG_API const char *gpujpeg_version_to_string(int version);

/** @return current time in seconds */
GPUJPEG_API double
gpujpeg_get_time(void);

/** Maximum number of devices for get device info */
#define GPUJPEG_MAX_DEVICE_COUNT 10

/** Device info for one device */
struct gpujpeg_device_info
{
    /// Device id
    int id;
    /// Device name
    char name[256];
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
gpujpeg_get_devices_info(void);

/**
 * Print information about available devices
 *
 * @return 0  if success
 * @return -1 for error (eg. no devices were found)
 */
GPUJPEG_API int
gpujpeg_print_devices_info(void);

/**
 * Init CUDA device
 *
 * @param device_id  CUDA device id (starting at 0)
 * @param flags  @ref Flags, e.g. if device info should be printed out (@ref GPUJPEG_INIT_DEV_VERBOSE) or
 *               enable OpenGL interoperability (@ref GPUJPEG_OPENGL_INTEROPERABILITY)
 * @return 0 if succeeds, otherwise nonzero
 */
GPUJPEG_API int
gpujpeg_init_device(int device_id, int flags);

enum restart_int {
    RESTART_AUTO = -1, ///< auto-select the best restart interval
    RESTART_NONE = 0,  ///< disabled; CPU Huffman encoder will be used
};

enum verbosity {
    GPUJPEG_LL_NORMAL = 0,  ///< normal verbosity
    GPUJPEG_LL_VERBOSE = 1, ///< print additional information
    GPUJPEG_LL_DEBUG = 2,   ///< print more information, including internal ones
    GPUJPEG_LL_DEBUG2 = 3,  ///< print maximum of information, including JPEG file internal structure
};

/**
 * JPEG parameters. This structure should not be initialized only be hand,
 * but at first gpujpeg_set_default_parameters should be call and then
 * some parameters could be changed.
 */
struct gpujpeg_parameters
{
    /// Verbosity level - show more information, collects duration of each phase, etc.
    /// @sa @ref verbosity for values
    int verbose;
    int perf_stats; /// record performance stats, set to 1 to allow gpujpeg_encoder_get_stats()

    /// Encoder quality level (0-100)
    int quality;

    /// Restart interval (0 means that restart interval is disabled and CPU huffman coder is used)
    /// @sa @rf restart_int for special values
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

    /// JPEG image component count; count of valid sampling_factor elements
    /// use gpujpeg_parameters_chroma_subsampling() to set comp_count and sampling_factor
    int comp_count;
    /// Sampling factors for each color component inside JPEG stream.
    struct gpujpeg_component_sampling_factor sampling_factor[GPUJPEG_MAX_COMPONENT_COUNT];

    /// Color space that is used inside JPEG stream = that is carried in JPEG format = to
    /// which are input data converted. Default value is @ref GPUJPEG_YCBCR_JPEG, changing it
    /// will change the JPEG format from @ref GPUJPEG_HEADER_JFIF "JFIF" (see also @ref
    /// gpujpeg_encoder_set_jpeg_header).
    /// - encoding: Set by user
    /// - decoding: Set by gpujpeg
    enum gpujpeg_color_space color_space_internal;
};

/**
 * Set default parameters for JPEG coder
 *
 * @param param  Parameters for JPEG coder
 * @return void
 * @sa gpujpeg_default_parameters
 */
GPUJPEG_API void
gpujpeg_set_default_parameters(struct gpujpeg_parameters* param);

/**
 * alternative to @ref gpujpeg_set_default_parameters allowing simultaneous
 * declaration and intialization for convenience.
 */
GPUJPEG_API struct gpujpeg_parameters
gpujpeg_default_parameters(void);

/** Sampling factor for all components */
typedef uint32_t gpujpeg_sampling_factor_t;
#define MK_SUBSAMPLING(comp1_factor_h, comp1_factor_v, comp2_factor_h, comp2_factor_v, comp3_factor_h, comp3_factor_v, \
                       comp4_factor_h, comp4_factor_v)                                                                 \
    ((comp1_factor_h) << 28U | (comp1_factor_v) << 24U | (comp2_factor_h) << 20U | (comp2_factor_v) << 16U |           \
     (comp3_factor_h) << 12U | (comp3_factor_v) << 8U | (comp4_factor_h) << 4U | (comp4_factor_v) << 0U)

#define GPUJPEG_SUBSAMPLING_UNKNOWN 0U
#define GPUJPEG_SUBSAMPLING_4444 MK_SUBSAMPLING(1, 1, 1, 1, 1, 1, 1, 1)
#define GPUJPEG_SUBSAMPLING_444 MK_SUBSAMPLING(1, 1, 1, 1, 1, 1, 0, 0)
#define GPUJPEG_SUBSAMPLING_440 MK_SUBSAMPLING(1, 2, 1, 1, 1, 1, 0, 0)
#define GPUJPEG_SUBSAMPLING_422 MK_SUBSAMPLING(2, 1, 1, 1, 1, 1, 0, 0)
#define GPUJPEG_SUBSAMPLING_420 MK_SUBSAMPLING(2, 2, 1, 1, 1, 1, 0, 0)
#define GPUJPEG_SUBSAMPLING_411 MK_SUBSAMPLING(4, 1, 1, 1, 1, 1, 0, 0)
#define GPUJPEG_SUBSAMPLING_410 MK_SUBSAMPLING(4, 2, 1, 1, 1, 1, 0, 0)
#define GPUJPEG_SUBSAMPLING_400 MK_SUBSAMPLING(1, 1, 0, 0, 0 ,0, 0, 0)
// non-standard sampling rates (in Y:Cb:Cr notation as described in
// <https://en.wikipedia.org/wiki/Chroma_subsampling#Different_Cb_and_Cr_rates>
#define GPUJPEG_SUBSAMPLING_442 MK_SUBSAMPLING(1, 2, 1, 2, 1, 1, 0, 0)
#define GPUJPEG_SUBSAMPLING_421 MK_SUBSAMPLING(2, 2, 2, 1, 1, 1, 0, 0)
/**
 * Set parameters for using specified chroma subsampling
 * @param param       parameters for coder
 * @param subsampling one of GPUJPEG_SUBSAMPLING_{444,422,420}
 */
GPUJPEG_API void
gpujpeg_parameters_chroma_subsampling(struct gpujpeg_parameters* param,
                                      gpujpeg_sampling_factor_t subsampling);

/**
 * Returns convenient name for subsampling (4:2:0 etc.). If it cannot be constructed
 * returns the format W1xH1:W2xH2:W3xH3.
 */
GPUJPEG_API const char*
gpujpeg_subsampling_get_name(int comp_count, const struct gpujpeg_component_sampling_factor *sampling_factor);

/**
 * returns gpujpeg_sampling_factor_t from textual representation of subsampling
 *
 * @retval subsampling, GPUJPEG_SUBSAMPLING_UNKNOWN if not recognized
 */
GPUJPEG_API gpujpeg_sampling_factor_t
gpujpeg_subsampling_from_name(const char* subsampling);

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
    /// Image data color space
    enum gpujpeg_color_space color_space;
    /// Image data sampling factor
    enum gpujpeg_pixel_format pixel_format;
    /// Number of bytes padded to each row
    int width_padding;
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
 * alternative to @ref gpujpeg_set_default_image_parameters allowing
 simultaneous * declaration and intialization for convenience.
 */
GPUJPEG_API struct gpujpeg_image_parameters
gpujpeg_default_image_parameters(void);

/**
 * Image file formats
 *
 * @note if modifying the enum, check the macros below
 */
enum gpujpeg_image_file_format {
    /// Unknown image file format
    GPUJPEG_IMAGE_FILE_UNKNOWN = 0,
    /// JPEG file format
    GPUJPEG_IMAGE_FILE_JPEG = 1,
    /// Raw file format
    /// @note all following formats must be raw
    GPUJPEG_IMAGE_FILE_RAW = 2,
    /// Gray file format
    GPUJPEG_IMAGE_FILE_GRAY,
    /// RGB file format, simple data format without header [R G B] [R G B] ...
    GPUJPEG_IMAGE_FILE_RGB,
    /// RGBA file format, simple data format without header [R G B A] [R G B A] ...
    GPUJPEG_IMAGE_FILE_RGBA,
    /// PNM file format
    GPUJPEG_IMAGE_FILE_PGM,
    GPUJPEG_IMAGE_FILE_PPM,
    GPUJPEG_IMAGE_FILE_PNM,
    /// PAM file format
    GPUJPEG_IMAGE_FILE_PAM,
    GPUJPEG_IMAGE_FILE_Y4M,
    /// YUV file format, simple data format without header [Y U V] [Y U V] ...
    /// @note all following formats must be YUV
    GPUJPEG_IMAGE_FILE_YUV,
    /// YUV file format with alpha channel [Y U V A] [Y U V A] ...
    GPUJPEG_IMAGE_FILE_YUVA,
    /// i420 file format
    GPUJPEG_IMAGE_FILE_I420,
    /// testing (empty) image, that is SW generated
    GPUJPEG_IMAGE_FILE_TST,
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
GPUJPEG_API size_t
gpujpeg_image_calculate_size(struct gpujpeg_image_parameters* param);

/**
 * Load image from file
 *
 * Allocated image must be freed by gpujpeg_image_load_from_file().
 *
 * @param         filaname    Image filename
 * @param[out]    image       Image data buffer allocated as CUDA host buffer
 * @param[in,out] image_size  Image data buffer size (can be specified for verification or 0 for retrieval)
 * @return 0 if succeeds, otherwise nonzero
 */
GPUJPEG_API int
gpujpeg_image_load_from_file(const char* filename, uint8_t** image, size_t* image_size);

/**
 * Save image to a file
 *
 * File format is deduced from the filename extension (if any).
 *
 * Use the extension "XXX" (eg. "image.XXX") to automatically select the
 * extension according to param_image. PNM is used for grayscale or RGB, PAM for
 * RGBA, Y4M ottherwise. "XXX" placeholder in filename is replaced with used
 * extension.
 *
 * @param filaname  Image filename
 * @param image  Image data buffer
 * @param image_size  Image data buffer size
 * @param param_image Image properties (may be NULL)
 * @return 0 if succeeds, otherwise nonzero
 */
GPUJPEG_API int
gpujpeg_image_save_to_file(char* filename, uint8_t* image, size_t image_size, const struct gpujpeg_image_parameters *param_image);

/**
 * Reads/obtains properties from uncompressed file (PNM etc.)
 *
 * For raw files it obtains only pixel format deduced from file extension.
 *
 * gpujpeg_image_parameters::comp_size is not set by the function and should be deduced from pixel_format
 *
 * May also return some with a null value - eg. when the file doesn't exist
 * but color space may be deduced from extension.
 * @retval 0 on success; != 0 on error
 * @retval -1 on error
 * @retval 1 only pixel format was deduced from file extension
 */
GPUJPEG_API int
gpujpeg_image_get_properties(const char *filename, struct gpujpeg_image_parameters *param_image, int file_exists);

/**
 * Destroys image allocated by GPUJPEG (eg. gpujpeg_image_load_from_file())
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
 * @note
 * currently defunct
 *
 * @param input
 * @param filename
 * @param param_image_from
 * @param param_image_to
 * @return 0 if succeeds, otherwise nonzero
 */
GPUJPEG_API int
gpujpeg_image_convert(const char* input, const char* output, struct gpujpeg_image_parameters param_image_from,
        struct gpujpeg_image_parameters param_image_to);

struct gpujpeg_opengl_context;
/**
 * Init OpenGL context
 *
 * This call is optional - initializes OpenGL context, thus doesn't need to be
 * called when context already exists. If not called, however, it may be required
 * to run glewInit() from client code prior to running GPUJPEG with GL interoperability.
 *
 * Returned pointer should be freed with gpujpeg_opengl_destroy() when done.
 *
 * @param[out] ctx pointer to OpenGL context data (to be passed to gpujpeg_opengl_destroy())
 * @return      0  if succeeds, otherwise nonzero
 * @return     -1  if initialization fails
 * @return     -2  if OpenGL support was not compiled in
 */
GPUJPEG_API int
gpujpeg_opengl_init(struct gpujpeg_opengl_context **ctx);

/**
 * Destroys OpenGL context created with gpujpeg_opengl_init()
 *
 * @return NULL if failed, otherwise state pointer
 */
GPUJPEG_API void
gpujpeg_opengl_destroy(struct gpujpeg_opengl_context *);

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
gpujpeg_opengl_texture_get_data(int texture_id, uint8_t* data, size_t* data_size);

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
gpujpeg_opengl_texture_map(struct gpujpeg_opengl_texture* texture, size_t* data_size);

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

GPUJPEG_API void
gpujpeg_print_pixel_formats();

/** Returns color space by string name */
GPUJPEG_API enum gpujpeg_color_space
gpujpeg_color_space_by_name(const char *name);

/** Returns number of color components in pixel format */
GPUJPEG_API int
gpujpeg_pixel_format_get_comp_count(enum gpujpeg_pixel_format pixel_format);

/** Returns name of the pixel format */
GPUJPEG_API const char*
gpujpeg_pixel_format_get_name(enum gpujpeg_pixel_format pixel_format);

/** Returns true if a pixel format is planar */
GPUJPEG_API int
gpujpeg_pixel_format_is_planar(enum gpujpeg_pixel_format pixel_format);

/** Calls cudaDeviceReset() */
GPUJPEG_API void
gpujpeg_device_reset(void);

#ifdef __cplusplus
}
#endif

#endif // GPUJPEG_COMMON_H
