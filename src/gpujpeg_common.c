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

#include "../libgpujpeg/gpujpeg_common.h"
#include "../libgpujpeg/gpujpeg_decoder.h" // decoder placeholders
#include "gpujpeg_common_internal.h"
#include "gpujpeg_util.h"
#ifdef HAVE_GPUJPEG_VERSION_H
#include <libgpujpeg/gpujpeg_version.h>
#endif // defined HAVE_GPUJPEG_VERSION_H
#include "utils/image_delegate.h"
#include <errno.h>                          // for errno
#include <math.h>
#include <stdbool.h>
#include <string.h>                         // for strcmp, strerror, strlen, strrchr
#if defined(_MSC_VER)
  #include <windows.h>
  #define strcasecmp _stricmp
#else
  #include <strings.h>
#endif
#ifdef GPUJPEG_USE_OPENGL
    #define GL_GLEXT_PROTOTYPES
    #include <GL/glew.h>
    #ifndef GL_VERSION_1_2
        #error "OpenGL 1.2 is required"
    #endif
    #if defined(GPUJPEG_USE_GLFW)
        #include <GLFW/glfw3.h>
    #elif defined(GPUJPEG_USE_GLX)
        #include <GL/glx.h>
    #endif
    #include <cuda_gl_interop.h>
#endif

#if _STDC_VERSION__ >= 201112L
#include <threads.h>
#elif __cplusplus < 201103L
#define thread_local
#endif

// rounds number of segment bytes up to next multiple of 128
#define SEGMENT_ALIGN(b) (((b) + 127) & ~127)

#if defined(_MSC_VER)
    /* Documented at declaration */
    double gpujpeg_get_time()
    {
        static double frequency = 0.0;
        static LARGE_INTEGER frequencyAsInt;
        LARGE_INTEGER timer;
        if (frequency == 0.0) {
            if (!QueryPerformanceFrequency(&frequencyAsInt)) {
                return -1.0;
            }
            else {
                frequency = (double)frequencyAsInt.QuadPart;
            }
        }
        QueryPerformanceCounter(&timer);
        return (double) timer.QuadPart / frequency;
    }
#elif defined(__linux__) || defined(__APPLE__)
    #include <sys/time.h>

    /* Documented at declaration */
    double gpujpeg_get_time(void)
    {
        struct timeval tv;
        gettimeofday(&tv, 0);
        return (double) tv.tv_sec + (double) tv.tv_usec * 0.000001;
    }
#endif

#define GPUJPEG_CUSTOM_TIMER_RESET(name)                                                                           \
        do {                                                                                                           \
            (name).started = 0;                                                                                        \
        } while ( 0 )

#define PLANAR   1u
static const struct {
    enum gpujpeg_pixel_format pixel_format;
    uint32_t flags;
    int comp_count;
    int bpp; ///< bytes per pixel, not relevant for planar formats
    const char *name;
    /// native sampling factor for the pixfmt
    struct gpujpeg_component_sampling_factor sampling_factor[GPUJPEG_MAX_COMPONENT_COUNT];
} gpujpeg_pixel_format_desc[] = {
    {GPUJPEG_PIXFMT_STD,        0,      0, 0, "(file standard)", {{0}}                           },
    {GPUJPEG_PIXFMT_NO_ALPHA,   0,      0, 0, "(without alpha)", {{0}}                           },
    {GPUJPEG_PIXFMT_AUTODETECT, 0,      0, 0, "(autodetect)",    {{0}}                           },
    {GPUJPEG_PIXFMT_NONE,       0,      0, 0, "(unknown)",       {{0}}                           },
    {GPUJPEG_U8,                0,      1, 1, "u8",              {{1, 1}}                        },
    {GPUJPEG_444_U8_P012,       0,      3, 3, "444-u8-p012",     {{1, 1}, {1, 1}, {1, 1}}        },
    {GPUJPEG_444_U8_P0P1P2,     PLANAR, 3, 0, "444-u8-p0p1p2",   {{1, 1}, {1, 1}, {1, 1}}        },
    {GPUJPEG_422_U8_P1020,      0,      3, 2, "422-u8-p1020",    {{2, 1}, {1, 1}, {1, 1}}        },
    {GPUJPEG_422_U8_P0P1P2,     PLANAR, 3, 0, "422-u8-p0p1p2",   {{2, 1}, {1, 1}, {1, 1}}        },
    {GPUJPEG_420_U8_P0P1P2,     PLANAR, 3, 0, "420-u8-p0p1p2",   {{2, 2}, {1, 1}, {1, 1}}        },
    {GPUJPEG_4444_U8_P0123,     0,      4, 4, "4444-u8-p0123",   {{1, 1}, {1, 1}, {1, 1}, {1, 1}}},
};

/* Documented at declaration */
struct gpujpeg_devices_info
gpujpeg_get_devices_info(void)
{
    struct gpujpeg_devices_info devices_info = { 0 };

    cudaGetDeviceCount(&devices_info.device_count);
    gpujpeg_cuda_check_error("Cannot get number of CUDA devices", return devices_info);

    if ( devices_info.device_count > GPUJPEG_MAX_DEVICE_COUNT ) {
        fprintf(stderr, "[GPUJPEG] [Warning] There are available more CUDA devices (%d) than maximum count (%d).\n",
            devices_info.device_count, GPUJPEG_MAX_DEVICE_COUNT);
        fprintf(stderr, "[GPUJPEG] [Warning] Using maximum count (%d).\n", GPUJPEG_MAX_DEVICE_COUNT);
        devices_info.device_count = GPUJPEG_MAX_DEVICE_COUNT;
    }

    for ( int device_id = 0; device_id < devices_info.device_count; device_id++ ) {
        struct cudaDeviceProp device_properties;
        cudaGetDeviceProperties(&device_properties, device_id);

        struct gpujpeg_device_info* device_info = &devices_info.device[device_id];

        device_info->id = device_id;
        strncpy(device_info->name, device_properties.name, sizeof device_info->name);
        device_info->cc_major = device_properties.major;
        device_info->cc_minor = device_properties.minor;
        device_info->global_memory = device_properties.totalGlobalMem;
        device_info->constant_memory = device_properties.totalConstMem;
        device_info->shared_memory = device_properties.sharedMemPerBlock;
        device_info->register_count = device_properties.regsPerBlock;
#if CUDART_VERSION >= 2000
        device_info->multiprocessor_count = device_properties.multiProcessorCount;
#endif
    }

    return devices_info;
}

/* Documented at declaration */
int
gpujpeg_print_devices_info(void)
{
    struct gpujpeg_devices_info devices_info = gpujpeg_get_devices_info();
    if ( devices_info.device_count == 0 ) {
        printf("There is no device supporting CUDA.\n");
        return -1;
    } else if ( devices_info.device_count == 1 ) {
        printf("There is 1 device supporting CUDA:\n");
    } else {
        printf("There are %d devices supporting CUDA:\n", devices_info.device_count);
    }

    for ( int device_id = 0; device_id < devices_info.device_count; device_id++ ) {
        struct gpujpeg_device_info* device_info = &devices_info.device[device_id];
        printf("\nDevice #%d: \"%s\"\n", device_info->id, device_info->name);
        printf("  Compute capability: %d.%d\n", device_info->cc_major, device_info->cc_minor);
        printf("  Total amount of global memory: %zu KiB\n", device_info->global_memory / 1024);
        printf("  Total amount of constant memory: %zu KiB\n", device_info->constant_memory / 1024);
        printf("  Total amount of shared memory per block: %zu KiB\n", device_info->shared_memory / 1024);
        printf("  Total number of registers available per block: %d\n", device_info->register_count);
        printf("  Multiprocessors: %d\n", device_info->multiprocessor_count);
    }
    return 0;
}

/* Documented at declaration */
int
gpujpeg_init_device(int device_id, int flags)
{
    int dev_count;
    cudaGetDeviceCount(&dev_count);
    gpujpeg_cuda_check_error("Cannot get number of CUDA devices", return -1);
    if ( dev_count == 0 ) {
        fprintf(stderr, "[GPUJPEG] [Error] No CUDA enabled device\n");
        return -1;
    }

    if ( device_id < 0 || device_id >= dev_count ) {
        fprintf(stderr, "[GPUJPEG] [Error] Selected device %d is out of bound. Devices on your system are in range %d - %d\n",
            device_id, 0, dev_count - 1);
        return -1;
    }

    struct cudaDeviceProp devProp;
    if ( cudaSuccess != cudaGetDeviceProperties(&devProp, device_id) ) {
        fprintf(stderr,
            "[GPUJPEG] [Error] Can't get CUDA device properties!\n"
            "[GPUJPEG] [Error] Do you have proper driver for CUDA installed?\n"
        );
        return -1;
    }

    if ( devProp.major < 1 ) {
        fprintf(stderr, "[GPUJPEG] [Error] Device %d does not support CUDA\n", device_id);
        return -1;
    }

#if defined GPUJPEG_USE_OPENGL && CUDART_VERSION < 5000
    if ( flags & GPUJPEG_OPENGL_INTEROPERABILITY ) {
        cudaGLSetGLDevice(device_id); // not needed since CUDA 5.0
        gpujpeg_cuda_check_error("Enabling OpenGL interoperability", return -1);
    }
#endif

    if ( flags & GPUJPEG_INIT_DEV_VERBOSE ) {
        int cuda_driver_version = 0;
        cudaDriverGetVersion(&cuda_driver_version);
        printf("CUDA driver version:   %d.%d\n", cuda_driver_version / 1000, (cuda_driver_version % 100) / 10);

        int cuda_runtime_version = 0;
        cudaRuntimeGetVersion(&cuda_runtime_version);
        printf("CUDA runtime version:  %d.%d\n", cuda_runtime_version / 1000, (cuda_runtime_version % 100) / 10);

        printf("Using Device #%d:       %s (c.c. %d.%d)\n", device_id, devProp.name, devProp.major, devProp.minor);
    }

    cudaSetDevice(device_id);
    gpujpeg_cuda_check_error("Set CUDA device", return -1);

    // Test by simple copying that the device is ready
    uint8_t data[] = {8};
    uint8_t* d_data = NULL;
    cudaMalloc((void**)&d_data, 1);
    cudaMemcpy(d_data, data, 1, cudaMemcpyHostToDevice);
    cudaFree(d_data);
    cudaError_t error = cudaGetLastError();
    if ( cudaSuccess != error ) {
        fprintf(stderr, "[GPUJPEG] [Error] Failed to initialize CUDA device: %s\n", cudaGetErrorString(error));
        if ( flags & GPUJPEG_OPENGL_INTEROPERABILITY )
            fprintf(stderr, "[GPUJPEG] [Info]  OpenGL interoperability is used, is OpenGL context available?\n");
        return -1;
    }

    return 0;
}

/* Documented at declaration */
void
gpujpeg_set_default_parameters(struct gpujpeg_parameters* param)
{
    param->verbose = GPUJPEG_LL_INFO;
    param->perf_stats = 0;
    param->quality = 75;
    param->restart_interval = 8;
    param->interleaved = 0;
    param->segment_info = 0;
    param->comp_count = 0;
    for ( int comp = 0; comp < GPUJPEG_MAX_COMPONENT_COUNT; comp++ ) {
        param->sampling_factor[comp].horizontal = 1;
        param->sampling_factor[comp].vertical = 1;
    }
    param->color_space_internal = GPUJPEG_YCBCR_BT601_256LVLS;
}

struct gpujpeg_parameters
gpujpeg_default_parameters()
{
    struct gpujpeg_parameters ret;
    gpujpeg_set_default_parameters(&ret);
    return ret;
}

void
gpujpeg_parameters_chroma_subsampling(struct gpujpeg_parameters* param,
                                      gpujpeg_sampling_factor_t subsampling)
{
    enum {
        HI_4_BIT_OFF = 28,
    };
    param->comp_count = 0;
    int comp = 0;
    for ( ; comp < GPUJPEG_MAX_COMPONENT_COUNT; comp++ ) {
        param->sampling_factor[comp].horizontal = subsampling >> HI_4_BIT_OFF;
        subsampling <<= 4U;
        param->sampling_factor[comp].vertical = subsampling >> HI_4_BIT_OFF;
        subsampling <<= 4U;
        if ( param->sampling_factor[comp].vertical * param->sampling_factor[comp].horizontal == 0 ) {
            break;
        }
        param->comp_count += 1;
    }
    GPUJPEG_ASSERT((subsampling == GPUJPEG_SUBSAMPLING_UNKNOWN || param->comp_count >= 1) && "Invalid subsampling!");
    for ( ; comp < GPUJPEG_MAX_COMPONENT_COUNT; ++comp ) {
        param->sampling_factor[comp].horizontal = 0;
        param->sampling_factor[comp].vertical = 0;
    }
}

/**
 * @returns true  if parameters are the same
 * @note
 * @ref gpujpeg_parameters members verbose, perf_stats and **quality**
 * ignored by the comparison
 */
static bool
gpujpeg_parameters_equals(const struct gpujpeg_parameters *p1 , const struct gpujpeg_parameters *p2)
{
    if ( p1->comp_count != p2->comp_count ||
            p1->restart_interval != p2->restart_interval ||
            p1->interleaved != p2->interleaved ||
            p1->segment_info != p2->segment_info ||
            p1->color_space_internal != p2->color_space_internal) {
        return false;
    }

    for ( int comp = 0; comp < p1->comp_count; comp++ ) {
        if (p1->sampling_factor[comp].horizontal != p2->sampling_factor[comp].horizontal ||
                p1->sampling_factor[comp].vertical != p2->sampling_factor[comp].vertical) {
            return false;
        }
    }

    return true;
}

/* Documented at declaration */
void
gpujpeg_image_set_default_parameters(struct gpujpeg_image_parameters* param)
{
    param->width = 0;
    param->height = 0;
    param->color_space = GPUJPEG_RGB;
    param->pixel_format = GPUJPEG_444_U8_P012;
    param->width_padding = 0;
}

struct gpujpeg_image_parameters
gpujpeg_default_image_parameters()
{
    struct gpujpeg_image_parameters ret;
    gpujpeg_image_set_default_parameters(&ret);
    return ret;
}

/**
 * @returns true  if parameters are the same
 */
bool
gpujpeg_image_parameters_equals(const struct gpujpeg_image_parameters *p1 , const struct gpujpeg_image_parameters *p2)
{
    return p1->width == p2->width &&
        p1->height == p2->height &&
        p1->color_space == p2->color_space &&
        p1->pixel_format == p2->pixel_format &&
        p1->width_padding == p2->width_padding;
}

/* Documented at declaration */
enum gpujpeg_image_file_format
gpujpeg_image_get_file_format(const char* filename)
{
    static const struct {
        const char *ext;
        enum gpujpeg_image_file_format format;
    } extensions[] = {
        { "raw",  GPUJPEG_IMAGE_FILE_RAW },
        { "rgb",  GPUJPEG_IMAGE_FILE_RGB},
        { "rgba", GPUJPEG_IMAGE_FILE_RGBA},
        { "yuv",  GPUJPEG_IMAGE_FILE_YUV},
        { "yuva", GPUJPEG_IMAGE_FILE_YUVA},
        { "i420", GPUJPEG_IMAGE_FILE_I420},
        { "r",    GPUJPEG_IMAGE_FILE_GRAY},
        { "jpg",  GPUJPEG_IMAGE_FILE_JPEG},
        { "jpeg", GPUJPEG_IMAGE_FILE_JPEG},
        { "jfif", GPUJPEG_IMAGE_FILE_JPEG},
        //{ "pbm",  GPUJPEG_IMAGE_FILE_PNM},
        { "pnm",  GPUJPEG_IMAGE_FILE_PNM},
        { "pgm",  GPUJPEG_IMAGE_FILE_PGM},
        { "ppm",  GPUJPEG_IMAGE_FILE_PPM},
        { "pam",  GPUJPEG_IMAGE_FILE_PAM},
        { "y4m",  GPUJPEG_IMAGE_FILE_Y4M},
        { "tst",  GPUJPEG_IMAGE_FILE_TST},
        { "XXX",  GPUJPEG_IMAGE_FILE_RAW},
    };

    const char * ext = strrchr(filename, '.');
    if ( ext == NULL )
        return GPUJPEG_IMAGE_FILE_UNKNOWN;
    ext++;
    if ( strcmp(ext, "help") == 0 ) {
        printf("Recognized extensions:\n");
    }
    for ( unsigned i = 0; i < sizeof extensions / sizeof extensions[0]; i++ ) {
        if ( strcmp(ext, "help") == 0 ) {
            printf("\t- %s\n", extensions[i].ext);
        }
        if ( strcasecmp(ext, extensions[i].ext) == 0 ) {
            return extensions[i].format;
        }
    }
    printf("\nUse \"help.tst\" (eg. `gpujpegtool help.tst null.jpg`) for test image usage).\n");
    return GPUJPEG_IMAGE_FILE_UNKNOWN;
}

static enum { FF_CS_NONE, FF_CS_RGB, FF_CS_YCBCR } get_file_type_cs(enum gpujpeg_image_file_format format)
{
    switch ( format ) {
    case GPUJPEG_IMAGE_FILE_UNKNOWN:
    case GPUJPEG_IMAGE_FILE_JPEG:
    case GPUJPEG_IMAGE_FILE_RAW:
    case GPUJPEG_IMAGE_FILE_TST:
        return FF_CS_NONE;
    case GPUJPEG_IMAGE_FILE_GRAY:
    case GPUJPEG_IMAGE_FILE_Y4M:
    case GPUJPEG_IMAGE_FILE_YUV:
    case GPUJPEG_IMAGE_FILE_YUVA:
    case GPUJPEG_IMAGE_FILE_I420:
        return FF_CS_YCBCR;
    case GPUJPEG_IMAGE_FILE_RGB:
    case GPUJPEG_IMAGE_FILE_RGBA:
    case GPUJPEG_IMAGE_FILE_PGM:
    case GPUJPEG_IMAGE_FILE_PPM:
    case GPUJPEG_IMAGE_FILE_PNM:
    case GPUJPEG_IMAGE_FILE_PAM:
        return FF_CS_RGB;
    }
    GPUJPEG_ASSERT(0 && "Unhandled file type!");
}

void gpujpeg_set_device(int index)
{
    cudaSetDevice(index);
}

/* Documented at declaration */
void
gpujpeg_component_print8(struct gpujpeg_component* component, uint8_t* d_data)
{
    int data_size = component->data_width * component->data_height;
    uint8_t* data = NULL;
    cudaMallocHost((void**)&data, data_size * sizeof(uint8_t));
    cudaMemcpy(data, d_data, data_size * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    printf("Print Data\n");
    for ( int y = 0; y < component->data_height; y++ ) {
        for ( int x = 0; x < component->data_width; x++ ) {
            printf("%3u ", data[y * component->data_width + x]);
        }
        printf("\n");
    }
    cudaFreeHost(data);
}

/* Documented at declaration */
void
gpujpeg_component_print16(struct gpujpeg_component* component, int16_t* d_data)
{
    int data_size = component->data_width * component->data_height;
    int16_t* data = NULL;
    cudaMallocHost((void**)&data, data_size * sizeof(int16_t));
    cudaMemcpy(data, d_data, data_size * sizeof(int16_t), cudaMemcpyDeviceToHost);

    printf("Print Data\n");
    for ( int y = 0; y < component->data_height; y++ ) {
        for ( int x = 0; x < component->data_width; x++ ) {
            printf("%3d ", data[y * component->data_width + x]);
        }
        printf("\n");
    }
    cudaFreeHost(data);
}

/* Documented at declaration */
int
gpujpeg_coder_init(struct gpujpeg_coder * coder)
{
    // Get info about the device
    struct cudaDeviceProp device_properties;
    int device_idx;
    GPUJPEG_CHECK(cudaGetDevice(&device_idx), return -1);
    cudaGetDeviceProperties(&device_properties, device_idx);
    gpujpeg_cuda_check_error("Device info getting", return -1);
    coder->cuda_cc_major = device_properties.major;
    coder->cuda_cc_minor = device_properties.minor;
    if (device_properties.major < 2) {
        fprintf(stderr, "GPUJPEG coder is currently broken on cards with cc < 2.0\n");
        return -1;
    }

    // Initialize coder for no image
    coder->param.quality = -1;
    coder->param.restart_interval = -1;
    coder->param.interleaved = -1;
    coder->param.segment_info = -1;
    coder->param.color_space_internal = GPUJPEG_NONE;
    coder->param_image.color_space = GPUJPEG_NONE;
    coder->preprocessor = NULL;
    coder->component = NULL;
    coder->d_component = NULL;
    coder->component_allocated_size = 0;
    coder->segment = NULL;
    coder->d_segment = NULL;
    coder->segment_allocated_size = 0;
    coder->block_list = NULL;
    coder->d_block_list = NULL;
    coder->block_allocated_size = 0;
    coder->d_data = NULL;
    coder->data_quantized = NULL;
    coder->d_data_quantized = NULL;
    coder->data_allocated_size = 0;
    coder->data_raw = NULL;
    coder->d_data_raw = NULL;
    coder->d_data_raw_allocated = NULL;
    coder->data_raw_allocated_size = 0;
    coder->data_compressed = NULL;
    coder->d_data_compressed = NULL;
    coder->d_temp_huffman = NULL;
    coder->data_compressed_allocated_size = 0;

    GPUJPEG_CUSTOM_TIMER_CREATE(coder->duration_memory_to, return -1);
    GPUJPEG_CUSTOM_TIMER_CREATE(coder->duration_memory_from, return -1);
    GPUJPEG_CUSTOM_TIMER_CREATE(coder->duration_memory_map, return -1);
    GPUJPEG_CUSTOM_TIMER_CREATE(coder->duration_memory_unmap, return -1);
    GPUJPEG_CUSTOM_TIMER_CREATE(coder->duration_preprocessor, return -1);
    GPUJPEG_CUSTOM_TIMER_CREATE(coder->duration_dct_quantization, return -1);
    GPUJPEG_CUSTOM_TIMER_CREATE(coder->duration_huffman_coder, return -1);
    GPUJPEG_CUSTOM_TIMER_CREATE(coder->duration_stream, return -1);
    GPUJPEG_CUSTOM_TIMER_CREATE(coder->duration_in_gpu, return -1);

    return 0;
}

static void
reset_timers(struct gpujpeg_coder* coder)
{
    GPUJPEG_CUSTOM_TIMER_RESET(coder->duration_memory_to);
    GPUJPEG_CUSTOM_TIMER_RESET(coder->duration_memory_from);
    GPUJPEG_CUSTOM_TIMER_RESET(coder->duration_memory_map);
    GPUJPEG_CUSTOM_TIMER_RESET(coder->duration_memory_unmap);
    GPUJPEG_CUSTOM_TIMER_RESET(coder->duration_preprocessor);
    GPUJPEG_CUSTOM_TIMER_RESET(coder->duration_dct_quantization);
    GPUJPEG_CUSTOM_TIMER_RESET(coder->duration_huffman_coder);
    GPUJPEG_CUSTOM_TIMER_RESET(coder->duration_stream);
    GPUJPEG_CUSTOM_TIMER_RESET(coder->duration_in_gpu);
}

int
gpujpeg_coder_allocate_cpu_huffman_buf(struct gpujpeg_coder * coder) {
    cudaMallocHost((void**)&coder->data_quantized, coder->data_size * sizeof(int16_t));
    gpujpeg_cuda_check_error("Coder quantized data host allocation", return -1);

    int16_t* comp_data_quantized = coder->data_quantized;
    for ( int comp = 0; comp < coder->param.comp_count; comp++ ) {
        struct gpujpeg_component* component = &coder->component[comp];
        component->data_quantized = comp_data_quantized;
        comp_data_quantized += (size_t) component->data_width * component->data_height;
    }

    return 0;
}

size_t
gpujpeg_coder_init_image(struct gpujpeg_coder * coder, const struct gpujpeg_parameters * param, const struct gpujpeg_image_parameters * param_image, cudaStream_t stream)
{
    reset_timers(coder);
    if (gpujpeg_parameters_equals(&coder->param, param) && gpujpeg_image_parameters_equals(&coder->param_image, param_image)) {
        // gpujpeg_parameters::{verbose,perf_stats,quality} may change without reconf but store them
        coder->param_image = *param_image;
        coder->param = *param;
        return coder->allocated_gpu_memory_size;
    }
    DEBUG_MSG(param->verbose, "coder image reconfiguration\n");
    size_t allocated_gpu_memory_size = 0;

    // Set parameters
    coder->param_image = *param_image;
    coder->param = *param;

    // Allocate color components
    if ( param->comp_count > coder->component_allocated_size ) {
        coder->component_allocated_size = 0;

        // (Re)allocate color components in host memory
        if (coder->component != NULL) {
            cudaFreeHost(coder->component);
            coder->component = NULL;
        }
        cudaMallocHost((void**)&coder->component, param->comp_count * sizeof(struct gpujpeg_component));
        gpujpeg_cuda_check_error("Coder color component host allocation", return 0);

        // (Re)allocate color components in device memory
        if (coder->d_component != NULL) {
            cudaFree(coder->d_component);
            coder->d_component = NULL;
        }
        cudaMalloc((void**)&coder->d_component, param->comp_count * sizeof(struct gpujpeg_component));
        gpujpeg_cuda_check_error("Coder color component device allocation", return 0);

        coder->component_allocated_size = param->comp_count;
    }
    allocated_gpu_memory_size += coder->component_allocated_size * sizeof(struct gpujpeg_component);

    // Calculate raw data size
    coder->data_raw_size = gpujpeg_image_calculate_size(&coder->param_image);

    // Initialize color components and compute maximum sampling factor to coder->sampling_factor
    coder->data_size = 0;
    coder->sampling_factor.horizontal = 0;
    coder->sampling_factor.vertical = 0;
    for ( int comp = 0; comp < coder->param.comp_count; comp++ ) {
        // Get component
        struct gpujpeg_component* component = &coder->component[comp];

        // Sampling factors
        component->sampling_factor = coder->param.sampling_factor[comp];
        assert(component->sampling_factor.horizontal >= 1 && component->sampling_factor.horizontal <= 15);
        assert(component->sampling_factor.vertical >= 1 && component->sampling_factor.vertical <= 15);
        if ( component->sampling_factor.horizontal > coder->sampling_factor.horizontal ) {
            coder->sampling_factor.horizontal = component->sampling_factor.horizontal;
        }
        if ( component->sampling_factor.vertical > coder->sampling_factor.vertical ) {
            coder->sampling_factor.vertical = component->sampling_factor.vertical;
        }

        // Set type
        component->type = (param->color_space_internal == GPUJPEG_RGB || comp == 0 || comp == 3)
                              ? GPUJPEG_COMPONENT_LUMINANCE
                              : GPUJPEG_COMPONENT_CHROMINANCE;

        // Set proper color component sizes in pixels based on sampling factors
        //
        // Assume unstrided dimensions, eg. for 5x5 I420 - planes Y: 5x5, U: 3x3, V: 3x3 = 43 B
        // This is different to how FFmpeg does (stride 2 also for Y) but consistent eg. with libyuv:
        // https://github.com/Bilibili/libyuv/blob/master/source/convert_to_i420.cc
        int div_h = coder->sampling_factor.horizontal / component->sampling_factor.horizontal;
        int div_v = coder->sampling_factor.vertical / component->sampling_factor.vertical;
        int width = ((coder->param_image.width + div_h - 1) / div_h) * div_h;
        int height = ((coder->param_image.height + div_v - 1) / div_v) * div_v;
        int samp_factor_h = component->sampling_factor.horizontal;
        int samp_factor_v = component->sampling_factor.vertical;
        component->width = (width * samp_factor_h) / coder->sampling_factor.horizontal;
        component->height = (height * samp_factor_v) / coder->sampling_factor.vertical;

        // Compute component MCU size
        component->mcu_size_x = GPUJPEG_BLOCK_SIZE;
        component->mcu_size_y = GPUJPEG_BLOCK_SIZE;
        if ( coder->param.interleaved == 1 ) {
            component->mcu_compressed_size = GPUJPEG_MAX_BLOCK_COMPRESSED_SIZE * samp_factor_h * samp_factor_v;
            component->mcu_size_x *= samp_factor_h;
            component->mcu_size_y *= samp_factor_v;
        } else {
            component->mcu_compressed_size = GPUJPEG_MAX_BLOCK_COMPRESSED_SIZE;
        }
        component->mcu_size = component->mcu_size_x * component->mcu_size_y;

        // Compute allocated data size
        component->data_width = gpujpeg_div_and_round_up(component->width, component->mcu_size_x) * component->mcu_size_x;
        component->data_height = gpujpeg_div_and_round_up(component->height, component->mcu_size_y) * component->mcu_size_y;
        component->data_size = (size_t) component->data_width * component->data_height;
        // Increase total data size
        coder->data_size += component->data_size;

        // Compute component MCU count
        component->mcu_count_x = gpujpeg_div_and_round_up(component->data_width, component->mcu_size_x);
        component->mcu_count_y = gpujpeg_div_and_round_up(component->data_height, component->mcu_size_y);
        component->mcu_count = component->mcu_count_x * component->mcu_count_y;

        // Compute MCU count per segment
        component->segment_mcu_count = coder->param.restart_interval;
        if ( component->segment_mcu_count == 0 ) {
            // If restart interval is disabled, restart interval is equal MCU count
            component->segment_mcu_count = component->mcu_count;
        }

        // Calculate segment count
        component->segment_count = gpujpeg_div_and_round_up(component->mcu_count, component->segment_mcu_count);

        //printf("Subsampling %dx%d, Resolution %d, %d, mcu size %d, mcu count %d\n",
        //    coder->param.sampling_factor[comp].horizontal, coder->param.sampling_factor[comp].vertical,
        //    component->data_width, component->data_height,
        //    component->mcu_compressed_size, component->mcu_count
        //);
    }

    // Maximum component data size for allocated buffers
    coder->data_width = gpujpeg_div_and_round_up(coder->param_image.width, GPUJPEG_BLOCK_SIZE) * GPUJPEG_BLOCK_SIZE;
    coder->data_height = gpujpeg_div_and_round_up(coder->param_image.height, GPUJPEG_BLOCK_SIZE) * GPUJPEG_BLOCK_SIZE;

    // Compute MCU size, MCU count, segment count and compressed data allocation size
    coder->mcu_count = 0;
    coder->mcu_size = 0;
    coder->mcu_compressed_size = 0;
    coder->segment_count = 0;
    coder->data_compressed_size = 0;
    if ( coder->param.interleaved == 1 ) {
        assert(coder->param.comp_count > 0);
        coder->mcu_count = coder->component[0].mcu_count;
        coder->segment_count = coder->component[0].segment_count;
        coder->segment_mcu_count = coder->component[0].segment_mcu_count;
        for ( int comp = 0; comp < coder->param.comp_count; comp++ ) {
            struct gpujpeg_component* component = &coder->component[comp];
            assert(coder->mcu_count == component->mcu_count);
            assert(coder->segment_mcu_count == component->segment_mcu_count);
            coder->mcu_size += component->mcu_size;
            coder->mcu_compressed_size += component->mcu_compressed_size;
        }
    } else {
        assert(coder->param.comp_count > 0);
        coder->mcu_size = coder->component[0].mcu_size;
        coder->mcu_compressed_size = coder->component[0].mcu_compressed_size;
        coder->segment_mcu_count = 0;
        for ( int comp = 0; comp < coder->param.comp_count; comp++ ) {
            struct gpujpeg_component* component = &coder->component[comp];
            assert(coder->mcu_size == component->mcu_size);
            assert(coder->mcu_compressed_size == component->mcu_compressed_size);
            coder->mcu_count += component->mcu_count;
            coder->segment_count += component->segment_count;
        }
    }
    //printf("mcu size %d -> %d, mcu count %d, segment mcu count %d\n", coder->mcu_size, coder->mcu_compressed_size, coder->mcu_count, coder->segment_mcu_count);

    // Allocate segments
    if (coder->segment_count > coder->segment_allocated_size) {
        coder->segment_allocated_size = 0;

        // (Re)allocate segments  in host memory
        if (coder->segment != NULL) {
            cudaFreeHost(coder->segment);
            coder->segment = NULL;
        }
        cudaMallocHost((void**)&coder->segment, coder->segment_count * sizeof(struct gpujpeg_segment));
        gpujpeg_cuda_check_error("Coder segment host allocation", return 0);

        // (Re)allocate segments in device memory
        if (coder->d_segment != NULL) {
            cudaFree(coder->d_segment);
            coder->d_segment = NULL;
        }
        cudaMalloc((void**)&coder->d_segment, coder->segment_count * sizeof(struct gpujpeg_segment));
        gpujpeg_cuda_check_error("Coder segment device allocation", return 0);

        coder->segment_allocated_size = coder->segment_count;
    }
    allocated_gpu_memory_size += coder->segment_allocated_size * sizeof(struct gpujpeg_segment);

    // Prepare segments
    // While preparing segments compute input size and compressed size
    size_t data_index = 0;
    size_t data_compressed_index = 0;
    // Prepare segments based on (non-)interleaved mode
    if ( coder->param.interleaved == 1 ) {
        // Prepare segments for encoding (only one scan for all color components)
        int mcu_index = 0;
        for ( int index = 0; index < coder->segment_count; index++ ) {
            // Prepare segment MCU count
            int mcu_count = coder->segment_mcu_count;
            if ( (mcu_index + mcu_count) >= coder->mcu_count )
                mcu_count = coder->mcu_count - mcu_index;
            // Set parameters for segment
            coder->segment[index].scan_index = 0;
            coder->segment[index].scan_segment_index = index;
            coder->segment[index].mcu_count = mcu_count;
            coder->segment[index].data_compressed_index = data_compressed_index;
            coder->segment[index].data_temp_index = data_compressed_index;
            coder->segment[index].data_compressed_size = 0;
            // Increase parameters for next segment
            data_index += mcu_count * coder->mcu_size;
            data_compressed_index += SEGMENT_ALIGN(mcu_count * coder->mcu_compressed_size);
            mcu_index += mcu_count;
        }
    }
    else {
        // Prepare segments for encoding (one scan for each color component)
        int index = 0;
        for ( int comp = 0; comp < coder->param.comp_count; comp++ ) {
            // Get component
            struct gpujpeg_component* component = &coder->component[comp];
            // Prepare component segments
            int mcu_index = 0;
            for ( int segment = 0; segment < component->segment_count; segment++ ) {
                // Prepare segment MCU count
                int mcu_count = component->segment_mcu_count;
                if ( (mcu_index + mcu_count) >= component->mcu_count )
                    mcu_count = component->mcu_count - mcu_index;
                // Set parameters for segment
                coder->segment[index].scan_index = comp;
                coder->segment[index].scan_segment_index = segment;
                coder->segment[index].mcu_count = mcu_count;
                coder->segment[index].data_compressed_index = data_compressed_index;
                coder->segment[index].data_temp_index = data_compressed_index;
                coder->segment[index].data_compressed_size = 0;
                // Increase parameters for next segment
                data_index += mcu_count * component->mcu_size;
                data_compressed_index += SEGMENT_ALIGN(mcu_count * component->mcu_compressed_size);
                mcu_index += mcu_count;
                index++;
            }
        }
    }
    // Check data size
    //printf("%d == %d\n", coder->data_size, data_index);
    assert(coder->data_size == data_index); (void) data_index;
    // Set compressed size
    coder->data_compressed_size = data_compressed_index;
    //printf("Compressed size %d (segments %d)\n", coder->data_compressed_size, coder->segment_count);

    // Print allocation info
    if ( coder->param.verbose >= GPUJPEG_LL_VERBOSE ) {
        int structures_size = 0;
        structures_size += coder->segment_count * sizeof(struct gpujpeg_segment);
        structures_size += coder->param.comp_count * sizeof(struct gpujpeg_component);
        size_t total_size = 0;
        total_size += structures_size;
        total_size += coder->data_raw_size;
        total_size += coder->data_size;
        total_size += coder->data_size * 2;
        total_size += coder->data_compressed_size;  // for Huffman coding output
        total_size += coder->data_compressed_size;  // for Hiffman coding temp buffer

        printf("\nAllocation Info:\n");
        printf("    Segment Count:            %d\n", coder->segment_count);
        printf("    Allocated Data Size:      %dx%d\n", coder->data_width, coder->data_height);
        printf("    Raw Buffer Size:          %0.1f MiB\n", (double)coder->data_raw_size / (1024.0 * 1024.0));
        printf("    Preprocessor Buffer Size: %0.1f MiB\n", (double)coder->data_size / (1024.0 * 1024.0));
        printf("    DCT Buffer Size:          %0.1f MiB\n", (double)2 * coder->data_size / (1024.0 * 1024.0));
        printf("    Compressed Buffer Size:   %0.1f MiB\n", (double)coder->data_compressed_size / (1024.0 * 1024.0));
        printf("    Huffman Temp buffer Size: %0.1f MiB\n", (double)coder->data_compressed_size / (1024.0 * 1024.0));
        printf("    Structures Size:          %0.1f KiB\n", (double)structures_size / (1024.0));
        printf("    Total GPU Memory Size:    %0.1f MiB\n", (double)total_size / (1024.0 * 1024.0));
        printf("\n");
    }

    // Allocate data buffers for all color components
    coder->d_data_raw = NULL;

    if (coder->component[0].data_width <= 0) {
        fprintf(stderr, "Data width should be positive!\n");
        return 0;
    }
    //for idct we must add some memory - it rounds up the block count, computes all and the extra bytes are omitted
    /// @todo idct_overhead computation looks suspicious
    size_t idct_overhead = ((size_t) GPUJPEG_IDCT_BLOCK_X * GPUJPEG_IDCT_BLOCK_Y * GPUJPEG_IDCT_BLOCK_Z / coder->component[0].data_width + 1)
      * GPUJPEG_BLOCK_SIZE * coder->component[0].data_width;
    if (coder->data_size + idct_overhead > coder->data_allocated_size) {
        coder->data_allocated_size = 0;

        // (Re)allocate preprocessor data in device memory
        if (coder->d_data != NULL) {
            cudaFree(coder->d_data);
            coder->d_data = NULL;
        }
        cudaMalloc((void**)&coder->d_data, (coder->data_size + idct_overhead) * sizeof(uint8_t));
        gpujpeg_cuda_check_error("Coder data device allocation", return 0);

        // Deallocate DCT and quantizer data in host memory, alloc just if needed
        // (gpujpeg_coder_allocate_cpu_huff_data())
        if ( coder->data_quantized != NULL ) {
            cudaFreeHost(coder->data_quantized);
            coder->data_quantized = NULL;
        }

        // (Re)allocated DCT and quantizer data in device memory
        if (coder->d_data_quantized != NULL) {
            cudaFree(coder->d_data_quantized);
            coder->d_data_quantized = NULL;
        }
        cudaMalloc((void**)&coder->d_data_quantized, (coder->data_size + idct_overhead) * sizeof(int16_t));
        gpujpeg_cuda_check_error("Coder quantized data device allocation", return 0);

        coder->data_allocated_size = coder->data_size + idct_overhead;
    }
    allocated_gpu_memory_size += coder->data_allocated_size * sizeof(uint8_t);
    allocated_gpu_memory_size += coder->data_allocated_size * sizeof(int16_t);

    if (coder->encoder) { // clear the buffer for preprocessor when the image size is not divisible by 8x8
        cudaMemset(coder->d_data, 0, coder->data_size * sizeof(uint8_t));
        gpujpeg_cuda_check_error("d_data memset failed", return 0);
    }

    // Set data buffer to color components
    uint8_t* d_comp_data = coder->d_data;
    int16_t* d_comp_data_quantized = coder->d_data_quantized;
    unsigned int data_quantized_index = 0;
    for ( int comp = 0; comp < coder->param.comp_count; comp++ ) {
        struct gpujpeg_component* component = &coder->component[comp];
        component->d_data = d_comp_data;
        component->d_data_quantized = d_comp_data_quantized;
        component->data_quantized_index = data_quantized_index;
        d_comp_data += component->data_width * component->data_height;
        d_comp_data_quantized += component->data_width * component->data_height;
        data_quantized_index += component->data_width * component->data_height;
    }

    // Allocate compressed data
    size_t max_compressed_data_size = coder->data_compressed_size;
    max_compressed_data_size += GPUJPEG_BLOCK_SIZE * GPUJPEG_BLOCK_SIZE;
    //max_compressed_data_size *= 2;
    if (max_compressed_data_size > coder->data_compressed_allocated_size) {
        coder->data_compressed_allocated_size = 0;

        // (Re)allocate huffman coder data in host memory
        if (coder->data_compressed != NULL) {
            cudaHostUnregister(coder->data_compressed);
            free(coder->data_compressed);
            coder->data_compressed = NULL;
        }
        coder->data_compressed = malloc(max_compressed_data_size);
        coder->data_compressed_pinned_sz = max_compressed_data_size / (GPUJPEG_MAX_BLOCK_COMPRESSED_SIZE
          / GPUJPEG_BLOCK_SQUARED_SIZE) / 3; // WxHxCH/3 bytes
        cudaHostRegister(coder->data_compressed, coder->data_compressed_pinned_sz, cudaHostRegisterDefault);
        gpujpeg_cuda_check_error("Coder data compressed host registration", return 0);

        // (Re)allocate huffman coder data in device memory
        if (coder->d_data_compressed != NULL) {
            cudaFree(coder->d_data_compressed);
            coder->d_data_compressed = NULL;
        }
        cudaMalloc((void**)&coder->d_data_compressed, max_compressed_data_size * sizeof(uint8_t));
        gpujpeg_cuda_check_error("Coder data compressed device allocation", return 0);

        // (Re)allocate Huffman coder temporary buffer
        if (coder->d_temp_huffman != NULL) {
            cudaFree(coder->d_temp_huffman);
            coder->d_temp_huffman = NULL;
        }
        cudaMalloc((void**)&coder->d_temp_huffman, max_compressed_data_size * sizeof(uint8_t));
        gpujpeg_cuda_check_error("Huffman temp buffer device allocation", return 0);

        coder->data_compressed_allocated_size = max_compressed_data_size;
    }
    allocated_gpu_memory_size += coder->data_compressed_allocated_size * sizeof(uint8_t);
    allocated_gpu_memory_size += coder->data_compressed_allocated_size * sizeof(uint8_t);

    // Allocate block lists in host memory
    coder->block_count = 0;
    for (int comp = 0; comp < coder->param.comp_count; comp++) {
        coder->block_count += (coder->component[comp].data_width * coder->component[comp].data_height) / (8 * 8);
    }
    if (coder->block_count > coder->block_allocated_size) {
        coder->block_allocated_size = 0;

        // (Re)allocate list of block indices in host memory
        if (coder->block_list != NULL) {
            cudaFreeHost(coder->block_list);
            coder->block_list = NULL;
        }
        cudaMallocHost((void**)&coder->block_list, coder->block_count * sizeof(*coder->block_list));
        gpujpeg_cuda_check_error("Coder block list host allocation", return 0);

        // (Re)allocate list of block indices in device memory
        if (coder->d_block_list != NULL) {
            cudaFree(coder->d_block_list);
            coder->d_block_list = NULL;
        }
        cudaMalloc((void**)&coder->d_block_list, coder->block_count * sizeof(*coder->d_block_list));
        gpujpeg_cuda_check_error("Coder block list device allocation", return 0);

        coder->block_allocated_size = coder->block_count;
    }
    allocated_gpu_memory_size += coder->block_allocated_size * sizeof(*coder->d_block_list);

    // Initialize block lists in host memory
    int block_idx = 0;
    int comp_count = 1;
    if ( coder->param.interleaved == 1 ) {
        comp_count = coder->param.comp_count;
    }
    assert(comp_count >= 1 && comp_count <= GPUJPEG_MAX_COMPONENT_COUNT);
    for (int segment_idx = 0; segment_idx < coder->segment_count; segment_idx++) {
        struct gpujpeg_segment* const segment = &coder->segment[segment_idx];
        segment->block_index_list_begin = block_idx;

        // Non-interleaving mode
        if ( comp_count == 1 ) {
            // Inspect MCUs in segment
            for ( int mcu_index = 0; mcu_index < segment->mcu_count; mcu_index++ ) {
                // Component for the scan
                struct gpujpeg_component* component = &coder->component[segment->scan_index];

                // Offset of component data for MCU
                uint64_t data_index = component->data_quantized_index + (segment->scan_segment_index * component->segment_mcu_count + mcu_index) * component->mcu_size;
                uint64_t component_type = component->type == GPUJPEG_COMPONENT_LUMINANCE ? 0x00 : 0x80;
                uint64_t dc_index = segment->scan_index;
                coder->block_list[block_idx++] = dc_index | component_type | (data_index << 8);
            }
        }
        // Interleaving mode
        else {
            // Encode MCUs in segment
            for ( int mcu_index = 0; mcu_index < segment->mcu_count; mcu_index++ ) {
                //assert(segment->scan_index == 0);
                for ( int comp = 0; comp < comp_count; comp++ ) {
                    struct gpujpeg_component* component = &coder->component[comp];

                    // Prepare mcu indexes
                    int mcu_index_x = (segment->scan_segment_index * component->segment_mcu_count + mcu_index) % component->mcu_count_x;
                    int mcu_index_y = (segment->scan_segment_index * component->segment_mcu_count + mcu_index) / component->mcu_count_x;
                    // Compute base data index
                    int data_index_base = component->data_quantized_index + mcu_index_y * (component->mcu_size * component->mcu_count_x) + mcu_index_x * (component->mcu_size_x * GPUJPEG_BLOCK_SIZE);

                    // For all vertical 8x8 blocks
                    for ( int y = 0; y < component->sampling_factor.vertical; y++ ) {
                        // Compute base row data index
                        int data_index_row = data_index_base + y * (component->mcu_count_x * component->mcu_size_x * GPUJPEG_BLOCK_SIZE);
                        // For all horizontal 8x8 blocks
                        for ( int x = 0; x < component->sampling_factor.horizontal; x++ ) {
                            // Compute 8x8 block data index
                            uint64_t data_index = data_index_row + x * GPUJPEG_BLOCK_SIZE * GPUJPEG_BLOCK_SIZE;
                            uint64_t component_type = component->type == GPUJPEG_COMPONENT_LUMINANCE ? 0x00 : 0x80;
                            uint64_t dc_index = comp;
                            coder->block_list[block_idx++] = dc_index | component_type | (data_index << 8);
                        }
                    }
                }
            }
        }
        segment->block_count = block_idx - segment->block_index_list_begin;
    }
    assert(block_idx == coder->block_count);

    // Copy components to device memory
    cudaMemcpyAsync(coder->d_component, coder->component, coder->param.comp_count * sizeof(struct gpujpeg_component),
                    cudaMemcpyHostToDevice, stream);
    gpujpeg_cuda_check_error("Coder component copy", return 0);

    // Copy block lists to device memory
    cudaMemcpyAsync(coder->d_block_list, coder->block_list, coder->block_count * sizeof(*coder->d_block_list), cudaMemcpyHostToDevice, stream);
    gpujpeg_cuda_check_error("Coder block list copy", return 0);

    // Copy segments to device memory
    cudaMemcpyAsync(coder->d_segment, coder->segment, coder->segment_count * sizeof(struct gpujpeg_segment), cudaMemcpyHostToDevice, stream);
    gpujpeg_cuda_check_error("Coder segment copy", return 0);

    coder->allocated_gpu_memory_size = allocated_gpu_memory_size;

    return allocated_gpu_memory_size;
}

/// @retval 0 on success, nonzero otherwise
int
gpujpeg_coder_get_stats(struct gpujpeg_coder *coder, struct gpujpeg_duration_stats *stats)
{
    stats->duration_memory_to = GPUJPEG_CUSTOM_TIMER_DURATION(coder->duration_memory_to);
    stats->duration_memory_from = GPUJPEG_CUSTOM_TIMER_DURATION(coder->duration_memory_from);
    stats->duration_memory_map = GPUJPEG_CUSTOM_TIMER_DURATION(coder->duration_memory_map);
    stats->duration_memory_unmap = GPUJPEG_CUSTOM_TIMER_DURATION(coder->duration_memory_unmap);
    stats->duration_preprocessor = GPUJPEG_CUSTOM_TIMER_DURATION(coder->duration_preprocessor);
    stats->duration_dct_quantization = GPUJPEG_CUSTOM_TIMER_DURATION(coder->duration_dct_quantization);
    stats->duration_huffman_coder = GPUJPEG_CUSTOM_TIMER_DURATION(coder->duration_huffman_coder);
    stats->duration_stream = GPUJPEG_CUSTOM_TIMER_DURATION(coder->duration_stream);
    stats->duration_in_gpu = GPUJPEG_CUSTOM_TIMER_DURATION(coder->duration_in_gpu);

    return stats->duration_memory_to >= 0 && stats->duration_memory_from >= 0 && stats->duration_memory_map >= 0 &&
                   stats->duration_memory_unmap >= 0 && stats->duration_preprocessor >= 0 &&
                   stats->duration_dct_quantization >= 0 && stats->duration_huffman_coder >= 0 &&
                   stats->duration_stream >= 0 && stats->duration_in_gpu >= 0
               ? 0
               : -1;
}

/* Documented at declaration */
int
gpujpeg_coder_deinit(struct gpujpeg_coder* coder)
{
    if (coder->component != NULL)
        cudaFreeHost(coder->component);
    if (coder->d_component != NULL)
        cudaFree(coder->d_component);
    if ( coder->data_raw != NULL )
        cudaFreeHost(coder->data_raw);
    if ( coder->d_data_raw_allocated != NULL )
        cudaFree(coder->d_data_raw_allocated);
    if ( coder->d_data != NULL )
        cudaFree(coder->d_data);
    if ( coder->data_quantized != NULL )
        cudaFreeHost(coder->data_quantized);
    if ( coder->d_data_quantized != NULL )
        cudaFree(coder->d_data_quantized);
    if ( coder->data_compressed != NULL ) {
        cudaHostUnregister(coder->data_compressed);
        free(coder->data_compressed);
    }
    if ( coder->d_data_compressed != NULL )
        cudaFree(coder->d_data_compressed);
    if ( coder->segment != NULL )
        cudaFreeHost(coder->segment);
    if ( coder->d_segment != NULL )
        cudaFree(coder->d_segment);
    if ( coder->d_temp_huffman != NULL )
        cudaFree(coder->d_temp_huffman);
    if ( coder->block_list != NULL )
        cudaFreeHost(coder->block_list);
    if ( coder->d_block_list != NULL )
        cudaFree(coder->d_block_list);

    GPUJPEG_CUSTOM_TIMER_DESTROY(coder->duration_memory_to, return -1);
    GPUJPEG_CUSTOM_TIMER_DESTROY(coder->duration_memory_from, return -1);
    GPUJPEG_CUSTOM_TIMER_DESTROY(coder->duration_memory_map, return -1);
    GPUJPEG_CUSTOM_TIMER_DESTROY(coder->duration_memory_unmap, return -1);
    GPUJPEG_CUSTOM_TIMER_DESTROY(coder->duration_preprocessor, return -1);
    GPUJPEG_CUSTOM_TIMER_DESTROY(coder->duration_dct_quantization, return -1);
    GPUJPEG_CUSTOM_TIMER_DESTROY(coder->duration_huffman_coder, return -1);
    GPUJPEG_CUSTOM_TIMER_DESTROY(coder->duration_stream, return -1);
    GPUJPEG_CUSTOM_TIMER_DESTROY(coder->duration_in_gpu, return -1);

    return 0;
}

/* Documented at declaration */
size_t
gpujpeg_image_calculate_size(struct gpujpeg_image_parameters* param)
{
    int bpp = gpujpeg_pixel_format_get_unit_size(param->pixel_format);
    if (bpp != 0) {
        return (size_t) param->width * param->height * bpp;
    }
    const int comp_count = gpujpeg_pixel_format_get_comp_count(param->pixel_format);
    switch ( param->pixel_format ) {
    case GPUJPEG_444_U8_P0P1P2:
        assert(comp_count == 3);
        return (size_t) param->width * param->height * comp_count;
    case GPUJPEG_422_U8_P0P1P2:
        assert(comp_count == 3);
        return (size_t) param->width * param->height + (size_t) 2 * ((param->width + 1) / 2) * param->height;
    case GPUJPEG_420_U8_P0P1P2:
        assert(comp_count == 3);
        return (size_t) param->width * param->height + (size_t) 2 * ((param->width + 1) / 2) * ((param->height + 1) / 2);
    default:
        assert(0);
        return 0;
    }
}

static void *gpujpeg_cuda_malloc_host(size_t size) {
    void *ptr;
    GPUJPEG_CHECK_EX(cudaMallocHost(&ptr, size), "Could not alloc host pointer", return NULL);
    return ptr;
}

/* Documented at declaration */
int
gpujpeg_image_load_from_file(const char* filename, uint8_t** image, size_t* image_size)
{
    enum gpujpeg_image_file_format format = gpujpeg_image_get_file_format(filename);
    image_load_delegate_t image_load_delegate = gpujpeg_get_image_load_delegate(format);
    if (image_load_delegate) {
        return image_load_delegate(filename, image_size, (void **) image, gpujpeg_cuda_malloc_host);
    }

    FILE* file;
    file = fopen(filename, "rb");
    if ( !file ) {
        fprintf(stderr, "[GPUJPEG] [Error] Failed open %s for reading: %s\n", filename, strerror(errno));
        return -1;
    }

    if ( *image_size == 0 ) {
        fseek(file, 0, SEEK_END);
        *image_size = ftell(file);
        rewind(file);
    }

    uint8_t* data = NULL;
    cudaMallocHost((void**)&data, *image_size * sizeof(uint8_t));
    gpujpeg_cuda_check_error("Initialize CUDA host buffer", return -1);
    if ( *image_size != fread(data, sizeof(uint8_t), *image_size, file) ) {
        fprintf(stderr, "[GPUJPEG] [Error] Failed to load image data [%zd bytes] from file %s!\n", *image_size, filename);
        return -1;
    }
    fclose(file);

    *image = data;

    return 0;
}

/// replace .XXX with eligible extension
static void
set_file_extension(char *filename,
                   const struct gpujpeg_image_parameters *param_image)
{
    const char *ext = NULL;
    if ( param_image->pixel_format != GPUJPEG_U8 &&
         param_image->color_space != GPUJPEG_RGB ) {
        ext = "y4m";
    }
    else if ( param_image->pixel_format == GPUJPEG_4444_U8_P0123 ) {
        ext = "pam";
    }
    else {
        ext = "pnm";
    }
    strcpy(strrchr(filename, '.') + 1, ext);
}

/* Documented at declaration */
int
gpujpeg_image_save_to_file(char* filename, const uint8_t* image, size_t image_size,
                           const struct gpujpeg_image_parameters* param_image)
{
    if (strrchr(filename, '.') != NULL &&
        strcmp(strrchr(filename, '.'), ".XXX") == 0) {
        set_file_extension(filename, param_image);
    }

    enum gpujpeg_image_file_format format = gpujpeg_image_get_file_format(filename);
    image_save_delegate_t image_save_delegate = gpujpeg_get_image_save_delegate(format);
    if (param_image && image_save_delegate) {
        return image_save_delegate(filename, param_image, (const char *) image);
    }

    FILE* file;
    file = fopen(filename, "wb");
    if ( !file ) {
        fprintf(stderr, "[GPUJPEG] [Error] Failed open %s for writing: %s\n", filename, strerror(errno));
        return -1;
    }

    if ( image_size != fwrite(image, sizeof(uint8_t), image_size, file) ) {
        fprintf(stderr, "[GPUJPEG] [Error] Failed to write image data [%zd bytes] to file %s!\n", image_size, filename);
        return -1;
    }
    fclose(file);

    return 0;
}

int
gpujpeg_image_get_properties(const char *filename, struct gpujpeg_image_parameters *param_image, int file_exists)
{
    const enum gpujpeg_image_file_format format = gpujpeg_image_get_file_format(filename);
    image_probe_delegate_t image_probe_delegate = gpujpeg_get_image_probe_delegate(format);
    if ( image_probe_delegate ) {
        return image_probe_delegate(filename, format, param_image, file_exists);
    }

    if ( get_file_type_cs(format) & FF_CS_RGB ) {
        param_image->color_space = GPUJPEG_RGB;
    }
    if ( get_file_type_cs(format) & FF_CS_YCBCR ) {
        param_image->color_space = GPUJPEG_YCBCR_JPEG;
    }

    switch (format) {
        case GPUJPEG_IMAGE_FILE_UNKNOWN:
            ERROR_MSG("GPUJPEG_IMAGE_FILE_UNKNOWN should not be passed!\n");
            return -1;
        case GPUJPEG_IMAGE_FILE_JPEG:
            ERROR_MSG("GPUJPEG_IMAGE_FILE_JPEG should not be passed!\n");
            return -1;
        case GPUJPEG_IMAGE_FILE_RAW:
            param_image->pixel_format = GPUJPEG_PIXFMT_STD;
            break;
        case GPUJPEG_IMAGE_FILE_GRAY:
            param_image->pixel_format = GPUJPEG_U8;
            break;
        case GPUJPEG_IMAGE_FILE_RGBA:
        case GPUJPEG_IMAGE_FILE_YUVA:
            param_image->pixel_format = GPUJPEG_4444_U8_P0123;
            break;
        case GPUJPEG_IMAGE_FILE_I420:
            param_image->pixel_format = GPUJPEG_420_U8_P0P1P2;
            break;
        case GPUJPEG_IMAGE_FILE_PAM:
        case GPUJPEG_IMAGE_FILE_PGM:
        case GPUJPEG_IMAGE_FILE_PNM:
        case GPUJPEG_IMAGE_FILE_PPM:
        case GPUJPEG_IMAGE_FILE_Y4M:
        case GPUJPEG_IMAGE_FILE_TST:
            GPUJPEG_ASSERT(0 && "image delegate should handle this file type!");
            break;
        case GPUJPEG_IMAGE_FILE_RGB:
        case GPUJPEG_IMAGE_FILE_YUV:
            param_image->pixel_format = GPUJPEG_444_U8_P012;
            break;
    }

    return 1;
}

/* Documented at declaration */
int
gpujpeg_image_destroy(uint8_t* image)
{
    cudaFreeHost(image);

    return 0;
}

/* Documented at declaration */
void
gpujpeg_image_range_info(const char* filename, int width, int height, enum gpujpeg_pixel_format pixel_format)
{
    // Load image
    size_t image_size = 0;
    uint8_t* image = NULL;
    if ( gpujpeg_image_load_from_file(filename, &image, &image_size) != 0 ) {
        fprintf(stderr, "[GPUJPEG] [Error] Failed to load image [%s]!\n", filename);
        return;
    }

    int c_min[3] = {256, 256, 256};
    int c_max[3] = {0, 0, 0};

    if ( pixel_format == GPUJPEG_444_U8_P012 ) {
        uint8_t* in_ptr = image;
        for ( int i = 0; i < width * height; i++ ) {
            for ( int c = 0; c < 3; c++ ) {
                if ( in_ptr[c] < c_min[c] )
                    c_min[c] = in_ptr[c];
                if ( in_ptr[c] > c_max[c] )
                    c_max[c] = in_ptr[c];
            }
            in_ptr += 3;
        }
    }
    else if ( pixel_format == GPUJPEG_422_U8_P1020 ) {
        uint8_t* in_ptr = image;
        for ( int i = 0; i < width * height; i++ ) {
            if ( in_ptr[1] < c_min[0] )
                c_min[0] = in_ptr[1];
            if ( in_ptr[1] > c_max[0] )
                c_max[0] = in_ptr[1];
            if ( i % 2 == 1 ) {
                if ( in_ptr[0] < c_min[1] )
                    c_min[1] = in_ptr[0];
                if ( in_ptr[0] > c_max[1] )
                    c_max[1] = in_ptr[0];
            } else {
                if ( in_ptr[0] < c_min[2] )
                    c_min[2] = in_ptr[0];
                if ( in_ptr[0] > c_max[2] )
                    c_max[2] = in_ptr[0];
            }

            in_ptr += 2;
        }
    }
    else {
        fprintf(stderr, "TODO: implement gpujpeg_image_range_info for pixel format %d.", pixel_format);
        return;
    }

    printf("Image Samples Range:\n");
    for ( int c = 0; c < 3; c++ ) {
        printf("Component %d: %d - %d\n", c + 1, c_min[c], c_max[c]);
    }

    // Destroy image
    gpujpeg_image_destroy(image);
}

/* Documented at declaration */
int
gpujpeg_image_convert(const char* input, const char* output, struct gpujpeg_image_parameters param_image_from,
        struct gpujpeg_image_parameters param_image_to)
{
    fprintf(stderr, "[GPUJPEG] [Error] GPUJPEG conversions are currently defunct, report to developers if needed!\n");
    (void) input, (void) output, (void) param_image_from, (void) param_image_to;
    return -1;
#if 0
    assert(param_image_from.width == param_image_to.width);
    assert(param_image_from.height == param_image_to.height);
    assert(param_image_from.comp_count == param_image_to.comp_count);

    // Load image
    int image_size = gpujpeg_image_calculate_size(&param_image_from);
    uint8_t* image = NULL;
    if ( gpujpeg_image_load_from_file(input, &image, &image_size) != 0 ) {
        fprintf(stderr, "[GPUJPEG] [Error] Failed to load image [%s]!\n", input);
        return -1;
    }

    struct gpujpeg_encoder * encoder = (struct gpujpeg_encoder *) malloc(sizeof(struct gpujpeg_encoder));
    struct gpujpeg_coder * coder = &encoder->coder;
    gpujpeg_set_default_parameters(&coder->param);
    coder->param.color_space_internal = GPUJPEG_RGB;

    // Initialize coder and preprocessor
    coder->param_image = param_image_from;
    GPUJPEG_ASSERT(gpujpeg_coder_init(coder) == 0);
    GPUJPEG_ASSERT(gpujpeg_preprocessor_encoder_init(coder) == 0);

    // Create buffers if not already created
    if (coder->data_raw == NULL) {
        if (cudaSuccess != cudaMallocHost((void**)&coder->data_raw, coder->data_raw_size * sizeof(uint8_t))) {
            return -1;
        }
    }
    if (coder->d_data_raw_allocated == NULL) {
        if (cudaSuccess != cudaMalloc((void**)&coder->d_data_raw_allocated, coder->data_raw_size * sizeof(uint8_t))) {
            return -1;
        }
    }

    coder->d_data_raw = coder->d_data_raw_allocated;

    // Perform preprocessor
    GPUJPEG_ASSERT(cudaMemcpy(coder->d_data_raw, image, coder->data_raw_size * sizeof(uint8_t), cudaMemcpyHostToDevice) == cudaSuccess);
    GPUJPEG_ASSERT(gpujpeg_preprocessor_encode(encoder) == 0);
    // Save preprocessor result
    uint8_t* buffer = NULL;
    GPUJPEG_ASSERT(cudaMallocHost((void**)&buffer, coder->data_size * sizeof(uint8_t)) == cudaSuccess);
    GPUJPEG_ASSERT(buffer != NULL);
    GPUJPEG_ASSERT(cudaMemcpy(buffer, coder->d_data, coder->data_size * sizeof(uint8_t), cudaMemcpyDeviceToHost) == cudaSuccess);
    // Deinitialize decoder
    gpujpeg_coder_deinit(coder);

    // Initialize coder and postprocessor
    coder->param_image = param_image_to;
    GPUJPEG_ASSERT(gpujpeg_coder_init(coder) == 0);
    GPUJPEG_ASSERT(gpujpeg_preprocessor_decoder_init(coder) == 0);
    // Perform postprocessor
    GPUJPEG_ASSERT(cudaMemcpy(coder->d_data, buffer, coder->data_size * sizeof(uint8_t), cudaMemcpyHostToDevice) == cudaSuccess);
    GPUJPEG_ASSERT(gpujpeg_preprocessor_decode(coder, NULL) == 0);
    // Save preprocessor result
    GPUJPEG_ASSERT(cudaMemcpy(coder->data_raw, coder->d_data_raw, coder->data_raw_size * sizeof(uint8_t), cudaMemcpyDeviceToHost) == cudaSuccess);
    if ( gpujpeg_image_save_to_file(output, coder->data_raw, coder->data_raw_size, &param_image_to ) != 0 ) {
        fprintf(stderr, "[GPUJPEG] [Error] Failed to save image [%s]!\n", output);
        return -1;
    }
    // Deinitialize decoder
    gpujpeg_coder_deinit(coder);

    return 0;
#endif
}

#if defined GPUJPEG_USE_GLFW
static void glfw_error_callback(int error, const char* description)
{
    fprintf(stderr, "[GPUJPEG] [Error] GLFW: %s (%d)\n", description, error);
}
#endif

struct gpujpeg_opengl_context {
#ifdef GPUJPEG_USE_GLFW
    GLFWwindow* glfw_window;
#elif defined GPUJPEG_USE_GLX
    Display* glx_display;
    Window glx_window;
#else
    int unused; // avoid empty struct if no extension is present
#endif
};

/* Documented at declaration */
int
gpujpeg_opengl_init(struct gpujpeg_opengl_context **ctx)
{
#ifdef GPUJPEG_USE_OPENGL
    #if defined(GPUJPEG_USE_GLX)
        // Open display
        Display* glx_display = XOpenDisplay(0);
        if ( glx_display == NULL ) {
            fprintf(stderr, "[GPUJPEG] [Error] Failed to open X display!\n");
            return -1;
        }

        // Choose visual
        static int attributes[] = {
            GLX_RGBA,
            GLX_DOUBLEBUFFER,
            GLX_RED_SIZE,   1,
            GLX_GREEN_SIZE, 1,
            GLX_BLUE_SIZE,  1,
            None
        };
        XVisualInfo* visual = glXChooseVisual(glx_display, DefaultScreen(glx_display), attributes);
        if ( visual == NULL ) {
            fprintf(stderr, "[GPUJPEG] [Error] Failed to choose visual!\n");
            return -1;
        }

        // Create OpenGL context
        GLXContext glx_context = glXCreateContext(glx_display, visual, 0, GL_TRUE);
        if ( glx_context == NULL ) {
            fprintf(stderr, "[GPUJPEG] [Error] Failed to create OpenGL context!\n");
            return -1;
        }

        // Create window
        Colormap colormap = XCreateColormap(glx_display, RootWindow(glx_display, visual->screen), visual->visual, AllocNone);
        XSetWindowAttributes swa = { 0 };
        swa.colormap = colormap;
        swa.border_pixel = 0;
        Window glx_window = XCreateWindow(
            glx_display,
            RootWindow(glx_display, visual->screen),
            0, 0, 640, 480,
            0, visual->depth, InputOutput, visual->visual,
            CWBorderPixel | CWColormap | CWEventMask,
            &swa
        );
        // Do not map window to display to keep it hidden
        //XMapWindow(glx_display, glx_window);

        glXMakeCurrent(glx_display, glx_window, glx_context);
    #elif defined(GPUJPEG_USE_GLFW)
        glfwSetErrorCallback(glfw_error_callback);
        if (!glfwInit()) {
            fprintf(stderr, "[GPUJPEG] [Error] glfwInit failed!\n");
            return -1;
        }

        glfwWindowHint(GLFW_VISIBLE, GL_FALSE);
        GLFWwindow* window = glfwCreateWindow(640, 480, "", NULL, NULL);
        if (window == NULL) {
            fprintf(stderr, "[GPUJPEG] [Error] Cannot create GLFW window!\n");
            return -1;
        }
        glfwMakeContextCurrent(window);
    #endif
        GLenum err = glewInit();
        if (err != GLEW_OK) {
            fprintf(stderr, "[GPUJPEG] [Error] glewInit: %s\n", glewGetErrorString(err));
            return -1;
        }
        struct gpujpeg_opengl_context *s = calloc(1, sizeof *s);
    #if defined(GPUJPEG_USE_GLFW)
        s->glfw_window = window;
    #elif defined(GPUJPEG_USE_GLX)
        s->glx_display = glx_display;
        s->glx_window = glx_window;
    #else
        fprintf(stderr, "[GPUJPEG] [Error] gpujpeg_opengl_init not implemented in current build!\n");
        fprintf(stderr, "[GPUJPEG] [Note] You can still use custom OpenGL context!\n");
        free(s);
        return -1;
    #endif

        *ctx = s;
        return 0;
#else
    (void) ctx;
    GPUJPEG_MISSING_OPENGL(return -2);
#endif
}

void
gpujpeg_opengl_destroy(struct gpujpeg_opengl_context *s)
{
    if (s == NULL) {
        return;
    }
#ifdef GPUJPEG_USE_GLFW
    glfwDestroyWindow(s->glfw_window);
    glfwTerminate();
#elif defined GPUJPEG_USE_GLX
    XDestroyWindow(s->glx_display, s->glx_window);
    XCloseDisplay(s->glx_display);
#endif // defined GPUJPEG_USE_GLX
    free(s);
}

/* Documented at declaration */
int
gpujpeg_opengl_texture_create(int width, int height, uint8_t* data)
{
#ifdef GPUJPEG_USE_OPENGL
    GLuint texture_id = 0;

    glGenTextures(1, &texture_id);
    if (texture_id == 0) {
        return 0;
    }
    glBindTexture(GL_TEXTURE_2D, texture_id);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);

    glBindTexture(GL_TEXTURE_2D, 0);

    return texture_id;
#else
    (void) width, (void) height, (void) data;
    GPUJPEG_MISSING_OPENGL(return 0);
#endif
}

/* Documented at declaration */
int
gpujpeg_opengl_texture_set_data(int texture_id, uint8_t* data)
{
#ifdef GPUJPEG_USE_OPENGL
    glBindTexture(GL_TEXTURE_2D, texture_id);

    int width = 0;
    int height = 0;
    glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &width);
    glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &height);
    assert(width != 0 && height != 0);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);

    glBindTexture(GL_TEXTURE_2D, 0);
    return 0;
#else
    (void) texture_id, (void) data;
    GPUJPEG_MISSING_OPENGL(return -1);
#endif
}

/* Documented at declaration */
int
gpujpeg_opengl_texture_get_data(int texture_id, uint8_t* data, size_t* data_size)
{
#ifdef GPUJPEG_USE_OPENGL
    glBindTexture(GL_TEXTURE_2D, texture_id);

    int width = 0;
    int height = 0;
    glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &width);
    glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &height);
    assert(width != 0 && height != 0);

    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
    if ( data_size != NULL )
        *data_size = width * height * 3;

    glBindTexture(GL_TEXTURE_2D, 0);
    return 0;
#else
    (void) texture_id, (void) data, (void) data_size;
    GPUJPEG_MISSING_OPENGL(return -1);
#endif
}

/* Documented at declaration */
void
gpujpeg_opengl_texture_destroy(int texture_id)
{
#ifdef GPUJPEG_USE_OPENGL
     glDeleteTextures(1, (GLuint*)&texture_id);
#else
    (void) texture_id;
    GPUJPEG_MISSING_OPENGL();
#endif
}

/* Documented at declaration */
struct gpujpeg_opengl_texture*
gpujpeg_opengl_texture_register(int texture_id, enum gpujpeg_opengl_texture_type texture_type)
{
    struct gpujpeg_opengl_texture* texture = NULL;
    cudaMallocHost((void**)&texture, sizeof(struct gpujpeg_opengl_texture));
    assert(texture != NULL);

    texture->texture_id = texture_id;
    texture->texture_type = texture_type;
    texture->texture_width = 0;
    texture->texture_height = 0;
    texture->texture_pbo_id = 0;
    texture->texture_pbo_type = 0;
    texture->texture_pbo_resource = 0;
    texture->texture_callback_param = NULL;
    texture->texture_callback_attach_opengl = NULL;
    texture->texture_callback_detach_opengl = NULL;

#ifdef GPUJPEG_USE_OPENGL
    glBindTexture(GL_TEXTURE_2D, texture->texture_id);
    glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &texture->texture_width);
    glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &texture->texture_height);
    glBindTexture(GL_TEXTURE_2D, 0);
    assert(texture->texture_width != 0 && texture->texture_height != 0);

    // Select PBO type
    if ( texture->texture_type == GPUJPEG_OPENGL_TEXTURE_READ ) {
        texture->texture_pbo_type = GL_PIXEL_PACK_BUFFER;
    } else if ( texture->texture_type == GPUJPEG_OPENGL_TEXTURE_WRITE ) {
        texture->texture_pbo_type = GL_PIXEL_UNPACK_BUFFER;
    } else {
        assert(0);
    }

    // Create PBO
    if (glGenBuffers == NULL) {
        fprintf(stderr, "GLEW wasn't initialized!\n");
        return NULL;
    }
    glGenBuffers(1, (GLuint*)&texture->texture_pbo_id);
    if (texture->texture_pbo_id == 0) {
        fprintf(stderr, "glGenBuffers returned zero!\n");
        return NULL;
    }
    glBindBuffer(texture->texture_pbo_type, texture->texture_pbo_id);
    glBufferData(texture->texture_pbo_type, texture->texture_width * texture->texture_height * 3 * sizeof(uint8_t), NULL, GL_DYNAMIC_DRAW);
    glBindBuffer(texture->texture_pbo_type, 0);

    // Create CUDA PBO Resource
    cudaGraphicsGLRegisterBuffer(&texture->texture_pbo_resource, texture->texture_pbo_id, cudaGraphicsMapFlagsNone);
    gpujpeg_cuda_check_error("Register OpenGL buffer", return NULL);

    return texture;
#else
    GPUJPEG_MISSING_OPENGL(return NULL);
#endif
}

/* Documented at declaration */
void
gpujpeg_opengl_texture_unregister(struct gpujpeg_opengl_texture* texture)
{
#ifdef GPUJPEG_USE_OPENGL
    assert(texture != NULL);

    if ( texture->texture_pbo_id != 0 ) {
     glDeleteBuffers(1, (GLuint*)&texture->texture_pbo_id);
    }
    if ( texture->texture_pbo_resource != NULL ) {
        cudaGraphicsUnregisterResource(texture->texture_pbo_resource);
    }
    cudaFreeHost(texture);
#else
    (void) texture;
    GPUJPEG_MISSING_OPENGL(return);
#endif
}

/* Documented at declaration */
uint8_t*
gpujpeg_opengl_texture_map(struct gpujpeg_opengl_texture* texture, size_t* data_size)
{
    assert(texture->texture_pbo_resource != NULL);
    assert((texture->texture_callback_attach_opengl == NULL && texture->texture_callback_detach_opengl == NULL) ||
           (texture->texture_callback_attach_opengl != NULL && texture->texture_callback_detach_opengl != NULL));

    // Attach OpenGL context by callback
    if ( texture->texture_callback_attach_opengl != NULL )
        texture->texture_callback_attach_opengl(texture->texture_callback_param);

#ifdef GPUJPEG_USE_OPENGL
    uint8_t* d_data = NULL;

    if ( texture->texture_type == GPUJPEG_OPENGL_TEXTURE_READ ) {
        assert(texture->texture_pbo_type == GL_PIXEL_PACK_BUFFER);

        glBindTexture(GL_TEXTURE_2D, texture->texture_id);
        glBindBuffer(GL_PIXEL_PACK_BUFFER, texture->texture_pbo_id);

        glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_UNSIGNED_BYTE, 0);

        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    // Map pixel buffer object to cuda
    cudaGraphicsMapResources(1, &texture->texture_pbo_resource, 0);
    gpujpeg_cuda_check_error("Encoder map texture PBO resource", return NULL);

    // Get device data pointer to pixel buffer object data
    size_t d_data_size;
    cudaGraphicsResourceGetMappedPointer((void **)&d_data, &d_data_size, texture->texture_pbo_resource);
    gpujpeg_cuda_check_error("Encoder get device pointer for texture PBO resource", return NULL);
    if ( data_size != NULL )
        *data_size = d_data_size;

    return d_data;
#else
    (void) data_size;
    GPUJPEG_MISSING_OPENGL(return NULL);
#endif
}

/* Documented at declaration */
void
gpujpeg_opengl_texture_unmap(struct gpujpeg_opengl_texture* texture)
{
    // Unmap pbo
    cudaGraphicsUnmapResources(1, &texture->texture_pbo_resource, 0);
    gpujpeg_cuda_check_error("Encoder unmap texture PBO resource", {});

#ifdef GPUJPEG_USE_OPENGL
    if ( texture->texture_type == GPUJPEG_OPENGL_TEXTURE_WRITE ) {
        assert(texture->texture_pbo_type == GL_PIXEL_UNPACK_BUFFER);

        glBindTexture(GL_TEXTURE_2D, texture->texture_id);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, texture->texture_pbo_id);

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, texture->texture_width, texture->texture_height, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        glBindTexture(GL_TEXTURE_2D, 0);
        glFinish();
    }

    // Dettach OpenGL context by callback
    if ( texture->texture_callback_detach_opengl != NULL )
        texture->texture_callback_detach_opengl(texture->texture_callback_param);
#else
    GPUJPEG_MISSING_OPENGL(return);
#endif
}

int gpujpeg_version(void)
{
#ifdef HAVE_GPUJPEG_VERSION_H
    return GPUJPEG_VERSION_INT;
#else
    return 0;
#endif // defined HAVE_GPUJPEG_VERSION_H
}

const char *gpujpeg_version_to_string(int version)
{
    thread_local static char buf[128];
    snprintf(buf, sizeof buf, "%d.%d.%d", version >> 16, (version >> 8) & 0xFF, version & 0xFF);
    return buf;
}

const char*
gpujpeg_subsampling_get_name(int comp_count, const struct gpujpeg_component_sampling_factor *sampling_factor)
{
    thread_local static char buf[128];
    if (comp_count == 1) { // monochrome
        strcpy(buf, "4:0:0");
        return buf;
    }

    const int J = 4;
    if (comp_count == 2 && sampling_factor[0].vertical == sampling_factor[1].vertical) { // monochrome + alpha
        snprintf(buf, sizeof buf, "4:0:0:%d", J / sampling_factor[0].horizontal * sampling_factor[1].horizontal);
        return buf;
    }

    if (sampling_factor[1].horizontal == sampling_factor[2].horizontal
            && sampling_factor[1].vertical == sampling_factor[2].horizontal && (comp_count == 3 ||
                (comp_count == 4 && sampling_factor[0].vertical == sampling_factor[3].vertical))) {
        int a = J / sampling_factor[0].horizontal * sampling_factor[1].horizontal;
        int vert_change = 2 / sampling_factor[0].vertical * sampling_factor[1].vertical == 2; // 1 if there is a vertical chroma change between 2 lines, 0 otherwise
        int b = a * vert_change;
        snprintf(buf, sizeof buf, "%d:%d:%d", J, a, b);
        if (comp_count == 4) {
            snprintf(buf + strlen(buf), sizeof buf - strlen(buf), ":%d", J / sampling_factor[0].horizontal * sampling_factor[3].horizontal);
        }
        return buf;
    }

    if ( gpujpeg_make_sampling_factor2(comp_count, sampling_factor) == GPUJPEG_SUBSAMPLING_442 ) {
        snprintf(buf, sizeof buf, "4:4:2");
        return buf;
    }
    if ( gpujpeg_make_sampling_factor2(comp_count, sampling_factor) == GPUJPEG_SUBSAMPLING_421 ) {
        snprintf(buf, sizeof buf, "4:2:1");
        return buf;
    }

    // other cases - simply write the subsampling factors in format v0-h0:v1-h1[:...]
    buf[0] = '\0';
    for (int i = 0; i < comp_count; ++i) {
        snprintf(buf + strlen(buf), sizeof buf - strlen(buf), "%s%d-%d", i != 0 ? ":" : "",
                sampling_factor[i].horizontal, sampling_factor[i].vertical);
    }

    return buf;
}

GPUJPEG_API gpujpeg_sampling_factor_t
gpujpeg_subsampling_from_name(const char* subsampling) {
    enum { UNKNOWN = 0 };
    if ( strcmp(subsampling, "help") == 0 ) {
        printf("Set subsampling in usual J:a:b[:alpha] format, eg. 4:2:2. Colons are optional.\n");
        printf("Non-standard subsamplings 4:4:2 and 4:2:1 are allowed.\n");
        return UNKNOWN;
    }
    int J = 0;
    int a = 0;
    int b = 0;
    int alpha = 0;
    int comp_count = sscanf(subsampling, "%d:%d:%d:%d", &J, &a, &b, &alpha);
    if (comp_count == 1 && J / (1000 == 4 || J / 100 == 4)) { // format without ':' delim
        if ( J / 1000 == 4 ) {
            comp_count = 4;
            alpha = J % 10;
            J /= 10;
        }
        else {
            comp_count = 3;
        }
        J -= 400;
        a = J / 10;
        b = J % 10;
        J = 4;
    }
    if (comp_count < 3 || J != 4 || (alpha != 4 && alpha != 0)) {
        return UNKNOWN;
    }
    if (a != b && b != 0) { // not in J:a:b format
        if ( a == 4 && b == 2 && alpha == 0 ) {
            return GPUJPEG_SUBSAMPLING_442;
        }
        if ( a == 2 && b == 1 && alpha == 0 ) {
            return GPUJPEG_SUBSAMPLING_421;
        }
        return UNKNOWN;
    }
    if (a == 0 && b == 0 && alpha == 0) {
        return GPUJPEG_SUBSAMPLING_400;
    }
    struct gpujpeg_component_sampling_factor factor[GPUJPEG_MAX_COMPONENT_COUNT] = {0};
    factor[0].horizontal = 4 / a;
    factor[0].vertical = a == b ? 1 : 2;
    factor[1].horizontal = factor[2].horizontal = 1;
    factor[1].vertical = factor[2].vertical = 1;
    if (alpha == 0) {
        return  gpujpeg_make_sampling_factor2(GPUJPEG_3_COMPONENTS, factor);
    }
    factor[3].horizontal = factor[0].horizontal;
    factor[3].vertical = factor[0].vertical;
    return gpujpeg_make_sampling_factor2(GPUJPEG_4_COMPONENTS, factor);
}

const char*
gpujpeg_color_space_get_name(enum gpujpeg_color_space color_space)
{
    switch ( color_space ) {
    case GPUJPEG_NONE:
        return "None";
    case GPUJPEG_RGB:
        return "RGB";
    case GPUJPEG_YUV:
        return "YUV";
    case GPUJPEG_YCBCR_BT601:
        return "YCbCr BT.601 (limtted range)";
    case GPUJPEG_YCBCR_BT601_256LVLS:
        return "YCbCr BT.601 256 Levels (YCbCr JPEG)";
    case GPUJPEG_YCBCR_BT709:
        return "YCbCr BT.709 (limited range)";
    }
    if ( color_space == GPUJPEG_CS_DEFAULT ) {
        return "(default CS)";
    }
    return "Unknown";
}

enum gpujpeg_pixel_format
gpujpeg_pixel_format_by_name(const char *name)
{
    for (size_t i = 0; i < sizeof gpujpeg_pixel_format_desc / sizeof gpujpeg_pixel_format_desc[0]; ++i) {
        if (strcmp(gpujpeg_pixel_format_desc[i].name, name) == 0) {
            return gpujpeg_pixel_format_desc[i].pixel_format;
        }
    }
    if (strcmp(name, "help") == 0) {
        gpujpeg_print_pixel_formats();
    }
    return GPUJPEG_PIXFMT_NONE;
}

void
gpujpeg_print_pixel_formats(void)
{
    printf("                          u8 (grayscale)          420-u8-p0p1p2 (planar 4:2:0)\n"
           "                          422-u8-p1020 (eg. UYVY) 422-u8-p0p1p2 (planar 4:2:2)\n"
           "                          444-u8-p012 (eg. RGB)   444-u8-p0p1p2 (planar 4:4:4)\n"
           "                          4444-u8-p0123 (RGBA)\n");
}

enum gpujpeg_color_space
gpujpeg_color_space_by_name(const char* name)
{
    if ( strcmp(name, "rgb") == 0 )  {
        return GPUJPEG_RGB;
    }
    if ( strcmp(name, "yuv") == 0 ) {
        return GPUJPEG_YUV;
    }
    if ( strcmp(name, "ycbcr") == 0 ) {
        return GPUJPEG_YCBCR;
    }
    if ( strcmp(name, "ycbcr-jpeg") == 0 ) {
        return GPUJPEG_YCBCR_BT601_256LVLS;
    }
    if ( strcmp(name, "ycbcr-bt601") == 0 ) {
        return GPUJPEG_YCBCR_BT601;
    }
    if ( strcmp(name, "ycbcr-bt709") == 0 ) {
        return GPUJPEG_YCBCR_BT709;
    }
    if ( strcmp(name, "help") == 0 ) {
        printf("Available color spaces:\n"
               "- rgb\n"
               "- yuv (deprecated)\n"
               "- ycbcr       - same as ycbcr-bt709\n"
               "- ycbcr-jpeg  - BT.601 full range\n"
               "- ycbcr-bt601 - limitted range\n"
               "- ycbcr-bt709 - limitted range\n");
    }
    return GPUJPEG_NONE;
}

int
gpujpeg_pixel_format_get_comp_count(enum gpujpeg_pixel_format pixel_format)
{
    for (size_t i = 0; i < sizeof gpujpeg_pixel_format_desc / sizeof gpujpeg_pixel_format_desc[0]; ++i) {
        if (gpujpeg_pixel_format_desc[i].pixel_format == pixel_format) {
            return gpujpeg_pixel_format_desc[i].comp_count;
        }
    }
    return 0;
}

const char*
gpujpeg_pixel_format_get_name(enum gpujpeg_pixel_format pixel_format)
{
    for (size_t i = 0; i < sizeof gpujpeg_pixel_format_desc / sizeof gpujpeg_pixel_format_desc[0]; ++i) {
        if (gpujpeg_pixel_format_desc[i].pixel_format == pixel_format) {
            return gpujpeg_pixel_format_desc[i].name;
        }
    }
    return NULL;
}

const struct gpujpeg_component_sampling_factor*
gpujpeg_pixel_format_get_sampling_factor(enum gpujpeg_pixel_format pixel_format)
{
    for (size_t i = 0; i < sizeof gpujpeg_pixel_format_desc / sizeof gpujpeg_pixel_format_desc[0]; ++i) {
        if (gpujpeg_pixel_format_desc[i].pixel_format == pixel_format) {
            return gpujpeg_pixel_format_desc[i].sampling_factor;
        }
    }
    return NULL;
}

int
gpujpeg_pixel_format_get_unit_size(enum gpujpeg_pixel_format pixel_format)
{
    for (size_t i = 0; i < sizeof gpujpeg_pixel_format_desc / sizeof gpujpeg_pixel_format_desc[0]; ++i) {
        if (gpujpeg_pixel_format_desc[i].pixel_format == pixel_format) {
            return gpujpeg_pixel_format_desc[i].bpp;
        }
    }
    return 0;
}

int
gpujpeg_pixel_format_is_interleaved(enum gpujpeg_pixel_format pixel_format)
{
    return gpujpeg_pixel_format_get_comp_count(pixel_format) > 1 && !gpujpeg_pixel_format_is_planar(pixel_format);
}

int gpujpeg_pixel_format_is_planar(enum gpujpeg_pixel_format pixel_format)
{
    for (size_t i = 0; i < sizeof gpujpeg_pixel_format_desc / sizeof gpujpeg_pixel_format_desc[0]; ++i) {
        if (gpujpeg_pixel_format_desc[i].pixel_format == pixel_format) {
            return (gpujpeg_pixel_format_desc[i].flags & PLANAR) != 0U;
        }
    }
    return -1;
}

/**
 * @retval >=0 on success
 * @retval  <0 on failure
 */
float gpujpeg_custom_timer_get_duration(cudaEvent_t start, cudaEvent_t stop) {
    float elapsedTime = NAN;
    cudaEventSynchronize(stop);
    gpujpeg_cuda_check_error("cudaEventSynchronize", return -1);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    gpujpeg_cuda_check_error("cudaEventElapsedTime", return -1);
    return elapsedTime;
}

void
gpujpeg_device_reset(void)
{
    cudaDeviceReset();
}

/**
 * If gpujpeg_coder.param.perf_stat is on, finishes statistics counting - mainly CPU time and its store for aggregate
 * printout in coder_process_stats_overall() - and prints per-image statistics.
 */
void
coder_process_stats(struct gpujpeg_coder* coder)
{
    if ( !coder->param.perf_stats ) {
        return;
    }
    coder->stop_time = gpujpeg_get_time();
    const double duration_ms = (coder->stop_time - coder->start_time) * 1000.0;
    const double duration_init_ms = (coder->init_end_time - coder->start_time) * 1000.0;

    if ( coder->frames == 0 ) {
        coder->first_frame_duration = duration_ms;
    }
    coder->aggregate_duration += duration_ms;
    coder->frames += 1;

    if ( coder->param.verbose < GPUJPEG_LL_STATUS ) {
        return;
    }

    struct gpujpeg_duration_stats stats;
    gpujpeg_coder_get_stats(coder, &stats);

    if ( coder->encoder ) {
        if ( coder->param.verbose >= GPUJPEG_LL_VERBOSE ) {
            printf(" -(Re)initialization:%10.4f ms\n", duration_init_ms);
            printf(" -Copy To Device:    %10.4f ms\n", stats.duration_memory_to);
            if ( stats.duration_memory_map != 0.0 && stats.duration_memory_unmap != 0.0 ) {
                printf(" -OpenGL Memory Map: %10.4f ms\n", stats.duration_memory_map);
                printf(" -OpenGL Memory Unmap:%9.4f ms\n", stats.duration_memory_unmap);
            }
            printf(" -Preprocessing:     %10.4f ms\n", stats.duration_preprocessor);
            printf(" -DCT & Quantization:%10.4f ms\n", stats.duration_dct_quantization);
            printf(" -Huffman Encoder:   %10.4f ms\n", stats.duration_huffman_coder);
            printf(" -Copy From Device:  %10.4f ms\n", stats.duration_memory_from);
            printf(" -Stream Formatter:  %10.4f ms\n", stats.duration_stream);
        }
        printf("Encode Image GPU:    %10.4f ms (only in-GPU processing)\n", stats.duration_in_gpu);
        printf("Encode Image Bare:   %10.4f ms (without copy to/from GPU memory)\n",
               duration_ms - stats.duration_memory_to - stats.duration_memory_from);
        printf("Encode Image:        %10.4f ms\n", duration_ms);
    }
    else {
        if ( coder->param.verbose >= GPUJPEG_LL_VERBOSE ) {
            printf(" -(Re)initialization:%10.4f ms\n", duration_init_ms);
            printf(" -Stream Reader:     %10.4f ms\n", stats.duration_stream);
            printf(" -Copy To Device:    %10.4f ms\n", stats.duration_memory_to);
            printf(" -Huffman Decoder:   %10.4f ms\n", stats.duration_huffman_coder);
            printf(" -DCT & Quantization:%10.4f ms\n", stats.duration_dct_quantization);
            printf(" -Postprocessing:    %10.4f ms\n", stats.duration_preprocessor);
            printf(" -Copy From Device:  %10.4f ms\n", stats.duration_memory_from);
            if ( stats.duration_memory_map != 0.0 && stats.duration_memory_unmap != 0.0 ) {
                printf(" -OpenGL Memory Map: %10.4f ms\n", stats.duration_memory_map);
                printf(" -OpenGL Memory Unmap:%9.4f ms\n", stats.duration_memory_unmap);
            }
        }
        printf("Decode Image GPU:    %10.4f ms (only in-GPU processing)\n", stats.duration_in_gpu);
        printf("Decode Image Bare:   %10.4f ms (without copy to/from GPU memory)\n",
               duration_ms - stats.duration_memory_to - stats.duration_memory_from);
        printf("Decode Image:        %10.4f ms\n", duration_ms);
    }
}

/**
 * @brief prints overal statistics
 * @sa coder_process_stats
 *
 * call on encoder/decoder destroy
 */
void
coder_process_stats_overall(struct gpujpeg_coder* coder) {
    if ( !coder->param.perf_stats || coder->frames <= 1 ) { // aggregate stats not needed for 0 or 1 frame
        return;
    }
    if ( coder->param.verbose <= GPUJPEG_LL_QUIET ) {
        return;
    }
    printf("\n");
    printf("Avg %s Duration: %10.4f ms\n", coder->encoder ? "Encode" : "Decode",
           coder->aggregate_duration / (double)coder->frames);
    if ( coder->param.verbose >= GPUJPEG_LL_VERBOSE ) {
        printf("Avg w/o 1st Iter:    %10.4f ms\n",
               (coder->aggregate_duration - coder->first_frame_duration) / ((double)coder->frames - 1));
    }
    printf("\n");
}

char*
format_number_with_delim(size_t num, char* buf, size_t buflen)
{
    assert(buflen >= 1);
    buf[buflen - 1] = '\0';
    char* ptr = buf + buflen - 1;
    int grp_count = 0;
    do {
        if ( ptr == buf || (grp_count == 3 && ptr == buf + 1) ) {
            snprintf(buf, buflen, "%s", "ERR");
            return buf;
        }
        if ( grp_count++ == 3 ) {
            grp_count = 1;
            *--ptr = ',';
        }
        *--ptr = (char)('0' + (num % 10));
        num /= 10;
    } while ( num != 0 );

    return ptr;
}

/* vi: set expandtab sw=4 : */
