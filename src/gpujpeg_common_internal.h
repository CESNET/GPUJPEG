/*
 * Copyright (c) 2011-2023, CESNET
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
/**
 * @file
 * This file contains common definitions and declarations that doesn't need to
 * be exposed to public API.
 */
 
#ifndef GPUJPEG_COMMON_INTERNAL_H
#define GPUJPEG_COMMON_INTERNAL_H

#include <cuda_runtime.h>
#include <math.h> // NAN
#include <stdio.h>
#include <stdlib.h>
#include "../libgpujpeg/gpujpeg_common.h"
#include "../libgpujpeg/gpujpeg_type.h"
#include "gpujpeg_util.h"

// VS 2015 compat
#if defined _MSC_VER && _MSC_VER <= 1900
#define __func__ ""
#define _Static_assert(cond, msg)
#endif // VS <=2015

/** Contants */
#define GPUJPEG_BLOCK_SIZE                      8
#define GPUJPEG_BLOCK_SQUARED_SIZE              64
#define GPUJPEG_MAX_BLOCK_COMPRESSED_SIZE       (GPUJPEG_BLOCK_SIZE * GPUJPEG_BLOCK_SIZE * 8)

#define GPUJPEG_IDCT_BLOCK_X	8
#define GPUJPEG_IDCT_BLOCK_Y 	8
#define GPUJPEG_IDCT_BLOCK_Z 	2

/** Maximum JPEG header size (MUST be divisible by 4!!!) */
#define GPUJPEG_MAX_HEADER_SIZE                 (65536 - 100)

/**
 * JPEG component type
 */
enum gpujpeg_component_type {
    GPUJPEG_COMPONENT_LUMINANCE = 0,
    GPUJPEG_COMPONENT_CHROMINANCE = 1,
    GPUJPEG_COMPONENT_TYPE_COUNT = 2
};

/**
 * JPEG huffman type
 */
enum gpujpeg_huffman_type {
    GPUJPEG_HUFFMAN_DC = 0,
    GPUJPEG_HUFFMAN_AC = 1,
    GPUJPEG_HUFFMAN_TYPE_COUNT = 2
};

enum {
    GPUJPEG_3_COMPONENTS = 3,
    GPUJPEG_4_COMPONENTS = 4,
};

/// unconditional assert
#define GPUJPEG_ASSERT(cond) do { if (!(cond)) { fprintf(stderr, "%s:%d: %s: Assertion `" #cond "' failed.\n", __FILE__, __LINE__, __func__); abort(); } } while(0)

#define ERROR_MSG(...) do { fprintf(stderr, "[GPUJPEG] [Error] " __VA_ARGS__); } while(0)
#define WARN_MSG(...) do { fprintf(stderr, "[GPUJPEG] [Warning] " __VA_ARGS__); } while(0)
#define VERBOSE_MSG(log_level, ...) do { if (log_level >= 1) fprintf(stderr, "[GPUJPEG] [Warning] " __VA_ARGS__); } while(0)
#define DEBUG_MSG(log_level, ...) do { if (log_level >= 2) fprintf(stderr, "[GPUJPEG] [Debug] " __VA_ARGS__); } while(0)
#define DEBUG2_MSG(log_level, ...) do { if (log_level >= 3) fprintf(stderr, "[GPUJPEG] [Debug2] " __VA_ARGS__); } while(0)

#ifndef MIN
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#endif

struct gpujpeg_timer {
    int started;
    cudaEvent_t start;
    cudaEvent_t stop;
};

#define GPUJPEG_CUSTOM_TIMER_CREATE(name, err_action) \
    do { \
        GPUJPEG_CHECK(cudaEventCreate(&(name).start), err_action); \
        GPUJPEG_CHECK(cudaEventCreate(&(name).stop), err_action); \
        (name).started = 0; \
    } while (0)

#define GPUJPEG_CUSTOM_TIMER_DESTROY(name, err_action) \
    do { \
        GPUJPEG_CHECK(cudaEventDestroy((name).start), err_action); \
        GPUJPEG_CHECK(cudaEventDestroy((name).stop), err_action); \
    } while (0)

/**
 * Start timer
 *
 * @param name
 * @todo stream
 */
#define GPUJPEG_CUSTOM_TIMER_START(name, record_perf, stream, err_action) \
    if (record_perf) { \
        (name).started = 1; \
        GPUJPEG_CHECK(cudaEventRecord((name).start, stream), err_action); \
    } else { \
        (name).started = -1; \
    }

/**
 * Stop timer
 *
 * @param name
 */
#define GPUJPEG_CUSTOM_TIMER_STOP(name, record_perf, stream, err_action) \
    if (record_perf) { \
        GPUJPEG_CHECK(cudaEventRecord((name).stop, stream), err_action); \
    }

/**
 * Get duration for timer
 *
 * @param name
 */
#define GPUJPEG_CUSTOM_TIMER_DURATION(name) \
    (name).started == 1 ? gpujpeg_custom_timer_get_duration((name).start, (name).stop) : (name).started == 0 ? 0 : ( fprintf(stderr, "Debug timer disabled!\n"), 0)

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/**
 * JPEG segment structure. Segment is data in scan generated by huffman coder
 * for N consecutive MCUs, where N is restart interval (e.g. data for MCUs between
 * restart markers)
 */
struct gpujpeg_segment
{
    /// Scan index (in which segment belongs)
    int scan_index;
    /// Segment index in the scan (position of segment in scan starting at 0)
    int scan_segment_index;
    /// MCU count in segment
    int mcu_count;

    /// Data compressed index (output/input data from/to segment for encoder/decoder)
    size_t data_compressed_index;
    /// Date temp index (temporary data of segment in CC 2.0 encoder)
    size_t data_temp_index;
    /// Data compressed size (output/input data from/to segment for encoder/decoder)
    size_t data_compressed_size;

    /// Offset of first block index
    int block_index_list_begin;
    /// Number of blocks of the segment
    int block_count;
};

/**
 * JPEG color component structure
 */
struct gpujpeg_component
{
    /// Component type (luminance or chrominance)
    enum gpujpeg_component_type type;

    /// Component sampling factor (horizontal and vertical)
    struct gpujpeg_component_sampling_factor sampling_factor;

    /// Real component width
    int width;
    /// Real component height
    int height;

    /// Allocated data width for component (rounded to 8 for 8x8 blocks)
    int data_width;
    /// Allocated data height for component (rounded to 8 for 8x8 blocks)
    int data_height;
    /// Allocated data size for component
    size_t data_size;

    /// MCU size for component (minimun coded unit size)
    int mcu_size;
    /// MCU size in component x-axis
    int mcu_size_x;
    /// MCU size in component y-axis
    int mcu_size_y;

    /// MCU maximum compressed size for component
    int mcu_compressed_size;

    /// MCU count for component (for interleaved mode the same value as [en|de]coder->mcu_count)
    int mcu_count;
    /// MCU count in component x-axis
    int mcu_count_x;
    /// MCU count in component y-axis
    int mcu_count_y;
    /// Segment count in component
    int segment_count;
    /// MCU count per segment in component (the last segment can contain less MCUs, but all other must contain this count)
    int segment_mcu_count;

    /// Preprocessor data in device memory (output/input for encoder/decoder)
    uint8_t* d_data;

    /// DCT and quantizer data in host memory (output/input for encoder/decoder)
    int16_t* data_quantized;
    /// DCT and quantizer data in device memory (output/input for encoder/decoder)
    int16_t* d_data_quantized;
    /// Index of DCT and quantizer data in device and host buffers
    unsigned int data_quantized_index;

    int dc_huff_idx; ///< Huffman DC table index
    int ac_huff_idx; ///< Huffman AC table index
};

/**
 * Print component data
 *
 * @param component
 * @param d_data
 */
void
gpujpeg_component_print8(struct gpujpeg_component* component, uint8_t* d_data);

/**
 * Print component data
 *
 * @param component
 * @param d_data
 */
void
gpujpeg_component_print16(struct gpujpeg_component* component, int16_t* d_data);

/**
 * JPEG coder structure
 */
struct gpujpeg_coder
{
    /// Parameters (quality, restart_interval, etc.)
    struct gpujpeg_parameters param;

    /// Parameters for image data (width, height, comp_count, etc.)
    struct gpujpeg_image_parameters param_image;

    /// Color components
    struct gpujpeg_component* component;
    /// Color components in device memory
    struct gpujpeg_component* d_component;
    /// Number of allocated components
    int component_allocated_size;

    /// Segments for all components
    struct gpujpeg_segment* segment;
    /// Segments in device memory for all components
    struct gpujpeg_segment* d_segment;
    /// Number of allocated segments
    int segment_allocated_size;

    /// Preprocessor data (kernel function pointer)
    void* preprocessor;

    /// Maximum sampling factor from components
    struct gpujpeg_component_sampling_factor sampling_factor;
    /// MCU size (for all components)
    int mcu_size;
    /// MCU compressed size (for all components)
    int mcu_compressed_size;
    /// MCU count (for all components)
    int mcu_count;
    /// Segment total count for all components
    int segment_count;
    /// MCU count per segment (the last segment can contain less MCUs, but all other must contain this count)
    int segment_mcu_count;

    /// Image data width
    int data_width;
    /// Image data height
    int data_height;
    /// Number of raw image bytes
    size_t data_raw_size;
    /// Number of coefficient count for all components
    size_t data_size;
    /// Number of compressed bytes
    size_t data_compressed_size;

    /// Number od 8x8 blocks in all components
    int block_count;

    /// List of block indices in host memory (lower 7 bits are index of component,
    /// 8th bit is 0 for luminance block or 1 for chroma block and bits from 9
    /// above are base index of the block in quantized buffer data)
    uint64_t* block_list;
    /// List of block indices in device memory (same format as host-memory block list)
    uint64_t* d_block_list;
    /// Number of allocated components
    int block_allocated_size;

    /// Raw image data in host memory (loaded from file for encoder, saved to file for decoder)
    uint8_t* data_raw;
    /// Raw image data in device memory (loaded from file for encoder, saved to file for decoder)
    uint8_t* d_data_raw;
    /// Memory allocated by gpujpeg
    uint8_t* d_data_raw_allocated;
    /// Allocated data size
    size_t data_raw_allocated_size;

    /// Preprocessor data in device memory (output/input for encoder/decoder)
    uint8_t* d_data;
    /// DCT and quantizer data in host memory (output/input for encoder/decoder)
    int16_t* data_quantized;
    /// DCT and quantizer data in device memory (output/input for encoder/decoder)
    int16_t* d_data_quantized;
    /// Allocated size
    size_t data_allocated_size;

    /// Huffman coder data in host memory (output/input for encoder/decoder)
    /// only **partially** pinned (needs special treatment - @sa data_compressed_pinned_sz occurrences)
    uint8_t* data_compressed;
    size_t data_compressed_pinned_sz; ///< amount of pinned memory from data_compressed
    /// Huffman coder data in device memory (output/input for encoder/decoder)
    uint8_t* d_data_compressed;
    /// Huffman coder temporary data (in device memory only)
    uint8_t* d_temp_huffman;
    /// Allocated size
    size_t data_compressed_allocated_size;

    int cuda_cc_major; ///< CUDA Compute capability (major version)
    int cuda_cc_minor; ///< CUDA Compute capability (minor version)

    // Operation durations
    double start_time;    ///< overal coding CPU start time
    double init_end_time; ///< CPU time after coder (re)initialization
    double stop_time;     ///< overal coding CPU end time
    struct gpujpeg_timer duration_memory_to;
    struct gpujpeg_timer duration_memory_from;
    struct gpujpeg_timer duration_memory_map;
    struct gpujpeg_timer duration_memory_unmap;
    struct gpujpeg_timer duration_preprocessor;
    struct gpujpeg_timer duration_dct_quantization;
    struct gpujpeg_timer duration_huffman_coder;
    struct gpujpeg_timer duration_stream;
    struct gpujpeg_timer duration_in_gpu;
    // aggregate statistics
    double first_frame_duration;
    double aggregate_duration;
    long frames;

    size_t allocated_gpu_memory_size; ///< for gpujpeg_encoder_max_pixels() only (remove?)

    int encoder; ///< 1 if we are encoder, 0 decoder
};

/**
 * Initialize JPEG coder (allocate buffers and initialize structures)
 *
 * @param codec  Codec structure
 * @return 0 if succeeds, otherwise nonzero
 */
int
gpujpeg_coder_init(struct gpujpeg_coder* coder);

/**
 * Initialize JPEG coder (allocate buffers and initialize structures)
 *
 * @param codec        Codec structure
 * @param param
 * @param param_image
 * @param stream       CUDA stream
 * @return size of allocated device memory in bytes if succeeds, otherwise 0
 */
size_t
gpujpeg_coder_init_image(struct gpujpeg_coder * coder, const struct gpujpeg_parameters * param, const struct gpujpeg_image_parameters * param_image, cudaStream_t stream);

/**
 * @brief allocate buffers for CPU Huffman coder
 * Separated from gpujpeg_coder_init_image() - called only if needed, which is not normally so (using GPU Huffman
 * coder if restart intervals are not disabled).
 * @return 0 if succeeds, otherwise nonzero
 */
int
gpujpeg_coder_allocate_cpu_huffman_buf(struct gpujpeg_coder * coder);

/**
 * Returns duration statistics for last coded image
 */
int
gpujpeg_coder_get_stats(struct gpujpeg_coder *coder, struct gpujpeg_duration_stats *stats);

/**
 * Deinitialize JPEG coder (free buffers)
 *
 * @param codec  Codec structure
 * @return 0 if succeeds, otherwise nonzero
 */
int
gpujpeg_coder_deinit(struct gpujpeg_coder* coder);

void
coder_process_stats(struct gpujpeg_coder* coder);
void
coder_process_stats_overall(struct gpujpeg_coder* coder);

struct gpujpeg_component;

/**
 * Returns subsampling configuration of a planar pixel format in array of 8
 * ints - [W0 H0 W1 H1 ...] where each number represent given component
 * horizontal/vertical sampling factor as defined for JPEG, eg. (2 1 1 1 1 1)
 * for 4:2:2 (first comonent - Y - is sampled in horizontal dimension twice
 * compared to the remaining components.
 *
 * @returns array of 8 components representing the sampling factor of the pixel format
 */
const struct gpujpeg_component_sampling_factor *
gpujpeg_pixel_format_get_sampling_factor(enum gpujpeg_pixel_format pixel_format);

/** Returns number of bytes per pixel */
int
gpujpeg_pixel_format_get_unit_size(enum gpujpeg_pixel_format pixel_format);

/** Returns true if a pixel format is interleaved (packed, more than one component) */
int
gpujpeg_pixel_format_is_interleaved(enum gpujpeg_pixel_format pixel_format);

/**
 * @retval 0 parameters are different
 * @retval 1 parameters are the same
 */
bool
gpujpeg_image_parameters_equals(const struct gpujpeg_image_parameters *p1 , const struct gpujpeg_image_parameters *p2);

/**
 * returns difference between specified CUDA events
 *
 * @returns duration in ms, 0.0F in case of error
 */
float
gpujpeg_custom_timer_get_duration(cudaEvent_t start, cudaEvent_t stop);

/// sets comp_count and sampling_factor from given pixel_format
void
gpujpeg_set_subsampling_from_pixel_format(enum gpujpeg_pixel_format pixel_format, int* comp_count,
                                          struct gpujpeg_component_sampling_factor* sampling_factor);

/**
 * @sa MK_SUBSAMPLING
 * This is mostly the same except but it zeroes unused coefficient (if comp_count < 4).
 */
inline gpujpeg_sampling_factor_t
gpujpeg_make_sampling_factor(int comp_count, int comp1_h, int comp1_v, int comp2_h, int comp2_v, int comp3_h,
                             int comp3_v, int comp4_h, int comp4_v)
{
    gpujpeg_sampling_factor_t sampling_factor =
        MK_SUBSAMPLING(comp1_h, comp1_v, comp2_h, comp2_v, comp3_h, comp3_v, comp4_h, comp4_v);
    const uint32_t mask = 0xFFFFFFFFU << (32U - comp_count * 8U);
    return sampling_factor & mask;
}

#define gpujpeg_make_sampling_factor2(comp_count, sampling_factor)                                                     \
    gpujpeg_make_sampling_factor(comp_count, (sampling_factor)[0].horizontal, (sampling_factor)[0].vertical,           \
                                 (sampling_factor)[1].horizontal, (sampling_factor)[1].vertical,                       \
                                 (sampling_factor)[2].horizontal, (sampling_factor)[2].vertical,                       \
                                 (sampling_factor)[3].horizontal, (sampling_factor)[3].vertical)


#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif // GPUJPEG_COMMON_INTERNAL_H

/* vi: set expandtab sw=4: */
