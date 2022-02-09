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

#ifndef GPUJPEG_TYPE_H
#define GPUJPEG_TYPE_H

#include <stdint.h>

#ifndef _MSC_VER
#define ATTRIBUTE_UNUSED __attribute__((unused))
#else
#define ATTRIBUTE_UNUSED
#endif

#ifdef __cplusplus
extern "C" {
#endif

/** Contants */
#define GPUJPEG_MAX_COMPONENT_COUNT             4

/** Flags */
#define GPUJPEG_VERBOSE                         1
#define GPUJPEG_OPENGL_INTEROPERABILITY         2

/** Maximum number of segment info header in stream */
#define GPUJPEG_MAX_SEGMENT_INFO_HEADER_COUNT   100

/** Errors */
#define GPUJPEG_NOERR                           0
#define GPUJPEG_ERROR                           (-1)
#define GPUJPEG_ERR_WRONG_SUBSAMPLING           (-2)
#define GPUJPEG_ERR_RESTART_CHANGE              (-3)

/**
 * Color spaces for JPEG codec
 */
enum gpujpeg_color_space {
    GPUJPEG_NONE = 0,
    GPUJPEG_RGB = 1,
    GPUJPEG_YCBCR_BT601 = 2,         ///< limited-range YCbCr BT.601
    GPUJPEG_YCBCR_BT601_256LVLS = 3, ///< full-range YCbCr BT.601
    GPUJPEG_YCBCR_JPEG = GPUJPEG_YCBCR_BT601_256LVLS,
    GPUJPEG_YCBCR_BT709 = 4,         ///< limited-range YCbCr BT.709
    GPUJPEG_YCBCR = GPUJPEG_YCBCR_BT709,
    GPUJPEG_YUV = 5                  ///< @deprecated will be removed soon (is this ever needed?), define ENABLE_YUV to enable pre/post processors
};

/**
 * Pixel format for input/output image data.
 */
enum gpujpeg_pixel_format {
    GPUJPEG_PIXFMT_NONE = -1,

    /// 8bit unsigned samples, 1 component
    GPUJPEG_U8 =  0,

    /// 8bit unsigned samples, 3 components, 4:4:4 sampling,
    /// sample order: comp#0 comp#1 comp#2, interleaved
    GPUJPEG_444_U8_P012 =  1,

    /// 8bit unsigned samples, 3 components, 4:4:4, planar
    GPUJPEG_444_U8_P0P1P2 = 2,

    /// 8bit unsigned samples, 3 components, 4:2:2,
    /// order of samples: comp#1 comp#0 comp#2 comp#0, interleaved
    GPUJPEG_422_U8_P1020 = 3,

    /// 8bit unsigned samples, planar, 3 components, 4:2:2, planar
    GPUJPEG_422_U8_P0P1P2 = 4,

    /// 8bit unsigned samples, planar, 3 components, 4:2:0, planar
    GPUJPEG_420_U8_P0P1P2 = 5,

    /// 8bit unsigned samples, 3 components, each pixel padded to 32bits
    /// with zero byte, 4:4:4 sampling, interleaved
    GPUJPEG_444_U8_P012Z =  6,

    /// 8bit unsigned samples, 3 components, each pixel padded to 32bits
    /// with all-one bits, 4:4:4 sampling, interleaved
    GPUJPEG_444_U8_P012A = 7,
};

/**
 * Sampling factor for color component in JPEG format
 */
struct gpujpeg_component_sampling_factor
{
    uint8_t horizontal;
    uint8_t vertical;
};

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

#ifdef __cplusplus
}
#endif

#endif // GPUJPEG_TYPE_H
