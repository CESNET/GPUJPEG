/**
 * @file
 * Copyright (c) 2011-2025, CESNET
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

#ifdef __cplusplus
#include <cstdint>
#else
#include <stdint.h>
#endif

#ifndef _MSC_VER
#define ATTRIBUTE_UNUSED __attribute__((unused))
#else
#define ATTRIBUTE_UNUSED
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @addtogroup Contants
 * @{
 */
#define GPUJPEG_MAX_COMPONENT_COUNT             4
/// @}

/**
 * @addtogroup Flags
 * @{
 */
#define GPUJPEG_INIT_DEV_VERBOSE                1
#define GPUJPEG_OPENGL_INTEROPERABILITY         2
/// @deprecated use @ref GPUJPEG_INIT_VERBOSE
#define GPUJPEG_VERBOSE                 GPUJPEG_INIT_DEV_VERBOSE
/// @}

/** Maximum number of segment info header in stream */
#define GPUJPEG_MAX_SEGMENT_INFO_HEADER_COUNT   100

/**
 * @addtogroup Errors
 * @{
 */
#define GPUJPEG_NOERR                           0
#define GPUJPEG_ERROR                           (-1)
#define GPUJPEG_ERR_RESTART_CHANGE              (-2)
/// @}

#define GPUJPEG_VAL_TRUE  "1"
#define GPUJPEG_VAL_FALSE "0"

/**
 * Color spaces for JPEG codec
 */
enum gpujpeg_color_space {
    GPUJPEG_NONE = 0,
    GPUJPEG_RGB = 1,
    GPUJPEG_YCBCR_BT601 = 2,         ///< limited-range YCbCr BT.601
    GPUJPEG_YCBCR_BT601_256LVLS = 3, ///< full-range YCbCr BT.601
    GPUJPEG_YCBCR_JPEG = GPUJPEG_YCBCR_BT601_256LVLS, ///< @ref GPUJPEG_YCBCR_BT601_256LVLS
    GPUJPEG_YCBCR_BT709 = 4,         ///< limited-range YCbCr BT.709
    GPUJPEG_YCBCR = GPUJPEG_YCBCR_BT709, ///< @ref GPUJPEG_YCBCR_BT709
    GPUJPEG_YUV = 5                  ///< @deprecated will be removed soon (is this ever needed?), define ENABLE_YUV to enable pre/post processors
};

enum gpujpeg_header_type {
    GPUJPEG_HEADER_DEFAULT = 0, ///< for 1 or 3 channel @ref GPUJPEG_YCBCR_JPEG @ref GPUJPEG_HEADER_JFIF, for @ref
                                ///< GPUJPEG_RGB @ref GPUJPEG_HEADER_ADOBE, @ref GPUJPEG_HEADER_SPIFF otherwise
    GPUJPEG_HEADER_JFIF = 1 << 0,
    GPUJPEG_HEADER_SPIFF = 1 << 1,
    GPUJPEG_HEADER_ADOBE = 1 << 2, ///< Adobe APP8 header
    GPUJPEG_HEADER_EXIF = 1 << 3,
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

    /// 8bit unsigned samples, 3 or 4 components, each pixel padded to 32bits
    /// with optional alpha or unused, 4:4:4(:4) sampling, interleaved
    GPUJPEG_4444_U8_P0123 =  6,
};

/**
 * Sampling factor for color component in JPEG format
 */
struct gpujpeg_component_sampling_factor
{
    uint8_t horizontal;
    uint8_t vertical;
};

enum {
    GPUJPEG_METADATA_ORIENTATION,
    GPUJPEG_METADATA_COUNT,
};
struct gpujpeg_orientation /// as defined in SPIFF
{
    unsigned rotation : 2; ///< in multiples of 90° clock-wise
    unsigned flip : 1;     ///< 1 - left-to-right orientation flipped after rotation applied
};
struct gpujpeg_image_metadata
{
    struct
    {
        union {
            struct gpujpeg_orientation orient;
        };
        unsigned set : 1; ///< item is set, otherwise the union value is undefined
    } vals[GPUJPEG_METADATA_COUNT];
};

#ifdef __cplusplus
}
#endif

#endif // GPUJPEG_TYPE_H
