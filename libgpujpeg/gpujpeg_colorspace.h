/**
 * Copyright (c) 2011, CESNET z.s.p.o
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

#ifndef GPUJPEG_COLORSPACE_H
#define GPUJPEG_COLORSPACE_H

#include "gpujpeg_type.h"

/**
 * Color transfor debug info
 */
#if __CUDA_ARCH__ >= 200
#define GPUJPEG_COLOR_TRANSFORM_DEBUG(FROM, TO, MESSAGE) /*\
    if ( blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0 ) { \
        printf("Color Transform %s -> %s (%s)\n", gpujpeg_color_space_get_name(FROM), gpujpeg_color_space_get_name(TO), MESSAGE); \
    }*/
#else
    #define GPUJPEG_COLOR_TRANSFORM_DEBUG(FROM, TO, MESSAGE)
#endif

/**
 * Clip [0,255] range
 */
inline __device__ float gpujpeg_clamp(float value)
{
    value = (value >= 0.0f) ? value : 0.0f;
    value = (value <= 255.0f) ? value : 255.0f;
    return value;
}

/**
 * Transform to color space by matrix
 *
 * @param bit_depth
 */
template<int bit_depth>
__device__ void
gpujpeg_color_transform_to(float & c1, float & c2, float & c3, const double matrix[9], int base1, int base2, int base3)
{
    // Prepare integer constants
    const int max = pow(2.0, bit_depth);
    const int middle = pow(2.0, bit_depth - 1);
    const int matrix_int[] = {
        round(matrix[0] * max), round(matrix[1] * max), round(matrix[2] * max),
        round(matrix[3] * max), round(matrix[4] * max), round(matrix[5] * max),
        round(matrix[6] * max), round(matrix[7] * max), round(matrix[8] * max),
    };

    // Perform color transform
    int r1 = c1;
    int r2 = c2;
    int r3 = c3;
    c1 = ((matrix_int[0] * r1 + matrix_int[1] * r2 + matrix_int[2] * r3 + middle) >> bit_depth) + base1;
    c2 = ((matrix_int[3] * r1 + matrix_int[4] * r2 + matrix_int[5] * r3 + middle) >> bit_depth) + base2;
    c3 = ((matrix_int[6] * r1 + matrix_int[7] * r2 + matrix_int[8] * r3 + middle) >> bit_depth) + base3;
}

/**
 * Transform from color space by matrix
 *
 * @param bit_depth
 */
template<int bit_depth>
__device__ void
gpujpeg_color_transform_from(float & c1, float & c2, float & c3, const double matrix[9], int base1, int base2, int base3)
{
    // Prepare integer constants
    const int max = pow(2.0, bit_depth);
    const int middle = pow(2.0, bit_depth - 1);
    const int matrix_int[] = {
        round(matrix[0] * max), round(matrix[1] * max), round(matrix[2] * max),
        round(matrix[3] * max), round(matrix[4] * max), round(matrix[5] * max),
        round(matrix[6] * max), round(matrix[7] * max), round(matrix[8] * max),
    };

    // Perform color transform
    int r1 = c1 - base1;
    int r2 = c2 - base2;
    int r3 = c3 - base3;
    c1 = gpujpeg_clamp((matrix_int[0] * r1 + matrix_int[1] * r2 + matrix_int[2] * r3 + middle) >> bit_depth);
    c2 = gpujpeg_clamp((matrix_int[3] * r1 + matrix_int[4] * r2 + matrix_int[5] * r3 + middle) >> bit_depth);
    c3 = gpujpeg_clamp((matrix_int[6] * r1 + matrix_int[7] * r2 + matrix_int[8] * r3 + middle) >> bit_depth);
}

/**
 * Color space transformation
 *
 * @param color_space_from
 * @param color_space_to
 */
template<enum gpujpeg_color_space color_space_from, enum gpujpeg_color_space color_space_to>
struct gpujpeg_color_transform
{
    static __device__ void
    perform(float & c1, float & c2, float & c3) {
        GPUJPEG_COLOR_TRANSFORM_DEBUG(color_space_from, color_space_to, "Undefined");
        assert(false);
    }
};

/** Specialization [color_space_from = color_space_to] */
template<enum gpujpeg_color_space color_space>
struct gpujpeg_color_transform<color_space, color_space> {
    /** None transform */
    static __device__ void
    perform(float & c1, float & c2, float & c3) {
        GPUJPEG_COLOR_TRANSFORM_DEBUG(color_space, color_space, "Do nothing");
        // Same color space thus do nothing
    }
};

/** Specialization [color_space_from = GPUJPEG_NONE] */
template<enum gpujpeg_color_space color_space>
struct gpujpeg_color_transform<GPUJPEG_NONE, color_space> {
    /** None transform */
    static __device__ void
    perform(float & c1, float & c2, float & c3) {
        GPUJPEG_COLOR_TRANSFORM_DEBUG(GPUJPEG_NONE, color_space, "Do nothing");
        // None color space thus do nothing
    }
};
/** Specialization [color_space_to = GPUJPEG_NONE] */
template<enum gpujpeg_color_space color_space>
struct gpujpeg_color_transform<color_space, GPUJPEG_NONE> {
    /** None transform */
    static __device__ void
    perform(float & c1, float & c2, float & c3) {
        GPUJPEG_COLOR_TRANSFORM_DEBUG(color_space, GPUJPEG_NONE, "Do nothing");
        // None color space thus do nothing
    }
};
/** Specialization [color_space_from = GPUJPEG_NONE, color_space_to = GPUJPEG_NONE] */
template<>
struct gpujpeg_color_transform<GPUJPEG_NONE, GPUJPEG_NONE> {
    /** None transform */
    static __device__ void
    perform(float & c1, float & c2, float & c3) {
        GPUJPEG_COLOR_TRANSFORM_DEBUG(GPUJPEG_NONE, GPUJPEG_NONE, "Do nothing");
        // None color space thus do nothing
    }
};

/** Specialization [color_space_from = GPUJPEG_RGB, color_space_to = GPUJPEG_YCBCR_BT601] */
template<>
struct gpujpeg_color_transform<GPUJPEG_RGB, GPUJPEG_YCBCR_BT601> {
    /** RGB -> YCbCr (ITU-R Recommendation BT.601) transform (8 bit) */
    static __device__ void
    perform(float & c1, float & c2, float & c3) {
        GPUJPEG_COLOR_TRANSFORM_DEBUG(GPUJPEG_RGB, GPUJPEG_YCBCR_BT601, "Transformation");
        // Source: http://www.equasys.de/colorconversion.html
        const double matrix[] = {
              0.257000,  0.504000,  0.098000,
             -0.148000, -0.291000,  0.439000,
              0.439000, -0.368000, -0.071000
        };
        gpujpeg_color_transform_to<8>(c1, c2, c3, matrix, 16, 128, 128);
    }
};
/** Specialization [color_space_from = GPUJPEG_YCBCR_BT601, color_space_to = GPUJPEG_RGB] */
template<>
struct gpujpeg_color_transform<GPUJPEG_YCBCR_BT601, GPUJPEG_RGB> {
    /** YCbCr (ITU-R Recommendation BT.601) -> RGB transform (8 bit) */
    static __device__ void
    perform(float & c1, float & c2, float & c3) {
        GPUJPEG_COLOR_TRANSFORM_DEBUG(GPUJPEG_YCBCR_BT601, GPUJPEG_RGB, "Transformation");
        // Source: http://www.equasys.de/colorconversion.html
        const double matrix[] = {
             1.164000,  0.000000,  1.596000,
             1.164000, -0.392000, -0.813000,
             1.164000,  2.017000,  0.000000
        };
        gpujpeg_color_transform_from<8>(c1, c2, c3, matrix, 16, 128, 128);
    }
};

/** Specialization [color_space_from = GPUJPEG_RGB, color_space_to = GPUJPEG_YCBCR_BT601_256LVLS] */
template<>
struct gpujpeg_color_transform<GPUJPEG_RGB, GPUJPEG_YCBCR_BT601_256LVLS> {
    /** RGB -> YCbCr (ITU-R Recommendation BT.601 with 256 levels) transform (8 bit) */
    static __device__ void
    perform(float & c1, float & c2, float & c3) {
        GPUJPEG_COLOR_TRANSFORM_DEBUG(GPUJPEG_RGB, GPUJPEG_YCBCR_BT601_256LVLS, "Transformation");
        // Source: http://www.ecma-international.org/publications/files/ECMA-TR/TR-098.pdf, page 3
        const double matrix[] = {
             0.299000,  0.587000,  0.114000,
            -0.168700, -0.331300,  0.500000,
             0.500000, -0.418700, -0.081300
        };
        gpujpeg_color_transform_to<8>(c1, c2, c3, matrix, 0, 128, 128);
    }
};
/** Specialization [color_space_from = GPUJPEG_YCBCR_BT601_256LVLS, color_space_to = GPUJPEG_RGB] */
template<>
struct gpujpeg_color_transform<GPUJPEG_YCBCR_BT601_256LVLS, GPUJPEG_RGB> {
    /** YCbCr (ITU-R Recommendation BT.601 with 256 levels) -> RGB transform (8 bit) */
    static __device__ void
    perform(float & c1, float & c2, float & c3) {
        GPUJPEG_COLOR_TRANSFORM_DEBUG(GPUJPEG_YCBCR_BT601_256LVLS, GPUJPEG_RGB, "Transformation");
        // Source: http://www.ecma-international.org/publications/files/ECMA-TR/TR-098.pdf, page 4
        const double matrix[] = {
            1.000000,  0.000000,  1.402000,
            1.000000, -0.344140, -0.714140,
            1.000000,  1.772000,  0.000000
        };
        gpujpeg_color_transform_from<8>(c1, c2, c3, matrix, 0, 128, 128);
    }
};

/** Specialization [color_space_from = GPUJPEG_RGB, color_space_to = GPUJPEG_YCBCR_BT709] */
template<>
struct gpujpeg_color_transform<GPUJPEG_RGB, GPUJPEG_YCBCR_BT709> {
    /** RGB -> YCbCr (ITU-R Recommendation BT.709) transform (8 bit) */
    static __device__ void
    perform(float & c1, float & c2, float & c3) {
        GPUJPEG_COLOR_TRANSFORM_DEBUG(GPUJPEG_RGB, GPUJPEG_YCBCR_BT709, "Transformation");
        // Source: http://www.equasys.de/colorconversion.html
        const double matrix[] = {
              0.183000,  0.614000,  0.062000,
             -0.101000, -0.339000,  0.439000,
              0.439000, -0.399000, -0.040000
        };
        gpujpeg_color_transform_to<8>(c1, c2, c3, matrix, 16, 128, 128);
    }
};
/** Specialization [color_space_from = GPUJPEG_YCBCR_BT709, color_space_to = GPUJPEG_RGB] */
template<>
struct gpujpeg_color_transform<GPUJPEG_YCBCR_BT709, GPUJPEG_RGB> {
    /** YCbCr (ITU-R Recommendation BT.709) -> RGB transform (8 bit) */
    static __device__ void
    perform(float & c1, float & c2, float & c3) {
        GPUJPEG_COLOR_TRANSFORM_DEBUG(GPUJPEG_YCBCR_BT709, GPUJPEG_RGB, "Transformation");
        // Source: http://www.equasys.de/colorconversion.html
        const double matrix[] = {
             1.164000,  0.000000,  1.793000,
             1.164000, -0.213000, -0.533000,
             1.164000,  2.112000,  0.000000
        };
        gpujpeg_color_transform_from<8>(c1, c2, c3, matrix, 16, 128, 128);
    }
};

/** Specialization [color_space_from = GPUJPEG_RGB, color_space_to = GPUJPEG_YUV] */
template<>
struct gpujpeg_color_transform<GPUJPEG_RGB, GPUJPEG_YUV> {
    /** RGB -> YUV transform (8 bit) */
    static __device__ void
    perform(float & c1, float & c2, float & c3) {
        GPUJPEG_COLOR_TRANSFORM_DEBUG(GPUJPEG_RGB, GPUJPEG_YUV, "Transformation");
        const double matrix[] = {
              0.299000,  0.587000,  0.114000,
             -0.147400, -0.289500,  0.436900,
              0.615000, -0.515000, -0.100000
        };
        gpujpeg_color_transform_to<8>(c1, c2, c3, matrix, 0, 128, 128);
    }
};
/** Specialization [color_space_from = GPUJPEG_YUV, color_space_to = GPUJPEG_RGB] */
template<>
struct gpujpeg_color_transform<GPUJPEG_YUV, GPUJPEG_RGB> {
    /** YUV -> RGB transform (8 bit) */
    static __device__ void
    perform(float & c1, float & c2, float & c3) {
        GPUJPEG_COLOR_TRANSFORM_DEBUG(GPUJPEG_YUV, GPUJPEG_RGB, "Transformation");
        const double matrix[] = {
             1.000000,  0.000000,  1.140000,
             1.000000, -0.395000, -0.581000,
             1.000000,  2.032000,  0.000000
        };
        gpujpeg_color_transform_from<8>(c1, c2, c3, matrix, 0, 128, 128);
    }
};

/** Specialization [color_space_from = GPUJPEG_YCBCR_BT601, color_space_to = GPUJPEG_YCBCR_BT601_256LVLS] */
template<>
struct gpujpeg_color_transform<GPUJPEG_YCBCR_BT601, GPUJPEG_YCBCR_BT601_256LVLS> {
    /** YCbCr (ITU-R Recommendation BT.709) -> YCbCr (ITU-R Recommendation BT.601 with 256 levels) transform (8 bit) */
    static __device__ void
    perform(float & c1, float & c2, float & c3) {
        gpujpeg_color_transform<GPUJPEG_YCBCR_BT601, GPUJPEG_RGB>::perform(c1,c2,c3);
        gpujpeg_color_transform<GPUJPEG_RGB, GPUJPEG_YCBCR_BT601_256LVLS>::perform(c1,c2,c3);

    }
};
/** Specialization [color_space_from = GPUJPEG_YCBCR_BT601_256LVLS, color_space_to = GPUJPEG_YCBCR_BT601] */
template<>
struct gpujpeg_color_transform<GPUJPEG_YCBCR_BT601_256LVLS, GPUJPEG_YCBCR_BT601> {
    /** YCbCr (ITU-R Recommendation BT.601 with 256 levels) -> YCbCr (ITU-R Recommendation BT.709) transform (8 bit) */
    static __device__ void
    perform(float & c1, float & c2, float & c3) {
        gpujpeg_color_transform<GPUJPEG_YCBCR_BT601_256LVLS, GPUJPEG_RGB>::perform(c1,c2,c3);
        gpujpeg_color_transform<GPUJPEG_RGB, GPUJPEG_YCBCR_BT601>::perform(c1,c2,c3);
    }
};

/** Specialization [color_space_from = GPUJPEG_YCBCR_BT709, color_space_to = GPUJPEG_YCBCR_BT601_256LVLS] */
template<>
struct gpujpeg_color_transform<GPUJPEG_YCBCR_BT709, GPUJPEG_YCBCR_BT601_256LVLS> {
    /** YCbCr (ITU-R Recommendation BT.709) -> YCbCr (ITU-R Recommendation BT.601 with 256 levels) transform (8 bit) */
    static __device__ void
    perform(float & c1, float & c2, float & c3) {
        gpujpeg_color_transform<GPUJPEG_YCBCR_BT709, GPUJPEG_RGB>::perform(c1,c2,c3);
        gpujpeg_color_transform<GPUJPEG_RGB, GPUJPEG_YCBCR_BT601_256LVLS>::perform(c1,c2,c3);

    }
};
/** Specialization [color_space_from = GPUJPEG_YCBCR_BT601_256LVLS, color_space_to = GPUJPEG_YCBCR_ITU_R] */
template<>
struct gpujpeg_color_transform<GPUJPEG_YCBCR_BT601_256LVLS, GPUJPEG_YCBCR_BT709> {
    /** YCbCr (ITU-R Recommendation BT.601 with 256 levels) -> YCbCr (ITU-R Recommendation BT.709) transform (8 bit) */
    static __device__ void
    perform(float & c1, float & c2, float & c3) {
        gpujpeg_color_transform<GPUJPEG_YCBCR_BT601_256LVLS, GPUJPEG_RGB>::perform(c1,c2,c3);
        gpujpeg_color_transform<GPUJPEG_RGB, GPUJPEG_YCBCR_BT709>::perform(c1,c2,c3);
    }
};

/** Specialization [color_space_from = GPUJPEG_YUV, color_space_to = GPUJPEG_YCBCR_BT601_256LVLS] */
template<>
struct gpujpeg_color_transform<GPUJPEG_YUV, GPUJPEG_YCBCR_BT601_256LVLS> {
    /** YUV -> YCbCr (ITU-R Recommendation BT.601 with 256 levels) transform (8 bit) */
    static __device__ void
    perform(float & c1, float & c2, float & c3) {
        gpujpeg_color_transform<GPUJPEG_YUV, GPUJPEG_RGB>::perform(c1,c2,c3);
        gpujpeg_color_transform<GPUJPEG_RGB, GPUJPEG_YCBCR_BT601_256LVLS>::perform(c1,c2,c3);

    }
};
/** Specialization [color_space_from = GPUJPEG_YCBCR_BT601_256LVLS, color_space_to = GPUJPEG_YUV] */
template<>
struct gpujpeg_color_transform<GPUJPEG_YCBCR_BT601_256LVLS, GPUJPEG_YUV> {
    /** YCbCr (ITU-R Recommendation BT.601 with 256 levels) -> YUV transform (8 bit) */
    static __device__ void
    perform(float & c1, float & c2, float & c3) {
        gpujpeg_color_transform<GPUJPEG_YCBCR_BT601_256LVLS, GPUJPEG_RGB>::perform(c1,c2,c3);
        gpujpeg_color_transform<GPUJPEG_RGB, GPUJPEG_YUV>::perform(c1,c2,c3);
    }
};

/**
 * Color components load order
 *
 * @param color_space
 */
template<enum gpujpeg_color_space color_space>
struct gpujpeg_color_order
{
    /** Change load order */
    static __device__ void
    perform_load(float & c1, float & c2, float & c3) {
        // Default order is not changed
    }
    /** Change load order */
    static __device__ void
    perform_store(float & c1, float & c2, float & c3) {
        // Default order is not changed
    }
};
/** YCbCr and YUV Specializations (UYV <-> YUV) */
template<>
struct gpujpeg_color_order<GPUJPEG_YCBCR_BT601>
{
    static __device__ void
    perform_load(float & c1, float & c2, float & c3) {
        float tmp = c1; c1 = c2; c2 = tmp;
    }
    static __device__ void
    perform_store(float & c1, float & c2, float & c3) {
        float tmp = c1; c1 = c2; c2 = tmp;
    }
};
template<>
struct gpujpeg_color_order<GPUJPEG_YCBCR_BT601_256LVLS>
{
    static __device__ void
    perform_load(float & c1, float & c2, float & c3) {
        float tmp = c1; c1 = c2; c2 = tmp;
    }
    static __device__ void
    perform_store(float & c1, float & c2, float & c3) {
        float tmp = c1; c1 = c2; c2 = tmp;
    }
};
template<>
struct gpujpeg_color_order<GPUJPEG_YCBCR_BT709>
{
    static __device__ void
    perform_load(float & c1, float & c2, float & c3) {
        float tmp = c1; c1 = c2; c2 = tmp;
    }
    static __device__ void
    perform_store(float & c1, float & c2, float & c3) {
        float tmp = c1; c1 = c2; c2 = tmp;
    }
};
template<>
struct gpujpeg_color_order<GPUJPEG_YUV>
{
    static __device__ void
    perform_load(float & c1, float & c2, float & c3) {
        float tmp = c1; c1 = c2; c2 = tmp;
    }
    static __device__ void
    perform_store(float & c1, float & c2, float & c3) {
        float tmp = c1; c1 = c2; c2 = tmp;
    }
};

#endif // GPUJPEG_COLORSPACE_H
