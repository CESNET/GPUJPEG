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

/**
 * Clip [0,255] range
 */
inline __device__ float gpujpeg_clamp(float value)
{
    value = (value >= 0.0f) ? value : 0.0f;
    value = (value <= 255.0) ? value : 255.0f;
    return value;
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
        //assert(false);
    }
};

/** Specialization [color_space_from = color_space_to] */
template<enum gpujpeg_color_space color_space>
struct gpujpeg_color_transform<color_space, color_space> {
    /** None transform */
    static __device__ void
    perform(float & c1, float & c2, float & c3) {
        // Same color space so do nothing
    }
};

/** Specialization [color_space_from = GPUJPEG_RGB, color_space_to = GPUJPEG_YCBCR_BT601] */
template<>
struct gpujpeg_color_transform<GPUJPEG_RGB, GPUJPEG_YCBCR_BT601> {
    /** RGB -> YCbCr (ITU-R Recommendation BT.601) transform (8 bit) */
    static __device__ void
    perform(float & c1, float & c2, float & c3) {
        // Source: http://www.equasys.de/colorconversion.html
        float r1 =  0.257000f * c1 + 0.504000f * c2 + 0.098000f * c3 + 16.0f;
        float r2 = -0.148000f * c1 - 0.291000f * c2 + 0.439000f * c3 + 128.0f;
        float r3 =  0.439000f * c1 - 0.368000f * c2 - 0.071000f * c3 + 128.0f;
        c1 = r1;
        c2 = r2;
        c3 = r3;
    }
};
/** Specialization [color_space_from = GPUJPEG_YCBCR_BT601, color_space_to = GPUJPEG_RGB] */
template<>
struct gpujpeg_color_transform<GPUJPEG_YCBCR_BT601, GPUJPEG_RGB> {
    /** YCbCr (ITU-R Recommendation BT.601) -> RGB transform (8 bit) */
    static __device__ void
    perform(float & c1, float & c2, float & c3) {
        // Source: http://www.equasys.de/colorconversion.html
        float r1 = c1 - 16.0f;
        float r2 = c2 - 128.0f;
        float r3 = c3 - 128.0f;
        c1 = gpujpeg_clamp(1.164000f * r1 + 0.000000f * r2 + 1.596000f * r3);
        c2 = gpujpeg_clamp(1.164000f * r1 - 0.392000f * r2 - 0.813000f * r3);
        c3 = gpujpeg_clamp(1.164000f * r1 + 2.017000f * r2 + 0.000000f * r3);
    }
};

/** Specialization [color_space_from = GPUJPEG_RGB, color_space_to = GPUJPEG_YCBCR_BT601_256LVLS] */
template<>
struct gpujpeg_color_transform<GPUJPEG_RGB, GPUJPEG_YCBCR_BT601_256LVLS> {
    /** RGB -> YCbCr (ITU-R Recommendation BT.601 with 256 levels) transform (8 bit) */
    static __device__ void
    perform(float & c1, float & c2, float & c3) {
        // Source: http://www.ecma-international.org/publications/files/ECMA-TR/TR-098.pdf, page 3
        float r1 =  0.299000f * c1 + 0.587000f * c2 + 0.114000f * c3;
        float r2 = -0.168736f * c1 - 0.331264f * c2 + 0.500000f * c3 + 128.0f;
        float r3 =  0.500000f * c1 - 0.418688f * c2 - 0.081312f * c3 + 128.0f;
        c1 = r1;
        c2 = r2;
        c3 = r3;
    }
};
/** Specialization [color_space_from = GPUJPEG_YCBCR_BT601_256LVLS, color_space_to = GPUJPEG_RGB] */
template<>
struct gpujpeg_color_transform<GPUJPEG_YCBCR_BT601_256LVLS, GPUJPEG_RGB> {
    /** YCbCr (ITU-R Recommendation BT.601 with 256 levels) -> RGB transform (8 bit) */
    static __device__ void
    perform(float & c1, float & c2, float & c3) {
        // Source: http://www.ecma-international.org/publications/files/ECMA-TR/TR-098.pdf, page 4
        float r1 = c1 - 0.0f;
        float r2 = c2 - 128.0f;
        float r3 = c3 - 128.0f;
        c1 = gpujpeg_clamp(1.000000f * r1 + 0.000000f * r2 + 1.402000f * r3);
        c2 = gpujpeg_clamp(1.000000f * r1 - 0.344136f * r2 - 0.714136f * r3);
        c3 = gpujpeg_clamp(1.000000f * r1 + 1.772000f * r2 + 0.000000f * r3);
    }
};

/** Specialization [color_space_from = GPUJPEG_RGB, color_space_to = GPUJPEG_YCBCR_BT709] */
template<>
struct gpujpeg_color_transform<GPUJPEG_RGB, GPUJPEG_YCBCR_BT709> {
    /** RGB -> YCbCr (ITU-R Recommendation BT.709) transform (8 bit) */
    static __device__ void
    perform(float & c1, float & c2, float & c3) {
        // Source: http://www.equasys.de/colorconversion.html
        float r1 =  0.183000f * c1 + 0.614000f * c2 + 0.062000f * c3 + 16.0f;
        float r2 = -0.101000f * c1 - 0.339000f * c2 + 0.439000f * c3 + 128.0f;
        float r3 =  0.439000f * c1 - 0.399000f * c2 - 0.040000f * c3 + 128.0f;
        c1 = r1;
        c2 = r2;
        c3 = r3;
    }
};
/** Specialization [color_space_from = GPUJPEG_YCBCR_BT709, color_space_to = GPUJPEG_RGB] */
template<>
struct gpujpeg_color_transform<GPUJPEG_YCBCR_BT709, GPUJPEG_RGB> {
    /** YCbCr (ITU-R Recommendation BT.709) -> RGB transform (8 bit) */
    static __device__ void
    perform(float & c1, float & c2, float & c3) {
        // Source: http://www.equasys.de/colorconversion.html
        float r1 = c1 - 16.0f;
        float r2 = c2 - 128.0f;
        float r3 = c3 - 128.0f;
        c1 = gpujpeg_clamp(1.164000f * r1 + 0.000000f * r2 + 1.793000f * r3);
        c2 = gpujpeg_clamp(1.164000f * r1 - 0.213000f * r2 - 0.533000f * r3);
        c3 = gpujpeg_clamp(1.164000f * r1 + 2.112000f * r2 + 0.000000f * r3);
    }
};

/** Specialization [color_space_from = GPUJPEG_RGB, color_space_to = GPUJPEG_YUV] */
template<>
struct gpujpeg_color_transform<GPUJPEG_RGB, GPUJPEG_YUV> {
    /** RGB -> YUV transform (8 bit) */
    static __device__ void
    perform(float & c1, float & c2, float & c3) {
        // Source: http://www.equasys.de/colorconversion.html
        float r1 =  0.299000f * c1 + 0.587000f * c2 + 0.114000f * c3;
        float r2 = -0.147400f * c1 - 0.289500f * c2 + 0.436900f * c3 + 128.0f;
        float r3 =  0.615000f * c1 - 0.515000f * c2 - 0.100000f * c3 + 128.0f;
        c1 = r1;
        c2 = r2;
        c3 = r3;
    }
};
/** Specialization [color_space_from = GPUJPEG_YUV, color_space_to = GPUJPEG_RGB] */
template<>
struct gpujpeg_color_transform<GPUJPEG_YUV, GPUJPEG_RGB> {
    /** YUV -> RGB transform (8 bit) */
    static __device__ void
    perform(float & c1, float & c2, float & c3) {
        // Source: http://www.equasys.de/colorconversion.html
        float r1 = c1 - 0.0f;
        float r2 = c2 - 128.0f;
        float r3 = c3 - 128.0f;
        c1 = gpujpeg_clamp(1.000000f * r1 + 0.000000f * r2 + 1.140000f * r3);
        c2 = gpujpeg_clamp(1.000000f * r1 - 0.395000f * r2 - 0.581000f * r3);
        c3 = gpujpeg_clamp(1.000000f * r1 + 2.032000f * r2 + 0.000000f * r3);
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
 * @param color_space_from
 * @param color_space_to
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
