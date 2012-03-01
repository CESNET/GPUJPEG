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
 
#include "gpujpeg_preprocessor.h"
#include "gpujpeg_util.h"

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
        assert(false);
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

#define RGB_8BIT_THREADS 256

/**
 * Preprocessor data for component
 */
struct gpujpeg_preprocessor_data_component
{
    uint8_t* d_data;
    int data_width;
    struct gpujpeg_component_sampling_factor sampling_factor;
};

/**
 * Preprocessor data
 */
struct gpujpeg_preprocessor_data
{
    struct gpujpeg_preprocessor_data_component comp[3];
};

/** Value that means that sampling factor has dynamic value */
#define GPUJPEG_DYNAMIC 16

/** Sampling factor for all components */
typedef int gpujpeg_preprocessor_sampling_factor_t;

/**
 * Compose sampling factor for all components to single type
 * 
 * @return integer that contains all sampling factors
 */
inline gpujpeg_preprocessor_sampling_factor_t
gpujpeg_preprocessor_make_sampling_factor(int comp1_h, int comp1_v, int comp2_h, int comp2_v, int comp3_h, int comp3_v)
{    
    gpujpeg_preprocessor_sampling_factor_t sampling_factor = 0;
    sampling_factor |= ((comp1_h << 4) | comp1_v) << 16;
    sampling_factor |= ((comp2_h << 4) | comp2_v) << 8;
    sampling_factor |= ((comp3_h << 4) | comp3_v) << 0;
    
    return sampling_factor;
}

/**
 * Store value to component data buffer in specified position by buffer size and subsampling
 */
template<
    uint8_t s_samp_factor_h = GPUJPEG_DYNAMIC,
    uint8_t s_samp_factor_v = GPUJPEG_DYNAMIC
>
struct gpujpeg_preprocessor_raw_to_comp_store
{
    static __device__ void
    perform(uint8_t value, int position_x, int position_y, struct gpujpeg_preprocessor_data_component & comp)
    {
        uint8_t samp_factor_h = s_samp_factor_h;
        if ( samp_factor_h == GPUJPEG_DYNAMIC ) {
            samp_factor_h = comp.sampling_factor.horizontal;
        }
        uint8_t samp_factor_v = s_samp_factor_v;
        if ( samp_factor_v == GPUJPEG_DYNAMIC ) {
            samp_factor_v = comp.sampling_factor.vertical;
        }
        
        if ( (position_x % samp_factor_h) != 0 && (position_x % samp_factor_v) != 0 )
            return;
        position_x = position_x / samp_factor_h;
        position_y = position_y / samp_factor_v;

        int data_position = position_y * comp.data_width + position_x;
        comp.d_data[data_position] = value;
    }
};
template<>
struct gpujpeg_preprocessor_raw_to_comp_store<1, 1>
{
    static __device__ void
    perform(uint8_t value, int position_x, int position_y, struct gpujpeg_preprocessor_data_component & comp)
    {
        int data_position = position_y * comp.data_width + position_x;
        comp.d_data[data_position] = value;
    }
};

/**
 * Kernel - Copy raw image source data into three separated component buffers
 */
typedef void (*gpujpeg_preprocessor_encode_kernel)(struct gpujpeg_preprocessor_data data, const uint8_t* d_data_raw, int image_width, int image_height);
 
/** Specialization [sampling factor is 4:4:4] */
template<
    enum gpujpeg_color_space color_space,
    uint8_t s_comp1_samp_factor_h, uint8_t s_comp1_samp_factor_v,
    uint8_t s_comp2_samp_factor_h, uint8_t s_comp2_samp_factor_v,
    uint8_t s_comp3_samp_factor_h, uint8_t s_comp3_samp_factor_v
>
__global__ void 
gpujpeg_preprocessor_raw_to_comp_kernel_4_4_4(struct gpujpeg_preprocessor_data data, const uint8_t* d_data_raw, int image_width, int image_height)
{
    int x  = threadIdx.x;
    int gX = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x;
            
    // Load to shared
    __shared__ unsigned char s_data[RGB_8BIT_THREADS * 3];
    if ( (x * 4) < RGB_8BIT_THREADS * 3 ) {
        int* s = (int*)d_data_raw;
        int* d = (int*)s_data;
        d[x] = s[((gX * 3) >> 2) + x];
    }
    __syncthreads();

    // Load
    int offset = x * 3;
    float r1 = (float)(s_data[offset]);
    float r2 = (float)(s_data[offset + 1]);
    float r3 = (float)(s_data[offset + 2]);

    // Color transform
    gpujpeg_color_transform<color_space, GPUJPEG_YCBCR_BT601_256LVLS>::perform(r1, r2, r3);
    
    // Position
    int image_position = gX + x;
    int image_position_x = image_position % image_width;
    int image_position_y = image_position / image_width;
        
    // Store
    if ( image_position < (image_width * image_height) ) {
        gpujpeg_preprocessor_raw_to_comp_store<s_comp1_samp_factor_h, s_comp1_samp_factor_v>::perform((uint8_t)r1, image_position_x, image_position_y, data.comp[0]);
        gpujpeg_preprocessor_raw_to_comp_store<s_comp2_samp_factor_h, s_comp2_samp_factor_v>::perform((uint8_t)r2, image_position_x, image_position_y, data.comp[1]);
        gpujpeg_preprocessor_raw_to_comp_store<s_comp3_samp_factor_h, s_comp3_samp_factor_v>::perform((uint8_t)r3, image_position_x, image_position_y, data.comp[2]);
    }
}

/** Specialization [sampling factor is 4:2:2] */
template<
    enum gpujpeg_color_space color_space,
    uint8_t s_comp1_samp_factor_h, uint8_t s_comp1_samp_factor_v,
    uint8_t s_comp2_samp_factor_h, uint8_t s_comp2_samp_factor_v,
    uint8_t s_comp3_samp_factor_h, uint8_t s_comp3_samp_factor_v
>
__global__ void 
gpujpeg_preprocessor_raw_to_comp_kernel_4_2_2(struct gpujpeg_preprocessor_data data, const uint8_t* d_data_raw, int image_width, int image_height)
{
    int x  = threadIdx.x;
    int gX = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x;
        
    // Load to shared
    __shared__ unsigned char s_data[RGB_8BIT_THREADS * 2];
    if ( (x * 4) < RGB_8BIT_THREADS * 2 ) {
        int* s = (int*)d_data_raw;
        int* d = (int*)s_data;
        d[x] = s[((gX * 2) >> 2) + x];
    }
    __syncthreads();

    // Load
    int offset = x * 2;
    float r1 = (float)(s_data[offset + 1]);
    float r2;
    float r3;
    if ( (gX + x) % 2 == 0 ) {
        r2 = (float)(s_data[offset]);
        r3 = (float)(s_data[offset + 2]);
    } else {
        r2 = (float)(s_data[offset - 2]);
        r3 = (float)(s_data[offset]);
    }
    
    // Color transform
    gpujpeg_color_transform<color_space, GPUJPEG_YCBCR_BT601_256LVLS>::perform(r1, r2, r3);
    
    // Position
    int image_position = gX + x;
    int image_position_x = image_position % image_width;
    int image_position_y = image_position / image_width;
    
    // Store
    if ( image_position < (image_width * image_height) ) {
        gpujpeg_preprocessor_raw_to_comp_store<s_comp1_samp_factor_h, s_comp1_samp_factor_v>::perform((uint8_t)r1, image_position_x, image_position_y, data.comp[0]);
        gpujpeg_preprocessor_raw_to_comp_store<s_comp2_samp_factor_h, s_comp2_samp_factor_v>::perform((uint8_t)r2, image_position_x, image_position_y, data.comp[1]);
        gpujpeg_preprocessor_raw_to_comp_store<s_comp3_samp_factor_h, s_comp3_samp_factor_v>::perform((uint8_t)r3, image_position_x, image_position_y, data.comp[2]);
    }
}

/**
 * Select preprocessor encode kernel
 * 
 * @param encoder
 * @return kernel
 */
gpujpeg_preprocessor_encode_kernel
gpujpeg_preprocessor_select_encode_kernel(struct gpujpeg_encoder* encoder)
{
    // Get coder
    struct gpujpeg_coder* coder = &encoder->coder;
    
    gpujpeg_preprocessor_sampling_factor_t sampling_factor = gpujpeg_preprocessor_make_sampling_factor(
        coder->sampling_factor.horizontal / coder->component[0].sampling_factor.horizontal,
        coder->sampling_factor.vertical / coder->component[0].sampling_factor.vertical,
        coder->sampling_factor.horizontal / coder->component[1].sampling_factor.horizontal,
        coder->sampling_factor.vertical / coder->component[1].sampling_factor.vertical,
        coder->sampling_factor.horizontal / coder->component[2].sampling_factor.horizontal,
        coder->sampling_factor.vertical / coder->component[2].sampling_factor.vertical
    );
    
#define RETURN_KERNEL_IF(KERNEL, COLOR, P1, P2, P3, P4, P5, P6) \
    if ( sampling_factor == gpujpeg_preprocessor_make_sampling_factor(P1, P2, P3, P4, P5, P6) ) { \
        int max_h = max(P1, max(P3, P5)); \
        int max_v = max(P2, max(P4, P6)); \
        if ( coder->param.verbose ) { \
            printf("Using faster kernel for preprocessor (precompiled %dx%d, %dx%d, %dx%d).\n", max_h / P1, max_v / P2, max_h / P3, max_v / P4, max_h / P5, max_v / P6); \
        } \
        return &KERNEL<COLOR, P1, P2, P3, P4, P5, P6>; \
    } 
#define RETURN_KERNEL(KERNEL, COLOR) \
    RETURN_KERNEL_IF(KERNEL, COLOR, 1, 1, 1, 1, 1, 1) \
    else RETURN_KERNEL_IF(KERNEL, COLOR, 1, 1, 2, 2, 2, 2) \
    else RETURN_KERNEL_IF(KERNEL, COLOR, 1, 1, 1, 2, 1, 2) \
    else RETURN_KERNEL_IF(KERNEL, COLOR, 1, 1, 2, 1, 2, 1) \
    else RETURN_KERNEL_IF(KERNEL, COLOR, 1, 1, 4, 4, 4, 4) \
    else { \
        if ( coder->param.verbose ) { \
            printf("Using slower kernel for preprocessor (dynamic %dx%d, %dx%d, %dx%d).\n", coder->component[0].sampling_factor.horizontal, coder->component[0].sampling_factor.vertical, coder->component[1].sampling_factor.horizontal, coder->component[1].sampling_factor.vertical, coder->component[2].sampling_factor.horizontal, coder->component[2].sampling_factor.vertical); \
        } \
        return &KERNEL<COLOR, GPUJPEG_DYNAMIC, GPUJPEG_DYNAMIC, GPUJPEG_DYNAMIC, GPUJPEG_DYNAMIC, GPUJPEG_DYNAMIC, GPUJPEG_DYNAMIC>; \
    } \

    // RGB color space
    if ( coder->param_image.color_space == GPUJPEG_RGB ) {
        assert(coder->param_image.sampling_factor == GPUJPEG_4_4_4);
        RETURN_KERNEL(gpujpeg_preprocessor_raw_to_comp_kernel_4_4_4, GPUJPEG_RGB);
    } 
    // YCbCr color space
    else if ( coder->param_image.color_space == GPUJPEG_YCBCR_BT601 ) {
        if ( coder->param_image.sampling_factor == GPUJPEG_4_4_4 ) {
            RETURN_KERNEL(gpujpeg_preprocessor_raw_to_comp_kernel_4_4_4, GPUJPEG_YCBCR_BT601);
        } else if ( coder->param_image.sampling_factor == GPUJPEG_4_2_2 ) {
            RETURN_KERNEL(gpujpeg_preprocessor_raw_to_comp_kernel_4_2_2, GPUJPEG_YCBCR_BT601);
        } else {
            assert(false);
        }
    } 
    // YCbCr color space
    else if ( coder->param_image.color_space == GPUJPEG_YCBCR_BT601_256LVLS ) {
        if ( coder->param_image.sampling_factor == GPUJPEG_4_4_4 ) {
            RETURN_KERNEL(gpujpeg_preprocessor_raw_to_comp_kernel_4_4_4, GPUJPEG_YCBCR_BT601_256LVLS);
        } else if ( coder->param_image.sampling_factor == GPUJPEG_4_2_2 ) {
            RETURN_KERNEL(gpujpeg_preprocessor_raw_to_comp_kernel_4_2_2, GPUJPEG_YCBCR_BT601_256LVLS);
        } else {
            assert(false);
        }
    }
    // YCbCr color space
    else if ( coder->param_image.color_space == GPUJPEG_YCBCR_BT709 ) {
        if ( coder->param_image.sampling_factor == GPUJPEG_4_4_4 ) {
            RETURN_KERNEL(gpujpeg_preprocessor_raw_to_comp_kernel_4_4_4, GPUJPEG_YCBCR_BT709);
        } else if ( coder->param_image.sampling_factor == GPUJPEG_4_2_2 ) {
            RETURN_KERNEL(gpujpeg_preprocessor_raw_to_comp_kernel_4_2_2, GPUJPEG_YCBCR_BT709);
        } else {
            assert(false);
        }
    }
    // Unknown color space
    else {
        assert(false);
    }
    
#undef RETURN_KERNEL_IF
#undef RETURN_KERNEL

    return NULL;
}

/** Documented at declaration */
int
gpujpeg_preprocessor_encoder_init(struct gpujpeg_encoder* encoder)
{
    encoder->coder.preprocessor = (void*)gpujpeg_preprocessor_select_encode_kernel(encoder);
    if ( encoder->coder.preprocessor == NULL )
        return -1;
    return 0;
}

/** Documented at declaration */
int
gpujpeg_preprocessor_encode(struct gpujpeg_encoder* encoder)
{    
    // Get coder
    struct gpujpeg_coder* coder = &encoder->coder;
    
    cudaMemset(coder->d_data, 0, coder->data_size * sizeof(uint8_t));

    // Select kernel
    gpujpeg_preprocessor_encode_kernel kernel = (gpujpeg_preprocessor_encode_kernel)coder->preprocessor;
    assert(kernel != NULL);
    
    int image_width = coder->param_image.width;
    int image_height = coder->param_image.height;
    
    // When loading 4:2:2 data of odd width, the data in fact has even width, so round it
    // (at least imagemagick convert tool generates data stream in this way)
    if ( coder->param_image.sampling_factor == GPUJPEG_4_2_2 )
        image_width = gpujpeg_div_and_round_up(coder->param_image.width, 2) * 2;
        
    // Prepare unit size
    assert(coder->param_image.sampling_factor == GPUJPEG_4_4_4 || coder->param_image.sampling_factor == GPUJPEG_4_2_2);
    int unitSize = coder->param_image.sampling_factor == GPUJPEG_4_4_4 ? 3 : 2;
    
    // Prepare kernel
    int alignedSize = gpujpeg_div_and_round_up(image_width * image_height, RGB_8BIT_THREADS) * RGB_8BIT_THREADS * unitSize;
    dim3 threads (RGB_8BIT_THREADS);
    dim3 grid (alignedSize / (RGB_8BIT_THREADS * unitSize));
    assert(alignedSize % (RGB_8BIT_THREADS * unitSize) == 0);
    if ( grid.x > GPUJPEG_CUDA_MAXIMUM_GRID_SIZE ) {
        grid.y = gpujpeg_div_and_round_up(grid.x, GPUJPEG_CUDA_MAXIMUM_GRID_SIZE);
        grid.x = GPUJPEG_CUDA_MAXIMUM_GRID_SIZE;
    }

    // Run kernel
    struct gpujpeg_preprocessor_data data;
    for ( int comp = 0; comp < 3; comp++ ) {
        assert(coder->sampling_factor.horizontal % coder->component[comp].sampling_factor.horizontal == 0);
        assert(coder->sampling_factor.vertical % coder->component[comp].sampling_factor.vertical == 0);
        data.comp[comp].d_data = coder->component[comp].d_data;
        data.comp[comp].sampling_factor.horizontal = coder->sampling_factor.horizontal / coder->component[comp].sampling_factor.horizontal;
        data.comp[comp].sampling_factor.vertical = coder->sampling_factor.vertical / coder->component[comp].sampling_factor.vertical;
        data.comp[comp].data_width = coder->component[comp].data_width;
    }
    kernel<<<grid, threads>>>(
        data,
        coder->d_data_raw,
        image_width,
        image_height
    );
    cudaThreadSynchronize();
    gpujpeg_cuda_check_error("Preprocessor encoding failed");
        
    return 0;
}

/**
 * Store value to component data buffer in specified position by buffer size and subsampling
 * 
 * @param value
 * @param position_x
 * @param position_y
 * @param comp
 */
 template<
    uint8_t s_samp_factor_h = GPUJPEG_DYNAMIC,
    uint8_t s_samp_factor_v = GPUJPEG_DYNAMIC
>
struct gpujpeg_preprocessor_comp_to_raw_load
{
    static __device__ void
    perform(float & value, int position_x, int position_y, struct gpujpeg_preprocessor_data_component & comp)
    {
        uint8_t samp_factor_h = s_samp_factor_h;
        if ( samp_factor_h == GPUJPEG_DYNAMIC ) {
            samp_factor_h = comp.sampling_factor.horizontal;
        }
        uint8_t samp_factor_v = s_samp_factor_v;
        if ( samp_factor_v == GPUJPEG_DYNAMIC ) {
            samp_factor_v = comp.sampling_factor.vertical;
        }
        
        position_x = position_x / samp_factor_h;
        position_y = position_y / samp_factor_v;
        
        int data_position = position_y * comp.data_width + position_x;
        value = (float)comp.d_data[data_position];
    }
};
template<>
struct gpujpeg_preprocessor_comp_to_raw_load<1, 1>
{
    static __device__ void
    perform(float & value, int position_x, int position_y, struct gpujpeg_preprocessor_data_component & comp)
    {
        int data_position = position_y * comp.data_width + position_x;
        value = (float)comp.d_data[data_position];
    }
};

/**
 * Kernel - Copy three separated component buffers into target image data
 *
 * @param d_c1  First component buffer
 * @param d_c2  Second component buffer
 * @param d_c3  Third component buffer
 * @param d_target  Image target data
 * @param pixel_count  Number of pixels to copy
 * @return void
 */
typedef void (*gpujpeg_preprocessor_decode_kernel)(struct gpujpeg_preprocessor_data data, uint8_t* d_data_raw, int image_width, int image_height);

/** Specialization [sampling factor is 4:4:4] */
template<
    enum gpujpeg_color_space color_space,
    uint8_t s_comp1_samp_factor_h, uint8_t s_comp1_samp_factor_v,
    uint8_t s_comp2_samp_factor_h, uint8_t s_comp2_samp_factor_v,
    uint8_t s_comp3_samp_factor_h, uint8_t s_comp3_samp_factor_v
>
__global__ void
gpujpeg_preprocessor_comp_to_raw_kernel_4_4_4(struct gpujpeg_preprocessor_data data, uint8_t* d_data_raw, int image_width, int image_height)
{
    int x  = threadIdx.x;
    int gX = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x;
    int image_position = gX + x;
    if ( image_position >= (image_width * image_height) )
        return;
    int image_position_x = image_position % image_width;
    int image_position_y = image_position / image_width;
        
    // Load
    float r1;
    float r2;
    float r3;
    gpujpeg_preprocessor_comp_to_raw_load<s_comp1_samp_factor_h, s_comp1_samp_factor_v>::perform(r1, image_position_x, image_position_y, data.comp[0]);
    gpujpeg_preprocessor_comp_to_raw_load<s_comp2_samp_factor_h, s_comp2_samp_factor_v>::perform(r2, image_position_x, image_position_y, data.comp[1]);
    gpujpeg_preprocessor_comp_to_raw_load<s_comp3_samp_factor_h, s_comp3_samp_factor_v>::perform(r3, image_position_x, image_position_y, data.comp[2]);
    
    // Color transform
    gpujpeg_color_transform<GPUJPEG_YCBCR_BT601_256LVLS, color_space>::perform(r1, r2, r3);
    
    // Save
    image_position = image_position * 3;
    d_data_raw[image_position + 0] = (uint8_t)round(r1);
    d_data_raw[image_position + 1] = (uint8_t)round(r2);
    d_data_raw[image_position + 2] = (uint8_t)round(r3);
}

/** Specialization [sampling factor is 4:2:2] */
template<
    enum gpujpeg_color_space color_space,
    uint8_t s_comp1_samp_factor_h, uint8_t s_comp1_samp_factor_v,
    uint8_t s_comp2_samp_factor_h, uint8_t s_comp2_samp_factor_v,
    uint8_t s_comp3_samp_factor_h, uint8_t s_comp3_samp_factor_v
>
__global__ void
gpujpeg_preprocessor_comp_to_raw_kernel_4_2_2(struct gpujpeg_preprocessor_data data, uint8_t* d_data_raw, int image_width, int image_height)
{
    int x  = threadIdx.x;
    int gX = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x;
    int image_position = gX + x;
    if ( image_position >= (image_width * image_height) )
        return;
    int image_position_x = image_position % image_width;
    int image_position_y = image_position / image_width;
        
    // Load
    float r1;
    float r2;
    float r3;
    gpujpeg_preprocessor_comp_to_raw_load<s_comp1_samp_factor_h, s_comp1_samp_factor_v>::perform(r1, image_position_x, image_position_y, data.comp[0]);
    gpujpeg_preprocessor_comp_to_raw_load<s_comp2_samp_factor_h, s_comp2_samp_factor_v>::perform(r2, image_position_x, image_position_y, data.comp[1]);
    gpujpeg_preprocessor_comp_to_raw_load<s_comp3_samp_factor_h, s_comp3_samp_factor_v>::perform(r3, image_position_x, image_position_y, data.comp[2]);
    
    // Color transform
    gpujpeg_color_transform<GPUJPEG_YCBCR_BT601_256LVLS, color_space>::perform(r1, r2, r3);
    
    // Save
    image_position = image_position * 2;
    d_data_raw[image_position + 1] = (uint8_t)round(r1);
    if ( (image_position_x % 2) == 0 )
        d_data_raw[image_position + 0] = (uint8_t)round(r2);
    else
        d_data_raw[image_position + 0] = (uint8_t)round(r3);
}

/**
 * Select preprocessor decode kernel
 * 
 * @param decoder
 * @return kernel
 */
gpujpeg_preprocessor_decode_kernel
gpujpeg_preprocessor_select_decode_kernel(struct gpujpeg_decoder* decoder)
{
    // Get coder
    struct gpujpeg_coder* coder = &decoder->coder;
    
    gpujpeg_preprocessor_sampling_factor_t sampling_factor = gpujpeg_preprocessor_make_sampling_factor(
        coder->sampling_factor.horizontal / coder->component[0].sampling_factor.horizontal,
        coder->sampling_factor.vertical / coder->component[0].sampling_factor.vertical,
        coder->sampling_factor.horizontal / coder->component[1].sampling_factor.horizontal,
        coder->sampling_factor.vertical / coder->component[1].sampling_factor.vertical,
        coder->sampling_factor.horizontal / coder->component[2].sampling_factor.horizontal,
        coder->sampling_factor.vertical / coder->component[2].sampling_factor.vertical
    );
    
#define RETURN_KERNEL_IF(KERNEL, COLOR, P1, P2, P3, P4, P5, P6) \
    if ( sampling_factor == gpujpeg_preprocessor_make_sampling_factor(P1, P2, P3, P4, P5, P6) ) { \
        int max_h = max(P1, max(P3, P5)); \
        int max_v = max(P2, max(P4, P6)); \
        if ( coder->param.verbose ) { \
            printf("Using faster kernel for postprocessor (precompiled %dx%d, %dx%d, %dx%d).\n", max_h / P1, max_v / P2, max_h / P3, max_v / P4, max_h / P5, max_v / P6); \
        } \
        return &KERNEL<COLOR, P1, P2, P3, P4, P5, P6>; \
    } 
#define RETURN_KERNEL(KERNEL, COLOR) \
    RETURN_KERNEL_IF(KERNEL, COLOR, 1, 1, 1, 1, 1, 1) \
    else RETURN_KERNEL_IF(KERNEL, COLOR, 1, 1, 2, 2, 2, 2) \
    else RETURN_KERNEL_IF(KERNEL, COLOR, 1, 1, 1, 2, 1, 2) \
    else RETURN_KERNEL_IF(KERNEL, COLOR, 1, 1, 2, 1, 2, 1) \
    else RETURN_KERNEL_IF(KERNEL, COLOR, 1, 1, 4, 4, 4, 4) \
    else { \
        if ( coder->param.verbose ) { \
            printf("Using slower kernel for postprocessor (dynamic %dx%d, %dx%d, %dx%d).\n", coder->component[0].sampling_factor.horizontal, coder->component[0].sampling_factor.vertical, coder->component[1].sampling_factor.horizontal, coder->component[1].sampling_factor.vertical, coder->component[2].sampling_factor.horizontal, coder->component[2].sampling_factor.vertical); \
        } \
        return &KERNEL<COLOR, GPUJPEG_DYNAMIC, GPUJPEG_DYNAMIC, GPUJPEG_DYNAMIC, GPUJPEG_DYNAMIC, GPUJPEG_DYNAMIC, GPUJPEG_DYNAMIC>; \
    } \
    
    // RGB color space
    if ( coder->param_image.color_space == GPUJPEG_RGB ) {
        assert(coder->param_image.sampling_factor == GPUJPEG_4_4_4);
        RETURN_KERNEL(gpujpeg_preprocessor_comp_to_raw_kernel_4_4_4, GPUJPEG_RGB);
    } 
    // YCbCr color space
    else if ( coder->param_image.color_space == GPUJPEG_YCBCR_BT601 ) {
        if ( coder->param_image.sampling_factor == GPUJPEG_4_4_4 ) {
            RETURN_KERNEL(gpujpeg_preprocessor_comp_to_raw_kernel_4_4_4, GPUJPEG_YCBCR_BT601)
        } else if ( coder->param_image.sampling_factor == GPUJPEG_4_2_2 ) {
            RETURN_KERNEL(gpujpeg_preprocessor_comp_to_raw_kernel_4_2_2, GPUJPEG_YCBCR_BT601)
        } else {
            assert(false);
        }
    }
    // YCbCr color space
    else if ( coder->param_image.color_space == GPUJPEG_YCBCR_BT601_256LVLS ) {
        if ( coder->param_image.sampling_factor == GPUJPEG_4_4_4 ) {
            RETURN_KERNEL(gpujpeg_preprocessor_comp_to_raw_kernel_4_4_4, GPUJPEG_YCBCR_BT601_256LVLS)
        } else if ( coder->param_image.sampling_factor == GPUJPEG_4_2_2 ) {
            RETURN_KERNEL(gpujpeg_preprocessor_comp_to_raw_kernel_4_2_2, GPUJPEG_YCBCR_BT601_256LVLS)
        } else {
            assert(false);
        }
    }
    // YCbCr color space
    else if ( coder->param_image.color_space == GPUJPEG_YCBCR_BT709 ) {
        if ( coder->param_image.sampling_factor == GPUJPEG_4_4_4 ) {
            RETURN_KERNEL(gpujpeg_preprocessor_comp_to_raw_kernel_4_4_4, GPUJPEG_YCBCR_BT709)
        } else if ( coder->param_image.sampling_factor == GPUJPEG_4_2_2 ) {
            RETURN_KERNEL(gpujpeg_preprocessor_comp_to_raw_kernel_4_2_2, GPUJPEG_YCBCR_BT709)
        } else {
            assert(false);
        }
    }
    // Unknown color space
    else {
        assert(false);
    }
    
#undef RETURN_KERNEL_IF
#undef RETURN_KERNEL
    
    return NULL;
}

/** Documented at declaration */
int
gpujpeg_preprocessor_decoder_init(struct gpujpeg_decoder* decoder)
{
    decoder->coder.preprocessor = (void*)gpujpeg_preprocessor_select_decode_kernel(decoder);
    if ( decoder->coder.preprocessor == NULL )
        return -1;
    return 0;
}

/** Documented at declaration */
int
gpujpeg_preprocessor_decode(struct gpujpeg_decoder* decoder)
{
    // Get coder
    struct gpujpeg_coder* coder = &decoder->coder;
    
    cudaMemset(coder->d_data_raw, 0, coder->data_raw_size * sizeof(uint8_t));
    
    // Select kernel
    gpujpeg_preprocessor_decode_kernel kernel = (gpujpeg_preprocessor_decode_kernel)coder->preprocessor;
    assert(kernel != NULL);
    
    int image_width = coder->param_image.width;
    int image_height = coder->param_image.height;
    
    // When saving 4:2:2 data of odd width, the data should have even width, so round it
    if ( coder->param_image.sampling_factor == GPUJPEG_4_2_2 )
        image_width = gpujpeg_div_and_round_up(coder->param_image.width, 2) * 2;
        
    // Prepare unit size
    assert(coder->param_image.sampling_factor == GPUJPEG_4_4_4 || coder->param_image.sampling_factor == GPUJPEG_4_2_2);
    int unitSize = coder->param_image.sampling_factor == GPUJPEG_4_4_4 ? 3 : 2;
    
    // Prepare kernel
    int alignedSize = gpujpeg_div_and_round_up(image_width * image_height, RGB_8BIT_THREADS) * RGB_8BIT_THREADS * unitSize;
    dim3 threads (RGB_8BIT_THREADS);
    dim3 grid (alignedSize / (RGB_8BIT_THREADS * unitSize));
    assert(alignedSize % (RGB_8BIT_THREADS * unitSize) == 0);
    if ( grid.x > GPUJPEG_CUDA_MAXIMUM_GRID_SIZE ) {
        grid.y = gpujpeg_div_and_round_up(grid.x, GPUJPEG_CUDA_MAXIMUM_GRID_SIZE);
        grid.x = GPUJPEG_CUDA_MAXIMUM_GRID_SIZE;
    }

    // Run kernel
    struct gpujpeg_preprocessor_data data;
    for ( int comp = 0; comp < 3; comp++ ) {
        assert(coder->sampling_factor.horizontal % coder->component[comp].sampling_factor.horizontal == 0);
        assert(coder->sampling_factor.vertical % coder->component[comp].sampling_factor.vertical == 0);
        data.comp[comp].d_data = coder->component[comp].d_data;
        data.comp[comp].sampling_factor.horizontal = coder->sampling_factor.horizontal / coder->component[comp].sampling_factor.horizontal;
        data.comp[comp].sampling_factor.vertical = coder->sampling_factor.vertical / coder->component[comp].sampling_factor.vertical;
        data.comp[comp].data_width = coder->component[comp].data_width;
    }
    kernel<<<grid, threads>>>(
        data,
        coder->d_data_raw, 
        image_width,
        image_height
    );
    cudaThreadSynchronize();
    gpujpeg_cuda_check_error("Preprocessor encoding failed");
    
    return 0;
}
