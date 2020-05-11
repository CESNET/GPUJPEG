/*
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
/**
 * @file
 * @brief
 * This file contains preprocessors and postprocessors between raw image and
 * a common format for computational kernels. It also does color space
 * transformations.
 * @todo
 * Preprocesors/postprocessors don't work for planar formats, only memcpy is
 * performed now (subsampling and color space cannot be changed).
 */

#include "gpujpeg_colorspace.h"
#include "gpujpeg_preprocessor.h"
#include "gpujpeg_util.h"

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
 * Prepares fixed divisor for dividing unsigned integers up to 2^31
 * with unsigned integers up to 2^31.
 * Source: http://www.hackersdelight.org/HDcode/magic.c.txt
 * Modified for positive numbers only.
 */
static void
gpujpeg_const_div_prepare(const uint32_t d, uint32_t & pre_div_mul, uint32_t & pre_div_shift) {
    if(d > 1) {
        uint32_t delta;
        const uint32_t two31 = 0x80000000; // 2**31.
        const uint32_t anc = two31 - 1 - two31 % d; // Absolute value of nc.
        int p = 31;                        // Init. p.
        uint32_t q1 = two31 / anc;         // Init. q1 = 2**p/|nc|.
        uint32_t r1 = two31 - q1 * anc;    // Init. r1 = rem(2**p, |nc|).
        uint32_t q2 = two31 / d;           // Init. q2 = 2**p/|d|.
        uint32_t r2 = two31 - q2 * d;      // Init. r2 = rem(2**p, |d|).
        do {
            p = p + 1;
            q1 = 2 * q1;                   // Update q1 = 2**p/|nc|.
            r1 = 2 * r1;                   // Update r1 = rem(2**p, |nc|).
            if (r1 >= anc) {               // (Must be an unsigned
                q1 = q1 + 1;               // comparison here).
                r1 = r1 - anc;
            }
            q2 = 2 * q2;                   // Update q2 = 2**p/|d|.
            r2 = 2 * r2;                   // Update r2 = rem(2**p, |d|).
            if (r2 >= d) {                 // (Must be an unsigned
                q2 = q2 + 1;               // comparison here).
                r2 = r2 - d;
            }
            delta = d - r2;
        } while (q1 < delta || (q1 == delta && r1 == 0));
        pre_div_mul = q2 + 1;
        pre_div_shift = p - 32;            // shift amount to return.
    } else {
        pre_div_mul = 0;                   // special case for d = 1
        pre_div_shift = 0;
    }
}


/**
 * Divides unsigned numerator (up to 2^31) by precomputed constant denominator.
 */
__device__ static uint32_t
gpujpeg_const_div_divide(const uint32_t numerator, const uint32_t pre_div_mul, const uint32_t pre_div_shift) {
    return pre_div_mul ? __umulhi(numerator, pre_div_mul) >> pre_div_shift : numerator;
}

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
    unsigned int s_samp_factor_h,
    unsigned int s_samp_factor_v
>
static __device__ void
gpujpeg_preprocessor_raw_to_comp_store(uint8_t value, unsigned int position_x, unsigned int position_y, struct gpujpeg_preprocessor_data_component & comp)
{
    const unsigned int samp_factor_h = ( s_samp_factor_h == GPUJPEG_DYNAMIC ) ? comp.sampling_factor.horizontal : s_samp_factor_h;
    const unsigned int samp_factor_v = ( s_samp_factor_v == GPUJPEG_DYNAMIC ) ? comp.sampling_factor.vertical : s_samp_factor_v;

    if ( (position_x % samp_factor_h) || (position_y % samp_factor_v) )
        return;

    position_x = position_x / samp_factor_h;
    position_y = position_y / samp_factor_v;

    const unsigned int data_position = position_y * comp.data_width + position_x;
    comp.d_data[data_position] = value;
}

/**
 * Kernel - Copy raw image source data into three separated component buffers
 */
typedef void (*gpujpeg_preprocessor_encode_kernel)(struct gpujpeg_preprocessor_data data, const uint8_t* d_data_raw, const uint8_t* d_data_raw_end, int image_width, int image_height, uint32_t width_div_mul, uint32_t width_div_shift);

/** Specialization [sampling factor is 4:4:4]
 * @tparam sample_width widht of the sample (3 for RGB, 4 for RGBX)
 */
template<
    enum gpujpeg_color_space color_space_internal,
    enum gpujpeg_color_space color_space,
    uint8_t s_comp1_samp_factor_h, uint8_t s_comp1_samp_factor_v,
    uint8_t s_comp2_samp_factor_h, uint8_t s_comp2_samp_factor_v,
    uint8_t s_comp3_samp_factor_h, uint8_t s_comp3_samp_factor_v,
    int sample_width
>
__global__ void
gpujpeg_preprocessor_raw_to_comp_kernel_4_4_4(struct gpujpeg_preprocessor_data data, const uint8_t* d_data_raw, const uint8_t* d_data_raw_end, int image_width, int image_height, uint32_t width_div_mul, uint32_t width_div_shift)
{
    int x  = threadIdx.x;
    int gX = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x;

    // Load to shared
    __shared__ unsigned char s_data[RGB_8BIT_THREADS * sample_width];
    if ( (x * 4) < RGB_8BIT_THREADS * sample_width ) {
        uint32_t* s = (uint32_t *) d_data_raw + ((gX * sample_width) >> 2) + x;
        uint32_t* d = (uint32_t *) s_data + x;
        if ((uint8_t *) s < d_data_raw_end) {
            *d = *s;
        }
    }
    __syncthreads();

    // Load
    int offset = x * sample_width;
    uint8_t r1 = s_data[offset];
    uint8_t r2 = s_data[offset + 1];
    uint8_t r3 = s_data[offset + 2];

    // Load Order
    gpujpeg_color_order<color_space>::perform_load(r1, r2, r3);

    // Color transform
    gpujpeg_color_transform<color_space, color_space_internal>::perform(r1, r2, r3);

    // Position
    int image_position = gX + x;
    int image_position_y = gpujpeg_const_div_divide(image_position, width_div_mul, width_div_shift);
    int image_position_x = image_position - (image_position_y * image_width);

    // Store
    if ( image_position < (image_width * image_height) ) {
        gpujpeg_preprocessor_raw_to_comp_store<s_comp1_samp_factor_h, s_comp1_samp_factor_v>(r1, image_position_x, image_position_y, data.comp[0]);
        gpujpeg_preprocessor_raw_to_comp_store<s_comp2_samp_factor_h, s_comp2_samp_factor_v>(r2, image_position_x, image_position_y, data.comp[1]);
        gpujpeg_preprocessor_raw_to_comp_store<s_comp3_samp_factor_h, s_comp3_samp_factor_v>(r3, image_position_x, image_position_y, data.comp[2]);
    }
}

/** Specialization [sampling factor is 4:2:2] */
template<
    enum gpujpeg_color_space color_space_internal,
    enum gpujpeg_color_space color_space,
    uint8_t s_comp1_samp_factor_h, uint8_t s_comp1_samp_factor_v,
    uint8_t s_comp2_samp_factor_h, uint8_t s_comp2_samp_factor_v,
    uint8_t s_comp3_samp_factor_h, uint8_t s_comp3_samp_factor_v
>
__global__ void
gpujpeg_preprocessor_raw_to_comp_kernel_4_2_2(struct gpujpeg_preprocessor_data data, const uint8_t* d_data_raw, const uint8_t* d_data_raw_end, int image_width, int image_height, uint32_t width_div_mul, uint32_t width_div_shift)
{
    int x  = threadIdx.x;
    int gX = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x;

    // Load to shared
    __shared__ unsigned char s_data[RGB_8BIT_THREADS * 2];
    if ( (x * 4) < RGB_8BIT_THREADS * 2 ) {
        uint32_t* s = (uint32_t *) d_data_raw + ((gX * 2) >> 2) + x;
        uint32_t* d = (uint32_t *) s_data + x;
        if ((uint8_t *) s < d_data_raw_end) {
            *d = *s;
        }
    }
    __syncthreads();

    // Load
    const unsigned int offset = x * 2;
    uint8_t r1;
    uint8_t r2 = s_data[offset + 1];
    uint8_t r3;
    if ( (gX + x) % 2 == 0 ) {
        r1 = s_data[offset];
        r3 = s_data[offset + 2];
    } else {
        r1 = s_data[offset - 2];
        r3 = s_data[offset];
    }

    // Load Order
    gpujpeg_color_order<color_space>::perform_load(r1, r2, r3);

    // Color transform
    gpujpeg_color_transform<color_space, color_space_internal>::perform(r1, r2, r3);

    // Position
    int image_position = gX + x;
    int image_position_y = gpujpeg_const_div_divide(image_position, width_div_mul, width_div_shift);
    int image_position_x = image_position - (image_position_y * image_width);

    // Store
    if ( image_position < (image_width * image_height) ) {
        gpujpeg_preprocessor_raw_to_comp_store<s_comp1_samp_factor_h, s_comp1_samp_factor_v>(r1, image_position_x, image_position_y, data.comp[0]);
        gpujpeg_preprocessor_raw_to_comp_store<s_comp2_samp_factor_h, s_comp2_samp_factor_v>(r2, image_position_x, image_position_y, data.comp[1]);
        gpujpeg_preprocessor_raw_to_comp_store<s_comp3_samp_factor_h, s_comp3_samp_factor_v>(r3, image_position_x, image_position_y, data.comp[2]);
    }
}

/**
 * Select preprocessor encode kernel
 *
 * @param encoder
 * @return kernel
 */
template<enum gpujpeg_color_space color_space_internal>
gpujpeg_preprocessor_encode_kernel
gpujpeg_preprocessor_select_encode_kernel(struct gpujpeg_coder* coder)
{
    gpujpeg_preprocessor_sampling_factor_t sampling_factor = gpujpeg_preprocessor_make_sampling_factor(
        coder->sampling_factor.horizontal / coder->component[0].sampling_factor.horizontal,
        coder->sampling_factor.vertical / coder->component[0].sampling_factor.vertical,
        coder->sampling_factor.horizontal / coder->component[1].sampling_factor.horizontal,
        coder->sampling_factor.vertical / coder->component[1].sampling_factor.vertical,
        coder->sampling_factor.horizontal / coder->component[2].sampling_factor.horizontal,
        coder->sampling_factor.vertical / coder->component[2].sampling_factor.vertical
    );

#define RETURN_KERNEL_IF(PIXEL_FORMAT, COLOR, P1, P2, P3, P4, P5, P6) \
    if ( sampling_factor == gpujpeg_preprocessor_make_sampling_factor(P1, P2, P3, P4, P5, P6) ) { \
        int max_h = max(P1, max(P3, P5)); \
        int max_v = max(P2, max(P4, P6)); \
        if ( coder->param.verbose ) { \
            printf("Using faster kernel for preprocessor (precompiled %dx%d, %dx%d, %dx%d).\n", max_h / P1, max_v / P2, max_h / P3, max_v / P4, max_h / P5, max_v / P6); \
        } \
        if ( PIXEL_FORMAT == GPUJPEG_444_U8_P012 ) { \
            return &gpujpeg_preprocessor_raw_to_comp_kernel_4_4_4<color_space_internal, COLOR, P1, P2, P3, P4, P5, P6, 3>; \
        } else if ( PIXEL_FORMAT == GPUJPEG_444_U8_P012Z ) { \
            return &gpujpeg_preprocessor_raw_to_comp_kernel_4_4_4<color_space_internal, COLOR, P1, P2, P3, P4, P5, P6, 4>; \
        } else if ( PIXEL_FORMAT == GPUJPEG_422_U8_P1020 ) { \
            return &gpujpeg_preprocessor_raw_to_comp_kernel_4_2_2<color_space_internal, COLOR, P1, P2, P3, P4, P5, P6>; \
        } else { \
            assert(false); \
        } \
    }
#define RETURN_KERNEL(PIXEL_FORMAT, COLOR) \
    RETURN_KERNEL_IF(PIXEL_FORMAT, COLOR, 1, 1, 1, 1, 1, 1) \
    else RETURN_KERNEL_IF(PIXEL_FORMAT, COLOR, 1, 1, 2, 2, 2, 2) \
    else RETURN_KERNEL_IF(PIXEL_FORMAT, COLOR, 1, 1, 1, 2, 1, 2) \
    else RETURN_KERNEL_IF(PIXEL_FORMAT, COLOR, 1, 1, 2, 1, 2, 1) \
    else RETURN_KERNEL_IF(PIXEL_FORMAT, COLOR, 1, 1, 4, 4, 4, 4) \
    else { \
        if ( coder->param.verbose ) { \
            printf("Using slower kernel for preprocessor (dynamic %dx%d, %dx%d, %dx%d).\n", coder->component[0].sampling_factor.horizontal, coder->component[0].sampling_factor.vertical, coder->component[1].sampling_factor.horizontal, coder->component[1].sampling_factor.vertical, coder->component[2].sampling_factor.horizontal, coder->component[2].sampling_factor.vertical); \
        } \
        if ( PIXEL_FORMAT == GPUJPEG_444_U8_P012 ) { \
            return &gpujpeg_preprocessor_raw_to_comp_kernel_4_4_4<color_space_internal, COLOR, GPUJPEG_DYNAMIC, GPUJPEG_DYNAMIC, GPUJPEG_DYNAMIC, GPUJPEG_DYNAMIC, GPUJPEG_DYNAMIC, GPUJPEG_DYNAMIC, 3>; \
        } else if ( PIXEL_FORMAT == GPUJPEG_444_U8_P012Z ) { \
            return &gpujpeg_preprocessor_raw_to_comp_kernel_4_4_4<color_space_internal, COLOR, GPUJPEG_DYNAMIC, GPUJPEG_DYNAMIC, GPUJPEG_DYNAMIC, GPUJPEG_DYNAMIC, GPUJPEG_DYNAMIC, GPUJPEG_DYNAMIC, 4>; \
        } else if ( PIXEL_FORMAT == GPUJPEG_422_U8_P1020 ) { \
            return &gpujpeg_preprocessor_raw_to_comp_kernel_4_2_2<color_space_internal, COLOR, GPUJPEG_DYNAMIC, GPUJPEG_DYNAMIC, GPUJPEG_DYNAMIC, GPUJPEG_DYNAMIC, GPUJPEG_DYNAMIC, GPUJPEG_DYNAMIC>; \
        } else { \
            assert(false); \
        } \
    } \

    // None color space
    if ( coder->param_image.color_space == GPUJPEG_NONE ) {
        RETURN_KERNEL(coder->param_image.pixel_format, GPUJPEG_NONE);
    }
    // RGB color space
    else if ( coder->param_image.color_space == GPUJPEG_RGB ) {
        RETURN_KERNEL(coder->param_image.pixel_format, GPUJPEG_RGB);
    }
    // YCbCr color space
    else if ( coder->param_image.color_space == GPUJPEG_YCBCR_BT601 ) {
        RETURN_KERNEL(coder->param_image.pixel_format, GPUJPEG_YCBCR_BT601);
    }
    // YCbCr color space
    else if ( coder->param_image.color_space == GPUJPEG_YCBCR_BT601_256LVLS ) {
        RETURN_KERNEL(coder->param_image.pixel_format, GPUJPEG_YCBCR_BT601_256LVLS);
    }
    // YCbCr color space
    else if ( coder->param_image.color_space == GPUJPEG_YCBCR_BT709 ) {
        RETURN_KERNEL(coder->param_image.pixel_format, GPUJPEG_YCBCR_BT709);
    }
    // YUV color space
    else if ( coder->param_image.color_space == GPUJPEG_YUV ) {
        RETURN_KERNEL(coder->param_image.pixel_format, GPUJPEG_YUV);
    }
    // Unknown color space
    else {
        assert(false);
    }

#undef RETURN_KERNEL_IF
#undef RETURN_KERNEL

    return NULL;
}

/* Documented at declaration */
int
gpujpeg_preprocessor_encoder_init(struct gpujpeg_coder* coder)
{
    if ( coder->param_image.comp_count == 1 ) {
        return 0;
    }

    assert(coder->param_image.comp_count == 3);

    switch (coder->param_image.pixel_format) {
        case GPUJPEG_444_U8_P012:
        case GPUJPEG_444_U8_P012Z:
        case GPUJPEG_422_U8_P1020:
        {
            coder->preprocessor = NULL;
            if (coder->param.color_space_internal == GPUJPEG_NONE) {
                coder->preprocessor = (void*)gpujpeg_preprocessor_select_encode_kernel<GPUJPEG_NONE>(coder);
            }
            else if (coder->param.color_space_internal == GPUJPEG_RGB) {
                coder->preprocessor = (void*)gpujpeg_preprocessor_select_encode_kernel<GPUJPEG_RGB>(coder);
            }
            else if (coder->param.color_space_internal == GPUJPEG_YCBCR_BT601_256LVLS) {
                coder->preprocessor = (void*)gpujpeg_preprocessor_select_encode_kernel<GPUJPEG_YCBCR_BT601_256LVLS>(coder);
            }
            else {
                assert(false);
            }
            if ( coder->preprocessor == NULL ) {
                return -1;
            }
            break;
        }
        default:
        {
            coder->preprocessor = NULL;
            break;
        }
    }
    return 0;
}

int
gpujpeg_preprocessor_encode_interlaced(struct gpujpeg_encoder * encoder)
{
    struct gpujpeg_coder* coder = &encoder->coder;

    assert(coder->param_image.comp_count == 3);

    cudaMemsetAsync(coder->d_data, 0, coder->data_size * sizeof(uint8_t), *(encoder->stream));
    gpujpeg_cuda_check_error("Preprocessor memset failed", return -1);

    // Select kernel
    gpujpeg_preprocessor_encode_kernel kernel = (gpujpeg_preprocessor_encode_kernel) coder->preprocessor;
    assert(kernel != NULL);

    int image_width = coder->param_image.width;
    int image_height = coder->param_image.height;

    // When loading 4:2:2 data of odd width, the data in fact has even width, so round it
    // (at least imagemagick convert tool generates data stream in this way)
    if (coder->param_image.pixel_format == GPUJPEG_422_U8_P1020) {
        image_width = (coder->param_image.width + 1) & ~1;
    }

    // Prepare unit size
    int unitSize = gpujpeg_pixel_format_get_unit_size(coder->param_image.pixel_format);
    assert(unitSize != 0);

    // Prepare kernel
    int alignedSize = gpujpeg_div_and_round_up(image_width * image_height, RGB_8BIT_THREADS) * RGB_8BIT_THREADS * unitSize;
    dim3 threads (RGB_8BIT_THREADS);
    dim3 grid (alignedSize / (RGB_8BIT_THREADS * unitSize));
    assert(alignedSize % (RGB_8BIT_THREADS * unitSize) == 0);
    while ( grid.x > GPUJPEG_CUDA_MAXIMUM_GRID_SIZE ) {
        grid.y *= 2;
        grid.x = gpujpeg_div_and_round_up(grid.x, 2);
    }

    // Decompose input image width for faster division using multiply-high and right shift
    uint32_t width_div_mul, width_div_shift;
    gpujpeg_const_div_prepare(image_width, width_div_mul, width_div_shift);

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
    kernel<<<grid, threads, 0, *(encoder->stream)>>>(
        data,
        coder->d_data_raw,
        coder->d_data_raw + coder->data_raw_size,
        image_width,
        image_height,
        width_div_mul,
        width_div_shift
    );
    gpujpeg_cuda_check_error("Preprocessor encoding failed", return -1);

    return 0;
}

/**
 * Copies raw data from source image to GPU memory without running
 * any preprocessor kernel.
 *
 * This assumes that the JPEG has same color space as input raw image and
 * currently also that the component subsampling correspond between raw and
 * JPEG (although at least different horizontal subsampling can be quite
 * easily done).
 *
 * @todo
 * Use preprocessing kernels as for packed formats.
 */
static int
gpujpeg_preprocessor_encoder_copy_planar_data(struct gpujpeg_encoder * encoder)
{
    struct gpujpeg_coder * coder = &encoder->coder;
    assert(coder->param_image.comp_count == 1 ||
            coder->param_image.comp_count == 3);

    if (coder->param_image.comp_count == 3 && coder->param_image.color_space != coder->param.color_space_internal) {
            fprintf(stderr, "Encoding JPEG from a planar pixel format supported only when no color transformation is required. "
                            "JPEG internal color space is set to \"%s\", image is \"%s\".\n",
                            gpujpeg_color_space_get_name(coder->param.color_space_internal),
                            gpujpeg_color_space_get_name(coder->param_image.color_space));
            return -1;
    }
    size_t data_raw_offset = 0;
    bool needs_stride = false; // true if width is not divisible by MCU width
    for (int i = 0; i < coder->param_image.comp_count; ++i) {
        needs_stride = needs_stride || coder->component[i].width != coder->component[i].data_width;
    }
    if (!needs_stride) {
            for (int i = 0; i < coder->param_image.comp_count; ++i) {
                    size_t component_size = coder->component[i].width * coder->component[i].height;
                    cudaMemcpyAsync(coder->component[i].d_data, coder->d_data_raw + data_raw_offset, component_size, cudaMemcpyDeviceToDevice, *(encoder->stream));
                    data_raw_offset += component_size;
            }
    } else {
            for (int i = 0; i < coder->param_image.comp_count; ++i) {
                    int spitch = coder->component[i].width;
                    int dpitch = coder->component[i].data_width;
                    size_t component_size = spitch * coder->component[i].height;
                    cudaMemcpy2DAsync(coder->component[i].d_data, dpitch, coder->d_data_raw + data_raw_offset, spitch, spitch, coder->component[i].height, cudaMemcpyDeviceToDevice, *(encoder->stream));
                    data_raw_offset += component_size;
            }
    }
    gpujpeg_cuda_check_error("Preprocessor copy failed", return -1);
    return 0;
}

/* Documented at declaration */
int
gpujpeg_preprocessor_encode(struct gpujpeg_encoder * encoder)
{
    struct gpujpeg_coder * coder = &encoder->coder;
    if (gpujpeg_pixel_format_is_interleaved(coder->param_image.pixel_format)) {
            return gpujpeg_preprocessor_encode_interlaced(encoder);
    } else {
        const int *sampling_factors = gpujpeg_pixel_format_get_sampling_factor(coder->param_image.pixel_format);
        for (int i = 0; i < coder->param_image.comp_count; ++i) {
            if (coder->component[i].sampling_factor.horizontal != sampling_factors[i * 2]
                    || coder->component[0].sampling_factor.vertical != sampling_factors[i * 2 + 1]) {
                fprintf(stderr, "Encoding JPEG from pixel format %s is supported only when %s subsampling inside JPEG is used.\n",
                        gpujpeg_pixel_format_get_name(coder->param_image.pixel_format),
                        gpujpeg_subsampling_get_name(coder->param_image.comp_count,
                            gpujpeg_get_component_subsampling(coder->param_image.pixel_format)));
                return GPUJPEG_ERR_WRONG_SUBSAMPLING;
            }
        }
        return gpujpeg_preprocessor_encoder_copy_planar_data(encoder);
    }
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
    perform(uint8_t & value, int position_x, int position_y, struct gpujpeg_preprocessor_data_component & comp)
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
        value = comp.d_data[data_position];
    }
};
template<>
struct gpujpeg_preprocessor_comp_to_raw_load<1, 1>
{
    static __device__ void
    perform(uint8_t & value, int position_x, int position_y, struct gpujpeg_preprocessor_data_component & comp)
    {
        int data_position = position_y * comp.data_width + position_x;
        value = comp.d_data[data_position];
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
    enum gpujpeg_color_space color_space_internal,
    enum gpujpeg_color_space color_space,
    uint8_t s_comp1_samp_factor_h, uint8_t s_comp1_samp_factor_v,
    uint8_t s_comp2_samp_factor_h, uint8_t s_comp2_samp_factor_v,
    uint8_t s_comp3_samp_factor_h, uint8_t s_comp3_samp_factor_v,
    int sample_width,
    uint8_t fill
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
    uint8_t r1;
    uint8_t r2;
    uint8_t r3;
    gpujpeg_preprocessor_comp_to_raw_load<s_comp1_samp_factor_h, s_comp1_samp_factor_v>::perform(r1, image_position_x, image_position_y, data.comp[0]);
    gpujpeg_preprocessor_comp_to_raw_load<s_comp2_samp_factor_h, s_comp2_samp_factor_v>::perform(r2, image_position_x, image_position_y, data.comp[1]);
    gpujpeg_preprocessor_comp_to_raw_load<s_comp3_samp_factor_h, s_comp3_samp_factor_v>::perform(r3, image_position_x, image_position_y, data.comp[2]);

    // Color transform
    gpujpeg_color_transform<color_space_internal, color_space>::perform(r1, r2, r3);

    // Store Order
    gpujpeg_color_order<color_space>::perform_store(r1, r2, r3);

    // Save
    image_position = image_position * sample_width;
    d_data_raw[image_position + 0] = r1;
    d_data_raw[image_position + 1] = r2;
    d_data_raw[image_position + 2] = r3;
    if (sample_width > 3) {
        d_data_raw[image_position + 3] = fill;
    }
}

/** Specialization [sampling factor is 4:2:2] */
template<
    enum gpujpeg_color_space color_space_internal,
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
    uint8_t r1;
    uint8_t r2;
    uint8_t r3;
    gpujpeg_preprocessor_comp_to_raw_load<s_comp1_samp_factor_h, s_comp1_samp_factor_v>::perform(r1, image_position_x, image_position_y, data.comp[0]);
    gpujpeg_preprocessor_comp_to_raw_load<s_comp2_samp_factor_h, s_comp2_samp_factor_v>::perform(r2, image_position_x, image_position_y, data.comp[1]);
    gpujpeg_preprocessor_comp_to_raw_load<s_comp3_samp_factor_h, s_comp3_samp_factor_v>::perform(r3, image_position_x, image_position_y, data.comp[2]);

    // Color transform
    gpujpeg_color_transform<color_space_internal, color_space>::perform(r1, r2, r3);

    // Store Order
    gpujpeg_color_order<color_space>::perform_store(r1, r2, r3);

    // Save
    image_position = image_position * 2;
    d_data_raw[image_position + 1] = r2;
    if ( (image_position_x % 2) == 0 )
        d_data_raw[image_position + 0] = r1;
    else
        d_data_raw[image_position + 0] = r3;
}

/**
 * Select preprocessor decode kernel
 *
 * @param decoder
 * @return kernel
 */
template<enum gpujpeg_color_space color_space_internal>
gpujpeg_preprocessor_decode_kernel
gpujpeg_preprocessor_select_decode_kernel(struct gpujpeg_coder* coder)
{
    gpujpeg_preprocessor_sampling_factor_t sampling_factor = gpujpeg_preprocessor_make_sampling_factor(
        coder->sampling_factor.horizontal / coder->component[0].sampling_factor.horizontal,
        coder->sampling_factor.vertical / coder->component[0].sampling_factor.vertical,
        coder->sampling_factor.horizontal / coder->component[1].sampling_factor.horizontal,
        coder->sampling_factor.vertical / coder->component[1].sampling_factor.vertical,
        coder->sampling_factor.horizontal / coder->component[2].sampling_factor.horizontal,
        coder->sampling_factor.vertical / coder->component[2].sampling_factor.vertical
    );

#define RETURN_KERNEL_IF(PIXEL_FORMAT, COLOR, P1, P2, P3, P4, P5, P6) \
    if ( sampling_factor == gpujpeg_preprocessor_make_sampling_factor(P1, P2, P3, P4, P5, P6) ) { \
        int max_h = max(P1, max(P3, P5)); \
        int max_v = max(P2, max(P4, P6)); \
        if ( coder->param.verbose ) { \
            printf("Using faster kernel for postprocessor (precompiled %dx%d, %dx%d, %dx%d).\n", max_h / P1, max_v / P2, max_h / P3, max_v / P4, max_h / P5, max_v / P6); \
        } \
        if ( PIXEL_FORMAT == GPUJPEG_444_U8_P012 ) { \
            return &gpujpeg_preprocessor_comp_to_raw_kernel_4_4_4<color_space_internal, COLOR, P1, P2, P3, P4, P5, P6, 3, 0>; \
        } else if ( PIXEL_FORMAT == GPUJPEG_444_U8_P012A ) { \
            return &gpujpeg_preprocessor_comp_to_raw_kernel_4_4_4<color_space_internal, COLOR, P1, P2, P3, P4, P5, P6, 4, 0xff>; \
        } else if ( PIXEL_FORMAT == GPUJPEG_444_U8_P012Z ) { \
            return &gpujpeg_preprocessor_comp_to_raw_kernel_4_4_4<color_space_internal, COLOR, P1, P2, P3, P4, P5, P6, 4, 0>; \
        } else if ( coder->param_image.pixel_format == GPUJPEG_422_U8_P1020 ) { \
            return &gpujpeg_preprocessor_comp_to_raw_kernel_4_2_2<color_space_internal, COLOR, P1, P2, P3, P4, P5, P6>; \
        } else { \
            assert(false); \
        } \
    }
#define RETURN_KERNEL(PIXEL_FORMAT, COLOR) \
    RETURN_KERNEL_IF(PIXEL_FORMAT, COLOR, 1, 1, 1, 1, 1, 1) \
    else RETURN_KERNEL_IF(PIXEL_FORMAT, COLOR, 1, 1, 2, 2, 2, 2) \
    else RETURN_KERNEL_IF(PIXEL_FORMAT, COLOR, 1, 1, 1, 2, 1, 2) \
    else RETURN_KERNEL_IF(PIXEL_FORMAT, COLOR, 1, 1, 2, 1, 2, 1) \
    else RETURN_KERNEL_IF(PIXEL_FORMAT, COLOR, 1, 1, 4, 4, 4, 4) \
    else { \
        if ( coder->param.verbose ) { \
            printf("Using slower kernel for postprocessor (dynamic %dx%d, %dx%d, %dx%d).\n", coder->component[0].sampling_factor.horizontal, coder->component[0].sampling_factor.vertical, coder->component[1].sampling_factor.horizontal, coder->component[1].sampling_factor.vertical, coder->component[2].sampling_factor.horizontal, coder->component[2].sampling_factor.vertical); \
        } \
        if ( PIXEL_FORMAT == GPUJPEG_444_U8_P012 ) { \
            return &gpujpeg_preprocessor_comp_to_raw_kernel_4_4_4<color_space_internal, COLOR, GPUJPEG_DYNAMIC, GPUJPEG_DYNAMIC, GPUJPEG_DYNAMIC, GPUJPEG_DYNAMIC, GPUJPEG_DYNAMIC, GPUJPEG_DYNAMIC, 3, 0>; \
        } else if ( PIXEL_FORMAT == GPUJPEG_444_U8_P012Z ) { \
            return &gpujpeg_preprocessor_comp_to_raw_kernel_4_4_4<color_space_internal, COLOR, GPUJPEG_DYNAMIC, GPUJPEG_DYNAMIC, GPUJPEG_DYNAMIC, GPUJPEG_DYNAMIC, GPUJPEG_DYNAMIC, GPUJPEG_DYNAMIC, 4, 0>; \
        } else if ( coder->param_image.pixel_format == GPUJPEG_422_U8_P1020 ) { \
            return &gpujpeg_preprocessor_comp_to_raw_kernel_4_2_2<color_space_internal, COLOR, GPUJPEG_DYNAMIC, GPUJPEG_DYNAMIC, GPUJPEG_DYNAMIC, GPUJPEG_DYNAMIC, GPUJPEG_DYNAMIC, GPUJPEG_DYNAMIC>; \
        } else { \
            assert(false); \
        } \
    } \

    // None color space
    if ( coder->param_image.color_space == GPUJPEG_NONE ) {
        RETURN_KERNEL(coder->param_image.pixel_format, GPUJPEG_NONE)
    }
    // RGB color space
    else if ( coder->param_image.color_space == GPUJPEG_RGB ) {
        RETURN_KERNEL(coder->param_image.pixel_format, GPUJPEG_RGB)
    }
    // YCbCr color space
    else if ( coder->param_image.color_space == GPUJPEG_YCBCR_BT601 ) {
        RETURN_KERNEL(coder->param_image.pixel_format, GPUJPEG_YCBCR_BT601)
    }
    // YCbCr color space
    else if ( coder->param_image.color_space == GPUJPEG_YCBCR_BT601_256LVLS ) {
        RETURN_KERNEL(coder->param_image.pixel_format, GPUJPEG_YCBCR_BT601_256LVLS)
    }
    // YCbCr color space
    else if ( coder->param_image.color_space == GPUJPEG_YCBCR_BT709 ) {
        RETURN_KERNEL(coder->param_image.pixel_format, GPUJPEG_YCBCR_BT709)
    }
    // YUV color space
    else if ( coder->param_image.color_space == GPUJPEG_YUV ) {
        RETURN_KERNEL(coder->param_image.pixel_format, GPUJPEG_YUV)
    }
    // Unknown color space
    else {
        assert(false);
    }

#undef RETURN_KERNEL_IF
#undef RETURN_KERNEL

    return NULL;
}

/* Documented at declaration */
int
gpujpeg_preprocessor_decoder_init(struct gpujpeg_coder* coder)
{
    if (!gpujpeg_pixel_format_is_interleaved(coder->param_image.pixel_format)) {
        return 0;
    }

    assert(coder->param_image.comp_count == 3);

    if (coder->param.color_space_internal == GPUJPEG_NONE) {
        coder->preprocessor = (void*)gpujpeg_preprocessor_select_decode_kernel<GPUJPEG_NONE>(coder);
    }
    else if (coder->param.color_space_internal == GPUJPEG_RGB) {
        coder->preprocessor = (void*)gpujpeg_preprocessor_select_decode_kernel<GPUJPEG_RGB>(coder);
    }
    else if (coder->param.color_space_internal == GPUJPEG_YCBCR_BT601_256LVLS) {
        coder->preprocessor = (void*)gpujpeg_preprocessor_select_decode_kernel<GPUJPEG_YCBCR_BT601_256LVLS>(coder);
    }
    else {
        assert(false);
    }
    if (coder->preprocessor == NULL) {
        return -1;
    }
    return 0;
}

/**
 * Copies raw data GPU memory without running any postprocessor kernel.
 *
 * This assumes that the JPEG has same color space as input raw image and
 * currently also that the component subsampling correspond between raw and
 * JPEG (although at least different horizontal subsampling can be quite
 * easily done).
 *
 * @todo
 * Use postprocessing kernels as for packed formats.
 */
static int
gpujpeg_preprocessor_decoder_copy_planar_data(struct gpujpeg_coder * coder, cudaStream_t stream)
{
    assert(coder->param_image.comp_count == 1 ||
            coder->param_image.comp_count == 3);
    if (coder->param_image.comp_count == 3 && coder->param_image.color_space != coder->param.color_space_internal) {
            fprintf(stderr, "Decoding JPEG to a planar pixel format is supported only when no color transformation is required. "
                            "JPEG internal color space is set to \"%s\", image is \"%s\".\n",
                            gpujpeg_color_space_get_name(coder->param.color_space_internal),
                            gpujpeg_color_space_get_name(coder->param_image.color_space));
            return -1;
    }

    const int *sampling_factors = gpujpeg_pixel_format_get_sampling_factor(coder->param_image.pixel_format);
    for (int i = 0; i < coder->param_image.comp_count; ++i) {
        if (coder->component[i].sampling_factor.horizontal != sampling_factors[i * 2]
                || coder->component[0].sampling_factor.vertical != sampling_factors[i * 2 + 1]) {
            const char *name = gpujpeg_pixel_format_get_name(coder->param_image.pixel_format);
            fprintf(stderr, "Decoding JPEG to a planar pixel format cannot change subsampling (%s to %s).\n",
                    gpujpeg_subsampling_get_name(coder->param_image.comp_count, coder->component),
                    gpujpeg_pixel_format_get_name(coder->param_image.pixel_format));
            return GPUJPEG_ERR_WRONG_SUBSAMPLING;
        }
    }

    size_t data_raw_offset = 0;
    bool needs_stride = false; // true if width is not divisible by MCU width
    for (int i = 0; i < coder->param_image.comp_count; ++i) {
        needs_stride = needs_stride || coder->component[i].width != coder->component[i].data_width;
    }
    if (!needs_stride) {
            for (int i = 0; i < coder->param_image.comp_count; ++i) {
                    size_t component_size = coder->component[i].width * coder->component[i].height;
                    cudaMemcpyAsync(coder->d_data_raw + data_raw_offset, coder->component[i].d_data, component_size, cudaMemcpyDeviceToDevice, stream);
                    data_raw_offset += component_size;
            }
    } else {
            for (int i = 0; i < coder->param_image.comp_count; ++i) {
                    int spitch = coder->component[i].data_width;
                    int dpitch = coder->component[i].width;
                    size_t component_size = spitch * coder->component[i].height;
                    cudaMemcpy2DAsync(coder->d_data_raw + data_raw_offset, dpitch, coder->component[i].d_data, spitch, coder->component[i].width, coder->component[i].height, cudaMemcpyDeviceToDevice, stream);
                    data_raw_offset += component_size;
            }
    }
    gpujpeg_cuda_check_error("Preprocessor copy failed", return -1);
    return 0;
}

/* Documented at declaration */
int
gpujpeg_preprocessor_decode(struct gpujpeg_coder* coder, cudaStream_t stream)
{
    if (!gpujpeg_pixel_format_is_interleaved(coder->param_image.pixel_format)) {
        return gpujpeg_preprocessor_decoder_copy_planar_data(coder, stream);
    }

    assert(coder->param_image.comp_count == 3);

    cudaMemsetAsync(coder->d_data_raw, 0, coder->data_raw_size * sizeof(uint8_t), stream);

    // Select kernel
    gpujpeg_preprocessor_decode_kernel kernel = (gpujpeg_preprocessor_decode_kernel)coder->preprocessor;
    assert(kernel != NULL);

    int image_width = coder->param_image.width;
    int image_height = coder->param_image.height;

    // When saving 4:2:2 data of odd width, the data should have even width, so round it
    if (coder->param_image.pixel_format == GPUJPEG_422_U8_P1020) {
        image_width = gpujpeg_div_and_round_up(coder->param_image.width, 2) * 2;
    }

    // Prepare unit size
    int unitSize = gpujpeg_pixel_format_get_unit_size(coder->param_image.pixel_format);

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
    kernel<<<grid, threads, 0, stream>>>(
        data,
        coder->d_data_raw,
        image_width,
        image_height
    );
    gpujpeg_cuda_check_error("Preprocessor encoding failed", return -1);

    return 0;
}

/* vi: set expandtab sw=4: */
