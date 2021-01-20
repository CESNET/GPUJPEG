/*
 * Copyright (c) 2011-2021, CESNET z.s.p.o
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
 * This file contains postprocessors a common format for computational kernels
 * to raw image. It also does color space transformations.
 */

#include "gpujpeg_colorspace.h"
#include "gpujpeg_preprocessor_common.h"
#include "gpujpeg_postprocessor.h"
#include "gpujpeg_util.h"

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

template<enum gpujpeg_pixel_format pixel_format>
inline __device__ void gpujpeg_comp_to_raw_store(uint8_t *d_data_raw, int &image_width, int &image_height, int &image_position, uint8_t &r1, uint8_t &r2, uint8_t &r3);

template<>
inline __device__ void gpujpeg_comp_to_raw_store<GPUJPEG_U8>(uint8_t *d_data_raw, int &image_width, int &image_height, int &image_position, uint8_t &r1, uint8_t &r2, uint8_t &r3)
{
    d_data_raw[image_position] = r1;
}

template<>
inline __device__ void gpujpeg_comp_to_raw_store<GPUJPEG_444_U8_P012>(uint8_t *d_data_raw, int &image_width, int &image_height, int &image_position, uint8_t &r1, uint8_t &r2, uint8_t &r3)
{
    image_position = image_position * 3;
    d_data_raw[image_position + 0] = r1;
    d_data_raw[image_position + 1] = r2;
    d_data_raw[image_position + 2] = r3;
}

template<>
inline __device__ void gpujpeg_comp_to_raw_store<GPUJPEG_444_U8_P012A>(uint8_t *d_data_raw, int &image_width, int &image_height, int &image_position, uint8_t &r1, uint8_t &r2, uint8_t &r3)
{
    image_position = image_position * 4;
    d_data_raw[image_position + 0] = r1;
    d_data_raw[image_position + 1] = r2;
    d_data_raw[image_position + 2] = r3;
    d_data_raw[image_position + 3] = 0xFF;
}

template<>
inline __device__ void gpujpeg_comp_to_raw_store<GPUJPEG_444_U8_P012Z>(uint8_t *d_data_raw, int &image_width, int &image_height, int &image_position, uint8_t &r1, uint8_t &r2, uint8_t &r3)
{
    image_position = image_position * 4;
    d_data_raw[image_position + 0] = r1;
    d_data_raw[image_position + 1] = r2;
    d_data_raw[image_position + 2] = r3;
    d_data_raw[image_position + 3] = 0x0;
}

template<>
inline __device__ void gpujpeg_comp_to_raw_store<GPUJPEG_444_U8_P0P1P2>(uint8_t *d_data_raw, int &image_width, int &image_height, int &image_position, uint8_t &r1, uint8_t &r2, uint8_t &r3)
{
    d_data_raw[image_position] = r1;
    d_data_raw[image_width * image_height + image_position] = r2;
    d_data_raw[2 * image_width * image_height + image_position] = r3;
}

template<>
inline __device__ void gpujpeg_comp_to_raw_store<GPUJPEG_422_U8_P0P1P2>(uint8_t *d_data_raw, int &image_width, int &image_height, int &image_position, uint8_t &r1, uint8_t &r2, uint8_t &r3)
{
    d_data_raw[image_position] = r1;
    if ( (image_position % 2) == 0 ) {
        d_data_raw[image_width * image_height + image_position / 2] = r2;
        d_data_raw[image_width * image_height + image_height * ((image_width + 1) / 2) + image_position / 2] = r3;
    }
}

template<>
inline __device__ void gpujpeg_comp_to_raw_store<GPUJPEG_422_U8_P1020>(uint8_t *d_data_raw, int &image_width, int &image_height, int &image_position, uint8_t &r1, uint8_t &r2, uint8_t &r3)
{
    image_position = image_position * 2;
    d_data_raw[image_position + 1] = r1;
    if ( (image_position % 2) == 0 )
        d_data_raw[image_position + 0] = r2;
    else
        d_data_raw[image_position + 0] = r3;
}

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

template<
    enum gpujpeg_color_space color_space_internal,
    enum gpujpeg_color_space color_space,
    enum gpujpeg_pixel_format pixel_format,
    uint8_t s_comp1_samp_factor_h, uint8_t s_comp1_samp_factor_v,
    uint8_t s_comp2_samp_factor_h, uint8_t s_comp2_samp_factor_v,
    uint8_t s_comp3_samp_factor_h, uint8_t s_comp3_samp_factor_v
>
__global__ void
gpujpeg_preprocessor_comp_to_raw_kernel(struct gpujpeg_preprocessor_data data, uint8_t* d_data_raw, int image_width, int image_height)
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

    // Save
    gpujpeg_comp_to_raw_store<pixel_format>(d_data_raw, image_width, image_height, image_position, r1, r2, r3);

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
        if ( coder->param.verbose >= 1 ) { \
            printf("Using faster kernel for postprocessor (precompiled %dx%d, %dx%d, %dx%d).\n", max_h / P1, max_v / P2, max_h / P3, max_v / P4, max_h / P5, max_v / P6); \
        } \
        if ( PIXEL_FORMAT == GPUJPEG_U8 ) { \
            return &gpujpeg_preprocessor_comp_to_raw_kernel<color_space_internal, COLOR, GPUJPEG_U8, P1, P2, P3, P4, P5, P6>; \
        } else if ( PIXEL_FORMAT == GPUJPEG_444_U8_P012 ) { \
            return &gpujpeg_preprocessor_comp_to_raw_kernel<color_space_internal, COLOR, GPUJPEG_444_U8_P012, P1, P2, P3, P4, P5, P6>; \
        } else if ( PIXEL_FORMAT == GPUJPEG_444_U8_P012A ) { \
            return &gpujpeg_preprocessor_comp_to_raw_kernel<color_space_internal, COLOR, GPUJPEG_444_U8_P012A, P1, P2, P3, P4, P5, P6>; \
        } else if ( PIXEL_FORMAT == GPUJPEG_444_U8_P012Z ) { \
            return &gpujpeg_preprocessor_comp_to_raw_kernel<color_space_internal, COLOR, GPUJPEG_444_U8_P012Z, P1, P2, P3, P4, P5, P6>; \
        } else if ( coder->param_image.pixel_format == GPUJPEG_422_U8_P1020 ) { \
            return &gpujpeg_preprocessor_comp_to_raw_kernel<color_space_internal, COLOR, GPUJPEG_422_U8_P1020, P1, P2, P3, P4, P5, P6>; \
        } else if ( coder->param_image.pixel_format == GPUJPEG_444_U8_P0P1P2 ) { \
            return &gpujpeg_preprocessor_comp_to_raw_kernel<color_space_internal, COLOR, GPUJPEG_444_U8_P0P1P2, P1, P2, P3, P4, P5, P6>; \
        } else if ( coder->param_image.pixel_format == GPUJPEG_422_U8_P0P1P2 ) { \
            return &gpujpeg_preprocessor_comp_to_raw_kernel<color_space_internal, COLOR, GPUJPEG_422_U8_P0P1P2, P1, P2, P3, P4, P5, P6>; \
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
        if ( coder->param.verbose >= 1 ) { \
            printf("Using slower kernel for postprocessor (dynamic %dx%d, %dx%d, %dx%d).\n", coder->component[0].sampling_factor.horizontal, coder->component[0].sampling_factor.vertical, coder->component[1].sampling_factor.horizontal, coder->component[1].sampling_factor.vertical, coder->component[2].sampling_factor.horizontal, coder->component[2].sampling_factor.vertical); \
        } \
        if ( PIXEL_FORMAT == GPUJPEG_U8 ) { \
            return &gpujpeg_preprocessor_comp_to_raw_kernel<color_space_internal, COLOR, GPUJPEG_U8, GPUJPEG_DYNAMIC, GPUJPEG_DYNAMIC, GPUJPEG_DYNAMIC, GPUJPEG_DYNAMIC, GPUJPEG_DYNAMIC, GPUJPEG_DYNAMIC>; \
        } else if ( PIXEL_FORMAT == GPUJPEG_444_U8_P012 ) { \
            return &gpujpeg_preprocessor_comp_to_raw_kernel<color_space_internal, COLOR, GPUJPEG_444_U8_P012, GPUJPEG_DYNAMIC, GPUJPEG_DYNAMIC, GPUJPEG_DYNAMIC, GPUJPEG_DYNAMIC, GPUJPEG_DYNAMIC, GPUJPEG_DYNAMIC>; \
        } else if ( PIXEL_FORMAT == GPUJPEG_444_U8_P012Z ) { \
            return &gpujpeg_preprocessor_comp_to_raw_kernel<color_space_internal, COLOR, GPUJPEG_444_U8_P012Z, GPUJPEG_DYNAMIC, GPUJPEG_DYNAMIC, GPUJPEG_DYNAMIC, GPUJPEG_DYNAMIC, GPUJPEG_DYNAMIC, GPUJPEG_DYNAMIC>; \
        } else if ( coder->param_image.pixel_format == GPUJPEG_422_U8_P1020 ) { \
            return &gpujpeg_preprocessor_comp_to_raw_kernel<color_space_internal, COLOR, GPUJPEG_422_U8_P1020, GPUJPEG_DYNAMIC, GPUJPEG_DYNAMIC, GPUJPEG_DYNAMIC, GPUJPEG_DYNAMIC, GPUJPEG_DYNAMIC, GPUJPEG_DYNAMIC>; \
        } else if ( coder->param_image.pixel_format == GPUJPEG_444_U8_P0P1P2 ) { \
            return &gpujpeg_preprocessor_comp_to_raw_kernel<color_space_internal, COLOR, GPUJPEG_444_U8_P0P1P2, GPUJPEG_DYNAMIC, GPUJPEG_DYNAMIC, GPUJPEG_DYNAMIC, GPUJPEG_DYNAMIC, GPUJPEG_DYNAMIC, GPUJPEG_DYNAMIC>; \
        } else if ( coder->param_image.pixel_format == GPUJPEG_422_U8_P0P1P2 ) { \
            return &gpujpeg_preprocessor_comp_to_raw_kernel<color_space_internal, COLOR, GPUJPEG_422_U8_P0P1P2, GPUJPEG_DYNAMIC, GPUJPEG_DYNAMIC, GPUJPEG_DYNAMIC, GPUJPEG_DYNAMIC, GPUJPEG_DYNAMIC, GPUJPEG_DYNAMIC>; \
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

static int gpujpeg_preprocessor_decode_no_transform(struct gpujpeg_coder * coder)
{
    if (coder->param_image.comp_count == 3 && coder->param_image.color_space != coder->param.color_space_internal) {
            /*fprintf(stderr, "Decoding JPEG to a planar pixel format is supported only when no color transformation is required. "
                            "JPEG internal color space is set to \"%s\", image is \"%s\".\n",
                            gpujpeg_color_space_get_name(coder->param.color_space_internal),
                            gpujpeg_color_space_get_name(coder->param_image.color_space));*/
            return 0;
    }

    const int *sampling_factors = gpujpeg_pixel_format_get_sampling_factor(coder->param_image.pixel_format);
    for (int i = 0; i < coder->param_image.comp_count; ++i) {
        if (coder->component[i].sampling_factor.horizontal != sampling_factors[i * 2]
                || coder->component[i].sampling_factor.vertical != sampling_factors[i * 2 + 1]) {
            const char *name = gpujpeg_pixel_format_get_name(coder->param_image.pixel_format);
            /*fprintf(stderr, "Decoding JPEG to a planar pixel format cannot change subsampling (%s to %s).\n",
                    gpujpeg_subsampling_get_name(coder->param_image.comp_count, coder->component),
                    gpujpeg_pixel_format_get_name(coder->param_image.pixel_format));*/
            return 0;
        }
    }
    return 1;
}

/* Documented at declaration */
int
gpujpeg_preprocessor_decoder_init(struct gpujpeg_coder* coder)
{
    coder->preprocessor = NULL;

    if (!gpujpeg_pixel_format_is_interleaved(coder->param_image.pixel_format) &&
            gpujpeg_preprocessor_decode_no_transform(coder)) {
        if ( coder->param.verbose >= 1 ) {
            printf("Matching format detected - not using postprocessor, using memcpy instead.");
        }
        return 0;
    }

    assert(coder->param_image.comp_count == 3);

    if (coder->param.color_space_internal == GPUJPEG_NONE) {
        coder->preprocessor = (void*)gpujpeg_preprocessor_select_decode_kernel<GPUJPEG_NONE>(coder);
    }
    else if (coder->param.color_space_internal == GPUJPEG_RGB) {
        coder->preprocessor = (void*)gpujpeg_preprocessor_select_decode_kernel<GPUJPEG_RGB>(coder);
    }
    else if (coder->param.color_space_internal == GPUJPEG_YCBCR_BT601) {
        coder->preprocessor = (void*)gpujpeg_preprocessor_select_decode_kernel<GPUJPEG_YCBCR_BT601>(coder);
    }
    else if (coder->param.color_space_internal == GPUJPEG_YCBCR_BT601_256LVLS) {
        coder->preprocessor = (void*)gpujpeg_preprocessor_select_decode_kernel<GPUJPEG_YCBCR_BT601_256LVLS>(coder);
    }
    else if (coder->param.color_space_internal == GPUJPEG_YCBCR_BT709) {
        coder->preprocessor = (void*)gpujpeg_preprocessor_select_decode_kernel<GPUJPEG_YCBCR_BT709>(coder);
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
 * @invariant gpujpeg_preprocessor_decode_no_transform(coder) != 0
 */
static int
gpujpeg_preprocessor_decoder_copy_planar_data(struct gpujpeg_coder * coder, cudaStream_t stream)
{
    assert(coder->param_image.comp_count == 1 ||
            coder->param_image.comp_count == 3);
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
    if (!coder->preprocessor) {
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
    /// @todo this stuff doesn't look correct - we multiply by unitSize and then divide by it
    int unitSize = gpujpeg_pixel_format_get_unit_size(coder->param_image.pixel_format);
    if (unitSize == 0) {
        unitSize = 1;
    }

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
