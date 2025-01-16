/*
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
/**
 * @file
 * @brief
 * This file contains postprocessors a common format for computational kernels
 * to raw image. It also does color space transformations.
 */

#include "gpujpeg_colorspace.h"
#include "gpujpeg_preprocessor_common.cuh"
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
struct gpujpeg_preprocessor_comp_to_raw_load_comp
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
struct gpujpeg_preprocessor_comp_to_raw_load_comp<1, 1>
{
    static __device__ void
    perform(uint8_t & value, int position_x, int position_y, struct gpujpeg_preprocessor_data_component & comp)
    {
        int data_position = position_y * comp.data_width + position_x;
        value = comp.d_data[data_position];
    }
};

template <>
struct gpujpeg_preprocessor_comp_to_raw_load_comp<0, 0>
{
    static __device__ void
    perform(uint8_t& value, int position_x, int position_y, struct gpujpeg_preprocessor_data_component& comp)
    {
    }
};

template<
    uint8_t s_comp1_samp_factor_h, uint8_t s_comp1_samp_factor_v,
    uint8_t s_comp2_samp_factor_h, uint8_t s_comp2_samp_factor_v,
    uint8_t s_comp3_samp_factor_h, uint8_t s_comp3_samp_factor_v,
    uint8_t s_comp4_samp_factor_h, uint8_t s_comp4_samp_factor_v
>
struct gpujpeg_preprocessor_comp_to_raw_load
{
    static __device__ void perform(uchar4 & value, int position_x, int position_y, struct gpujpeg_preprocessor_data & data) {
        gpujpeg_preprocessor_comp_to_raw_load_comp<s_comp1_samp_factor_h, s_comp1_samp_factor_v>::perform(value.x, position_x, position_y, data.comp[0]);
        gpujpeg_preprocessor_comp_to_raw_load_comp<s_comp2_samp_factor_h, s_comp2_samp_factor_v>::perform(value.y, position_x, position_y, data.comp[1]);
        gpujpeg_preprocessor_comp_to_raw_load_comp<s_comp3_samp_factor_h, s_comp3_samp_factor_v>::perform(value.z, position_x, position_y, data.comp[2]);
        gpujpeg_preprocessor_comp_to_raw_load_comp<s_comp4_samp_factor_h, s_comp4_samp_factor_v>::perform(value.w, position_x, position_y, data.comp[3]);
    }
};

template<enum gpujpeg_pixel_format pixel_format>
inline __device__ void gpujpeg_comp_to_raw_store(uint8_t *d_data_raw, int &image_width, int &image_height, int &image_position, int &x, int &y, uchar4 &r);

template<>
inline __device__ void gpujpeg_comp_to_raw_store<GPUJPEG_U8>(uint8_t *d_data_raw, int &image_width, int &image_height, int &image_position, int &x, int &y, uchar4 &r)
{
    d_data_raw[image_position] = r.x;
}

template<>
inline __device__ void gpujpeg_comp_to_raw_store<GPUJPEG_444_U8_P012>(uint8_t *d_data_raw, int &image_width, int &image_height, int &image_position, int &x, int &y, uchar4 &r)
{
    image_position = image_position * 3;
    d_data_raw[image_position + 0] = r.x;
    d_data_raw[image_position + 1] = r.y;
    d_data_raw[image_position + 2] = r.z;
}

template<>
inline __device__ void gpujpeg_comp_to_raw_store<GPUJPEG_4444_U8_P0123>(uint8_t *d_data_raw, int &image_width, int &image_height, int &image_position, int &x, int &y, uchar4 &r)
{
    image_position = image_position * 4;
    d_data_raw[image_position + 0] = r.x;
    d_data_raw[image_position + 1] = r.y;
    d_data_raw[image_position + 2] = r.z;
    d_data_raw[image_position + 3] = r.w;
}

template<>
inline __device__ void gpujpeg_comp_to_raw_store<GPUJPEG_444_U8_P0P1P2>(uint8_t *d_data_raw, int &image_width, int &image_height, int &image_position, int &x, int &y, uchar4 &r)
{
    d_data_raw[image_position] = r.x;
    d_data_raw[image_width * image_height + image_position] = r.y;
    d_data_raw[2 * image_width * image_height + image_position] = r.z;
}

template<>
inline __device__ void gpujpeg_comp_to_raw_store<GPUJPEG_422_U8_P0P1P2>(uint8_t *d_data_raw, int &image_width, int &image_height, int &image_position, int &x, int &y, uchar4 &r)
{
    d_data_raw[image_position] = r.x;
    if ( (x % 2) == 0 ) {
        d_data_raw[image_width * image_height + image_position / 2] = r.y;
        d_data_raw[image_width * image_height + image_height * ((image_width + 1) / 2) + image_position / 2] = r.z;
    }
}

template<>
inline __device__ void gpujpeg_comp_to_raw_store<GPUJPEG_422_U8_P1020>(uint8_t *d_data_raw, int &image_width, int &image_height, int &image_position, int &x, int &y, uchar4 &r)
{
    image_position = image_position * 2;
    d_data_raw[image_position + 1] = r.x;
    if ( (x % 2) == 0 )
        d_data_raw[image_position + 0] = r.y;
    else
        d_data_raw[image_position + 0] = r.z;
}

template<>
inline __device__ void gpujpeg_comp_to_raw_store<GPUJPEG_420_U8_P0P1P2>(uint8_t *d_data_raw, int &image_width, int &image_height, int &image_position, int &x, int &y, uchar4 &r)
{
    d_data_raw[image_position] = r.x;
    if ( (image_position % 2) == 0 && (y % 2) == 0 ) {
        d_data_raw[image_width * image_height + y / 2 * ((image_width + 1) / 2) + x / 2] = r.y;
        d_data_raw[image_width * image_height + ((image_height + 1) / 2 + y / 2) * ((image_width + 1) / 2) + x / 2] = r.z;
    }
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
    int comp_count,
    uint8_t s_comp1_samp_factor_h, uint8_t s_comp1_samp_factor_v,
    uint8_t s_comp2_samp_factor_h, uint8_t s_comp2_samp_factor_v,
    uint8_t s_comp3_samp_factor_h, uint8_t s_comp3_samp_factor_v,
    uint8_t s_comp4_samp_factor_h, uint8_t s_comp4_samp_factor_v
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
    uchar4 r;
    gpujpeg_preprocessor_comp_to_raw_load<s_comp1_samp_factor_h, s_comp1_samp_factor_v, s_comp2_samp_factor_h, s_comp2_samp_factor_v, s_comp3_samp_factor_h, s_comp3_samp_factor_v, s_comp4_samp_factor_h, s_comp4_samp_factor_v>::perform(r, image_position_x, image_position_y, data);

    // Color transform
    gpujpeg_color_transform<color_space_internal, color_space>::perform(r);

    // Save
    if (pixel_format == GPUJPEG_4444_U8_P0123 && comp_count == 3) {
        r.w = 0xFF;
    }
    gpujpeg_comp_to_raw_store<pixel_format>(d_data_raw, image_width, image_height, image_position, image_position_x, image_position_y, r);
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
    gpujpeg_sampling_factor_t sampling_factor = gpujpeg_preprocessor_make_sampling_factor_i(
        coder->param.comp_count, coder->sampling_factor.horizontal, coder->sampling_factor.vertical,
        coder->component[0].sampling_factor.horizontal, coder->component[0].sampling_factor.vertical,
        coder->component[1].sampling_factor.horizontal, coder->component[1].sampling_factor.vertical,
        coder->component[2].sampling_factor.horizontal, coder->component[2].sampling_factor.vertical,
        coder->component[3].sampling_factor.horizontal, coder->component[3].sampling_factor.vertical);

#define RETURN_KERNEL_SWITCH(PIXEL_FORMAT, COLOR, P1, P2, P3, P4, P5, P6, P7, P8) \
        switch ( PIXEL_FORMAT ) { \
            case GPUJPEG_U8: return &gpujpeg_preprocessor_comp_to_raw_kernel<color_space_internal, COLOR, GPUJPEG_U8, 3, P1, P2, P3, P4, P5, P6, P7, P8>; \
            case GPUJPEG_444_U8_P012: return &gpujpeg_preprocessor_comp_to_raw_kernel<color_space_internal, COLOR, GPUJPEG_444_U8_P012, 3, P1, P2, P3, P4, P5, P6, P7, P8>; \
            case GPUJPEG_4444_U8_P0123: return &gpujpeg_preprocessor_comp_to_raw_kernel<color_space_internal, COLOR, GPUJPEG_4444_U8_P0123, 3, P1, P2, P3, P4, P5, P6, P7, P8>;\
            case GPUJPEG_422_U8_P1020: return &gpujpeg_preprocessor_comp_to_raw_kernel<color_space_internal, COLOR, GPUJPEG_422_U8_P1020, 3, P1, P2, P3, P4, P5, P6, P7, P8>; \
            case GPUJPEG_444_U8_P0P1P2: return &gpujpeg_preprocessor_comp_to_raw_kernel<color_space_internal, COLOR, GPUJPEG_444_U8_P0P1P2, 3, P1, P2, P3, P4, P5, P6, P7, P8>; \
            case GPUJPEG_422_U8_P0P1P2: return &gpujpeg_preprocessor_comp_to_raw_kernel<color_space_internal, COLOR, GPUJPEG_422_U8_P0P1P2, 3, P1, P2, P3, P4, P5, P6, P7, P8>; \
            case GPUJPEG_420_U8_P0P1P2: return &gpujpeg_preprocessor_comp_to_raw_kernel<color_space_internal, COLOR, GPUJPEG_420_U8_P0P1P2, 3, P1, P2, P3, P4, P5, P6, P7, P8>; \
            case GPUJPEG_PIXFMT_NONE: GPUJPEG_ASSERT(0 && "Postprocess to GPUJPEG_PIXFMT_NONE not allowed"); \
        } \

#define RETURN_KERNEL_IF(PIXEL_FORMAT, COLOR, COMP_COUNT, P1, P2, P3, P4, P5, P6, P7, P8) \
    if ( sampling_factor == gpujpeg_make_sampling_factor(COMP_COUNT, P1, P2, P3, P4, P5, P6, P7, P8) ) { \
        if ( coder->param.verbose >= GPUJPEG_LL_VERBOSE ) {                                                            \
            print_kernel_configuration(                                                                                \
                "Using faster kernel for postprocessor (precompiled %dx%d, %dx%d, %dx%d, %dx%d).\n");                  \
        } \
        RETURN_KERNEL_SWITCH(PIXEL_FORMAT, COLOR, P1, P2, P3, P4, P5, P6, P7, P8) \
    }

#define RETURN_KERNEL(PIXEL_FORMAT, COLOR) \
    RETURN_KERNEL_IF(PIXEL_FORMAT, COLOR, 3, 1, 1, 1, 1, 1, 1, 0, 0) /* 4:4:4 */ \
    else RETURN_KERNEL_IF(PIXEL_FORMAT, COLOR, 3, 1, 1, 2, 2, 2, 2, 0, 0) /* 4:2:0 */ \
    else RETURN_KERNEL_IF(PIXEL_FORMAT, COLOR, 3, 1, 1, 1, 2, 1, 2, 0, 0) /* 4:4:0 */ \
    else RETURN_KERNEL_IF(PIXEL_FORMAT, COLOR, 3, 1, 1, 2, 1, 2, 1, 0, 0) /* 4:2:2 */ \
    else { \
        if ( coder->param.verbose >= GPUJPEG_LL_INFO ) {                                                               \
            print_kernel_configuration(                                                                                \
                "Using slower kernel for postprocessor (dynamic %dx%d, %dx%d, %dx%d, %dx%d).\n");                      \
        } \
        RETURN_KERNEL_SWITCH(PIXEL_FORMAT, COLOR, GPUJPEG_DYNAMIC, GPUJPEG_DYNAMIC, GPUJPEG_DYNAMIC, GPUJPEG_DYNAMIC, GPUJPEG_DYNAMIC, GPUJPEG_DYNAMIC, GPUJPEG_DYNAMIC, GPUJPEG_DYNAMIC) \
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
#ifndef ENABLE_YUV
    // YUV color space
    else if ( coder->param_image.color_space == GPUJPEG_YUV ) {
        RETURN_KERNEL(coder->param_image.pixel_format, GPUJPEG_YUV)
    }
#endif
    // Unknown color space
    else {
        assert(false);
    }

#undef RETURN_KERNEL_SWITCH
#undef RETURN_KERNEL_IF
#undef RETURN_KERNEL

    return NULL;
}

static int gpujpeg_preprocessor_decode_no_transform(struct gpujpeg_coder * coder)
{
    if ( coder->param.comp_count == 3 && coder->param_image.color_space != coder->param.color_space_internal ) {
            /*fprintf(stderr, "Decoding JPEG to a planar pixel format is supported only when no color transformation is required. "
                            "JPEG internal color space is set to \"%s\", image is \"%s\".\n",
                            gpujpeg_color_space_get_name(coder->param.color_space_internal),
                            gpujpeg_color_space_get_name(coder->param_image.color_space));*/
            return 0;
    }

    const struct gpujpeg_component_sampling_factor* sampling_factors =
        gpujpeg_pixel_format_get_sampling_factor(coder->param_image.pixel_format);
    for ( int i = 0; i < coder->param.comp_count; ++i ) {
        if ( coder->component[i].sampling_factor.horizontal != sampling_factors[i].horizontal ||
             coder->component[i].sampling_factor.vertical != sampling_factors[i].vertical ) {
            // const char *name = gpujpeg_pixel_format_get_name(coder->param_image.pixel_format);
            /*fprintf(stderr, "Decoding JPEG to a planar pixel format cannot change subsampling (%s to %s).\n",
                    gpujpeg_subsampling_get_name(coder->param.comp_count, coder->component),
                    gpujpeg_pixel_format_get_name(coder->param_image.pixel_format));*/
            return 0;
        }
    }
    return 1;
}

static int gpujpeg_preprocessor_decode_aligned(struct gpujpeg_coder * coder)
{
    for ( int i = 0; i < coder->param.comp_count; ++i ) {
        if (coder->component[i].data_width !=  coder->component[i].width) {
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
            gpujpeg_preprocessor_decode_no_transform(coder) &&
            gpujpeg_preprocessor_decode_aligned(coder)) {
        if ( coder->param.verbose >= GPUJPEG_LL_DEBUG ) {
            printf("Matching format detected - not using postprocessor, using memcpy instead.\n");
        }
        return 0;
    }

    if ( coder->param.comp_count == 1 && gpujpeg_pixel_format_get_comp_count(coder->param_image.pixel_format) > 1 ) {
        ERROR_MSG("Decoding single component JPEG allowed only to single component output format!\n");
        return -1;
    }

    // assert(coder->param.comp_count == 3 || coder->param.comp_count == 4);

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
    assert(coder->param.comp_count == 1 || coder->param.comp_count == 3);
    size_t data_raw_offset = 0;
    bool needs_stride = false; // true if width is not divisible by MCU width
    for ( int i = 0; i < coder->param.comp_count; ++i ) {
        needs_stride = needs_stride || coder->component[i].width != coder->component[i].data_width;
    }
    if (!needs_stride) {
            for ( int i = 0; i < coder->param.comp_count; ++i ) {
                    size_t component_size = coder->component[i].width * coder->component[i].height;
                    cudaMemcpyAsync(coder->d_data_raw + data_raw_offset, coder->component[i].d_data, component_size, cudaMemcpyDeviceToDevice, stream);
                    data_raw_offset += component_size;
            }
    } else {
            for ( int i = 0; i < coder->param.comp_count; ++i ) {
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
    for ( int comp = 0; comp < coder->param.comp_count; comp++ ) {
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
