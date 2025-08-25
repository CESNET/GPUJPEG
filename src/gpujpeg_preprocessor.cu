/*
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
/**
 * @file
 * @brief
 * This file contains preprocessors from raw image to a common format for
 * computational kernels. It also does color space transformations.
 */

#include "gpujpeg_preprocessor.h"

#include "gpujpeg_colorspace.h"
#include "gpujpeg_encoder_internal.h"
#include "gpujpeg_preprocessor_common.cuh"
#include "gpujpeg_util.h"

/**
 * Store value to component data buffer in specified position by buffer size and subsampling
 */
template<
    unsigned int s_samp_factor_h,
    unsigned int s_samp_factor_v
>
static __device__ void
gpujpeg_preprocessor_raw_to_comp_store_comp(uint8_t value, unsigned int position_x, unsigned int position_y, struct gpujpeg_preprocessor_data_component & comp)
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

template <>
__device__ void
gpujpeg_preprocessor_raw_to_comp_store_comp<0, 0>(uint8_t value, unsigned int position_x, unsigned int position_y, struct gpujpeg_preprocessor_data_component& comp)
{
}

template<
    uint8_t s_comp1_samp_factor_h, uint8_t s_comp1_samp_factor_v,
    uint8_t s_comp2_samp_factor_h, uint8_t s_comp2_samp_factor_v,
    uint8_t s_comp3_samp_factor_h, uint8_t s_comp3_samp_factor_v,
    uint8_t s_comp4_samp_factor_h, uint8_t s_comp4_samp_factor_v
>
struct gpujpeg_preprocessor_raw_to_comp_store {
    static __device__ void perform (uchar4 value, unsigned int position_x, unsigned int position_y, struct gpujpeg_preprocessor_data & data) {
        gpujpeg_preprocessor_raw_to_comp_store_comp<s_comp1_samp_factor_h, s_comp1_samp_factor_v>(value.x, position_x, position_y, data.comp[0]);
        gpujpeg_preprocessor_raw_to_comp_store_comp<s_comp2_samp_factor_h, s_comp2_samp_factor_v>(value.y, position_x, position_y, data.comp[1]);
        gpujpeg_preprocessor_raw_to_comp_store_comp<s_comp3_samp_factor_h, s_comp3_samp_factor_v>(value.z, position_x, position_y, data.comp[2]);
        gpujpeg_preprocessor_raw_to_comp_store_comp<s_comp4_samp_factor_h, s_comp4_samp_factor_v>(value.w, position_x, position_y, data.comp[3]);
    }
};

template<enum gpujpeg_pixel_format>
inline __device__ void raw_to_comp_load(const uint8_t* d_data_raw, int &image_width, int &image_height, int &image_position, int &x, int &y, uchar4 &r);

template<>
inline __device__ void raw_to_comp_load<GPUJPEG_U8>(const uint8_t* d_data_raw, int &image_width, int &image_height, int &image_position, int &x, int &y, uchar4 &r)
{
    r.x = d_data_raw[image_position];
    r.y = 128;
    r.z = 128;
}

template<>
inline __device__ void raw_to_comp_load<GPUJPEG_444_U8_P0P1P2>(const uint8_t* d_data_raw, int &image_width, int &image_height, int &image_position, int &x, int &y, uchar4 &r)
{
    r.x = d_data_raw[image_position];
    r.y = d_data_raw[image_width * image_height + image_position];
    r.z = d_data_raw[2 * image_width * image_height + image_position];
}

template<>
inline __device__ void raw_to_comp_load<GPUJPEG_422_U8_P0P1P2>(const uint8_t* d_data_raw, int &image_width, int &image_height, int &image_position, int &x, int &y, uchar4 &r)
{
    r.x = d_data_raw[image_position];
    r.y = d_data_raw[image_width * image_height + image_position / 2];
    r.z = d_data_raw[image_width * image_height + image_height * ((image_width + 1) / 2) + image_position / 2];
}

template<>
inline __device__ void raw_to_comp_load<GPUJPEG_420_U8_P0P1P2>(const uint8_t* d_data_raw, int &image_width, int &image_height, int &image_position, int &x, int &y, uchar4 &r)
{
    r.x = d_data_raw[image_position];
    r.y = d_data_raw[image_width * image_height + y / 2 * ((image_width + 1) / 2) + x / 2];
    r.z = d_data_raw[image_width * image_height + ((image_height + 1) / 2 + y / 2) * ((image_width + 1) / 2) + x / 2];
}

template<>
inline __device__ void raw_to_comp_load<GPUJPEG_444_U8_P012>(const uint8_t* d_data_raw, int &image_width, int &image_height, int &offset, int &x, int &y, uchar4 &r)
{
    r.x = d_data_raw[offset];
    r.y = d_data_raw[offset + 1];
    r.z = d_data_raw[offset + 2];
}

template<>
inline __device__ void raw_to_comp_load<GPUJPEG_4444_U8_P0123>(const uint8_t* d_data_raw, int &image_width, int &image_height, int &offset, int &x, int &y, uchar4 &r)
{
    r.x = d_data_raw[offset];
    r.y = d_data_raw[offset + 1];
    r.z = d_data_raw[offset + 2];
    r.w = d_data_raw[offset + 3];
}

template<>
inline __device__ void raw_to_comp_load<GPUJPEG_422_U8_P1020>(const uint8_t* d_data_raw, int &image_width, int &image_height, int &offset, int &x, int &y, uchar4 &r)
{
    r.x = d_data_raw[offset + 1];
    if ( offset % 4 == 0 ) {
        r.y = d_data_raw[offset];
        r.z = d_data_raw[offset + 2];
    } else {
        r.y = d_data_raw[offset - 2];
        r.z = d_data_raw[offset];
    }
}

/**
 * Kernel - Copy raw image source data into three separated component buffers
 */
typedef void (*gpujpeg_preprocessor_encode_kernel)(struct gpujpeg_preprocessor_data data, const uint8_t* d_data_raw, int image_width_padding, int image_width, int image_height, uint32_t width_div_mul, uint32_t width_div_shift);

/**
 * @note
 * In previous versions, there was an optimalization with aligned preloading to shared memory.
 * This was, however, removed because it didn't exhibit any performance improvement anymore
 * (actually removing that yields slight performance gain).
 */
template<
    enum gpujpeg_color_space color_space_internal,
    enum gpujpeg_color_space color_space,
    enum gpujpeg_pixel_format pixel_format,
    uint8_t s_comp1_samp_factor_h, uint8_t s_comp1_samp_factor_v,
    uint8_t s_comp2_samp_factor_h, uint8_t s_comp2_samp_factor_v,
    uint8_t s_comp3_samp_factor_h, uint8_t s_comp3_samp_factor_v,
    uint8_t s_comp4_samp_factor_h, uint8_t s_comp4_samp_factor_v
>
__global__ void
gpujpeg_preprocessor_raw_to_comp_kernel(struct gpujpeg_preprocessor_data data, const uint8_t* d_data_raw, int image_width_padding, int image_width, int image_height, uint32_t width_div_mul, uint32_t width_div_shift)
{
    int x  = threadIdx.x;
    int gX = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x;

    // Position
    int image_position = gX + x;
    int image_position_y = gpujpeg_const_div_divide(image_position, width_div_mul, width_div_shift);
    int image_position_x = image_position - (image_position_y * image_width);

    if ( image_position >= (image_width * image_height) ) {
        return;
    }

    // Load
    uchar4 r;
    int offset = image_position * unit_size<pixel_format>() + image_width_padding * image_position_y;
    raw_to_comp_load<pixel_format>(d_data_raw, image_width, image_height, offset, image_position_x, image_position_y, r);

    // Color transform
    gpujpeg_color_transform<color_space, color_space_internal>::perform(r);

    // Store
    gpujpeg_preprocessor_raw_to_comp_store<s_comp1_samp_factor_h, s_comp1_samp_factor_v, s_comp2_samp_factor_h,
                                           s_comp2_samp_factor_v, s_comp3_samp_factor_h, s_comp3_samp_factor_v,
                                           s_comp4_samp_factor_h, s_comp4_samp_factor_v>::perform(r, image_position_x,
                                                                                                  image_position_y,
                                                                                                  data);
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
    gpujpeg_sampling_factor_t sampling_factor = gpujpeg_preprocessor_make_sampling_factor_i(
        coder->param.comp_count, coder->sampling_factor.horizontal, coder->sampling_factor.vertical,
        coder->component[0].sampling_factor.horizontal, coder->component[0].sampling_factor.vertical,
        coder->component[1].sampling_factor.horizontal, coder->component[1].sampling_factor.vertical,
        coder->component[2].sampling_factor.horizontal, coder->component[2].sampling_factor.vertical,
        coder->component[3].sampling_factor.horizontal, coder->component[3].sampling_factor.vertical);

#define RETURN_KERNEL_SWITCH(PIXEL_FORMAT, COLOR, P1, P2, P3, P4, P5, P6, P7, P8) \
        switch ( PIXEL_FORMAT ) { \
            case GPUJPEG_444_U8_P012: return &gpujpeg_preprocessor_raw_to_comp_kernel<color_space_internal, COLOR, GPUJPEG_444_U8_P012, P1, P2, P3, P4, P5, P6, P7, P8>; \
            case GPUJPEG_4444_U8_P0123: return &gpujpeg_preprocessor_raw_to_comp_kernel<color_space_internal, COLOR, GPUJPEG_4444_U8_P0123, P1, P2, P3, P4, P5, P6, P7, P8>; \
            case GPUJPEG_422_U8_P1020: return &gpujpeg_preprocessor_raw_to_comp_kernel<color_space_internal, COLOR, GPUJPEG_422_U8_P1020, P1, P2, P3, P4, P5, P6, P7, P8>; \
            case GPUJPEG_444_U8_P0P1P2: return &gpujpeg_preprocessor_raw_to_comp_kernel<color_space_internal, COLOR, GPUJPEG_444_U8_P0P1P2, P1, P2, P3, P4, P5, P6, P7, P8>; \
            case GPUJPEG_422_U8_P0P1P2: return &gpujpeg_preprocessor_raw_to_comp_kernel<color_space_internal, COLOR, GPUJPEG_422_U8_P0P1P2, P1, P2, P3, P4, P5, P6, P7, P8>; \
            case GPUJPEG_420_U8_P0P1P2: return &gpujpeg_preprocessor_raw_to_comp_kernel<color_space_internal, COLOR, GPUJPEG_420_U8_P0P1P2, P1, P2, P3, P4, P5, P6, P7, P8>; \
            case GPUJPEG_U8: return &gpujpeg_preprocessor_raw_to_comp_kernel<color_space_internal, COLOR, GPUJPEG_U8, P1, P2, P3, P4, P5, P6, P7, P8>; \
            case GPUJPEG_PIXFMT_NONE: GPUJPEG_ASSERT(0 && "Preprocess from GPUJPEG_PIXFMT_NONE not allowed" ); \
        }

#define RETURN_KERNEL_IF(PIXEL_FORMAT, COLOR, COMP_COUNT, P1, P2, P3, P4, P5, P6, P7, P8) \
    if ( sampling_factor == gpujpeg_make_sampling_factor(COMP_COUNT, P1, P2, P3, P4, P5, P6, P7, P8) ) { \
        if ( coder->param.verbose >= GPUJPEG_LL_VERBOSE ) {                                                            \
            print_kernel_configuration(                                                                                \
                "Using faster kernel for preprocessor (precompiled %dx%d, %dx%d, %dx%d, %dx%d).\n");                   \
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
                "Using slower kernel for preprocessor (dynamic %dx%d, %dx%d, %dx%d, %dx%d).\n");                       \
        } \
        RETURN_KERNEL_SWITCH(PIXEL_FORMAT, COLOR, GPUJPEG_DYNAMIC, GPUJPEG_DYNAMIC, GPUJPEG_DYNAMIC, GPUJPEG_DYNAMIC,  \
                             GPUJPEG_DYNAMIC, GPUJPEG_DYNAMIC, GPUJPEG_DYNAMIC, GPUJPEG_DYNAMIC)                       \
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
#ifdef ENABLE_YUV
    // YUV color space
    else if ( coder->param_image.color_space == GPUJPEG_YUV ) {
        RETURN_KERNEL(coder->param_image.pixel_format, GPUJPEG_YUV);
    }
#endif
    // Unknown color space
    else {
        assert(false);
    }

#undef RETURN_KERNEL_IF
#undef RETURN_KERNEL

    return NULL;
}

static bool
gpujpeg_preprocessor_encode_no_transform(struct gpujpeg_coder *coder)
{
    if (gpujpeg_pixel_format_is_interleaved(coder->param_image.pixel_format)) {
        return false;
    }

    if ( coder->param.comp_count == 3 && coder->param_image.color_space != coder->param.color_space_internal ) {
        return false;
    }

    const struct gpujpeg_component_sampling_factor* sampling_factors =
        gpujpeg_pixel_format_get_sampling_factor(coder->param_image.pixel_format);
    for ( int i = 0; i < coder->param.comp_count; ++i ) {
        if ( coder->component[i].sampling_factor.horizontal != sampling_factors[i].horizontal ||
             coder->component[i].sampling_factor.vertical != sampling_factors[i].vertical ) {
            return false;
        }
    }
    return true;
}

/* Documented at declaration */
int
gpujpeg_preprocessor_encoder_init(struct gpujpeg_coder* coder)
{
    coder->preprocessor.kernel = nullptr;
    struct gpujpeg_preprocessor_data *data = &coder->preprocessor.data;
    for ( int comp = 0; comp < coder->param.comp_count; comp++ ) {
        assert(coder->sampling_factor.horizontal % coder->component[comp].sampling_factor.horizontal == 0);
        assert(coder->sampling_factor.vertical % coder->component[comp].sampling_factor.vertical == 0);
        data->comp[comp].d_data = coder->component[comp].d_data;
        data->comp[comp].sampling_factor.horizontal = coder->sampling_factor.horizontal / coder->component[comp].sampling_factor.horizontal;
        data->comp[comp].sampling_factor.vertical = coder->sampling_factor.vertical / coder->component[comp].sampling_factor.vertical;
        data->comp[comp].data_width = coder->component[comp].data_width;
    }

    if ( gpujpeg_preprocessor_encode_no_transform(coder) ) {
        DEBUG_MSG(coder->param.verbose, "Matching format detected - not using preprocessor, using memcpy instead.\n");
        return 0;
    }

    if (coder->param.color_space_internal == GPUJPEG_NONE) {
        coder->preprocessor.kernel = (void*)gpujpeg_preprocessor_select_encode_kernel<GPUJPEG_NONE>(coder);
    }
    else if (coder->param.color_space_internal == GPUJPEG_RGB) {
        coder->preprocessor.kernel = (void*)gpujpeg_preprocessor_select_encode_kernel<GPUJPEG_RGB>(coder);
    }
    else if (coder->param.color_space_internal == GPUJPEG_YCBCR_BT601) {
        coder->preprocessor.kernel = (void*)gpujpeg_preprocessor_select_encode_kernel<GPUJPEG_YCBCR_BT601>(coder);
    }
    else if (coder->param.color_space_internal == GPUJPEG_YCBCR_BT601_256LVLS) {
        coder->preprocessor.kernel = (void*)gpujpeg_preprocessor_select_encode_kernel<GPUJPEG_YCBCR_BT601_256LVLS>(coder);
    }
    else if (coder->param.color_space_internal == GPUJPEG_YCBCR_BT709) {
        coder->preprocessor.kernel = (void*)gpujpeg_preprocessor_select_encode_kernel<GPUJPEG_YCBCR_BT709>(coder);
    }

    if ( coder->preprocessor.kernel == nullptr ) {
        return -1;
    }

    return 0;
}

int
gpujpeg_preprocessor_encode_interlaced(struct gpujpeg_encoder * encoder)
{
    struct gpujpeg_coder* coder = &encoder->coder;

    // Select kernel
    gpujpeg_preprocessor_encode_kernel kernel = (gpujpeg_preprocessor_encode_kernel) coder->preprocessor.kernel;
    assert(kernel != NULL);

    int image_width = coder->param_image.width;
    int image_height = coder->param_image.height;

    // When loading 4:2:2 data of odd width, the data in fact has even width, so round it
    // (at least imagemagick convert tool generates data stream in this way)
    if (coder->param_image.pixel_format == GPUJPEG_422_U8_P1020) {
        image_width = (coder->param_image.width + 1) & ~1;
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
    while ( grid.x > GPUJPEG_CUDA_MAXIMUM_GRID_SIZE ) {
        grid.y *= 2;
        grid.x = gpujpeg_div_and_round_up(grid.x, 2);
    }

    // Decompose input image width for faster division using multiply-high and right shift
    uint32_t width_div_mul, width_div_shift;
    gpujpeg_const_div_prepare(image_width, width_div_mul, width_div_shift);

    // Run kernel
    kernel<<<grid, threads, 0, encoder->stream>>>(
        coder->preprocessor.data,
        coder->d_data_raw,
        coder->param_image.width_padding,
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
 * @invariant gpujpeg_preprocessor_encode_no_transform(coder) != 0
 */
static int
gpujpeg_preprocessor_encoder_copy_planar_data(struct gpujpeg_encoder * encoder)
{
    struct gpujpeg_coder * coder = &encoder->coder;
    assert(coder->param.comp_count == 1 || coder->param.comp_count == 3);

    size_t data_raw_offset = 0;
    bool needs_stride = false; // true if width is not divisible by MCU width
    for ( int i = 0; i < coder->param.comp_count; ++i ) {
        int component_width = coder->component[i].width + coder->param_image.width_padding;
        needs_stride = needs_stride || component_width != coder->component[i].data_width;
    }
    if (!needs_stride) {
            for ( int i = 0; i < coder->param.comp_count; ++i ) {
                    size_t component_size = coder->component[i].width * coder->component[i].height;
                    cudaMemcpyAsync(coder->component[i].d_data, coder->d_data_raw + data_raw_offset, component_size, cudaMemcpyDeviceToDevice, encoder->stream);
                    data_raw_offset += component_size;
            }
    } else {
           for ( int i = 0; i < coder->param.comp_count; ++i ) {
                    int spitch = coder->component[i].width + coder->param_image.width_padding;
                    int dpitch = coder->component[i].data_width;
                    size_t component_size = spitch * coder->component[i].height;
                    cudaMemcpy2DAsync(coder->component[i].d_data, dpitch, coder->d_data_raw + data_raw_offset, spitch,
                                      coder->component[i].width, coder->component[i].height, cudaMemcpyDeviceToDevice,
                                      encoder->stream);
                    data_raw_offset += component_size;
            }
    }
    gpujpeg_cuda_check_error("Preprocessor copy failed", return -1);
    return 0;
}

static __global__ void
vertical_flip_kernel(uint32_t* data,
                     int width, // image linesize/4
                     int height // image height in pixels
)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x; // column index
    int y = blockIdx.y * blockDim.y + threadIdx.y; // row index

    if ( x < width ) {
        // Flipped row index
        int flipped_y = height - 1 - y;
        uint32_t tmp = data[y * width + x];
        data[y * width + x] = data[flipped_y * width + x];
        data[flipped_y * width + x] = tmp;
    }
}

int
gpujpeg_preprocessor_flip_lines(struct gpujpeg_coder* coder, cudaStream_t stream)
{
    for ( int i = 0; i < coder->param.comp_count; ++i ) {
        dim3 block(RGB_8BIT_THREADS, 1);
        int width = coder->component[i].data_width / 4;
        int height = coder->component[i].data_height;
        dim3 grid((width + block.x - 1) / block.x, height / 2); // only half of height
        vertical_flip_kernel<<<grid, block, 0, stream>>>((uint32_t*)coder->component[i].d_data, width, height);
    }
    gpujpeg_cuda_check_error("Preprocessor flip failed", return -1);
    return 0;
}

template <enum gpujpeg_pixel_format pixel_format>
__global__ void
channel_remap_kernel(uint8_t* data, int width, int pitch, int height, unsigned int byte_map)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x; // column index
    int y = blockIdx.y * blockDim.y + threadIdx.y; // row index

    if ( x >= width || y >= height ) {
        return;
    }

    // Load
    uchar4 r;
    int offset = y * pitch + x * unit_size<pixel_format>();
    raw_to_comp_load<pixel_format>(data, width, height, offset, x, y, r);

    // Permutation
    uint32_t val = r.w << 24 |  r.z << 16 | r.y << 8 | r.x;
    val = __byte_perm(val, 0xFF, byte_map);
    r.w = val >> 24;
    r.z = (val >> 16) & 0xFF;
    r.y = (val >> 8) & 0xFF;
    r.x = val & 0xFF;

    // Store
    gpujpeg_comp_to_raw_store<pixel_format>(data, width, height, offset, x, y, r);
}

/**
 * remaps color channels according to coder->preprocessor.channel_remap
 * if requested by user (with an option)
 */
int
gpujpeg_preprocessor_channel_remap(struct gpujpeg_coder* coder, cudaStream_t stream)
{
    const unsigned comp_count = gpujpeg_pixel_format_get_comp_count(coder->param_image.pixel_format);
    const unsigned mapped_count = coder->preprocessor.channel_remap >> 24;
    if (comp_count != mapped_count) {
        ERROR_MSG("Wrong channel remapping given, given %u channels but pixel format has %u!\n", mapped_count,
                  comp_count);
        return -1;
    }
    const unsigned mapping = coder->preprocessor.channel_remap & 0xFFFF;

    dim3 block(16, 16);
    int width = coder->param_image.width;
    int height = coder->param_image.height;
    int pitch = (width * gpujpeg_pixel_format_get_comp_count(coder->param_image.pixel_format)) +
                coder->param_image.width_padding;
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    auto* kernel = channel_remap_kernel<GPUJPEG_U8>;
#define SWITCH_KERNEL(pf)                                                                                              \
    case pf:                                                                                                           \
        kernel = channel_remap_kernel<pf>;                                                                             \
        break
    switch ( coder->param_image.pixel_format ) {
        SWITCH_KERNEL(GPUJPEG_444_U8_P012);
        SWITCH_KERNEL(GPUJPEG_4444_U8_P0123);
        SWITCH_KERNEL(GPUJPEG_422_U8_P1020);
        SWITCH_KERNEL(GPUJPEG_444_U8_P0P1P2);
        SWITCH_KERNEL(GPUJPEG_422_U8_P0P1P2);
        SWITCH_KERNEL(GPUJPEG_420_U8_P0P1P2);
        SWITCH_KERNEL(GPUJPEG_U8);
    case GPUJPEG_PIXFMT_NONE:
        GPUJPEG_ASSERT(0 && "Preprocess from GPUJPEG_PIXFMT_NONE not allowed");
    }
#undef SWITDH_KERNEL
    kernel<<<grid, block, 0, stream>>>(coder->d_data_raw, width, pitch, height, mapping);
    gpujpeg_cuda_check_error("channel_remap_kernel failed", return -1);
    return 0;
}

/* Documented at declaration */
int
gpujpeg_preprocessor_encode(struct gpujpeg_encoder * encoder)
{
    struct gpujpeg_coder * coder = &encoder->coder;
    /// @todo ensure that all combinations work so the assert is really unneeded
    // assert(coder->param_image.width_padding == 0 ||
    //        (coder->param_image.pixel_format == GPUJPEG_444_U8_P012 && coder->preprocessor.kernel != nullptr));

    if ( coder->preprocessor.channel_remap != 0 ) {
        const int ret = gpujpeg_preprocessor_channel_remap(coder, encoder->stream);
        if ( ret != 0 ) {
            return ret;
        }
    }

    int ret = coder->preprocessor.kernel != nullptr ? gpujpeg_preprocessor_encode_interlaced(encoder)
                                                    : gpujpeg_preprocessor_encoder_copy_planar_data(encoder);
    if (ret != 0) {
        return ret;
    }
    if ( coder->preprocessor.flipped ) {
        return gpujpeg_preprocessor_flip_lines(coder, encoder->stream);
    }
    return ret;
}

/* vi: set expandtab sw=4: */
