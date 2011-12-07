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

/** Specialization [color_space_from = GPUJPEG_RGB, color_space_to = GPUJPEG_YCBCR_JPEG] */
template<>
struct gpujpeg_color_transform<GPUJPEG_RGB, GPUJPEG_YCBCR_JPEG> {
    /** RGB -> YCbCr transform (8 bit) */
    static __device__ void 
    perform(float & c1, float & c2, float & c3) {
        float r1 = 0.299f * c1 + 0.587f * c2 + 0.114f * c3;
        float r2 = -0.1687f * c1 - 0.3313f * c2 + 0.5f * c3 + 128.0f;
        float r3 = 0.5f * c1 - 0.4187f * c2 - 0.0813f * c3 + 128.0f;
        c1 = r1;
        c2 = r2;
        c3 = r3;
    }
};

/** Specialization [color_space_from = GPUJPEG_YCBCR_ITU_R, color_space_to = GPUJPEG_YCBCR_JPEG] */
template<>
struct gpujpeg_color_transform<GPUJPEG_YCBCR_ITU_R, GPUJPEG_YCBCR_JPEG> {
    /** YUV -> YCbCr transform (8 bit) */
    static __device__ void 
    perform(float & c1, float & c2, float & c3) {
        c1 -= 16;
        // Check minimum value 0
        c1 = (c1 >= 0.0f) ? c1 : 0.0f;
    }
};

/** Specialization [color_space_from = GPUJPEG_YCBCR_JPEG, color_space_to = GPUJPEG_RGB] */
template<>
struct gpujpeg_color_transform<GPUJPEG_YCBCR_JPEG, GPUJPEG_RGB> {
    /** YCbCr -> RGB transform (8 bit) */
    static __device__ void 
    perform(float & c1, float & c2, float & c3) {
        // Update values
        float r1 = c1 - 0.0f;
        float r2 = c2 - 128.0f;
        float r3 = c3 - 128.0f;
        // Perfomr YCbCr -> RGB conversion
        c1 = (1.0f * r1 + 0.0f * r2 + 1.402f * r3);
        c2 = (1.0f * r1 - 0.344136f * r2 - 0.714136f * r3);
        c3 = (1.0f * r1 + 1.772f * r2 + 0.0f * r3);
        // Check minimum value 0
        c1 = (c1 >= 0.0f) ? c1 : 0.0f;
        c2 = (c2 >= 0.0f) ? c2 : 0.0f;
        c3 = (c3 >= 0.0f) ? c3 : 0.0f;
        // Check maximum value 255
        c1 = (c1 <= 255.0) ? c1 : 255.0f;
        c2 = (c2 <= 255.0) ? c2 : 255.0f;
        c3 = (c3 <= 255.0) ? c3 : 255.0f;    
    }
};

/** Specialization [color_space_from = GPUJPEG_YCBCR_JPEG, color_space_to = GPUJPEG_YCBCR_ITU_R] */
template<>
struct gpujpeg_color_transform<GPUJPEG_YCBCR_JPEG, GPUJPEG_YCBCR_ITU_R> {
    /** YCbCr -> YUV transform (8 bit) */
    static __device__ void 
    perform(float & c1, float & c2, float & c3) {
        c1 += 16;
        // Check maximum value 255
        c1 = (c1 <= 255.0) ? c1 : 255.0f;
    }
};

#define RGB_8BIT_THREADS 256

/**
 * Kernel - Copy raw image source data into three separated component buffers
 *
 * @param d_c1  First component buffer
 * @param d_c2  Second component buffer
 * @param d_c3  Third component buffer
 * @param d_source  Image source data
 * @param pixel_count  Number of pixels to copy
 * @return void
 */
typedef void (*gpujpeg_preprocessor_encode_kernel)(uint8_t* d_c1, uint8_t* d_c2, uint8_t* d_c3, const uint8_t* d_source, int image_width, int image_height, int data_width, int data_height);
 
/** Specialization [sampling factor is 4:4:4] */
template<enum gpujpeg_color_space color_space>
__global__ void 
gpujpeg_preprocessor_raw_to_comp_kernel_4_4_4(uint8_t* d_c1, uint8_t* d_c2, uint8_t* d_c3, const uint8_t* d_source, int image_width, int image_height, int data_width, int data_height)
{
    int x  = threadIdx.x;
    int gX = blockDim.x * blockIdx.x;
        
    // Load to shared
    __shared__ unsigned char s_data[RGB_8BIT_THREADS * 3];
    if ( (x * 4) < RGB_8BIT_THREADS * 3 ) {
        int* s = (int*)d_source;
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
    gpujpeg_color_transform<color_space, GPUJPEG_YCBCR_JPEG>::perform(r1, r2, r3);
    // Store
    int image_position = gX + x;
    if ( image_position < (image_width * image_height) ) {
        int data_position = (image_position / image_width) * data_width + image_position % image_width;
        d_c1[data_position] = (uint8_t)r1;
        d_c2[data_position] = (uint8_t)r2;
        d_c3[data_position] = (uint8_t)r3;
    }
}

/** Specialization [sampling factor is 4:2:2] */
template<enum gpujpeg_color_space color_space>
__global__ void 
gpujpeg_preprocessor_raw_to_comp_kernel_4_2_2(uint8_t* d_c1, uint8_t* d_c2, uint8_t* d_c3, const uint8_t* d_source, int image_width, int image_height, int data_width, int data_height)
{
    int x  = threadIdx.x;
    int gX = blockDim.x * blockIdx.x;
        
    // Load to shared
    __shared__ unsigned char s_data[RGB_8BIT_THREADS * 2];
    if ( (x * 4) < RGB_8BIT_THREADS * 2 ) {
        int* s = (int*)d_source;
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
    gpujpeg_color_transform<color_space, GPUJPEG_YCBCR_JPEG>::perform(r1, r2, r3);
    // Store
    int image_position = gX + x;
    if ( image_position < (image_width * image_height) ) {
        int data_position = (image_position / image_width) * data_width + image_position % image_width;
        d_c1[data_position] = (uint8_t)r1;
        d_c2[data_position] = (uint8_t)r2;
        d_c3[data_position] = (uint8_t)r3;
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
    // RGB color space
    if ( encoder->param_image.color_space == GPUJPEG_RGB ) {
        assert(encoder->param_image.sampling_factor == GPUJPEG_4_4_4);
        return &gpujpeg_preprocessor_raw_to_comp_kernel_4_4_4<GPUJPEG_RGB>;
    } 
    // YCbCr ITU-R color space
    else if ( encoder->param_image.color_space == GPUJPEG_YCBCR_ITU_R ) {
        if ( encoder->param_image.sampling_factor == GPUJPEG_4_4_4 ) {
            return &gpujpeg_preprocessor_raw_to_comp_kernel_4_4_4<GPUJPEG_YCBCR_ITU_R>;
        } else if ( encoder->param_image.sampling_factor == GPUJPEG_4_2_2 ) {
            return &gpujpeg_preprocessor_raw_to_comp_kernel_4_2_2<GPUJPEG_YCBCR_ITU_R>;
        } else {
            assert(false);
        }
    } 
    // YCbCr JPEG color space
    else if ( encoder->param_image.color_space == GPUJPEG_YCBCR_JPEG ) {
        if ( encoder->param_image.sampling_factor == GPUJPEG_4_4_4 ) {
            return &gpujpeg_preprocessor_raw_to_comp_kernel_4_4_4<GPUJPEG_YCBCR_JPEG>;
        } else if ( encoder->param_image.sampling_factor == GPUJPEG_4_2_2 ) {
            return &gpujpeg_preprocessor_raw_to_comp_kernel_4_2_2<GPUJPEG_YCBCR_JPEG>;
        } else {
            assert(false);
        }
    }
    // Unknown color space
    else {
        assert(false);
    }
    return NULL;
}

/** Documented at declaration */
int
gpujpeg_preprocessor_encode(struct gpujpeg_encoder* encoder)
{    
    cudaMemset(encoder->d_data, 0, encoder->data_size * sizeof(uint8_t));

    // Select kernel
    gpujpeg_preprocessor_encode_kernel kernel = gpujpeg_preprocessor_select_encode_kernel(encoder);
    
    int image_width = encoder->param_image.width;
    int image_height = encoder->param_image.height;
    
    // When loading 4:2:2 data of odd width, the data in fact has even width, so round it
    // (at least imagemagick convert tool generates data stream in this way)
    if ( encoder->param_image.sampling_factor == GPUJPEG_4_2_2 )
        image_width = gpujpeg_div_and_round_up(encoder->param_image.width, 2) * 2;
        
    // Prepare unit size
    assert(encoder->param_image.sampling_factor == GPUJPEG_4_4_4 || encoder->param_image.sampling_factor == GPUJPEG_4_2_2);
    int unitSize = encoder->param_image.sampling_factor == GPUJPEG_4_4_4 ? 3 : 2;
    
    // Prepare kernel
    int alignedSize = gpujpeg_div_and_round_up(image_width * image_height, RGB_8BIT_THREADS) * RGB_8BIT_THREADS * unitSize;
    dim3 threads (RGB_8BIT_THREADS);
    dim3 grid (alignedSize / (RGB_8BIT_THREADS * unitSize));
    assert(alignedSize % (RGB_8BIT_THREADS * unitSize) == 0);

    // Run kernel
    int data_comp_size = encoder->data_width * encoder->data_height;
    uint8_t* d_c1 = &encoder->d_data[0 * data_comp_size];
    uint8_t* d_c2 = &encoder->d_data[1 * data_comp_size];
    uint8_t* d_c3 = &encoder->d_data[2 * data_comp_size];
    kernel<<<grid, threads>>>(
        d_c1, 
        d_c2, 
        d_c3, 
        encoder->d_data_source, 
        image_width,
        image_height,
        encoder->data_width,
        encoder->data_height
    );
    cudaError cuerr = cudaThreadSynchronize();
    if ( cuerr != cudaSuccess ) {
        fprintf(stderr, "Preprocessor encoding failed: %s!\n", cudaGetErrorString(cuerr));
        return -1;
    }
        
    return 0;
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
typedef void (*gpujpeg_preprocessor_decode_kernel)(const uint8_t* d_c1, const uint8_t* d_c2, const uint8_t* d_c3, uint8_t* d_target, int image_width, int image_height, int data_width, int data_height);

/** Specialization [sampling factor is 4:4:4] */
template<enum gpujpeg_color_space color_space>
__global__ void
gpujpeg_preprocessor_comp_to_raw_kernel_4_4_4(const uint8_t* d_c1, const uint8_t* d_c2, const uint8_t* d_c3, uint8_t* d_target, int image_width, int image_height, int data_width, int data_height)
{
    int x  = threadIdx.x;
    int gX = blockDim.x * blockIdx.x;
    int image_position = gX + x;
    if ( image_position >= (image_width * image_height) )
        return;
    int data_position = (image_position / image_width) * data_width + image_position % image_width;
    image_position = image_position * 3;
    
    // Load
    float r1 = (float)(d_c1[data_position]);
    float r2 = (float)(d_c2[data_position]);
    float r3 = (float)(d_c3[data_position]);
    // Color transform
    gpujpeg_color_transform<GPUJPEG_YCBCR_JPEG, color_space>::perform(r1, r2, r3);
    // Save
    d_target[image_position + 0] = (uint8_t)r1;
    d_target[image_position + 1] = (uint8_t)r2;
    d_target[image_position + 2] = (uint8_t)r3;
}

/** Specialization [sampling factor is 4:2:2] */
template<enum gpujpeg_color_space color_space>
__global__ void
gpujpeg_preprocessor_comp_to_raw_kernel_4_2_2(const uint8_t* d_c1, const uint8_t* d_c2, const uint8_t* d_c3, uint8_t* d_target, int image_width, int image_height, int data_width, int data_height)
{
    int x  = threadIdx.x;
    int gX = blockDim.x * blockIdx.x;
    int image_position = gX + x;
    if ( image_position >= (image_width * image_height) )
        return;
    int image_x = image_position % image_width;
    int data_position = (image_position / image_width) * data_width + image_position % image_width;
    image_position = image_position * 2;
    
    // Load
    float r1 = (float)(d_c1[data_position]);
    float r2 = (float)(d_c2[data_position]);
    float r3 = (float)(d_c3[data_position]);
    // Color transform
    gpujpeg_color_transform<GPUJPEG_YCBCR_JPEG, color_space>::perform(r1, r2, r3);
    // Save
    d_target[image_position + 1] = (uint8_t)r1;
    if ( (image_x % 2) == 0 )
        d_target[image_position + 0] = (uint8_t)r2;
    else
        d_target[image_position + 0] = (uint8_t)r3;
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
    // RGB color space
    if ( decoder->param_image.color_space == GPUJPEG_RGB ) {
        assert(decoder->param_image.sampling_factor == GPUJPEG_4_4_4);
        return &gpujpeg_preprocessor_comp_to_raw_kernel_4_4_4<GPUJPEG_RGB>;
    } 
    // YCbCr ITU-R color space
    else if ( decoder->param_image.color_space == GPUJPEG_YCBCR_ITU_R ) {
        if ( decoder->param_image.sampling_factor == GPUJPEG_4_4_4 ) {
            return &gpujpeg_preprocessor_comp_to_raw_kernel_4_4_4<GPUJPEG_YCBCR_ITU_R>;
        } else if ( decoder->param_image.sampling_factor == GPUJPEG_4_2_2 ) {
            return &gpujpeg_preprocessor_comp_to_raw_kernel_4_2_2<GPUJPEG_YCBCR_ITU_R>;
        } else {
            assert(false);
        }
    }
    // YCbCr JPEG color space
    else if ( decoder->param_image.color_space == GPUJPEG_YCBCR_JPEG ) {
        if ( decoder->param_image.sampling_factor == GPUJPEG_4_4_4 ) {
            return &gpujpeg_preprocessor_comp_to_raw_kernel_4_4_4<GPUJPEG_YCBCR_JPEG>;
        } else if ( decoder->param_image.sampling_factor == GPUJPEG_4_2_2 ) {
            return &gpujpeg_preprocessor_comp_to_raw_kernel_4_2_2<GPUJPEG_YCBCR_JPEG>;
        } else {
            assert(false);
        }
    }
    // Unknown color space
    else {
        assert(false);
    }
    return NULL;
}

/** Documented at declaration */
int
gpujpeg_preprocessor_decode(struct gpujpeg_decoder* decoder)
{
    cudaMemset(decoder->d_data_target, 0, decoder->data_target_size * sizeof(uint8_t));
    
    // Select kernel
    gpujpeg_preprocessor_decode_kernel kernel = gpujpeg_preprocessor_select_decode_kernel(decoder);
    
    int image_width = decoder->param_image.width;
    int image_height = decoder->param_image.height;
    
    // When saving 4:2:2 data of odd width, the data should have even width, so round it
    if ( decoder->param_image.sampling_factor == GPUJPEG_4_2_2 )
        image_width = gpujpeg_div_and_round_up(decoder->param_image.width, 2) * 2;
        
    // Prepare unit size
    assert(decoder->param_image.sampling_factor == GPUJPEG_4_4_4 || decoder->param_image.sampling_factor == GPUJPEG_4_2_2);
    int unitSize = decoder->param_image.sampling_factor == GPUJPEG_4_4_4 ? 3 : 2;
    
    // Prepare kernel
    int alignedSize = gpujpeg_div_and_round_up(image_width * image_height, RGB_8BIT_THREADS) * RGB_8BIT_THREADS * unitSize;
    dim3 threads (RGB_8BIT_THREADS);
    dim3 grid (alignedSize / (RGB_8BIT_THREADS * unitSize));
    assert(alignedSize % (RGB_8BIT_THREADS * unitSize) == 0);

    // Run kernel
    int data_comp_size = decoder->data_width * decoder->data_height;
    uint8_t* d_c1 = &decoder->d_data[0 * data_comp_size];
    uint8_t* d_c2 = &decoder->d_data[1 * data_comp_size];
    uint8_t* d_c3 = &decoder->d_data[2 * data_comp_size];
    kernel<<<grid, threads>>>(
        d_c1, 
        d_c2, 
        d_c3, 
        decoder->d_data_target, 
        image_width,
        image_height,
        decoder->data_width,
        decoder->data_height
    );
    cudaError cuerr = cudaThreadSynchronize();
    if ( cuerr != cudaSuccess ) {
        fprintf(stderr, "Preprocessing decoding failed: %s!\n", cudaGetErrorString(cuerr));
        return -1;
    }
    
    return 0;
}
