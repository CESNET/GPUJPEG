/**
 * Copyright (c) 2011, Martin Srom
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
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
 
#include "jpeg_preprocessor.h"
#include "jpeg_util.h"

#define RGB_8BIT_THREADS 256

/**
 * Kernel - Copy image source data into three separated component buffers
 *
 * @param d_c1  First component buffer
 * @param d_c2  Second component buffer
 * @param d_c3  Third component buffer
 * @param d_source  Image source data
 * @param pixel_count  Number of pixels to copy
 * @return void
 */
__global__ void d_rgb_to_comp(uint8_t* d_c1, uint8_t* d_c2, uint8_t* d_c3, const uint8_t* d_source, int pixel_count)
{
    int x  = threadIdx.x;
    int gX = blockDim.x * blockIdx.x;

    __shared__ unsigned char s_data[RGB_8BIT_THREADS * 3];

    if ( (x * 4) < RGB_8BIT_THREADS * 3 ) {
        int* s = (int*)d_source;
        int* d = (int*)s_data;
        d[x] = s[((gX * 3) >> 2) + x];
    }
    __syncthreads();

    int offset = x * 3;
    float r1 = (float)(s_data[offset]);
    float r2 = (float)(s_data[offset + 1]);
    float r3 = (float)(s_data[offset + 2]);
    int globalOutputPosition = gX + x;
    if ( globalOutputPosition < pixel_count ) {
                
        d_c1[globalOutputPosition] = (uint8_t)(0.299 * r1 + 0.587 * r2 + 0.114 * r3);
        d_c2[globalOutputPosition] = (uint8_t)(-0.1687 * r1 - 0.3313 * r2 + 0.5 * r3 + 128);
        d_c3[globalOutputPosition] = (uint8_t)(0.5 * r1 - 0.4187 * r2 - 0.0813 * r3 + 128);
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
__global__ void d_comp_to_rgb(const uint8_t* d_c1, const uint8_t* d_c2, const uint8_t* d_c3, uint8_t* d_target, int pixel_count)
{
    int x  = threadIdx.x;
    int gX = blockDim.x * blockIdx.x;
    
    int globalInputPosition = gX + x;
    if ( globalInputPosition >= pixel_count )
        return;
        
    float r1 = (float)(d_c1[globalInputPosition] - 16);
    float r2 = (float)(d_c2[globalInputPosition] - 128);
    float r3 = (float)(d_c3[globalInputPosition] - 128);

    int globalOutputPosition = (gX + x) * 3;
    d_target[globalOutputPosition + 0] = (uint8_t)(1.0 * r1 + 0.0 * r2 + 1.402 * r3);
    d_target[globalOutputPosition + 1] = (uint8_t)(1.0 * r1 - 0.344136 * r2 - 0.714136 * r3);
    d_target[globalOutputPosition + 2] = (uint8_t)(1.0 * r1 + 1.772 * r2 + 0.0 * r3);
}

/** Documented at declaration */
int
jpeg_preprocessor_encode(struct jpeg_encoder* encoder)
{
    int pixel_count = encoder->width * encoder->height;
    int alignedSize = (pixel_count / RGB_8BIT_THREADS + 1) * RGB_8BIT_THREADS * 3;
        
    // Kernel
    dim3 threads (RGB_8BIT_THREADS);
    dim3 grid (alignedSize / (RGB_8BIT_THREADS * 3));
    assert(alignedSize % (RGB_8BIT_THREADS * 3) == 0);

    uint8_t* d_c1 = &encoder->d_data[0 * pixel_count];
    uint8_t* d_c2 = &encoder->d_data[1 * pixel_count];
    uint8_t* d_c3 = &encoder->d_data[2 * pixel_count];
    d_rgb_to_comp<<<grid, threads>>>(d_c1, d_c2, d_c3, encoder->d_data_source, pixel_count);
    
    cudaError cuerr = cudaThreadSynchronize();
    if ( cuerr != cudaSuccess ) {
        fprintf(stderr, "Preprocessor encoding failed: %s!\n", cudaGetErrorString(cuerr));
        return -1;
    }
    
    return 0;
}

/** Documented at declaration */
int
jpeg_preprocessor_decode(struct jpeg_decoder* decoder)
{
    int pixel_count = decoder->width * decoder->height;
    int alignedSize = (pixel_count / RGB_8BIT_THREADS + 1) * RGB_8BIT_THREADS * 3;
        
    // Kernel
    dim3 threads (RGB_8BIT_THREADS);
    dim3 grid (alignedSize / (RGB_8BIT_THREADS * 3));
    assert(alignedSize % (RGB_8BIT_THREADS * 3) == 0);

    uint8_t* d_c1 = &decoder->d_data[0 * pixel_count];
    uint8_t* d_c2 = &decoder->d_data[1 * pixel_count];
    uint8_t* d_c3 = &decoder->d_data[2 * pixel_count];
    d_comp_to_rgb<<<grid, threads>>>(d_c1, d_c2, d_c3, decoder->d_data_target, pixel_count);
    
    cudaError cuerr = cudaThreadSynchronize();
    if ( cuerr != cudaSuccess ) {
        fprintf(stderr, "Preprocessing decoding failed: %s!\n", cudaGetErrorString(cuerr));
        return -1;
    }
    
    return 0;
}