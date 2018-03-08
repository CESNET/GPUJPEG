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

#include "gpujpeg_dct_gpu.h"
#include <libgpujpeg/gpujpeg_util.h>

/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/** Fast integer multiplication */
#define FMUL(x,y)   (__mul24(x,y))
//#define FMUL(x,y)   ((x)*(y))

// X block count which will be processed by one thread block
#define GPUJPEG_DCT_BLOCK_COUNT_X       4
// Y block count which will be processed by one thread block
#define GPUJPEG_DCT_BLOCK_COUNT_Y       4

// Thread block width
#define GPUJPEG_DCT_THREAD_BLOCK_WIDTH  (GPUJPEG_BLOCK_SIZE * GPUJPEG_DCT_BLOCK_COUNT_X)
// Thread block height
#define GPUJPEG_DCT_THREAD_BLOCK_HEIGHT (GPUJPEG_BLOCK_SIZE * GPUJPEG_DCT_BLOCK_COUNT_Y)

// Stride of shared memory buffer (short kernel)
#define GPUJPEG_DCT_THREAD_BLOCK_STRIDE (GPUJPEG_DCT_THREAD_BLOCK_WIDTH + 4)

#define IMAD(a, b, c) ( ((a) * (b)) + (c) )
#define IMUL(a, b) ((a) * (b))

#define SIN_1_4     0x5A82
#define COS_1_4     0x5A82
#define SIN_1_8     0x30FC
#define COS_1_8     0x7642

#define OSIN_1_16   0x063E
#define OSIN_3_16   0x11C7
#define OSIN_5_16   0x1A9B
#define OSIN_7_16   0x1F63

#define OCOS_1_16   0x1F63
#define OCOS_3_16   0x1A9B
#define OCOS_5_16   0x11C7
#define OCOS_7_16   0x063E

/**
 * Package of 2 shorts into 1 int - designed to perform i/o by integers to avoid bank conflicts
 */
union PackedInteger
{
    struct __align__(8)
    {
        int16_t hShort1;
        int16_t hShort2;
    };
    int32_t hInt;
};

/**
 * Converts fixed point value to short value
 */
__device__ inline int16_t
unfixh(int x)
{
    return (int16_t)((x + 0x8000) >> 16);
}

/**
 * Converts fixed point value to short value
 */
__device__ inline int
unfixo(int x)
{
    return (x + 0x1000) >> 13;
}

/**
 * Performs in-place IDCT of vector of 8 elements (used to access columns in shared memory).
 *
 * @param SrcDst [IN/OUT] - Pointer to the first element of vector
 * @param Stride [IN] - Value to add to ptr to access other elements
 * @return None
 */
__device__ void
gpujpeg_idct_gpu_kernel_inplace(int16_t* SrcDst, int Stride)
{
    int in0, in1, in2, in3, in4, in5, in6, in7;
    int tmp10, tmp11, tmp12, tmp13;
    int tmp20, tmp21, tmp22, tmp23;
    int tmp30, tmp31;
    int tmp40, tmp41, tmp42, tmp43;
    int tmp50, tmp51, tmp52, tmp53;

    int16_t *DstPtr = SrcDst;
    in0 = *DstPtr;
    DstPtr += Stride;
    in1 = *DstPtr;
    DstPtr += Stride;
    in2 = *DstPtr;
    DstPtr += Stride;
    in3 = *DstPtr;
    DstPtr += Stride;
    in4 = *DstPtr;
    DstPtr += Stride;
    in5 = *DstPtr;
    DstPtr += Stride;
    in6 = *DstPtr;
    DstPtr += Stride;
    in7 = *DstPtr;

    tmp10 = FMUL(in0 + in4, COS_1_4);
    tmp11 = FMUL(in0 - in4, COS_1_4);
    tmp12 = FMUL(in2, SIN_1_8) - FMUL(in6, COS_1_8);
    tmp13 = FMUL(in6, SIN_1_8) + FMUL(in2, COS_1_8);

    tmp20 = tmp10 + tmp13;
    tmp21 = tmp11 + tmp12;
    tmp22 = tmp11 - tmp12;
    tmp23 = tmp10 - tmp13;

    tmp30 = unfixo(FMUL(in3 + in5, COS_1_4));
    tmp31 = unfixo(FMUL(in3 - in5, COS_1_4));

    in1 <<= 2;
    in7 <<= 2;

    tmp40 = in1 + tmp30;
    tmp41 = in7 + tmp31;
    tmp42 = in1 - tmp30;
    tmp43 = in7 - tmp31;

    tmp50 = FMUL(tmp40, OCOS_1_16) + FMUL(tmp41, OSIN_1_16);
    tmp51 = FMUL(tmp40, OSIN_1_16) - FMUL(tmp41, OCOS_1_16);
    tmp52 = FMUL(tmp42, OCOS_5_16) + FMUL(tmp43, OSIN_5_16);
    tmp53 = FMUL(tmp42, OSIN_5_16) - FMUL(tmp43, OCOS_5_16);

    DstPtr = SrcDst;
    *DstPtr = unfixh(tmp20 + tmp50);
    DstPtr += Stride;
    *DstPtr = unfixh(tmp21 + tmp53);
    DstPtr += Stride;
    *DstPtr = unfixh(tmp22 + tmp52);
    DstPtr += Stride;
    *DstPtr = unfixh(tmp23 + tmp51);
    DstPtr += Stride;
    *DstPtr = unfixh(tmp23 - tmp51);
    DstPtr += Stride;
    *DstPtr = unfixh(tmp22 - tmp52);
    DstPtr += Stride;
    *DstPtr = unfixh(tmp21 - tmp53);
    DstPtr += Stride;
    *DstPtr = unfixh(tmp20 - tmp50);
}

/**
 * Performs in-place IDCT of vector of 8 elements (used to access rows in shared memory).
 *
 * @param V8 [IN/OUT] - Pointer to the first two elements of vector
 * @return None
 */
__device__ void
gpujpeg_idct_gpu_kernel_inplace(uint32_t* V8)
{
    int in0, in1, in2, in3, in4, in5, in6, in7;
    int tmp10, tmp11, tmp12, tmp13;
    int tmp20, tmp21, tmp22, tmp23;
    int tmp30, tmp31;
    int tmp40, tmp41, tmp42, tmp43;
    int tmp50, tmp51, tmp52, tmp53;
    PackedInteger sh0, sh1, sh2, sh3;

    sh0.hInt = V8[0];
    sh1.hInt = V8[1];
    sh2.hInt = V8[2];
    sh3.hInt = V8[3];
    in0 = sh0.hShort1;
    in1 = sh0.hShort2;
    in2 = sh1.hShort1;
    in3 = sh1.hShort2;
    in4 = sh2.hShort1;
    in5 = sh2.hShort2;
    in6 = sh3.hShort1;
    in7 = sh3.hShort2;

    tmp10 = FMUL(in0 + in4, COS_1_4);
    tmp11 = FMUL(in0 - in4, COS_1_4);
    tmp12 = FMUL(in2, SIN_1_8) - FMUL(in6, COS_1_8);
    tmp13 = FMUL(in6, SIN_1_8) + FMUL(in2, COS_1_8);

    tmp20 = tmp10 + tmp13;
    tmp21 = tmp11 + tmp12;
    tmp22 = tmp11 - tmp12;
    tmp23 = tmp10 - tmp13;

    tmp30 = unfixo(FMUL(in3 + in5, COS_1_4));
    tmp31 = unfixo(FMUL(in3 - in5, COS_1_4));

    in1 <<= 2;
    in7 <<= 2;

    tmp40 = in1 + tmp30;
    tmp41 = in7 + tmp31;
    tmp42 = in1 - tmp30;
    tmp43 = in7 - tmp31;

    tmp50 = FMUL(tmp40, OCOS_1_16) + FMUL(tmp41, OSIN_1_16);
    tmp51 = FMUL(tmp40, OSIN_1_16) - FMUL(tmp41, OCOS_1_16);
    tmp52 = FMUL(tmp42, OCOS_5_16) + FMUL(tmp43, OSIN_5_16);
    tmp53 = FMUL(tmp42, OSIN_5_16) - FMUL(tmp43, OCOS_5_16);

    sh0.hShort1 = unfixh(tmp20 + tmp50);
    sh0.hShort2 = unfixh(tmp21 + tmp53);
    sh1.hShort1 = unfixh(tmp22 + tmp52);
    sh1.hShort2 = unfixh(tmp23 + tmp51);
    sh2.hShort1 = unfixh(tmp23 - tmp51);
    sh2.hShort2 = unfixh(tmp22 - tmp52);
    sh3.hShort1 = unfixh(tmp21 - tmp53);
    sh3.hShort2 = unfixh(tmp20 - tmp50);

    V8[0] = sh0.hInt;
    V8[1] = sh1.hInt;
    V8[2] = sh2.hInt;
    V8[3] = sh3.hInt;
}


/**
 * 1D 8point DCT, with optional level shift (must be premultiplied).
 * Based on based on Arai, Agui, and Nakajima's DCT algorithm. (Trans. IEICE E-71(11):1095)
 * Implementation inspired by Independent JPEG Group JPEG implementation, file jfdctflt.c,
 * but optimized for CUDA (cheap floating point MAD instructions).
 */
template <typename T>
__device__ static inline void
gpujpeg_dct_gpu(const T in0, const T in1, const T in2, const T in3, const T in4, const T in5, const T in6, const T in7,
                T & out0, T & out1, T & out2, T & out3, T & out4, T & out5, T & out6, T & out7,
                const float level_shift_8 = 0.0f)
{
    const float diff0 = in0 + in7;
    const float diff1 = in1 + in6;
    const float diff2 = in2 + in5;
    const float diff3 = in3 + in4;
    const float diff4 = in3 - in4;
    const float diff5 = in2 - in5;
    const float diff6 = in1 - in6;
    const float diff7 = in0 - in7;

    const float even0 = diff0 + diff3;
    const float even1 = diff1 + diff2;
    const float even2 = diff1 - diff2;
    const float even3 = diff0 - diff3;

    const float even_diff = even2 + even3;

    const float odd0 = diff4 + diff5;
    const float odd1 = diff5 + diff6;
    const float odd2 = diff6 + diff7;

    const float odd_diff5 = (odd0 - odd2) * 0.382683433f;
    const float odd_diff4 = 1.306562965f * odd2 + odd_diff5;
    const float odd_diff3 = diff7 - odd1 * 0.707106781f;
    const float odd_diff2 = 0.541196100f * odd0 + odd_diff5;
    const float odd_diff1 = diff7 + odd1 * 0.707106781f;

    out0 = even0 + even1 + level_shift_8;
    out1 = odd_diff1 + odd_diff4;
    out2 = even3 + even_diff * 0.707106781f;
    out3 = odd_diff3 - odd_diff2;
    out4 = even0 - even1;
    out5 = odd_diff3 + odd_diff2;
    out6 = even3 - even_diff * 0.707106781f;
    out7 = odd_diff1 - odd_diff4;
}

/** Constant memory copy of transposed quantization table pre-divided with DCT output weights. */
__constant__ float gpujpeg_dct_gpu_quantization_table_const[64];

/**
 * Performs 8x8 block-wise Forward Discrete Cosine Transform of the given
 * image plane and outputs result to the array of coefficients. Short implementation.
 * This kernel is designed to process image by blocks of blocks8x8 that
 * utilize maximum warps capacity, assuming that it is enough of 8 threads
 * per block8x8.
 *
 * @param source        [IN]  - Source coefficients
 * @param source_stride [IN]  - Stride of source
 * @param output        [OUT] - Source coefficients
 * @param output_stride [OUT] - Stride of source
 * @param quant_table   [IN]  - Quantization table, pre-divided with DCT output scales
 * @return None
 */
template <int WARP_COUNT>
__global__ void
gpujpeg_dct_gpu_kernel(int block_count_x, int block_count_y, uint8_t* source, const unsigned int source_stride,
                       int16_t* output, int output_stride, const float * const quant_table)
{
    // each warp processes 4 8x8 blocks (horizontally neighboring)
    const int block_idx_x = threadIdx.x >> 3;
    const int block_idx_y = threadIdx.y;

    // offset of threadblocks's blocks in the image (along both axes)
    const int block_offset_x = blockIdx.x * 4;
    const int block_offset_y = blockIdx.y * WARP_COUNT;

    // stop if thread's block is out of image
    const bool processing = block_offset_x + block_idx_x < block_count_x
                         && block_offset_y + block_idx_y < block_count_y;
    if(!processing) {
        return;
    }

    // index of row/column processed by this thread within its 8x8 block
    const int dct_idx = threadIdx.x & 7;

    // data type of transformed coefficients
    typedef float dct_t;

    // dimensions of shared buffer (compile time constants)
    enum {
        // 4 8x8 blocks, padded to odd number of 4byte banks
        SHARED_STRIDE = ((32 * sizeof(dct_t)) | 4) / sizeof(dct_t),

        // number of shared buffer items needed for 1 warp
        SHARED_SIZE_WARP = SHARED_STRIDE * 8,

        // total number of items in shared buffer
        SHARED_SIZE_TOTAL = SHARED_SIZE_WARP * WARP_COUNT
    };

    // buffer for transpositions of all blocks
    __shared__ dct_t s_transposition_all[SHARED_SIZE_TOTAL];

    // pointer to begin of transposition buffer for thread's block
    dct_t * const s_transposition = s_transposition_all + block_idx_y * SHARED_SIZE_WARP + block_idx_x * 8;

    // input coefficients pointer (each thread loads 1 column of 8 coefficients from its 8x8 block)
    const int in_x = (block_offset_x + block_idx_x) * 8 + dct_idx;
    const int in_y = (block_offset_y + block_idx_y) * 8;
    const int in_offset = in_x + in_y * source_stride;
    const uint8_t * in = source + in_offset;

    // load all 8 coefficients of thread's column, but do NOT apply level shift now - will be applied as part of DCT
    dct_t src0 = *in;
    in += source_stride;
    dct_t src1 = *in;
    in += source_stride;
    dct_t src2 = *in;
    in += source_stride;
    dct_t src3 = *in;
    in += source_stride;
    dct_t src4 = *in;
    in += source_stride;
    dct_t src5 = *in;
    in += source_stride;
    dct_t src6 = *in;
    in += source_stride;
    dct_t src7 = *in;

    // destination pointer into shared transpose buffer (each thread saves one column)
    dct_t * const s_dest = s_transposition + dct_idx;

    // transform the column (vertically) and save it into the transpose buffer
    gpujpeg_dct_gpu(src0, src1, src2, src3, src4, src5, src6, src7,
                    s_dest[SHARED_STRIDE * 0],
                    s_dest[SHARED_STRIDE * 1],
                    s_dest[SHARED_STRIDE * 2],
                    s_dest[SHARED_STRIDE * 3],
                    s_dest[SHARED_STRIDE * 4],
                    s_dest[SHARED_STRIDE * 5],
                    s_dest[SHARED_STRIDE * 6],
                    s_dest[SHARED_STRIDE * 7],
                    -1024.0f  // = 8 * -128 ... level shift sum for all 8 coefficients
    );

    // read coefficients back - each thread reads one row (no need to sync - only threads within same warp work on each block)
    // ... and transform the row horizontally
    volatile dct_t * s_src = s_transposition + SHARED_STRIDE * dct_idx;
    dct_t dct0, dct1, dct2, dct3, dct4, dct5, dct6, dct7;
    gpujpeg_dct_gpu(s_src[0], s_src[1], s_src[2], s_src[3], s_src[4], s_src[5], s_src[6], s_src[7],
                    dct0, dct1, dct2, dct3, dct4, dct5, dct6, dct7);

    // apply quantization to the row of coefficients (quantization table is actually transposed in global memory for coalesced memory acceses)
    #if __CUDA_ARCH__ < 200
    const float * const quantization_row = gpujpeg_dct_gpu_quantization_table_const + dct_idx; // Quantization table in constant memory for CCs < 2.0
    #else
    const float * const quantization_row = quant_table + dct_idx; // Cached global memory reads for CCs >= 2.0
    #endif
    const int out0 = rintf(dct0 * quantization_row[0 * 8]);
    const int out1 = rintf(dct1 * quantization_row[1 * 8]);
    const int out2 = rintf(dct2 * quantization_row[2 * 8]);
    const int out3 = rintf(dct3 * quantization_row[3 * 8]);
    const int out4 = rintf(dct4 * quantization_row[4 * 8]);
    const int out5 = rintf(dct5 * quantization_row[5 * 8]);
    const int out6 = rintf(dct6 * quantization_row[6 * 8]);
    const int out7 = rintf(dct7 * quantization_row[7 * 8]);

    // using single write, save output row packed into 16 bytes
    const int out_x = (block_offset_x + block_idx_x) * 64; // 64 coefficients per one transformed and quantized block
    const int out_y = (block_offset_y + block_idx_y) * output_stride;
    ((uint4*)(output + out_x + out_y))[dct_idx] = make_uint4(
        (out0 & 0xFFFF) + (out1 << 16),
        (out2 & 0xFFFF) + (out3 << 16),
        (out4 & 0xFFFF) + (out5 << 16),  // ... & 0xFFFF keeps only lower 16 bits - useful for negative numbers, which have 1s in upper bits
        (out6 & 0xFFFF) + (out7 << 16)
    );
}

/** Quantization table */
__constant__ uint16_t gpujpeg_idct_gpu_quantization_table[64];

/**
 * Performs 8x8 block-wise Inverse Discrete Cosine Transform of the given
 * image plane and outputs result to the array of coefficients. Short implementation.
 * This kernel is designed to process image by blocks of blocks8x8 that
 * utilize maximum warps capacity, assuming that it is enough of 8 threads
 * per block8x8.
 *
 * @param source        [IN]  - Source coefficients
 * @param source_stride [IN]  - Stride of source
 * @param output        [OUT] - Source coefficients
 * @param output_stride [OUT] - Stride of source
 * @param table         [IN]  - Quantization table
 * @return None
 */
__global__ void
gpujpeg_idct_gpu_kernel(int block_count_x, int block_count_y, int16_t* source, int source_stride,
                        uint8_t* output, int output_stride, uint16_t* quantization_table)
{
// For pre-fermi GPUs, quantization table in constant memory is faster
#if __CUDA_ARCH__ < 200
    quantization_table = gpujpeg_idct_gpu_quantization_table;
#endif

    // Shared data
    __shared__ int16_t block[GPUJPEG_DCT_THREAD_BLOCK_HEIGHT * GPUJPEG_DCT_THREAD_BLOCK_STRIDE];

    // Block position
    int block_x = IMAD(blockIdx.x, GPUJPEG_DCT_BLOCK_COUNT_X, threadIdx.y);
    int block_y = IMAD(blockIdx.y, GPUJPEG_DCT_BLOCK_COUNT_Y, threadIdx.z);

    // Thread position in thread block
    int thread_x = IMAD(threadIdx.y, GPUJPEG_BLOCK_SIZE, threadIdx.x);
    int thread_y = IMUL(threadIdx.z, GPUJPEG_BLOCK_SIZE);
    int thread_x_permutated = (thread_x & 0xFFFFFFE0) | (((thread_x << 1) | ((thread_x >> 4) & 0x1)) & 0x1F);

    // Determine position into shared buffer
    int16_t* block_ptr = block + IMAD(thread_y, GPUJPEG_DCT_THREAD_BLOCK_STRIDE, thread_x);

    // Determine position in source buffer and apply it
    int source_x = IMAD(block_x, GPUJPEG_BLOCK_SQUARED_SIZE, threadIdx.x * 2);
    int source_y = block_y;
    source += IMAD(source_y, source_stride, source_x);

    // Load data to shared memory, only half of threads in each cell performs data moving (each thread moves 2 shorts)
    if ( block_x < block_count_x && block_y < block_count_y ) {
        int16_t* block_load_ptr = block_ptr + threadIdx.x; // Shortcut for "IMAD(..., threadIdx.x * 2)"
        if ( threadIdx.x < (GPUJPEG_BLOCK_SIZE / 2) ) {
            #pragma unroll
            for(int i = 0; i < GPUJPEG_BLOCK_SIZE; i++)
                ((int*)block_load_ptr)[i * (GPUJPEG_DCT_THREAD_BLOCK_STRIDE / 2)] = ((int*)source)[i * (GPUJPEG_BLOCK_SIZE / 2)];
        }
    }
    __syncthreads();

    // Quantization
    for(int i = 0; i < GPUJPEG_BLOCK_SIZE; i++) {
        int16_t quantization = quantization_table[i * GPUJPEG_BLOCK_SIZE + threadIdx.x];
        int16_t coefficient = block_ptr[i * GPUJPEG_DCT_THREAD_BLOCK_STRIDE];

        coefficient = coefficient * quantization;

        block_ptr[i * GPUJPEG_DCT_THREAD_BLOCK_STRIDE] = coefficient;
    }

    // Perform IDCT
    __syncthreads();
    gpujpeg_idct_gpu_kernel_inplace(block + thread_y * GPUJPEG_DCT_THREAD_BLOCK_STRIDE + thread_x_permutated, GPUJPEG_DCT_THREAD_BLOCK_STRIDE);
    __syncthreads();
    gpujpeg_idct_gpu_kernel_inplace((uint32_t*)(block + (thread_y + threadIdx.x) * GPUJPEG_DCT_THREAD_BLOCK_STRIDE + threadIdx.y * GPUJPEG_BLOCK_SIZE));
    __syncthreads();

     // Determine position in output buffer and apply it
    int output_x = IMAD(blockIdx.x, GPUJPEG_DCT_THREAD_BLOCK_WIDTH, thread_x);
    int output_y = IMAD(blockIdx.y, GPUJPEG_DCT_THREAD_BLOCK_HEIGHT, thread_y);
    output += IMAD(output_y, output_stride, output_x);

// For pre-fermi GPUs, storing to global memory by 4 bytes is faster
#if __CUDA_ARCH__ < 200
    __shared__ uint8_t block_byte[GPUJPEG_DCT_THREAD_BLOCK_HEIGHT * GPUJPEG_DCT_THREAD_BLOCK_STRIDE];
    uint8_t* block_byte_ptr = block_byte + IMAD(thread_y, GPUJPEG_DCT_THREAD_BLOCK_STRIDE, thread_x);
    uint8_t* __output = output;
    int __output_stride = output_stride;
    output = block_byte_ptr;
    output_stride = GPUJPEG_DCT_THREAD_BLOCK_STRIDE;
#endif

    // Store data to global memory
    if ( block_x < block_count_x && block_y < block_count_y ) {
        #pragma unroll
        for(int i = 0; i < GPUJPEG_BLOCK_SIZE; i++) {
            int16_t coefficient = block_ptr[i * GPUJPEG_DCT_THREAD_BLOCK_STRIDE];
            coefficient += 128;
            if ( coefficient > 255 )
                coefficient = 255;
            if ( coefficient < 0 )
                coefficient = 0;
            output[i * output_stride] = (uint8_t)coefficient;
        }

// For pre-fermi GPUs, storing to global memory by 4 bytes is faster
#if __CUDA_ARCH__ < 200
        if ( threadIdx.x % 4 == 0 ) {
            #pragma unroll
            for(int i = 0; i < GPUJPEG_BLOCK_SIZE; i++)
                ((uint32_t*)__output)[i * (__output_stride / 4)] = ((uint32_t*)block_byte_ptr)[i * (GPUJPEG_DCT_THREAD_BLOCK_STRIDE / 4)];
        }
#endif
    }
}

/** Documented at declaration */
int
gpujpeg_dct_gpu(struct gpujpeg_encoder* encoder)
{
    // Get coder
    struct gpujpeg_coder* coder = &encoder->coder;

    // Encode each component
    for ( int comp = 0; comp < coder->param_image.comp_count; comp++ ) {
        // Get component
        struct gpujpeg_component* component = &coder->component[comp];

        // Get quantization table
        enum gpujpeg_component_type type = (comp == 0) ? GPUJPEG_COMPONENT_LUMINANCE : GPUJPEG_COMPONENT_CHROMINANCE;
        const float* const d_quantization_table = encoder->table_quantization[type].d_table_forward;

        // copy the quantization table into constant memory for devices of CC < 2.0
        if( encoder->coder.cuda_cc_major < 2 ) {
            cudaMemcpyToSymbolAsync(
                gpujpeg_dct_gpu_quantization_table_const,
                d_quantization_table,
                sizeof(gpujpeg_dct_gpu_quantization_table_const),
                0,
                cudaMemcpyDeviceToDevice,
                *(encoder->stream)
            );
            gpujpeg_cuda_check_error("Quantization table memcpy failed", return -1);
        }

        int roi_width = component->data_width;
        int roi_height = component->data_height;
        assert(GPUJPEG_BLOCK_SIZE == 8);

        int block_count_x = roi_width / GPUJPEG_BLOCK_SIZE;
        int block_count_y = roi_height / GPUJPEG_BLOCK_SIZE;

        enum { WARP_COUNT = 4 };

        // Perform block-wise DCT processing
        dim3 dct_grid(
            gpujpeg_div_and_round_up(block_count_x, 4),
            gpujpeg_div_and_round_up(block_count_y, WARP_COUNT),
            1
        );
        dim3 dct_block(4 * 8, WARP_COUNT);
        gpujpeg_dct_gpu_kernel<WARP_COUNT><<<dct_grid, dct_block, 0, *(encoder->stream)>>>(
            block_count_x,
            block_count_y,
            component->d_data,
            component->data_width,
            component->d_data_quantized,
            component->data_width * GPUJPEG_BLOCK_SIZE,
            d_quantization_table
        );
        gpujpeg_cuda_check_error("Quantization table memcpy failed", return -1);
    }

    return 0;
}

/** Documented at declaration */
int
gpujpeg_idct_gpu(struct gpujpeg_decoder* decoder)
{
    // Get coder
    struct gpujpeg_coder* coder = &decoder->coder;

    // Encode each component
    for ( int comp = 0; comp < coder->param_image.comp_count; comp++ ) {
        // Get component
        struct gpujpeg_component* component = &coder->component[comp];

        // Determine table type
        enum gpujpeg_component_type type = (comp == 0) ? GPUJPEG_COMPONENT_LUMINANCE : GPUJPEG_COMPONENT_CHROMINANCE;

        int roi_width = component->data_width;
        int roi_height = component->data_height;
        assert(GPUJPEG_BLOCK_SIZE == 8);

        int block_count_x = roi_width / GPUJPEG_BLOCK_SIZE;
        int block_count_y = roi_height / GPUJPEG_BLOCK_SIZE;

        // Get quantization table
        uint16_t* d_quantization_table = decoder->table_quantization[type].d_table;

        // Copy quantization table to constant memory
        cudaMemcpyToSymbolAsync(
            gpujpeg_idct_gpu_quantization_table,
            d_quantization_table,
            64 * sizeof(uint16_t),
            0,
            cudaMemcpyDeviceToDevice,
            *(decoder->stream)
        );
        gpujpeg_cuda_check_error("Copy IDCT quantization table to constant memory", return -1);

        // Perform block-wise IDCT processing
        dim3 dct_grid(
            gpujpeg_div_and_round_up(block_count_x, GPUJPEG_DCT_BLOCK_COUNT_X),
            gpujpeg_div_and_round_up(block_count_y, GPUJPEG_DCT_BLOCK_COUNT_Y),
            1
        );
        dim3 dct_block(
            GPUJPEG_BLOCK_SIZE,
            GPUJPEG_DCT_BLOCK_COUNT_X,
            GPUJPEG_DCT_BLOCK_COUNT_Y
        );
        gpujpeg_idct_gpu_kernel<<<dct_grid, dct_block, 0, *(decoder->stream)>>>(
            block_count_x,
            block_count_y,
            component->d_data_quantized,
            component->data_width * GPUJPEG_BLOCK_SIZE,
            component->d_data,
            component->data_width,
            d_quantization_table
        );
        gpujpeg_cuda_check_error("Inverse Integer DCT failed", return -1);
    }

    return 0;
}
