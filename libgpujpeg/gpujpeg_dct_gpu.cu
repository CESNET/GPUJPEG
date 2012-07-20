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
#include "gpujpeg_util.h"

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


template <typename T>
__device__ static inline void
dct(const T in0, const T in1, const T in2, const T in3, const T in4, const T in5, const T in6, const T in7,
    volatile T & out0, volatile T & out1, volatile T & out2, volatile T & out3, volatile T & out4, volatile T & out5, volatile T & out6, volatile T & out7)
{
//     const int tmp0 = in7 + in0;
//     const int tmp1 = in6 + in1;
//     const int tmp2 = in5 + in2;
//     const int tmp3 = in4 + in3;
//     const int tmp4 = in3 - in4;
//     const int tmp5 = in2 - in5;
//     const int tmp6 = in1 - in6;
//     const int tmp7 = in0 - in7;
// 
//     const int tmp10 = tmp3 + tmp0;
//     const int tmp11 = tmp2 + tmp1;
//     const int tmp12 = tmp1 - tmp2;
//     const int tmp13 = tmp0 - tmp3;
// 
//     const int tmp16 = unfixo(FMUL(tmp6 + tmp5, SIN_1_4));
//     const int tmp15 = unfixo(FMUL(tmp6 - tmp5, COS_1_4));
// 
//     const int tmp4b = tmp4 << 2;
//     const int tmp7b = tmp7 << 2;
// 
//     const int tmp14 = tmp4b + tmp15;
//     const int tmp25 = tmp4b - tmp15;
//     const int tmp26 = tmp7b - tmp16;
//     const int tmp17 = tmp7b + tmp16;
//     
//     out0 = unfixh(FMUL(tmp10 + tmp11, SIN_1_4));
//     out1 = unfixh(FMUL(tmp17, OCOS_1_16) + FMUL(tmp14, OSIN_1_16));
//     out2 = unfixh(FMUL(tmp13, COS_1_8) + FMUL(tmp12, SIN_1_8));
//     out3 = unfixh(FMUL(tmp26, OCOS_3_16) - FMUL(tmp25, OSIN_3_16));
//     out4 = unfixh(FMUL(tmp10 - tmp11, COS_1_4));
//     out5 = unfixh(FMUL(tmp26, OCOS_5_16) + FMUL(tmp25, OSIN_5_16));
//     out6 = unfixh(FMUL(tmp13, SIN_1_8) - FMUL(tmp12, COS_1_8));
//     out7 = unfixh(FMUL(tmp17, OCOS_7_16) - FMUL(tmp14, OSIN_7_16));


    const float scale0 = 0.353553390593274f; // sin(pi / 4) / 2
    const float scale1 = 0.509795579104159f; // 1 / (2 * sin(7 * pi / 16))
    const float scale2 = 0.541196100146197f; // 1 / (2 * sin(3 * pi / 8))
    const float scale3 = 0.601344886935045f; // 1 / (2 * cos(3 * pi / 16))
    const float scale4 = 0.707106781186547f; // sin(pi / 4)
    const float scale5 = 0.415734806151273f; // cos(3 * pi / 16) / 2
    const float scale6 = 0.461939766255643f; // sin(3 * pi / 8) / 2
    const float scale7 = 0.490392640201615f; // sin(7 * pi / 16) / 2
    
    const float p1 = 0.4142135623f;
    const float p2 = 0.6681786379f;
    const float p3 = 0.1989123673f;
    const float p4 = 0.4142135623f;
    const float p5 = 0.4142135623f;
    const float u1 = 0.3535533905f;
    const float u2 = 0.4619397662f;
    const float u3 = 0.1913417161f;
    const float u4 = 0.7071067811f;
    
    float a0 = in7 + in0;
    float a1 = in6 + in1;
    float a2 = in5 + in2;
    float a3 = in4 + in3;
    float a4 = in3 - in4;
    float a5 = in2 - in5;
    float a6 = in1 - in6;
    float a7 = in0 - in7;
    
    a5 = a5 - a6 * p4;
    a6 = a6 + a5 * u4;
    a5 = a6 * p5 - a5;
    
    float b0 = a0 + a3;
    float b1 = a1 + a2;
    float b2 = a1 - a2;
    float b3 = a0 - a3;
    float b4 = a4 + a5;
    float b5 = a4 - a5;
    float b6 = a7 - a6;
    float b7 = a7 + a6;
    
    b0 = b0 + b1;
    b1 = 0.5f * b0 - b1;
    
    b2 = p1 * b3 - b2;
    b3 = b3 - u1 * b2;
    
    b4 = p3 * b7 - b4;
    b7 = b7 - u3 * b4;
    
    b5 = b5 + p2 * b6;
    b6 = b6 - u2 * b5;
    
    out0 = b0 * scale0;
    out1 = b7 * scale7;
    out2 = b3 * scale3;
    out3 = b6 * scale6;
    out4 = b1 * scale1;
    out5 = b5 * scale5;
    out6 = b2 * scale2;
    out7 = b4 * scale4;
}


/**
 * Performs in-place DCT of vector of 8 elements (used to access columns in shared memory).
 *
 * @param SrcDst [IN/OUT] - Pointer to the first element of vector
 * @param Stride [IN] - Value to add to ptr to access other elements
 * @return None
 */
__device__ void
gpujpeg_dct_gpu_kernel_inplace(float* SrcDst, int Stride)
{
    dct(SrcDst[Stride * 0], SrcDst[Stride * 1], SrcDst[Stride * 2], SrcDst[Stride * 3],
        SrcDst[Stride * 4], SrcDst[Stride * 5], SrcDst[Stride * 6], SrcDst[Stride * 7],
        SrcDst[Stride * 0], SrcDst[Stride * 1], SrcDst[Stride * 2], SrcDst[Stride * 3],
        SrcDst[Stride * 4], SrcDst[Stride * 5], SrcDst[Stride * 6], SrcDst[Stride * 7]);
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

/** Quantization table */
__constant__ uint16_t gpujpeg_dct_gpu_quantization_table[64];

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
 * @param table         [IN]  - Quantization table
 * @return None
 */
template <int WARP_COUNT>
__global__ void
gpujpeg_dct_gpu_kernel(int block_count_x, int block_count_y, uint8_t* source, int source_stride,
                       int16_t* output, int output_stride, uint16_t* quantization_table)
{
    // each warp processes 4 8x8 blocks (horizontally neighboring)
    const int block_idx_x = threadIdx.x >> 3;
    const int block_idx_y = threadIdx.y;
    
    // offset of threadblocks's blocks in the image (along both axes)
    const int block_offset_x = blockIdx.x * 4;
    const int block_offset_y = blockIdx.y * WARP_COUNT;
    
    // true if thread's block is not out of image
    const bool processing = block_offset_x + block_idx_x < block_count_x
                         && block_offset_y + block_idx_y < block_count_y;
    
    // stop if out of block range
    if(!processing) {
        return;
    }
    
    // index of row/column processed by this thread within its 8x8 block
    const int dct_idx = threadIdx.x & 7;
    
    
    
    // Data type of transformed coefficients
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
    __shared__ volatile dct_t s_transposition_all[SHARED_SIZE_TOTAL];
    
    // pointer to begin of transposition buffer for thread's block
    volatile dct_t * const s_transposition = s_transposition_all + block_idx_y * SHARED_SIZE_WARP + block_idx_x * 8;
    
    
    
    
    
    
    // Load input coefficients (each thread loads 1 row of 8 coefficients from its 8x8 block)
    const int in_x = (block_offset_x + block_idx_x) * 8;
    const int in_y = (block_offset_y + block_idx_y) * 8 + dct_idx;
    const int in_offset = in_x + in_y * source_stride;
    const ushort4 in = *((ushort4*)(source + in_offset));
    
    // separate input coefficients and apply level shift (assuming little endian hardware)
    dct_t dct0 = (in.x & 0xFF) - 128;
    dct_t dct1 = (in.x >> 8) - 128;
    dct_t dct2 = (in.y & 0xFF) - 128;
    dct_t dct3 = (in.y >> 8) - 128;
    dct_t dct4 = (in.z & 0xFF) - 128;
    dct_t dct5 = (in.z >> 8) - 128;
    dct_t dct6 = (in.w & 0xFF) - 128;
    dct_t dct7 = (in.w >> 8) - 128;
    
    
    
    
    
    // repeat "1D transform and transpose" two times
    #pragma unroll
    for(int i = 0; i < 2; i++) {
        // destination pointer into shared transpose buffer (each thread saves one column)
        volatile dct_t * const s_dest = s_transposition + dct_idx;
        
        dct(dct0, dct1, dct2, dct3, dct4, dct5, dct6, dct7,
            s_dest[SHARED_STRIDE * 0],
            s_dest[SHARED_STRIDE * 1],
            s_dest[SHARED_STRIDE * 2],
            s_dest[SHARED_STRIDE * 3],
            s_dest[SHARED_STRIDE * 4],
            s_dest[SHARED_STRIDE * 5],
            s_dest[SHARED_STRIDE * 6],
            s_dest[SHARED_STRIDE * 7]
        );
        
        // read coefficients back - each thread read one row (no need to sync - only thread within same warp work on each block)
        volatile dct_t * s_src = s_transposition + SHARED_STRIDE * dct_idx;
        dct0 = s_src[0];
        dct1 = s_src[1];
        dct2 = s_src[2];
        dct3 = s_src[3];
        dct4 = s_src[4];
        dct5 = s_src[5];
        dct6 = s_src[6];
        dct7 = s_src[7];
    }
    
    // apply qunatzation to the row of coefficients
    const uint16_t * const quantization_row = quantization_table + 8 * dct_idx;
    dct0 *= quantization_row[0] * 3.05185094759972e-05f;
    dct1 *= quantization_row[1] * 3.05185094759972e-05f;
    dct2 *= quantization_row[2] * 3.05185094759972e-05f;
    dct3 *= quantization_row[3] * 3.05185094759972e-05f;
    dct4 *= quantization_row[4] * 3.05185094759972e-05f;
    dct5 *= quantization_row[5] * 3.05185094759972e-05f;
    dct6 *= quantization_row[6] * 3.05185094759972e-05f;
    dct7 *= quantization_row[7] * 3.05185094759972e-05f;
    
    // save output row packed into 16 bytes
    const int out_x = (block_offset_x + block_idx_x) * 64; // 64 coefficients per one transformed and quantized block
    const int out_y = (block_offset_y + block_idx_y) * output_stride;
    ((uint4*)(output + out_x + out_y))[dct_idx] = make_uint4(
        (int)dct0 + 0x10000 * (int)dct1,
        (int)dct2 + 0x10000 * (int)dct3,
        (int)dct4 + 0x10000 * (int)dct5,
        (int)dct6 + 0x10000 * (int)dct7
    );
    
    
    
    
//     // Shared data
//     __shared__ float block[GPUJPEG_DCT_THREAD_BLOCK_HEIGHT * GPUJPEG_DCT_THREAD_BLOCK_STRIDE];
// 
//     // Block position
//     int block_x = IMAD(blockIdx.x, GPUJPEG_DCT_BLOCK_COUNT_X, threadIdx.y);
//     int block_y = IMAD(blockIdx.y, GPUJPEG_DCT_BLOCK_COUNT_Y, threadIdx.z);
// 
//     // Thread position in thread block
//     int thread_x = IMAD(threadIdx.y, GPUJPEG_BLOCK_SIZE, threadIdx.x);
//     int thread_y = IMUL(threadIdx.z, GPUJPEG_BLOCK_SIZE);
//     int thread_x_permutated = (thread_x & 0xFFFFFFE0) | (((thread_x << 1) | ((thread_x >> 4) & 0x1)) & 0x1F);
// 
//     // Determine position into shared buffer
//     float* block_ptr = block + IMAD(thread_y, GPUJPEG_DCT_THREAD_BLOCK_STRIDE, thread_x);
// 
//     // Determine position in source buffer and apply it
//     int source_x = IMAD(block_x, GPUJPEG_BLOCK_SIZE, threadIdx.x);
//     int source_y = IMUL(block_y, GPUJPEG_BLOCK_SIZE);
//     source += IMAD(source_y, source_stride, source_x);
// 
//     // Load data to shared memory memory
//     if ( block_x < block_count_x && block_y < block_count_y ) {
//         #pragma unroll
//         for(int i = 0; i < GPUJPEG_BLOCK_SIZE; i++) {
//             float coefficient = (int16_t)(source[i * source_stride]);
//             coefficient -= 128.0f;
//             block_ptr[i * GPUJPEG_DCT_THREAD_BLOCK_STRIDE] = coefficient;
//         }
//     }
// 
//     // Perform DCT
//     __syncthreads();
//     gpujpeg_dct_gpu_kernel_inplace(block + thread_y * GPUJPEG_DCT_THREAD_BLOCK_STRIDE + thread_x_permutated, GPUJPEG_DCT_THREAD_BLOCK_STRIDE);
//     __syncthreads();
//     gpujpeg_dct_gpu_kernel_inplace(block + (thread_y + threadIdx.x) * GPUJPEG_DCT_THREAD_BLOCK_STRIDE + threadIdx.y * GPUJPEG_BLOCK_SIZE, 1);
//     __syncthreads();
// 
//     // Quantization
//     for(int i = 0; i < GPUJPEG_BLOCK_SIZE; i++) {
//         float quantization = (quantization_table[i * GPUJPEG_BLOCK_SIZE + threadIdx.x]) / 32767.0f;
//         float coefficient = block_ptr[i * GPUJPEG_DCT_THREAD_BLOCK_STRIDE];
//         block_ptr[i * GPUJPEG_DCT_THREAD_BLOCK_STRIDE] = coefficient * quantization;
//     }
//     __syncthreads();
// 
//     // Determine position in output buffer and apply it
//     int output_x = IMAD(IMAD(blockIdx.x, GPUJPEG_DCT_BLOCK_COUNT_X, threadIdx.y), GPUJPEG_BLOCK_SQUARED_SIZE, threadIdx.x);
//     int output_y = IMAD(blockIdx.y, GPUJPEG_DCT_BLOCK_COUNT_Y, threadIdx.z);
//     output += IMAD(output_y, output_stride, output_x);
// 
//     // Store data to global memory
//     if ( block_x < block_count_x && block_y < block_count_y ) {
//         #pragma unroll
//         for(int i = 0; i < GPUJPEG_BLOCK_SIZE; i++)
//             output[i * GPUJPEG_BLOCK_SIZE] = round(block_ptr[i * GPUJPEG_DCT_THREAD_BLOCK_STRIDE]);
//     }
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
void
gpujpeg_dct_gpu(struct gpujpeg_encoder* encoder)
{
    // Get coder
    struct gpujpeg_coder* coder = &encoder->coder;

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
        uint16_t* d_quantization_table = encoder->table_quantization[type].d_table;
        
        // Copy quantization table to constant memory
        cudaMemcpyToSymbol(
            (const char*)gpujpeg_dct_gpu_quantization_table,
            d_quantization_table, 
            64 * sizeof(uint16_t),
            0,
            cudaMemcpyDeviceToDevice
        );
        gpujpeg_cuda_check_error("Copy DCT quantization table to constant memory");

        enum { WARP_COUNT = 4 };
        
        // Perform block-wise DCT processing
        dim3 dct_grid(
            gpujpeg_div_and_round_up(block_count_x, 4),
            gpujpeg_div_and_round_up(block_count_y, WARP_COUNT),
            1
        );
        dim3 dct_block(4 * 8, WARP_COUNT);
        gpujpeg_dct_gpu_kernel<WARP_COUNT><<<dct_grid, dct_block>>>(
            block_count_x,
            block_count_y,
            component->d_data,
            component->data_width,
            component->d_data_quantized,
            component->data_width * GPUJPEG_BLOCK_SIZE,
            d_quantization_table
        );
        cudaThreadSynchronize();
        gpujpeg_cuda_check_error("Forward Integer DCT failed");
    }
}

/** Documented at declaration */
void
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
        cudaMemcpyToSymbol(
            (const char*)gpujpeg_idct_gpu_quantization_table,
            d_quantization_table, 
            64 * sizeof(uint16_t),
            0,
            cudaMemcpyDeviceToDevice
        );
        gpujpeg_cuda_check_error("Copy IDCT quantization table to constant memory");

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
        gpujpeg_idct_gpu_kernel<<<dct_grid, dct_block>>>(
            block_count_x,
            block_count_y,
            component->d_data_quantized,
            component->data_width * GPUJPEG_BLOCK_SIZE,
            component->d_data,
            component->data_width,
            d_quantization_table
        );
        cudaThreadSynchronize();
        gpujpeg_cuda_check_error("Inverse Integer DCT failed");
    }
}
