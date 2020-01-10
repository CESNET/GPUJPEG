/**
 * @file
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

//PTX code for IDCT (GPUJPEG_IDCT_GPU_KERNEL_INPLACE macro) should be a bit faster
//but maybe won't work for newer CCs
#define GPUJPEG_IDCT_USE_ASM 0

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
//TODO zmenit na float
__constant__ uint16_t gpujpeg_idct_gpu_quantization_table[64];

#if !GPUJPEG_IDCT_USE_ASM

/**
 * Performs in-place IDCT of vector of 8 elements (used to access rows 
 * or columns in a vector).
 * With a use of a scheme presented in Jie Liang - Approximating the DCT 
 * with the lifting scheme: systematic design and applications; online:
 * http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=910943
 *
 * @param V8 [IN/OUT] - Pointer to the first element of vector
 * @return None
 */
__device__ void
gpujpeg_idct_gpu_kernel_inplace(float* V8)
{
	//costants which are used more than once
	const float koeficient[6] = {0.4142135623f, 0.3535533905f, 0.4619397662f, 0.1989123673f, 0.7071067811f, -2.0f};
	
	V8[2] *= 0.5411961f;
	V8[4] *= 0.509795579f;
	V8[5] *= 0.601344887f;
	
	V8[1] = (V8[0] - V8[1]) * koeficient[1];
	V8[0] = V8[0] * koeficient[4] - V8[1];

	V8[3] = V8[2] * koeficient[1] + V8[3] * koeficient[2];
	V8[2] = V8[3] * koeficient[0] - V8[2];

	V8[6] = V8[5] * koeficient[2] + V8[6] * koeficient[0];
	V8[5] = -0.6681786379f * V8[6] + V8[5];

	V8[7] = V8[4] * koeficient[3] + V8[7] * 0.49039264f;
	V8[4] = V8[7] * koeficient[3] - V8[4];

	//instead of float tmp = V8[1]; V8[1] = V8[2] + V8[1]; V8[2] = tmp - V8[2];
	//we use this two operations (with a use of a multiply-add instruction)
	V8[1] = V8[2] + V8[1];
	V8[2] = koeficient[5] * V8[2] + V8[1];

	V8[4] = V8[5] + V8[4];
	V8[5] = 2.0f * V8[5] - V8[4];

	V8[7] = V8[6] + V8[7];
	V8[6] = koeficient[5] * V8[6] + V8[7];

	V8[0] = V8[3] + V8[0];
	V8[3] = koeficient[5] * V8[3] + V8[0];

	V8[5] = V8[6] * koeficient[0] + V8[5];
	V8[6] = V8[5] * -koeficient[4] + V8[6];
	V8[5] = V8[6] * koeficient[0] + V8[5];

	V8[3] = V8[3] + V8[4];
	V8[4] = koeficient[5] * V8[4] + V8[3];

	V8[2] = V8[2] + V8[5];
	V8[5] = koeficient[5] * V8[5] + V8[2];

	V8[1] = V8[6] + V8[1];
	V8[6] = koeficient[5] * V8[6] + V8[1];

	V8[0] = V8[0] + V8[7];
	V8[7] = koeficient[5] * V8[7] + V8[0];
}
#else

#if __CUDA_ARCH__ >= 200
#define MULTIPLY_ADD "fma.rn.f32	"
#else
#define MULTIPLY_ADD "mad.f32	"
#endif

//instead of float tmp = V8[1]; V8[1] = V8[2] + V8[1]; V8[2] = tmp - V8[2];
//we use this two operations (with a use of a multiply-add instruction)
#define ASM_X_PLUS_Y_SIMULTANEOUSLY_WITH_X_MINUS_Y(x, y) \
	"add.f32	" #x ", " #x ", " #y ";	\n\t"	\
	MULTIPLY_ADD #y ", " #y ", 0fc0000000, " #x ";	\n\t"

/**
 * Performs in-place IDCT of 8 elements (rows or columns). A PTX implementation.
 * With a use of a scheme presented in Jie Liang - Approximating the DCT 
 * with the lifting scheme: systematic design and applications; online:
 * http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=910943
 */
#define GPUJPEG_IDCT_GPU_KERNEL_INPLACE(in0, in1, in2, in3, in4, in5, in6, in7, \
		out0, out1, out2, out3, out4, out5, out6, out7)	\
		asm( \
			/* negreg register used for negating variables (e.g. for */	\
			/* a * b - c we neagte c into negreg and use multiply-add) */	\
			"{.reg .f32 negreg;	\n\t"	\
\
			"mul.f32	%9, %9, 0fbeb504f3;	\n\t"	\
			MULTIPLY_ADD "%9, %8, 0f3eb504f3, %9;	\n\t"	\
			"neg.f32 	negreg, %9;	\n\t"	\
			MULTIPLY_ADD "%8, %8, 0f3f3504f3, negreg;	\n\t"	\
\
			"mul.f32	%10, %10, 0f3f0a8bd4;	\n\t"	\
			"mul.f32	%11, %11, 0f3eec835e;	\n\t"	\
			MULTIPLY_ADD "%11, %10, 0f3eb504f3, %11;	\n\t"	\
			"neg.f32	%10, %10;	\n\t"	\
			MULTIPLY_ADD "%10, %11, 0f3ed413cd, %10;	\n\t"	\
\
			"mul.f32	%13, %13, 0f3f19f1bd;	\n\t"	\
			"mul.f32	%14, %14, 0f3ed4db31;	\n\t"	\
			MULTIPLY_ADD "%14, %13, 0f3eec835e, %14;	\n\t"	\
			MULTIPLY_ADD "%13, %14, 0fbf2b0dc7, %13;	\n\t"	\
\
			"mul.f32	%12, %12, 0f3f0281f7;	\n\t"	\
			"mul.f32	%15, %15, 0f3efb14be;	\n\t"	\
			MULTIPLY_ADD "%15, %12, 0f3e4bafaf, %15;	\n\t"	\
			"neg.f32	%12, %12;	\n\t"	\
			MULTIPLY_ADD "%12, %15, 0f3e4bafaf, %12;	\n\t"	\
\
			ASM_X_PLUS_Y_SIMULTANEOUSLY_WITH_X_MINUS_Y(%9, %10) \
\
			ASM_X_PLUS_Y_SIMULTANEOUSLY_WITH_X_MINUS_Y(%12, %13) \
\
			ASM_X_PLUS_Y_SIMULTANEOUSLY_WITH_X_MINUS_Y(%8, %11) \
\
			ASM_X_PLUS_Y_SIMULTANEOUSLY_WITH_X_MINUS_Y(%15, %14) \
\
			MULTIPLY_ADD "%13, %14, 0fbed413db, %13;	\n\t"	\
			MULTIPLY_ADD "%14, %13, 0f3f3504f3, %14;	\n\t"	\
			"neg.f32 negreg, %13;	\n\t"	\
			MULTIPLY_ADD "%13, %14, 0f3ed413cd, negreg;	\n\t"	\
\
			/* writing into output registers */	\
			"add.f32	%3, %11, %12;	\n\t"	\
			"sub.f32	%4, %11, %12;	\n\t"	\
\
			"add.f32	%2, %10, %13;	\n\t"	\
			"sub.f32	%5, %10, %13;	\n\t"	\
\
			"add.f32	%1, %14, %9;	\n\t"	\
			"sub.f32	%6, %9, %14;	\n\t"	\
\
			"add.f32	%0, %8, %15;	\n\t"	\
			"sub.f32	%7, %8, %15;	\n\t"	\
			"}"	\
\
			: "=f"((out0)),	\
			"=f"((out1)),	\
			"=f"((out2)),	\
			"=f"((out3)),	\
			"=f"((out4)),	\
			"=f"((out5)),	\
			"=f"((out6)),	\
			"=f"((out7))	\
			: "f"((in0)),	\
			"f"((in1)),	\
			"f"((in2)),	\
			"f"((in3)),	\
			"f"((in4)),	\
			"f"((in5)),	\
			"f"((in6)),	\
			"f"((in7))	\
	);

#endif

/**
 * Performs 8x8 block-wise Inverse Discrete Cosine Transform of the given
 * image plane and outputs result to the array of coefficients. Float implementation.
 * This kernel is designed to process image by blocks of blocks8x8 that
 * utilize maximum warps capacity. Prepared for 8*8*2 threads in a block
 *
 * @param source             [IN]  - Source coefficients
 * @param output             [OUT] - Result coefficients
 * @param output_stride      [OUT] - Stride of result (image width)
 * @param quantization_table [IN]  - Quantization table
 * @return None
 */
__global__ void
gpujpeg_idct_gpu_kernel(int16_t* source, uint8_t* result, int output_stride, uint16_t* quantization_table)
{
	//here the grid is assumed to be only in x - it saves a few operations; if a larger
	//block count is used (e. g. GPUJPEG_IDCT_BLOCK_Z == 1), it would need to be adjusted,
	//the blockIdx.x not to exceed 65535. In the current state this function is good 
	//enough for a 67.1 MPix picture (8K is 33.1 MPix)

	//the first block of picture processed in this thread block
	unsigned int picBlockNumber = (blockIdx.x) * GPUJPEG_IDCT_BLOCK_Y * GPUJPEG_IDCT_BLOCK_X
			* GPUJPEG_IDCT_BLOCK_Z;

	//pointer to the begin of data for this thread block
	int16_t* sourcePtr = (int16_t*) (source) + picBlockNumber * 8;

	__shared__ float data[GPUJPEG_IDCT_BLOCK_Z][8][GPUJPEG_IDCT_BLOCK_Y][GPUJPEG_IDCT_BLOCK_X + 1];

	//variables to be used later more times (only one multiplication here)
	unsigned int z64 = threadIdx.z * 64;
	unsigned int x8 = threadIdx.x * 8;

	//data copying global -> shared, type casting int16_t -> float and dequantization.
	//16b reading gives only 50% efectivity but another ways are too complicated 
	//so this proves to be the fastest way
#pragma unroll
	for (int i = 0; i < 8; i++) {
		data[threadIdx.z][i][threadIdx.x][threadIdx.y] = sourcePtr[x8
				+ threadIdx.y + i * GPUJPEG_IDCT_BLOCK_X * GPUJPEG_IDCT_BLOCK_Y + z64 * 8]
				* quantization_table[threadIdx.x * 8 + threadIdx.y];
	}
	
	__syncthreads();

	float x[8];

	//kompilator delal hrozne psi kusy - zbytecne kopirovani konstant do
	//registru atp., bylo jednodussi napsat to v assembleru nez snazit se ho
	//presvedcit, aby nedelal blbosti; vsechny konstanty se pouzivaji primo
	//hodnotou, nestrkaji se zbytecne do registru

	//here the data are being processed by columns - each thread processes one column
#if GPUJPEG_IDCT_USE_ASM
	GPUJPEG_IDCT_GPU_KERNEL_INPLACE(data[threadIdx.z][threadIdx.x][0][threadIdx.y],
			data[threadIdx.z][threadIdx.x][4][threadIdx.y],
			data[threadIdx.z][threadIdx.x][6][threadIdx.y],
			data[threadIdx.z][threadIdx.x][2][threadIdx.y],
			data[threadIdx.z][threadIdx.x][7][threadIdx.y],
			data[threadIdx.z][threadIdx.x][5][threadIdx.y],
			data[threadIdx.z][threadIdx.x][3][threadIdx.y],
			data[threadIdx.z][threadIdx.x][1][threadIdx.y],

			data[threadIdx.z][threadIdx.x][0][threadIdx.y],
			data[threadIdx.z][threadIdx.x][1][threadIdx.y],
			data[threadIdx.z][threadIdx.x][2][threadIdx.y],
			data[threadIdx.z][threadIdx.x][3][threadIdx.y],
			data[threadIdx.z][threadIdx.x][4][threadIdx.y],
			data[threadIdx.z][threadIdx.x][5][threadIdx.y],
			data[threadIdx.z][threadIdx.x][6][threadIdx.y],
			data[threadIdx.z][threadIdx.x][7][threadIdx.y])
#else
	x[0] = data[threadIdx.z][threadIdx.x][0][threadIdx.y];
	x[1] = data[threadIdx.z][threadIdx.x][4][threadIdx.y];
	x[2] = data[threadIdx.z][threadIdx.x][6][threadIdx.y];
	x[3] = data[threadIdx.z][threadIdx.x][2][threadIdx.y];
	x[4] = data[threadIdx.z][threadIdx.x][7][threadIdx.y];
	x[5] = data[threadIdx.z][threadIdx.x][5][threadIdx.y];
	x[6] = data[threadIdx.z][threadIdx.x][3][threadIdx.y];
	x[7] = data[threadIdx.z][threadIdx.x][1][threadIdx.y];
	
	gpujpeg_idct_gpu_kernel_inplace(x);

	data[threadIdx.z][threadIdx.x][0][threadIdx.y] = x[0];
	data[threadIdx.z][threadIdx.x][1][threadIdx.y] = x[1];
	data[threadIdx.z][threadIdx.x][2][threadIdx.y] = x[2];
	data[threadIdx.z][threadIdx.x][3][threadIdx.y] = x[3];
	data[threadIdx.z][threadIdx.x][4][threadIdx.y] = x[4];
	data[threadIdx.z][threadIdx.x][5][threadIdx.y] = x[5];
	data[threadIdx.z][threadIdx.x][6][threadIdx.y] = x[6];
	data[threadIdx.z][threadIdx.x][7][threadIdx.y] = x[7];
#endif
	//between data writing and sync it's good to compute something useful 
	// - the sync will be shorter.
	
	//output pointer (the begin for this thread block)
	unsigned int firstByteOfActualBlock = x8 + z64 + picBlockNumber;

	//output pointer for this thread + output row shift; each thread writes 1 row of an 
	//output block (8B), threads [0 - 7] in threadIdx.x write blocks next to each other,
	//threads [1 - 7] in threadIdx.y write next rows of a block; threads [0 - 1] in 
	//threadIdx.z write next 8 blocks
	uint8_t* resultPtr = ((uint8_t*) result) + firstByteOfActualBlock
			+ (threadIdx.y + ((firstByteOfActualBlock / output_stride) * 7))
					* output_stride;

	__syncthreads();

#if GPUJPEG_IDCT_USE_ASM
	//here the data are being processed by rows - each thread processes one row
	GPUJPEG_IDCT_GPU_KERNEL_INPLACE(data[threadIdx.z][threadIdx.x][threadIdx.y][0],
			data[threadIdx.z][threadIdx.x][threadIdx.y][4],
			data[threadIdx.z][threadIdx.x][threadIdx.y][6],
			data[threadIdx.z][threadIdx.x][threadIdx.y][2],
			data[threadIdx.z][threadIdx.x][threadIdx.y][7],
			data[threadIdx.z][threadIdx.x][threadIdx.y][5],
			data[threadIdx.z][threadIdx.x][threadIdx.y][3],
			data[threadIdx.z][threadIdx.x][threadIdx.y][1],

			x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7])
#else
	x[0] = data[threadIdx.z][threadIdx.x][threadIdx.y][0];
	x[1] = data[threadIdx.z][threadIdx.x][threadIdx.y][4];
	x[2] = data[threadIdx.z][threadIdx.x][threadIdx.y][6];
	x[3] = data[threadIdx.z][threadIdx.x][threadIdx.y][2];
	x[4] = data[threadIdx.z][threadIdx.x][threadIdx.y][7];
	x[5] = data[threadIdx.z][threadIdx.x][threadIdx.y][5];
	x[6] = data[threadIdx.z][threadIdx.x][threadIdx.y][3];
	x[7] = data[threadIdx.z][threadIdx.x][threadIdx.y][1];

	gpujpeg_idct_gpu_kernel_inplace(x);
#endif

	//output will be written by 8B (one row) which is the most effective way
	uint64_t tempResult;
	uint64_t* tempResultP = &tempResult;

#pragma unroll
	for (int i = 0; i < 8; i++) {
		//this would be faster but will work only for 100% quality otherwise some values overflow 255
		//((uint8_t*) tempResultP)[i] = __float2uint_rz(x[i] + ((float) 128.0));

		//cast float to uint8_t with saturation (.sat) which cuts values higher than 
		//255 to 255 and smaller than 0 to 0; cuda can't use a reg smaller than 32b 
		//(though it can convert to 8b for the saturation purposes and save to 32b reg)
		uint32_t save;
		asm("cvt.rni.u8.f32.sat	%0, %1;" : "=r"(save) : "f"(x[i] + ((float) 128.0)));
		((uint8_t*) tempResultP)[i] = save;
	}

	//writing result - one row of a picture block by a thread
	*((uint64_t*) resultPtr) = tempResult;
}

/* Documented at declaration */
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

/* Documented at declaration */
int
gpujpeg_idct_gpu(struct gpujpeg_decoder* decoder)
{
    // Get coder
    struct gpujpeg_coder* coder = &decoder->coder;

    // Encode each component
    for ( int comp = 0; comp < coder->param_image.comp_count; comp++ ) {
        // Get component
        struct gpujpeg_component* component = &coder->component[comp];

        int roi_width = component->data_width;
        int roi_height = component->data_height;
        assert(GPUJPEG_BLOCK_SIZE == 8);

        int block_count_x = roi_width / GPUJPEG_BLOCK_SIZE;
        int block_count_y = roi_height / GPUJPEG_BLOCK_SIZE;

        // Get quantization table
        uint16_t* d_quantization_table = decoder->table_quantization[decoder->comp_table_quantization_map[comp]].d_table;

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

        dim3 dct_grid(gpujpeg_div_and_round_up(block_count_x * block_count_y,
				(GPUJPEG_IDCT_BLOCK_X * GPUJPEG_IDCT_BLOCK_Y * GPUJPEG_IDCT_BLOCK_Z) / GPUJPEG_BLOCK_SIZE), 1);
        dim3 dct_block(GPUJPEG_IDCT_BLOCK_X, GPUJPEG_IDCT_BLOCK_Y, GPUJPEG_IDCT_BLOCK_Z);
 
        gpujpeg_idct_gpu_kernel<<<dct_grid, dct_block, 0, *(decoder->stream)>>>(
            component->d_data_quantized,
            component->d_data,
            component->data_width,
            d_quantization_table
        );
        gpujpeg_cuda_check_error("Inverse Integer DCT failed", return -1);
    }

    return 0;
}
