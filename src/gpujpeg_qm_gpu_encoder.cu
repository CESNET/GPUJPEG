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

#include "gpujpeg_qm_gpu_encoder.h"
#include <libgpujpeg/gpujpeg_util.h>
#include <stdlib.h>

#define WARPS_NUM 8 // for compaction kernel

/** One row in probability estimation state machine. */
struct TableRow {
    uint32_t Q_e;
    uint8_t nextIndexLPS;
    uint8_t nextIndexMPS;
    uint8_t switchMPS;
    
    TableRow(uint32_t Q_e, uint8_t nextIndexLPS, uint8_t nextIndexMPS, uint8_t switchMPS)
    : Q_e(Q_e), nextIndexLPS(nextIndexLPS), nextIndexMPS(nextIndexMPS), switchMPS(switchMPS)
    {}

    TableRow() {}
};

/** Stat structure for DC and AC statistical models. */
struct Stat {
	uint8_t mps = 0;
	uint8_t index = 0;
};


/** Arithmetic coder table */
__constant__ struct TableRow gpujpeg_arithmetic_gpu_encoder_state_machine[128];

/** Natural order in constant memory */
__constant__ int gpujpeg_qm_gpu_encoder_order_natural[GPUJPEG_ORDER_NATURAL_SIZE];


struct gpujpeg_qm_gpu_encoder
{
    /** Size of occupied part of output buffer */
    unsigned int * d_gpujpeg_qm_output_byte_count;
};

/** Arithmetic encoder structure.
 *  Doesnt contain A and C.
 */
struct gpujpeg_arithmetic_gpu_coder{
	// Statistical areas for encoding DC and AC coefficients
	// One array for luma, one for chroma
	Stat DC_stats[2][49];
	Stat AC_stats[2][245];
	Stat fixed;

	// previous DC difference index (F.1.4.4.1.2) for each component
	int previous_diff_index[4] = {0};
	// value of previous DC for each component
	int previous_DC[4] = {0};
	
	// renormalization shift counter
	int32_t CT = 11;
	// stack counter
	int32_t ST = 0;

	// last emitted byte
	int last_byte = -1;
	// current component index
	int current_comp;
	// current component type
	enum gpujpeg_component_type c_type;
};


/**
 * This is allocation kernel from Huffman coder.
 * Can be used for QM.
 *
 * QM coder compact output allocation kernel - serially reserves
 * some space for compressed output of segments in output buffer.
 *
 * Only single threadblock with 512 threads is launched.
 */
__global__ static void
gpujpeg_qm_encoder_allocation_kernel (
    struct gpujpeg_segment* const d_segment,
    const int segment_count,
    unsigned int * d_gpujpeg_qm_output_byte_count
) {
    // offsets of segments
    __shared__ unsigned int s_segment_offsets[512];

    // cumulative sum of bytes of all segments
    unsigned int total_byte_count = 0;

    // iterate over all segments
    const unsigned int segment_idx_end = (segment_count + 511) & ~511;
    for(unsigned int segment_idx = threadIdx.x; segment_idx < segment_idx_end; segment_idx += 512) {
        // all threads load byte sizes of their segments (rounded up to next multiple of 16 B) into the shared array
        s_segment_offsets[threadIdx.x] = segment_idx < segment_count
                ? (d_segment[segment_idx].data_compressed_size + 15) & ~15
                : 0;

        // first thread runs a sort of serial prefix sum over the segment sizes to get their offsets
        __syncthreads();
        if(0 == threadIdx.x) {
            #pragma unroll 4
            for(int i = 0; i < 512; i++) {
                const unsigned int segment_size = s_segment_offsets[i];
                s_segment_offsets[i] = total_byte_count;
                total_byte_count += segment_size;
            }
        }
        __syncthreads();

        // all threads write offsets back into corresponding segment structures
        if(segment_idx < segment_count) {
            d_segment[segment_idx].data_compressed_index = s_segment_offsets[threadIdx.x];
        }
    }

    // first thread finally saves the total sum of bytes needed for compressed data
    if(threadIdx.x == 0) {
        *d_gpujpeg_qm_output_byte_count = total_byte_count;
    }
}


/**
 * This is compaction kernel from Huffman coder.
 * Can be used for QM.
 * 
 * QM coder output compaction kernel.
 *
 * @return void
 */
__global__ static void
gpujpeg_qm_encoder_compaction_kernel (
    struct gpujpeg_segment* const d_segment,
    const int segment_count,
    const uint8_t* const d_src,
    uint8_t* const d_dest,
    unsigned int * d_gpujpeg_qm_output_byte_count
) {
    // get some segment (size of threadblocks is 32 x N, so threadIdx.y is warp index)
    const int block_idx = blockIdx.x + blockIdx.y * gridDim.x;
    const int segment_idx = threadIdx.y + block_idx * blockDim.y;
    if(segment_idx >= segment_count) {
        return;
    }

    // temp variables for all warps
    __shared__ uint4* volatile s_out_ptrs[WARPS_NUM];

    // get info about the segment
    const unsigned int segment_byte_count = (d_segment[segment_idx].data_compressed_size + 15) & ~15;  // number of bytes rounded up to multiple of 16
    const unsigned int segment_in_offset = d_segment[segment_idx].data_temp_index;  // this should be aligned at least to 16byte boundary

    // first thread of each warp reserves space in output buffer
    if(0 == threadIdx.x) {
        // Load precomputed output offset
        const unsigned int segment_out_offset = d_segment[segment_idx].data_compressed_index;
        s_out_ptrs[threadIdx.y] = (uint4*)(d_dest + segment_out_offset);
    }

    // all threads read output buffer offset for their segment and prepare input and output pointers and number of copy iterations
    const uint4 * d_in = threadIdx.x + (uint4*)(d_src + segment_in_offset);
    uint4 * d_out = threadIdx.x + s_out_ptrs[threadIdx.y];
    unsigned int copy_iterations = segment_byte_count / 512; // 512 is number of bytes copied in each iteration (32 threads * 16 bytes per thread)

    // copy the data!
    while(copy_iterations--) {
        *d_out = *d_in;
        d_out += 32;
        d_in += 32;
    }

    // copy remaining bytes (less than 512 bytes)
    if((threadIdx.x * 16) < (segment_byte_count & 511)) {
        *d_out = *d_in;
    }
}

// Threadblock size for kernel
#define THREAD_BLOCK_SIZE 32

/**
 * Write one byte to compressed data
 *
 * @param data_compressed  Data compressed
 * @param value  Byte value to write
 * @return void
 */
#define gpujpeg_qm_gpu_encoder_emit_byte(data_compressed, value) { \
    *data_compressed = (uint8_t)(value); \
    data_compressed++; }

/**
 * Write two bytes to compressed data
 *
 * @param data_compressed  Data compressed
 * @param value  Two-byte value to write
 * @return void
 */
#define gpujpeg_qm_gpu_encoder_emit_2byte(data_compressed, value) { \
    *data_compressed = (uint8_t)(((value) >> 8) & 0xFF); \
    data_compressed++; \
    *data_compressed = (uint8_t)((value) & 0xFF); \
    data_compressed++; }

/**
 * Write marker to compressed data
 *
 * @param data_compressed  Data compressed
 * @oaran marker  Marker to write (JPEG_MARKER_...)
 * @return void
 */
#define gpujpeg_qm_gpu_encoder_marker(data_compressed, marker) { \
    *data_compressed = 0xFF;\
    data_compressed++; \
    *data_compressed = (uint8_t)(marker); \
    data_compressed++; }

/** Emit byte using writer.
 *  Byte passed as int to this function isnt emitted asap,
 *  instead it is stored in buffer.
 *  Byte stored in buffer before is emitted instead.
 */
__device__ void emit_byte(struct gpujpeg_arithmetic_gpu_coder* coder, uint8_t* & data_compressed, int byte) {
	if (coder->last_byte != -1) {
		gpujpeg_qm_gpu_encoder_emit_byte(data_compressed, coder->last_byte);
    }
	coder->last_byte = byte;
}

/** Stuffs a zero byte whenever the addition of the carry to the data
 *  already in the entropy-coded segments creates a X’FF’ byte (D.1.6).
 */
__device__ void stuff_0(struct gpujpeg_arithmetic_gpu_coder* coder, uint8_t* & data_compressed) {
	if (coder->last_byte == 0xFF)
		emit_byte(coder, data_compressed, 0);
}

/** Put stacked 0xFFs in the entropy-coded segment */
__device__ void output_stacked_XFFs(struct gpujpeg_arithmetic_gpu_coder* coder, uint8_t* & data_compressed) {
	while (coder->ST != 0) {
		emit_byte(coder, data_compressed, 0xFF);
		emit_byte(coder, data_compressed, 0);
		coder->ST--;
	}
}

/** Place stacked output bytes converted to zero
 *  by carry-over in stuff_0 in the entropy-coded segment (D.1.6).
 */
__device__ void output_stacked_zeros(struct gpujpeg_arithmetic_gpu_coder* coder, uint8_t* & data_compressed) {
	while (coder->ST != 0) {
		emit_byte(coder, data_compressed, 0);
		coder->ST--;
	}
}

/** Removes byte of compressed data from C
 *  and sends it in the entropy-coded segment
 */
__device__ void byte_out(struct gpujpeg_arithmetic_gpu_coder* coder, uint8_t* & data_compressed, int32_t *A, int32_t *C) {
	int T = *C >> 19;
	if (T > 0xFF) {
		coder->last_byte++;
		stuff_0(coder, data_compressed);
		output_stacked_zeros(coder, data_compressed);
		emit_byte(coder, data_compressed, T);
	} else {
		if (T == 0xFF)
			coder->ST++;
		else {
			output_stacked_XFFs(coder, data_compressed);
			emit_byte(coder, data_compressed, T);
		}
	}
	*C &= 0x7FFFF;
}

/** Sets as many low order bits of the code register to zero as possible
 *  without pointing outside of the final interval
 */
__device__ void clear_final_bits(struct gpujpeg_arithmetic_gpu_coder* coder, int32_t *A, int32_t *C) {
	unsigned int T = *C + *A - 1;
	T &= 0xFFFF0000;
	if (T < *C)
		T += 0x8000;
	*C = T;
}

/** Terminates the arithmetic encoding procedures and prepares
 *  the entropy-coded segment for the addition of the X’FF’ 
 */
__device__ void flush(struct gpujpeg_arithmetic_gpu_coder* coder, uint8_t* & data_compressed, int32_t *A, int32_t *C) {
	clear_final_bits(coder, A, C);
    *C <<= coder->CT;
	byte_out(coder, data_compressed, A, C);
	*C <<= 8;
	byte_out(coder, data_compressed, A, C);
	if (coder->last_byte != 0) gpujpeg_qm_gpu_encoder_emit_byte(data_compressed, coder->last_byte);
}

/** Performs encoder renormalization */
__device__ void renorm_e(struct gpujpeg_arithmetic_gpu_coder* coder, uint8_t* & data_compressed, int32_t *A, int32_t *C) {
	do {
		*A <<= 1;
		*C <<= 1;
		coder->CT--;
		if (coder->CT == 0) {
			byte_out(coder, data_compressed, A, C);
			coder->CT = 8;
		}
	} while (*A < 0x8000);
}

/** Estimates new Stat after LPS coding.
 *  May also change sense of MPS.
 */
__device__ void estimate_Qe_after_LPS(struct gpujpeg_arithmetic_gpu_coder* coder, Stat* S) {
	int I = S->index;
	if (gpujpeg_arithmetic_gpu_encoder_state_machine[I].switchMPS == 1)
		S->mps = 1 - S->mps;
	I = gpujpeg_arithmetic_gpu_encoder_state_machine[I].nextIndexLPS;
	S->index = I;
}

/** Estimates new Stat after MPS coding. */
__device__ void estimate_Qe_after_MPS(struct gpujpeg_arithmetic_gpu_coder* coder, Stat* S) {
	int I = S->index;
	I = gpujpeg_arithmetic_gpu_encoder_state_machine[I].nextIndexMPS;
	S->index = I;
}

/** Codes LPS. */
__device__ void code_LPS(struct gpujpeg_arithmetic_gpu_coder* coder, Stat* S, uint8_t* & data_compressed, int32_t *A, int32_t *C) {
	*A -= gpujpeg_arithmetic_gpu_encoder_state_machine[S->index].Q_e;
	if (*A >= gpujpeg_arithmetic_gpu_encoder_state_machine[S->index].Q_e) {
		*C += *A;
		*A = gpujpeg_arithmetic_gpu_encoder_state_machine[S->index].Q_e;
	}
	estimate_Qe_after_LPS(coder, S);
	renorm_e(coder, data_compressed, A, C);
}

/** Codes MPS. */
__device__ void code_MPS(struct gpujpeg_arithmetic_gpu_coder* coder, Stat* S, uint8_t* & data_compressed, int32_t *A, int32_t *C) {
	*A -= gpujpeg_arithmetic_gpu_encoder_state_machine[S->index].Q_e;
	if (*A < 0x8000) {
		if (*A < gpujpeg_arithmetic_gpu_encoder_state_machine[S->index].Q_e) {
			*C += *A;
			*A = gpujpeg_arithmetic_gpu_encoder_state_machine[S->index].Q_e; 
		}
		estimate_Qe_after_MPS(coder, S);
		renorm_e(coder, data_compressed, A, C);
	}
}

/** Codes bit 1. */
__device__ void code_1(struct gpujpeg_arithmetic_gpu_coder* coder, Stat* S, uint8_t* & data_compressed, int32_t *A, int32_t *C) {
	if (S->mps == 1) {
		code_MPS(coder, S, data_compressed, A, C);
	}
	else {
		code_LPS(coder, S, data_compressed, A, C);
	}
}

/** Codes bit 0. */
__device__ void code_0(struct gpujpeg_arithmetic_gpu_coder* coder, Stat* S, uint8_t* & data_compressed, int32_t *A, int32_t *C) {
	if (S->mps == 0) {
        code_MPS(coder, S, data_compressed, A, C);
	}
    else {
        code_LPS(coder, S, data_compressed, A, C);
	}
}

/** Encodes sign of DC difference. */
__device__ int encode_sign_of_V_DC(struct gpujpeg_arithmetic_gpu_coder* coder, int S_i, const int V, uint8_t* & data_compressed, int32_t *A, int32_t *C) {
	if (V < 0) {
		code_1(coder, &coder->DC_stats[coder->c_type][S_i + 1], data_compressed, A, C); // code SS
		S_i += 3; //S = SN
		coder->previous_diff_index[coder->current_comp] = 8; //-small
	} else {
		code_0(coder, &coder->DC_stats[coder->c_type][S_i + 1], data_compressed, A, C);
		S_i += 2; // S = SP
		coder->previous_diff_index[coder->current_comp] = 4; //+small
	}
	if (abs(V) > 2)
		coder->previous_diff_index[coder->current_comp] += 8; //+-large
	return S_i;
}

/** Encodes sign of AC coefficient. */
__device__ int encode_sign_of_V_AC(struct gpujpeg_arithmetic_gpu_coder* coder, int S_i, const int V, uint8_t* & data_compressed, int32_t *A, int32_t *C) {
    if (V < 0)
        code_1(coder, &coder->fixed, data_compressed, A, C); // code SS
    else
        code_0(coder, &coder->fixed, data_compressed, A, C); // code SS
    return S_i + 1; // S = SP/SN
}

/** Encodes magnitude category of DC difference.  */
__device__ int encode_log2_Sz_DC(struct gpujpeg_arithmetic_gpu_coder* coder, int Sz, int S_i, int* M, uint8_t* & data_compressed, int32_t *A, int32_t *C) {
	if (Sz >= *M) {
		code_1(coder, &coder->DC_stats[coder->c_type][S_i], data_compressed, A, C);
		*M = 2;
		S_i = 20; // S = X1
		if (Sz >= *M) {
			code_1(coder, &coder->DC_stats[coder->c_type][S_i], data_compressed, A, C);
			*M = 4;
			S_i++; // S = X2
			while (Sz >= *M) {
				code_1(coder, &coder->DC_stats[coder->c_type][S_i], data_compressed, A, C);
				*M <<= 1;
				S_i++;
			}
		}
	}
	code_0(coder, &coder->DC_stats[coder->c_type][S_i], data_compressed, A, C);
	*M >>= 1;
	return S_i;
}

/** Encodes magnitude category of AC coefficient */
__device__ int encode_log2_Sz_AC(struct gpujpeg_arithmetic_gpu_coder* coder, int Sz, int S_i, const int K, int* M, uint8_t* & data_compressed, int32_t *A, int32_t *C) {
	if (Sz >= *M) {
        code_1(coder, &coder->AC_stats[coder->c_type][S_i], data_compressed, A, C);
        *M = 2;
        if (Sz >= *M) {
            code_1(coder, &coder->AC_stats[coder->c_type][S_i], data_compressed, A, C);
            *M = 4;
            S_i = K <= 5 ? 189 : 217; // S = X2
            while (Sz >= *M) {
                code_1(coder, &coder->AC_stats[coder->c_type][S_i], data_compressed, A, C);
                *M <<= 1;
                S_i++;

            }
        }
    }
    code_0(coder, &coder->AC_stats[coder->c_type][S_i], data_compressed, A, C);
    *M >>= 1;
    return S_i;
}


/** Encodes value of magnitude of DC difference or AC coefficient  */
__device__ void encode_Sz_bits(struct gpujpeg_arithmetic_gpu_coder* coder, const int Sz, int S_i, int M, bool isDC, uint8_t* & data_compressed, int32_t *A, int32_t *C) {
	S_i += 14;
	while (M >>= 1) {
		int T = M & Sz;
		if (T == 0)
			if (isDC)
				code_0(coder, &coder->DC_stats[coder->c_type][S_i], data_compressed, A, C);
			else
				code_0(coder, &coder->AC_stats[coder->c_type][S_i], data_compressed, A, C);
		else
			if (isDC)
				code_1(coder, &coder->DC_stats[coder->c_type][S_i], data_compressed, A, C);
			else
				code_1(coder, &coder->AC_stats[coder->c_type][S_i], data_compressed, A, C);
	}
}

/** Encodes DC difference if difference is not equal to 0 */
__device__ void encode_V_DC(struct gpujpeg_arithmetic_gpu_coder* coder, int S_i, const int V, uint8_t* & data_compressed, int32_t *A, int32_t *C) {
	int s_i = encode_sign_of_V_DC(coder, S_i, V, data_compressed, A, C);
	int Sz = abs(V) - 1;
	int M = 1;
	s_i = encode_log2_Sz_DC(coder, Sz, s_i, &M, data_compressed, A, C);
	encode_Sz_bits(coder, Sz, s_i, M, true, data_compressed, A, C);
}

/** Encodes AC coefficient */
__device__ void encode_V_AC(struct gpujpeg_arithmetic_gpu_coder* coder, int S_i, const int V, int K, uint8_t* & data_compressed,	int32_t *A, int32_t *C) {
	int s_i = encode_sign_of_V_AC(coder, S_i, V, data_compressed, A, C);
    int Sz = abs(V) - 1;
	int M = 1;
    s_i = encode_log2_Sz_AC(coder, Sz, s_i, K, &M, data_compressed, A, C);
    encode_Sz_bits(coder, Sz, s_i, M, false, data_compressed, A, C);
}

/** Encodes DC difference (F.1.4.1) */
__device__ void encode_DC_diff(struct gpujpeg_arithmetic_gpu_coder* coder, const int dc, uint8_t* & data_compressed, int32_t *A, int32_t *C) {
	int V = dc - coder->previous_DC[coder->current_comp];
	coder->previous_DC[coder->current_comp] = dc;
	int S0 = coder->previous_diff_index[coder->current_comp];
	if (V == 0) {
		code_0(coder, &coder->DC_stats[coder->c_type][S0], data_compressed, A, C);
		coder->previous_diff_index[coder->current_comp] = 0; //zero diff
	} else {
		code_1(coder, &coder->DC_stats[coder->c_type][S0], data_compressed, A, C);
		encode_V_DC(coder, S0, V, data_compressed, A, C);
	}
}

/** Encodes all AC coefficients (F.1.4.2) */
__device__ void encode_AC_coefficients(struct gpujpeg_arithmetic_gpu_coder* coder, int16_t* block, uint8_t* & data_compressed, int32_t *A, int32_t *C) {
	int K = 1;
	int SE = 3 * (K - 1);

	// Finds EOB index. EOB is index of last non-null coefficient in zig-zag sequence increased by 1.
	int EOB = 63;
	for (; EOB > 0; EOB--)
		if (block[gpujpeg_qm_gpu_encoder_order_natural[EOB]]) break;
	EOB++;

	while(true) {
		if (K == EOB) {
			code_1(coder, &coder->AC_stats[coder->c_type][SE], data_compressed, A, C);
			return;
		} else {
			code_0(coder, &coder->AC_stats[coder->c_type][SE], data_compressed, A, C);
			int V = block[gpujpeg_qm_gpu_encoder_order_natural[K]];
			while (V == 0) {
				code_0(coder, &coder->AC_stats[coder->c_type][SE + 1], data_compressed, A, C);
				K++;
				SE += 3;
				V = block[gpujpeg_qm_gpu_encoder_order_natural[K]];
			}
			code_1(coder, &coder->AC_stats[coder->c_type][SE + 1], data_compressed, A, C);
			encode_V_AC(coder, SE + 1, V, K, data_compressed, A, C);
			if (K == 63) return; // K == Se
			K++;
			SE = 3 * (K - 1);
		}
	}
}

/** Encodes 8x8  block of DCT coefficients */
__device__ void gpujpeg_qm_cpu_encoder_encode_block(int16_t *block, struct gpujpeg_arithmetic_gpu_coder* coder, uint8_t* & data_compressed, int32_t *A, int32_t *C) {
	encode_DC_diff(coder, block[gpujpeg_qm_gpu_encoder_order_natural[0]], data_compressed, A, C);
	encode_AC_coefficients(coder, block, data_compressed, A, C);
}

/**
 * QM encoder kernel
 *
 * @return void
 */
__global__ static void
gpujpeg_qm_encoder_encode_kernel(
    struct gpujpeg_component* d_component,
    struct gpujpeg_segment* d_segment,
    int comp_count,
    int segment_count,
    uint8_t* d_data_compressed,
    unsigned int * d_gpujpeg_qm_output_byte_count
)
{
    int segment_index = blockIdx.x * blockDim.x + threadIdx.x;
    if ( segment_index >= segment_count)
        return;

    struct gpujpeg_segment* segment = &d_segment[segment_index];

    // first thread initializes compact output size for next kernel
    if(0 == segment_index) {
        *d_gpujpeg_qm_output_byte_count = 0;
    }

    // Initialize arithmetic coder
    struct gpujpeg_arithmetic_gpu_coder coder;
    int32_t A = 0x10000;
    int32_t C = 0;
    coder.fixed.mps = 0;
    coder.fixed.index = 113;

    // Prepare data pointers
    uint8_t* data_compressed = &d_data_compressed[segment->data_temp_index];
    uint8_t* data_compressed_start = data_compressed;
	
    // Non-interleaving mode
    if ( comp_count == 1 ) {
        int segment_index = segment->scan_segment_index;
        // Encode MCUs in segment
        for ( int mcu_index = 0; mcu_index < segment->mcu_count; mcu_index++ ) {
            // Get component for current scan
            struct gpujpeg_component* component = &d_component[segment->scan_index];

	    // Set component info in arithmetic coder
	    coder.current_comp = 0;
	    coder.c_type = GPUJPEG_COMPONENT_LUMINANCE;

            // Get component data for MCU
            int16_t* block = &component->d_data_quantized[(segment_index * component->segment_mcu_count + mcu_index) * component->mcu_size];
			
            // Encode 8x8 block
            gpujpeg_qm_cpu_encoder_encode_block(block, &coder, data_compressed, &A, &C);
        }
    }
    // Interleaving mode
    else {
        int segment_index = segment->scan_segment_index;
        // Encode MCUs in segment
        for ( int mcu_index = 0; mcu_index < segment->mcu_count; mcu_index++ ) {
            for ( int comp = 0; comp < comp_count; comp++ ) {
                struct gpujpeg_component* component = &d_component[comp];

				// Set component info in arithmetic coder 
				coder.current_comp = comp;
				coder.c_type = component->type;

                // Prepare mcu indexes
                int mcu_index_x = (segment_index * component->segment_mcu_count + mcu_index) % component->mcu_count_x;
                int mcu_index_y = (segment_index * component->segment_mcu_count + mcu_index) / component->mcu_count_x;
                // Compute base data index
                int data_index_base = mcu_index_y * (component->mcu_size * component->mcu_count_x) + mcu_index_x * (component->mcu_size_x * GPUJPEG_BLOCK_SIZE);

                // For all vertical 8x8 blocks
                for ( int y = 0; y < component->sampling_factor.vertical; y++ ) {
                    // Compute base row data index
                    int data_index_row = data_index_base + y * (component->mcu_count_x * component->mcu_size_x * GPUJPEG_BLOCK_SIZE);
                    // For all horizontal 8x8 blocks
                    for ( int x = 0; x < component->sampling_factor.horizontal; x++ ) {
                        // Compute 8x8 block data index
                        int data_index = data_index_row + x * GPUJPEG_BLOCK_SIZE * GPUJPEG_BLOCK_SIZE;

                        // Get component data for MCU
                        int16_t* block = &component->d_data_quantized[data_index];

                        // Encode 8x8 block
                        gpujpeg_qm_cpu_encoder_encode_block(block, &coder, data_compressed, &A, &C);
                    }
                }
            }
        }
    }

    // Flush arithmetic coder 
    flush(&coder, data_compressed, &A, &C);
    // Output restart marker
    int restart_marker = GPUJPEG_MARKER_RST0 + (segment->scan_segment_index % 8);
    gpujpeg_qm_gpu_encoder_marker(data_compressed, restart_marker);

    // Set compressed size
    segment->data_compressed_size = data_compressed - data_compressed_start;
}


/** Documented at declaration */
struct gpujpeg_qm_gpu_encoder *
gpujpeg_qm_gpu_encoder_create(const struct gpujpeg_encoder * encoder)
{
    struct gpujpeg_qm_gpu_encoder * qm_gpu_encoder = (struct gpujpeg_qm_gpu_encoder *) malloc(sizeof(struct gpujpeg_qm_gpu_encoder));
    if ( qm_gpu_encoder == NULL ) {
        return NULL;
    }
    memset(qm_gpu_encoder, 0, sizeof(struct gpujpeg_qm_gpu_encoder));

    // Allocate
    cudaMalloc((void**)&qm_gpu_encoder->d_gpujpeg_qm_output_byte_count, sizeof(unsigned int));
    gpujpeg_cuda_check_error("Allocation of qm output byte count failed", return NULL);

    // Copy natural order to constant device memory
    cudaMemcpyToSymbol(
        gpujpeg_qm_gpu_encoder_order_natural,
        gpujpeg_order_natural,
        GPUJPEG_ORDER_NATURAL_SIZE * sizeof(int),
        0,
        cudaMemcpyHostToDevice
    );
    gpujpeg_cuda_check_error("qm encoder init (natural order copy)", return NULL);

    // Define probability state machine
    TableRow arithMachine[128] = {
  TableRow(0x5a1d,   1,   1, 1 ), TableRow(0x2586,  14,   2, 0 ),
  TableRow(0x1114,  16,   3, 0 ), TableRow(0x080b,  18,   4, 0 ),
  TableRow(0x03d8,  20,   5, 0 ), TableRow(0x01da,  23,   6, 0 ),
  TableRow(0x00e5,  25,   7, 0 ), TableRow(0x006f,  28,   8, 0 ),
  TableRow(0x0036,  30,   9, 0 ), TableRow(0x001a,  33,  10, 0 ),
  TableRow(0x000d,  35,  11, 0 ), TableRow(0x0006,   9,  12, 0 ),
  TableRow(0x0003,  10,  13, 0 ), TableRow(0x0001,  12,  13, 0 ),
  TableRow(0x5a7f,  15,  15, 1 ), TableRow(0x3f25,  36,  16, 0 ),
  TableRow(0x2cf2,  38,  17, 0 ), TableRow(0x207c,  39,  18, 0 ),
  TableRow(0x17b9,  40,  19, 0 ), TableRow(0x1182,  42,  20, 0 ),
  TableRow(0x0cef,  43,  21, 0 ), TableRow(0x09a1,  45,  22, 0 ),
  TableRow(0x072f,  46,  23, 0 ), TableRow(0x055c,  48,  24, 0 ),
  TableRow(0x0406,  49,  25, 0 ), TableRow(0x0303,  51,  26, 0 ),
  TableRow(0x0240,  52,  27, 0 ), TableRow(0x01b1,  54,  28, 0 ),
  TableRow(0x0144,  56,  29, 0 ), TableRow(0x00f5,  57,  30, 0 ),
  TableRow(0x00b7,  59,  31, 0 ), TableRow(0x008a,  60,  32, 0 ),
  TableRow(0x0068,  62,  33, 0 ), TableRow(0x004e,  63,  34, 0 ),
  TableRow(0x003b,  32,  35, 0 ), TableRow(0x002c,  33,   9, 0 ),
  TableRow(0x5ae1,  37,  37, 1 ), TableRow(0x484c,  64,  38, 0 ),
  TableRow(0x3a0d,  65,  39, 0 ), TableRow(0x2ef1,  67,  40, 0 ),
  TableRow(0x261f,  68,  41, 0 ), TableRow(0x1f33,  69,  42, 0 ),
  TableRow(0x19a8,  70,  43, 0 ), TableRow(0x1518,  72,  44, 0 ),
  TableRow(0x1177,  73,  45, 0 ), TableRow(0x0e74,  74,  46, 0 ),
  TableRow(0x0bfb,  75,  47, 0 ), TableRow(0x09f8,  77,  48, 0 ),
  TableRow(0x0861,  78,  49, 0 ), TableRow(0x0706,  79,  50, 0 ),
  TableRow(0x05cd,  48,  51, 0 ), TableRow(0x04de,  50,  52, 0 ),
  TableRow(0x040f,  50,  53, 0 ), TableRow(0x0363,  51,  54, 0 ),
  TableRow(0x02d4,  52,  55, 0 ), TableRow(0x025c,  53,  56, 0 ),
  TableRow(0x01f8,  54,  57, 0 ), TableRow(0x01a4,  55,  58, 0 ),
  TableRow(0x0160,  56,  59, 0 ), TableRow(0x0125,  57,  60, 0 ),
  TableRow(0x00f6,  58,  61, 0 ), TableRow(0x00cb,  59,  62, 0 ),
  TableRow(0x00ab,  61,  63, 0 ), TableRow(0x008f,  61,  32, 0 ),
  TableRow(0x5b12,  65,  65, 1 ), TableRow(0x4d04,  80,  66, 0 ),
  TableRow(0x412c,  81,  67, 0 ), TableRow(0x37d8,  82,  68, 0 ),
  TableRow(0x2fe8,  83,  69, 0 ), TableRow(0x293c,  84,  70, 0 ),
  TableRow(0x2379,  86,  71, 0 ), TableRow(0x1edf,  87,  72, 0 ),
  TableRow(0x1aa9,  87,  73, 0 ), TableRow(0x174e,  72,  74, 0 ),
  TableRow(0x1424,  72,  75, 0 ), TableRow(0x119c,  74,  76, 0 ),
  TableRow(0x0f6b,  74,  77, 0 ), TableRow(0x0d51,  75,  78, 0 ),
  TableRow(0x0bb6,  77,  79, 0 ), TableRow(0x0a40,  77,  48, 0 ),
  TableRow(0x5832,  80,  81, 1 ), TableRow(0x4d1c,  88,  82, 0 ),
  TableRow(0x438e,  89,  83, 0 ), TableRow(0x3bdd,  90,  84, 0 ),
  TableRow(0x34ee,  91,  85, 0 ), TableRow(0x2eae,  92,  86, 0 ),
  TableRow(0x299a,  93,  87, 0 ), TableRow(0x2516,  86,  71, 0 ),
  TableRow(0x5570,  88,  89, 1 ), TableRow(0x4ca9,  95,  90, 0 ),
  TableRow(0x44d9,  96,  91, 0 ), TableRow(0x3e22,  97,  92, 0 ),
  TableRow(0x3824,  99,  93, 0 ), TableRow(0x32b4,  99,  94, 0 ),
  TableRow(0x2e17,  93,  86, 0 ), TableRow(0x56a8,  95,  96, 1 ),
  TableRow(0x4f46, 101,  97, 0 ), TableRow(0x47e5, 102,  98, 0 ),
  TableRow(0x41cf, 103,  99, 0 ), TableRow(0x3c3d, 104, 100, 0 ),
  TableRow(0x375e,  99,  93, 0 ), TableRow(0x5231, 105, 102, 0 ),
  TableRow(0x4c0f, 106, 103, 0 ), TableRow(0x4639, 107, 104, 0 ),
  TableRow(0x415e, 103,  99, 0 ), TableRow(0x5627, 105, 106, 1 ),
  TableRow(0x50e7, 108, 107, 0 ), TableRow(0x4b85, 109, 103, 0 ),
  TableRow(0x5597, 110, 109, 0 ), TableRow(0x504f, 111, 107, 0 ),
  TableRow(0x5a10, 110, 111, 1 ), TableRow(0x5522, 112, 109, 0 ),
  TableRow(0x59eb, 112, 111, 1 ), TableRow(0x5a1d, 113, 113, 0 ),
  TableRow(0x0, 114, 114, 0 ), // filler
  TableRow(0x0, 114, 114, 0 ), // filler
  TableRow(0x0, 114, 114, 0 ), // filler
  TableRow(0x0, 114, 114, 0 ), // filler
  TableRow(0x0, 114, 114, 0 ), // filler
  TableRow(0x0, 114, 114, 0 ), // filler
  TableRow(0x0, 114, 114, 0 ), // filler
  TableRow(0x0, 114, 114, 0 ), // filler
  TableRow(0x0, 114, 114, 0 ), // filler
  TableRow(0x0, 114, 114, 0 ), // filler
  TableRow(0x0, 114, 114, 0 ), // filler
  TableRow(0x0, 114, 114, 0 ), // filler
  TableRow(0x0, 114, 114, 0 ), // filler
  TableRow(0x0, 114, 114, 0 ), // filler
};

	// Copy probability estimation state machine to constant device memory
	cudaMemcpyToSymbol(
    	gpujpeg_arithmetic_gpu_encoder_state_machine,
    	arithMachine,
    	128 * sizeof(TableRow),
    	0,
    	cudaMemcpyHostToDevice
	);
	gpujpeg_cuda_check_error("Arithmetic encoder init (probability state machine)", return NULL);


    // Configure more shared memory for all kernels
    cudaFuncSetCacheConfig(gpujpeg_qm_encoder_compaction_kernel, cudaFuncCachePreferShared);
    cudaFuncSetCacheConfig(gpujpeg_qm_encoder_encode_kernel, cudaFuncCachePreferShared);
    cudaFuncSetCacheConfig(gpujpeg_qm_encoder_allocation_kernel, cudaFuncCachePreferShared);

    return qm_gpu_encoder;
}

/** Documented at declaration */
void
gpujpeg_qm_gpu_encoder_destroy(struct gpujpeg_qm_gpu_encoder * qm_gpu_encoder)
{
    assert(qm_gpu_encoder != NULL);

    if (qm_gpu_encoder->d_gpujpeg_qm_output_byte_count != NULL) {
        cudaFree(qm_gpu_encoder->d_gpujpeg_qm_output_byte_count);
    }

    free(qm_gpu_encoder);
}


/**
 * Get grid size for specified count of threadblocks. (Grid size is limited
 * to 65536 in both directions, so if we need more threadblocks, we must use
 * both x and y coordinates.)
 */
dim3
gpujpeg_qm_gpu_encoder_grid_size(int tblock_count)
{
    dim3 size(tblock_count);
    while(size.x > 0xffff) {
        size.x = (size.x + 1) >> 1;
        size.y <<= 1;
    }
    return size;
}

/** Documented at declaration */
int
gpujpeg_qm_gpu_encoder_encode(struct gpujpeg_encoder* encoder, struct gpujpeg_qm_gpu_encoder * qm_gpu_encoder, unsigned int * output_byte_count)
{
    // Get coder
    struct gpujpeg_coder* coder = &encoder->coder;
    assert(coder->param.restart_interval > 0);

    // Select encoder kernel which either expects continuos segments of blocks or uses block lists
    int comp_count = 1;
    if ( coder->param.interleaved == 1 )
        comp_count = coder->param_image.comp_count;
    assert(comp_count >= 1 && comp_count <= GPUJPEG_MAX_COMPONENT_COUNT);

    // Run kernel
    dim3 thread(THREAD_BLOCK_SIZE);
    dim3 grid(gpujpeg_div_and_round_up(coder->segment_count, thread.x));
    gpujpeg_qm_encoder_encode_kernel<<<grid, thread, 0, *(encoder->stream)>>>(
        coder->d_component,
        coder->d_segment,
        comp_count,
        coder->segment_count,
        coder->d_temp_huffman,
        qm_gpu_encoder->d_gpujpeg_qm_output_byte_count
    );
    gpujpeg_cuda_check_error("qm encoding failed", return -1);

    // Run output size computation kernel to allocate the output buffer space
    gpujpeg_qm_encoder_allocation_kernel<<<1, 512, 0, *(encoder->stream)>>>
		(coder->d_segment, coder->segment_count, qm_gpu_encoder->d_gpujpeg_qm_output_byte_count);
    gpujpeg_cuda_check_error("qm encoder output allocation failed", return -1);

    // Run output compaction kernel (one warp per segment)
    const dim3 compaction_thread(32, WARPS_NUM);
    const dim3 compaction_grid = gpujpeg_qm_gpu_encoder_grid_size(gpujpeg_div_and_round_up(coder->segment_count, WARPS_NUM));
    gpujpeg_qm_encoder_compaction_kernel<<<compaction_grid, compaction_thread, 0, *(encoder->stream)>>>(
        coder->d_segment,
        coder->segment_count,
        coder->d_temp_huffman,
        coder->d_data_compressed,
        qm_gpu_encoder->d_gpujpeg_qm_output_byte_count
    );
    gpujpeg_cuda_check_error("qm output compaction failed", return -1);

    // Read and return number of occupied bytes
    cudaMemcpyAsync(output_byte_count, qm_gpu_encoder->d_gpujpeg_qm_output_byte_count, sizeof(unsigned int), cudaMemcpyDeviceToHost, *(encoder->stream));
    gpujpeg_cuda_check_error("qm output size getting failed", return -1);

    // indicate success
    return 0;
}
