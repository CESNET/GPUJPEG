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
 
#include "gpujpeg_qm_cpu_encoder.h"
#include <libgpujpeg/gpujpeg_util.h>
#include <vector>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <array>

/** One row in probability estimation state machine. */
struct TableRow {
    uint32_t Q_e;
    uint8_t nextIndexLPS;
    uint8_t nextIndexMPS;
    uint8_t switchMPS;
    
    TableRow(uint32_t Q_e, uint8_t nextIndexLPS, uint8_t nextIndexMPS, uint8_t switchMPS)
    : Q_e(Q_e), nextIndexLPS(nextIndexLPS), nextIndexMPS(nextIndexMPS), switchMPS(switchMPS)
    {}
};

/** Probability estimation state machine. */
TableRow arithTable[114] = {
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
};


/** Stat structure for DC and AC statistical models. */
struct Stat {
    uint8_t mps;
    uint8_t index;
};


/** QM encoder structure. */
struct gpujpeg_qm_cpu_encoder
{
	// Color components
	struct gpujpeg_component* component;

	// JPEG writer structure
	struct gpujpeg_writer* writer;

    // Component count (1 means non-interleaving grayscale, > 1 means interleaving color)
    int comp_count;

	// Statistical areas for encoding DC and AC coefficients
	// One array for luma, one for chroma
	Stat DC_stats[2][49];
	Stat AC_stats[2][245];
	Stat fixed;

	// previous DC difference index (F.1.4.4.1.2) for each component
	int previous_diff_index[3];
	// value of previous DC for each component
	int previous_DC[3];
	
	// probability interval
	uint32_t A;
	// code register
	uint32_t C;
	// renormalization shift counter
	int CT;
	// stack counter
	int ST; 

	// last emitted byte
	int last_byte;
	// current component index
	int current_comp;
	// current component type
	enum gpujpeg_component_type c_type;
};

/** Initialize statistics. */
void initStatistics(struct gpujpeg_qm_cpu_encoder* coder) {
	for (int i = 0; i < 3; i++) {
		coder->previous_diff_index[i] = 0;
       	coder->previous_DC[i] = 0;
	}

	for (int j = 0; j < 2; j++) {
    	for (int i = 0; i < 49; i++) {
       		coder->DC_stats[j][i].mps = 0;
       		coder->DC_stats[j][i].index = 0;
       	}
	}
	
	for (int j = 0; j < 2; j++) {
    	for (int i = 0; i < 245; i++) {
       		coder->AC_stats[j][i].mps = 0;
       		coder->AC_stats[j][i].index = 0;
    	}
	}
	
    coder->fixed.mps = 0;
    coder->fixed.index = 113;
}

/** Initiaze whole qm coder structure. */
gpujpeg_qm_cpu_encoder init(struct gpujpeg_encoder* encoder) {
	struct gpujpeg_qm_cpu_encoder coder;
	coder.writer = encoder->writer;
	coder.component = encoder->coder.component;
    coder.comp_count = encoder->coder.param_image.comp_count;
	
	coder.A = 0x10000;
	coder.C = 0;
	coder.CT = 11;
	coder.ST = 0;
	coder.last_byte = -1; // no byte emitted yet
	coder.current_comp = -1;
	coder.c_type = GPUJPEG_COMPONENT_LUMINANCE;

	initStatistics(&coder);
	return coder;		
}

/** Emit byte using writer.
 *  Byte passed as int to this function isnt emitted asap,
 *  instead it is stored in buffer.
 *  Byte stored in buffer before is emitted instead.
 */
void emit_byte(struct gpujpeg_qm_cpu_encoder* coder, int byte) {
	if (coder->last_byte != -1) {
		gpujpeg_writer_emit_byte(coder->writer, coder->last_byte);
    }
	coder->last_byte = byte;
}

/** Stuffs a zero byte whenever the addition of the carry to the data
 *  already in the entropy-coded segments creates a X’FF’ byte (D.1.6).
 */
void stuff_0(struct gpujpeg_qm_cpu_encoder* coder) {
	if (coder->last_byte == 0xFF)
		emit_byte(coder, 0);
}

/** Place stacked output bytes converted to zero
 *  by carry-over in stuff_0 in the entropy-coded segment (D.1.6).
 */
void output_stacked_zeros(struct gpujpeg_qm_cpu_encoder* coder) {
	while (coder->ST != 0) {
		emit_byte(coder, 0);
		coder->ST--;
	}
}

/** Put stacked 0xFFs in the entropy-coded segment */
void output_stacked_XFFs(struct gpujpeg_qm_cpu_encoder* coder) {
	while (coder->ST != 0) {
		emit_byte(coder, 0xFF);
		emit_byte(coder, 0);
		coder->ST--;
	}
}

/** Removes byte of compressed data from C
 *  and sends it in the entropy-coded segment
 */
void byte_out(struct gpujpeg_qm_cpu_encoder* coder) {
	int T = coder->C >> 19;
	if (T > 0xFF) {
		coder->last_byte++;
		stuff_0(coder);
		output_stacked_zeros(coder);
		emit_byte(coder, T);
	} else {
		if (T == 0xFF)
			coder->ST++;
		else {
			output_stacked_XFFs(coder);
			emit_byte(coder, T);
		}
	}
	coder->C &= 0x7FFFF;
}

/** Sets as many low order bits of the code register to zero as possible
 *  without pointing outside of the final interval
 */
void clear_final_bits(struct gpujpeg_qm_cpu_encoder* coder) {
	unsigned int T = coder->C + coder->A - 1;
	T &= 0xFFFF0000;
	if (T < coder->C)
		T += 0x8000;
	coder->C = T;
}

/** Terminates the arithmetic encoding procedures and prepares
 *  the entropy-coded segment for the addition of the X’FF’ 
 */
void flush(struct gpujpeg_qm_cpu_encoder* coder) {
	clear_final_bits(coder);
    coder->C <<= coder->CT;
	byte_out(coder);
	coder->C <<= 8;
	byte_out(coder);
	if (coder->last_byte != 0) gpujpeg_writer_emit_byte(coder->writer, coder->last_byte);
}

/** Performs encoder renormalization */
void renorm_e(struct gpujpeg_qm_cpu_encoder* coder) {
	do {
		coder->A <<= 1;
		coder->C <<= 1;
		coder->CT--;
		if (coder->CT == 0) {
			byte_out(coder);
			coder->CT = 8;
		}
	} while (coder->A < 0x8000);
}

/** Estimates new Stat after LPS coding.
 *  May also change sense of MPS.
 */
void estimate_Qe_after_LPS(struct gpujpeg_qm_cpu_encoder* coder, Stat* S) {
	int I = S->index;
	if (arithTable[I].switchMPS == 1)
		S->mps = 1 - S->mps;
	I = arithTable[I].nextIndexLPS;
	S->index = I;
}

/** Estimates new Stat after MPS coding. */
void estimate_Qe_after_MPS(struct gpujpeg_qm_cpu_encoder* coder, Stat* S) {
	int I = S->index;
	I = arithTable[I].nextIndexMPS;
	S->index = I;
}

/** Codes LPS. */
void code_LPS(struct gpujpeg_qm_cpu_encoder* coder, Stat* S) {
	coder->A -= arithTable[S->index].Q_e;
	if (coder->A >= arithTable[S->index].Q_e) {
		coder->C += coder->A;
		coder->A = arithTable[S->index].Q_e;
	}
	estimate_Qe_after_LPS(coder, S);
	renorm_e(coder);
}

/** Codes MPS. */
void code_MPS(struct gpujpeg_qm_cpu_encoder* coder, Stat* S) {
	coder->A -= arithTable[S->index].Q_e;
	if (coder->A < 0x8000) {
		if (coder->A < arithTable[S->index].Q_e) {
			coder->C += coder->A;
			coder->A = arithTable[S->index].Q_e; 
		}
		estimate_Qe_after_MPS(coder, S);
		renorm_e(coder);
	}
}

/** Codes bit 1. */
void code_1(struct gpujpeg_qm_cpu_encoder* coder, Stat* S) {
	if (S->mps == 1) {
		code_MPS(coder, S);
	}
	else {
		code_LPS(coder, S);
	}
}

/** Codes bit 0. */
void code_0(struct gpujpeg_qm_cpu_encoder* coder, Stat* S) {
	if (S->mps == 0) {
        code_MPS(coder, S);
	}
    else {
        code_LPS(coder, S);
	}
}

/** Encodes sign of DC difference. */
int encode_sign_of_V_DC(struct gpujpeg_qm_cpu_encoder* coder, int S_i, const int V) {
	if (V < 0) {
		code_1(coder, &coder->DC_stats[coder->c_type][S_i + 1]); // code SS
		S_i += 3; //S = SN
		coder->previous_diff_index[coder->current_comp] = 8; //-small
	} else {
		code_0(coder, &coder->DC_stats[coder->c_type][S_i + 1]);
		S_i += 2; // S = SP
		coder->previous_diff_index[coder->current_comp] = 4; //+small
	}
	if (abs(V) > 2)
		coder->previous_diff_index[coder->current_comp] += 8; //+-large
	return S_i;
}

/** Encodes sign of AC coefficient. */
int encode_sign_of_V_AC(struct gpujpeg_qm_cpu_encoder* coder, int S_i, const int V) {
    if (V < 0)
        code_1(coder, &coder->fixed); // code SS
    else
        code_0(coder, &coder->fixed); // code SS
    return S_i + 1; // S = SP/SN
}

/** Encodes magnitude category of DC difference.  */
int encode_log2_Sz_DC(struct gpujpeg_qm_cpu_encoder* coder, int Sz, int S_i, int* M) {
	if (Sz >= *M) {
		code_1(coder, &coder->DC_stats[coder->c_type][S_i]);
		*M = 2;
		S_i = 20; // S = X1
		if (Sz >= *M) {
			code_1(coder, &coder->DC_stats[coder->c_type][S_i]);
			*M = 4;
			S_i++; // S = X2
			while (Sz >= *M) {
				code_1(coder, &coder->DC_stats[coder->c_type][S_i]);
				*M <<= 1;
				S_i++;
			}
		}
	}
	code_0(coder, &coder->DC_stats[coder->c_type][S_i]);
	*M >>= 1;
	return S_i;
}

/** Encodes magnitude category of AC coefficient */
int encode_log2_Sz_AC(struct gpujpeg_qm_cpu_encoder* coder, int Sz, int S_i, const int K, int* M) {
	if (Sz >= *M) {
        code_1(coder, &coder->AC_stats[coder->c_type][S_i]);
        *M = 2;
        if (Sz >= *M) {
            code_1(coder, &coder->AC_stats[coder->c_type][S_i]);
            *M = 4;
            S_i = K <= 5 ? 189 : 217; // S = X2
            while (Sz >= *M) {
                code_1(coder, &coder->AC_stats[coder->c_type][S_i]);
                *M <<= 1;
                S_i++;

            }
        }
    }
    code_0(coder, &coder->AC_stats[coder->c_type][S_i]);
    *M >>= 1;
    return S_i;
}

/** Encodes value of magnitude of DC difference or AC coefficient  */
void encode_Sz_bits(struct gpujpeg_qm_cpu_encoder* coder, const int Sz, int S_i, int M, bool isDC) {
	S_i += 14;
	while (M >>= 1) {
		int T = M & Sz;
		if (T == 0)
			if (isDC)
				code_0(coder, &coder->DC_stats[coder->c_type][S_i]);
			else
				code_0(coder, &coder->AC_stats[coder->c_type][S_i]);
		else
			if (isDC)
				code_1(coder, &coder->DC_stats[coder->c_type][S_i]);
			else
				code_1(coder, &coder->AC_stats[coder->c_type][S_i]);
	}
}

/** Encodes DC difference if difference is not equal to 0 */
void encode_V_DC(struct gpujpeg_qm_cpu_encoder* coder, int S_i, const int V) {
	int s_i = encode_sign_of_V_DC(coder, S_i, V);
	int Sz = abs(V) - 1;
	int M = 1;
	s_i = encode_log2_Sz_DC(coder, Sz, s_i, &M);
	encode_Sz_bits(coder, Sz, s_i, M, true);
}

/** Encodes AC coefficient */
void encode_V_AC(struct gpujpeg_qm_cpu_encoder* coder, int S_i, const int V, int K) {
	int s_i = encode_sign_of_V_AC(coder, S_i, V);
    int Sz = abs(V) - 1;
	int M = 1;
    s_i = encode_log2_Sz_AC(coder, Sz, s_i, K, &M);
    encode_Sz_bits(coder, Sz, s_i, M, false);
}

/** Encodes DC difference (F.1.4.1) */
void encode_DC_diff(struct gpujpeg_qm_cpu_encoder* coder, const int dc) {
	int V = dc - coder->previous_DC[coder->current_comp];
	coder->previous_DC[coder->current_comp] = dc;
	int S0 = coder->previous_diff_index[coder->current_comp];
	if (V == 0) {
		code_0(coder, &coder->DC_stats[coder->c_type][S0]);
		coder->previous_diff_index[coder->current_comp] = 0; //zero diff
	} else {
		code_1(coder, &coder->DC_stats[coder->c_type][S0]);
		encode_V_DC(coder, S0, V);
	}
}

/** Encodes all AC coefficients (F.1.4.2) */
void encode_AC_coefficients(struct gpujpeg_qm_cpu_encoder* coder, int16_t* block) {
	int K = 1;
	int SE = 3 * (K - 1);

	// Finds EOB index. EOB is index of last non-null coefficient in zig-zag sequence increased by 1.
	int EOB = 63;
	for (; EOB > 0; EOB--)
		if (block[gpujpeg_order_natural[EOB]]) break;
	EOB++;

	// Loop through AC coefficients.
	while(true) {
		if (K == EOB) {
			code_1(coder, &coder->AC_stats[coder->c_type][SE]);
			return;
		} else {
			code_0(coder, &coder->AC_stats[coder->c_type][SE]);
			int V = block[gpujpeg_order_natural[K]];
			// V == 0 is simply encoded as 0, no encode_V needed.
			while (V == 0) {
				code_0(coder, &coder->AC_stats[coder->c_type][SE + 1]);
				K++;
				SE += 3;
				V = block[gpujpeg_order_natural[K]];
			}
			code_1(coder, &coder->AC_stats[coder->c_type][SE + 1]);
			encode_V_AC(coder, SE + 1, V, K);
			if (K == 63) return; // K == Se
			K++;
			SE = 3 * (K - 1);
		}
	}
}

/** Encodes 8x8 block of DCT coefficients */
void gpujpeg_qm_cpu_encoder_encode_block(struct gpujpeg_qm_cpu_encoder* coder, int16_t *block) {
	encode_DC_diff(coder, block[gpujpeg_order_natural[0]]);
	encode_AC_coefficients(coder, block);
}


/**
 * Encode one MCU
 *
 * @return 0 if succeeds, otherwise nonzero
 */
int
gpujpeg_qm_cpu_encoder_encode_mcu(struct gpujpeg_qm_cpu_encoder* coder, int segment_index, int mcu_index)
{
    // Non-interleaving mode
    if ( coder->comp_count == 1) {
		coder->current_comp = 0;
        // Get component for current scan
        struct gpujpeg_component* component = &coder->component[0];
        // Get component data for MCU
        int16_t* block = &component->data_quantized[(segment_index * component->segment_mcu_count + mcu_index) * component->mcu_size];
		gpujpeg_qm_cpu_encoder_encode_block(coder, block);
    } else {
        for ( int comp = 0; comp < coder->comp_count; comp++ ) {
            struct gpujpeg_component* component = &coder->component[comp];
			coder->current_comp = comp;
			coder->c_type = component->type;
            // Prepare mcu indexes
            int mcu_index_x = (segment_index * component->segment_mcu_count + mcu_index) % component->mcu_count_x;
            int mcu_index_y = (segment_index * component->segment_mcu_count + mcu_index) / component->mcu_count_x;
            // Compute base data index
            int data_index_base = mcu_index_y * (component->mcu_size * component->mcu_count_x) + mcu_index_x * (component->mcu_size_x * GPUJPEG_BLOCK_SIZE);
            // For all vertical 8x8 blocks
            for ( int y = 0; y < component->sampling_factor.vertical; y++ ) {
                // Compute base row data index
                assert((component->mcu_count_x * component->mcu_size_x) == component->data_width);
                int data_index_row = data_index_base + y * (component->mcu_count_x * component->mcu_size_x * GPUJPEG_BLOCK_SIZE);
                // For all horizontal 8x8 blocks
                for ( int x = 0; x < component->sampling_factor.horizontal; x++ ) {
                    // Compute 8x8 block data index
                    int data_index = data_index_row + x * GPUJPEG_BLOCK_SIZE * GPUJPEG_BLOCK_SIZE;
                    
                    // Get component data for MCU
                    int16_t* block = &component->data_quantized[data_index];
                    // Encode 8x8 block
                    gpujpeg_qm_cpu_encoder_encode_block(coder, block);
                }
            }
        }
	}
	return 0;
}

/** Documented at declaration */
int gpujpeg_qm_cpu_encoder_encode(struct gpujpeg_encoder* encoder) {

	gpujpeg_qm_cpu_encoder coder;
	gpujpeg_writer_write_scan_header(encoder, 0);

    // Encode all segments
    for ( int segment_index = 0; segment_index < encoder->coder.segment_count; segment_index++ ) {
		coder = init(encoder);
		struct gpujpeg_segment* segment = &encoder->coder.segment[segment_index];
        // Encode segment MCUs
        for ( int mcu_index = 0; mcu_index < segment->mcu_count; mcu_index++ ) {
            if ( gpujpeg_qm_cpu_encoder_encode_mcu(&coder, segment->scan_segment_index, mcu_index) != 0 ) {
                fprintf(stderr, "[GPUJPEG] [Error] QM encoder failed at block [%d, %d]!\n", segment_index, mcu_index);
                return -1;
            }
        }
		if (segment_index + 1 < encoder->coder.segment_count && encoder->coder.param.restart_interval) {
			flush(&coder);
            // Output restart marker
            int restart_marker = GPUJPEG_MARKER_RST0 + (segment->scan_segment_index & 0x7);
            gpujpeg_writer_emit_marker(encoder->writer, restart_marker);
        }
    }
	if (!encoder->coder.param.restart_interval) flush(&coder);
    return 0;
}

//Test from K.4.1
void test_coder(struct gpujpeg_encoder* encoder) {
	std::string binary = "0000000000000010000000000101000100000000000000000000000011000000000000110101001010000111001010101010101010101010101010101010101010000010110000000010000000000000111111001101011110011110111101100111010011101010101010111111011101101001011111101110011101001100";

    gpujpeg_qm_cpu_encoder coder = init(encoder);

    Stat stat;
    stat.mps = 0;
    stat.index = 0;

    for (int i = 0; i < binary.length(); i++) {
        std::cout << std::dec << i + 1 << " -- D: " << binary[i]  << ", mps: " << stat.mps << ", Qe: " << std::hex
				  << arithTable[stat.index].Q_e << ", A: " << std::hex << coder.A << ", C: " << std::hex << coder.C
				  << ", CT: " << std::dec << coder.CT
				  << ", ST: " << coder.ST << "\n";

        if (binary[i] == '1') code_1(&coder, &stat);
        else code_0(&coder, &stat);
    }
    
	std::cout << std::dec << "flush" << " -- A: " << std::hex << coder.A
              << ", C: " << std::hex << coder.C << ", CT: " << std::dec << coder.CT << ", ST: " << coder.ST << "\n";
    flush(&coder);
}

