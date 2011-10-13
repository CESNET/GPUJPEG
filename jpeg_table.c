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
 
#include "jpeg_table.h"
#include "jpeg_util.h"

/**
 * Initialize huffman DC and AC table for component type
 * 
 * @param table  Table structure that contains all tables for component type
 * @param type  Component type (luminance/chrominance)
 * @return void
 */
void
jpeg_table_init_huffman(struct jpeg_table* table, enum jpeg_component_type type);

/** 
 * Compute huffman table from bits and values arrays
 * 
 * @param bits
 * @param values
 * @return void
 */
void
jpeg_table_compute_huffman(unsigned char* bits, unsigned char* values, struct jpeg_table_huffman* table);

/** Raw Quantization Table */
Npp8u table_raw_default[64] = { 
    16, 11, 12, 14, 12, 10, 16, 14,
    13, 14, 18, 17, 16, 19, 24, 40,
    26, 24, 22, 22, 24, 49, 35, 37,
    29, 40, 58, 51, 61, 60, 57, 51,
    56, 55, 64, 72, 92, 78, 64, 68,
    87, 69, 55, 56, 80, 109, 81, 87,
    95, 98, 103, 104, 103, 62, 77, 113,
    121, 112, 100, 120, 92, 101, 103, 99 
};

/** Documented at declaration */
struct jpeg_table*
jpeg_table_create(enum jpeg_component_type type, int quality)
{
    struct jpeg_table* table = malloc(sizeof(struct jpeg_table));
    if ( table == NULL )
        return NULL;
        
    // Setup raw table
    nppiSetDefaultQuantTable(table->table_raw, (int)type);
    // Other default raw table
    //for ( int i = 0; i < 64; i++ ) {
    //    table->table_raw[i] = table_raw_default[i];
    //}
    
    // Init raw table
    nppiQuantFwdRawTableInit_JPEG_8u(table->table_raw, quality);
    
    // Setup forward table
    const int scale = (1 << 15);
    for (int i = 0; i < 64; ++i) {
        table->table_forward[jpeg_order_natural[i]] = (scale / (double) table->table_raw[i]) + 0.5;
    }
    // Setup forward table by npp (with bug)
    //nppiQuantFwdTableInit_JPEG_8u16u(table->table_raw, table->table_forward);
    
    // Setup inverse table
    for (int i = 0; i < 64; ++i) {
        table->table_inverse[jpeg_order_natural[i]] = table->table_raw[i];
    }
    // Setup inverse table by npp (with bug)
    //nppiQuantInvTableInit_JPEG_8u16u(table->table_raw, table->table_inverse);
    
    // Allocate device memory for tables
    if ( cudaSuccess != cudaMalloc((void**)&table->d_table_forward, 64 * sizeof(uint16_t)) )
        return NULL;
    if ( cudaSuccess != cudaMalloc((void**)&table->d_table_inverse, 64 * sizeof(uint16_t)) )
        return NULL;
    
    // Copy tables to device memory
    if ( cudaSuccess != cudaMemcpy(table->d_table_forward, table->table_forward, 64 * sizeof(uint16_t), cudaMemcpyHostToDevice) )
        return NULL;
    if ( cudaSuccess != cudaMemcpy(table->d_table_inverse, table->table_inverse, 64 * sizeof(uint16_t), cudaMemcpyHostToDevice) )
        return NULL;
        
    // Init huffman tables
    jpeg_table_init_huffman(table, type);
    
    return table;
}

/** Documented at declaration */
int
jpeg_table_destroy(struct jpeg_table* table)
{
    assert(table != NULL);
    cudaFree(table->d_table_forward);
    cudaFree(table->d_table_inverse);
    free(table);
    return 0;
}

/** Documented at declaration */
void
jpeg_table_print(struct jpeg_table* table)
{
    puts("Raw Table (with quality):");
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            printf("%4u", table->table_raw[i * 8 + j]);
        }
        puts("");
    }
    
    puts("Forward Table:");
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            printf("%6u", table->table_forward[i * 8 + j]);
        }
        puts("");
    }
    
    puts("Inverse Table:");
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            printf("%6u", table->table_inverse[i * 8 + j]);
        }
        puts("");
    }
}

/** Documented at declaration */
void
jpeg_table_init_huffman(struct jpeg_table* table, enum jpeg_component_type type) 
{
    // DC for Y component
    static unsigned char bitsYDC[17] = {
        0, 0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0 
    };
	static unsigned char valYDC[] = { 
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 
    };
	// DC for Cb or Cr component
	static unsigned char bitsCbCrDC[17] = { 
        0, 0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0 
    };
	static unsigned char valCbCrDC[] = { 
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 
    };
    
	// AC for Y component
	static unsigned char bitsYAC[17] = { 
        0, 0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d 
    };
	static unsigned char valYAC[] = { 
        0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12,
        0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
        0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08,
        0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0,
        0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16,
        0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28,
        0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39,
        0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
        0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
        0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
        0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79,
        0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
        0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98,
        0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
        0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6,
        0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5,
        0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4,
        0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2,
        0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea,
        0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
        0xf9, 0xfa 
    };
	// AC for Cb or Cr component
	static unsigned char bitsCbCrAC[17] = { 
        0, 0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77 
    };
	static unsigned char valCbCrAC[] = { 
        0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21,
        0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71,
        0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91,
        0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52, 0xf0,
        0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34,
        0xe1, 0x25, 0xf1, 0x17, 0x18, 0x19, 0x1a, 0x26,
        0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38,
        0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
        0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58,
        0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
        0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78,
        0x79, 0x7a, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
        0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96,
        0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5,
        0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4,
        0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3,
        0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2,
        0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda,
        0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9,
        0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
        0xf9, 0xfa 
    };

    // Compute huffman table for correct component type
    if ( type == JPEG_COMPONENT_LUMINANCE ) {
        jpeg_table_compute_huffman(bitsYDC, valYDC, &table->table_huffman_dc);
        jpeg_table_compute_huffman(bitsYAC, valYAC, &table->table_huffman_ac);
    } else if ( type == JPEG_COMPONENT_CHROMINANCE ) {
        jpeg_table_compute_huffman(bitsCbCrDC, valCbCrDC, &table->table_huffman_dc);
        jpeg_table_compute_huffman(bitsCbCrAC, valCbCrAC, &table->table_huffman_ac); 
    } else {
        assert(0);
    }
}

/** Documented at declaration */
void
jpeg_table_compute_huffman(unsigned char* bits, unsigned char* values, struct jpeg_table_huffman* table)
{
	char huffsize[257];
	unsigned int huffcode[257];

	// First we copy bits and huffval
	memcpy(table->bits, bits, sizeof(table->bits));
	memcpy(table->huffval, values, sizeof(table->huffval));

	// Figure C.1: make table of Huffman code length for each symbol
	// Note that this is in code-length order
	int p = 0;
	for ( int l = 1; l <= 16; l++ ) {
		for ( int i = 1; i <= (int) bits[l]; i++ )
			huffsize[p++] = (char) l;
	}
	huffsize[p] = 0;
	int lastp = p;

	// Figure C.2: generate the codes themselves
	// Note that this is in code-length order

	unsigned int code = 0;
	int si = huffsize[0];
	p = 0;
	while ( huffsize[p] ) {
		while ( ((int) huffsize[p]) == si ) {
			huffcode[p++] = code;
			code++;
		}
		code <<= 1;
		si++;
	}

	// Figure C.3: generate encoding tables
	// These are code and size indexed by symbol value

	// Set any codeless symbols to have code length 0;
	// this allows EmitBits to detect any attempt to emit such symbols.
	memset(table->size, 0, sizeof(table->size));

	for (p = 0; p < lastp; p++) {
		table->code[values[p]] = huffcode[p];
		table->size[values[p]] = huffsize[p];
	}
}

