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

#ifndef JPEG_TABLE
#define JPEG_TABLE

#include "jpeg_type.h"

static const int jpeg_order_zigzag[64] = { 
     0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63 
};

static const int jpeg_order_natural[64] = {
    0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63
};

/** JPEG table for huffman coding */
struct jpeg_table_huffman {
    // Code for each symbol 
    unsigned int code[256];	
    // Length of code for each symbol 
	char size[256];
    // If no code has been allocated for a symbol S, size[S] is 0 

	// These two fields directly represent the contents of a JPEG DHT marker
    // bits[k] = # of symbols with codes of length k bits; bits[0] is unused
    unsigned char bits[17];
    // The symbols, in order of incr code length
    // This field is used only during compression.  It's initialized false when
	// the table is created, and set true when it's been output to the file.
	// You could suppress output of a table by setting this to true.
	// (See jpeg_suppress_tables for an example.)
    unsigned char huffval[256];		
};

/** JPEG table structure for one component type (luminance/chrominance) */
struct jpeg_table
{
    // Quantization raw table
    uint8_t table_raw[64];
    // Quantization forward table
    uint16_t table_forward[64];
    // Quantization inverse table
    uint16_t table_inverse[64];
    // Quantization forward table in device memory
    uint16_t* d_table_forward;
    // Quantization inverse table in device memory
    uint16_t* d_table_inverse;
    // Huffman table DC
    struct jpeg_table_huffman table_huffman_dc;
    // Huffman table AC
    struct jpeg_table_huffman table_huffman_ac;
};

/**
 * Create JPEG table for quantizatin
 * 
 * @param type  Type of component for table
 * @param quality  Quality (0-100)
 * @return table structure if succeeds, otherwise NULL
 */
struct jpeg_table*
jpeg_table_create(enum jpeg_component_type type, int quality);

/**
 * Destory JPEG table
 * 
 * @param table  Table structure
 * @return 0 if succeeds, otherwise nonzero
 */
int
jpeg_table_destroy(struct jpeg_table* table);

/**
 * Print JPEG table
 * 
 * @param table  Table structure
 * @return void
 */
void
jpeg_table_print(struct jpeg_table* table);

#endif // JPEG_TABLE