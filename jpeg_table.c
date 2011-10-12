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
 * Table used to convert zigzad order to the natural-order
 */
const int convert_zigzag2natural[64] = { 
     0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63 
};

/**
 * Raw Quantization Table
 */
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
        table->table_forward[convert_zigzag2natural[i]] = (scale / (double) table->table_raw[i]) + 0.5;
    }
    // Setup forward table by npp (with bug)
    //nppiQuantFwdTableInit_JPEG_8u16u(table->table_raw, table->table_forward);
    
    // Setup inverse table
    for (int i = 0; i < 64; ++i) {
        table->table_inverse[convert_zigzag2natural[i]] = table->table_raw[i];
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
