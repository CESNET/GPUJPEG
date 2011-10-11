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

/** Documented at declaration */
struct jpeg_table*
jpeg_table_create(enum jpeg_table_type type, int quality)
{
    struct jpeg_table* table = malloc(sizeof(struct jpeg_table));
    if ( table == NULL )
        return NULL;
        
    // Setup raw table
    nppiSetDefaultQuantTable(table->table_raw, (int)type);
    
    // Setup forward table
    //nppiQuantFwdTableInit_JPEG_8u16u(table->table_raw, table->table_forward);
    for (int i = 0; i < 64; ++i) {
        table->table_forward[convert_zigzag2natural[i]] = ((1 << 15) / (double) table->table_raw[i]) + 0.5;
    }
    
    // Setup inverse table
    //nppiQuantInvTableInit_JPEG_8u16u(table->table_raw, table->table_inverse);
    for (int i = 0; i < 64; ++i) {
        table->table_inverse[convert_zigzag2natural[i]] = table->table_raw[i];
    }
    
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
