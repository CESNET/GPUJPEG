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

/**
 * JPEG table structure
 */
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