/**
 * @file
 * Copyright (c) 2011-2023, CESNET z.s.p.o
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

#ifndef GPUJPEG_DECODER_INTERNAL_H
#define GPUJPEG_DECODER_INTERNAL_H

#include "../libgpujpeg/gpujpeg_common.h"
#include "gpujpeg_common_internal.h"
#include "gpujpeg_table.h"

/**
 * JPEG decoder structure
 */
struct gpujpeg_decoder
{
    /// JPEG coder structure
    struct gpujpeg_coder coder;
    
    struct gpujpeg_huffman_gpu_decoder *huffman_gpu_decoder;

    uint8_t comp_id[GPUJPEG_MAX_COMPONENT_COUNT]; /// component IDs defined by SOF
    
    /// Quantization tables
    struct gpujpeg_table_quantization table_quantization[GPUJPEG_MAX_COMPONENT_COUNT];
    int comp_table_quantization_map[GPUJPEG_MAX_COMPONENT_COUNT]; ///< mapping component -> index to table_quantization
    
    /// Huffman coder tables
    struct gpujpeg_table_huffman_decoder table_huffman[GPUJPEG_MAX_COMPONENT_COUNT][GPUJPEG_HUFFMAN_TYPE_COUNT];
    // Huffman coder tables in device memory
    struct gpujpeg_table_huffman_decoder* d_table_huffman[GPUJPEG_MAX_COMPONENT_COUNT][GPUJPEG_HUFFMAN_TYPE_COUNT];
    int comp_table_huffman_map[GPUJPEG_MAX_COMPONENT_COUNT][GPUJPEG_HUFFMAN_TYPE_COUNT]; ///< mapping component -> indices (AC/DC) to table_huffman
    
    /// Current segment count for decoded image
    int segment_count;
    
    /// Current data compressed size for decoded image
    size_t data_compressed_size;

    // Stream
    cudaStream_t stream;

    enum gpujpeg_pixel_format req_pixel_format;
};

#endif // GPUJPEG_DECODER_INTERNAL_H

