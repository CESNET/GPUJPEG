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
 
#include "jpeg_huffman_gpu_decoder.h"
#include "jpeg_format_type.h"
#include "jpeg_util.h"

/** Natural order in constant memory */
__constant__ int jpeg_huffman_gpu_decoder_order_natural[64];

/** Documented at declaration */
int
jpeg_huffman_gpu_decoder_init()
{
    // Copy natural order to constant device memory
    cudaMemcpyToSymbol(
        "jpeg_huffman_gpu_decoder_order_natural",
        jpeg_order_natural, 
        64 * sizeof(int),
        0,
        cudaMemcpyHostToDevice
    );
    cudaCheckError("Huffman decoder init");
    
    return 0;
}

/** Documented at declaration */
int
jpeg_huffman_gpu_decoder_decode(struct jpeg_decoder* decoder)
{    
    /*int block_width = 8;
    int block_height = 8;
    int block_cx = (encoder->width + block_width - 1) / block_width;
    int block_cy = (encoder->height + block_height - 1) / block_height;
    int block_count = block_cx * block_cy;
    int segment_count = (block_count / encoder->restart_interval + 1);
            
    // Run kernel
    dim3 thread(32);
    dim3 grid(segment_count / thread.x + 1, encoder->comp_count);
    jpeg_huffman_encoder_encode_kernel<<<grid, thread>>>(
        encoder->restart_interval,
        block_count, 
        encoder->d_segments, 
        segment_count,        
        encoder->d_data_quantized, 
        encoder->d_data_compressed, 
        encoder->d_table_huffman[JPEG_COMPONENT_LUMINANCE][JPEG_HUFFMAN_DC],
        encoder->d_table_huffman[JPEG_COMPONENT_LUMINANCE][JPEG_HUFFMAN_AC],
        encoder->d_table_huffman[JPEG_COMPONENT_CHROMINANCE][JPEG_HUFFMAN_DC],
        encoder->d_table_huffman[JPEG_COMPONENT_CHROMINANCE][JPEG_HUFFMAN_AC]
    );
    cudaError cuerr = cudaThreadSynchronize();
    if ( cuerr != cudaSuccess ) {
        fprintf(stderr, "Huffman decoding failed: %s!\n", cudaGetErrorString(cuerr));
        return -1;
    }*/
    
    return 0;
}