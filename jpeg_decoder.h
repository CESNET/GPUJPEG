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

#ifndef JPEG_DECODER
#define JPEG_DECODER

#include "jpeg_table.h"
#include "jpeg_reader.h"

#define JPEG_MAX_COMPONENT_COUNT 3

/** JPEG reader scan structure */
struct jpeg_decoder_scan
{
    uint8_t* data;
    int data_size;
};

/**
 * JPEG decoder structure
 */
struct jpeg_decoder
{  
    // Image width
    int width;
    // Image height
    int height;
    // Component count
    int comp_count;
    
    // Quantization tables
    struct jpeg_table_quantization table_quantization[JPEG_COMPONENT_TYPE_COUNT];
    
    // Huffman coder tables
    struct jpeg_table_huffman_decoder table_huffman[JPEG_COMPONENT_TYPE_COUNT][JPEG_HUFFMAN_TYPE_COUNT];
    
    // JPEG reader structure
    struct jpeg_reader* reader;
    
    // Scan count
    int scan_count;
    
    // Scan definitions
    struct jpeg_decoder_scan scan[JPEG_MAX_COMPONENT_COUNT];
    
    // Scan restart interval
    int restart_interval;
    
    // Data quantized
    int16_t* data_quantized;
    
    // Data quantized in device memory
    int16_t* d_data_quantized;
    
    // Datain device memory
    uint8_t* d_data;
    
    // Data target
    uint8_t* data_target;
    
    // Data target in device memory
    uint8_t* d_data_target;
};

/**
 * Create JPEG decoder
 * 
 * @param width  Width of decodable images
 * @param height  Height of decodable images
 * @param comp_count  Component count
 * @return encoder structure if succeeds, otherwise NULL
 */
struct jpeg_decoder*
jpeg_decoder_create(int width, int height, int comp_count);

/**
 * Init JPEG decoder for specific image size
 * 
 * @param decoder  Decoder structure
 * @param width  Width of decodable images
 * @param height  Height of decodable images
 * @param comp_count  Component count
 * @return 0 if succeeds, otherwise nonzero
 */
int
jpeg_decoder_init(struct jpeg_decoder* decoder, int width, int height, int comp_count);

/**
 * Decompress image by decoder
 * 
 * @param decoder  Decoder structure
 * @param image  Source image data
 * @param image_size  Source image data size
 * @param image_decompressed  Pointer to variable where decompressed image data buffer will be placed
 * @param image_decompressed_size  Pointer to variable where decompressed image size will be placed
 * @return 0 if succeeds, otherwise nonzero
 */
int
jpeg_decoder_decode(struct jpeg_decoder* decoder, uint8_t* image, int image_size, uint8_t** image_decompressed, int* image_decompressed_size);

/**
 * Destory JPEG decoder
 * 
 * @param decoder  Decoder structure
 * @return 0 if succeeds, otherwise nonzero
 */
int
jpeg_decoder_destroy(struct jpeg_decoder* decoder);

#endif // JPEG_ENCODER