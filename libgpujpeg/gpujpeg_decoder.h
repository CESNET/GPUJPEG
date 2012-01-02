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

#ifndef GPUJPEG_DECODER_H
#define GPUJPEG_DECODER_H

#include "gpujpeg_common.h"
#include "gpujpeg_table.h"
#include "gpujpeg_reader.h"

/** JPEG reader scan structure */
struct gpujpeg_decoder_scan
{
    // Index into array of segment indexes [decoder->data_scan_index] for the first byte of scan
    int segment_index;
    // Segment count
    int segment_count;
};

/**
 * JPEG decoder structure
 */
struct gpujpeg_decoder
{  
    // Parameters for image data (width, height, comp_count, etc.)
    struct gpujpeg_image_parameters param_image;
    
    // Quantization tables
    struct gpujpeg_table_quantization table_quantization[GPUJPEG_COMPONENT_TYPE_COUNT];
    
    // Huffman coder tables
    struct gpujpeg_table_huffman_decoder table_huffman[GPUJPEG_COMPONENT_TYPE_COUNT][GPUJPEG_HUFFMAN_TYPE_COUNT];
    // Huffman coder tables in device memory
    struct gpujpeg_table_huffman_decoder* d_table_huffman[GPUJPEG_COMPONENT_TYPE_COUNT][GPUJPEG_HUFFMAN_TYPE_COUNT];
    
    // JPEG reader structure
    struct gpujpeg_reader* reader;
    
    // Restart interval for all scans (number of MCU that can be coded independatly, 
    // 0 means seqeuential coding, 1 means every MCU can be coded independantly)
    int restart_interval;
    
    // Flag which determines if interleaved format of JPEG stream should be used (only
    // one scan which includes all color components, e.g. Y Cb Cr Y Cb Cr ...),
    // or one scan for each color component (e.g. Y Y Y ..., Cb Cb Cb ..., Cr Cr Cr ...)
    int interleaved;
    
    // Data buffer for all scans
    uint8_t* data_scan;
    // Data buffer for all scans in device memory
    uint8_t* d_data_scan;
    
    // Size for data buffer for all scans
    int data_scan_size;
    
    // Indexes into scan data buffer for all segments (index point to segment data start in buffer)
    int* data_scan_index;
    // Indexes into scan data buffer for all segments in device memory (index point to segment data start in buffer)
    int* d_data_scan_index;
    
    // Total segment count for all scans
    int segment_count;
    
    // Allocated data width
    int data_width;
    // Allocated data height
    int data_height;
    // Allocated data coefficient count
    int data_size;
    
    // Data quantized (output from huffman coder)
    int16_t* data_quantized;
    // Data quantized in device memory (output from huffman coder)
    int16_t* d_data_quantized;
    
    // Data in device memory (output from inverse DCT and quantization)
    uint8_t* d_data;
    
    // Target image data coefficient count
    int data_target_size;
    
    // Data target (output from preprocessing)
    uint8_t* data_target;
    // Data target in device memory (output from preprocessing)
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
struct gpujpeg_decoder*
gpujpeg_decoder_create(struct gpujpeg_image_parameters* param_image);

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
gpujpeg_decoder_init(struct gpujpeg_decoder* decoder, int width, int height, int comp_count);

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
gpujpeg_decoder_decode(struct gpujpeg_decoder* decoder, uint8_t* image, int image_size, uint8_t** image_decompressed, int* image_decompressed_size);

/**
 * Destory JPEG decoder
 * 
 * @param decoder  Decoder structure
 * @return 0 if succeeds, otherwise nonzero
 */
int
gpujpeg_decoder_destroy(struct gpujpeg_decoder* decoder);

#endif // GPUJPEG_DECODER_H
