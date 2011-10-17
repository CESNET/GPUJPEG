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
 
#include "jpeg_decoder.h"
#include "jpeg_huffman_decoder.h"
#include "jpeg_preprocessor.h"
#include "jpeg_util.h"

/** Documented at declaration */
struct jpeg_decoder*
jpeg_decoder_create(int width, int height, int comp_count)
{
    struct jpeg_decoder* decoder = malloc(sizeof(struct jpeg_decoder));
    if ( decoder == NULL )
        return NULL;
        
    // Set parameters
    decoder->width = 0;
    decoder->height = 0;
    decoder->comp_count = 0;
    decoder->restart_interval = 0;
    decoder->data_quantized = NULL;
    decoder->d_data_quantized = NULL;
    decoder->d_data = NULL;
    decoder->data_target = NULL;
    decoder->d_data_target = NULL;
    for ( int comp = 0; comp < JPEG_MAX_COMPONENT_COUNT; comp++ )
        decoder->scan[comp].data = NULL;
    
    int result = 1;
    
    // Create reader
    decoder->reader = jpeg_reader_create();
    if ( decoder->reader == NULL )
        result = 0;
    
    // Allocate quantization tables in device memory
    for ( int comp_type = 0; comp_type < JPEG_COMPONENT_TYPE_COUNT; comp_type++ ) {
        if ( cudaSuccess != cudaMalloc((void**)&decoder->table_quantization[comp_type].d_table, 64 * sizeof(uint16_t)) ) 
            result = 0;
    }
    
    // Init decoder
    if ( width != 0 && height != 0 ) {
        if ( jpeg_decoder_init(decoder, width, height, comp_count) != 0 )
            result = 0;
    }
    
    if ( result == 0 ) {
        jpeg_decoder_destroy(decoder);
        return NULL;
    }
    
    return decoder;
}

/** Documented at declaration */
int
jpeg_decoder_init(struct jpeg_decoder* decoder, int width, int height, int comp_count)
{
    assert(comp_count == 3);
    
    // No reinialization needed
    if ( decoder->width == width && decoder->height == height && decoder->comp_count == comp_count ) {
        return 0;
    }
    
    // For now we can't reinitialize decoder, we can only do first initialization
    if ( decoder->width != 0 || decoder->height != 0 || decoder->comp_count != 0 ) {
        fprintf(stderr, "Can't reinitialize decoder, implement if needed!\n");
        return -1;
    }
    
    decoder->width = width;
    decoder->height = height;
    decoder->comp_count = comp_count;
    
    // Allocate scans
    int data_comp_size = decoder->width * decoder->height;
    for ( int comp = 0; comp < decoder->comp_count; comp++ ) {
        decoder->scan[comp].data = malloc(data_comp_size * sizeof(uint8_t));
        if ( decoder->scan[comp].data == NULL )
            return -1;
    }
    
    // Allocate buffers
    int data_size = decoder->width * decoder->width * decoder->comp_count;
    if ( cudaSuccess != cudaMallocHost((void**)&decoder->data_quantized, data_size * sizeof(int16_t)) ) 
        return -1;
    if ( cudaSuccess != cudaMalloc((void**)&decoder->d_data_quantized, data_size * sizeof(int16_t)) ) 
        return -1;
    if ( cudaSuccess != cudaMalloc((void**)&decoder->d_data, data_size * sizeof(uint8_t)) ) 
        return -1;
    if ( cudaSuccess != cudaMallocHost((void**)&decoder->data_target, data_size * sizeof(uint8_t)) ) 
        return -1;
    if ( cudaSuccess != cudaMalloc((void**)&decoder->d_data_target, data_size * sizeof(uint8_t)) ) 
        return -1;
    
    return 0;
}

void
jpeg_decoder_print8(struct jpeg_decoder* decoder, uint8_t* d_data)
{
    int data_size = decoder->width * decoder->height;
    uint8_t* data = NULL;
    cudaMallocHost((void**)&data, data_size * sizeof(uint8_t)); 
    cudaMemcpy(data, d_data, data_size * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    
    printf("Print Data\n");
    for ( int y = 0; y < decoder->height; y++ ) {
        for ( int x = 0; x < decoder->width; x++ ) {
            printf("%3u ", data[y * decoder->width + x]);
        }
        printf("\n");
    }
    cudaFreeHost(data);
}

void
jpeg_decoder_print16(struct jpeg_decoder* decoder, int16_t* d_data)
{
    int data_size = decoder->width * decoder->height;
    int16_t* data = NULL;
    cudaMallocHost((void**)&data, data_size * sizeof(int16_t)); 
    cudaMemcpy(data, d_data, data_size * sizeof(int16_t), cudaMemcpyDeviceToHost);
    
    printf("Print Data\n");
    for ( int y = 0; y < decoder->height; y++ ) {
        for ( int x = 0; x < decoder->width; x++ ) {
            printf("%3d ", data[y * decoder->width + x]);
        }
        printf("\n");
    }
    cudaFreeHost(data);
}

/** Documented at declaration */
int
jpeg_decoder_decode(struct jpeg_decoder* decoder, uint8_t* image, int image_size, uint8_t** image_decompressed, int* image_decompressed_size)
{
    int data_size = decoder->width * decoder->height * decoder->comp_count;
    
    // Read JPEG image data
    if ( jpeg_reader_read_image(decoder, image, image_size) != 0 ) {
        fprintf(stderr, "Decoder failed when decoding image data!\n");
        return -1;
    }
    
    // Perform huffman decoding for all components
    for ( int index = 0; index < decoder->scan_count; index++ ) {
        // Get scan and data buffer
        struct jpeg_decoder_scan* scan = &decoder->scan[index];
        int16_t* data_quantized_comp = &decoder->data_quantized[index * decoder->width * decoder->height];
        // Determine table type
        enum jpeg_component_type type = (index == 0) ? JPEG_COMPONENT_LUMINANCE : JPEG_COMPONENT_CHROMINANCE;
        // Huffman decode
        if ( jpeg_huffman_decoder_decode(decoder, type, scan->data, scan->data_size, data_quantized_comp) != 0 ) {
            fprintf(stderr, "Huffman decoder failed for scan at index %d!\n", index);
            return -1;
        }
    }
    
    // Copy quantized data to device memory from cpu memory    
    cudaMemcpy(decoder->d_data_quantized, decoder->data_quantized, data_size * sizeof(int16_t), cudaMemcpyHostToDevice);
    
    // Perform IDCT and dequantization
    for ( int comp = 0; comp < decoder->comp_count; comp++ ) {
        uint8_t* d_data_comp = &decoder->d_data[comp * decoder->width * decoder->height];
        int16_t* d_data_quantized_comp = &decoder->d_data_quantized[comp * decoder->width * decoder->height];
        
        // Determine table type
        enum jpeg_component_type type = (comp == 0) ? JPEG_COMPONENT_LUMINANCE : JPEG_COMPONENT_CHROMINANCE;
        
        //jpeg_decoder_print16(decoder, d_data_quantized_comp);
        
        cudaMemset(d_data_comp, 0, decoder->width * decoder->height * sizeof(int16_t));
        
        //Perform inverse DCT
        NppiSize inv_roi;
        inv_roi.width = 64 * decoder->width / 8;
        inv_roi.height = decoder->height / 8;
        NppStatus status = nppiDCTQuantInv8x8LS_JPEG_16s8u_C1R(
            d_data_quantized_comp, 
            decoder->width * 8 * sizeof(int16_t), 
            d_data_comp, 
            decoder->width * sizeof(uint8_t), 
            decoder->table_quantization[type].d_table, 
            inv_roi
        );
        if ( status != 0 )
            printf("Error %d\n", status);
        
        //jpeg_decoder_print8(decoder, d_data_comp);
    }
    
    // Preprocessing
    if ( jpeg_preprocessor_decode(decoder) != 0 )
        return -1;
    
    cudaMemcpy(decoder->data_target, decoder->d_data_target, data_size * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    
    // Set decompressed image
    *image_decompressed = decoder->data_target;
    *image_decompressed_size = data_size * sizeof(uint8_t);
    
    return 0;
}

/** Documented at declaration */
int
jpeg_decoder_destroy(struct jpeg_decoder* decoder)
{
    assert(decoder != NULL);
    
    for ( int comp_type = 0; comp_type < JPEG_COMPONENT_TYPE_COUNT; comp_type++ ) {
        if ( decoder->table_quantization[comp_type].d_table != NULL )
            cudaFree(decoder->table_quantization[comp_type].d_table);
    }
    
    if ( decoder->reader != NULL )
        jpeg_reader_destroy(decoder->reader);
    
    for ( int comp = 0; comp < decoder->comp_count; comp++ ) {
        if ( decoder->scan[comp].data != NULL )
            free(decoder->scan[comp].data);
    }
        
    if ( decoder->data_quantized != NULL )
        cudaFreeHost(decoder->data_quantized);
    if ( decoder->d_data_quantized != NULL )
        cudaFree(decoder->d_data_quantized);
    if ( decoder->d_data != NULL )
        cudaFree(decoder->d_data);
    if ( decoder->data_target != NULL )
        cudaFreeHost(decoder->data_target);
    if ( decoder->d_data_target != NULL )
        cudaFree(decoder->d_data_target);
    
    free(decoder);
    
    return 0;
}
