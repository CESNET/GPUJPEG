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
 
#include "gpujpeg_decoder.h"
#include "gpujpeg_preprocessor.h"
#include "gpujpeg_huffman_cpu_decoder.h"
#include "gpujpeg_huffman_gpu_decoder.h"
#include "gpujpeg_util.h"

/** Documented at declaration */
struct gpujpeg_decoder*
gpujpeg_decoder_create(struct gpujpeg_image_parameters* param_image)
{    
    struct gpujpeg_decoder* decoder = malloc(sizeof(struct gpujpeg_decoder));
    if ( decoder == NULL )
        return NULL;
        
    // Set parameters
    decoder->param_image = *param_image;
    decoder->param_image.width = 0;
    decoder->param_image.height = 0;
    decoder->param_image.comp_count = 0;
    decoder->restart_interval = 0;
    decoder->interleaved = 0;
    decoder->data_width = 0;
    decoder->data_height = 0;
    decoder->data_size = 0;
    decoder->data_quantized = NULL;
    decoder->d_data_quantized = NULL;
    decoder->d_data = NULL;
    decoder->data_target = NULL;
    decoder->d_data_target = NULL;
    
    int result = 1;
    
    // Create reader
    decoder->reader = gpujpeg_reader_create();
    if ( decoder->reader == NULL )
        result = 0;
    
    // Allocate quantization tables in device memory
    for ( int comp_type = 0; comp_type < GPUJPEG_COMPONENT_TYPE_COUNT; comp_type++ ) {
        if ( cudaSuccess != cudaMalloc((void**)&decoder->table_quantization[comp_type].d_table, 64 * sizeof(uint16_t)) ) 
            result = 0;
    }
    // Allocate huffman tables in device memory
    for ( int comp_type = 0; comp_type < GPUJPEG_COMPONENT_TYPE_COUNT; comp_type++ ) {
        for ( int huff_type = 0; huff_type < GPUJPEG_HUFFMAN_TYPE_COUNT; huff_type++ ) {
            if ( cudaSuccess != cudaMalloc((void**)&decoder->d_table_huffman[comp_type][huff_type], sizeof(struct gpujpeg_table_huffman_decoder)) )
                result = 0;
        }
    }
    gpujpeg_cuda_check_error("Decoder table allocation");
    
    // Init decoder
    if ( param_image->width != 0 && param_image->height != 0 ) {
        if ( gpujpeg_decoder_init(decoder, param_image->width, param_image->height, param_image->comp_count) != 0 )
            result = 0;
    }
    
    // Init huffman encoder
    if ( gpujpeg_huffman_gpu_decoder_init() != 0 )
        result = 0;
    
    if ( result == 0 ) {
        gpujpeg_decoder_destroy(decoder);
        return NULL;
    }
    
    return decoder;
}

/** Documented at declaration */
int
gpujpeg_decoder_init(struct gpujpeg_decoder* decoder, int width, int height, int comp_count)
{
    assert(comp_count == 3);
    
    // No reinialization needed
    if ( decoder->param_image.width == width && decoder->param_image.height == height && decoder->param_image.comp_count == comp_count ) {
        return 0;
    }
    
    // For now we can't reinitialize decoder, we can only do first initialization
    if ( decoder->param_image.width != 0 || decoder->param_image.height != 0 || decoder->param_image.comp_count != 0 ) {
        fprintf(stderr, "Can't reinitialize decoder, implement if needed!\n");
        return -1;
    }
    
    decoder->param_image.width = width;
    decoder->param_image.height = height;
    decoder->param_image.comp_count = comp_count;    
    
    // Allocate scan data
    int data_scan_size = decoder->param_image.comp_count * decoder->param_image.width * decoder->param_image.height;
    // Add some space for right, bottom corners (when image size isn't divisible by 8)
    data_scan_size += gpujpeg_div_and_round_up(decoder->param_image.width, GPUJPEG_BLOCK_SIZE) * GPUJPEG_BLOCK_SIZE * GPUJPEG_BLOCK_SIZE;
    data_scan_size += gpujpeg_div_and_round_up(decoder->param_image.height, GPUJPEG_BLOCK_SIZE) * GPUJPEG_BLOCK_SIZE * GPUJPEG_BLOCK_SIZE;
    data_scan_size += GPUJPEG_BLOCK_SIZE * GPUJPEG_BLOCK_SIZE;
    // We need more data ie. twice, restart_interval could be 1 so a lot of data
    data_scan_size = data_scan_size * 2;
    // Allocate indexes to data for each segment too
    int max_segment_count = decoder->param_image.comp_count * ((decoder->param_image.width + GPUJPEG_BLOCK_SIZE - 1) / GPUJPEG_BLOCK_SIZE) * ((decoder->param_image.height + GPUJPEG_BLOCK_SIZE - 1) / GPUJPEG_BLOCK_SIZE);
    if ( cudaSuccess != cudaMallocHost((void**)&decoder->data_scan, data_scan_size * sizeof(uint8_t)) ) 
        return -1;
    if ( cudaSuccess != cudaMalloc((void**)&decoder->d_data_scan, data_scan_size * sizeof(uint8_t)) ) 
        return -1;
    if ( cudaSuccess != cudaMallocHost((void**)&decoder->data_scan_index, max_segment_count * sizeof(int)) ) 
        return -1;
    if ( cudaSuccess != cudaMalloc((void**)&decoder->d_data_scan_index, max_segment_count * sizeof(int)) ) 
        return -1;
    gpujpeg_cuda_check_error("Decoder scan allocation");
    
    // Calculate data size
    decoder->data_width = gpujpeg_div_and_round_up(decoder->param_image.width, GPUJPEG_BLOCK_SIZE) * GPUJPEG_BLOCK_SIZE;
    decoder->data_height = gpujpeg_div_and_round_up(decoder->param_image.height, GPUJPEG_BLOCK_SIZE) * GPUJPEG_BLOCK_SIZE;
    decoder->data_size = decoder->data_width * decoder->data_height * decoder->param_image.comp_count;
    decoder->data_target_size = gpujpeg_image_calculate_size(&decoder->param_image);
    
    // Allocate buffers
    if ( cudaSuccess != cudaMallocHost((void**)&decoder->data_quantized, decoder->data_size * sizeof(int16_t)) ) 
        return -1;
    if ( cudaSuccess != cudaMalloc((void**)&decoder->d_data_quantized, decoder->data_size * sizeof(int16_t)) ) 
        return -1;
    if ( cudaSuccess != cudaMalloc((void**)&decoder->d_data, decoder->data_size * sizeof(uint8_t)) ) 
        return -1;
    if ( cudaSuccess != cudaMallocHost((void**)&decoder->data_target, decoder->data_target_size * sizeof(uint8_t)) ) 
        return -1;
    if ( cudaSuccess != cudaMalloc((void**)&decoder->d_data_target, decoder->data_target_size * sizeof(uint8_t)) ) 
        return -1;
    gpujpeg_cuda_check_error("Decoder data allocation");
    
    return 0;
}

void
gpujpeg_decoder_print8(struct gpujpeg_decoder* decoder, uint8_t* d_data)
{
    int data_size = decoder->data_width * decoder->data_height;
    uint8_t* data = NULL;
    cudaMallocHost((void**)&data, data_size * sizeof(uint8_t)); 
    cudaMemcpy(data, d_data, data_size * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    
    printf("Print Data\n");
    for ( int y = 0; y < decoder->data_height; y++ ) {
        for ( int x = 0; x < decoder->data_width; x++ ) {
            printf("%3u ", data[y * decoder->data_width + x]);
        }
        printf("\n");
    }
    cudaFreeHost(data);
}

void
gpujpeg_decoder_print16(struct gpujpeg_decoder* decoder, int16_t* d_data)
{
    int data_size = decoder->data_width * decoder->data_height;
    int16_t* data = NULL;
    cudaMallocHost((void**)&data, data_size * sizeof(int16_t)); 
    cudaMemcpy(data, d_data, data_size * sizeof(int16_t), cudaMemcpyDeviceToHost);
    
    printf("Print Data\n");
    for ( int y = 0; y < decoder->data_height; y++ ) {
        for ( int x = 0; x < decoder->data_width; x++ ) {
            printf("%3d ", data[y * decoder->data_width + x]);
        }
        printf("\n");
    }
    cudaFreeHost(data);
}

/** Documented at declaration */
int
gpujpeg_decoder_decode(struct gpujpeg_decoder* decoder, uint8_t* image, int image_size, uint8_t** image_decompressed, int* image_decompressed_size)
{    
    //GPUJPEG_TIMER_INIT();
    //GPUJPEG_TIMER_START();
    
    // Read JPEG image data
    if ( gpujpeg_reader_read_image(decoder, image, image_size) != 0 ) {
        fprintf(stderr, "Decoder failed when decoding image data!\n");
        return -1;
    }
    
    //GPUJPEG_TIMER_STOP_PRINT("-Stream Reader:     ");
    //GPUJPEG_TIMER_START();
    
    // Perform huffman decoding on CPU (when restart interval is not set)
    if ( decoder->restart_interval == 0 ) {
        // Perform huffman decoding for all components
        for ( int index = 0; index < decoder->reader->scan_count; index++ ) {
            // Get scan and data buffer
            struct gpujpeg_reader_scan* scan = &decoder->reader->scan[index];
            int16_t* data_quantized_comp = &decoder->data_quantized[index * decoder->data_width * decoder->data_height];
            // Determine table type
            enum gpujpeg_component_type type = (index == 0) ? GPUJPEG_COMPONENT_LUMINANCE : GPUJPEG_COMPONENT_CHROMINANCE;
            // Huffman decode
            if ( gpujpeg_huffman_cpu_decoder_decode(decoder, type, scan, data_quantized_comp) != 0 ) {
                fprintf(stderr, "Huffman decoder failed for scan at index %d!\n", index);
                return -1;
            }
        }
        
        // Copy quantized data to device memory from cpu memory    
        cudaMemcpy(decoder->d_data_quantized, decoder->data_quantized, decoder->data_size * sizeof(int16_t), cudaMemcpyHostToDevice);
    }
    // Perform huffman decoding on GPU (when restart interval is set)
    else {
        cudaMemset(decoder->d_data_quantized, 0, decoder->data_size * sizeof(int16_t));
        
        // Copy scan data to device memory
        cudaMemcpy(decoder->d_data_scan, decoder->data_scan, decoder->data_scan_size * sizeof(uint8_t), cudaMemcpyHostToDevice);
        gpujpeg_cuda_check_error("Decoder copy scan data");
        // Copy scan data to device memory
        cudaMemcpy(decoder->d_data_scan_index, decoder->data_scan_index, decoder->segment_count * sizeof(int), cudaMemcpyHostToDevice);
        gpujpeg_cuda_check_error("Decoder copy scan data index");
        
        // Zero output memory
        cudaMemset(decoder->d_data_quantized, 0, decoder->data_size * sizeof(int16_t));
        
        // Perform huffman decoding
        if ( gpujpeg_huffman_gpu_decoder_decode(decoder) != 0 ) {
            fprintf(stderr, "Huffman decoder on GPU failed!\n");
            return -1;
        }
    }
    
    //GPUJPEG_TIMER_STOP_PRINT("-Huffman Decoder:   ");
    //GPUJPEG_TIMER_START();
    
    // Perform IDCT and dequantization
    for ( int comp = 0; comp < decoder->param_image.comp_count; comp++ ) {
        uint8_t* d_data_comp = &decoder->d_data[comp * decoder->data_width * decoder->data_height];
        int16_t* d_data_quantized_comp = &decoder->d_data_quantized[comp * decoder->data_width * decoder->data_height];
        
        // Determine table type
        enum gpujpeg_component_type type = (comp == 0) ? GPUJPEG_COMPONENT_LUMINANCE : GPUJPEG_COMPONENT_CHROMINANCE;
        
        //gpujpeg_decoder_print16(decoder, d_data_quantized_comp);
        
        cudaMemset(d_data_comp, 0, decoder->param_image.width * decoder->param_image.height * sizeof(uint8_t));
        
        //Perform inverse DCT
        NppiSize inv_roi;
        inv_roi.width = decoder->data_width * GPUJPEG_BLOCK_SIZE;
        inv_roi.height = decoder->data_height / GPUJPEG_BLOCK_SIZE;
        assert(GPUJPEG_BLOCK_SIZE == 8);
        NppStatus status = nppiDCTQuantInv8x8LS_JPEG_16s8u_C1R(
            d_data_quantized_comp, 
            decoder->data_width * GPUJPEG_BLOCK_SIZE * sizeof(int16_t), 
            d_data_comp, 
            decoder->data_width * sizeof(uint8_t), 
            decoder->table_quantization[type].d_table, 
            inv_roi
        );
        if ( status != 0 )
            printf("Error %d\n", status);
            
        //gpujpeg_decoder_print8(decoder, d_data_comp);
    }
    
    //GPUJPEG_TIMER_STOP_PRINT("-DCT & Quantization:");
    //GPUJPEG_TIMER_START();
    
    // Preprocessing
    if ( gpujpeg_preprocessor_decode(decoder) != 0 )
        return -1;
        
    //GPUJPEG_TIMER_STOP_PRINT("-Postprocessing:    ");
    
    cudaMemcpy(decoder->data_target, decoder->d_data_target, decoder->data_target_size * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    
    // Set decompressed image
    *image_decompressed = decoder->data_target;
    *image_decompressed_size = decoder->data_target_size * sizeof(uint8_t);
    
    return 0;
}

/** Documented at declaration */
int
gpujpeg_decoder_destroy(struct gpujpeg_decoder* decoder)
{    
    assert(decoder != NULL);
    
    for ( int comp_type = 0; comp_type < GPUJPEG_COMPONENT_TYPE_COUNT; comp_type++ ) {
        if ( decoder->table_quantization[comp_type].d_table != NULL )
            cudaFree(decoder->table_quantization[comp_type].d_table);
    }
    
    if ( decoder->reader != NULL )
        gpujpeg_reader_destroy(decoder->reader);
    
    if ( decoder->data_scan != NULL )
        cudaFreeHost(decoder->data_scan);
    if ( decoder->data_scan != NULL )
        cudaFree(decoder->d_data_scan);
    if ( decoder->data_scan_index != NULL )
        cudaFreeHost(decoder->data_scan_index);
    if ( decoder->data_scan_index != NULL )
        cudaFree(decoder->d_data_scan_index);
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
