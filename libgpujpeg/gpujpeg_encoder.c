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
 
#include "gpujpeg_encoder.h"
#include "gpujpeg_preprocessor.h"
#include "gpujpeg_huffman_cpu_encoder.h"
#include "gpujpeg_huffman_gpu_encoder.h"
#include "gpujpeg_format_type.h"
#include "gpujpeg_util.h"

/** Documented at declaration */
void
gpujpeg_encoder_set_default_parameters(struct gpujpeg_encoder_parameters* param)
{
    param->quality = 75;
    param->restart_interval = 8;
    param->interleaved = 0;
    for ( int comp = 0; comp < GPUJPEG_MAX_COMPONENT_COUNT; comp++ ) {
        param->sampling_factor[comp].horizontal = 1;
        param->sampling_factor[comp].vertical = 1;
    }
}

/** Documented at declaration */
void
gpujpeg_encoder_parameters_chroma_subsampling(struct gpujpeg_encoder_parameters* param)
{
    for ( int comp = 0; comp < GPUJPEG_MAX_COMPONENT_COUNT; comp++ ) {
        if ( comp == 0 ) {
            param->sampling_factor[comp].horizontal = 2;
            param->sampling_factor[comp].vertical = 2;
        } else {
            param->sampling_factor[comp].horizontal = 1;
            param->sampling_factor[comp].vertical = 1;
        }
    }
}

/** Documented at declaration */
struct gpujpeg_encoder*
gpujpeg_encoder_create(struct gpujpeg_image_parameters* param_image, struct gpujpeg_encoder_parameters* param)
{
    assert(param_image->comp_count == 3);
    assert(param_image->comp_count <= GPUJPEG_MAX_COMPONENT_COUNT);
    assert(param->quality >= 0 && param->quality <= 100);
    assert(param->restart_interval >= 0);
    assert(param->interleaved == 0 || param->interleaved == 1);
    
    struct gpujpeg_encoder* encoder = malloc(sizeof(struct gpujpeg_encoder));
    if ( encoder == NULL )
        return NULL;
        
    // Set parameters
    memset(encoder, 0, sizeof(struct gpujpeg_encoder));
    encoder->param_image = *param_image;
    encoder->param = *param;
    
    int result = 1;
    
    // Create writer
    encoder->writer = gpujpeg_writer_create(encoder);
    if ( encoder->writer == NULL )
        result = 0;
    
    // Allocate color components
    cudaMallocHost((void**)&encoder->component, encoder->param_image.comp_count * sizeof(struct gpujpeg_encoder_component));
    if ( encoder->component == NULL )
        result = 0;
    // Allocate color components in device memory
    if ( cudaSuccess != cudaMalloc((void**)&encoder->d_component, encoder->param_image.comp_count * sizeof(struct gpujpeg_encoder_component)) )
        result = 0;
    gpujpeg_cuda_check_error("Encoder color component allocation");
        
    // Initialize sampling factors and compute maximum sampling factor to encoder->sampling_factor
    encoder->sampling_factor.horizontal = 0;
    encoder->sampling_factor.vertical = 0;
    for ( int comp = 0; comp < encoder->param_image.comp_count; comp++ ) {
        assert(encoder->param.sampling_factor[comp].horizontal >= 1 && encoder->param.sampling_factor[comp].horizontal <= 15);
        assert(encoder->param.sampling_factor[comp].vertical >= 1 && encoder->param.sampling_factor[comp].vertical <= 15);
        encoder->component[comp].sampling_factor = encoder->param.sampling_factor[comp];
        if ( encoder->component[comp].sampling_factor.horizontal > encoder->sampling_factor.horizontal )
            encoder->sampling_factor.horizontal = encoder->component[comp].sampling_factor.horizontal;
        if ( encoder->component[comp].sampling_factor.vertical > encoder->sampling_factor.vertical )
            encoder->sampling_factor.vertical = encoder->component[comp].sampling_factor.vertical;
    }
    
    // Calculate data size
    encoder->data_source_size = encoder->param_image.width * encoder->param_image.height * encoder->param_image.comp_count;
    encoder->data_size = 0;
    
    // Initialize color components
    for ( int comp = 0; comp < encoder->param_image.comp_count; comp++ ) {
        // Set type
        encoder->component[comp].type = (comp == 0) ? GPUJPEG_COMPONENT_LUMINANCE : GPUJPEG_COMPONENT_CHROMINANCE;
        
        // Set proper color component sizes in pixels based on sampling factors
        int samp_factor_h = encoder->component[comp].sampling_factor.horizontal;
        int samp_factor_v = encoder->component[comp].sampling_factor.vertical;
        encoder->component[comp].width = (encoder->param_image.width * samp_factor_h) / encoder->sampling_factor.horizontal;
        encoder->component[comp].height = (encoder->param_image.height * samp_factor_v) / encoder->sampling_factor.vertical;
        
        // Compute component MCU size
        encoder->component[comp].mcu_size_x = GPUJPEG_BLOCK_SIZE;
        encoder->component[comp].mcu_size_y = GPUJPEG_BLOCK_SIZE;
        if ( encoder->param.interleaved == 1 ) {
            encoder->component[comp].mcu_compressed_size = GPUJPEG_MAX_BLOCK_COMPRESSED_SIZE * samp_factor_h * samp_factor_v;
            encoder->component[comp].mcu_size_x *= samp_factor_h;
            encoder->component[comp].mcu_size_y *= samp_factor_v;
        } else {
            encoder->component[comp].mcu_compressed_size = GPUJPEG_MAX_BLOCK_COMPRESSED_SIZE;
        }
        encoder->component[comp].mcu_size = encoder->component[comp].mcu_size_x * encoder->component[comp].mcu_size_y;
        
        // Compute allocated data size
        encoder->component[comp].data_width = gpujpeg_div_and_round_up(encoder->component[comp].width, encoder->component[comp].mcu_size_x) * encoder->component[comp].mcu_size_x;
        encoder->component[comp].data_height = gpujpeg_div_and_round_up(encoder->component[comp].height, encoder->component[comp].mcu_size_y) * encoder->component[comp].mcu_size_y;
        encoder->component[comp].data_size = encoder->component[comp].data_width * encoder->component[comp].data_height;
        // Increase total data size
        encoder->data_size += encoder->component[comp].data_size;
        
        // Compute component MCU count
        encoder->component[comp].mcu_count_x = gpujpeg_div_and_round_up(encoder->component[comp].data_width, encoder->component[comp].mcu_size_x);
        encoder->component[comp].mcu_count_y = gpujpeg_div_and_round_up(encoder->component[comp].data_height, encoder->component[comp].mcu_size_y);
        encoder->component[comp].mcu_count = encoder->component[comp].mcu_count_x * encoder->component[comp].mcu_count_y;
        
        // Compute MCU count per segment
        encoder->component[comp].segment_mcu_count = encoder->param.restart_interval;
        if ( encoder->component[comp].segment_mcu_count == 0 ) {
            // If restart interval is disabled, restart interval is equal MCU count
            encoder->component[comp].segment_mcu_count = encoder->component[comp].mcu_count;
        }
        
        // Calculate segment count
        encoder->component[comp].segment_count = gpujpeg_div_and_round_up(encoder->component[comp].mcu_count, encoder->component[comp].segment_mcu_count);
        
        //printf("Subsampling %dx%d, Resolution %d, %d, mcu size %d, mcu count %d\n",
        //    encoder->param.sampling_factor[comp].horizontal, encoder->param.sampling_factor[comp].vertical,
        //    encoder->component[comp].data_width, encoder->component[comp].data_height,
        //    encoder->component[comp].mcu_compressed_size, encoder->component[comp].mcu_count
        //);
    }
    
    // Maximum component data size for allocated buffers
    encoder->data_width = gpujpeg_div_and_round_up(param_image->width, GPUJPEG_BLOCK_SIZE) * GPUJPEG_BLOCK_SIZE;
    encoder->data_height = gpujpeg_div_and_round_up(param_image->height, GPUJPEG_BLOCK_SIZE) * GPUJPEG_BLOCK_SIZE;
    
    // Allocate data buffers for all color components
    if ( cudaSuccess != cudaMalloc((void**)&encoder->d_data_source, encoder->data_source_size * sizeof(uint8_t)) ) 
        result = 0;
    if ( cudaSuccess != cudaMalloc((void**)&encoder->d_data, encoder->data_size * sizeof(uint8_t)) ) 
        result = 0;
    if ( cudaSuccess != cudaMallocHost((void**)&encoder->data_quantized, encoder->data_size * sizeof(int16_t)) ) 
        result = 0;
    if ( cudaSuccess != cudaMalloc((void**)&encoder->d_data_quantized, encoder->data_size * sizeof(int16_t)) ) 
        result = 0;
	gpujpeg_cuda_check_error("Encoder data allocation");
    
    // Set data buffer to color components
    uint8_t* d_comp_data = encoder->d_data;
    int16_t* d_comp_data_quantized = encoder->d_data_quantized;
    int16_t* comp_data_quantized = encoder->data_quantized;
    for ( int comp = 0; comp < encoder->param_image.comp_count; comp++ ) {
        encoder->component[comp].d_data = d_comp_data;
        encoder->component[comp].d_data_quantized = d_comp_data_quantized;
        encoder->component[comp].data_quantized = comp_data_quantized;
        d_comp_data += encoder->component[comp].data_width * encoder->component[comp].data_height;
        d_comp_data_quantized += encoder->component[comp].data_width * encoder->component[comp].data_height;
        comp_data_quantized += encoder->component[comp].data_width * encoder->component[comp].data_height;
    }
    
    // Copy components to device memory
    if ( cudaSuccess != cudaMemcpy(encoder->d_component, encoder->component, encoder->param_image.comp_count * sizeof(struct gpujpeg_encoder_component), cudaMemcpyHostToDevice) )
        result = 0;
    gpujpeg_cuda_check_error("Encoder component copy");
    
    // Compute MCU size, MCU count, segment count and compressed data allocation size
    encoder->mcu_count = 0;
    encoder->mcu_size = 0;
    encoder->mcu_compressed_size = 0;
    encoder->segment_count = 0;
    encoder->data_compressed_size = 0;
    if ( encoder->param.interleaved == 1 ) {
        assert(encoder->param_image.comp_count > 0);
        encoder->mcu_count = encoder->component[0].mcu_count;
        encoder->segment_count = encoder->component[0].segment_count;
        encoder->segment_mcu_count = encoder->component[0].segment_mcu_count;
        for ( int comp = 0; comp < encoder->param_image.comp_count; comp++ ) {
            assert(encoder->mcu_count == encoder->component[comp].mcu_count);
            assert(encoder->segment_mcu_count == encoder->component[comp].segment_mcu_count);
            encoder->mcu_size += encoder->component[comp].mcu_size;
            encoder->mcu_compressed_size += encoder->component[comp].mcu_compressed_size;
        }
    } else {
        assert(encoder->param_image.comp_count > 0);
        encoder->mcu_size = encoder->component[0].mcu_size;
        encoder->mcu_compressed_size = encoder->component[0].mcu_compressed_size;
        encoder->segment_mcu_count = 0;
        for ( int comp = 0; comp < encoder->param_image.comp_count; comp++ ) {
            assert(encoder->mcu_size == encoder->component[comp].mcu_size);
            assert(encoder->mcu_compressed_size == encoder->component[comp].mcu_compressed_size);
            encoder->mcu_count += encoder->component[comp].mcu_count;
            encoder->segment_count += encoder->component[comp].segment_count;
        }
    }
    //printf("mcu size %d -> %d, mcu count %d, segment mcu count %d\n", encoder->mcu_size, encoder->mcu_compressed_size, encoder->mcu_count, encoder->segment_mcu_count);

    // Allocate segments
    cudaMallocHost((void**)&encoder->segment, encoder->segment_count * sizeof(struct gpujpeg_encoder_segment));
    if ( encoder->segment == NULL )
        result = 0;
    // Allocate segments in device memory
    if ( cudaSuccess != cudaMalloc((void**)&encoder->d_segment, encoder->segment_count * sizeof(struct gpujpeg_encoder_segment)) )
        result = 0;
    gpujpeg_cuda_check_error("Encoder segment allocation");
    
    // Prepare segments
    if ( result == 1 ) {            
        // While preparing segments compute input size and compressed size
        int data_index = 0;
        int data_compressed_index = 0;
        
        // Prepare segments based on (non-)interleaved mode
        if ( encoder->param.interleaved == 1 ) {
            // Prepare segments for encoding (only one scan for all color components)
            int mcu_index = 0;
            for ( int index = 0; index < encoder->segment_count; index++ ) {
                // Prepare segment MCU count
                int mcu_count = encoder->segment_mcu_count;
                if ( (mcu_index + mcu_count) >= encoder->mcu_count )
                    mcu_count = encoder->mcu_count - mcu_index;
                // Set parameters for segment
                encoder->segment[index].scan_index = 0;
                encoder->segment[index].scan_segment_index = index;
                encoder->segment[index].mcu_count = mcu_count;
                encoder->segment[index].data_compressed_index = data_compressed_index;
                encoder->segment[index].data_compressed_size = 0;
                // Increase parameters for next segment
                data_index += mcu_count * encoder->mcu_size;
                data_compressed_index += mcu_count * encoder->mcu_compressed_size;
                mcu_index += mcu_count;
            }
        } else {
            // Prepare segments for encoding (one scan for each color component)
            int index = 0;
            for ( int comp = 0; comp < encoder->param_image.comp_count; comp++ ) {
                // Prepare component segments
                int mcu_index = 0;
                for ( int segment = 0; segment < encoder->component[comp].segment_count; segment++ ) {
                    // Prepare segment MCU count
                    int mcu_count = encoder->component[comp].segment_mcu_count;
                    if ( (mcu_index + mcu_count) >= encoder->component[comp].mcu_count )
                        mcu_count = encoder->component[comp].mcu_count - mcu_index;
                    // Set parameters for segment
                    encoder->segment[index].scan_index = comp;
                    encoder->segment[index].scan_segment_index = segment;
                    encoder->segment[index].mcu_count = mcu_count;
                    encoder->segment[index].data_compressed_index = data_compressed_index;
                    encoder->segment[index].data_compressed_size = 0;
                    // Increase parameters for next segment
                    data_index += mcu_count * encoder->component[comp].mcu_size;
                    data_compressed_index += mcu_count * encoder->component[comp].mcu_compressed_size;
                    mcu_index += mcu_count;
                    index++;
                }
            }
        }
        
        // Check data size
        //printf("%d == %d\n", encoder->data_size, data_index);
        assert(encoder->data_size == data_index);
            
        // Set compressed size
        encoder->data_compressed_size = data_compressed_index;
    }
    //printf("Compressed size %d (segments %d)\n", encoder->data_compressed_size, encoder->segment_count);
        
    // Copy segments to device memory
    if ( cudaSuccess != cudaMemcpy(encoder->d_segment, encoder->segment, encoder->segment_count * sizeof(struct gpujpeg_encoder_segment), cudaMemcpyHostToDevice) )
        result = 0;
        
        // Allocate compressed data
    if ( cudaSuccess != cudaMallocHost((void**)&encoder->data_compressed, encoder->data_compressed_size * sizeof(uint8_t)) ) 
        result = 0;   
    if ( cudaSuccess != cudaMalloc((void**)&encoder->d_data_compressed, encoder->data_compressed_size * sizeof(uint8_t)) ) 
        result = 0;   
	gpujpeg_cuda_check_error("Encoder data compressed allocation");
     
    // Allocate quantization tables in device memory
    for ( int comp_type = 0; comp_type < GPUJPEG_COMPONENT_TYPE_COUNT; comp_type++ ) {
        if ( cudaSuccess != cudaMalloc((void**)&encoder->table_quantization[comp_type].d_table, 64 * sizeof(uint16_t)) ) 
            result = 0;
    }
    // Allocate huffman tables in device memory
    for ( int comp_type = 0; comp_type < GPUJPEG_COMPONENT_TYPE_COUNT; comp_type++ ) {
        for ( int huff_type = 0; huff_type < GPUJPEG_HUFFMAN_TYPE_COUNT; huff_type++ ) {
            if ( cudaSuccess != cudaMalloc((void**)&encoder->d_table_huffman[comp_type][huff_type], sizeof(struct gpujpeg_table_huffman_encoder)) )
                result = 0;
        }
    }
	gpujpeg_cuda_check_error("Encoder table allocation");
    
    // Init quantization tables for encoder
    for ( int comp_type = 0; comp_type < GPUJPEG_COMPONENT_TYPE_COUNT; comp_type++ ) {
        if ( gpujpeg_table_quantization_encoder_init(&encoder->table_quantization[comp_type], comp_type, encoder->param.quality) != 0 )
            result = 0;
    }
    // Init huffman tables for encoder
    for ( int comp_type = 0; comp_type < GPUJPEG_COMPONENT_TYPE_COUNT; comp_type++ ) {
        for ( int huff_type = 0; huff_type < GPUJPEG_HUFFMAN_TYPE_COUNT; huff_type++ ) {
            if ( gpujpeg_table_huffman_encoder_init(&encoder->table_huffman[comp_type][huff_type], encoder->d_table_huffman[comp_type][huff_type], comp_type, huff_type) != 0 )
                result = 0;
        }
    }
	gpujpeg_cuda_check_error("Encoder table init");
    
    // Init huffman encoder
    if ( gpujpeg_huffman_gpu_encoder_init() != 0 )
        result = 0;
    
    if ( result == 0 ) {
        gpujpeg_encoder_destroy(encoder);
        return NULL;
    }
    
    return encoder;
}

void
gpujpeg_encoder_print8(struct gpujpeg_encoder_component* component, uint8_t* d_data)
{
    int data_size = component->data_width * component->data_height;
    uint8_t* data = NULL;
    cudaMallocHost((void**)&data, data_size * sizeof(uint8_t)); 
    cudaMemcpy(data, d_data, data_size * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    
    printf("Print Data\n");
    for ( int y = 0; y < component->data_height; y++ ) {
        for ( int x = 0; x < component->data_width; x++ ) {
            printf("%3u ", data[y * component->data_width + x]);
        }
        printf("\n");
    }
    cudaFreeHost(data);
}

void
gpujpeg_encoder_print16(struct gpujpeg_encoder_component* component, int16_t* d_data)
{
    int data_size = component->data_width * component->data_height;
    int16_t* data = NULL;
    cudaMallocHost((void**)&data, data_size * sizeof(int16_t)); 
    cudaMemcpy(data, d_data, data_size * sizeof(int16_t), cudaMemcpyDeviceToHost);
    
    printf("Print Data\n");
    for ( int y = 0; y < component->data_height; y++ ) {
        for ( int x = 0; x < component->data_width; x++ ) {
            printf("%3d ", data[y * component->data_width + x]);
        }
        printf("\n");
    }
    cudaFreeHost(data);
}

/** Documented at declaration */
int
gpujpeg_encoder_encode(struct gpujpeg_encoder* encoder, uint8_t* image, uint8_t** image_compressed, int* image_compressed_size)
{    
    //GPUJPEG_TIMER_INIT();
    //GPUJPEG_TIMER_START();
    
    // Copy image to device memory
    if ( cudaSuccess != cudaMemcpy(encoder->d_data_source, image, encoder->data_source_size * sizeof(uint8_t), cudaMemcpyHostToDevice) )
        return -1;
    
    //gpujpeg_table_print(encoder->table[JPEG_COMPONENT_LUMINANCE]);
    //gpujpeg_table_print(encoder->table[JPEG_COMPONENT_CHROMINANCE]);
    
    // Preprocessing
    if ( gpujpeg_preprocessor_encode(encoder) != 0 )
        return -1;
        
    //GPUJPEG_TIMER_STOP_PRINT("-Preprocessing:     ");
    //GPUJPEG_TIMER_START();
        
    // Perform DCT and quantization
    for ( int comp = 0; comp < encoder->param_image.comp_count; comp++ ) {        
        // Determine table type
        enum gpujpeg_component_type type = (comp == 0) ? GPUJPEG_COMPONENT_LUMINANCE : GPUJPEG_COMPONENT_CHROMINANCE;
        
        //gpujpeg_encoder_print8(&encoder->component[comp], encoder->component[comp].d_data);
        
        //Perform forward DCT
        NppiSize fwd_roi;
        fwd_roi.width = encoder->component[comp].data_width;
        fwd_roi.height = encoder->component[comp].data_height;
        NppStatus status = nppiDCTQuantFwd8x8LS_JPEG_8u16s_C1R(
            encoder->component[comp].d_data, 
            encoder->component[comp].data_width * sizeof(uint8_t), 
            encoder->component[comp].d_data_quantized, 
            encoder->component[comp].data_width * GPUJPEG_BLOCK_SIZE * sizeof(int16_t), 
            encoder->table_quantization[type].d_table, 
            fwd_roi
        );
        if ( status != 0 ) {
            fprintf(stderr, "Forward DCT failed for component at index %d [error %d]!\n", comp, status);		
            return -1;
        }
        
        //gpujpeg_encoder_print16(&encoder->component[comp], encoder->component[comp].d_data_quantized);
    }
    
    // Initialize writer output buffer current position
    encoder->writer->buffer_current = encoder->writer->buffer;
    
    // Write header
    gpujpeg_writer_write_header(encoder);
    
    //GPUJPEG_TIMER_STOP_PRINT("-DCT & Quantization:");
    //GPUJPEG_TIMER_START();
    
    // Perform huffman coding on CPU (when restart interval is not set)
    if ( encoder->param.restart_interval == 0 ) {
        // Copy quantized data from device memory to cpu memory
        cudaMemcpy(encoder->data_quantized, encoder->d_data_quantized, encoder->data_size * sizeof(int16_t), cudaMemcpyDeviceToHost);
        
        // Perform huffman coding
        if ( gpujpeg_huffman_cpu_encoder_encode(encoder) != 0 ) {
            fprintf(stderr, "Huffman encoder on CPU failed!\n");
            return -1;
        }
    }
    // Perform huffman coding on GPU (when restart interval is set)
    else {
        // Perform huffman coding
        if ( gpujpeg_huffman_gpu_encoder_encode(encoder) != 0 ) {
            fprintf(stderr, "Huffman encoder on GPU failed!\n");
            return -1;
        }
        
        // Copy compressed data from device memory to cpu memory
        if ( cudaSuccess != cudaMemcpy(encoder->data_compressed, encoder->d_data_compressed, encoder->data_compressed_size * sizeof(uint8_t), cudaMemcpyDeviceToHost) != 0 )
            return -1;
        // Copy segments to device memory
        if ( cudaSuccess != cudaMemcpy(encoder->segment, encoder->d_segment, encoder->segment_count * sizeof(struct gpujpeg_encoder_segment), cudaMemcpyDeviceToHost) )
            return -1;
            
        if ( encoder->param.interleaved == 1 ) {
            // Write scan header (only one scan is written, that contains all color components data)
            gpujpeg_writer_write_scan_header(encoder, 0);
            // Write scan data
            for ( int segment_index = 0; segment_index < encoder->segment_count; segment_index++ ) {
                struct gpujpeg_encoder_segment* segment = &encoder->segment[segment_index];
                    
                // Copy compressed data to writer
                memcpy(
                    encoder->writer->buffer_current, 
                    &encoder->data_compressed[segment->data_compressed_index],
                    segment->data_compressed_size
                );
                encoder->writer->buffer_current += segment->data_compressed_size;
                //printf("Compressed data %d bytes\n", segment->data_compressed_size);
            }
        } else {
            // Write huffman coder results as one scan for each color component
            int segment_index = 0;
            for ( int comp = 0; comp < encoder->param_image.comp_count; comp++ ) {
                // Write scan header
                gpujpeg_writer_write_scan_header(encoder, comp);
                // Write scan data
                for ( int index = 0; index < encoder->component[comp].segment_count; index++ ) {
                    struct gpujpeg_encoder_segment* segment = &encoder->segment[segment_index];
                    
                    // Copy compressed data to writer
                    memcpy(
                        encoder->writer->buffer_current, 
                        &encoder->data_compressed[segment->data_compressed_index],
                        segment->data_compressed_size
                    );
                    encoder->writer->buffer_current += segment->data_compressed_size;
                    //printf("Compressed data %d bytes\n", segment->data_compressed_size);
                    
                    segment_index++;
                }
            }
        }
    }
    gpujpeg_writer_emit_marker(encoder->writer, GPUJPEG_MARKER_EOI);
    
    //GPUJPEG_TIMER_STOP_PRINT("-Huffman Encoder:   ");
    
    // Set compressed image
    *image_compressed = encoder->writer->buffer;
    *image_compressed_size = encoder->writer->buffer_current - encoder->writer->buffer;
    
    return 0;
}

/** Documented at declaration */
int
gpujpeg_encoder_destroy(struct gpujpeg_encoder* encoder)
{
    assert(encoder != NULL);
    
    for ( int comp_type = 0; comp_type < GPUJPEG_COMPONENT_TYPE_COUNT; comp_type++ ) {
        if ( encoder->table_quantization[comp_type].d_table != NULL )
            cudaFree(encoder->table_quantization[comp_type].d_table);
    }
    for ( int comp_type = 0; comp_type < GPUJPEG_COMPONENT_TYPE_COUNT; comp_type++ ) {
        for ( int huff_type = 0; huff_type < GPUJPEG_HUFFMAN_TYPE_COUNT; huff_type++ ) {
            if ( encoder->d_table_huffman[comp_type][huff_type] != NULL )
                cudaFree(encoder->d_table_huffman[comp_type][huff_type]);
        }
    }
    
    if ( encoder->writer != NULL )
        gpujpeg_writer_destroy(encoder->writer);
    
    if ( encoder->d_data_source != NULL )
        cudaFree(encoder->d_data_source);
    if ( encoder->d_data != NULL )
        cudaFree(encoder->d_data);
    if ( encoder->data_quantized != NULL )
        cudaFreeHost(encoder->data_quantized);    
    if ( encoder->d_data_quantized != NULL )
        cudaFree(encoder->d_data_quantized);    
    if ( encoder->data_compressed != NULL )
        cudaFreeHost(encoder->data_compressed);    
    if ( encoder->d_data_compressed != NULL )
        cudaFree(encoder->d_data_compressed);    
    if ( encoder->segment != NULL )
        cudaFreeHost(encoder->segment); 
    if ( encoder->d_segment != NULL )
        cudaFree(encoder->d_segment);
    
    free(encoder);
    
    return 0;
}
