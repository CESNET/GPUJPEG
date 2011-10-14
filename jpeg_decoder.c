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
    
    // Create reader
    decoder->reader = jpeg_reader_create();
    if ( decoder->reader == NULL )
        return NULL;
    
    // Allocate quantization tables in device memory
    for ( int comp_type = 0; comp_type < JPEG_COMPONENT_TYPE_COUNT; comp_type++ ) {
        if ( cudaSuccess != cudaMalloc((void**)&decoder->table_quantization[comp_type].d_table, 64 * sizeof(uint16_t)) ) 
            return NULL;
    }
    
    // Init decoder
    if ( width != 0 && height != 0 ) {
        if ( jpeg_decoder_init(decoder, width, height, comp_count) != 0 )
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
    
    int data_comp_size = decoder->width * decoder->height;
    for ( int comp = 0; comp < decoder->comp_count; comp++ ) {
        decoder->scan[comp].data = malloc(data_comp_size * sizeof(uint8_t));
        if ( decoder->scan[comp].data == NULL )
            return -1;
    }
    
    return 0;
}

/** Documented at declaration */
int
jpeg_decoder_decode(struct jpeg_decoder* decoder, uint8_t* image, int image_size, uint8_t** image_decompressed, int* image_decompressed_size)
{
    // Read JPEG image data
    if ( jpeg_reader_read_image(decoder, image, image_size) != 0 ) {
        fprintf(stderr, "Decoder failed when decoding image data!\n");
        return -1;
    }
    
    for ( int index = 0; index < decoder->scan_count; index++ ) {
        struct jpeg_decoder_scan* scan = &decoder->scan[index];
        
        // TODO: Decode
        fprintf(stderr, "Todo: Decode scan data %d bytes!\n", scan->data_size);
    }
    
    // Preprocessing
    /*if ( jpeg_preprocessor_process(encoder, image) != 0 )
        return -1;
        
    for ( int comp = 0; comp < encoder->comp_count; comp++ ) {
        uint8_t* d_data_comp = &encoder->d_data[comp * encoder->width * encoder->height];
        int16_t* d_data_quantized_comp = &encoder->d_data_quantized[comp * encoder->width * encoder->height];
        
        // Determine table type
        enum jpeg_component_type type = (comp == 0) ? JPEG_COMPONENT_LUMINANCE : JPEG_COMPONENT_CHROMINANCE;
        
        //jpeg_encoder_print8(encoder, d_data_comp);
        
        cudaMemset(d_data_quantized_comp, 0, encoder->width * encoder->height * sizeof(int16_t));
        
        //Perform forward DCT
        NppiSize fwd_roi;
        fwd_roi.width = encoder->width;
        fwd_roi.height = encoder->height;
        NppStatus status = nppiDCTQuantFwd8x8LS_JPEG_8u16s_C1R(
            d_data_comp, 
            encoder->width * sizeof(uint8_t), 
            d_data_quantized_comp, 
            encoder->width * 8 * sizeof(int16_t), 
            encoder->table[type]->d_table_forward, 
            fwd_roi
        );
        if ( status != 0 )
            printf("Error %d\n", status);
            
        
        //jpeg_encoder_print16(encoder, d_data_quantized_comp);
        
        cudaMemset(d_data_comp, 0, encoder->width * encoder->height * sizeof(int16_t));
        
        //Perform inverse DCT
        NppiSize inv_roi;
        inv_roi.width = 64;
        inv_roi.height = 3;
        status = nppiDCTQuantInv8x8LS_JPEG_16s8u_C1R(
            d_data_quantized_comp, 
            4, 
            d_data_comp, 
            32 * sizeof(uint8_t), 
            encoder->table[type]->d_table_inverse, 
            inv_roi
        );
        if ( status != 0 )
            printf("Error %d\n", status);
        
        jpeg_encoder_print8(encoder, d_data_comp);
        
        //exit(0);
    }
    
    // Initialize writer output buffer current position
    encoder->writer->buffer_current = encoder->writer->buffer;
    
    // Write header
    jpeg_writer_write_header(encoder);
    
    // Copy quantized data from device memory to cpu memory
    int data_size = encoder->width * encoder->height * encoder->comp_count;
    int16_t* data = NULL;
    cudaMallocHost((void**)&data, data_size * sizeof(int16_t)); 
    cudaMemcpy(data, encoder->d_data_quantized, data_size * sizeof(int16_t), cudaMemcpyDeviceToHost);
    
    // Perform huffman coding for all components
    for ( int comp = 0; comp < encoder->comp_count; comp++ ) {
        // Get data buffer for component
        int16_t* data_comp = &data[comp * encoder->width * encoder->height];
        // Determine table type
        enum jpeg_component_type type = (comp == 0) ? JPEG_COMPONENT_LUMINANCE : JPEG_COMPONENT_CHROMINANCE;
        // Write scan header
        jpeg_writer_write_scan_header(encoder, comp, type);
        // Perform huffman coding
        if ( jpeg_huffman_coder_encode(encoder, type, data_comp) != 0 ) {
            fprintf(stderr, "Huffman coder failed for component at index %d!\n", comp);
            return -1;
        }
    }
    
    jpeg_writer_emit_marker(encoder->writer, JPEG_MARKER_EOI);
    
    // Set compressed image
    *image_compressed = encoder->writer->buffer;
    *image_compressed_size = encoder->writer->buffer_current - encoder->writer->buffer;
    */
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
    
    assert(decoder->reader != NULL);
    jpeg_reader_destroy(decoder->reader);
    
    for ( int comp = 0; comp < decoder->comp_count; comp++ )
        free(decoder->scan[comp].data);
    
    free(decoder);
    
    return 0;
}
