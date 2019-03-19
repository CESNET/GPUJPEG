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

#include <string.h>
#include <libgpujpeg/gpujpeg_encoder.h>
#include "gpujpeg_preprocessor.h"
#include "gpujpeg_dct_cpu.h"
#include "gpujpeg_dct_gpu.h"
#include "gpujpeg_huffman_cpu_encoder.h"
#include "gpujpeg_huffman_gpu_encoder.h"
#include <math.h>
#include <libgpujpeg/gpujpeg_util.h>

/** Documented at declaration */
void
gpujpeg_encoder_input_set_image(struct gpujpeg_encoder_input* input, uint8_t* image)
{
    input->type = GPUJPEG_ENCODER_INPUT_IMAGE;
    input->image = image;
    input->texture = NULL;
}

/** Documented at declaration */
void
gpujpeg_encoder_input_set_gpu_image(struct gpujpeg_encoder_input* input, uint8_t* image)
{
    input->type = GPUJPEG_ENCODER_INPUT_GPU_IMAGE;
    input->image = image;
    input->texture = NULL;
}

/** Documented at declaration */
void
gpujpeg_encoder_input_set_texture(struct gpujpeg_encoder_input* input, struct gpujpeg_opengl_texture* texture)
{
    input->type = GPUJPEG_ENCODER_INPUT_OPENGL_TEXTURE;
    input->image = NULL;
    input->texture = texture;
}

/** Documented at declaration */
struct gpujpeg_encoder*
gpujpeg_encoder_create(cudaStream_t * stream)
{
    struct gpujpeg_encoder* encoder = (struct gpujpeg_encoder*) malloc(sizeof(struct gpujpeg_encoder));
    if ( encoder == NULL ) {
        return NULL;
    }
    memset(encoder, 0, sizeof(struct gpujpeg_encoder));

    // Get coder
    struct gpujpeg_coder* coder = &encoder->coder;

    int result = 1;

    // Create writer
    encoder->writer = gpujpeg_writer_create(encoder);
    if ( encoder->writer == NULL )
        result = 0;

    // Initialize coder
    if ( gpujpeg_coder_init(coder) != 0 )
        result = 0;

    // Allocate quantization tables in device memory
    for ( int comp_type = 0; comp_type < GPUJPEG_COMPONENT_TYPE_COUNT; comp_type++ ) {
        if ( cudaSuccess != cudaMalloc((void**)&encoder->table_quantization[comp_type].d_table, 64 * sizeof(uint16_t)) ) {
            result = 0;
        }
        if ( cudaSuccess != cudaMalloc((void**)&encoder->table_quantization[comp_type].d_table_forward, 64 * sizeof(float)) ) {
            result = 0;
        }
    }
    gpujpeg_cuda_check_error("Encoder table allocation", return NULL);

    // Init huffman tables for encoder
    for ( int comp_type = 0; comp_type < GPUJPEG_COMPONENT_TYPE_COUNT; comp_type++ ) {
        for ( int huff_type = 0; huff_type < GPUJPEG_HUFFMAN_TYPE_COUNT; huff_type++ ) {
            if ( gpujpeg_table_huffman_encoder_init(&encoder->table_huffman[comp_type][huff_type], (enum gpujpeg_component_type)comp_type, (enum gpujpeg_huffman_type)huff_type) != 0 )
                result = 0;
        }
    }
    gpujpeg_cuda_check_error("Encoder table init", return NULL);

    // Init huffman encoder
    encoder->huffman_gpu_encoder = gpujpeg_huffman_gpu_encoder_create(encoder);
    if (encoder->huffman_gpu_encoder == NULL) {
        result = 0;
    }

    // Stream
    encoder->stream = stream;
    if (encoder->stream == NULL) {
        encoder->allocatedStream = (cudaStream_t *) malloc(sizeof(cudaStream_t));
        if (cudaSuccess != cudaStreamCreate(encoder->allocatedStream)) {
            result = 0;
        }
        encoder->stream = encoder->allocatedStream;
    }

    if ( result == 0 ) {
        gpujpeg_encoder_destroy(encoder);
        return NULL;
    }

    return encoder;
}

/** Documented at declaration */
size_t gpujpeg_encoder_max_pixels(struct gpujpeg_parameters * param, struct gpujpeg_image_parameters * param_image, enum gpujpeg_encoder_input_type image_input_type, size_t memory_size, int * max_pixels)
{
    struct gpujpeg_coder coder;
    if (0 != gpujpeg_coder_init(&coder)) {
        return 0;
    }

    size_t encoder_memory_size = 0;
    encoder_memory_size += 2 * 64 * sizeof(uint16_t); // Quantization tables
    encoder_memory_size += 2 * 64 * sizeof(float);    // Quantization tables

    int current_max_pixels = 0;
    size_t current_max_pixels_memory_size = 0;
    int result = 0;
    int pixels = 10000;
    int iteration = 0;
    while (true) {
        param_image->width = (int) sqrt((float) pixels);
        param_image->height = (pixels + param_image->width - 1) / param_image->width;
        //printf("\nIteration #%d (pixels: %d, size: %dx%d)\n", iteration++, pixels, param_image->width, param_image->height);
        size_t image_memory_size = gpujpeg_coder_init_image(&coder, param, param_image, NULL);
        if (image_memory_size == 0) {
            break;
        }
        size_t allocated_memory_size = 0;
        allocated_memory_size += encoder_memory_size;
        allocated_memory_size += image_memory_size;
        if (image_input_type == GPUJPEG_ENCODER_INPUT_IMAGE || image_input_type == GPUJPEG_ENCODER_INPUT_OPENGL_TEXTURE) {
            allocated_memory_size += coder.data_raw_size;
        }
        if (allocated_memory_size > 0 && allocated_memory_size <= memory_size) {
            current_max_pixels = pixels;
            current_max_pixels_memory_size = allocated_memory_size;

            // TODO: increase number of pixels
            double used_memory_size = (double) current_max_pixels_memory_size / (double) memory_size;
            //printf("  Max Pixels: %d (used %d/%d bytes, %0.2f%%)\n", current_max_pixels, current_max_pixels_memory_size, memory_size, used_memory_size * 100.0);

            // Check next
            int next_pixels = pixels * (0.99 / used_memory_size);
            if (next_pixels <= pixels) {
                break;
            }
            pixels = next_pixels;
        }
        else  {
            if (current_max_pixels == 0){
                result = -1;
            }
            break;
        }
    }

    if (0 != gpujpeg_coder_deinit(&coder)) {
        return 0;
    }
    if (max_pixels != NULL) {
        *max_pixels = current_max_pixels;
    }
    return current_max_pixels_memory_size;
}

/** Documented at declaration */
size_t gpujpeg_encoder_max_memory(struct gpujpeg_parameters * param, struct gpujpeg_image_parameters * param_image, enum gpujpeg_encoder_input_type image_input_type, int max_pixels)
{
    struct gpujpeg_coder coder;
    if (0 != gpujpeg_coder_init(&coder)) {
        return 0;
    }

    size_t encoder_memory_size = 0;
    encoder_memory_size += 2 * 64 * sizeof(uint16_t); // Quantization tables
    encoder_memory_size += 2 * 64 * sizeof(float);    // Quantization tables

    param_image->width = (int) sqrt((float) max_pixels);
    param_image->height = (max_pixels + param_image->width - 1) / param_image->width;

    size_t image_memory_size = gpujpeg_coder_init_image(&coder, param, param_image, NULL);
    if (image_memory_size == 0) {
        return 0;
    }

    size_t allocated_memory_size = 0;
    allocated_memory_size += encoder_memory_size;
    allocated_memory_size += image_memory_size;
    if (image_input_type == GPUJPEG_ENCODER_INPUT_IMAGE || image_input_type == GPUJPEG_ENCODER_INPUT_OPENGL_TEXTURE) {
        allocated_memory_size += coder.data_raw_size;
    }

    if (0 != gpujpeg_coder_deinit(&coder)) {
        return 0;
    }

    return allocated_memory_size;
}

/** Documented at declaration */
int gpujpeg_encoder_allocate(struct gpujpeg_encoder * encoder, struct gpujpeg_parameters * param, struct gpujpeg_image_parameters * param_image, enum gpujpeg_encoder_input_type image_input_type, int pixels)
{
    // Get coder
    struct gpujpeg_coder* coder = &encoder->coder;

    // Quality should be initialized when encoding
    struct gpujpeg_parameters tmp_param;
    tmp_param = *param;
    tmp_param.quality = -1;

    // Set image size from pixels
    struct gpujpeg_image_parameters tmp_param_image;
    tmp_param_image = *param_image;
    tmp_param_image.width = (int) sqrt((float) pixels);
    tmp_param_image.height = (pixels + tmp_param_image.width - 1) / tmp_param_image.width;

    // Allocate internal buffers
    if (0 == gpujpeg_coder_init_image(coder, &tmp_param, &tmp_param_image, NULL)) {
        fprintf(stderr, "[GPUJPEG] [Error] Failed to pre-allocate encoding!\n");
        return -1;
    }

    // Allocate input raw buffer
    if (image_input_type == GPUJPEG_ENCODER_INPUT_IMAGE || image_input_type == GPUJPEG_ENCODER_INPUT_OPENGL_TEXTURE) {
        // Allocate raw data internal buffer
        if (coder->data_raw_size > coder->data_raw_allocated_size) {
            coder->data_raw_allocated_size = 0;

            // (Re)allocate raw data in device memory
            if (coder->d_data_raw_allocated != NULL) {
                cudaFree(coder->d_data_raw_allocated);
                coder->d_data_raw_allocated = NULL;
            }
            cudaMalloc((void**)&coder->d_data_raw_allocated, coder->data_raw_size);
            gpujpeg_cuda_check_error("Encoder raw data allocation", return -1);

            coder->data_raw_allocated_size = coder->data_raw_size;
        }
    }

    return 0;
}

/** Documented at declaration */
int
gpujpeg_encoder_encode(struct gpujpeg_encoder* encoder, struct gpujpeg_parameters* param, struct gpujpeg_image_parameters* param_image, struct gpujpeg_encoder_input* input, uint8_t** image_compressed, int* image_compressed_size)
{
    assert(param_image->comp_count == 1 || param_image->comp_count == 3);
    assert(param_image->comp_count <= GPUJPEG_MAX_COMPONENT_COUNT);
    assert(param->quality >= 0 && param->quality <= 100);
    assert(param->restart_interval >= 0);
    assert(param->interleaved == 0 || param->interleaved == 1);

    // Get coder
    struct gpujpeg_coder* coder = &encoder->coder;

    // (Re)initialize encoder
    if (coder->param.quality != param->quality) {
        // Init quantization tables for encoder
        for (int comp_type = 0; comp_type < GPUJPEG_COMPONENT_TYPE_COUNT; comp_type++) {
            if (gpujpeg_table_quantization_encoder_init(&encoder->table_quantization[comp_type], (enum gpujpeg_component_type)comp_type, param->quality) != 0) {
                return -1;
            }
        }
        gpujpeg_cuda_check_error("Quantization init", return -1);
    }
    if (0 == gpujpeg_coder_init_image(coder, param, param_image, encoder->stream)) {
        fprintf(stderr, "[GPUJPEG] [Error] Failed to init image encoding!\n");
        return -1;
    }

    // (Re)initialize writer
    if (gpujpeg_writer_init(encoder->writer, &encoder->coder.param_image) != 0) {
        fprintf(stderr, "[GPUJPEG] [Error] Failed to init writer!\n");
        return -1;
    }

    // (Re)initialize preprocessor
    if (gpujpeg_preprocessor_encoder_init(&encoder->coder) != 0) {
        fprintf(stderr, "[GPUJPEG] [Error] Failed to init preprocessor!\n");
        return -1;
    }

    // Reset durations
    coder->duration_memory_map = 0.0;
    coder->duration_memory_unmap = 0.0;
    coder->duration_stream = 0.0;
    coder->duration_in_gpu = 0.0;
    coder->duration_waiting = 0.0;

    // Load input image
    if ( input->type == GPUJPEG_ENCODER_INPUT_IMAGE ) {
        // Allocate raw data internal buffer
        if (coder->data_raw_size > coder->data_raw_allocated_size) {
            coder->data_raw_allocated_size = 0;

            // (Re)allocate raw data in device memory
            if (coder->d_data_raw_allocated != NULL) {
                cudaFree(coder->d_data_raw_allocated);
                coder->d_data_raw_allocated = NULL;
            }
            cudaMalloc((void**)&coder->d_data_raw_allocated, coder->data_raw_size);
            gpujpeg_cuda_check_error("Encoder raw data allocation", return -1);

            coder->data_raw_allocated_size = coder->data_raw_size;
        }
        // User internal buffer for raw data
        coder->d_data_raw = coder->d_data_raw_allocated;

        // Copy image to device memory
        cudaMemcpyAsync(coder->d_data_raw, input->image, coder->data_raw_size * sizeof(uint8_t), cudaMemcpyHostToDevice, *(encoder->stream));
        gpujpeg_cuda_check_error("Encoder raw data copy", return -1);
    }
    else if (input->type == GPUJPEG_ENCODER_INPUT_GPU_IMAGE) {
        coder->d_data_raw = input->image;
    }
    else if ( input->type == GPUJPEG_ENCODER_INPUT_OPENGL_TEXTURE ) {
        assert(input->texture != NULL);

        GPUJPEG_CUSTOM_TIMER_START(encoder->def);

        // Create buffers if not already created
        if (coder->data_raw_size > coder->data_raw_allocated_size) {
            coder->data_raw_allocated_size = 0;

            // (Re)allocate raw data in device memory
            if (coder->d_data_raw_allocated != NULL) {
                cudaFree(coder->d_data_raw_allocated);
                coder->d_data_raw_allocated = NULL;
            }
            cudaMalloc((void**)&coder->d_data_raw_allocated, coder->data_raw_size);
            gpujpeg_cuda_check_error("Encoder raw data allocation", return -1);

            coder->data_raw_allocated_size = coder->data_raw_size;
        }
        coder->d_data_raw = coder->d_data_raw_allocated;

        // Map texture to CUDA
        int data_size = 0;
        uint8_t* d_data = gpujpeg_opengl_texture_map(input->texture, &data_size);
        assert(data_size == (coder->data_raw_size));

        GPUJPEG_CUSTOM_TIMER_STOP(encoder->def);
        coder->duration_memory_map = GPUJPEG_CUSTOM_TIMER_DURATION(encoder->def);

        // Copy image data from texture pixel buffer object to device data
        cudaMemcpyAsync(coder->d_data_raw, d_data, coder->data_raw_size * sizeof(uint8_t), cudaMemcpyDeviceToDevice, *(encoder->stream));

        GPUJPEG_CUSTOM_TIMER_START(encoder->def);

        // Unmap texture from CUDA
        gpujpeg_opengl_texture_unmap(input->texture);

        GPUJPEG_CUSTOM_TIMER_STOP(encoder->def);
        coder->duration_memory_unmap = GPUJPEG_CUSTOM_TIMER_DURATION(encoder->def);
    }
    else {
        // Unknown output type
        assert(0);
    }

    //gpujpeg_table_print(encoder->table[JPEG_COMPONENT_LUMINANCE]);
    //gpujpeg_table_print(encoder->table[JPEG_COMPONENT_CHROMINANCE]);

    GPUJPEG_CUSTOM_TIMER_START(encoder->in_gpu);

    // Preprocessing
    if (0 != gpujpeg_preprocessor_encode(encoder)) {
        return -1;
    }

    // Perform DCT and quantization
    if (0 != gpujpeg_dct_gpu(encoder)) {
        return -1;
    }

    // If restart interval is 0 then the GPU processing is in the end (even huffman coder will be performed on CPU)
    if (coder->param.restart_interval == 0) {
        GPUJPEG_CUSTOM_TIMER_STOP(encoder->in_gpu);
        coder->duration_in_gpu = GPUJPEG_CUSTOM_TIMER_DURATION(encoder->in_gpu);
    }

    // Initialize writer output buffer current position
    encoder->writer->buffer_current = encoder->writer->buffer;

    // Write header
    gpujpeg_writer_write_header(encoder);

    // Perform huffman coding on CPU (when restart interval is not set)
    if ( coder->param.restart_interval == 0 ) {
        // Copy quantized data from device memory to cpu memory
        cudaMemcpyAsync(coder->data_quantized, coder->d_data_quantized, coder->data_size * sizeof(int16_t), cudaMemcpyDeviceToHost, *(encoder->stream));

        // Wait for async operations before the coding
        cudaStreamSynchronize(*(encoder->stream));

        // Perform huffman coding
        if ( gpujpeg_huffman_cpu_encoder_encode(encoder) != 0 ) {
            fprintf(stderr, "[GPUJPEG] [Error] Huffman encoder on CPU failed!\n");
            return -1;
        }
    }
    // Perform huffman coding on GPU (when restart interval is set)
    else {
        // Perform huffman coding
        unsigned int output_size;
        if ( gpujpeg_huffman_gpu_encoder_encode(encoder, encoder->huffman_gpu_encoder, &output_size) != 0 ) {
            fprintf(stderr, "[GPUJPEG] [Error] Huffman encoder on GPU failed!\n");
            return -1;
        }

        // Copy compressed data from device memory to cpu memory
        if ( cudaSuccess != cudaMemcpyAsync(coder->data_compressed, coder->d_data_compressed, output_size, cudaMemcpyDeviceToHost, *(encoder->stream)) != 0 ) {
            return -1;
        }
        // Copy segments from device memory
        if ( cudaSuccess != cudaMemcpyAsync(coder->segment, coder->d_segment, coder->segment_count * sizeof(struct gpujpeg_segment), cudaMemcpyDeviceToHost, *(encoder->stream)) ) {
            return -1;
        }

        // Wait for async operations before formatting
        GPUJPEG_CUSTOM_TIMER_START(encoder->def);
        cudaStreamSynchronize(*(encoder->stream));
        GPUJPEG_CUSTOM_TIMER_STOP(encoder->def);
        coder->duration_waiting = GPUJPEG_CUSTOM_TIMER_DURATION(encoder->def);

        GPUJPEG_CUSTOM_TIMER_STOP(encoder->in_gpu);
        coder->duration_in_gpu = GPUJPEG_CUSTOM_TIMER_DURATION(encoder->in_gpu);

        GPUJPEG_CUSTOM_TIMER_START(encoder->def);
        if ( coder->param.interleaved == 1 ) {
            // Write scan header (only one scan is written, that contains all color components data)
            gpujpeg_writer_write_scan_header(encoder, 0);

            // Write scan data
            for ( int segment_index = 0; segment_index < coder->segment_count; segment_index++ ) {
                struct gpujpeg_segment* segment = &coder->segment[segment_index];

                gpujpeg_writer_write_segment_info(encoder);

                // Copy compressed data to writer
                memcpy(
                    encoder->writer->buffer_current,
                    &coder->data_compressed[segment->data_compressed_index],
                    segment->data_compressed_size
                );
                encoder->writer->buffer_current += segment->data_compressed_size;
                //printf("Compressed data %d bytes\n", segment->data_compressed_size);
            }
            // Remove last restart marker in scan (is not needed)
            encoder->writer->buffer_current -= 2;

            gpujpeg_writer_write_segment_info(encoder);
        }
        else {
            // Write huffman coder results as one scan for each color component
            int segment_index = 0;
            for ( int comp = 0; comp < coder->param_image.comp_count; comp++ ) {
                // Write scan header
                gpujpeg_writer_write_scan_header(encoder, comp);
                // Write scan data
                for ( int index = 0; index < coder->component[comp].segment_count; index++ ) {
                    struct gpujpeg_segment* segment = &coder->segment[segment_index];

                    gpujpeg_writer_write_segment_info(encoder);

                    // Copy compressed data to writer
                    memcpy(
                        encoder->writer->buffer_current,
                        &coder->data_compressed[segment->data_compressed_index],
                        segment->data_compressed_size
                    );
                    encoder->writer->buffer_current += segment->data_compressed_size;
                    //printf("Compressed data %d bytes\n", segment->data_compressed_size);

                    segment_index++;
                }
                // Remove last restart marker in scan (is not needed)
                encoder->writer->buffer_current -= 2;

                gpujpeg_writer_write_segment_info(encoder);
            }
        }
        GPUJPEG_CUSTOM_TIMER_STOP(encoder->def);
        coder->duration_stream = GPUJPEG_CUSTOM_TIMER_DURATION(encoder->def);
    }
    gpujpeg_writer_emit_marker(encoder->writer, GPUJPEG_MARKER_EOI);

    // Set compressed image
    *image_compressed = encoder->writer->buffer;
    *image_compressed_size = encoder->writer->buffer_current - encoder->writer->buffer;

    coder->d_data_raw = NULL;

    return 0;
}

/** Documented at declaration */
int
gpujpeg_encoder_destroy(struct gpujpeg_encoder* encoder)
{
    assert(encoder != NULL);

    if (encoder->huffman_gpu_encoder != NULL) {
        gpujpeg_huffman_gpu_encoder_destroy(encoder->huffman_gpu_encoder);
    }
    if (gpujpeg_coder_deinit(&encoder->coder) != 0) {
        return -1;
    }
    for (int comp_type = 0; comp_type < GPUJPEG_COMPONENT_TYPE_COUNT; comp_type++) {
        if (encoder->table_quantization[comp_type].d_table != NULL) {
            cudaFree(encoder->table_quantization[comp_type].d_table);
        }
    }
    if (encoder->writer != NULL) {
        gpujpeg_writer_destroy(encoder->writer);
    }
    if (encoder->allocatedStream != NULL) {
        cudaStreamDestroy(*(encoder->allocatedStream));
        free(encoder->allocatedStream);
        encoder->allocatedStream = NULL;
        encoder->stream = NULL;
    }

    free(encoder);

    return 0;
}
