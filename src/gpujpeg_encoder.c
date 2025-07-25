/**
 * @file
 * Copyright (c) 2011-2025, CESNET
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

#include <assert.h>
#include <math.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#define strcasecmp _stricmp
#endif

#include "../libgpujpeg/gpujpeg_common.h"
#include "../libgpujpeg/gpujpeg_encoder.h"
#include "gpujpeg_common_internal.h"
#include "gpujpeg_preprocessor.h"
#include "gpujpeg_dct_cpu.h"
#include "gpujpeg_dct_gpu.h"
#include "gpujpeg_huffman_cpu_encoder.h"
#include "gpujpeg_huffman_gpu_encoder.h"
#include "gpujpeg_marker.h"
#include "gpujpeg_util.h"

/* Documented at declaration */
void
gpujpeg_encoder_input_set_image(struct gpujpeg_encoder_input* input, uint8_t* image)
{
    input->type = GPUJPEG_ENCODER_INPUT_IMAGE;
    input->image = image;
    input->texture = NULL;
}

/* Documented at declaration */
void
gpujpeg_encoder_input_set_gpu_image(struct gpujpeg_encoder_input* input, uint8_t* image)
{
    input->type = GPUJPEG_ENCODER_INPUT_GPU_IMAGE;
    input->image = image;
    input->texture = NULL;
}

/* Documented at declaration */
void
gpujpeg_encoder_input_set_texture(struct gpujpeg_encoder_input* input, struct gpujpeg_opengl_texture* texture)
{
    input->type = GPUJPEG_ENCODER_INPUT_OPENGL_TEXTURE;
    input->image = NULL;
    input->texture = texture;
}

struct gpujpeg_encoder_input
gpujpeg_encoder_input_image(uint8_t* image)
{
    struct gpujpeg_encoder_input ret;
    gpujpeg_encoder_input_set_image(&ret, image);
    return ret;
}

struct gpujpeg_encoder_input
gpujpeg_encoder_input_gpu_image(uint8_t* image)
{
    struct gpujpeg_encoder_input ret;
    gpujpeg_encoder_input_set_gpu_image(&ret, image);
    return ret;
}

struct gpujpeg_encoder_input
gpujpeg_encoder_input_texture(struct gpujpeg_opengl_texture* texture)
{
    struct gpujpeg_encoder_input ret;
    gpujpeg_encoder_input_set_texture(&ret, texture);
    return ret;
}

/* Documented at declaration */
struct gpujpeg_encoder*
gpujpeg_encoder_create(cudaStream_t stream)
{
    gpujpeg_init_term_colors();
    struct gpujpeg_encoder* encoder = (struct gpujpeg_encoder*) calloc(1, sizeof(struct gpujpeg_encoder));
    if ( encoder == NULL ) {
        return NULL;
    }

    // Stream
    encoder->stream = stream;

    // Get coder
    struct gpujpeg_coder* coder = &encoder->coder;

    int result = 1;

    // Create writer
    encoder->writer = gpujpeg_writer_create();
    if ( encoder->writer == NULL )
        result = 0;

    // Initialize coder
    if ( gpujpeg_coder_init(coder) != 0 )
        result = 0;
    coder->encoder = 1;

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

    if ( result == 0 ) {
        gpujpeg_encoder_destroy(encoder);
        return NULL;
    }

    return encoder;
}

/* Documented at declaration */
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
    int pixels = 10000;
    while (true) {
        param_image->width = (int) sqrt((float) pixels);
        param_image->height = (pixels + param_image->width - 1) / param_image->width;
        //printf("\nIteration #%d (pixels: %d, size: %dx%d)\n", iteration++, pixels, param_image->width, param_image->height);
        size_t image_memory_size = gpujpeg_coder_init_image(&coder, param, param_image, cudaStreamDefault);
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

/* Documented at declaration */
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

    size_t image_memory_size = gpujpeg_coder_init_image(&coder, param, param_image, cudaStreamDefault);
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

/* Documented at declaration */
int gpujpeg_encoder_allocate(struct gpujpeg_encoder * encoder, const struct gpujpeg_parameters * param, const struct gpujpeg_image_parameters * param_image, enum gpujpeg_encoder_input_type image_input_type)
{
    // Get coder
    struct gpujpeg_coder* coder = &encoder->coder;

    // Allocate internal buffers
    if (0 == gpujpeg_coder_init_image(coder, param, param_image, encoder->stream)) {
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

int
gpujpeg_encoder_suggest_restart_interval(const struct gpujpeg_image_parameters* param_image,
                                         gpujpeg_sampling_factor_t subsampling, bool interleaved, int verbose)
{
    int restart_interval;
    const int comp_count = gpujpeg_pixel_format_get_comp_count(param_image->pixel_format);
    // Adjust according to Mpix count
    double coefficient = ((double)param_image->width * param_image->height * comp_count) / (1000000.0 * 3.0);
    if ( coefficient < 1.0 ) {
        restart_interval = 4;
    } else if ( coefficient < 3.0 ) {
        restart_interval = 8;
    } else if ( coefficient < 9.0 ) {
        restart_interval = 10;
    } else {
        restart_interval = 12;
    }
    // when chroma subsampling and interleaving is enabled, the restart interval should be smaller
    if ( subsampling != GPUJPEG_SUBSAMPLING_444 && interleaved ) {
        restart_interval /= 2;
    }
    // when not interleaved, restart interval applies to each segment so actual number is higher
    if ( !interleaved ) {
        restart_interval *= comp_count;
    }
    VERBOSE_MSG(verbose, "Auto-adjusting restart interval to %d for better performance.\n", restart_interval);
    return restart_interval;
}

static struct gpujpeg_parameters
adjust_params(struct gpujpeg_coder* coder, const struct gpujpeg_parameters* param,
              const struct gpujpeg_image_parameters* param_image, bool img_changed)
{
    struct gpujpeg_parameters param_adjusted = *param;
    if ( param->comp_count == 0 ) {
        if ( img_changed ) {

            param_adjusted.comp_count =
                MIN(gpujpeg_pixel_format_get_comp_count(param_image->pixel_format), GPUJPEG_3_COMPONENTS);
            memcpy(param_adjusted.sampling_factor, gpujpeg_pixel_format_get_sampling_factor(param_image->pixel_format),
                   sizeof param_adjusted.sampling_factor);
        }
        else {
            param_adjusted.comp_count = coder->param.comp_count;
            memcpy(param_adjusted.sampling_factor, coder->param.sampling_factor, sizeof param_adjusted.sampling_factor);
        }
    }
    if ( param->restart_interval == RESTART_AUTO ) {
        if ( img_changed || param_adjusted.interleaved != coder->param.interleaved ) {
            param_adjusted.restart_interval = gpujpeg_encoder_suggest_restart_interval(
                param_image, gpujpeg_make_sampling_factor2(param_adjusted.comp_count, param_adjusted.sampling_factor),
                param_adjusted.interleaved, param_adjusted.verbose);
        }
        else {
            param_adjusted.restart_interval = coder->param.restart_interval;
        }
    }
    return param_adjusted;
}

/* Documented at declaration */
int
gpujpeg_encoder_encode(struct gpujpeg_encoder* encoder, const struct gpujpeg_parameters* param,
                       const struct gpujpeg_image_parameters* param_image, const struct gpujpeg_encoder_input* input,
                       uint8_t** image_compressed, size_t* image_compressed_size)
{
    assert(param->comp_count <= GPUJPEG_MAX_COMPONENT_COUNT);
    assert(param->quality >= 0 && param->quality <= 100);
    assert(param->restart_interval >= RESTART_AUTO);
    assert(param->interleaved == 0 || param->interleaved == 1);

    // Get coder
    struct gpujpeg_coder* coder = &encoder->coder;
    int rc;

    const bool img_changed = !gpujpeg_image_parameters_equals(&coder->param_image, param_image);
    struct gpujpeg_parameters param_adjusted = adjust_params(coder, param, param_image, img_changed);
    param_adjusted.perf_stats = param->perf_stats || param->verbose >= GPUJPEG_LL_STATUS;

    coder->start_time = param_adjusted.perf_stats ? gpujpeg_get_time() : 0;

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
    if (0 == gpujpeg_coder_init_image(coder, &param_adjusted, param_image, encoder->stream)) {
        fprintf(stderr, "[GPUJPEG] [Error] Failed to init image encoding!\n");
        return -1;
    }

    // (Re)initialize writer
    if ( gpujpeg_writer_init(encoder->writer, coder->param.comp_count, &encoder->coder.param_image) != 0 ) {
        fprintf(stderr, "[GPUJPEG] [Error] Failed to init writer!\n");
        return -1;
    }

    if ( img_changed ) {
        // (Re)initialize preprocessor
        if ( gpujpeg_preprocessor_encoder_init(&encoder->coder) != 0 ) {
            fprintf(stderr, "[GPUJPEG] [Error] Failed to init preprocessor!\n");
            return -1;
        }
    }
    coder->init_end_time = param_adjusted.perf_stats ? gpujpeg_get_time() : 0;

    // Load input image
    if ( input->type == GPUJPEG_ENCODER_INPUT_IMAGE ) {
        GPUJPEG_CUSTOM_TIMER_START(coder->duration_memory_to, coder->param.perf_stats, encoder->stream, return -1);

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
        cudaMemcpyAsync(coder->d_data_raw, input->image, coder->data_raw_size * sizeof(uint8_t), cudaMemcpyHostToDevice, encoder->stream);
        gpujpeg_cuda_check_error("Encoder raw data copy", return -1);

        GPUJPEG_CUSTOM_TIMER_STOP(coder->duration_memory_to, coder->param.perf_stats, encoder->stream, return -1);
    }
    else if (input->type == GPUJPEG_ENCODER_INPUT_GPU_IMAGE) {
        coder->d_data_raw = input->image;
    }
    else if ( input->type == GPUJPEG_ENCODER_INPUT_OPENGL_TEXTURE ) {
        assert(input->texture != NULL);

        GPUJPEG_CUSTOM_TIMER_START(coder->duration_memory_map, coder->param.perf_stats, encoder->stream, return -1);

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
        size_t data_size = 0;
        uint8_t* d_data = gpujpeg_opengl_texture_map(input->texture, &data_size);
        assert(data_size == (coder->data_raw_size));

        GPUJPEG_CUSTOM_TIMER_STOP(coder->duration_memory_map, coder->param.perf_stats, encoder->stream, return -1);

        GPUJPEG_CUSTOM_TIMER_START(coder->duration_memory_to, coder->param.perf_stats, encoder->stream, return -1);
        // Copy image data from texture pixel buffer object to device data
        cudaMemcpyAsync(coder->d_data_raw, d_data, coder->data_raw_size * sizeof(uint8_t), cudaMemcpyDeviceToDevice, encoder->stream);
        GPUJPEG_CUSTOM_TIMER_STOP(coder->duration_memory_to, coder->param.perf_stats, encoder->stream, return -1);

        GPUJPEG_CUSTOM_TIMER_START(coder->duration_memory_unmap, coder->param.perf_stats, encoder->stream, return -1);

        // Unmap texture from CUDA
        gpujpeg_opengl_texture_unmap(input->texture);

        GPUJPEG_CUSTOM_TIMER_STOP(coder->duration_memory_unmap, coder->param.perf_stats, encoder->stream, return -1);
    }
    else {
        // Unknown output type
        assert(0);
    }

    //gpujpeg_table_print(encoder->table[JPEG_COMPONENT_LUMINANCE]);
    //gpujpeg_table_print(encoder->table[JPEG_COMPONENT_CHROMINANCE]);

    GPUJPEG_CUSTOM_TIMER_START(coder->duration_in_gpu, coder->param.perf_stats, encoder->stream, return -1);
    GPUJPEG_CUSTOM_TIMER_START(coder->duration_preprocessor, coder->param.perf_stats, encoder->stream, return -1);

    // Preprocessing
    rc = gpujpeg_preprocessor_encode(encoder);
    if (rc != GPUJPEG_NOERR) {
        return rc;
    }

    GPUJPEG_CUSTOM_TIMER_STOP(coder->duration_preprocessor, coder->param.perf_stats, encoder->stream, return -1);
    GPUJPEG_CUSTOM_TIMER_START(coder->duration_dct_quantization, coder->param.perf_stats, encoder->stream, return -1);

    // Perform DCT and quantization
    if (0 != gpujpeg_dct_gpu(encoder)) {
        return -1;
    }

    // If restart interval is 0 then the GPU processing is in the end (even huffman coder will be performed on CPU)
    if (coder->param.restart_interval == 0) {
        GPUJPEG_CUSTOM_TIMER_STOP(coder->duration_in_gpu, coder->param.perf_stats, encoder->stream, return -1);
    }

    // Initialize writer output buffer current position
    encoder->writer->buffer_current = encoder->writer->buffer;

    // Write header
    gpujpeg_writer_write_header(encoder);

    GPUJPEG_CUSTOM_TIMER_STOP(coder->duration_dct_quantization, coder->param.perf_stats, encoder->stream, return -1);

    // Perform huffman coding on CPU (when restart interval is not set)
    if ( coder->param.restart_interval == 0 ) {
        GPUJPEG_CUSTOM_TIMER_START(coder->duration_memory_from, coder->param.perf_stats, encoder->stream, return -1);
        if (coder->data_quantized == NULL) {
            if (gpujpeg_coder_allocate_cpu_huffman_buf(coder) != 0) {
                return -1;
            }
        }
        // Copy quantized data from device memory to cpu memory
        cudaMemcpyAsync(coder->data_quantized, coder->d_data_quantized, coder->data_size * sizeof(int16_t), cudaMemcpyDeviceToHost, encoder->stream);

        // Wait for async operations before the coding
        cudaStreamSynchronize(encoder->stream);
        GPUJPEG_CUSTOM_TIMER_STOP(coder->duration_memory_from, coder->param.perf_stats, encoder->stream, return -1);

        GPUJPEG_CUSTOM_TIMER_START(coder->duration_huffman_coder, coder->param.perf_stats, encoder->stream, return -1);
        // Perform huffman coding
        if ( gpujpeg_huffman_cpu_encoder_encode(encoder) != 0 ) {
            fprintf(stderr, "[GPUJPEG] [Error] Huffman encoder on CPU failed!\n");
            return -1;
        }
        GPUJPEG_CUSTOM_TIMER_STOP(coder->duration_huffman_coder, coder->param.perf_stats, encoder->stream, return -1);
    }
    // Perform huffman coding on GPU (when restart interval is set)
    else {
        GPUJPEG_CUSTOM_TIMER_START(coder->duration_huffman_coder, coder->param.perf_stats, encoder->stream, return -1);
        // Perform huffman coding
        unsigned int output_size;
        if ( gpujpeg_huffman_gpu_encoder_encode(encoder, encoder->huffman_gpu_encoder, &output_size) != 0 ) {
            fprintf(stderr, "[GPUJPEG] [Error] Huffman encoder on GPU failed!\n");
            return -1;
        }

        GPUJPEG_CUSTOM_TIMER_STOP(coder->duration_huffman_coder, coder->param.perf_stats, encoder->stream, return -1);
        GPUJPEG_CUSTOM_TIMER_STOP(coder->duration_in_gpu, coder->param.perf_stats, encoder->stream, return -1);
        GPUJPEG_CUSTOM_TIMER_START(coder->duration_memory_from, coder->param.perf_stats, encoder->stream, return -1);

        // Copy compressed data from device memory to cpu memory
        gpujpeg_cuda_memcpy_async_partially_pinned(coder->data_compressed, coder->d_data_compressed, output_size,
                                                   cudaMemcpyDeviceToHost, encoder->stream,
                                                   coder->data_compressed_pinned_sz);
        gpujpeg_cuda_check_error("Encoder copy compressed data to memory", return -1);

        // Copy segments from device memory
        if ( cudaSuccess != cudaMemcpyAsync(coder->segment, coder->d_segment, coder->segment_count * sizeof(struct gpujpeg_segment), cudaMemcpyDeviceToHost, encoder->stream) ) {
            return -1;
        }

        // Wait for async operations before formatting
        cudaStreamSynchronize(encoder->stream);
        GPUJPEG_CUSTOM_TIMER_STOP(coder->duration_memory_from, coder->param.perf_stats, encoder->stream, return -1);

        GPUJPEG_CUSTOM_TIMER_START(coder->duration_stream, coder->param.perf_stats, encoder->stream, return -1);
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
            for ( int comp = 0; comp < coder->param.comp_count; comp++ ) {
                // Write scan header
                gpujpeg_writer_write_scan_header(encoder, comp);
                // Write scan data
                for ( int index = 0; index < coder->component[comp].segment_count; index++ ) {
                    struct gpujpeg_segment* segment = &coder->segment[segment_index];

                    gpujpeg_writer_write_segment_info(encoder);

                    assert(encoder->writer->buffer_current +
                               segment->data_compressed_size <=
                           encoder->writer->buffer +
                               encoder->writer->buffer_allocated_size);
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
        GPUJPEG_CUSTOM_TIMER_STOP(coder->duration_stream, coder->param.perf_stats, encoder->stream, return -1);
    }
    gpujpeg_writer_emit_marker(encoder->writer, GPUJPEG_MARKER_EOI);

    // Set compressed image
    *image_compressed = encoder->writer->buffer;
    *image_compressed_size = encoder->writer->buffer_current - encoder->writer->buffer;

    coder->d_data_raw = NULL;

    coder_process_stats(coder);
    if (coder->param.verbose >= GPUJPEG_LL_STATUS ) {
        const char* interleaved =
            coder->param.comp_count == 1 ? "" : (coder->param.interleaved ? " interleaved" : " non-interleaved");
        char buf[21];
        char* comp_size_delim = format_number_with_delim(*image_compressed_size, buf, sizeof buf);
        PRINTF("Compressed Size:%15s bytes %dx%d %s %s%s\n", comp_size_delim, coder->param_image.width,
               coder->param_image.height, gpujpeg_color_space_get_name(coder->param.color_space_internal),
               gpujpeg_subsampling_get_name(coder->param.comp_count, coder->param.sampling_factor), interleaved);
    }

    return 0;
}

int
gpujpeg_encoder_get_stats(struct gpujpeg_encoder *encoder, struct gpujpeg_duration_stats *stats)
{
    return gpujpeg_coder_get_stats(&encoder->coder, stats);
}

// Documented at declaration
void
gpujpeg_encoder_set_jpeg_header(struct gpujpeg_encoder *encoder, enum gpujpeg_header_type header_type)
{
    encoder->header_type = header_type;
}

static int
enc_opt_set_channel_remap(struct gpujpeg_encoder* encoder, const char* val)
{
    if ( strcmp(val, "help") == 0 ) {
        printf("syntax for " GPUJPEG_ENC_OPT_CHANNEL_REMAP ":\n");
        printf("\t\"XYZ\" or \"XYZW\" where the letters are input channel indices\n");
        printf("\tplaceholder 'Z' or 'F' can be used to set the channel to all-zeros or all-ones\n");
        printf("\n");
        printf("examples:\n");
        printf("\t\"1230\" or \"123F\" to map ARGB to RGBA\n");
        return GPUJPEG_ERROR;
    }
    const int mapped_count = strlen(val);
    if ( mapped_count > GPUJPEG_MAX_COMPONENT_COUNT ) {
        ERROR_MSG("Mapping for more than %d channels specified!\n", GPUJPEG_MAX_COMPONENT_COUNT);
        return GPUJPEG_ERROR;
    }
    encoder->coder.preprocessor.channel_remap = 0; // clear old
    while ( *val != '\0' ) {
        encoder->coder.preprocessor.channel_remap >>= 4;
        int src_chan = *val - '0';
        if ( *val == 'F' ) {
            src_chan = 4;
        }
        else if ( *val == 'Z' ) {
            src_chan = 5;
        }
        else if ( src_chan < 0 || src_chan >= mapped_count ) {
            ERROR_MSG("Invalid channel index %c for " GPUJPEG_ENC_OPT_CHANNEL_REMAP " (mapping %d channels)!\n", *val,
                      mapped_count);
            return GPUJPEG_ERROR;
        }
        encoder->coder.preprocessor.channel_remap |= src_chan << 12;
        val++;
    }
    encoder->coder.preprocessor.channel_remap |= mapped_count << 24;
    return GPUJPEG_NOERR;
}

GPUJPEG_API int
gpujpeg_encoder_set_option(struct gpujpeg_encoder* encoder, const char *opt, const char* val)
{
    if ( encoder == NULL || opt == NULL || val == NULL ) {
        return GPUJPEG_ERROR;
    }
    if ( strcmp(opt, GPUJPEG_ENC_OPT_OUT) == 0 ) {
        if ( strcmp(val, GPUJPEG_ENC_OUT_VAL_PAGEABLE) != 0 && strcmp(val, GPUJPEG_ENC_OUT_VAL_PINNED) != 0 ) {
            ERROR_MSG("Unexpeceted value %s for " GPUJPEG_ENC_OPT_OUT "\n", val);
            return GPUJPEG_ERROR;
        }
        encoder->writer->buffer_pinned = strcmp(val, GPUJPEG_ENC_OUT_VAL_PINNED) == 0;
        return GPUJPEG_NOERR;
    }
    if ( strcmp(opt, GPUJPEG_ENCODER_OPT_OUT_PINNED) == 0 ) {
        WARN_MSG("deprecated, use GPUJPEG_ENC_OUT_VAL_PAGEABLE (" GPUJPEG_ENC_OUT_VAL_PAGEABLE ") instead!\n");
        if ( strcmp(val, GPUJPEG_VAL_TRUE) != 0 && strcmp(val, GPUJPEG_VAL_FALSE) != 0 ) {
            ERROR_MSG("Unexpeceted value %s for " GPUJPEG_ENCODER_OPT_OUT_PINNED "\n", val);
            return GPUJPEG_ERROR;
        }
        encoder->writer->buffer_pinned = strcmp(val, GPUJPEG_VAL_TRUE) == 0;
        return GPUJPEG_NOERR;
    }
    if ( strcmp(opt, GPUJPEG_ENC_OPT_HDR) == 0 ) {
        if (strcasecmp(val, GPUJPEG_ENC_HDR_VAL_JFIF) == 0) {
            encoder->header_type = GPUJPEG_HEADER_JFIF;
        } else if (strcasecmp(val, GPUJPEG_ENC_HDR_VAL_ADOBE) == 0) {
            encoder->header_type = GPUJPEG_HEADER_ADOBE;
        } else if (strcasecmp(val, GPUJPEG_ENC_HDR_VAL_SPIFF) == 0) {
            encoder->header_type = GPUJPEG_HEADER_SPIFF;
        } else {
            ERROR_MSG("Unknown encoder header type: %s\n", val);
            return GPUJPEG_ERROR;
        }
        return GPUJPEG_NOERR;
    }
    if ( strcmp(opt, GPUJPEG_ENC_OPT_FLIPPED_BOOL) == 0 ) {
        if ( strcasecmp(val, GPUJPEG_VAL_TRUE) == 0 ) {
            encoder->coder.preprocessor.input_flipped = true;
        }
        else if ( strcasecmp(val, GPUJPEG_VAL_FALSE) == 0 ) {
            encoder->coder.preprocessor.input_flipped = false;
        }
        else {
            ERROR_MSG("Unknown option %s for " GPUJPEG_ENC_OPT_FLIPPED_BOOL "\n", val);
            return GPUJPEG_ERROR;
        }
        return GPUJPEG_NOERR;
    }
    if ( strcmp(opt, GPUJPEG_ENC_OPT_CHANNEL_REMAP) == 0 ) {
        return enc_opt_set_channel_remap(encoder, val);
    }
    ERROR_MSG("Invalid encoder option: %s!\n", opt);
    return GPUJPEG_ERROR;
}

/* Documented at declaration */
int
gpujpeg_encoder_destroy(struct gpujpeg_encoder* encoder)
{
    assert(encoder != NULL);

    coder_process_stats_overall(&encoder->coder);

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
        if (encoder->table_quantization[comp_type].d_table_forward != NULL) {
            cudaFree(encoder->table_quantization[comp_type].d_table_forward);
        }
    }
    if (encoder->writer != NULL) {
        gpujpeg_writer_destroy(encoder->writer);
    }

    free(encoder);

    return 0;
}
/* vim: set expandtab sw=4: */
