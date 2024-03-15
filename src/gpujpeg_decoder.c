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

#include "../libgpujpeg/gpujpeg_decoder.h"
#include "gpujpeg_dct_cpu.h"
#include "gpujpeg_dct_gpu.h"
#include "gpujpeg_decoder_internal.h"
#include "gpujpeg_huffman_cpu_decoder.h"
#include "gpujpeg_huffman_gpu_decoder.h"
#include "gpujpeg_postprocessor.h"
#include "gpujpeg_util.h"

/* Documented at declaration */
void
gpujpeg_decoder_output_set_default(struct gpujpeg_decoder_output* output)
{

    output->type = GPUJPEG_DECODER_OUTPUT_INTERNAL_BUFFER;
    output->data = NULL;
    output->data_size = 0;
    output->texture = NULL;
}

/* Documented at declaration */
void
gpujpeg_decoder_output_set_custom(struct gpujpeg_decoder_output* output, uint8_t* custom_buffer)
{
    output->type = GPUJPEG_DECODER_OUTPUT_CUSTOM_BUFFER;
    output->data = custom_buffer;
    output->data_size = 0;
    output->texture = NULL;
}

/* Documented at declaration */
void
gpujpeg_decoder_output_set_texture(struct gpujpeg_decoder_output* output, struct gpujpeg_opengl_texture* texture)
{
    output->type = GPUJPEG_DECODER_OUTPUT_OPENGL_TEXTURE;
    output->data = NULL;
    output->data_size = 0;
    output->texture = texture;
}

/* Documented at declaration */
void
gpujpeg_decoder_output_set_cuda_buffer(struct gpujpeg_decoder_output* output)
{
    output->type = GPUJPEG_DECODER_OUTPUT_CUDA_BUFFER;
    output->data = NULL;
    output->data_size = 0;
    output->texture = NULL;
}

/* Documented at declaration */
void
gpujpeg_decoder_output_set_custom_cuda(struct gpujpeg_decoder_output* output, uint8_t* d_custom_buffer)
{
    output->type = GPUJPEG_DECODER_OUTPUT_CUSTOM_CUDA_BUFFER;
    output->data = d_custom_buffer;
    output->data_size = 0;
    output->texture = NULL;
}

/* Documented at declaration */
struct gpujpeg_decoder*
gpujpeg_decoder_create(cudaStream_t stream)
{
    struct gpujpeg_decoder* decoder = (struct gpujpeg_decoder*) calloc(1, sizeof(struct gpujpeg_decoder));
    if ( decoder == NULL )
        return NULL;

    // Get coder
    struct gpujpeg_coder* coder = &decoder->coder;

    // Initialize coder
    if ( gpujpeg_coder_init(coder) != 0 ) {
        return NULL;
    }

    // Set parameters
    gpujpeg_set_default_parameters(&coder->param);
    gpujpeg_image_set_default_parameters(&coder->param_image);
    coder->param_image.comp_count = 0;
    coder->param_image.width = 0;
    coder->param_image.height = 0;
    coder->param.restart_interval = 0;

    int result = 1;

    // Create reader
    decoder->reader = gpujpeg_reader_create();
    if ( decoder->reader == NULL )
        result = 0;

    // Allocate quantization tables in device memory
    for ( int comp_type = 0; comp_type < GPUJPEG_MAX_COMPONENT_COUNT; comp_type++ ) {
        if ( cudaSuccess != cudaMalloc((void**)&decoder->table_quantization[comp_type].d_table, 64 * sizeof(uint16_t)) )
            result = 0;
    }
    // Allocate huffman tables in device memory
    for ( int comp_type = 0; comp_type < GPUJPEG_MAX_COMPONENT_COUNT; comp_type++ ) {
        for ( int huff_type = 0; huff_type < GPUJPEG_HUFFMAN_TYPE_COUNT; huff_type++ ) {
            if ( cudaSuccess != cudaMalloc((void**)&decoder->d_table_huffman[comp_type][huff_type], sizeof(struct gpujpeg_table_huffman_decoder)) )
                result = 0;
            // gpujpeg_huffman_decoder_table_kernel() computes quick tables for 2 pair of Huffman tables, but eg. for grayscale only one pair is
            // present which causes the function potentially crash because computing from garbage values - memsetting to 0 fixes that
            if ( cudaSuccess != cudaMemset(decoder->d_table_huffman[comp_type][huff_type], 0, sizeof(struct gpujpeg_table_huffman_decoder)) ) {
                result = 0;
            }
        }
    }
    gpujpeg_cuda_check_error("Decoder table allocation", return NULL);

    // Init huffman encoder
    if ((decoder->huffman_gpu_decoder = gpujpeg_huffman_gpu_decoder_init()) == NULL) {
        result = 0;
    }

    // Stream
    decoder->stream = stream;

    if (result == 0) {
        gpujpeg_decoder_destroy(decoder);
        return NULL;
    }

    return decoder;
}

/* Documented at declaration */
int
gpujpeg_decoder_init(struct gpujpeg_decoder* decoder, const struct gpujpeg_parameters* param, const struct gpujpeg_image_parameters* param_image)
{
    // Get coder
    struct gpujpeg_coder* coder = &decoder->coder;

    coder->param.verbose = param->verbose;
    coder->param.perf_stats = param->perf_stats;
    if (param_image->width * param_image->height * param_image->comp_count == 0) {
        return 0;
    }

    // Check if (re)inialization is needed
    int change = 0;
    change |= coder->param_image.width != param_image->width;
    change |= coder->param_image.height != param_image->height;
    change |= coder->param_image.comp_count != param_image->comp_count;
    change |= coder->param.restart_interval != param->restart_interval;
    change |= coder->param.interleaved != param->interleaved;
    change |= coder->param.color_space_internal != param->color_space_internal;
    for ( int comp = 0; comp < param_image->comp_count; comp++ ) {
        change |= coder->param.sampling_factor[comp].horizontal != param->sampling_factor[comp].horizontal;
        change |= coder->param.sampling_factor[comp].vertical != param->sampling_factor[comp].vertical;
    }
    if ( change == 0 )
        return 0;

    // For now we can't reinitialize decoder, we can only do first initialization
    if ( coder->param_image.width != 0 || coder->param_image.height != 0 || coder->param_image.comp_count != 0 ) {
        fprintf(stderr, "[GPUJPEG] [Info] Reinitializing decoder.\n");
    }

    if (0 == gpujpeg_coder_init_image(coder, param, param_image, decoder->stream)) {
        fprintf(stderr, "[GPUJPEG] [Error] Failed to init coder image!\n");
        return -1;
    }

    // Init postprocessor
    if ( gpujpeg_preprocessor_decoder_init(&decoder->coder) != 0 ) {
        fprintf(stderr, "[GPUJPEG] [Error] Failed to init postprocessor!\n");
        return -1;
    }

    return 0;
}

/* Documented at declaration */
int
gpujpeg_decoder_decode(struct gpujpeg_decoder* decoder, uint8_t* image, size_t image_size, struct gpujpeg_decoder_output* output)
{
    // Get coder
    struct gpujpeg_coder* coder = &decoder->coder;
    int rc;
    int unsupp_gpu_huffman_params = 0;

    GPUJPEG_CUSTOM_TIMER_START(coder->duration_stream, coder->param.perf_stats, decoder->stream, return -1);

    // Read JPEG image data
    if (0 != (rc = gpujpeg_reader_read_image(decoder, image, image_size))) {
        fprintf(stderr, "[GPUJPEG] [Error] Decoder failed when decoding image data!\n");
        return rc;
    }

    GPUJPEG_CUSTOM_TIMER_STOP(coder->duration_stream, coder->param.perf_stats, decoder->stream, return -1);

    // check if params is ok for GPU decoder
    for (int i = 0; i < decoder->coder.param_image.comp_count; ++i) {
        // packed_block_info_ptr holds only component type
        if ( decoder->comp_table_huffman_map[i][GPUJPEG_HUFFMAN_DC] != decoder->comp_table_huffman_map[i][GPUJPEG_HUFFMAN_AC] ) {
            fprintf(stderr, "[GPUJPEG] [Warning] Using different table DC/AC indices (%d and %d) for component %d (ID %d)! Using Huffman CPU decoder. Please report to GPUJPEG developers.\n", decoder->comp_table_huffman_map[i][GPUJPEG_HUFFMAN_AC], decoder->comp_table_huffman_map[i][GPUJPEG_HUFFMAN_DC], i, decoder->comp_id[i]);
            unsupp_gpu_huffman_params = 1;
        }
        // only DC/AC tables 0 and 1 are processed gpujpeg_huffman_decoder_table_kernel()
        if ( decoder->comp_table_huffman_map[i][GPUJPEG_HUFFMAN_DC] > 1 || decoder->comp_table_huffman_map[i][GPUJPEG_HUFFMAN_AC] > 1 ) {
            fprintf(stderr, "[GPUJPEG] [Warning] Using Huffman tables (%d, %d) implies extended process! Using Huffman CPU decoder. Please report to GPUJPEG developers.\n", decoder->comp_table_huffman_map[i][GPUJPEG_HUFFMAN_AC], decoder->comp_table_huffman_map[i][GPUJPEG_HUFFMAN_DC]);
            unsupp_gpu_huffman_params = 1;
        }
    }

    // Perform huffman decoding on CPU (when there are not enough segments to saturate GPU)
    if (coder->segment_count < 32 || unsupp_gpu_huffman_params) {
        GPUJPEG_CUSTOM_TIMER_START(coder->duration_huffman_coder, coder->param.perf_stats, decoder->stream, return -1);
        if (0 != gpujpeg_huffman_cpu_decoder_decode(decoder)) {
            fprintf(stderr, "[GPUJPEG] [Error] Huffman decoder failed!\n");
            return -1;
        }
        GPUJPEG_CUSTOM_TIMER_STOP(coder->duration_huffman_coder, coder->param.perf_stats, decoder->stream, return -1);

        // Copy quantized data to device memory from cpu memory
        GPUJPEG_CUSTOM_TIMER_START(coder->duration_memory_to, coder->param.perf_stats, decoder->stream, return -1);
        cudaMemcpyAsync(coder->d_data_quantized, coder->data_quantized, coder->data_size * sizeof(int16_t), cudaMemcpyHostToDevice, decoder->stream);
        GPUJPEG_CUSTOM_TIMER_STOP(coder->duration_memory_to, coder->param.perf_stats, decoder->stream, return -1);

        GPUJPEG_CUSTOM_TIMER_START(coder->duration_in_gpu, coder->param.perf_stats, decoder->stream, return -1);
    }
    // Perform huffman decoding on GPU (when there are enough segments to saturate GPU)
    else {
        GPUJPEG_CUSTOM_TIMER_START(coder->duration_memory_to, coder->param.perf_stats, decoder->stream, return -1);

        // Copy scan data to device memory
        cudaMemcpyAsync(coder->d_data_compressed, coder->data_compressed, decoder->data_compressed_size * sizeof(uint8_t), cudaMemcpyHostToDevice, decoder->stream);
        gpujpeg_cuda_check_error("Decoder copy compressed data", return -1);

        // Copy segments to device memory
        cudaMemcpyAsync(coder->d_segment, coder->segment, decoder->segment_count * sizeof(struct gpujpeg_segment), cudaMemcpyHostToDevice, decoder->stream);
        gpujpeg_cuda_check_error("Decoder copy compressed data", return -1);

        GPUJPEG_CUSTOM_TIMER_STOP(coder->duration_memory_to, coder->param.perf_stats, decoder->stream, return -1);

        GPUJPEG_CUSTOM_TIMER_START(coder->duration_in_gpu, coder->param.perf_stats, decoder->stream, return -1);
        GPUJPEG_CUSTOM_TIMER_START(coder->duration_huffman_coder, coder->param.perf_stats, decoder->stream, return -1);
        // Perform huffman decoding
        if (0 != gpujpeg_huffman_gpu_decoder_decode(decoder)) {
            fprintf(stderr, "[GPUJPEG] [Error] Huffman decoder on GPU failed!\n");
            return -1;
        }
        GPUJPEG_CUSTOM_TIMER_STOP(coder->duration_huffman_coder, coder->param.perf_stats, decoder->stream, return -1);
    }

    GPUJPEG_CUSTOM_TIMER_START(coder->duration_dct_quantization, coder->param.perf_stats, decoder->stream, return -1);

    // Perform IDCT and dequantization (own CUDA implementation)
    if (0 != gpujpeg_idct_gpu(decoder)) {
        return -1;
    }

    GPUJPEG_CUSTOM_TIMER_STOP(coder->duration_dct_quantization, coder->param.perf_stats, decoder->stream, return -1);

    // Create buffers if not already created
    if (coder->data_raw == NULL) {
        if (cudaSuccess != cudaMallocHost((void**)&coder->data_raw, coder->data_raw_size * sizeof(uint8_t))) {
            return -1;
        }
    }
    if (coder->d_data_raw_allocated == NULL) {
        if (cudaSuccess != cudaMalloc((void**)&coder->d_data_raw_allocated, coder->data_raw_size * sizeof(uint8_t))) {
            return -1;
        }
    }

    // Select CUDA output buffer
    if (output->type == GPUJPEG_DECODER_OUTPUT_CUSTOM_CUDA_BUFFER) {
        // Image should be directly decoded into custom CUDA buffer
        coder->d_data_raw = output->data;
    }
    else if (output->type == GPUJPEG_DECODER_OUTPUT_OPENGL_TEXTURE && output->texture->texture_callback_attach_opengl == NULL) {
        GPUJPEG_CUSTOM_TIMER_START(coder->duration_memory_map, coder->param.perf_stats, decoder->stream, return -1);

        // Use OpenGL texture as decoding destination
        size_t data_size = 0;
        uint8_t* d_data = gpujpeg_opengl_texture_map(output->texture, &data_size);
        assert(data_size == coder->data_raw_size);
        coder->d_data_raw = d_data;

        GPUJPEG_CUSTOM_TIMER_STOP(coder->duration_memory_map, coder->param.perf_stats, decoder->stream, return -1);
    }
    else {
        // Use internal CUDA buffer as decoding destination
        coder->d_data_raw = coder->d_data_raw_allocated;
    }

    // Preprocessing
    GPUJPEG_CUSTOM_TIMER_START(coder->duration_preprocessor, coder->param.perf_stats, decoder->stream, return -1);
    rc = gpujpeg_preprocessor_decode(&decoder->coder, decoder->stream);
    if (rc != GPUJPEG_NOERR) {
        return rc;
    }
    GPUJPEG_CUSTOM_TIMER_STOP(coder->duration_preprocessor, coder->param.perf_stats, decoder->stream, return -1);

    // Wait for async operations before copying from the device
    cudaStreamSynchronize(decoder->stream);

    GPUJPEG_CUSTOM_TIMER_STOP(coder->duration_in_gpu, coder->param.perf_stats, decoder->stream, return -1);

    // Set decompressed image size
    output->data_size = coder->data_raw_size * sizeof(uint8_t);
    output->param_image = decoder->coder.param_image;
    if (output->param_image.color_space == GPUJPEG_NONE) {
        output->param_image.color_space = decoder->coder.param.color_space_internal;
    }

    // Set decompressed image
    if (output->type == GPUJPEG_DECODER_OUTPUT_INTERNAL_BUFFER) {
        GPUJPEG_CUSTOM_TIMER_START(coder->duration_memory_from, coder->param.perf_stats, decoder->stream, return -1);

        // Copy decompressed image to host memory
        cudaMemcpy(coder->data_raw, coder->d_data_raw, coder->data_raw_size * sizeof(uint8_t), cudaMemcpyDeviceToHost);

        GPUJPEG_CUSTOM_TIMER_STOP(coder->duration_memory_from, coder->param.perf_stats, decoder->stream, return -1);

        // Set output to internal buffer
        output->data = coder->data_raw;
    }
    else if (output->type == GPUJPEG_DECODER_OUTPUT_CUSTOM_BUFFER) {
        GPUJPEG_CUSTOM_TIMER_START(coder->duration_memory_from, coder->param.perf_stats, decoder->stream, return -1);

        assert(output->data != NULL);

        // Copy decompressed image to host memory
        cudaMemcpy(output->data, coder->d_data_raw, coder->data_raw_size * sizeof(uint8_t), cudaMemcpyDeviceToHost);

        GPUJPEG_CUSTOM_TIMER_STOP(coder->duration_memory_from, coder->param.perf_stats, decoder->stream, return -1);
    }
    else if (output->type == GPUJPEG_DECODER_OUTPUT_OPENGL_TEXTURE) {
        // If OpenGL texture wasn't mapped and used directly for decoding into it
        if (output->texture->texture_callback_attach_opengl != NULL) {
            GPUJPEG_CUSTOM_TIMER_START(coder->duration_memory_map, coder->param.perf_stats, decoder->stream, return -1);

            // Map OpenGL texture
            size_t data_size = 0;
            uint8_t* d_data = gpujpeg_opengl_texture_map(output->texture, &data_size);
            assert(data_size == coder->data_raw_size);

            GPUJPEG_CUSTOM_TIMER_STOP(coder->duration_memory_map, coder->param.perf_stats, decoder->stream, return -1);

            GPUJPEG_CUSTOM_TIMER_START(coder->duration_memory_from, coder->param.perf_stats, decoder->stream, return -1);

            // Copy decompressed image to texture pixel buffer object device data
            cudaMemcpy(d_data, coder->d_data_raw, coder->data_raw_size * sizeof(uint8_t), cudaMemcpyDeviceToDevice);

            GPUJPEG_CUSTOM_TIMER_STOP(coder->duration_memory_from, coder->param.perf_stats, decoder->stream, return -1);
        }

        GPUJPEG_CUSTOM_TIMER_START(coder->duration_memory_unmap, coder->param.perf_stats, decoder->stream, return -1);

        // Unmap OpenGL texture
        gpujpeg_opengl_texture_unmap(output->texture);

        GPUJPEG_CUSTOM_TIMER_STOP(coder->duration_memory_unmap, coder->param.perf_stats, decoder->stream, return -1);
    }
    else if (output->type == GPUJPEG_DECODER_OUTPUT_CUDA_BUFFER) {
        // Copy decompressed image to texture pixel buffer object device data
        output->data = coder->d_data_raw;
    }
    else if (output->type == GPUJPEG_DECODER_OUTPUT_CUSTOM_CUDA_BUFFER) {
        // Image was already directly decoded into custom CUDA buffer
        output->data = coder->d_data_raw;
    }
    else {
        // Unknown output type
        assert(0);
    }

    return 0;
}

int
gpujpeg_decoder_get_stats(struct gpujpeg_decoder *decoder, struct gpujpeg_duration_stats *stats)
{
    return gpujpeg_coder_get_stats(&decoder->coder, stats);
}

void
gpujpeg_decoder_set_output_format(struct gpujpeg_decoder* decoder,
                enum gpujpeg_color_space color_space,
                enum gpujpeg_pixel_format sampling_factor)
{
        decoder->coder.param_image.color_space = color_space;
        decoder->coder.param_image.pixel_format = sampling_factor;
}

/* Documented at declaration */
int
gpujpeg_decoder_destroy(struct gpujpeg_decoder* decoder)
{
    assert(decoder != NULL);

    if (0 != gpujpeg_coder_deinit(&decoder->coder)) {
        return -1;
    }

    for (int comp_type = 0; comp_type < GPUJPEG_MAX_COMPONENT_COUNT; comp_type++) {
        if (decoder->table_quantization[comp_type].d_table != NULL) {
            cudaFree(decoder->table_quantization[comp_type].d_table);
        }
    }

    for ( int comp_type = 0; comp_type < GPUJPEG_MAX_COMPONENT_COUNT; comp_type++ ) {
        for ( int huff_type = 0; huff_type < GPUJPEG_HUFFMAN_TYPE_COUNT; huff_type++ ) {
            cudaFree(decoder->d_table_huffman[comp_type][huff_type]);
        }
    }

    if (decoder->reader != NULL) {
        gpujpeg_reader_destroy(decoder->reader);
    }

    if (decoder->huffman_gpu_decoder != NULL) {
        gpujpeg_huffman_gpu_decoder_destroy(decoder->huffman_gpu_decoder);
    }

    free(decoder);

#ifdef DEBUG
    cudaDeviceReset(); // to allow "cuda-memcheck --leak-check full"
#endif // defined DEBUG

    return 0;
}

int
gpujpeg_decoder_get_image_info(uint8_t *image, size_t image_size, struct gpujpeg_image_parameters *param_image, struct gpujpeg_parameters *param, int *segment_count) {
    return gpujpeg_reader_get_image_info(image, image_size, param_image, param, segment_count);
}

/* vi: set expandtab sw=4 : */
