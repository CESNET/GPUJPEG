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

#ifndef GPUJPEG_ENCODER_H
#define GPUJPEG_ENCODER_H

#include "gpujpeg_common.h"

#ifdef __cplusplus
extern "C" {
#endif

struct gpujpeg_encoder;

/**
 * Encoder input type
 */
enum gpujpeg_encoder_input_type {
    /// Encoder will use custom input buffer
    GPUJPEG_ENCODER_INPUT_IMAGE,
    /// Encoder will use OpenGL Texture PBO Resource as input buffer
    GPUJPEG_ENCODER_INPUT_OPENGL_TEXTURE,
    /// Encoder will use custom GPU input buffer
    GPUJPEG_ENCODER_INPUT_GPU_IMAGE,
};

/**
 * Encoder input structure
 */
struct gpujpeg_encoder_input
{
    /// Output type
    enum gpujpeg_encoder_input_type type;

    /// Image data
    uint8_t* image;

    /// Registered OpenGL Texture
    struct gpujpeg_opengl_texture* texture;
};

/**
 * Set encoder input to image data
 *
 * @param encoder_input  Encoder input structure
 * @param image  Input image data
 * @return void
 */
GPUJPEG_API void
gpujpeg_encoder_input_set_image(struct gpujpeg_encoder_input* input, uint8_t* image);

/**
 * Set encoder input to GPU image data
 *
 * @param encoder_input  Encoder input structure
 * @param image GPU image data
 * @return void
 */
GPUJPEG_API void
gpujpeg_encoder_input_set_gpu_image(struct gpujpeg_encoder_input* input, uint8_t* image);

/**
 * Set encoder input to OpenGL texture
 *
 * @param encoder_input  Encoder input structure
 * @param texture_id  OpenGL texture id
 * @return void
 */
GPUJPEG_API void
gpujpeg_encoder_input_set_texture(struct gpujpeg_encoder_input* input, struct gpujpeg_opengl_texture* texture);

/**
 * Create JPEG encoder
 *
 * @param stream CUDA stream to be used, may be cudaStreamDefault (0x00)
 * @return encoder structure if succeeds, otherwise NULL
 */
GPUJPEG_API struct gpujpeg_encoder*
gpujpeg_encoder_create(cudaStream_t stream);

/**
 * Compute maximum number of image pixels (width x height) which can be encoded by given memory size.
 *
 * @param encoder
 * @param param
 * @param param_image
 * @param image_input_type
 * @param memory_size
 * @param max_pixels
 * @return size of used device memory in bytes if succeeds, otherwise 0
 */
GPUJPEG_API size_t
gpujpeg_encoder_max_pixels(struct gpujpeg_parameters * param, struct gpujpeg_image_parameters * param_image, enum gpujpeg_encoder_input_type image_input_type, size_t memory_size, int * max_pixels);

/**
 * Compute maximum size of device memory which will be used for encoding image with given number of pixels.
 *
 * @param encoder
 * @param param
 * @param param_image
 * @param image_input_type
 * @param max_pixels
 * @return size of required device memory in bytes if succeeds, otherwise 0
 */
GPUJPEG_API size_t
gpujpeg_encoder_max_memory(struct gpujpeg_parameters * param, struct gpujpeg_image_parameters * param_image, enum gpujpeg_encoder_input_type image_input_type, int max_pixels);

/**
 * Pre-allocate all encoding buffers for given image pixels.
 *
 * @param encoder
 * @param param
 * @param param_image
 * @param image_input_type
 * @return 0 if succeeds, otherwise nonzero
 */
GPUJPEG_API int
gpujpeg_encoder_allocate(struct gpujpeg_encoder* encoder, const struct gpujpeg_parameters * param, const struct gpujpeg_image_parameters * param_image, enum gpujpeg_encoder_input_type image_input_type);

/**
 * Compress image by encoder
 *
 * @param encoder  Encoder structure
 * @param param  Parameters for coder
 * @param param_image  Parameters for image data
 * @param image  Source image data
 * @param image_compressed  Pointer to variable where compressed image data buffer will be placed
 * @param image_compressed_size  Pointer to variable where compressed image size will be placed.
 *                               Buffer is owned by encoder and must not be freed by caller. Buffer
 *                               is valid until next gpujpeg_encoder_encode() call.
 * @return 0 if succeeds, otherwise nonzero
 */
GPUJPEG_API int
gpujpeg_encoder_encode(struct gpujpeg_encoder* encoder, const struct gpujpeg_parameters* param,
                       const struct gpujpeg_image_parameters* param_image, const struct gpujpeg_encoder_input* input,
                       uint8_t** image_compressed, size_t* image_compressed_size);

/**
 * Returns duration statistics for last encoded image
 * @return 0 if succeeds, otherwise nonzero
 * @note
 * The values are only informative and for debugging only and thus this is
 * not considered as a part of a public API.
 */
GPUJPEG_API int
gpujpeg_encoder_get_stats(struct gpujpeg_encoder *encoder, struct gpujpeg_duration_stats *stats);

enum gpujpeg_header_type {
        GPUJPEG_HEADER_DEFAULT = 0, ///< for 1 or 3 channel @ref GPUJPEG_YCBCR_JPEG @ref GPUJPEG_HEADER_JFIF, for @ref
                                    ///< GPUJPEG_RGB @ref GPUJPEG_HEADER_ADOBE, @ref GPUJPEG_HEADER_SPIFF otherwise
        GPUJPEG_HEADER_JFIF    = 1<<0,
        GPUJPEG_HEADER_SPIFF   = 1<<1,
        GPUJPEG_HEADER_ADOBE   = 1<<2, ///< Adobe APP8 header
};

/**
 * Forces JPEG header to be emitted.
 *
 * Header type should be capable of describing the resulting JPEG, eg. JFIF only for BT.601
 * full-scale YCbCr images. If not, resulting JPEG image may be incompatible with decoders.
 */
GPUJPEG_API void
gpujpeg_encoder_set_jpeg_header(struct gpujpeg_encoder *encoder, enum gpujpeg_header_type header_type);

/**
 * Suggests optimal restart interval to be used for given param_image balancing both image
 * size and performance.
 * @param subsampling 444 422 or 420
 */
GPUJPEG_API int
gpujpeg_encoder_suggest_restart_interval(const struct gpujpeg_image_parameters* param_image,
                                         gpujpeg_sampling_factor_t subsampling, bool interleaved, int verbose);

/**
 * Destory JPEG encoder
 *
 * @param encoder  Encoder structure
 * @return 0 if succeeds, otherwise nonzero
 */
GPUJPEG_API int
gpujpeg_encoder_destroy(struct gpujpeg_encoder* encoder);

#ifdef __cplusplus
}
#endif

#endif // GPUJPEG_ENCODER_H
