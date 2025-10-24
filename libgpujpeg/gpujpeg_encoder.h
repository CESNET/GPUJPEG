/**
 * @file
 * Copyright (c) 2011-2025, CESNET z.s.p.o
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
 * @sa gpujpeg_encoder_input_image
 */
GPUJPEG_API void
gpujpeg_encoder_input_set_image(struct gpujpeg_encoder_input* input, uint8_t* image);

/**
 * Set encoder input to GPU image data
 *
 * @param encoder_input  Encoder input structure
 * @param image GPU image data
 * @return void
 * @sa gpujpeg_encoder_input_gpu_image
 */
GPUJPEG_API void
gpujpeg_encoder_input_set_gpu_image(struct gpujpeg_encoder_input* input, uint8_t* image);

/**
 * Set encoder input to OpenGL texture
 *
 * @param encoder_input  Encoder input structure
 * @param texture_id  OpenGL texture id
 * @return void
 * @sa gpujpeg_encoder_input_set_texture
 */
GPUJPEG_API void
gpujpeg_encoder_input_set_texture(struct gpujpeg_encoder_input* input, struct gpujpeg_opengl_texture* texture);

/// alternative to @ref gpujpeg_encoder_input_set_image returning the struct as a return value
GPUJPEG_API struct gpujpeg_encoder_input
gpujpeg_encoder_input_image(uint8_t* image);
/// alternative to @ref gpujpeg_encoder_input_set_gpu_image returning the struct as a return value
GPUJPEG_API struct gpujpeg_encoder_input
gpujpeg_encoder_input_gpu_image(uint8_t* image);
/// alternative to @ref gpujpeg_encoder_input_set_texture returning the struct as a return value
GPUJPEG_API struct gpujpeg_encoder_input
gpujpeg_encoder_input_texture(struct gpujpeg_opengl_texture* texture);

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
 * @param[out] image_compressed  Pointer to variable where compressed image data buffer will be placed.
 *                               Returned host buffer is owned by encoder and must not be freed by the caller.
 *                               Buffer is valid until next gpujpeg_encoder_encode() call.
 * @param[out] image_compressed_size  Pointer to variable where compressed image size will be placed.
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
 * @deprecated
 * The encoder now prints the statistics to stdout if gpujpeg_parameters.perf_stats is set.
 * May be removed in future versions - please report if using this function.
 */
GPUJPEG_DEPRECATED GPUJPEG_API int
gpujpeg_encoder_get_stats(struct gpujpeg_encoder *encoder, struct gpujpeg_duration_stats *stats);

/**
 * Forces JPEG header to be emitted.
 *
 * Header type should be capable of describing the resulting JPEG, eg. JFIF only for BT.601
 * full-scale YCbCr images. If not, resulting JPEG image may be incompatible with decoders.
 *
 * @deprecated use gpujpeg_encoder_set_option() with @ref GPUJPEG_ENC_OPT_HDR
 */
GPUJPEG_DEPRECATED GPUJPEG_API void
gpujpeg_encoder_set_jpeg_header(struct gpujpeg_encoder *encoder, enum gpujpeg_header_type header_type);

/**
 * Suggests optimal restart interval to be used for given param_image balancing both image
 * size and performance.
 * @param subsampling 444 422 or 420
 */
GPUJPEG_API int
gpujpeg_encoder_suggest_restart_interval(const struct gpujpeg_image_parameters* param_image,
                                         gpujpeg_sampling_factor_t subsampling, bool interleaved, int verbose);

#define GPUJPEG_ENCODER_OPT_OUT_PINNED  "enc_out_pinned" ///< deprecated - use GPUJPEG_ENC_OPT_OUT
/// location of buffer returned from gpujpeg_encoder_encode()
#define GPUJPEG_ENC_OPT_OUT          "enc_opt_out"
#define GPUJPEG_ENC_OUT_VAL_PAGEABLE "enc_out_val_pageable"  ///< default
#define GPUJPEG_ENC_OUT_VAL_PINNED   "enc_out_val_pinned"

#define GPUJPEG_ENC_OPT_HDR          "enc_hdr"
/// @defgroup enc_hdr_types
/// @{
#define GPUJPEG_ENC_HDR_VAL_JFIF     "JFIF"
#define GPUJPEG_ENC_HDR_VAL_EXIF     "Exif"
#define GPUJPEG_ENC_HDR_VAL_ADOBE    "Adobe"
#define GPUJPEG_ENC_HDR_VAL_SPIFF    "SPIFF"
/// @}

/// input image is vertically flipped (bottom-up): values @ref GPUJPEG_VAL_TRUE or @ref GPUJPEG_VAL_FALSE
#define GPUJPEG_ENC_OPT_FLIPPED_BOOL "enc_opt_flipped"
/// custom exif tag in format <key>:TYPE=<value>
#define GPUJPEG_ENC_OPT_EXIF_TAG "enc_exif_tag"
/// set image orientation - syntax "<name>=<deg>[-]" or "help"; only if header supports (Exif, SPIFF)
#define GPUJPEG_ENC_OPT_METADATA "enc_metadata"

/**
 * remap channel order
 *
 * Format is "XYZ" or "XYZW" where the letters stand for input channel indices (0-indexed) mapped to output position;
 * so eg. for remapping from ARGB to RGBA the option value will be "1230".
 *
 * The number of image channes must equal the length of the string. Set to "" (empty string) to revert this setting.
 * Letters 'Z' or 'F' can be used instead of indices to fill given output channel with zeros or all-ones.
 */
#define GPUJPEG_ENC_OPT_CHANNEL_REMAP "enc_opt_channel_remap"

/**
 * sets encoder option
 * @retval GPUJPEG_NOERR  option was sucessfully set
 * @retval GPUJPEG_ERROR  invalid argument passed
 */
GPUJPEG_API int
gpujpeg_encoder_set_option(struct gpujpeg_encoder* encoder, const char* opt, const char* val);
GPUJPEG_API void
gpujpeg_encoder_print_options();

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
