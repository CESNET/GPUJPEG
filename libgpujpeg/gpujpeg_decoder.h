/**
 * @file
 * Copyright (c) 2011-2024, CESNET
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
#include "gpujpeg_type.h"

#ifdef __cplusplus
extern "C" {
#endif

struct gpujpeg_decoder;

/**
 * Decoder output type
 */
enum gpujpeg_decoder_output_type {
    /// Decoder will use it's internal output buffer
    GPUJPEG_DECODER_OUTPUT_INTERNAL_BUFFER,
    /// Decoder will use custom output buffer
    GPUJPEG_DECODER_OUTPUT_CUSTOM_BUFFER,
    /// Decoder will use OpenGL Texture PBO Resource as output buffer
    GPUJPEG_DECODER_OUTPUT_OPENGL_TEXTURE,
    /// Decoder will use internal CUDA buffer as output buffer
    GPUJPEG_DECODER_OUTPUT_CUDA_BUFFER,
    /// Decoder will use custom CUDA buffer as output buffer
    GPUJPEG_DECODER_OUTPUT_CUSTOM_CUDA_BUFFER,
};

/**
 * Decoder output structure
 */
struct gpujpeg_decoder_output
{
    /// Output type
    enum gpujpeg_decoder_output_type type;

    /// Decompressed data
    uint8_t* data;

    /// Decompressed data size
    size_t data_size;

    /// Decoded image parameters
    struct gpujpeg_image_parameters param_image;

    /// OpenGL texture
    struct gpujpeg_opengl_texture* texture;
};

/**
 * Set default parameters to decoder output structure
 *
 * @param output  Decoder output structure
 * @return void
 */
GPUJPEG_API void
gpujpeg_decoder_output_set_default(struct gpujpeg_decoder_output* output);

/**
 * Setup decoder output to custom buffer
 *
 * @param output        Decoder output structure
 * @param custom_buffer Custom buffer
 * @return void
 */
GPUJPEG_API void
gpujpeg_decoder_output_set_custom(struct gpujpeg_decoder_output* output, uint8_t* custom_buffer);

/**
 * Set decoder output to OpenGL texture
 *
 * @param output  Decoder output structure
 * @return void
 */
GPUJPEG_API void
gpujpeg_decoder_output_set_texture(struct gpujpeg_decoder_output* output, struct gpujpeg_opengl_texture* texture);

/**
 * Sets output to CUDA buffer
 *
 * @param output  Decoder output structure
 */
GPUJPEG_API void
gpujpeg_decoder_output_set_cuda_buffer(struct gpujpeg_decoder_output* output);

/**
 * Setup decoder output to custom CUDA buffer
 *
 * @param output          Decoder output structure
 * @param d_custom_buffer Custom buffer in CUDA device memory
 * @return void
 */
GPUJPEG_API void
gpujpeg_decoder_output_set_custom_cuda(struct gpujpeg_decoder_output* output, uint8_t* d_custom_buffer);

/**
 * Create JPEG decoder
 *
 * @param stream CUDA stream to be used, may be cudaStreamDefault (0x00)
 * @return decoder structure if succeeds, otherwise NULL
 */
GPUJPEG_API struct gpujpeg_decoder*
gpujpeg_decoder_create(cudaStream_t stream);

/**
 * Init JPEG decoder for specific image properties
 *
 * Following properties are relevant:
 * - image dimensions, commonent count
 * - output pixel format that will be requested
 * - interleaving, restart interval, color_space_internal (usually GPUJPEG_YCBCR_BT601_256LVLS)
 * - correct subsampling setting
 *
 * @note
 * Doesn't need to be called from user code, buffers will be initialized automatically according to
 * image properties during decompression.
 *
 * @param decoder  Decoder structure
 * @param[in] param        Parameters for coder, pointed structure is copied
 * @param[in] param_image  Parameters for image data, pointed structure is copied
 * @return 0 if succeeds, otherwise nonzero
 */
GPUJPEG_API int
gpujpeg_decoder_init(struct gpujpeg_decoder* decoder, const struct gpujpeg_parameters* param, const struct gpujpeg_image_parameters* param_image);

/**
 * Decompress image by decoder
 *
 * @param decoder  Decoder structure
 * @param image  Source image data
 * @param image_size  Source image data size
 * @param image_decompressed  Pointer to variable where decompressed image data buffer will be placed
 * @param image_decompressed_size  Pointer to variable where decompressed image size will be placed
 * @return @ref Errors
 */
GPUJPEG_API int
gpujpeg_decoder_decode(struct gpujpeg_decoder* decoder, uint8_t* image, size_t image_size, struct gpujpeg_decoder_output* output);

/**
 * Returns duration statistics for last decoded image
 * @return 0 if succeeds, otherwise nonzero
 * @note
 * The values are only informative and for debugging only and thus this is
 * not considered as a part of a public API.
 */
GPUJPEG_API int
gpujpeg_decoder_get_stats(struct gpujpeg_decoder *decoder, struct gpujpeg_duration_stats *stats);

/**
 * Destory JPEG decoder
 *
 * @param decoder  Decoder structure
 * @return 0 if succeeds, otherwise nonzero
 */
GPUJPEG_API int
gpujpeg_decoder_destroy(struct gpujpeg_decoder* decoder);

/**
 * @defgroup decoder_pixfmt_placeholders
 * @{
 * following format placeholders are special values that may be passed
 * to the decoeer in order to detect the format with optional constraints.
 * Defined outside the enums to avoid -Wswitch warns.
 */
/// decoder default pixfmt - usually @ref GPUJPEG_444_U8_P012;
/// @ref GPUJPEG_U8 for grayscale and @ref GPUJPEG_444_U8_P0123 if alpha present
///
#define GPUJPEG_PIXFMT_AUTODETECT ((enum gpujpeg_pixel_format)(GPUJPEG_PIXFMT_NONE - 1))
/// as @ref GPUJPEG_PIXFMT_AUTODETECT, but alpha stripped if present
#define GPUJPEG_PIXFMT_NO_ALPHA ((enum gpujpeg_pixel_format)(GPUJPEG_PIXFMT_AUTODETECT - 1))
/// pixel format that may be stored in a PAM or Y4M file - a planar pixel
/// format that is either 444, 422 or 420 for YUV, P012(3) otherwise
#define GPUJPEG_PIXFMT_STD ((enum gpujpeg_pixel_format)(GPUJPEG_PIXFMT_NO_ALPHA - 1))
/// @}
/// Decode RGB for 3 or 4 channels, GPUJPEG_YCBCR for grayscale.
/// decoder only, valid only if passed to gpujpeg_decoder_set_output_format()
#define GPUJPEG_CS_DEFAULT ((enum gpujpeg_color_space)(GPUJPEG_NONE - 1))

/**
 * Sets output format
 *
 * If not called, @ref GPUJPEG_CS_DEFAULT and @ref GPUJPEG_PIXFMT_AUTODETECT
 * are used.
 *
 * @param decoder         Decoder structure
 * @param color_space     Requested output color space,
 *                        use @ref GPUJPEG_NONE to keep JPEG internal color space;
 *                        special value @ref GPUJPEG_CS_DEFAULT to decode RGB
 *                        (or luma for grayscale)
 * @param sampling_factor Requestd color sampling factor; special values
 *                        @ref decoder_pixfmt_placeholders can be used
 */
GPUJPEG_API void
gpujpeg_decoder_set_output_format(struct gpujpeg_decoder* decoder,
                enum gpujpeg_color_space color_space,
                enum gpujpeg_pixel_format pixel_format);

/**
 * @copydoc gpujpeg_reader_get_image_info
 */
GPUJPEG_API int
gpujpeg_decoder_get_image_info(uint8_t *image, size_t image_size, struct gpujpeg_image_parameters *param_image, struct gpujpeg_parameters *param, int *segment_count);

#ifdef __cplusplus
}
#endif

#endif // GPUJPEG_DECODER_H
