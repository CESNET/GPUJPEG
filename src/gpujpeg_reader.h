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

#ifndef GPUJPEG_READER_H
#define GPUJPEG_READER_H

#include "../libgpujpeg/gpujpeg_common.h"

#ifdef __cplusplus
extern "C" {
#endif

/** JPEG decoder structure predeclaration */
struct gpujpeg_decoder;
struct gpujpeg_reader;

/**
 * Create JPEG reader
 *
 * @return reader structure if succeeds, otherwise NULL
 */
struct gpujpeg_reader*
gpujpeg_reader_create(void);

/**
 * Destroy JPEG reader
 *
 * @param reader  Reader structure
 * @return 0 if succeeds, otherwise nonzero
 */
int
gpujpeg_reader_destroy(struct gpujpeg_reader* reader);

/**
 * Read JPEG image from data buffer
 *
 * @param image  Image data
 * @param image_size  Image data size
 * @return 0 if succeeds, otherwise nonzero
 */
int
gpujpeg_reader_read_image(struct gpujpeg_decoder* decoder, uint8_t* image, size_t image_size);

/**
 * Read image info from JPEG file
 *
 * Values read (if present) are: width, height, comp_count, color_space.
 * If a value of a parameter cannot be read/deduced, corresponding member
 * of gpujpeg_image_parameters is not modified. Thus the caller may initialize
 * the members with some distictive values to detect this.
 *
 * @param image  Image data
 * @param image_size  Image data size
 * @param[out]    param_image   parameters obtained from image, must not be NULL
 * @param[in,out] param         parameters obtained from image (verbose parameter is used as an input param), non-NULL
 * @param[out]    segment_count number of segments (may be NULL if parameter segment_count is not needed)
 * @return 0 if succeeds, otherwise nonzero
 *
 * @todo refactorize common code with gpujpeg_reader_read_image()
 */
int
gpujpeg_reader_get_image_info(uint8_t *image, size_t image_size, struct gpujpeg_image_parameters *param_image, struct gpujpeg_parameters *param, int *segment_count);

#ifdef __cplusplus
}
#endif

#endif // GPUJPEG_READER_H
