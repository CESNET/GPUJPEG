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

#ifndef GPUJPEG_READER_H
#define GPUJPEG_READER_H

#include <libgpujpeg/gpujpeg_common.h>

#ifdef __cplusplus
extern "C" {
#endif

/** JPEG decoder structure predeclaration */
struct gpujpeg_decoder;

/** JPEG reader scan structure */
struct gpujpeg_reader_scan
{
    // Global segment index
    int segment_index;
    // Segment count in scan
    int segment_count;
};

/** JPEG reader structure */
struct gpujpeg_reader
{
    // Parameters
    struct gpujpeg_parameters param;

    // Parameters for image data
    struct gpujpeg_image_parameters param_image;

    // Loaded component count
    int comp_count;

    // Loaded scans
    struct gpujpeg_reader_scan scan[GPUJPEG_MAX_COMPONENT_COUNT];

    // Loaded scans count
    int scan_count;

    // Total segment count
    int segment_count;

    // Total readed size
    int data_compressed_size;

    // Segment info (every buffer is placed inside another header)
    uint8_t* segment_info[GPUJPEG_MAX_SEGMENT_INFO_HEADER_COUNT];
    // Segment info buffers count (equals number of segment info headers)
    int segment_info_count;
    // Segment info total buffers size
    int segment_info_size;
};

/**
 * Create JPEG reader
 *
 * @return reader structure if succeeds, otherwise NULL
 */
struct gpujpeg_reader*
gpujpeg_reader_create();

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
gpujpeg_reader_read_image(struct gpujpeg_decoder* decoder, uint8_t* image, int image_size);

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
 * @param param_image parameters obtained from image
 * @return 0 if succeeds, otherwise nonzero
 */
GPUJPEG_API int
gpujpeg_reader_get_image_info(uint8_t *image, int image_size, struct gpujpeg_image_parameters *param_image, int *segment_count);

#ifdef __cplusplus
}
#endif

#endif // GPUJPEG_READER_H
