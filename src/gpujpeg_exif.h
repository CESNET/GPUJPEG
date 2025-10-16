/**
 * @file
 * Copyright (c) 2025 CESNET, zájmové sdružení právnických osob
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

#ifndef GPUJPEG_EXIF_H_6755D546_2A90_46DF_9ECA_22F575C3E7E3
#define GPUJPEG_EXIF_H_6755D546_2A90_46DF_9ECA_22F575C3E7E3

#ifndef __cplusplus
#include <stdbool.h>
#include <stdint.h>
#endif // not defined __cplusplus

struct gpujpeg_exif_tags;

struct gpujpeg_encoder;
void
gpujpeg_writer_write_exif(struct gpujpeg_encoder* encoder);

bool
gpujpeg_exif_add_tag(struct gpujpeg_exif_tags** exif_tags, const char *cfg);
void
gpujpeg_exif_tags_destroy(struct gpujpeg_exif_tags* exif_tags);

void
gpujpeg_exif_parse(uint8_t** image, const uint8_t* image_end, int verbose);

#endif // defined GPUJPEG_EXIF_H_6755D546_2A90_46DF_9ECA_22F575C3E7E3
