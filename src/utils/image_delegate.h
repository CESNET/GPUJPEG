/*
 * Copyright (c) 2020, CESNET z.s.p.o
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


#ifndef GPUJPEG_IMAGE_DELEGATE_H_0EE4DE91_F6E7_4C02_A4A6_0FFF8C402AE8
#define GPUJPEG_IMAGE_DELEGATE_H_0EE4DE91_F6E7_4C02_A4A6_0FFF8C402AE8

#include <libgpujpeg/gpujpeg_common.h>

#ifdef __cplusplus
extern "C" {
#endif // defined __cplusplus

typedef void *(*allocator_t)(size_t);
typedef int (*image_load_delegate_t)(const char *filename, int *image_size, void **image_data, allocator_t alloc);
typedef int (*image_probe_delegate_t)(const char *filename, int *width, int *height, enum gpujpeg_color_space *, enum gpujpeg_pixel_format *, int file_exists);
typedef int (*image_save_delegate_t)(const char *filename, int width, int height, enum gpujpeg_color_space, enum gpujpeg_pixel_format, const char *data);

image_load_delegate_t gpujpeg_get_image_load_delegate(enum gpujpeg_image_file_format format);
image_probe_delegate_t gpujpeg_get_image_probe_delegate(enum gpujpeg_image_file_format format);
image_save_delegate_t gpujpeg_get_image_save_delegate(enum gpujpeg_image_file_format format);

#ifdef __cplusplus
} // extern "C"
#endif // defined __cplusplus

#endif // ! defined GPUJPEG_IMAGE_DELEGATE_H_0EE4DE91_F6E7_4C02_A4A6_0FFF8C402AE8

