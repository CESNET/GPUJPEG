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

#include "libgpujpeg/gpujpeg_common.h"

#ifdef __cplusplus
extern "C" {
#endif // defined __cplusplus

/// @brief malloc-compatible allocator to allocate data
typedef void *(*allocator_t)(size_t);
/**
 * @param[in]  filename   image filename
 * @param[out] image_size loaded image size
 * @param[out] image_data loaded image data allocated with alloc
 * @param[in]  alloc      allocator to allocate image_data
 * @returns               0 if succeeded; non-zero otherwise
 */
typedef int (*image_load_delegate_t)(const char *filename, int *image_size, void **image_data, allocator_t alloc);
/**
 * Reads image metadata without actually reading the file contents.
 * @param[in]  filename    image filename
 * @param[out] params      probed image parameters
 * @param[in]  file_exists boolean value to indicated if the file exists. The aim of this option is to allow
 *                         obtaining at least a piece of information, namely channel count from pnm-family ext.
 * @returns                0 if succeeded; non-zero otherwise
 */
typedef int (*image_probe_delegate_t)(const char *filename, struct gpujpeg_image_parameters *params, int file_exists);
/**
 * Writes image data with appropriate filetype header.
 * @param[in]  filename    output image filename
 * @param[out] params      image parameters to use
 * @param[in]  data        image contents to be written, length will be deduced from parameters thus not given explicitly
 * @returns                0 if succeeded; non-zero otherwise
 */
typedef int (*image_save_delegate_t)(const char *filename, const struct gpujpeg_image_parameters *params, const char *data);

image_load_delegate_t gpujpeg_get_image_load_delegate(enum gpujpeg_image_file_format format);
image_probe_delegate_t gpujpeg_get_image_probe_delegate(enum gpujpeg_image_file_format format);
image_save_delegate_t gpujpeg_get_image_save_delegate(enum gpujpeg_image_file_format format);

#ifdef __cplusplus
} // extern "C"
#endif // defined __cplusplus

#endif // ! defined GPUJPEG_IMAGE_DELEGATE_H_0EE4DE91_F6E7_4C02_A4A6_0FFF8C402AE8

