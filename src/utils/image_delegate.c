/*
 * Copyright (c) 2020-2024, CESNET z.s.p.o
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
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#include "image_delegate.h"
#include "pam.h"
#include "y4m.h"

enum {
  DEPTH_8B = 8,
  MAXVAL_8B = 255,
};

static int pam_load_delegate(const char *filename, size_t *image_size, void **image_data, allocator_t alloc) {
    struct pam_metadata info;
    bool ret = pam_read(filename, &info, (unsigned char **) image_data, alloc);
    if (ret) {
        assert(info.maxval == 255);
        *image_size = (size_t) info.width * info.height * info.depth;
    }
    return ret ? 0 : 1;
}

static int
pampnm_probe_delegate(const char* filename, struct gpujpeg_image_parameters* param_image)
{
    struct pam_metadata info;
    if (!pam_read(filename, &info, NULL, NULL)) {
        return GPUJPEG_ERROR;
    }
    if (info.maxval != MAXVAL_8B) {
        fprintf(
            stderr,
            "[GPUJPEG] [Error] PAM image %s reports %d levels but only 255 are "
            "currently supported!\n",
            filename, info.maxval);
        return GPUJPEG_ERROR;
    }
    param_image->width = info.width;
    param_image->height = info.height;
    switch (info.depth) {
    case 4:
        param_image->pixel_format = GPUJPEG_4444_U8_P0123;
        break;
    case 3:
        param_image->pixel_format = GPUJPEG_444_U8_P012;
        break;
    case 1:
        param_image->color_space = GPUJPEG_YCBCR_BT601_256LVLS;
        param_image->pixel_format = GPUJPEG_U8;
        break;
    default:
        fprintf(stderr, "Wrong pam component count %d!\n", info.depth);
        return GPUJPEG_ERROR;
    }
    return 0;
}

static int
pam_probe_delegate(const char* filename, struct gpujpeg_image_parameters* param_image, int file_exists)
{
    if (!file_exists) {
        param_image->pixel_format = GPUJPEG_PIXFMT_AUTODETECT;
        return true;
    }
    return pampnm_probe_delegate(filename, param_image);
}

static int
pnm_probe_delegate(const char* filename, struct gpujpeg_image_parameters* param_image, int file_exists)
{
    if (!file_exists) {
        param_image->pixel_format = GPUJPEG_PIXFMT_NO_ALPHA;
        return true;
    }
    return pampnm_probe_delegate(filename, param_image);
}

int pampnm_save_delegate(const char *filename, const struct gpujpeg_image_parameters *param_image, const char *data, bool pnm)
{
    if (param_image->pixel_format != GPUJPEG_U8 && param_image->color_space != GPUJPEG_RGB) {
        fprintf(stderr, "Wrong color space %s for PAM!\n", gpujpeg_color_space_get_name(param_image->color_space));
        return -1;
    }
    int depth;
    switch (param_image->pixel_format) {
    case GPUJPEG_U8:
        depth = 1;
        break;
    case GPUJPEG_444_U8_P012:
        depth = 3;
        break;
    case GPUJPEG_4444_U8_P0123:
        depth = 4;
        break;
    default:
        fprintf(stderr,
                "Wrong pixel format %s for PAM/PNM! Only packed formats "
                "without subsampling are supported.\n",
                gpujpeg_pixel_format_get_name(param_image->pixel_format));
        return -1;
    }

    bool ret = pam_write(filename, param_image->width, param_image->height, depth, 255, (const unsigned char *) data, pnm);
    return ret ? 0 : -1;
}

int pam_save_delegate(const char *filename, const struct gpujpeg_image_parameters *param_image, const char *data)
{
    return pampnm_save_delegate(filename, param_image, data, false);
}

int pnm_save_delegate(const char *filename, const struct gpujpeg_image_parameters *param_image, const char *data)
{
    return pampnm_save_delegate(filename, param_image, data, true);
}

static int y4m_probe_delegate(const char *filename, struct gpujpeg_image_parameters *param_image, int file_exists) {
    if (!file_exists) {
        param_image->color_space = GPUJPEG_YCBCR_BT601_256LVLS;
        param_image->pixel_format = (enum gpujpeg_pixel_format) GPUJPEG_PIXFMT_PLANAR_STD;
        return 0;
    }

    struct y4m_metadata info = { 0 };
    if (!y4m_read(filename, &info, NULL, NULL)) {
        return -1;
    }
    param_image->width = info.width;
    param_image->height = info.height;
    if (info.bitdepth != DEPTH_8B) {
        fprintf(stderr,
                "[GPUJPEG] [Error] Currently only 8-bit Y4M pictures are "
                "supported but %s has %d bits!\n",
                filename, info.bitdepth);
        return GPUJPEG_ERROR;
    }
    switch (info.subsampling) {
        case Y4M_SUBS_MONO:
            param_image->pixel_format = GPUJPEG_U8;
            break;
        case Y4M_SUBS_420:
            param_image->pixel_format = GPUJPEG_420_U8_P0P1P2;
            break;
        case Y4M_SUBS_422:
            param_image->pixel_format = GPUJPEG_422_U8_P0P1P2;
            break;
        case Y4M_SUBS_444:
            param_image->pixel_format = GPUJPEG_444_U8_P0P1P2;
            break;
        case Y4M_SUBS_YUVA:
            fprintf(stderr, "Planar YCbCr with alpha is not currently supported!\n");
            return -1;
        default:
            fprintf(stderr, "[GPUJPEG] [Error] Unknown subsamplig in Y4M!\n");
            return GPUJPEG_ERROR;
    }
    param_image->color_space = info.limited ? GPUJPEG_YCBCR_BT601 : GPUJPEG_YCBCR_BT601_256LVLS;
    return 0;
}

static int y4m_load_delegate(const char *filename, size_t *image_size, void **image_data, allocator_t alloc) {
    struct y4m_metadata info = { 0 };
    if ((*image_size = y4m_read(filename, &info, (unsigned char **) image_data, alloc)) == 0) {
        return 1;
    }
    return 0;
}

int y4m_save_delegate(const char *filename, const struct gpujpeg_image_parameters *param_image, const char *data)
{
    if (param_image->color_space == GPUJPEG_RGB) {
        fprintf(stderr, "Y4M cannot use RGB colorspace!\n");
        return -1;
    }
    int subsampling = 0;
    switch (param_image->pixel_format) {
    case GPUJPEG_U8:
        subsampling = Y4M_SUBS_MONO;
        break;
    case GPUJPEG_420_U8_P0P1P2:
        subsampling = Y4M_SUBS_420;
        break;
    case GPUJPEG_422_U8_P0P1P2:
        subsampling = Y4M_SUBS_422;
        break;
    case GPUJPEG_444_U8_P0P1P2:
        subsampling = Y4M_SUBS_444;
        break;
    default:
        fprintf(stderr, "Wrong pixel format %s for Y4M! Only planar formats are supported.\n", gpujpeg_pixel_format_get_name(param_image->pixel_format));
        return -1;
    }

    _Bool limited = param_image->color_space != GPUJPEG_YCBCR_JPEG;

    struct y4m_metadata info = { .width = param_image->width, .height = param_image->height, .bitdepth = 8, .subsampling = subsampling, .limited = limited };
    return y4m_write(filename, &info, (const unsigned char *) data) ? 0 : -1;
}

image_load_delegate_t gpujpeg_get_image_load_delegate(enum gpujpeg_image_file_format format) {
    switch (format) {
    case GPUJPEG_IMAGE_FILE_PAM:
    case GPUJPEG_IMAGE_FILE_PNM:
        return pam_load_delegate;
    case GPUJPEG_IMAGE_FILE_Y4M:
        return y4m_load_delegate;
    default:
        return NULL;
    }
}

image_probe_delegate_t gpujpeg_get_image_probe_delegate(enum gpujpeg_image_file_format format)
{
    switch (format) {
    case GPUJPEG_IMAGE_FILE_PAM:
        return pam_probe_delegate;
    case GPUJPEG_IMAGE_FILE_PNM:
        return pnm_probe_delegate;
    case GPUJPEG_IMAGE_FILE_Y4M:
        return y4m_probe_delegate;
    default:
        return NULL;
    }
}

image_save_delegate_t gpujpeg_get_image_save_delegate(enum gpujpeg_image_file_format format)
{
    switch (format) {
    case GPUJPEG_IMAGE_FILE_PAM:
        return pam_save_delegate;
    case GPUJPEG_IMAGE_FILE_PNM:
        return pnm_save_delegate;
    case GPUJPEG_IMAGE_FILE_Y4M:
        return y4m_save_delegate;
    default:
        return NULL;
    }
}

/* vi: set expandtab sw=4 : */
