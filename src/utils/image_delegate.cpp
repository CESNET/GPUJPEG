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

#include <iostream>
#include <string.h>

#include "image_delegate.h"
#include "pam.hpp"
#include "pnm.hpp"

#if defined(_MSC_VER)
  #define strcasecmp _stricmp
#else
  #include <strings.h>
#endif

using std::cerr;

static int pam_load_delegate(const char *filename, int *image_size, void **image_data, allocator_t alloc) {
    unsigned int w, h;
    int d;
    bool ret = pam_read(filename, &w, &h, &d, (unsigned char **) image_data, alloc);
    if (ret) {
        *image_size = w * h * d;
    }
    return ret ? 0 : 1;
}

static int pam_probe_delegate(const char *filename, struct gpujpeg_image_parameters *param_image, int file_exists) {
    if (!file_exists) {
        return 0;
    }
    int depth;
    unsigned int w, h;
    if (pam_read(filename, &w, &h, &depth, nullptr, nullptr)) {
        param_image->width = w;
        param_image->height = h;
        switch (depth) {
        case 4:
            param_image->pixel_format = GPUJPEG_444_U8_P012A;
            break;
        case 3:
            param_image->pixel_format = GPUJPEG_444_U8_P012;
            break;
        case 1:
            param_image->color_space = GPUJPEG_YCBCR_BT601_256LVLS;
            param_image->pixel_format = GPUJPEG_U8;
            break;
        default:
            fprintf(stderr, "Wrong pam component count %d!\n", depth);
            return GPUJPEG_ERROR;
        }
    } else {
        param_image->width = 0;
        param_image->height = 0;
        param_image->pixel_format = GPUJPEG_PIXFMT_NONE;
    }
    return 0;
}

int pam_save_delegate(const char *filename, const struct gpujpeg_image_parameters *param_image, const char *data)
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
    case GPUJPEG_444_U8_P012A:
    case GPUJPEG_444_U8_P012Z:
        depth = 4;
        break;
    default:
        fprintf(stderr, "Wrong pixel format for PAM! Only planar formats are supported.\n");
        return -1;
    }

    bool ret = pam_write(filename, param_image->width, param_image->height, depth, (const unsigned char *) data);
    return ret ? 0 : -1;
}


static int pnm_load_delegate(const char *filename, int *image_size, void **image_data, allocator_t alloc) {
    std::ifstream ifs( filename, std::ios_base::binary | std::ios::ate );
    if (!ifs.is_open()) {
        cerr << "[GPUJPEG] [Error] Failed open " << filename << " for reading!\n";
        return 1;
    }
    ifs.exceptions(std::ifstream::failbit | std::ifstream::badbit);

    PNM::Info info;

    *image_data = alloc(ifs.tellg());
    ifs.seekg( 0, std::ios::beg );

    ifs >> PNM::load( (uint8_t*) *image_data, info );
    return info.valid() ? 0 : 1;
}

static int pnm_probe_delegate(const char *filename, struct gpujpeg_image_parameters *param_image, int file_exists) {
    if (!file_exists) {
        // deduce from file extension
        const char *suffix = strrchr(filename, '.');
        if (!suffix) {
            return 0;
        }
        suffix++;
        if (strcasecmp(suffix, "pgm") == 0) {
            param_image->color_space = GPUJPEG_YCBCR_BT601_256LVLS;
            param_image->pixel_format = GPUJPEG_U8;
        } else if (strcasecmp(suffix, "ppm") == 0) {
            param_image->pixel_format = GPUJPEG_444_U8_P012;
        }
        return 0;
    }
    std::ifstream ifs( filename, std::ios_base::binary );
    if (!ifs.is_open()) {
        cerr << "[GPUJPEG] [Error] Failed open " << filename << " for reading!\n";
        return 1;
    }
    ifs.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    PNM::Info info;

    ifs >> PNM::probe( info );
    if (info.valid()) {
        param_image->width = info.width();
        param_image->height = info.height();
        switch (info.channel()) {
        case 3:
            param_image->pixel_format = GPUJPEG_444_U8_P012;
            break;
        case 1:
            param_image->color_space = GPUJPEG_YCBCR_BT601_256LVLS;
            param_image->pixel_format = GPUJPEG_U8;
            break;
        default:
            fprintf(stderr, "Wrong PNM component count %zd!\n", info.channel());
            return GPUJPEG_ERROR;
        }
    }
    return 0;
}

int pnm_save_delegate(const char *filename, const struct gpujpeg_image_parameters *param_image, const char *data)
{
    if (param_image->color_space != GPUJPEG_RGB) {
        fprintf(stderr, "Wrong color space %s for PNM!\n", gpujpeg_color_space_get_name(param_image->color_space));
        return -1;
    }

    std::ofstream ofs( filename );
    PNM::type type;
    switch (param_image->pixel_format) {
    case GPUJPEG_U8:
        type = PNM::P5;
        break;
    case GPUJPEG_444_U8_P012:
        type = PNM::P6;
        break;
    default:
        fprintf(stderr, "Wrong pixel format for PNM! Only planar RGB and grayscale is supported!\n");
        return -1;
    }

    ofs << PNM::save( (uint8_t *) data, param_image->width, param_image->height, type );

    return 0;
}

image_load_delegate_t gpujpeg_get_image_load_delegate(enum gpujpeg_image_file_format format) {
    switch (format) {
    case GPUJPEG_IMAGE_FILE_PAM:
        return pam_load_delegate;
    case GPUJPEG_IMAGE_FILE_PNM:
        return pnm_load_delegate;
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
    default:
        return NULL;
    }
}

/* vi: set expandtab sw=4 : */
