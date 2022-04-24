/*
 * Copyright (c) 2020-2022, CESNET z.s.p.o
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

#include <fstream>
#include <iostream>
#include <string.h>

#include "image_delegate.h"
#include "image_delegate_pnm.hpp"
#include "pnm.hpp"

#if defined(_MSC_VER)
  #define strcasecmp _stricmp
#else
  #include <strings.h>
#endif

using std::cerr;

int pnm_load_delegate(const char *filename, int *image_size, void **image_data, allocator_t alloc) {
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
    if (!info.valid()) {
        return 1;
    }
    *image_size = info.width() * info.height() * info.channel() * info.depth() / 8;
    return 0;
}

int pnm_probe_delegate(const char *filename, struct gpujpeg_image_parameters *param_image, int file_exists) {
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
            param_image->color_space = GPUJPEG_RGB;
            param_image->pixel_format = GPUJPEG_444_U8_P012;
        } else if (strcasecmp(suffix, "pnm") == 0) {
            param_image->pixel_format = (gpujpeg_pixel_format) GPUJPEG_PIXFMT_NO_ALPHA;
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
        fprintf(stderr, "Wrong pixel format %s for PNM! Only packed formats with 1 or 3 channels are supported.\n", gpujpeg_pixel_format_get_name(param_image->pixel_format));
        return -1;
    }

    ofs << PNM::save( (uint8_t *) data, param_image->width, param_image->height, type );

    return 0;
}

/* vi: set expandtab sw=4 : */
