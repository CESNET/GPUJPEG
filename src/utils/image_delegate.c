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
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#define PATH_MAX MAX_PATH
#else
#include <limits.h>
#endif

#ifdef __linux__
#define strtok_s strtok_r
#endif

#include "../gpujpeg_common_internal.h"
#include "../../libgpujpeg/gpujpeg_decoder.h" // ddecoder placeholders
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
        *image_size = (size_t) info.width * info.height * info.ch_count;
    }
    return ret ? 0 : 1;
}

static enum gpujpeg_pixel_format
pampnm_from_filetype(enum gpujpeg_image_file_format format)
{
    switch (format) {
    case GPUJPEG_IMAGE_FILE_PGM:
        return GPUJPEG_U8;
    case GPUJPEG_IMAGE_FILE_PPM:
        return GPUJPEG_444_U8_P012;
    case GPUJPEG_IMAGE_FILE_PNM:
        return GPUJPEG_PIXFMT_NO_ALPHA;
    case GPUJPEG_IMAGE_FILE_PAM:
        return GPUJPEG_PIXFMT_AUTODETECT;
    default:
        GPUJPEG_ASSERT(0 && "Unsupported file format passed to PAM/PNM handler!");
    }
}

static int
pampnm_probe_delegate(const char* filename, enum gpujpeg_image_file_format format,
                      struct gpujpeg_image_parameters* param_image, bool file_exists)
{
    if ( !file_exists ) {
        param_image->pixel_format = pampnm_from_filetype(format);
        param_image->color_space = format == GPUJPEG_IMAGE_FILE_PGM ? GPUJPEG_YCBCR_JPEG : GPUJPEG_CS_DEFAULT;
        return true;
    }
    struct pam_metadata info;
    if (!pam_read(filename, &info, NULL, NULL)) {
        return GPUJPEG_ERROR;
    }
    if (info.maxval != MAXVAL_8B) {
        fprintf(stderr,
                "[GPUJPEG] [Error] PAM/PNM image %s reports %d levels but only 255 are "
                "currently supported!\n",
                filename, info.maxval);
        return GPUJPEG_ERROR;
    }
    param_image->width = info.width;
    param_image->height = info.height;
    param_image->color_space = GPUJPEG_RGB;
    switch (info.ch_count) {
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
        fprintf(stderr, "Unsupported PAM/PNM component count %d!\n", info.ch_count);
        return GPUJPEG_ERROR;
    }
    return 0;
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

static int
y4m_probe_delegate(const char* filename, enum gpujpeg_image_file_format format,
                   struct gpujpeg_image_parameters* param_image, bool file_exists)
{
    assert(format == GPUJPEG_IMAGE_FILE_Y4M); (void) format;
    if (!file_exists) {
        param_image->color_space = GPUJPEG_YCBCR_BT601_256LVLS;
        param_image->pixel_format = (enum gpujpeg_pixel_format) GPUJPEG_PIXFMT_STD;
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

static void
tst_usage()
{
    PRINTF("Usage:\n"
           "\t<W>x<H>[.c_<CS>][.p_<PF>][.<pattern>].tst\n");
    PRINTF("\nOptional options are sepearated by a dot, key is after an underscore, order\n"
           "doesn't matter.\n");
    PRINTF("\nOptions:\n"
           "\t- c_<CS> - color space\n"
           "\t- p_<PF> - pixel format\n"
           "\t- blank  - use blank pattern\n"
           "\t- gradient - use gradient pattern (default)\n"
           "\t- noise  - use white noise\n"
           "\t- random[=seed] - same as noise, but use deterministic pattern\n"
            );
    PRINTF("\nExamples:\n"
           "\t- 1920x1080.tst              - use FullHD image\n"
           "\t- 1920x1080.c_ycbcr-jpeg.tst - \" with YCbCr color space\n"
           "\t- 1920x1080.p_u8.tst         - FHD grayscale (u8 pixel format)\n"
           "\t- 1920x1080.noise.tst        - FHD RGB noise\n"
           "\t- 1920x1080.c_ycbcr-jpeg.p_422-u8-p1020.tst - YCbCr 4:2:2\n");
    PRINTF("\n");
}

enum tst_pattern {
    TST_GRADIENT,
    TST_DEFAULT = TST_GRADIENT,
    TST_BLANK,
    TST_NOISE,
    TST_RANDOM,
};

static int
tst_image_parse_filename(const char* filename, struct gpujpeg_image_parameters* param_image, enum tst_pattern* pattern,
                         int* seed)
{
    char fname[PATH_MAX];
    snprintf(fname, sizeof fname, "%s", filename);
    // drop extension
    char* ext_dot = strrchr(fname, '.');
    assert(ext_dot != NULL && strlen(ext_dot + 1) == 3); // 3 char ext (.tst)
    *ext_dot = '\0';

    char* endptr = "";
    param_image->width = (int)strtoul(fname, &endptr, 10);
    if ( *endptr != 'x' ) {
        tst_usage();
        return -1;
    }
    endptr += 1;
    param_image->height = (int)strtoul(endptr, &endptr, 10);
    if (param_image->height == 0) {
        tst_usage();
        return -1;
    }

    // defaults
    param_image->color_space = GPUJPEG_RGB;
    param_image->pixel_format = GPUJPEG_444_U8_P012;
    *pattern  = TST_GRADIENT;

    char *item = NULL;
    char *saveptr = NULL;
    while ((item = strtok_s(endptr, ".", &saveptr)) != 0) {
        const char* value = strchr(item, '_') + 1;
        if ( strstr(item, "c_") == item ) {
            param_image->color_space = gpujpeg_color_space_by_name(value);
            if ( param_image->color_space == GPUJPEG_NONE ) {
                fprintf(stderr, "Unknown color space: %s\n", value);
                return -1;
            }
        }
        else if ( strstr(item, "p_") == item ) {
            param_image->pixel_format = gpujpeg_pixel_format_by_name(value);
            if ( param_image->pixel_format == GPUJPEG_PIXFMT_NONE ) {
                fprintf(stderr, "Unknown pixel format: %s\n", value);
                return -1;
            }
        }
        else if ( strcmp(item, "noise") == 0) {
            *pattern = TST_NOISE;
        }
        else if ( strstr(item, "random") == item ) {
            *pattern = TST_RANDOM;
            if ( strstr(item, "random_") == item ) {
                *seed = atoi(strchr(item, '_') + 1);
            }
        }
        else if ( strcmp(item, "blank") == 0) {
            *pattern = TST_BLANK;
        }
        else if ( strcmp(item, "gradient") == 0) {
            *pattern = TST_GRADIENT;
        }
        else {
            fprintf(stderr, "unknown test image option: %s!\n", item);
            return -1;
        }
        endptr = NULL;
    }

    return 0;
}

static int
tst_image_probe_delegate(const char* filename, enum gpujpeg_image_file_format format,
                   struct gpujpeg_image_parameters* param_image, bool file_exists)
{
    (void)format;
    (void)file_exists;
    enum tst_pattern unused_pattern = {0};
    int unused_seed = 0;

    return tst_image_parse_filename(filename, param_image, &unused_pattern, &unused_seed);
}

/// generates deterministic random pattern
static void
gen_pseudorandom(unsigned char* data, size_t len, int seed)
{
    // Linear Congruential Generator (LCG) parameters
    enum {
        A = 1664525,
        C = 1013904223,
        M = 2147483647,
    };
    uint32_t state = seed; // Change this for a different sequence
    for ( size_t i = 0; i < len; i++ ) {
        // Generate the next state in the LCG sequence
        state = (A * state + C) % M;

        // Extract the byte (0-255) from the state
        unsigned char byte = (unsigned char)(state % 256);

        // Append the byte to the buffer
        data[i] = byte;
    }
}

static int
tst_image_load_delegate(const char* filename, size_t* image_size, void** image_data, allocator_t alloc)
{
    struct gpujpeg_image_parameters param_image;
    enum tst_pattern pattern = {0};
    int random_seed = 12345;
    if ( tst_image_parse_filename(filename, &param_image, &pattern, &random_seed) != 0 ) {
        return -1;
    }
    *image_size = gpujpeg_image_calculate_size(&param_image);
    *image_data = alloc(*image_size);

    // fill some data
    switch (pattern) {
        case TST_GRADIENT: {
            struct gpujpeg_image_parameters param_oneline = param_image;
            param_oneline.height = 1;
            const size_t linesize = gpujpeg_image_calculate_size(&param_oneline);
            for ( int i = 0; i < param_image.height; ++i ) {
                memset((char*)*image_data + i * linesize, i * 255 / param_image.height, linesize);
            }
            break;
        }
        case TST_NOISE: {
            char* data = *image_data;
            for ( unsigned i = 0; i < *image_size; ++i ) {
                data[i] = rand();
            }
            break;
        }
        case TST_RANDOM: {
            gen_pseudorandom((unsigned char*)*image_data, *image_size, random_seed);
            break;
        }
        case TST_BLANK: {
            memset(*image_data, 0, *image_size);
            break;
        }
    }

    return 0;
}

image_load_delegate_t gpujpeg_get_image_load_delegate(enum gpujpeg_image_file_format format) {
    switch (format) {
    case GPUJPEG_IMAGE_FILE_PGM:
    case GPUJPEG_IMAGE_FILE_PPM:
    case GPUJPEG_IMAGE_FILE_PNM:
    case GPUJPEG_IMAGE_FILE_PAM:
        return pam_load_delegate;
    case GPUJPEG_IMAGE_FILE_Y4M:
        return y4m_load_delegate;
    case GPUJPEG_IMAGE_FILE_TST:
        return tst_image_load_delegate;
    default:
        return NULL;
    }
}

image_probe_delegate_t gpujpeg_get_image_probe_delegate(enum gpujpeg_image_file_format format)
{
    switch (format) {
    case GPUJPEG_IMAGE_FILE_PGM:
    case GPUJPEG_IMAGE_FILE_PPM:
    case GPUJPEG_IMAGE_FILE_PNM:
    case GPUJPEG_IMAGE_FILE_PAM:
        return pampnm_probe_delegate;
    case GPUJPEG_IMAGE_FILE_Y4M:
        return y4m_probe_delegate;
    case GPUJPEG_IMAGE_FILE_TST:
        return tst_image_probe_delegate;
    default:
        return NULL;
    }
}

image_save_delegate_t gpujpeg_get_image_save_delegate(enum gpujpeg_image_file_format format)
{
    switch (format) {
    case GPUJPEG_IMAGE_FILE_PAM:
        return pam_save_delegate;
    case GPUJPEG_IMAGE_FILE_PGM:
    case GPUJPEG_IMAGE_FILE_PPM:
    case GPUJPEG_IMAGE_FILE_PNM:
        return pnm_save_delegate;
    case GPUJPEG_IMAGE_FILE_Y4M:
        return y4m_save_delegate;
    default:
        return NULL;
    }
}

/* vi: set expandtab sw=4 : */
