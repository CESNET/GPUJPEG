/**
 * @file   utils/y4m.h
 * @author Martin Pulec     <pulec@cesnet.cz>
 *
 * This file is part of GPUJPEG.
 */
/*
 * Copyright (c) 2022, CESNET z.s.p.o.
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

#ifndef Y4M_H_DB8BEC68_AA48_4A63_81C3_DC3821F5555B
#define Y4M_H_DB8BEC68_AA48_4A63_81C3_DC3821F5555B

#ifdef __cplusplus
#include <cerrno>
#include <cstdio>
#include <string>
#else
#include <errno.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#endif

#ifdef __GNUC__
#define PAM_ATTRIBUTE_UNUSED __attribute__((unused))
#else
#define PAM_ATTRIBUTE_UNUSED
#endif

static inline bool y4m_write(const char *filename, int width, int height, int subsampling, int depth, bool limited, const unsigned char *data) PAM_ATTRIBUTE_UNUSED;
static inline bool y4m_write(const char *filename, int width, int height, int subsampling, int depth, bool limited, const unsigned char *data) {
        const char *chroma_type = NULL;
        size_t len = 0;
        switch (subsampling) {
                case 400: chroma_type = "mono"; len = (size_t) width * height; break;
                case 420: chroma_type = "420"; len = (size_t) width * height + (size_t) 2 * ((width + 1) / 2) * ((height + 1) / 2); break;
                case 422: chroma_type = "422"; len = (size_t) width * height + (size_t) 2 * ((width + 1) / 2) * height; break;
                case 444: chroma_type = "444"; len = (size_t) width * height * 3; break;
                default:
                          fprintf(stderr, "Wrong subsampling '%d'", subsampling);
                          return false;
        }
        errno = 0;
        FILE *file = fopen(filename, "wb");
        if (!file) {
                fprintf(stderr, "Failed to open %s for writing: %s", filename, strerror(errno));
                return false;
        }
        char depth_suffix[20] = "";
        if (depth > 8) {
                len *= 2;
                snprintf(depth_suffix, sizeof depth_suffix, "p%d", depth);
        }

        fprintf(file, "YUV4MPEG2 W%d H%d F25:1 Ip A0:0 C%s%s XCOLORRANGE=%s\nFRAME\n",
                        width, height, chroma_type, depth_suffix, limited ? "LIMITED" : "FULL");
        fwrite((const char *) data, len, 1, file);
        bool ret = !ferror(file);
        if (!ret) {
                perror("Unable to write Y4M data");
        }
        fclose(file);
        return ret;
}

#endif // defined Y4M_H_DB8BEC68_AA48_4A63_81C3_DC3821F5555B
