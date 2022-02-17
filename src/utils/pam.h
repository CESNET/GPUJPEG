/**
 * @file   utils/pam.hpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 *
 * This file is a part of UltraGrid.
 */
/*
 * Copyright (c) 2013-2020 CESNET z.s.p.o.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, is permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * 3. Neither the name of CESNET nor the names of its contributors may be
 *    used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHORS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING,
 * BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
 * AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef PAM_H_7E23A609_963A_45A8_88E2_ED4D3FDFF69F
#define PAM_H_7E23A609_963A_45A8_88E2_ED4D3FDFF69F

#ifdef __cplusplus
#include <cstdio>
#include <cstdlib>
#else
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#endif

static inline bool pam_read(const char *filename, unsigned int *width, unsigned int *height, int *depth, unsigned char **data, void *(*allocator)(size_t)) {
        char line[128];
        FILE *file = fopen(filename, "rb");
        if (!file) {
                fprintf(stderr, "Failed to open %s!", filename);
                return false;
        }
        fgets(line, sizeof line - 1, file);
        if (feof(file) || ferror(file) || strcmp(line, "P7\n") != 0) {
               fprintf(stderr, "File '%s' doesn't seem to be valid PAM.", filename);
               fclose(file);
               return false;
        }
        fgets(line, sizeof line - 1, file);
        *width = 0, *height = 0, *depth = 0;
        while (!feof(file) && !ferror(file)) {
                char *spc = strchr(line, ' ');
                if (spc != NULL) {
                        const char *key = line;
                        *spc = '\0';
                        const char *val = spc + 1;
                        if (strcmp(key, "WIDTH") == 0) {
                                *width = atoi(val);
                        } else if (strcmp(key, "HEIGHT") == 0) {
                                *height = atoi(val);
                        } else if (strcmp(key, "DEPTH") == 0) {
                                *depth = atoi(val);
                        } else if (strcmp(key, "MAXVAL") == 0) {
                                if (atoi(val) != 255) {
                                        fprintf(stderr, "Only supported maxval is 255.\n");
                                        fclose(file);
                                        return false;
                                }
                        } else if (strcmp(key, "TUPLETYPE") == 0) {
                                // ignored - assuming MAXVAL == 255, value of DEPTH is sufficient
                                // to determine pixel format
                        }
                } else if (strcmp(line, "ENDHDR\n") == 0) {
                        break;
                }
                fgets(line, sizeof line - 1, file);
        }
        if (*width * *height == 0) {
                fprintf(stderr, "Unspecified size header field!");
                fclose(file);
                return false;
        }
        if (*depth == 0) {
                fprintf(stderr, "Unspecified depth header field!");
                fclose(file);
                return false;
        }
        if (data != NULL && allocator != NULL) {
                int datalen = *depth * *width * *height;
                *data = (unsigned char *) allocator(datalen);
                if (!*data) {
                        fprintf(stderr, "Unspecified depth header field!");
                        fclose(file);
                        return false;
                }
                fread((char *) *data, datalen, 1, file);
                if (feof(file) || ferror(file)) {
                        fprintf(stderr, "Unable to load PAM data from file.");
                        fclose(file);
                        return false;
                }
        }
        fclose(file);
        return true;
}

static bool pam_write(const char *filename, unsigned int width, unsigned int height, int depth, const unsigned char *data) {
        FILE *file = fopen(filename, "wb");
        if (!file) {
                fprintf(stderr, "Failed to open %s for writing!", filename);
                return false;
        }
        const char *tuple_type = "INVALID";
        switch (depth) {
                case 4: tuple_type = "RGB_ALPHA"; break;
                case 3: tuple_type = "RGB"; break;
                case 2: tuple_type = "GRAYSCALE_ALPHA"; break;
                case 1: tuple_type = "GRAYSCALE"; break;
                default: fprintf(stderr, "Wrong depth: %d\n", depth);
        }
        fprintf(file, "P7\n"
                "WIDTH %u\n"
                "HEIGHT %u\n"
                "DEPTH %d\n"
                "MAXVAL 255\n"
                "TUPLTYPE %s\n"
                "ENDHDR\n",
                width, height, depth, tuple_type);
        fwrite((const char *) data, width * height * depth, 1, file);
        bool ret = !ferror(file);
        fclose(file);
        return ret;
}

#endif // defined PAM_H_7E23A609_963A_45A8_88E2_ED4D3FDFF69F
