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

#ifndef GPUJPEG_TYPE_H
#define GPUJPEG_TYPE_H

#include <stdint.h>

static const int GPUJPEG_BLOCK_SIZE = 8;

/**
 * Color spaces for JPEG codec
 */
enum gpujpeg_color_space {
    GPUJPEG_RGB = 1,
    GPUJPEG_YUV = 2,
    GPUJPEG_YCBCR = 3,
};

/**
 * Sampling factor for JPEG codec
 */
enum gpujpeg_sampling_factor {
    GPUJPEG_4_4_4 = ((4 << 16) | (4 << 8) | 4),
    GPUJPEG_4_2_2 = ((4 << 16) | (2 << 8) | 2),
};

/**
 * JPEG component type
 */
enum gpujpeg_component_type {
    GPUJPEG_COMPONENT_LUMINANCE = 0,
    GPUJPEG_COMPONENT_CHROMINANCE = 1,
    GPUJPEG_COMPONENT_TYPE_COUNT = 2
};

/** 
 * JPEG huffman type 
 */
enum gpujpeg_huffman_type {
    GPUJPEG_HUFFMAN_DC = 0,
    GPUJPEG_HUFFMAN_AC = 1,
    GPUJPEG_HUFFMAN_TYPE_COUNT = 2
};

#endif // GPUJPEG_TYPE_H
