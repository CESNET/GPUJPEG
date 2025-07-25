/**
 * @file
 * Copyright (c) 2011-2025, CESNET
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

#ifndef GPUJPEG_PREPROCESSOR_H
#define GPUJPEG_PREPROCESSOR_H

#ifndef __cplusplus
#include <stdbool.h>
#endif

#include "../libgpujpeg/gpujpeg_type.h"


#ifdef __cplusplus
extern "C" {
#endif

struct gpujpeg_coder;
struct gpujpeg_encoder;

/**
 * Preprocessor/postprocessor data for component
 // */
struct gpujpeg_preprocessor_data_component
{
    uint8_t* d_data;
    int data_width;
    struct gpujpeg_component_sampling_factor sampling_factor;
};

/**
 * Preprocessor/postprocessor data
 */
struct gpujpeg_preprocessor_data
{
    struct gpujpeg_preprocessor_data_component comp[GPUJPEG_MAX_COMPONENT_COUNT];
};

/**
 * Preprocessor/postprocessor state
 */
struct gpujpeg_preprocessor
{
    void* kernel;       // function poitner
    bool input_flipped;         ///< [preprocess only] input buf is flipped
    unsigned int channel_remap; ///< remap input channels if != 0; currently preproecss only
                                ///< format: count_8b | 00000000 | idx0_4b | idx1_4b | idx2_4b | idx3_4b
    struct gpujpeg_preprocessor_data data;
};

/**
 * Init preprocessor encoder
 *
 * @param encoder
 * @return 0 if succeeds, otherwise nonzero
 */
int
gpujpeg_preprocessor_encoder_init(struct gpujpeg_coder* coder);

/**
 * Preprocessor encode
 *
 * @param encoder  Encoder structure
 * @param image  Image source data
 * @return 0 if succeeds, otherwise nonzero
 * @retval  0 if succeeds
 * @retval -1 general error
 * @retval -2 planar pixel format cannot be encoded with specified subsampling
 */
int
gpujpeg_preprocessor_encode(struct gpujpeg_encoder * encoder);

#ifdef __cplusplus
}
#endif

#endif // GPUJPEG_PREPROCESSOR_H
