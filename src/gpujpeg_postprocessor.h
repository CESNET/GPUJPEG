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

#ifndef GPUJPEG_POSTPROCESSOR_H
#define GPUJPEG_POSTPROCESSOR_H

#include "gpujpeg_decoder_internal.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Init preprocessor decoder
 *
 * @param encoder
 * @return 0 if succeeds, otherwise nonzero
 */
int
gpujpeg_postprocessor_decoder_init(struct gpujpeg_coder* coder);

/**
 * Preprocessor decode
 *
 * @param coder
 * @param stream
 * @retval  0 if succeeds
 * @retval -1 general error
 * @retval -2 JPEG with source subsampling cannot be decoded to specified planar pixel format
 */
int
gpujpeg_postprocessor_decode(struct gpujpeg_coder* coder, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif // GPUJPEG_POSTPROCESSOR_H
