/**
 * @file
 * Copyright (c) 2011-2023, CESNET z.s.p.o
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
 
#ifndef GPUJPEG_UTIL_H
#define GPUJPEG_UTIL_H

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GPUJPEG_CLAMP(x, low, high)  (((x) > (high)) ? (high) : (((x) < (low)) ? (low) : (x)))
    
// CUDA check error
#define gpujpeg_cuda_check_error(msg, action) \
    { \
        cudaError_t err = cudaGetLastError(); \
        if( cudaSuccess != err) { \
            fprintf(stderr, "[GPUJPEG] [Error] %s (line %i): %s: %s.\n", \
                __FILE__, __LINE__, msg, cudaGetErrorString( err) ); \
            action; \
        } \
    } \

#define GPUJPEG_CHECK_EX(cmd, msg, action) do {\
        cmd;\
        gpujpeg_cuda_check_error(msg, action)\
} while(0)
#define GPUJPEG_CHECK(cmd, action) GPUJPEG_CHECK_EX(cmd, #cmd, action)
    
// Divide and round up
#define gpujpeg_div_and_round_up(value, div) \
    ((((value) % (div)) != 0) ? ((value) / (div) + 1) : ((value) / (div)))

// CUDA maximum grid size
#define GPUJPEG_CUDA_MAXIMUM_GRID_SIZE 65535

// CUDA C++ extension for Eclipse CDT
#ifdef __CDT_PARSER__
struct { int x; int y; int z; } threadIdx;
struct { int x; int y; int z; } blockIdx;
struct { int x; int y; int z; } blockDim;
struct { int x; int y; int z; } gridDim;
#endif

// OpenGL missing error
#define GPUJPEG_MISSING_OPENGL(action) \
    fprintf(stderr, "[GPUJPEG] [Error] Can't use OpenGL. The codec was compiled without OpenGL!\n"); \
    action; \

#define ARR_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))

/**
 * formats num with thousands delimitered by a comma (,)
 * @param buf    output buffer, must not be NULL or zero bytes long
 * @param buflen buffer len, should be long enough to hold the result
 * @returns pointer to buf (not necessarily the beginning) containng the represented number of "ERR"
 *          if not enough space
 */
char*
format_number_with_delim(size_t num, char* buf, size_t buflen);

#ifdef __cplusplus
}
#endif

#endif // GPUJPEG_UTIL_H
