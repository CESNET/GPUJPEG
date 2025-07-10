/*
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

#ifndef GPUJPEG_PREPROCESSOR_COMMON_CUH_DCC657E3_2EDF_47E2_90F4_F7CA26829E81
#define GPUJPEG_PREPROCESSOR_COMMON_CUH_DCC657E3_2EDF_47E2_90F4_F7CA26829E81

#include "../libgpujpeg/gpujpeg_common.h"
#include "../libgpujpeg/gpujpeg_type.h"

#define RGB_8BIT_THREADS 256

/**
 * Preprocessor data for component
 */
struct gpujpeg_preprocessor_data_component
{
    uint8_t* d_data;
    int data_width;
    struct gpujpeg_component_sampling_factor sampling_factor;
};

/**
 * Preprocessor data
 */
struct gpujpeg_preprocessor_data
{
    struct gpujpeg_preprocessor_data_component comp[GPUJPEG_MAX_COMPONENT_COUNT];
};

struct gpujpeg_preprocessor {
    void* kernel; // function poitner
    struct gpujpeg_preprocessor_data data;
};

#ifdef PREPROCESSOR_INTERNAL_API
#include "gpujpeg_common_internal.h"
#include <cassert>
#include <cstdint>
/** Value that means that sampling factor has dynamic value */
#define GPUJPEG_DYNAMIC 16

/**
 * Prepares fixed divisor for dividing unsigned integers up to 2^31
 * with unsigned integers up to 2^31.
 * Source: http://www.hackersdelight.org/HDcode/magic.c.txt
 * Modified for positive numbers only.
 */
static void inline
gpujpeg_const_div_prepare(const uint32_t d, uint32_t & pre_div_mul, uint32_t & pre_div_shift) {
    if(d > 1) {
        uint32_t delta;
        const uint32_t two31 = 0x80000000; // 2**31.
        const uint32_t anc = two31 - 1 - two31 % d; // Absolute value of nc.
        int p = 31;                        // Init. p.
        uint32_t q1 = two31 / anc;         // Init. q1 = 2**p/|nc|.
        uint32_t r1 = two31 - q1 * anc;    // Init. r1 = rem(2**p, |nc|).
        uint32_t q2 = two31 / d;           // Init. q2 = 2**p/|d|.
        uint32_t r2 = two31 - q2 * d;      // Init. r2 = rem(2**p, |d|).
        do {
            p = p + 1;
            q1 = 2 * q1;                   // Update q1 = 2**p/|nc|.
            r1 = 2 * r1;                   // Update r1 = rem(2**p, |nc|).
            if (r1 >= anc) {               // (Must be an unsigned
                q1 = q1 + 1;               // comparison here).
                r1 = r1 - anc;
            }
            q2 = 2 * q2;                   // Update q2 = 2**p/|d|.
            r2 = 2 * r2;                   // Update r2 = rem(2**p, |d|).
            if (r2 >= d) {                 // (Must be an unsigned
                q2 = q2 + 1;               // comparison here).
                r2 = r2 - d;
            }
            delta = d - r2;
        } while (q1 < delta || (q1 == delta && r1 == 0));
        pre_div_mul = q2 + 1;
        pre_div_shift = p - 32;            // shift amount to return.
    } else {
        pre_div_mul = 0;                   // special case for d = 1
        pre_div_shift = 0;
    }
}


/**
 * Divides unsigned numerator (up to 2^31) by precomputed constant denominator.
 */
__device__ static uint32_t
gpujpeg_const_div_divide(const uint32_t numerator, const uint32_t pre_div_mul, const uint32_t pre_div_shift) {
    return pre_div_mul ? __umulhi(numerator, pre_div_mul) >> pre_div_shift : numerator;
}

inline gpujpeg_sampling_factor_t
gpujpeg_preprocessor_make_sampling_factor_i(int comp_count, int numerator_h, int numerator_v, int comp1_h, int comp1_v, int comp2_h, int comp2_v,
                                            int comp3_h, int comp3_v, int comp4_h, int comp4_v) {
    return gpujpeg_make_sampling_factor(
        comp_count, numerator_h / comp1_h, numerator_v / comp1_v,
        comp2_h != 0 ? numerator_h / comp2_h : 0, comp2_v != 0 ? numerator_v / comp2_v : 0,
        comp3_h != 0 ? numerator_h / comp3_h : 0, comp3_v != 0 ? numerator_v / comp3_v : 0,
        comp4_h != 0 ? numerator_h / comp4_h : 0, comp4_v != 0 ? numerator_v / comp4_v : 0);
}

#define print_kernel_configuration(msg)                                                                                \
    PRINTF(msg, coder->component[0].sampling_factor.horizontal, coder->component[0].sampling_factor.vertical,          \
           coder->component[1].sampling_factor.horizontal, coder->component[1].sampling_factor.vertical,               \
           coder->component[2].sampling_factor.horizontal, coder->component[2].sampling_factor.vertical,               \
           coder->component[3].sampling_factor.horizontal, coder->component[3].sampling_factor.vertical)
#endif // defined PREPROCESSOR_INTERNAL_API

#endif // defined GPUJPEG_PREPROCESSOR_COMMON_CUH_DCC657E3_2EDF_47E2_90F4_F7CA26829E81
/* vi: set expandtab sw=4: */
