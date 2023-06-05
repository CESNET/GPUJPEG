/*
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

#include "../libgpujpeg/gpujpeg_type.h"

#include <cstdint>

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

/** Value that means that sampling factor has dynamic value */
#define GPUJPEG_DYNAMIC 16

/** Sampling factor for all components */
typedef uint32_t gpujpeg_preprocessor_sampling_factor_t;

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

/**
 * Compose sampling factor for all components to single type
 *
 * @return integer that contains all sampling factors
 */
inline gpujpeg_preprocessor_sampling_factor_t
gpujpeg_preprocessor_make_sampling_factor(int comp1_h, int comp1_v, int comp2_h, int comp2_v, int comp3_h, int comp3_v, int comp4_h, int comp4_v)
{
    gpujpeg_preprocessor_sampling_factor_t sampling_factor = 0;
    sampling_factor |= ((comp1_h << 4U) | comp1_v) << 24U;
    sampling_factor |= ((comp2_h << 4U) | comp2_v) << 16U;
    sampling_factor |= ((comp3_h << 4U) | comp3_v) << 8U;
    sampling_factor |= ((comp4_h << 4U) | comp4_v) << 0U;

    return sampling_factor;
}

/* vi: set expandtab sw=4: */
