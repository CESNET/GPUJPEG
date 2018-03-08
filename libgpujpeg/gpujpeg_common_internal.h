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
 
#ifndef GPUJPEG_COMMON_INTERNAL_H
#define GPUJPEG_COMMON_INTERNAL_H

#include "cuda_runtime.h"

/**
 * Declare timer
 *
 * @param name
 */
#define GPUJPEG_CUSTOM_TIMER_DECLARE(name) \
    double name ## _elapsedTime__;

/**
 * Start timer
 *
 * @param name
 */
#define GPUJPEG_CUSTOM_TIMER_START(name) \
    name ## _elapsedTime__ = gpujpeg_get_time();

/**
 * Stop timer
 *
 * @param name
 */
#define GPUJPEG_CUSTOM_TIMER_STOP(name) \
    name ## _elapsedTime__ = gpujpeg_get_time() - name ## _elapsedTime__;

/**
 * Get duration for timer
 *
 * @param name
 */
#define GPUJPEG_CUSTOM_TIMER_DURATION(name) (name ## _elapsedTime__ * 1000.0)

/**
 * Stop timer and print result
 *
 * @param name
 * @param text
 */
#define GPUJPEG_CUSTOM_TIMER_STOP_PRINT(name, text) \
    GPUJPEG_CUSTOM_TIMER_STOP(name); \
    printf("%s %0.2f ms\n", text, (name ## _elapsedTime__ * 1000.0))

/**
 * Default timer implementation
 */
#define GPUJPEG_TIMER_INIT() \
    GPUJPEG_CUSTOM_TIMER_DECLARE(def)
#define GPUJPEG_TIMER_START() GPUJPEG_CUSTOM_TIMER_START(def)
#define GPUJPEG_TIMER_STOP() GPUJPEG_CUSTOM_TIMER_STOP(def)
#define GPUJPEG_TIMER_DURATION() GPUJPEG_CUSTOM_TIMER_DURATION(def)
#define GPUJPEG_TIMER_STOP_PRINT(text) GPUJPEG_CUSTOM_TIMER_STOP_PRINT(def, text)

#endif // GPUJPEG_COMMON_INTERNAL_H
