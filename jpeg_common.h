/**
 * Copyright (c) 2011, Martin Srom
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
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
 
#ifndef JPEG_COMMON_H
#define JPEG_COMMON_H

#include <stdint.h>

/**
 * Load RGB image from file
 * 
 * @param filaname  Image filename
 * @param width  Image width in pixels
 * @param height  Image height in pixels
 * @param image  Image data buffer
 * @return 0 if succeeds, otherwise nonzero
 */
int
jpeg_image_load_from_file(const char* filename, int width, int height, uint8_t** image);

/**
 * Save RGB image to file
 * 
 * @param filaname  Image filename
 * @param image   Image data buffer
 * @param image_size  Image data buffer size
 * @return 0 if succeeds, otherwise nonzero
 */
int
jpeg_image_save_to_file(const char* filename, uint8_t* image, int image_size);

/**
 * Destroy DXT image
 * 
 * @param image  Image data buffer
 * @return 0 if succeeds, otherwise nonzero
 */
int
jpeg_image_destroy(uint8_t* image);

#endif // JPEG_COMMON_H