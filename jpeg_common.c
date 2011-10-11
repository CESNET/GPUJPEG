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
 
#include "jpeg_common.h"
#include "jpeg_util.h"

/** Documented at declaration */
int
jpeg_image_load_from_file(const char* filename, int width, int height, unsigned char** image)
{
    FILE* file;
	file = fopen(filename, "rb");
	if ( !file ) {
		fprintf(stderr, "Failed open %s for reading!\n", filename);
		return -1;
	}

    int data_size = width * height * 3;
    unsigned char* data = (unsigned char*)malloc(data_size * sizeof(unsigned char));
    if ( data_size != fread(data, sizeof(unsigned char), data_size, file) ) {
        fprintf(stderr, "Failed to load image data [%d bytes] from file %s!\n", data_size, filename);
        return -1;
    }
    fclose(file);
    
    *image = data;
    
    return 0;
}

/** Documented at declaration */
int
jpeg_image_save_to_file(const char* filename, unsigned char* image, int width, int height)
{
    FILE* file;
	file = fopen(filename, "wb");
	if ( !file ) {
		fprintf(stderr, "Failed open %s for writing!\n", filename);
		return -1;
	}
    
    int data_size = width * height * 3 * sizeof(unsigned char);
    if ( data_size != fwrite(image, sizeof(unsigned char), data_size, file) ) {
        fprintf(stderr, "Failed to write image data [%d bytes] to file %s!\n", width * height * 3, filename);
        return -1;
    }
    fclose(file);
    
    return 0;
}

/** Documented at declaration */
int
jpeg_image_destroy(unsigned char* image)
{
    free(image);
}
