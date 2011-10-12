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
 
#include "jpeg_writer.h"
#include "jpeg_writer_type.h"
#include "jpeg_encoder.h"
#include "jpeg_util.h"

/** Documented at declaration */
struct jpeg_writer*
jpeg_writer_create(struct jpeg_encoder* encoder)
{
    struct jpeg_writer* writer = malloc(sizeof(struct jpeg_writer));
    if ( writer == NULL )
        return NULL;
    
    // Allocate output buffer
    writer->buffer = malloc(encoder->width * encoder->height * encoder->comp_count * sizeof(uint8_t));
    if ( writer->buffer == NULL )
        return NULL;
    
    return writer;
}

/** Documented at declaration */
int
jpeg_writer_destroy(struct jpeg_writer* writer)
{
    assert(writer != NULL);
    assert(writer->buffer != NULL);
    free(writer->buffer);
    free(writer);
    return 0;
}