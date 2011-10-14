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
 
#include "jpeg_reader.h"
#include "jpeg_format_type.h"
#include "jpeg_util.h"

/** Documented at declaration */
struct jpeg_reader*
jpeg_reader_create()
{
    struct jpeg_reader* reader = malloc(sizeof(struct jpeg_reader));
    if ( reader == NULL )
        return NULL;
    
    return reader;
}

/** Documented at declaration */
int
jpeg_reader_destroy(struct jpeg_reader* reader)
{
    assert(reader != NULL);
    free(reader);
    return 0;
}

#define jpeg_reader_read_byte(image) \
    (uint8_t)(*(image)++)
    
#define jpeg_reader_read_2byte(image) \
    (uint16_t)(((*(image)) << 8) + (*((image) + 1))); \
    image += 2;

int
jpeg_reader_read_marker(uint8_t** image)
{
	if( jpeg_reader_read_byte(*image) != 0xFF )
		return -1;
	int marker = jpeg_reader_read_byte(*image);
	return marker;
}

void
jpeg_reader_skip_marker_content(uint8_t** image)
{
    int length = (int)jpeg_reader_read_2byte(*image);

	*image += length - 2;
}

int
jpeg_reader_read_app0(uint8_t** image)
{
    int length = (int)jpeg_reader_read_2byte(*image);
    if ( length != 16 ) {
        fprintf(stderr, "APP0 marker length should be 16 but %d was presented!\n", length);
        return -1;
    }

    char jfif[4];
    jfif[0] = jpeg_reader_read_byte(*image);
    jfif[1] = jpeg_reader_read_byte(*image);
    jfif[2] = jpeg_reader_read_byte(*image);
    jfif[3] = jpeg_reader_read_byte(*image);
    jfif[4] = jpeg_reader_read_byte(*image);
    if ( strcmp(jfif, "JFIF") != 0 ) {
        fprintf(stderr, "APP0 marker identifier should be 'JFIF' but '%s' was presented!\n", jfif);
        return -1;
    }

    int version_major = jpeg_reader_read_byte(*image);
    int version_minor = jpeg_reader_read_byte(*image);
    if ( version_major != 1 || version_minor != 1 ) {
        fprintf(stderr, "APP0 marker version should be 1.1 but %d.%d was presented!\n", version_major, version_minor);
        return -1;
    }
    
    int pixel_units = jpeg_reader_read_byte(*image);
    int pixel_xdpu = jpeg_reader_read_2byte(*image);
    int pixel_ydpu = jpeg_reader_read_2byte(*image);
    int thumbnail_width = jpeg_reader_read_byte(*image);
    int thumbnail_height = jpeg_reader_read_byte(*image);

    return 0;
}

int
jpeg_reader_read_dqt(uint8_t** image)
{
    fprintf(stderr, "Todo: Read DQT marker!\n");
    jpeg_reader_skip_marker_content(image);
    return 0;
}

int
jpeg_reader_read_sof0(uint8_t** image)
{
    fprintf(stderr, "Todo: Read SOF0 marker!\n");
    jpeg_reader_skip_marker_content(image);
    return 0;
}

int
jpeg_reader_read_dht(uint8_t** image)
{
    fprintf(stderr, "Todo: Read DHT marker!\n");
    jpeg_reader_skip_marker_content(image);
    return 0;
}

int
jpeg_reader_read_sos(uint8_t** image, uint8_t* image_end)
{    
    uint8_t buffer[1920 * 1080];
    uint8_t byte = 0;
    uint8_t byte_previous = 0;
    int byte_count = 0;
    do {
        byte_previous = byte;
        byte = jpeg_reader_read_byte(*image);
        byte_count++;
        
        if ( byte_previous == 0xFF && (byte == JPEG_MARKER_EOI || byte == JPEG_MARKER_SOS) ) {
            fprintf(stderr, "Todo: Save SOS marker data %d bytes!\n", byte_count);
            *image -= 2;
            return 0;
        }
    } while( *image < image_end );
    
    fprintf(stderr, "JPEG data unexpected ended while reading SOS marker!\n");
    
    return -1;
}

/** Documented at declaration */
int
jpeg_reader_read_image(struct jpeg_decoder* decoder, uint8_t* image, int image_size)
{
    // Get image end
    uint8_t* image_end = image + image_size;
    
    // Check first SOI marker
    int marker_soi = jpeg_reader_read_marker(&image);
    if ( marker_soi != JPEG_MARKER_SOI ) {
        fprintf(stderr, "Error: JPEG data should begin with SOI marker, but marker %s was found!\n", jpeg_marker_name(marker_soi));
        return -1;
    }
        
    int eoi_presented = 0;
    while ( eoi_presented == 0 ) {
        // Read marker
        int marker = jpeg_reader_read_marker(&image);

        // Read more info according to the marker
        // the order of cases is in jpg file made by ms paint
        switch (marker) 
        {
        case JPEG_MARKER_APP0:
            if ( jpeg_reader_read_app0(&image) != 0 )
                return -1;
            break;
        case JPEG_MARKER_APP1:
        case JPEG_MARKER_APP2:
        case JPEG_MARKER_APP3:
        case JPEG_MARKER_APP4:
        case JPEG_MARKER_APP5:
        case JPEG_MARKER_APP6:
        case JPEG_MARKER_APP7:
        case JPEG_MARKER_APP8:
        case JPEG_MARKER_APP9:
        case JPEG_MARKER_APP10:
        case JPEG_MARKER_APP11:
        case JPEG_MARKER_APP12:
        case JPEG_MARKER_APP13:
        case JPEG_MARKER_APP14:
        case JPEG_MARKER_APP15:
            fprintf(stderr, "Warning: JPEG data contains not supported %s marker\n", jpeg_marker_name(marker));
            jpeg_reader_skip_marker_content(&image);
            break;
            
        case JPEG_MARKER_DQT:
            if ( jpeg_reader_read_dqt(&image) != 0 )
                return -1;
            break;

        case JPEG_MARKER_SOF0: 
            // Baseline
            if ( jpeg_reader_read_sof0(&image) != 0 )
                return -1;
            break;
        case JPEG_MARKER_SOF1:
            // Extended sequential with Huffman coder
            fprintf(stderr, "Warning: Reading SOF1 as it was SOF0 marker (should work but verify it)!\n", jpeg_marker_name(marker));
            if ( jpeg_reader_read_sof0(&image) != 0 )
                return -1;
            break;
        case JPEG_MARKER_SOF2:
            fprintf(stderr, "Error: Marker SOF2 (Progressive with Huffman coding) is not supported!");
            return -1;
        case JPEG_MARKER_SOF3:
            fprintf(stderr, "Error: Marker SOF3 (Lossless with Huffman coding) is not supported!");
            return -1;
        case JPEG_MARKER_SOF5:
            fprintf(stderr, "Error: Marker SOF5 (Differential sequential with Huffman coding) is not supported!");
            return -1;
        case JPEG_MARKER_SOF6:
            fprintf(stderr, "Error: Marker SOF6 (Differential progressive with Huffman coding) is not supported!");
            return -1;
        case JPEG_MARKER_SOF7:
            fprintf(stderr, "Error: Marker SOF7 (Extended lossless with Arithmetic coding) is not supported!");
            return -1;
        case JPEG_MARKER_JPG:
            fprintf(stderr, "Error: Marker JPG (Reserved for JPEG extensions ) is not supported!");
            return -1;
        case JPEG_MARKER_SOF10:
            fprintf(stderr, "Error: Marker SOF10 (Progressive with Arithmetic coding) is not supported!");
            return -1;
        case JPEG_MARKER_SOF11:
            fprintf(stderr, "Error: Marker SOF11 (Lossless with Arithmetic coding) is not supported!");
            return -1;
        case JPEG_MARKER_SOF13:
            fprintf(stderr, "Error: Marker SOF13 (Differential sequential with Arithmetic coding) is not supported!");
            return -1;
        case JPEG_MARKER_SOF14:
            fprintf(stderr, "Error: Marker SOF14 (Differential progressive with Arithmetic coding) is not supported!");
            return -1;
        case JPEG_MARKER_SOF15:
            fprintf(stderr, "Error: Marker SOF15 (Differential lossless with Arithmetic coding) is not supported!");
            return -1;
            
        case JPEG_MARKER_DHT:
            if ( jpeg_reader_read_dht(&image) != 0 )
                return -1;
            break;

        case JPEG_MARKER_SOS:
            if ( jpeg_reader_read_sos(&image, image_end) != 0 )
                return -1;
            break;
            
        case JPEG_MARKER_EOI:
            eoi_presented = 1;
            break;

        case JPEG_MARKER_COM:
        case JPEG_MARKER_DRI:
        case JPEG_MARKER_DAC:
        case JPEG_MARKER_DNL:
            fprintf(stderr, "Warning: JPEG data contains not supported %s marker\n", jpeg_marker_name(marker));
            jpeg_reader_skip_marker_content(&image);
            break;
            
        default:   
            fprintf(stderr, "Error: JPEG data contains not supported %s marker!\n", jpeg_marker_name(marker));
            jpeg_reader_skip_marker_content(&image);
            return -1;
        }
    }
    
    // Check EOI marker
    if ( eoi_presented == 0 ) {
        fprintf(stderr, "Error: JPEG data should end with EOI marker!\n");
        return -1;
    }
    
    return 0;
}