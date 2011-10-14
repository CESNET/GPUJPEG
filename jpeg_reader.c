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
	if( jpeg_reader_read_byte(*image) != 255 )
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


/** Documented at declaration */
int
jpeg_reader_read_image(struct jpeg_decoder* decoder, uint8_t* image, int image_size)
{
    // Check first SOI marker
    int marker_soi = jpeg_reader_read_marker(&image);
    if ( marker_soi != JPEG_MARKER_SOI ) {
        fprintf(stderr, "Error: JPEG data should begin with SOI marker, but marker %s was found!\n", jpeg_marker_name(marker_soi));
        return -1;
    }
        
    // Step 1:
    int retval = -1;
    while ( 1 ) {
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

        /*case JPEG_MARKER_DQT:// maybe twice, one for Y, another for Cb/Cr
            GetDqt();
            break;

        case JPEG_MARKER_SOF0:        //* Baseline
        case JPEG_MARKER_SOF1:        //* Extended sequential, Huffman 
            get_sof(false, false);
            break;

        case JPEG_MARKER_SOF2:        //* Progressive, Huffman 
            //get_sof(true, false);    
            rgpuTrace("Prog + Huff is not supported");
            return -1;

        case JPEG_MARKER_SOF9:        //* Extended sequential, arithmetic 
            //get_sof(false, true);
            rgpuTrace("sequential + Arith is not supported");
            return -1;

        case JPEG_MARKER_SOF10:        //* Progressive, arithmetic 
            //get_sof(true, true);
            rgpuTrace("Prog + Arith is not supported");
            return -1;

        case JPEG_MARKER_DHT:
            get_dht();//4 tables: dc/ac * Y/CbCr
            break;

        case JPEG_MARKER_SOS://Start of Scan
            get_sos();
            retval = 0;//JPEG_REACHED_SOS;

            nWidth = gnJPEGDecoderWidth;
            nHeight = gnJPEGDecoderHeight;
            nHeadSize = m_pData - pInBuf;
            return retval;

        //the following marker are not needed for jpg made by ms paint
        case JPEG_MARKER_COM:
            SkipMarker();
            break;

        case JPEG_MARKER_DRI:
            get_dri();
            break;*/


/*            
        Currently unsupported SOFn types 
        case JPEG_MARKER_SOF3:         Lossless, Huffman
        case JPEG_MARKER_SOF5:         Differential sequential, Huffman 
        case JPEG_MARKER_SOF6:         Differential progressive, Huffman 
        case JPEG_MARKER_SOF7:         Differential lossless, Huffman 
        case JPEG_MARKER_JPG:             Reserved for JPEG extensions 
        case JPEG_MARKER_SOF11:         Lossless, arithmetic 
        case JPEG_MARKER_SOF13:         Differential sequential, arithmetic 
        case JPEG_MARKER_SOF14:         Differential progressive, arithmetic
        case JPEG_MARKER_SOF15:         Differential lossless, arithmetic 
            return -1;//ERREXIT1(cinfo, JERR_SOF_UNSUPPORTED, cinfo->unread_marker);
            break;
            
        case JPEG_MARKER_EOI:
            TRACEMS(cinfo, 1, JTRC_EOI);
            cinfo->unread_marker = 0;    
            return 1;//JPEG_REACHED_EOI;
            
        case JPEG_MARKER_DAC:
            if (! get_dac(cinfo))
                return -1;//JPEG_SUSPENDED;
            break;

        case JPEG_MARKER_RST0:        
        case JPEG_MARKER_RST1:
        case JPEG_MARKER_RST2:
        case JPEG_MARKER_RST3:
        case JPEG_MARKER_RST4:
        case JPEG_MARKER_RST5:
        case JPEG_MARKER_RST6:
        case JPEG_MARKER_RST7:
        case JPEG_MARKER_TEM:
            break;
            
        case JPEG_MARKER_DNL:            
        if (! skip_variable(cinfo))
                return -1;//JPEG_SUSPENDED;
            break;
*/            
        default:   
            fprintf(stderr, "Error: JPEG data contains not supported %s marker!\n", jpeg_marker_name(marker));
            jpeg_reader_skip_marker_content(&image);
            /* must be DHP, EXP, JPGn, or RESn */
            /* For now, we treat the reserved markers as fatal errors since they are
            * likely to be used to signal incompatible JPEG Part 3 extensions.
            * Once the JPEG 3 version-number marker is well defined, this code
            * ought to change!
            */
            return -1;
        }
        /* Successfully processed marker, so reset state variable */
        //unread_marker = 0;
    }
    
    /*if(( gnJPEGDecoderWidth <= 0 )||( gnJPEGDecoderHeight <= 0 ))
        return false;
    m_nDataBytesLeft = cbInBuf - nHeadSize;*/
    
    return -1;
}