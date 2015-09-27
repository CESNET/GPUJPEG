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
 
#include <libgpujpeg/gpujpeg_reader.h>
#include <libgpujpeg/gpujpeg_decoder.h>
#include <libgpujpeg/gpujpeg_decoder_internal.h>
#include <libgpujpeg/gpujpeg_util.h>

/** Documented at declaration */
struct gpujpeg_reader*
gpujpeg_reader_create()
{
    struct gpujpeg_reader* reader = (struct gpujpeg_reader*)
            malloc(sizeof(struct gpujpeg_reader));
    if ( reader == NULL )
        return NULL;
    reader->comp_count = 0;
    reader->scan_count = 0;
    reader->segment_count = 0;
    reader->segment_info_count = 0;
    reader->segment_info_size = 0;
    
    return reader;
}

/** Documented at declaration */
int
gpujpeg_reader_destroy(struct gpujpeg_reader* reader)
{
    assert(reader != NULL);
    free(reader);
    return 0;
}

/**
 * Read byte from image data
 * 
 * @param image
 * @return byte
 */
#define gpujpeg_reader_read_byte(image) \
    (uint8_t)(*(image)++)

/**
 * Read two-bytes from image data
 * 
 * @param image
 * @return 2 bytes
 */    
#define gpujpeg_reader_read_2byte(image) \
    (uint16_t)(((*(image)) << 8) + (*((image) + 1))); \
    image += 2;

/**
 * Read marker from image data
 * 
 * @param image
 * @return marker code or -1 if failed
 */
int
gpujpeg_reader_read_marker(uint8_t** image)
{
    uint8_t byte = gpujpeg_reader_read_byte(*image);
    if( byte != 0xFF ) {
        fprintf(stderr, "[GPUJPEG] [Error] Failed to read marker from JPEG data (0xFF was expected but 0x%X was presented)\n", byte);
        return -1;
    }
    int marker = gpujpeg_reader_read_byte(*image);
    return marker;
}

/**
 * Skip marker content (read length and that much bytes - 2)
 * 
 * @param image
 * @return void
 */
void
gpujpeg_reader_skip_marker_content(uint8_t** image)
{
    int length = (int)gpujpeg_reader_read_2byte(*image);

    *image += length - 2;
}

/**
 * Read application ifno block from image
 * 
 * @param image
 * @return 0 if succeeds, otherwise nonzero
 */
int
gpujpeg_reader_read_app0(uint8_t** image)
{
    int length = (int)gpujpeg_reader_read_2byte(*image);
    if ( length != 16 ) {
        fprintf(stderr, "[GPUJPEG] [Error] APP0 marker length should be 16 but %d was presented!\n", length);
        return -1;
    }

    char jfif[5];
    jfif[0] = gpujpeg_reader_read_byte(*image);
    jfif[1] = gpujpeg_reader_read_byte(*image);
    jfif[2] = gpujpeg_reader_read_byte(*image);
    jfif[3] = gpujpeg_reader_read_byte(*image);
    jfif[4] = gpujpeg_reader_read_byte(*image);
    if ( strcmp(jfif, "JFIF") != 0 ) {
        fprintf(stderr, "[GPUJPEG] [Error] APP0 marker identifier should be 'JFIF' but '%s' was presented!\n", jfif);
        return -1;
    }

    int version_major = gpujpeg_reader_read_byte(*image);
    int version_minor = gpujpeg_reader_read_byte(*image);
    if ( version_major != 1 || version_minor != 1 ) {
        fprintf(stderr, "[GPUJPEG] [Error] APP0 marker version should be 1.1 but %d.%d was presented!\n", version_major, version_minor);
        return -1;
    }
    
    int pixel_units = gpujpeg_reader_read_byte(*image);
    int pixel_xdpu = gpujpeg_reader_read_2byte(*image);
    int pixel_ydpu = gpujpeg_reader_read_2byte(*image);
    int thumbnail_width = gpujpeg_reader_read_byte(*image);
    int thumbnail_height = gpujpeg_reader_read_byte(*image);

    return 0;
}

/**
 * Read quantization table definition block from image
 * 
 * @param decoder
 * @param image
 * @return 0 if succeeds, otherwise nonzero
 */
int
gpujpeg_reader_read_dqt(struct gpujpeg_decoder* decoder, uint8_t** image)
{    
    int length = (int)gpujpeg_reader_read_2byte(*image);
    length -= 2;
    
    if ( (length % 65) != 0 ) {
        fprintf(stderr, "[GPUJPEG] [Error] DQT marker length should be 65 but %d was presented!\n", length);
        return -1;
    }

    for (;length > 0; length-=65) {
	int index = gpujpeg_reader_read_byte(*image);
	struct gpujpeg_table_quantization* table;
	if( index == 0 ) {
	    table = &decoder->table_quantization[GPUJPEG_COMPONENT_LUMINANCE];
	} else if ( index == 1 ) {
	    table = &decoder->table_quantization[GPUJPEG_COMPONENT_CHROMINANCE];
	} else {
	    fprintf(stderr, "[GPUJPEG] [Error] DQT marker index should be 0 or 1 but %d was presented!\n", index);
	    return -1;
	}

	for ( int i = 0; i < 64; i++ ) {
	    table->table_raw[i] = gpujpeg_reader_read_byte(*image);
	}
	// Prepare quantization table for read raw table
	gpujpeg_table_quantization_decoder_compute(table);
    }
    return 0;
}

/**
 * Read start of frame block from image
 * 
 * @param image
 * @return 0 if succeeds, otherwise nonzero
 */
int
gpujpeg_reader_read_sof0(struct gpujpeg_decoder* decoder, uint8_t** image)
{    
    int length = (int)gpujpeg_reader_read_2byte(*image);
    if ( length < 6 ) {
        fprintf(stderr, "[GPUJPEG] [Error] SOF0 marker length should be greater than 6 but %d was presented!\n", length);
        return -1;
    }
    length -= 2;

    int precision = (int)gpujpeg_reader_read_byte(*image);
    if ( precision != 8 ) {
        fprintf(stderr, "[GPUJPEG] [Error] SOF0 marker precision should be 8 but %d was presented!\n", precision);
        return -1;
    }
    
    decoder->reader->param_image.height = (int)gpujpeg_reader_read_2byte(*image);
    decoder->reader->param_image.width = (int)gpujpeg_reader_read_2byte(*image);
    decoder->reader->param_image.comp_count = (int)gpujpeg_reader_read_byte(*image);
    length -= 6;

    for ( int comp = 0; comp < decoder->reader->param_image.comp_count; comp++ ) {
        int index = (int)gpujpeg_reader_read_byte(*image);
        if ( index != (comp + 1) ) {
            fprintf(stderr, "[GPUJPEG] [Error] SOF0 marker component %d id should be %d but %d was presented!\n", comp, comp + 1, index);
            return -1;
        }
        
        int sampling = (int)gpujpeg_reader_read_byte(*image);
        decoder->reader->param.sampling_factor[comp].horizontal = (sampling >> 4) & 15;
        decoder->reader->param.sampling_factor[comp].vertical = (sampling) & 15;
        
        int table_index = (int)gpujpeg_reader_read_byte(*image);
        if ( comp == 0 && table_index != 0 ) {
            fprintf(stderr, "[GPUJPEG] [Error] SOF0 marker component Y should have quantization table index 0 but %d was presented!\n", table_index);
            return -1;
        }
        if ( (comp == 1 || comp == 2) && table_index != 1 ) {
            fprintf(stderr, "[GPUJPEG] [Error] SOF0 marker component Cb or Cr should have quantization table index 1 but %d was presented!\n", table_index);
            return -1;
        }
        length -= 3;
    }
    
    // Check length
    if ( length > 0 ) {
        fprintf(stderr, "[GPUJPEG] [Warning] SOF0 marker contains %d more bytes than needed!\n", length);
        *image += length;
    }
    
    return 0;
}

/**
 * Read huffman table definition block from image
 * 
 * @param decoder
 * @param image
 * @return 0 if succeeds, otherwise nonzero
 */
int
gpujpeg_reader_read_dht(struct gpujpeg_decoder* decoder, uint8_t** image)
{    
    int length = (int)gpujpeg_reader_read_2byte(*image);
    length -= 2;

    while (length > 0) {
	int index = gpujpeg_reader_read_byte(*image);
	struct gpujpeg_table_huffman_decoder* table = NULL;
	struct gpujpeg_table_huffman_decoder* d_table = NULL;
	switch(index) {
	case 0:
	    table = &decoder->table_huffman[GPUJPEG_COMPONENT_LUMINANCE][GPUJPEG_HUFFMAN_DC];
	    d_table = decoder->d_table_huffman[GPUJPEG_COMPONENT_LUMINANCE][GPUJPEG_HUFFMAN_DC];
	    break;
	case 16:
	    table = &decoder->table_huffman[GPUJPEG_COMPONENT_LUMINANCE][GPUJPEG_HUFFMAN_AC];
	    d_table = decoder->d_table_huffman[GPUJPEG_COMPONENT_LUMINANCE][GPUJPEG_HUFFMAN_AC];
	    break;
	case 1:
	    table = &decoder->table_huffman[GPUJPEG_COMPONENT_CHROMINANCE][GPUJPEG_HUFFMAN_DC];
	    d_table = decoder->d_table_huffman[GPUJPEG_COMPONENT_CHROMINANCE][GPUJPEG_HUFFMAN_DC];
	    break;
	case 17:
	    table = &decoder->table_huffman[GPUJPEG_COMPONENT_CHROMINANCE][GPUJPEG_HUFFMAN_AC];
	    d_table = decoder->d_table_huffman[GPUJPEG_COMPONENT_CHROMINANCE][GPUJPEG_HUFFMAN_AC];
	    break;
	default:
	    fprintf(stderr, "[GPUJPEG] [Error] DHT marker index should be 0, 1, 16 or 17 but %d was presented!\n", index);
	    return -1;
	}
	length -= 1;
    
	// Read in bits[]
	table->bits[0] = 0;
	int count = 0;
	for ( int i = 1; i <= 16; i++ ) {
	    table->bits[i] = gpujpeg_reader_read_byte(*image);
	    count += table->bits[i];
	    if ( length > 0 ) {
		length--;
	    } else {
		fprintf(stderr, "[GPUJPEG] [Error] DHT marker unexpected end when reading bit counts!\n", index);
		return -1;
	    }
	}   

	// Read in huffval
	for ( int i = 0; i < count; i++ ){
	    table->huffval[i] = gpujpeg_reader_read_byte(*image);
	    if ( length > 0 ) {
		length--;
	    } else {
		fprintf(stderr, "[GPUJPEG] [Error] DHT marker unexpected end when reading huffman values!\n", index);
		return -1;
	    }
	}
	// Compute huffman table for read values
	gpujpeg_table_huffman_decoder_compute(table, d_table);
    }
    return 0;
}

/**
 * Read restart interval block from image
 * 
 * @param decoder
 * @param image
 * @return 0 if succeeds, otherwise nonzero
 */
int
gpujpeg_reader_read_dri(struct gpujpeg_decoder* decoder, uint8_t** image)
{
    int length = (int)gpujpeg_reader_read_2byte(*image);
    if ( length != 4 ) {
        fprintf(stderr, "[GPUJPEG] [Error] DRI marker length should be 4 but %d was presented!\n", length);
        return -1;
    }
    
    int restart_interval = gpujpeg_reader_read_2byte(*image);
    if ( restart_interval == decoder->reader->param.restart_interval )
        return 0;
    
    if ( decoder->reader->param.restart_interval != 0 ) {
        fprintf(stderr, "[GPUJPEG] [Error] DRI marker can't redefine restart interval!");
        fprintf(stderr, "This may be caused when more DRI markers are presented which is not supported!\n");
        return -1;
    }
    
    decoder->reader->param.restart_interval = restart_interval;
    
    return 0;
}

/**
 * Read segment info for following scan
 *
 * @param decoder
 * @param image
 * @return 0 if succeeds, otherwise nonzero
 */
int
gpujpeg_reader_read_segment_info(struct gpujpeg_decoder* decoder, uint8_t** image)
{
    int length = (int)gpujpeg_reader_read_2byte(*image);
    int scan_index = (int)gpujpeg_reader_read_byte(*image);
    if ( length <= 3 ) {
        fprintf(stderr, "[GPUJPEG] [Error] %s marker (segment info) length should be greater than 3 but %d was presented!\n",
                gpujpeg_marker_name(GPUJPEG_MARKER_SEGMENT_INFO), length);
        return -1;
    }
    if ( scan_index != decoder->reader->scan_count ) {
        fprintf(stderr, "[GPUJPEG] [Error] %s marker (segment info) scan index should be %d but %d was presented!\n",
                gpujpeg_marker_name(GPUJPEG_MARKER_SEGMENT_INFO), decoder->reader->scan_count, scan_index);
        return -1;
    }

    int data_size = length - 3;
    decoder->reader->segment_info[decoder->reader->segment_info_count] = *image;
    decoder->reader->segment_info_count++;
    decoder->reader->segment_info_size += data_size;

    *image += data_size;

    return 0;
}

/**
 * Read scan content by parsing byte-by-byte
 *
 * @param decoder
 * @param image
 * @param image_end
 * @param scan
 * @param scan_index
 * @return 0 if succeeds, otherwise nonzero
 */
int
gpujpeg_reader_read_scan_content_by_parsing(struct gpujpeg_decoder* decoder, uint8_t** image, uint8_t* image_end,
        struct gpujpeg_reader_scan* scan, int scan_index)
{
    // Get first segment in scan
    struct gpujpeg_segment* segment = &decoder->coder.segment[scan->segment_index];
    segment->scan_index = scan_index;
    segment->scan_segment_index = scan->segment_count;
    segment->data_compressed_index = decoder->reader->data_compressed_size;
    scan->segment_count++;
    
    // Read scan data
    int result = -1;
    uint8_t byte = 0;
    uint8_t byte_previous = 0;
    uint8_t previous_marker = GPUJPEG_MARKER_RST0 - 1;
    do {
        byte_previous = byte;
        byte = gpujpeg_reader_read_byte(*image);
        decoder->coder.data_compressed[decoder->reader->data_compressed_size] = byte;
        decoder->reader->data_compressed_size++;        
        
        // Check markers
        if ( byte_previous == 0xFF ) {
            // Check zero byte
            if ( byte == 0 ) {
                continue;
            }
            // Check restart marker
            else if ( byte >= GPUJPEG_MARKER_RST0 && byte <= GPUJPEG_MARKER_RST7 ) {
                // Check expected marker
                uint8_t expected_marker = (previous_marker < GPUJPEG_MARKER_RST7) ? (previous_marker + 1) : GPUJPEG_MARKER_RST0;
                if ( expected_marker != byte ) {
                    fprintf(stderr, "[GPUJPEG] [Error] Expected marker 0x%X but 0x%X was presented!\n", expected_marker, byte);

                    // Skip bytes to expected marker
                    int found_expected_marker = 0;
                    int skip_count = 0;
                    byte_previous = byte;
                    while ( *image < image_end ) {
                        skip_count++;
                        byte = gpujpeg_reader_read_byte(*image);
                        if ( byte_previous == 0xFF ) {
                            // Expected marker was found so notify about it
                            if ( byte == expected_marker ) {
                                fprintf(stderr, "[GPUJPEG] [Recovery] Skipping %d bytes of data until marker 0x%X was found!\n", skip_count, expected_marker, byte);
                                found_expected_marker = 1;
                                break;
                            } else if ( byte == GPUJPEG_MARKER_EOI || byte == GPUJPEG_MARKER_SOS ) {
                                // Go back last marker (will be read again by main read cycle)
                                *image -= 2;
                                break;
                            }
                        }
                        byte_previous = byte;
                    }

                    // If expected marker was not found to end of stream
                    if ( found_expected_marker == 0 ) {
                        fprintf(stderr, "[GPUJPEG] [Error] No marker 0x%X was found until end of current scan!\n", expected_marker);
                        continue;
                    }
                }
                // Set previous marker
                previous_marker = byte;

                decoder->reader->data_compressed_size -= 2;
                
                // Set segment byte count
                segment->data_compressed_size = decoder->reader->data_compressed_size - segment->data_compressed_index;
                
                // Start new segment in scan
                segment = &decoder->coder.segment[scan->segment_index + scan->segment_count];
                segment->scan_index = scan_index;
                segment->scan_segment_index = scan->segment_count;
                segment->data_compressed_index = decoder->reader->data_compressed_size;
                scan->segment_count++;    
            }
            // Check scan end
            else if ( byte == GPUJPEG_MARKER_EOI || byte == GPUJPEG_MARKER_SOS || (byte >= GPUJPEG_MARKER_APP0 && byte <= GPUJPEG_MARKER_APP15) ) {
                *image -= 2;
                decoder->reader->data_compressed_size -= 2;
                
                // Set segment byte count
                segment->data_compressed_size = decoder->reader->data_compressed_size - segment->data_compressed_index;
                
                // Add scan segment count to decoder segment count
                decoder->reader->segment_count += scan->segment_count;

                // Successfully read end of scan, so the result is OK
                result = 0;
                break;
            } else {
                fprintf(stderr, "[GPUJPEG] [Error] JPEG scan contains unexpected marker 0x%X!\n", byte);
                return -1;
            }
        }
    } while( *image < image_end );
    
    if ( result == -1) {
        fprintf(stderr, "[GPUJPEG] [Error] JPEG data unexpected ended while reading SOS marker!\n");
    }
    return result;
}

/**
 * Read scan content by segment info contained inside the JPEG stream
 *
 * @param decoder
 * @param image
 * @param image_end
 * @param scan
 * @param scan_index
 * @return 0 if succeeds, otherwise nonzero
 */
int
gpujpeg_reader_read_scan_content_by_segment_info(struct gpujpeg_decoder* decoder, uint8_t** image, uint8_t* image_end,
        struct gpujpeg_reader_scan* scan, int scan_index)
{
    // Calculate segment count
    int segment_count = decoder->reader->segment_info_size / 4 - 1;

    // Read first record from segment info, which means beginning of the first segment
    int scan_start = (decoder->reader->segment_info[0][0] << 24)
        + (decoder->reader->segment_info[0][1] << 16)
        + (decoder->reader->segment_info[0][2] << 8)
        + (decoder->reader->segment_info[0][3] << 0);

    // Read all segments from segment info
    for ( int segment_index = 0; segment_index < segment_count; segment_index++ ) {
        // Determine header index
        int header_index = ((segment_index + 1) * 4) / GPUJPEG_MAX_HEADER_SIZE;

        // Determine header data index
        int header_data_index = ((segment_index + 1) * 4) % GPUJPEG_MAX_HEADER_SIZE;

        // Determine segment ending in the scan
        int scan_end = (decoder->reader->segment_info[header_index][header_data_index + 0] << 24)
            + (decoder->reader->segment_info[header_index][header_data_index + 1] << 16)
            + (decoder->reader->segment_info[header_index][header_data_index + 2] << 8)
            + (decoder->reader->segment_info[header_index][header_data_index + 3] << 0);

        // Setup segment
        struct gpujpeg_segment* segment = &decoder->coder.segment[scan->segment_index + segment_index];
        segment->scan_index = scan_index;
        segment->scan_segment_index = segment_index;
        segment->data_compressed_index = decoder->reader->data_compressed_size + scan_start;
        segment->data_compressed_size = decoder->reader->data_compressed_size + scan_end - segment->data_compressed_index;

        // If segment is not last it contains restart marker at the end so remove it
        if ( (segment_index + 1) < segment_count ) {
            segment->data_compressed_size -= 2;
        }

        // Move info for next segment
        scan_start = scan_end;
    }

    // Set segment count in the scan
    scan->segment_count = segment_count;

    // Increase number of segment count in reader
    decoder->reader->segment_count += scan->segment_count;

    // Copy scan data to buffer
    memcpy(
        &decoder->coder.data_compressed[decoder->reader->data_compressed_size],
        *image,
        scan_start
    );
    *image += scan_start;
    decoder->reader->data_compressed_size += scan_start;

    // Reset segment info, for next scan it has to be loaded again from other header
    decoder->reader->segment_info_count = 0;
    decoder->reader->segment_info_size = 0;

    return 0;
}

/**
 * Read start of scan block from image
 *
 * @param decoder
 * @param image
 * @param image_end
 * @return 0 if succeeds, otherwise nonzero
 */
int
gpujpeg_reader_read_sos(struct gpujpeg_decoder* decoder, uint8_t** image, uint8_t* image_end)
{
    int length = (int)gpujpeg_reader_read_2byte(*image);
    length -= 2;

    int comp_count = (int)gpujpeg_reader_read_byte(*image);
    // Not interleaved mode
    if ( comp_count == 1 ) {
        decoder->reader->param.interleaved = 0;
    }
    // Interleaved mode
    else if ( comp_count == decoder->reader->param_image.comp_count ) {
        if ( decoder->reader->comp_count != 0 ) {
            fprintf(stderr, "[GPUJPEG] [Error] SOS marker component count %d is not supported for multiple scans!\n", comp_count);
            return -1;
        }
        decoder->reader->param.interleaved = 1;
    }
    // Unknown mode
    else {
        fprintf(stderr, "[GPUJPEG] [Error] SOS marker component count %d is not supported (should be 1 or equals to total component count)!\n", comp_count);
        return -1;
    }

    // We must init decoder before data is loaded into it
    if ( decoder->reader->comp_count == 0 ) {
        // Init decoder
        gpujpeg_decoder_init(decoder, &decoder->reader->param, &decoder->reader->param_image);
    }

    // Check maximum component count
    decoder->reader->comp_count += comp_count;
    if ( decoder->reader->comp_count > decoder->reader->param_image.comp_count ) {
        fprintf(stderr, "[GPUJPEG] [Error] SOS marker component count for all scans %d exceeds maximum component count %d!\n",
            decoder->reader->comp_count, decoder->reader->param_image.comp_count);
    }

    // Collect the component-spec parameters
    for ( int comp = 0; comp < comp_count; comp++ )
    {
        int index = (int)gpujpeg_reader_read_byte(*image);
        int table = (int)gpujpeg_reader_read_byte(*image);
        int table_dc = (table >> 4) & 15;
        int table_ac = table & 15;

        if ( index == 1 && (table_ac != 0 || table_dc != 0) ) {
            fprintf(stderr, "[GPUJPEG] [Error] SOS marker for Y should have huffman tables 0,0 but %d,%d was presented!\n", table_dc, table_ac);
            return -1;
        }
        if ( (index == 2 || index == 3) && (table_ac != 1 || table_dc != 1) ) {
            fprintf(stderr, "[GPUJPEG] [Error] SOS marker for Cb or Cr should have huffman tables 1,1 but %d,%d was presented!\n", table_dc, table_ac);
            return -1;
        }
    }

    // Collect the additional scan parameters Ss, Se, Ah/Al.
    int Ss = (int)gpujpeg_reader_read_byte(*image);
    int Se = (int)gpujpeg_reader_read_byte(*image);
    int Ax = (int)gpujpeg_reader_read_byte(*image);
    int Ah = (Ax >> 4) & 15;
    int Al = (Ax) & 15;

    // Check maximum scan count
    if ( decoder->reader->scan_count >= GPUJPEG_MAX_COMPONENT_COUNT ) {
        fprintf(stderr, "[GPUJPEG] [Error] SOS marker reached maximum number of scans (3)!\n");
        return -1;
    }

    int scan_index = decoder->reader->scan_count;

    // Get scan structure
    struct gpujpeg_reader_scan* scan = &decoder->reader->scan[scan_index];
    decoder->reader->scan_count++;
    // Scan segments begin at the end of previous scan segments or from zero index
    scan->segment_index = decoder->reader->segment_count;
    scan->segment_count = 0;

    // Read scan content
    if ( decoder->reader->segment_info_count > 0 ) {
        // Read scan content by segment info contained in special header
        if ( gpujpeg_reader_read_scan_content_by_segment_info(decoder, image, image_end, scan, scan_index) != 0 )
            return -1;
    } else {
        // Read scan content byte-by-byte
        if ( gpujpeg_reader_read_scan_content_by_parsing(decoder, image, image_end, scan, scan_index) != 0 )
            return -1;
    }

    return 0;
}

/** Documented at declaration */
int
gpujpeg_reader_read_image(struct gpujpeg_decoder* decoder, uint8_t* image, int image_size)
{
    // Setup reader and decoder
    decoder->reader->param = decoder->coder.param;
    decoder->reader->param_image = decoder->coder.param_image;
    decoder->reader->comp_count = 0;
    decoder->reader->scan_count = 0;
    decoder->reader->segment_count = 0;
    decoder->reader->data_compressed_size = 0;
    decoder->reader->segment_info_count = 0;
    decoder->reader->segment_info_size = 0;
    
    // Get image end
    uint8_t* image_end = image + image_size;
    
    // Check first SOI marker
    int marker_soi = gpujpeg_reader_read_marker(&image);
    if ( marker_soi != GPUJPEG_MARKER_SOI ) {
        fprintf(stderr, "[GPUJPEG] [Error] JPEG data should begin with SOI marker, but marker %s was found!\n", gpujpeg_marker_name((enum gpujpeg_marker_code)marker_soi));
        return -1;
    }
        
    int eoi_presented = 0;
    while ( eoi_presented == 0 ) {
        // Read marker
        int marker = gpujpeg_reader_read_marker(&image);
        if ( marker == -1 ) {
            return -1;
        }

        // Read more info according to the marker
        switch (marker) 
        {
        case GPUJPEG_MARKER_SEGMENT_INFO:
            if ( gpujpeg_reader_read_segment_info(decoder, &image) != 0 )
                return -1;
            break;

        case GPUJPEG_MARKER_APP0:
            if ( gpujpeg_reader_read_app0(&image) != 0 )
                return -1;
            break;
        case GPUJPEG_MARKER_APP1:
        case GPUJPEG_MARKER_APP2:
        case GPUJPEG_MARKER_APP3:
        case GPUJPEG_MARKER_APP4:
        case GPUJPEG_MARKER_APP5:
        case GPUJPEG_MARKER_APP6:
        case GPUJPEG_MARKER_APP7:
        case GPUJPEG_MARKER_APP8:
        case GPUJPEG_MARKER_APP9:
        case GPUJPEG_MARKER_APP10:
        case GPUJPEG_MARKER_APP11:
        case GPUJPEG_MARKER_APP12:
        //case GPUJPEG_MARKER_APP13:
        case GPUJPEG_MARKER_APP14:
        case GPUJPEG_MARKER_APP15:
            fprintf(stderr, "[GPUJPEG] [Warning] JPEG data contains not supported %s marker\n", gpujpeg_marker_name((enum gpujpeg_marker_code)marker));
            gpujpeg_reader_skip_marker_content(&image);
            break;
            
        case GPUJPEG_MARKER_DQT:
            if ( gpujpeg_reader_read_dqt(decoder, &image) != 0 )
                return -1;
            break;

        case GPUJPEG_MARKER_SOF0: 
            // Baseline
            if ( gpujpeg_reader_read_sof0(decoder, &image) != 0 )
                return -1;
            break;
        case GPUJPEG_MARKER_SOF1:
            // Extended sequential with Huffman coder
            fprintf(stderr, "[GPUJPEG] [Warning] Reading SOF1 as it was SOF0 marker (should work but verify it)!\n", gpujpeg_marker_name((enum gpujpeg_marker_code)marker));
            if ( gpujpeg_reader_read_sof0(decoder, &image) != 0 )
                return -1;
            break;
        case GPUJPEG_MARKER_SOF2:
            fprintf(stderr, "[GPUJPEG] [Error] Marker SOF2 (Progressive with Huffman coding) is not supported!");
            return -1;
        case GPUJPEG_MARKER_SOF3:
            fprintf(stderr, "[GPUJPEG] [Error] Marker SOF3 (Lossless with Huffman coding) is not supported!");
            return -1;
        case GPUJPEG_MARKER_SOF5:
            fprintf(stderr, "[GPUJPEG] [Error] Marker SOF5 (Differential sequential with Huffman coding) is not supported!");
            return -1;
        case GPUJPEG_MARKER_SOF6:
            fprintf(stderr, "[GPUJPEG] [Error] Marker SOF6 (Differential progressive with Huffman coding) is not supported!");
            return -1;
        case GPUJPEG_MARKER_SOF7:
            fprintf(stderr, "[GPUJPEG] [Error] Marker SOF7 (Extended lossless with Arithmetic coding) is not supported!");
            return -1;
        case GPUJPEG_MARKER_JPG:
            fprintf(stderr, "[GPUJPEG] [Error] Marker JPG (Reserved for JPEG extensions ) is not supported!");
            return -1;
        case GPUJPEG_MARKER_SOF10:
            fprintf(stderr, "[GPUJPEG] [Error] Marker SOF10 (Progressive with Arithmetic coding) is not supported!");
            return -1;
        case GPUJPEG_MARKER_SOF11:
            fprintf(stderr, "[GPUJPEG] [Error] Marker SOF11 (Lossless with Arithmetic coding) is not supported!");
            return -1;
        case GPUJPEG_MARKER_SOF13:
            fprintf(stderr, "[GPUJPEG] [Error] Marker SOF13 (Differential sequential with Arithmetic coding) is not supported!");
            return -1;
        case GPUJPEG_MARKER_SOF14:
            fprintf(stderr, "[GPUJPEG] [Error] Marker SOF14 (Differential progressive with Arithmetic coding) is not supported!");
            return -1;
        case GPUJPEG_MARKER_SOF15:
            fprintf(stderr, "[GPUJPEG] [Error] Marker SOF15 (Differential lossless with Arithmetic coding) is not supported!");
            return -1;
            
        case GPUJPEG_MARKER_DHT:
            if ( gpujpeg_reader_read_dht(decoder, &image) != 0 )
                return -1;
            break;
            
        case GPUJPEG_MARKER_DRI:
            if ( gpujpeg_reader_read_dri(decoder, &image) != 0 )
                return -1;
            break;

        case GPUJPEG_MARKER_SOS:
            if ( gpujpeg_reader_read_sos(decoder, &image, image_end) != 0 )
                return -1;
            break;
            
        case GPUJPEG_MARKER_EOI:
            eoi_presented = 1;
            break;

        case GPUJPEG_MARKER_COM:
            gpujpeg_reader_skip_marker_content(&image);
            break;

        case GPUJPEG_MARKER_DAC:
        case GPUJPEG_MARKER_DNL:
            fprintf(stderr, "[GPUJPEG] [Warning] JPEG data contains not supported %s marker\n", gpujpeg_marker_name((enum gpujpeg_marker_code)marker));
            gpujpeg_reader_skip_marker_content(&image);
            break;
            
        default:   
            fprintf(stderr, "[GPUJPEG] [Error] JPEG data contains not supported %s marker!\n", gpujpeg_marker_name((enum gpujpeg_marker_code)marker));
            gpujpeg_reader_skip_marker_content(&image);
            return -1;
        }
    }
    
    // Check EOI marker
    if ( eoi_presented == 0 ) {
        fprintf(stderr, "[GPUJPEG] [Error] JPEG data should end with EOI marker!\n");
        return -1;
    }
    
    // Set decoder parameters
    decoder->segment_count = decoder->reader->segment_count;
    decoder->data_compressed_size = decoder->reader->data_compressed_size;
    
    if ( decoder->segment_count > decoder->coder.segment_count ) {
        fprintf(stderr, "[GPUJPEG] [Error] Decoder can't decode image that has segment count %d (maximum segment count for specified parameters is %d)!\n",
            decoder->segment_count, decoder->coder.segment_count);
        return -1;
    }
    
    return 0;
}
