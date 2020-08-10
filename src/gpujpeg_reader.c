/**
 * @file
 * Copyright (c) 2011-2020, CESNET z.s.p.o
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

#include <string.h>
#include <libgpujpeg/gpujpeg_decoder.h>
#include "gpujpeg_decoder_internal.h"
#include "gpujpeg_marker.h"
#include "gpujpeg_reader.h"
#include "gpujpeg_util.h"

/* Documented at declaration */
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

/* Documented at declaration */
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
 * @param length  length field of the marker
 */
int gpujpeg_reader_read_jfif(uint8_t** image, int length)
{
    int version_major = gpujpeg_reader_read_byte(*image);
    int version_minor = gpujpeg_reader_read_byte(*image);
    if ( version_major != 1 || ( version_minor < 0 || version_minor > 2) ) {
        fprintf(stderr, "[GPUJPEG] [Error] JFIF marker version should be 1.00 to 1.02 but %d.%02d was presented!\n", version_major, version_minor);
        return -1;
    }

    int pixel_units = gpujpeg_reader_read_byte(*image);
    int pixel_xdpu = gpujpeg_reader_read_2byte(*image);
    int pixel_ydpu = gpujpeg_reader_read_2byte(*image);
    int thumbnail_width = gpujpeg_reader_read_byte(*image);
    int thumbnail_height = gpujpeg_reader_read_byte(*image);

    int thumbnail_length = thumbnail_width * thumbnail_height * 3;
    int expected_length = 16 + thumbnail_length;
    if ( length != expected_length ) {
        fprintf(stderr, "[GPUJPEG] [Warning] JFIF marker length should be %d (with thumbnail %dx%d) but %d was presented!\n",
                expected_length, thumbnail_width, thumbnail_height, length);
    }

    *image += length - 16;
    return 0;
}

/**
 * @param length  length field of the marker
 */
static int gpujpeg_reader_skip_jfxx(uint8_t** image, int length)
{
    *image += length - 7;

    return 0;
}

/**
 * Read segment info for following scan
 *
 * @param decoder
 * @param image
 * @param length length of the marker
 * @return 0 if succeeds, otherwise nonzero
 */
static int
gpujpeg_reader_read_segment_info(struct gpujpeg_decoder* decoder, uint8_t** image, int length)
{
    int scan_index = (int)gpujpeg_reader_read_byte(*image);
    int data_size = length - 3;
    if ( scan_index != decoder->reader->scan_count ) {
        fprintf(stderr, "[GPUJPEG] [Warning] %s marker (segment info) scan index should be %d but %d was presented!"
                " (marker not a segment info?)\n",
                gpujpeg_marker_name(GPUJPEG_MARKER_SEGMENT_INFO), decoder->reader->scan_count, scan_index);
        *image += data_size;
        return -1;
    }

    decoder->reader->segment_info[decoder->reader->segment_info_count] = *image;
    decoder->reader->segment_info_count++;
    decoder->reader->segment_info_size += data_size;

    *image += data_size;

    return 0;
}

/**
 * Read application info block from image
 *
 * @param image
 * @return 0 if succeeds, otherwise nonzero
 */
int
gpujpeg_reader_read_app0(uint8_t** image)
{
    int length = (int)gpujpeg_reader_read_2byte(*image);

    char marker[5];
    marker[0] = gpujpeg_reader_read_byte(*image);
    marker[1] = gpujpeg_reader_read_byte(*image);
    marker[2] = gpujpeg_reader_read_byte(*image);
    marker[3] = gpujpeg_reader_read_byte(*image);
    marker[4] = gpujpeg_reader_read_byte(*image);
    if ( strcmp(marker, "JFIF") == 0 ) {
        return gpujpeg_reader_read_jfif(image, length);
    } else if ( strcmp(marker, "JFXX") == 0 ) {
        return gpujpeg_reader_skip_jfxx(image, length);
    } else {
        fprintf(stderr, "[GPUJPEG] [Error] APP0 marker identifier should be 'JFIF' or 'JFXX' but '%s' was presented!\n", marker);
        return -1;
    }
}

/**
 * Read Adobe APP13 marker (used as a segment info by GPUJPEG)
 *
 * @param decoder decoder state
 * @param image   JPEG data
 * @return        0 if succeeds, otherwise nonzero
 *
 * @todo
 * Segment info should have an identifying header like other application markers
 * (see http://www.ozhiker.com/electronics/pjmt/jpeg_info/app_segments.html)
 */
static int
gpujpeg_reader_read_app13(struct gpujpeg_decoder* decoder, uint8_t** image)
{
    int length = (int)gpujpeg_reader_read_2byte(*image);
    if ( length <= 3 ) {
        fprintf(stderr, "[GPUJPEG] [Error] APP13 marker (segment info) length should be greater than 3 but %d was presented!\n",
                length);
        return -1;
    }

    char *ignored_hdrs[] = { "Adobe_Photoshop2.5", "Photoshop 3.0", "Adobe_CM" };
    for (size_t i = 0; i < sizeof ignored_hdrs / sizeof ignored_hdrs[0]; ++i) {
        if (length >= 2 + sizeof ignored_hdrs[i] - 1 && memcmp((char *) *image, ignored_hdrs[i], sizeof ignored_hdrs[i] - 1) == 0) {
            fprintf(stderr, "[GPUJPEG] [Warning] Skipping unsupported %s APP13 marker!\n", ignored_hdrs[i]);
            *image += length - 2;
            return 0;
        }
    }

    // forward compatibility - expect future GPUJPEG version to tag the segment info marker,
    // which is currently not written
    char gpujpeg_str[] = "GPUJPEG"; // with terminating '\0'
    if (length >= 2 + sizeof gpujpeg_str && memcmp((char *) image, gpujpeg_str, sizeof gpujpeg_str) == 0) {
        *image += sizeof gpujpeg_str;
        length -= sizeof gpujpeg_str;
        return gpujpeg_reader_read_segment_info(decoder, image, length);
    }
    // suppose segment info marker - current GPUJPEG doesn't tag it
    gpujpeg_reader_read_segment_info(decoder, image, length); // ignore return code since we are
                                                              // not sure if it is ours
    return 0;
}

/**
 * Read Adobe APP14 marker (used for RGB images by GPUJPEG)
 *
 * Obtains colorspace from APP14.
 *
 * @param decoder decoder state
 * @param image   JPEG data
 * @return        0 if succeeds, otherwise nonzero
 */
static int
gpujpeg_reader_read_app14(uint8_t** image, enum gpujpeg_color_space *color_space)
{
    int length = gpujpeg_reader_read_2byte(*image);
    if (length != 14) {
        fprintf(stderr, "[GPUJPEG] [Error] APP14 marker length should be 14 but %d was presented!\n", length);
        return -1;
    }

    char adobe[6] = "";
    adobe[0] = gpujpeg_reader_read_byte(*image);
    adobe[1] = gpujpeg_reader_read_byte(*image);
    adobe[2] = gpujpeg_reader_read_byte(*image);
    adobe[3] = gpujpeg_reader_read_byte(*image);
    adobe[4] = gpujpeg_reader_read_byte(*image);
    if (strcmp(adobe, "Adobe") != 0) {
        fprintf(stderr, "[GPUJPEG] [Error] APP14 marker identifier should be 'Adobe' but '%s' was presented!\n", adobe);
        return -1;
    }

    int version = gpujpeg_reader_read_2byte(*image);
    int flags0 = gpujpeg_reader_read_2byte(*image);
    int flags1 = gpujpeg_reader_read_2byte(*image);
    int color_transform = gpujpeg_reader_read_byte(*image);

    if (color_transform == 0) {
        *color_space = GPUJPEG_RGB;
    } else if (color_transform == 1) {
        *color_space = GPUJPEG_YCBCR_BT601_256LVLS;
    } else if (color_transform == 2) {
        fprintf(stderr, "[GPUJPEG] [Error] Unsupported YCCK color transformation was presented!\n");
        return -1;
    } else {
        fprintf(stderr, "[GPUJPEG] [Error] Unsupported color transformation value '%d' was presented in APP14 marker!\n", color_transform);
        return -1;
    }

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
        int Pq = index >> 4; // precision - 0 for 8-bit, 1 for 16-bit
        int Tq = index & 0xF; // destination (0-3)
        if (Pq != 0 && Pq != 1) {
            fprintf(stderr, "[GPUJPEG] [Error] DQT marker Pq should be 0 or 1 (8 or 16-bit) but %d was presented!\n", Pq);
            return -1;
        }
        if (Pq != 0) {
            fprintf(stderr, "[GPUJPEG] [Error] Unsupported DQT Pq 1 (16-bit table) presented!\n");
            return -1;
        }
        table = &decoder->table_quantization[Tq];

        for ( int i = 0; i < 64; i++ ) {
            table->table_raw[i] = gpujpeg_reader_read_byte(*image);
        }
        // Prepare quantization table for read raw table
        gpujpeg_table_quantization_decoder_compute(table);
    }
    return 0;
}

/**
 * Return component ID that matches given index and color space.
 */
static uint8_t gpujpeg_reader_get_component_id(int index, enum gpujpeg_color_space color_space) {
    if (color_space == GPUJPEG_RGB) {
            assert(index < 3);
            static const uint8_t rgb_ids[3] = { 'R', 'G', 'B' };
            return rgb_ids[index];
    } else {
            return index + 1;
    }
}

/**
 * Read start of frame block from image
 *
 * @param image
 * @return 0 if succeeds, otherwise nonzero
 */
static int
gpujpeg_reader_read_sof0(struct gpujpeg_parameters * param, struct gpujpeg_image_parameters * param_image, int quant_map[4], uint8_t comp_id[4], uint8_t** image)
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

    param_image->height = (int)gpujpeg_reader_read_2byte(*image);
    param_image->width = (int)gpujpeg_reader_read_2byte(*image);
    param_image->comp_count = (int)gpujpeg_reader_read_byte(*image);
    length -= 6;

    for ( int comp = 0; comp < param_image->comp_count; comp++ ) {
        int id = (int)gpujpeg_reader_read_byte(*image);
        int expected_id = gpujpeg_reader_get_component_id(comp, param->color_space_internal);
        if ( id != expected_id ) {
            fprintf(stderr, "[GPUJPEG] [Error] SOF0 marker component %d id should be %d but %d was presented!\n", comp, expected_id, id);
            return -1;
        }
        comp_id[comp] = id;

        int sampling = (int)gpujpeg_reader_read_byte(*image);
        param->sampling_factor[comp].horizontal = (sampling >> 4) & 15;
        param->sampling_factor[comp].vertical = (sampling) & 15;

        int table_index = (int)gpujpeg_reader_read_byte(*image);
        if ( table_index < 0 || table_index > 3 ) {
            fprintf(stderr, "[GPUJPEG] [Error] SOF0 marker contains unexpected quantization table index %d!\n", table_index);
            return -1;
        }
        quant_map[comp] = table_index;
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
        int Tc = index >> 4; // class - 0 = DC, 1 = AC
        int Th = index & 0xF; // destination (0-1 baseline, 0-3 extended), usually 0 = luminance, 1 = chrominance
        if (Tc != 0 && Tc != 1) {
            fprintf(stderr, "[GPUJPEG] [Error] DHT marker Tc should be 0 or 1 but %d was presented!\n", Tc);
            return -1;
        }
        table = &decoder->table_huffman[Th][Tc];
        d_table = decoder->d_table_huffman[Th][Tc];
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
                fprintf(stderr, "[GPUJPEG] [Error] DHT marker unexpected end when reading bit counts!\n");
                return -1;
            }
        }

        // Read in huffval
        for ( int i = 0; i < count; i++ ){
            table->huffval[i] = gpujpeg_reader_read_byte(*image);
            if ( length > 0 ) {
                length--;
            } else {
                fprintf(stderr, "[GPUJPEG] [Error] DHT marker unexpected end when reading huffman values!\n");
                return -1;
            }
        }
        // Compute huffman table for read values
        gpujpeg_table_huffman_decoder_compute(table);

        // Copy table to device memory
        cudaMemcpyAsync(d_table, table, sizeof(struct gpujpeg_table_huffman_decoder), cudaMemcpyHostToDevice, decoder->stream);
        gpujpeg_cuda_check_error("Decoder copy huffman table ", return -1);
    }
    return 0;
}

/**
 * Read restart interval block from image
 *
 * @param decoder
 * @param image
 * @retval  0 if succeeds
 * @retval -3 on restart interval redefinition
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

    if ( decoder->reader->param.restart_interval != 0
            && decoder->reader->param.restart_interval != restart_interval ) {
        fprintf(stderr, "[GPUJPEG] [Error] DRI marker can't redefine restart interval (%d to %d)!\n",
                decoder->reader->param.restart_interval, restart_interval );
        fprintf(stderr, "This may be caused when more DRI markers are presented which is not supported!\n");
        return GPUJPEG_ERR_RESTART_CHANGE;
    }

    decoder->reader->param.restart_interval = restart_interval;

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
    size_t data_compressed_offset = decoder->reader->data_compressed_size;

    // Get first segment in scan
    uint8_t * segment_data_start = *image;
    struct gpujpeg_segment* segment = &decoder->coder.segment[scan->segment_index];
    segment->scan_index = scan_index;
    segment->scan_segment_index = scan->segment_count;
    segment->data_compressed_index = data_compressed_offset;
    scan->segment_count++;

    // Read scan data
    int result = -1;
    uint8_t byte = 0;
    uint8_t byte_previous = 0;
    uint8_t previous_marker = GPUJPEG_MARKER_RST0 - 1;
    do {
        byte_previous = byte;
        byte = gpujpeg_reader_read_byte(*image);
        data_compressed_offset++;

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
                                fprintf(stderr, "[GPUJPEG] [Recovery] Skipping %d bytes of data until marker 0x%X was found!\n", skip_count, expected_marker);
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

                // Set segment byte count
                data_compressed_offset -= 2;
                segment->data_compressed_size = data_compressed_offset - segment->data_compressed_index;
                memcpy(&decoder->coder.data_compressed[segment->data_compressed_index], segment_data_start, segment->data_compressed_size);

                // Start new segment in scan
                segment_data_start = *image;
                segment = &decoder->coder.segment[scan->segment_index + scan->segment_count];
                segment->scan_index = scan_index;
                segment->scan_segment_index = scan->segment_count;
                segment->data_compressed_index = data_compressed_offset;
                scan->segment_count++;
            }
            // Check scan end
            else if ( byte == GPUJPEG_MARKER_EOI || byte == GPUJPEG_MARKER_SOS || (byte >= GPUJPEG_MARKER_APP0 && byte <= GPUJPEG_MARKER_APP15) ) {
                *image -= 2;

                // Set segment byte count
                data_compressed_offset -= 2;
                segment->data_compressed_size = data_compressed_offset - segment->data_compressed_index;
                memcpy(&decoder->coder.data_compressed[segment->data_compressed_index], segment_data_start, segment->data_compressed_size);

                if ( segment->data_compressed_size == 0 ) { // skip FFMPEG empty segments after last RST before EOF (FF bug #8412)
                    fprintf(stderr, "[GPUJPEG] [Warning] Empty segment detected!\n");
                    scan->segment_count -= 1;
                }

                // Add scan segment count to decoder segment count
                decoder->reader->segment_count += scan->segment_count;

                // Successfully read end of scan, so the result is OK
                result = 0;
                break;
            }
            else {
                fprintf(stderr, "[GPUJPEG] [Error] JPEG scan contains unexpected marker 0x%X!\n", byte);
                return -1;
            }
        }
    } while( *image < image_end );

    decoder->reader->data_compressed_size = data_compressed_offset;

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
        if ( gpujpeg_decoder_init(decoder, &decoder->reader->param, &decoder->reader->param_image) != 0 ) {
            return -1;
        }
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
        int comp_id = (int)gpujpeg_reader_read_byte(*image);
        int table = (int)gpujpeg_reader_read_byte(*image);
        int table_dc = (table >> 4) & 15;
        int table_ac = table & 15;

        int component_index = -1;
        for ( int i = 0; i < decoder->reader->param_image.comp_count; ++i ) {
            if (decoder->comp_id[i] == comp_id) {
                component_index = i;
                break;
            }
        }
        if ( component_index == -1 ) {
            fprintf(stderr, "[GPUJPEG] [Error] Unexpected component ID '%d' present SOS marker (not defined by SOF marker)!\n", comp_id);
            return -1;
        }

        decoder->comp_table_huffman_map[component_index][GPUJPEG_HUFFMAN_DC] = table_dc;
        decoder->comp_table_huffman_map[component_index][GPUJPEG_HUFFMAN_AC] = table_ac;
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

/* Documented at declaration */
int
gpujpeg_reader_read_image(struct gpujpeg_decoder* decoder, uint8_t* image, int image_size)
{
    int rc;
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
        case GPUJPEG_MARKER_APP0:
            if ( gpujpeg_reader_read_app0(&image) != 0 )
                return -1;
            break;
        case GPUJPEG_MARKER_APP13:
            if ( gpujpeg_reader_read_app13(decoder, &image) != 0 )
                return -1;
            break;
        case GPUJPEG_MARKER_APP14:
            if ( gpujpeg_reader_read_app14(&image, &decoder->reader->param.color_space_internal) != 0 )
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
            if ( gpujpeg_reader_read_sof0(&decoder->reader->param, &decoder->reader->param_image, decoder->comp_table_quantization_map,
                       decoder->comp_id, &image) != 0 )
                return -1;
            break;
        case GPUJPEG_MARKER_SOF1:
            // Extended sequential with Huffman coder
            fprintf(stderr, "[GPUJPEG] [Warning] Reading SOF1 as it was SOF0 marker (should work but verify it)!\n");
            if ( gpujpeg_reader_read_sof0(&decoder->reader->param, &decoder->reader->param_image, decoder->comp_table_quantization_map,
                        decoder->comp_id, &image) != 0 )
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
            rc = gpujpeg_reader_read_dri(decoder, &image);
            if ( rc != 0 )
                return rc;
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

/* Documented at declaration */
int
gpujpeg_reader_get_image_info(uint8_t *image, int image_size, struct gpujpeg_image_parameters *param_image, int *segment_count)
{
    struct gpujpeg_parameters param = {0};
    int segments = 0;
    int interleaved = 0;
    int unused[4];
    uint8_t unused2[4];

    // Check first SOI marker
    int marker_soi = gpujpeg_reader_read_marker(&image);
    if (marker_soi != GPUJPEG_MARKER_SOI) {
        fprintf(stderr, "[GPUJPEG] [Error] JPEG data should begin with SOI marker, but marker %s was found!\n", gpujpeg_marker_name((enum gpujpeg_marker_code)marker_soi));
        return -1;
    }

    int eoi_presented = 0;
    while (eoi_presented == 0) {
        // Read marker
        int marker = gpujpeg_reader_read_marker(&image);
        if (marker == -1) {
            return -1;
        }

        // Read more info according to the marker
        switch (marker)
        {
        case GPUJPEG_MARKER_APP0:
        {
            if (gpujpeg_reader_read_app0(&image) == 0) {
                // if the marker defines a valid JFIF, it is YCbCr (CCIR 601-256 levels)
                param_image->color_space = GPUJPEG_YCBCR_BT601_256LVLS;
            } else {
                return -1;
            }
            break;
        }
        case GPUJPEG_MARKER_APP14:
        {
            if (gpujpeg_reader_read_app14(&image, &param_image->color_space) != 0) {
                return -1;
            }
            break;
        }
        case GPUJPEG_MARKER_SOF0: // Baseline
        case GPUJPEG_MARKER_SOF1: // Extended sequential with Huffman coder
        {
            param.color_space_internal = param_image->color_space;
            if (gpujpeg_reader_read_sof0(&param, param_image, unused, unused2, &image) != 0) {
                return -1;
            }
            break;
        }
        case GPUJPEG_MARKER_SOF2:
        case GPUJPEG_MARKER_SOF3:
        case GPUJPEG_MARKER_SOF5:
        case GPUJPEG_MARKER_SOF6:
        case GPUJPEG_MARKER_SOF7:
        case GPUJPEG_MARKER_SOF9:
        case GPUJPEG_MARKER_SOF10:
        case GPUJPEG_MARKER_SOF11:
        case GPUJPEG_MARKER_SOF13:
        case GPUJPEG_MARKER_SOF14:
        case GPUJPEG_MARKER_SOF15:
        {
            fprintf(stderr, "Unsupported encoding process!\n");
            return -1;
        }
        case GPUJPEG_MARKER_RST0:
        case GPUJPEG_MARKER_RST1:
        case GPUJPEG_MARKER_RST2:
        case GPUJPEG_MARKER_RST3:
        case GPUJPEG_MARKER_RST4:
        case GPUJPEG_MARKER_RST5:
        case GPUJPEG_MARKER_RST6:
        case GPUJPEG_MARKER_RST7:
        case GPUJPEG_MARKER_SOS:
        {
            segments++;
            if (marker == GPUJPEG_MARKER_SOS) {
                int length = (int) gpujpeg_reader_read_2byte(image); // length
                assert(length > 3);
                int comp_count = (int) gpujpeg_reader_read_byte(image); // comp count in the segment
                if (comp_count > 1) {
                    interleaved = 1;
                }
                if (segment_count == NULL) { // if not counting segments, we can skip the rest
                    eoi_presented = 1;
                }
            }
            while (*image != 0xff || (*image == 0xff && image[1] == 0x00)) { if (*image == 0xff) image++; image++; }
            break;
        }

        case GPUJPEG_MARKER_EOI:
        {
            eoi_presented = 1;
            break;
        }
        default:
            gpujpeg_reader_skip_marker_content(&image);
            break;
        }
    }

    if (segment_count != NULL) {
        *segment_count = segments;
    }

    param_image->pixel_format = GPUJPEG_PIXFMT_NONE;
    if (param_image->comp_count == 1) {
        param_image->pixel_format = GPUJPEG_U8;
    } else if (param_image->comp_count == 3
            && param.sampling_factor[1].horizontal == 1 && param.sampling_factor[1].vertical == 1
            && param.sampling_factor[2].horizontal == 1 && param.sampling_factor[2].vertical == 1) {
        int sum = interleaved << 16 | param.sampling_factor[0].horizontal << 8 |  param.sampling_factor[0].vertical;
        switch (sum) {
        case 1<<16 | 1<<8 | 1: param_image->pixel_format = GPUJPEG_444_U8_P012; break;
        case 0<<16 | 1<<8 | 1: param_image->pixel_format = GPUJPEG_444_U8_P0P1P2; break;
        case 1<<16 | 2<<8 | 1: param_image->pixel_format = GPUJPEG_422_U8_P1020; break;
        case 0<<16 | 2<<8 | 1: param_image->pixel_format = GPUJPEG_422_U8_P0P1P2; break;
        case 1<<16 | 2<<8 | 2: // we have only one pixfmt for 420, so use for both
        case 0<<16 | 2<<8 | 2: param_image->pixel_format = GPUJPEG_420_U8_P0P1P2; break;
        }
    }

    return 0;
}

/* vim: set expandtab sw=4: */
