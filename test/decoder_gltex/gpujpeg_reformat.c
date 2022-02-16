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

#include "gpujpeg_reformat.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

#define GPUJPEG_BLOCK_SIZE           8
#define GPUJPEG_MAX_COMPONENT_COUNT  3
#define GPUJPEG_MAX_HEADER_COUNT     100

/// Maximum JPEG header size (MUST be divisible by 4!!!)
#define GPUJPEG_MAX_HEADER_SIZE      (65536 - 100)

/**
 * JPEG marker codes
 */
enum gpujpeg_marker_code {
    GPUJPEG_MARKER_SOF0  = 0xc0,
    GPUJPEG_MARKER_SOF1  = 0xc1,
    GPUJPEG_MARKER_RST0  = 0xd0,
    GPUJPEG_MARKER_RST1  = 0xd1,
    GPUJPEG_MARKER_RST2  = 0xd2,
    GPUJPEG_MARKER_RST3  = 0xd3,
    GPUJPEG_MARKER_RST4  = 0xd4,
    GPUJPEG_MARKER_RST5  = 0xd5,
    GPUJPEG_MARKER_RST6  = 0xd6,
    GPUJPEG_MARKER_RST7  = 0xd7,
    GPUJPEG_MARKER_SOI   = 0xd8,
    GPUJPEG_MARKER_EOI   = 0xd9,
    GPUJPEG_MARKER_SOS   = 0xda,
    GPUJPEG_MARKER_DRI   = 0xdd,
    GPUJPEG_MARKER_APP0  = 0xe0,
    GPUJPEG_MARKER_APP13 = 0xed,
    GPUJPEG_MARKER_APP15 = 0xef,
};

/// Sampling factor for color component in JPEG format
struct gpujpeg_component_sampling_factor
{
    uint8_t horizontal;
    uint8_t vertical;
};

//// Divide and round up
#define gpujpeg_div_and_round_up(value, div) \
    (((value % div) != 0) ? (value / div + 1) : (value / div))

/**
 * Read byte from image data
 *
 * @param image
 * @return byte
 */
#define gpujpeg_reformat_read_byte(image) \
    (uint8_t)(*(image)++)

/**
 * Read two-bytes from image data
 *
 * @param image
 * @return 2 bytes
 */
#define gpujpeg_reformat_read_2byte(image) \
    (uint16_t)(((*(image)) << 8) + (*((image) + 1))); \
    image += 2;

/**
 * Read marker from image data
 *
 * @param image
 * @return marker code or -1 if failed
 */
int
gpujpeg_reformat_read_marker(uint8_t** image)
{
    uint8_t byte = gpujpeg_reformat_read_byte(*image);
    if( byte != 0xFF ) {
        fprintf(stderr, "[GPUJPEG] [Error] Failed to read marker from JPEG data (0xFF was expected but 0x%X was presented)\n", byte);
        return -1;
    }
    int marker = gpujpeg_reformat_read_byte(*image);
    return marker;
}

/**
 * Skip marker content (read length and that much bytes - 2)
 *
 * @param image
 * @return void
 */
void
gpujpeg_reformat_skip_marker_content(uint8_t** image)
{
    int length = (int) gpujpeg_reformat_read_2byte(*image);

    *image += length - 2;
}

/**
 * Write one byte to file
 *
 * @param writer  Writer structure
 * @param value  Byte value to write
 * @return void
 */
#define gpujpeg_reformat_emit_byte(image, value) { \
    *image = (uint8_t)(value); \
    image++; }

/**
 * Write two bytes to file
 *
 * @param writer  Writer structure
 * @param value  Two-byte value to write
 * @return void
 */
#define gpujpeg_reformat_emit_2byte(image, value) { \
    *image = (uint8_t)(((value) >> 8) & 0xFF); \
    image++; \
    *image = (uint8_t)((value) & 0xFF); \
    image++; }

/**
 * Write marker to file
 *
 * @param writer  Writer structure
 * @oaran marker  Marker to write (JPEG_MARKER_...)
 * @return void
 */
#define gpujpeg_reformat_emit_marker(image, marker) { \
    *image = 0xFF;\
    image++; \
    *image = (uint8_t)(marker); \
    image++; }

struct gpujpeg_reformat_scan {
    /// Offset of the SOS marker in JPEG stream.
    int offset;

    /// Computed number of segments in scan.
    int segment_count;

    /// Parsed number of segments in scan.
    int segment_index;

    /// Buffer of offsets to all segments.
    int * segment_offset;
};

struct gpujpeg_rewriter
{
    /// Restart interval (0 means that restart interval is disabled and CPU huffman coder is used)
    int restart_interval;

    /// Flag which determines if interleaved format of JPEG stream should be used, "1" = only
    /// one scan which includes all color components (e.g. Y Cb Cr Y Cb Cr ...),
    /// or "0" = one scan for each color component (e.g. Y Y Y ..., Cb Cb Cb ..., Cr Cr Cr ...)
    int interleaved;

    /// Image data width
    int width;

    /// Image data height
    int height;

    /// Image data component count
    int comp_count;

    /// Sampling factors for each color component inside JPEG stream.
    struct gpujpeg_component_sampling_factor sampling_factor[GPUJPEG_MAX_COMPONENT_COUNT];

    /// Index of the next color component
    int comp_index;

    /// Info about scans.
    struct gpujpeg_reformat_scan scans[GPUJPEG_MAX_COMPONENT_COUNT];

    /// Number of scans.
    int scan_count;
};

/**
 * Read start of frame block from image
 *
 * @param param
 * @param param_image
 * @param image
 * @return 0 if succeeds, otherwise nonzero
 */
int
gpujpeg_reformat_read_sof0(struct gpujpeg_rewriter * rewriter, uint8_t** image)
{
    int length = (int) gpujpeg_reformat_read_2byte(*image);
    if (length < 6) {
        fprintf(stderr, "[GPUJPEG] [Error] SOF0 marker length should be greater than 6 but %d was presented!\n", length);
        return -1;
    }
    length -= 2;

    int precision = (int) gpujpeg_reformat_read_byte(*image);
    if ( precision != 8 ) {
        fprintf(stderr, "[GPUJPEG] [Error] SOF0 marker precision should be 8 but %d was presented!\n", precision);
        return -1;
    }

    rewriter->height = (int) gpujpeg_reformat_read_2byte(*image);
    rewriter->width = (int) gpujpeg_reformat_read_2byte(*image);
    rewriter->comp_count = (int) gpujpeg_reformat_read_byte(*image);
    length -= 6;

    for ( int comp = 0; comp < rewriter->comp_count; comp++ ) {
        int index = (int) gpujpeg_reformat_read_byte(*image);
        if ( index != (comp + 1) ) {
            fprintf(stderr, "[GPUJPEG] [Error] SOF0 marker component %d id should be %d but %d was presented!\n", comp, comp + 1, index);
            return -1;
        }

        int sampling = (int) gpujpeg_reformat_read_byte(*image);
        rewriter->sampling_factor[comp].horizontal = (sampling >> 4) & 15;
        rewriter->sampling_factor[comp].vertical = (sampling) & 15;

        int table_index = (int) gpujpeg_reformat_read_byte(*image);
        if (comp == 0 && table_index != 0) {
            fprintf(stderr, "[GPUJPEG] [Error] SOF0 marker component Y should have quantization table index 0 but %d was presented!\n", table_index);
            return -1;
        }
        if ((comp == 1 || comp == 2) && table_index != 1) {
            fprintf(stderr, "[GPUJPEG] [Error] SOF0 marker component Cb or Cr should have quantization table index 1 but %d was presented!\n", table_index);
            return -1;
        }
        length -= 3;
    }

    // Check length
    if (length > 0) {
        fprintf(stderr, "[GPUJPEG] [Warning] SOF0 marker contains %d more bytes than needed!\n", length);
        *image += length;
    }

    return 0;
}

/**
 * Read restart interval block from image
 *
 * @param param
 * @param image
 * @return 0 if succeeds, otherwise nonzero
 */
int
gpujpeg_reformat_read_dri(struct gpujpeg_rewriter * rewriter, uint8_t** image)
{
    int length = (int) gpujpeg_reformat_read_2byte(*image);
    if (length != 4) {
        fprintf(stderr, "[GPUJPEG] [Error] DRI marker length should be 4 but %d was presented!\n", length);
        return -1;
    }
    rewriter->restart_interval = gpujpeg_reformat_read_2byte(*image);
    return 0;
}

/**
 * Read start of scan block from image
 *
 * @param rewriter
 * @param image
 * @param image_end
 * @return 0 if succeeds, otherwise nonzero
 */
int
gpujpeg_reformat_read_sos(struct gpujpeg_rewriter * rewriter, uint8_t** image, uint8_t* image_end)
{
    int length = (int) gpujpeg_reformat_read_2byte(*image);
    length -= 2;

    int comp_count = (int) gpujpeg_reformat_read_byte(*image);
    // Not interleaved mode
    if (comp_count == 1) {
        rewriter->interleaved = 0;
    }
    // Interleaved mode
    else if (comp_count == rewriter->comp_count) {
        if (rewriter->comp_index != 0) {
            fprintf(stderr, "[GPUJPEG] [Error] SOS marker component count %d is not supported for multiple scans!\n", comp_count);
            return -1;
        }
        rewriter->interleaved = 1;
    }
    // Unknown mode
    else {
        fprintf(stderr, "[GPUJPEG] [Error] SOS marker component count %d is not supported (should be 1 or equals to total component count)!\n", comp_count);
        return -1;
    }

    // We must init decoder before data is loaded into it
    if ( rewriter->comp_index == 0 ) {
        // Compute number of segments
        struct gpujpeg_component_sampling_factor  sampling_factor;
        sampling_factor.horizontal = 0;
        sampling_factor.vertical = 0;
        if (rewriter->comp_count > GPUJPEG_MAX_COMPONENT_COUNT) {
            fprintf(stderr, "[GPUJPEG] [Error] Number of color components (%d) exceed maximum (%d)\n", rewriter->comp_count, GPUJPEG_MAX_COMPONENT_COUNT);
            return -1;
        }
        for (int comp = 0; comp < rewriter->comp_count; comp++) {
            // Sampling factors
            assert(rewriter->sampling_factor[comp].horizontal >= 1 && rewriter->sampling_factor[comp].horizontal <= 15);
            assert(rewriter->sampling_factor[comp].vertical >= 1 && rewriter->sampling_factor[comp].vertical <= 15);
            struct gpujpeg_component_sampling_factor component_sampling_factor = rewriter->sampling_factor[comp];
            if ( component_sampling_factor.horizontal > sampling_factor.horizontal ) {
                sampling_factor.horizontal = component_sampling_factor.horizontal;
            }
            if ( component_sampling_factor.vertical > sampling_factor.vertical ) {
                sampling_factor.vertical = component_sampling_factor.vertical;
            }

            // Set proper color component sizes in pixels based on sampling factors
            int width = ((rewriter->width + sampling_factor.horizontal - 1) / sampling_factor.horizontal) * sampling_factor.horizontal;
            int height = ((rewriter->height + sampling_factor.vertical - 1) / sampling_factor.vertical) * sampling_factor.vertical;
            int samp_factor_h = component_sampling_factor.horizontal;
            int samp_factor_v = component_sampling_factor.vertical;
            int component_width = (width * samp_factor_h) / sampling_factor.horizontal;
            int component_height = (height * samp_factor_v) / sampling_factor.vertical;

            // Compute component MCU size
            int component_mcu_size_x = GPUJPEG_BLOCK_SIZE;
            int component_mcu_size_y = GPUJPEG_BLOCK_SIZE;
            if (rewriter->interleaved == 1) {
                component_mcu_size_x *= samp_factor_h;
                component_mcu_size_y *= samp_factor_v;
            }

            // Compute allocated data size
            int component_data_width = gpujpeg_div_and_round_up(component_width, component_mcu_size_x) * component_mcu_size_x;
            int component_data_height = gpujpeg_div_and_round_up(component_height, component_mcu_size_y) * component_mcu_size_y;

            // Compute component MCU count
            int component_mcu_count_x = gpujpeg_div_and_round_up(component_data_width, component_mcu_size_x);
            int component_mcu_count_y = gpujpeg_div_and_round_up(component_data_height, component_mcu_size_y);
            int component_mcu_count = component_mcu_count_x * component_mcu_count_y;

            // Compute MCU count per segment
            int component_segment_mcu_count = rewriter->restart_interval;
            if (component_segment_mcu_count == 0) {
                // If restart interval is disabled, restart interval is equal MCU count
                component_segment_mcu_count = component_mcu_count;
            }

            // Calculate segment count
            int component_segment_count = gpujpeg_div_and_round_up(component_mcu_count, component_segment_mcu_count);
            if (rewriter->interleaved == 1) {
                // Only one scan exists
                rewriter->scans[0].segment_count = component_segment_count;
                rewriter->scans[0].segment_offset = (int *) malloc((component_segment_count + 1) * sizeof(int));
                break;
            }
            else {
                // Scan for each color component exists
                rewriter->scans[comp].segment_count = component_segment_count;
                rewriter->scans[comp].segment_offset = (int *) malloc((component_segment_count + 1) * sizeof(int));
            }
        }
    }

    // Check maximum component count
    rewriter->comp_index += comp_count;
    if (rewriter->comp_index > rewriter->comp_count) {
        fprintf(stderr, "[GPUJPEG] [Error] SOS marker component count for all scans %d exceeds maximum component count %d!\n",
            rewriter->comp_index, rewriter->comp_count);
    }

    // Collect the component-spec parameters
    for (int comp = 0; comp < comp_count; comp++) {
        int index = (int)gpujpeg_reformat_read_byte(*image);
        int table = (int)gpujpeg_reformat_read_byte(*image);
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
    int Ss = (int)gpujpeg_reformat_read_byte(*image);
    int Se = (int)gpujpeg_reformat_read_byte(*image);
    int Ax = (int)gpujpeg_reformat_read_byte(*image);
    int Ah = (Ax >> 4) & 15;
    int Al = (Ax) & 15;
    (void) Ss, (void) Se, (void) Ax, (void) Ah, (void) Al;

    // Check maximum scan count
    if ( rewriter->scan_count >= GPUJPEG_MAX_COMPONENT_COUNT ) {
        fprintf(stderr, "[GPUJPEG] [Error] SOS marker reached maximum number of scans (3)!\n");
        return -1;
    }

    struct gpujpeg_reformat_scan * scan = &rewriter->scans[rewriter->scan_count];
    rewriter->scan_count++;

    // TODO: get current offset
    size_t current_data_offset = 0;

    // First segment
    scan->segment_offset[scan->segment_index] = current_data_offset;
    scan->segment_index++;

    // Read scan data
    int result = -1;
    uint8_t byte = 0;
    uint8_t byte_previous = 0;
    uint8_t previous_marker = GPUJPEG_MARKER_RST0 - 1;
    do {
        byte_previous = byte;
        byte = gpujpeg_reformat_read_byte(*image);
        current_data_offset++;

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
                        byte = gpujpeg_reformat_read_byte(*image);
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

                // Next segment
                scan->segment_offset[scan->segment_index] = current_data_offset;
                scan->segment_index++;
            }
            // Check scan end
            else if ( byte == GPUJPEG_MARKER_EOI || byte == GPUJPEG_MARKER_SOS || (byte >= GPUJPEG_MARKER_APP0 && byte <= GPUJPEG_MARKER_APP15) ) {
                *image -= 2;
                current_data_offset -= 2;

                // Offset after the last segment
                scan->segment_offset[scan->segment_index] = current_data_offset;

                // Successfully read end of scan, so the result is OK
                result = 0;
                break;
            }
            else {
                fprintf(stderr, "[GPUJPEG] [Error] JPEG scan contains unexpected marker 0x%X!\n", byte);
                return -1;
            }
        }
    } while(*image < image_end);

    if (result == -1) {
        fprintf(stderr, "[GPUJPEG] [Error] JPEG data unexpected ended while reading SOS marker!\n");
    }
    return result;
}

int
gpujpeg_reformat_parse_image(uint8_t * image, int image_size, struct gpujpeg_rewriter * rewriter)
{
    uint8_t * image_begin = image;
    uint8_t * image_end = image + image_size;

    // Check first SOI marker
    int marker_soi = gpujpeg_reformat_read_marker(&image);
    if (marker_soi != GPUJPEG_MARKER_SOI) {
        fprintf(stderr, "[GPUJPEG] [Error] JPEG data should begin with SOI marker, but marker 0x%X was found!\n", marker_soi);
        return -1;
    }

    int eoi_presented = 0;
    while (eoi_presented == 0) {
        // Read marker
        int marker = gpujpeg_reformat_read_marker(&image);
        if (marker == -1) {
            return -1;
        }

        // Read more info according to the marker
        switch (marker)
        {
        case GPUJPEG_MARKER_SOF0: // Baseline
        case GPUJPEG_MARKER_SOF1: // Extended sequential with Huffman coder
        {
            if (0 != gpujpeg_reformat_read_sof0(rewriter, &image)) {
                return -1;
            }
            break;
        }
        case GPUJPEG_MARKER_DRI:
        {
            if (0 != gpujpeg_reformat_read_dri(rewriter, &image)) {
                return -1;
            }
            break;
        }
        case GPUJPEG_MARKER_SOS:
        {
            rewriter->scans[rewriter->scan_count].offset = image - image_begin - 2;
            if (0 != gpujpeg_reformat_read_sos(rewriter, &image, image_end)) {
                return -1;
            }
            break;
        }
        case GPUJPEG_MARKER_EOI:
        {
            eoi_presented = 1;
            break;
        }
        default:
            gpujpeg_reformat_skip_marker_content(&image);
            break;
        }
    }
    return 0;
}

int
gpujpeg_reformat(uint8_t * in_image, int in_image_size, uint8_t ** out_image, int * out_image_size)
{
    // Parse input image
    struct gpujpeg_rewriter rewriter;
    memset(&rewriter, 0, sizeof(struct gpujpeg_rewriter));
    gpujpeg_reformat_parse_image(in_image, in_image_size, &rewriter);

    // Check scans and compute maximum size of new headers
    int new_header_size = GPUJPEG_MAX_HEADER_COUNT * (2 + 4);
    for (int scan_index = 0; scan_index < rewriter.scan_count; scan_index++) {
        struct gpujpeg_reformat_scan * scan = &rewriter.scans[scan_index];
        if (scan->segment_count != scan->segment_index) {
            fprintf(stderr, "[GPUJPEG] [Error] Number of parsed segments %d in scan #%d doesn't equal the expected number %d!\n", scan->segment_index, scan_index, scan->segment_count);
            return -1;
        }
        new_header_size += scan->segment_count * 4;
    }

    // Allocate new JPEG stream    
    *out_image = (uint8_t *) malloc(in_image_size + new_header_size);
    uint8_t * image = *out_image;

    // Copy headers until first SOS
    memcpy(image, in_image, rewriter.scans[0].offset);
    image += rewriter.scans[0].offset;

    // Write new headers for each scan
    for (int scan_index = 0; scan_index < rewriter.scan_count; scan_index++) {
        struct gpujpeg_reformat_scan * scan = &rewriter.scans[scan_index];
        int data_size = (scan->segment_count + 1) * 4;
        int segment_index = 0;

        // Emit headers (each header can have data size of only 2^16)        
        while (data_size > 0) {
            // Determine current header size
            int header_size = data_size;
            if ( header_size > GPUJPEG_MAX_HEADER_SIZE ) {
                header_size = GPUJPEG_MAX_HEADER_SIZE;
            }
            data_size -= header_size;

            // Header marker
            gpujpeg_reformat_emit_marker(image, GPUJPEG_MARKER_APP13);

            // Write custom application header
            gpujpeg_reformat_emit_2byte(image, 3 + header_size);
            gpujpeg_reformat_emit_byte(image, scan_index);

            // Write segments
            while (header_size > 0) {
                int segment_offset = scan->segment_offset[segment_index];
                gpujpeg_reformat_emit_byte(image, (uint8_t)(((segment_offset) >> 24) & 0xFF));
                gpujpeg_reformat_emit_byte(image, (uint8_t)(((segment_offset) >> 16) & 0xFF));
                gpujpeg_reformat_emit_byte(image, (uint8_t)(((segment_offset) >> 8) & 0xFF));
                gpujpeg_reformat_emit_byte(image, (uint8_t)(((segment_offset) >> 0) & 0xFF));
                segment_index++;
                header_size -= 4;
            }
        }

        // Write scan
        int scan_size = (((scan_index + 1) < rewriter.scan_count) ? rewriter.scans[scan_index + 1].offset : in_image_size) - scan->offset;
        memcpy(image, in_image + scan->offset, scan_size);
        image += scan_size;
    }
    *out_image_size = image - *out_image;

    // Cleanup
    for (int scan_index = 0; scan_index < rewriter.scan_count; scan_index++) {
        struct gpujpeg_reformat_scan * scan = &rewriter.scans[scan_index];
        if (scan->segment_offset != NULL) {
            free(scan->segment_offset);
        }
    }

    return 0;
}
