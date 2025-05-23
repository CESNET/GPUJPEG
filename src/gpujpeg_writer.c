/**
 * @file
 * Copyright (c) 2011-2025, CESNET z.s.p.o
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

#include "../libgpujpeg/gpujpeg_encoder.h"
#include "gpujpeg_encoder_internal.h"
#include "gpujpeg_marker.h"
#include "gpujpeg_util.h"
#include "gpujpeg_writer.h"

#if defined _MSC_VER && _MSC_VER < 1900 // VS 2015
#define snprintf _snprintf
#endif

/* Documented at declaration */
struct gpujpeg_writer*
gpujpeg_writer_create(void)
{
    struct gpujpeg_writer* writer = (struct gpujpeg_writer*) malloc(sizeof(struct gpujpeg_writer));
    if (writer == NULL) {
        return NULL;
    }

    writer->buffer_allocated_size = 0;
    writer->buffer = NULL;

    return writer;
}

static void free_buffer(struct gpujpeg_writer *writer) {
    if ( writer->buffer == NULL ) {
        return;
    }
    if ( writer->buffer_pinned ) {
        cudaFreeHost(writer->buffer);
    }
    else {
        free(writer->buffer);
    }
    writer->buffer = NULL;
}

/* Documented at declaration */
int
gpujpeg_writer_init(struct gpujpeg_writer* writer, int comp_count, struct gpujpeg_image_parameters* param_image)
{
    // Allocate output buffer
    size_t buffer_size = 1000;
    buffer_size += (size_t) param_image->width * param_image->height * comp_count * 2;

    if (buffer_size > writer->buffer_allocated_size) {
        writer->buffer_allocated_size = 0;
        free_buffer(writer);
        if ( writer->buffer_pinned ) {
            cudaMallocHost((void**)&writer->buffer, buffer_size * sizeof(uint8_t));
        }
        else {
            writer->buffer = (uint8_t*)malloc(buffer_size * sizeof(uint8_t));
        }
        if (writer->buffer == NULL) {
            return -1;
        }
        writer->buffer_allocated_size = buffer_size;
    }
    writer->buffer_current = NULL;
    writer->segment_info_count = 0;
    writer->segment_info_index = 0;
    writer->segment_info_position = 0;
    return 0;
}

/* Documented at declaration */
int
gpujpeg_writer_destroy(struct gpujpeg_writer* writer)
{
    assert(writer != NULL);
    free_buffer(writer);
    free(writer);
    return 0;
}

/**
 * Write SOI
 *
 * @param writer  Writer structure
 * @return void
 */
void
gpujpeg_writer_write_soi(struct gpujpeg_writer* writer)
{
    gpujpeg_writer_emit_marker(writer, GPUJPEG_MARKER_SOI);
}

/**
 * Write APP0 block
 *
 * @param writer  Writer structure
 * @return void
 */
void gpujpeg_writer_write_app0(struct gpujpeg_writer* writer)
{
    // Length of APP0 block    (2 bytes)
    // Block ID                (4 bytes - ASCII "JFIF")
    // Zero byte               (1 byte to terminate the ID string)
    // Version Major, Minor    (2 bytes - 0x01, 0x01)
    // Units                   (1 byte - 0x00 = none, 0x01 = inch, 0x02 = cm)
    // Xdpu                    (2 bytes - dots per unit horizontal)
    // Ydpu                    (2 bytes - dots per unit vertical)
    // Thumbnail X size        (1 byte)
    // Thumbnail Y size        (1 byte)
    gpujpeg_writer_emit_marker(writer, GPUJPEG_MARKER_APP0);

    // Length
    gpujpeg_writer_emit_2byte(writer, 2 + 4 + 1 + 2 + 1 + 2 + 2 + 1 + 1);

    // Identifier: ASCII "JFIF"
    gpujpeg_writer_emit_byte(writer, 0x4A);
    gpujpeg_writer_emit_byte(writer, 0x46);
    gpujpeg_writer_emit_byte(writer, 0x49);
    gpujpeg_writer_emit_byte(writer, 0x46);
    gpujpeg_writer_emit_byte(writer, 0);

    // We currently emit version code 1.01 since we use no 1.02 features.
    // This may avoid complaints from some older decoders.
    // Major version
    gpujpeg_writer_emit_byte(writer, 1);
    // Minor version
    gpujpeg_writer_emit_byte(writer, 1);
    // Pixel size information
    gpujpeg_writer_emit_byte(writer, 1);
    gpujpeg_writer_emit_2byte(writer, 300);
    gpujpeg_writer_emit_2byte(writer, 300);
    // No thumbnail image
    gpujpeg_writer_emit_byte(writer, 0);
    gpujpeg_writer_emit_byte(writer, 0);
}

static void
gpujpeg_writer_write_app8(struct gpujpeg_writer* writer)
{
    gpujpeg_writer_emit_marker(writer, GPUJPEG_MARKER_APP8);
}

/**
 * Write SPIFF header block
 *
 * This marker is used for color spaces that neither JFIF nor Adobe APP14 cannot represent
 *
 * @param writer  Writer structure
 */
static void gpujpeg_writer_write_spiff_header(struct gpujpeg_encoder* encoder)
{
    struct gpujpeg_writer* writer = encoder->writer;
    gpujpeg_writer_write_app8(writer);

    // Length
    gpujpeg_writer_emit_2byte(writer, SPIFF_MARKER_LEN)

    // Identifier: ASCII "SPIFF\0"
    char spiff[] = { 'S', 'P', 'I', 'F', 'F', '\0' };
    for (unsigned i = 0; i < sizeof spiff; ++i) {
        gpujpeg_writer_emit_byte(writer, spiff[i]);
    }
    int color_space;
    if ( encoder->coder.param.comp_count == 1 ) {
        color_space = 8;
    }  else {
        switch (encoder->coder.param.color_space_internal) {
            case GPUJPEG_YCBCR_BT709:
                color_space = 1;
                break;
            case GPUJPEG_YCBCR_BT601_256LVLS:
                color_space = 3;
                break;
            case GPUJPEG_YCBCR_BT601:
                color_space = 4;
                break;
            case GPUJPEG_RGB:
                color_space = 10;
                break;
            default:
                color_space = 2; // no color space specified
        }
    }
    int profile = color_space == 3 || color_space == 8 ? 1 : 0; // 0 = No profile
    gpujpeg_writer_emit_2byte(writer, SPIFF_VERSION);
    gpujpeg_writer_emit_byte(writer, profile);   // ProfileID
    gpujpeg_writer_emit_byte(writer, encoder->coder.param.comp_count);   // number of components
    gpujpeg_writer_emit_4byte(writer, encoder->coder.param_image.height);
    gpujpeg_writer_emit_4byte(writer, encoder->coder.param_image.width);
    gpujpeg_writer_emit_byte(writer, color_space);
    gpujpeg_writer_emit_byte(writer, 8);   // bits per sample
    gpujpeg_writer_emit_byte(writer, SPIFF_COMPRESSION_JPEG);   // compression type: 5 - JPEG
    gpujpeg_writer_emit_byte(writer, 0);   // resolution units: 1 - ratio
    gpujpeg_writer_emit_4byte(writer, 1U); // VerticalResolution
    gpujpeg_writer_emit_4byte(writer, 1U); // HorizontalResolution
}

static void gpujpeg_writer_write_spiff_directory_eod(struct gpujpeg_writer* writer)
{
    gpujpeg_writer_write_app8(writer);
    gpujpeg_writer_emit_2byte(writer, SPIFF_ENTRY_TAG_EOD_LENGHT)
    gpujpeg_writer_emit_4byte(writer, SPIFF_ENTRY_TAG_EOD)
}

static void gpujpeg_writer_write_spiff_directory(struct gpujpeg_encoder* encoder)
{
    struct gpujpeg_writer* writer = encoder->writer;
    gpujpeg_writer_write_spiff_directory_eod(writer); // this must be last directory entry
}

static void gpujpeg_writer_write_spiff(struct gpujpeg_encoder* encoder)
{
    gpujpeg_writer_write_spiff_header(encoder);
    gpujpeg_writer_write_spiff_directory(encoder);
    gpujpeg_writer_write_soi(encoder->writer); // in SPIFF there is SOI once more starting the actual image data
}

/**
 * Write APP14 block
 *
 * This marker is used for RGB images - JFIF supports only YCbCr. This Apple
 * marker allows us to store RGB files. Inspired by libjpeg-turbo.
 *
 * @param writer  Writer structure
 */
static void gpujpeg_writer_write_app14(struct gpujpeg_writer* writer)
{
    gpujpeg_writer_emit_marker(writer, GPUJPEG_MARKER_APP14);

    // Length
    gpujpeg_writer_emit_2byte(writer, APP14_ADOBE_MARKER_LEN)

    // Identifier: ASCII "Adobe"
    gpujpeg_writer_emit_byte(writer, 0x41);
    gpujpeg_writer_emit_byte(writer, 0x64);
    gpujpeg_writer_emit_byte(writer, 0x6F);
    gpujpeg_writer_emit_byte(writer, 0x62);
    gpujpeg_writer_emit_byte(writer, 0x65);

    gpujpeg_writer_emit_2byte(writer, 100); // Version
    gpujpeg_writer_emit_2byte(writer, 0);   // Flags0
    gpujpeg_writer_emit_2byte(writer, 0);   // Flags1
    gpujpeg_writer_emit_byte(writer, 0);    // Color transform - 1 YCbCr, 2 YCCK, 0 otherwise (RGB or CMYK)
}

/**
 * Write DQT block
 *
 * @param encoder  Encoder structure
 * @param type  Component type for table retrieve
 * @return void
 */
void
gpujpeg_writer_write_dqt(struct gpujpeg_encoder* encoder, enum gpujpeg_component_type type)
{
    gpujpeg_writer_emit_marker(encoder->writer, GPUJPEG_MARKER_DQT);

    // Length
    gpujpeg_writer_emit_2byte(encoder->writer, 67);

    // Index: Y component = 0, Cb or Cr component = 1
    gpujpeg_writer_emit_byte(encoder->writer, (int)type);

    // Table changed from default with quality
    uint8_t* dqt = encoder->table_quantization[type].table_raw;

    // Emit table in zig-zag order
    for ( int i = 0; i < 64; i++ )  {
        unsigned char qval = (unsigned char)((char)(dqt[i]));
        gpujpeg_writer_emit_byte(encoder->writer, qval);
    }
}

static uint8_t gpujpeg_writer_get_component_id(int index, enum gpujpeg_color_space color_space) {
    if (color_space == GPUJPEG_RGB) {
            static const uint8_t rgb_ids[4] = { 'R', 'G', 'B', 'A' };
            assert((unsigned) index < sizeof rgb_ids / sizeof rgb_ids[0]);
            return rgb_ids[index];
    } else {
            return index + 1;
    }
}

/**
 * Currently support GPUJPEG_MARKER_SOF0 baseline implementation
 *
 * @param encoder  Encoder structure
 * @return void
 */
void
gpujpeg_writer_write_sof0(struct gpujpeg_encoder* encoder)
{
    gpujpeg_writer_emit_marker(encoder->writer, GPUJPEG_MARKER_SOF0);

    // Length
    gpujpeg_writer_emit_2byte(encoder->writer, 8 + 3 * encoder->coder.param.comp_count);

    // Precision (bit depth)
    gpujpeg_writer_emit_byte(encoder->writer, 8);
    // Dimensions
    gpujpeg_writer_emit_2byte(encoder->writer, encoder->coder.param_image.height);
    gpujpeg_writer_emit_2byte(encoder->writer, encoder->coder.param_image.width);

    // Number of components
    gpujpeg_writer_emit_byte(encoder->writer, encoder->coder.param.comp_count);

    // Components
    for ( int comp_index = 0; comp_index < encoder->coder.param.comp_count; comp_index++ ) {
        // Get component
        struct gpujpeg_component* component = &encoder->coder.component[comp_index];

        // Component ID
        gpujpeg_writer_emit_byte(encoder->writer, gpujpeg_writer_get_component_id(comp_index, encoder->coder.param.color_space_internal));

        // Sampling factors (1 << 4) + 1 (sampling h: 1, v: 1)
        gpujpeg_writer_emit_byte(encoder->writer, (component->sampling_factor.horizontal << 4) + component->sampling_factor.vertical);

        // Quantization table index
        if ( component->type == GPUJPEG_COMPONENT_LUMINANCE ) {
            gpujpeg_writer_emit_byte(encoder->writer, 0);
        } else if ( component->type == GPUJPEG_COMPONENT_CHROMINANCE ) {
            gpujpeg_writer_emit_byte(encoder->writer, 1);
        } else {
            assert(0);
        }
    }
}

/**
 * Write DHT block
 *
 * @param encoder  Encoder structure
 * @param type  Component type for table retrieve
 * @param is_ac  Flag if table AC or DC should be written
 * @return void
 */
void
gpujpeg_writer_write_dht(struct gpujpeg_encoder* encoder, enum gpujpeg_component_type comp_type, enum gpujpeg_huffman_type huff_type)
{
    // Get proper table and its index
    struct gpujpeg_table_huffman_encoder* table = NULL;
    int index;
    if ( comp_type == GPUJPEG_COMPONENT_LUMINANCE ) {
        if ( huff_type == GPUJPEG_HUFFMAN_AC ) {
            table = &encoder->table_huffman[comp_type][huff_type];
            index = 16;
        } else {
            table = &encoder->table_huffman[comp_type][huff_type];
            index = 0;
        }
    } else {
        if ( huff_type == GPUJPEG_HUFFMAN_AC ) {
            table = &encoder->table_huffman[comp_type][huff_type];
            index = 17;
        } else {
            table = &encoder->table_huffman[comp_type][huff_type];
            index = 1;
        }
    }

    gpujpeg_writer_emit_marker(encoder->writer, GPUJPEG_MARKER_DHT);

    int length = 0;
    for ( int i = 1; i <= 16; i++ )
        length += table->bits[i];

    gpujpeg_writer_emit_2byte(encoder->writer, length + 2 + 1 + 16);

    gpujpeg_writer_emit_byte(encoder->writer, index);

    for ( int i = 1; i <= 16; i++ )
        gpujpeg_writer_emit_byte(encoder->writer, table->bits[i]);

    // Varible-length
    for ( int i = 0; i < length; i++ )
        gpujpeg_writer_emit_byte(encoder->writer, table->huffval[i]);
}

/**
 * Write restart interval
 *
 * @param encoder  Encoder structure
 * @return void
 */
void
gpujpeg_writer_write_dri(struct gpujpeg_encoder* encoder)
{
    gpujpeg_writer_emit_marker(encoder->writer, GPUJPEG_MARKER_DRI);

    // Length
    gpujpeg_writer_emit_2byte(encoder->writer, 4);

    // Restart interval
    gpujpeg_writer_emit_2byte(encoder->writer, encoder->coder.param.restart_interval);
}

static void
gpujpeg_writer_write_com(struct gpujpeg_encoder* encoder, const char *str)
{
    gpujpeg_writer_emit_marker(encoder->writer, GPUJPEG_MARKER_COM);
    // Length
    gpujpeg_writer_emit_2byte(encoder->writer, 2 + strlen(str) + 1);

    for ( unsigned i = 0; i <= strlen(str); i++ )  { // include terminating '\0'
        gpujpeg_writer_emit_byte(encoder->writer, str[i]);
    }
}

/**
 * Write encoding library in comment marker
 */
static void
gpujpeg_writer_write_com_library(struct gpujpeg_encoder* encoder)
{
    char creator[] = "CREATOR: GPUJPEG, quality = \0\0\0";
    snprintf(creator + strlen(creator), sizeof creator - strlen(creator), "%d",
                    GPUJPEG_CLAMP(encoder->coder.param.quality, 1, 100));
    gpujpeg_writer_write_com(encoder, creator);
}

/* Documented at declaration */
void
gpujpeg_writer_write_header(struct gpujpeg_encoder* encoder)
{
    gpujpeg_writer_write_soi(encoder->writer);

    switch (encoder->header_type) {
    case GPUJPEG_HEADER_DEFAULT:
        if ( encoder->coder.param.comp_count == 4 ) {
            gpujpeg_writer_write_spiff(encoder);
        } else {
            switch (encoder->coder.param.color_space_internal) {
                case GPUJPEG_YCBCR_BT601:
                case GPUJPEG_YCBCR_BT709:
                    gpujpeg_writer_write_spiff(encoder);
                    break;
                case GPUJPEG_RGB:
                    gpujpeg_writer_write_app14(encoder->writer);
                    break;
                default: // ordinary JFIF
                    gpujpeg_writer_write_app0(encoder->writer);
                    break;
            }
        }
        break;
    case GPUJPEG_HEADER_JFIF:
        gpujpeg_writer_write_app0(encoder->writer);
        break;
    case GPUJPEG_HEADER_SPIFF:
        gpujpeg_writer_write_spiff(encoder);
        break;
    case GPUJPEG_HEADER_ADOBE:
        gpujpeg_writer_write_app14(encoder->writer);
        break;
    }

    unsigned dqt_type_emitted = 0U;
    for ( int i = 0; i < encoder->coder.param.comp_count; ++i ) {
        if ((dqt_type_emitted & (1U << encoder->coder.component[i].type)) == 0) {
            gpujpeg_writer_write_dqt(encoder, encoder->coder.component[i].type);
            dqt_type_emitted |= 1U << encoder->coder.component[i].type;
        }
    }

    gpujpeg_writer_write_sof0(encoder);

    // emit GPUJPEG_COMPONENT_CHROMINANCE only if needed (not used for RGB or grayscale)
    unsigned dht_type_emitted = 0U; ///< bitfield of already output component types (CHROMINANCE/LUMINANCE)
    for ( int i = 0; i < encoder->coder.param.comp_count; ++i ) {
        if ((dht_type_emitted & (1U << encoder->coder.component[i].type)) == 0) {
            gpujpeg_writer_write_dht(encoder, encoder->coder.component[i].type, GPUJPEG_HUFFMAN_DC);   // DC table
            gpujpeg_writer_write_dht(encoder, encoder->coder.component[i].type, GPUJPEG_HUFFMAN_AC);   // AC table
            dht_type_emitted |= 1U << encoder->coder.component[i].type;
        }
    }

    gpujpeg_writer_write_dri(encoder);

    gpujpeg_writer_write_com_library(encoder);

    if (encoder->coder.param.color_space_internal == GPUJPEG_YCBCR_BT601) {
            gpujpeg_writer_write_com(encoder, "CS=ITU601");
    }
}

/* Documented at declaration */
void
gpujpeg_writer_write_segment_info(struct gpujpeg_encoder* encoder)
{
    if ( encoder->coder.param.segment_info ) {
        // First setup of position
        if ( encoder->writer->segment_info_position == 0 ) {
            encoder->writer->segment_info_position = encoder->writer->buffer_current;
        }
        // Get segment position in scan
        int position = encoder->writer->buffer_current - encoder->writer->segment_info_position;

        // Determine right header index
        int header_index = (encoder->writer->segment_info_index * 4) / GPUJPEG_MAX_HEADER_SIZE;
        assert(header_index < encoder->writer->segment_info_count);

        // Save segment position into segment info data to right header
        int header_data_index = (encoder->writer->segment_info_index * 4) % GPUJPEG_MAX_HEADER_SIZE;
        encoder->writer->segment_info[header_index][header_data_index + 0] = (uint8_t)(((position) >> 24) & 0xFF);
        encoder->writer->segment_info[header_index][header_data_index + 1] = (uint8_t)(((position) >> 16) & 0xFF);
        encoder->writer->segment_info[header_index][header_data_index + 2] = (uint8_t)(((position) >> 8) & 0xFF);
        encoder->writer->segment_info[header_index][header_data_index + 3] = (uint8_t)(((position) >> 0) & 0xFF);

        // Increase segment info index
        encoder->writer->segment_info_index++;
    }
}

/* Documented at declaration */
void
gpujpeg_writer_write_scan_header(struct gpujpeg_encoder* encoder, int scan_index)
{
    // Write custom application header containing info about scan segments
    if ( encoder->coder.param.segment_info && encoder->coder.param.restart_interval > 0 ) {
        // Get segment count in scan
        int segment_count = 0;
        if ( encoder->coder.param.interleaved == 1 ) {
            // Get segment count for all components
            segment_count = encoder->coder.segment_count;
        } else {
            // Component index
            int comp_index = scan_index;
            // Get segment count for one component
            segment_count = encoder->coder.component[comp_index].segment_count;
        }

        // Calculate header data size
        int data_size = (segment_count + 1) * 4;

        // Reset current position in the scan and header count
        encoder->writer->segment_info_count = 0;
        encoder->writer->segment_info_index = 0;
        encoder->writer->segment_info_position = 0;

        // Emit headers (each header can have data size of only 2^16)
        while ( data_size > 0 ) {
            // Determine current header size
            int header_size = data_size;
            if ( header_size > GPUJPEG_MAX_HEADER_SIZE ) {
                header_size = GPUJPEG_MAX_HEADER_SIZE;
            }
            data_size -= header_size;

            // Header marker
            gpujpeg_writer_emit_marker(encoder->writer, GPUJPEG_MARKER_SEGMENT_INFO);

            // Write custom application header
            gpujpeg_writer_emit_2byte(encoder->writer, 3 + header_size);
            gpujpeg_writer_emit_byte(encoder->writer, scan_index);

            // Set pointer to current segment info data block, where segment positions will be placed
            encoder->writer->segment_info[encoder->writer->segment_info_count] = encoder->writer->buffer_current;

            // Prepare size for segment info data
            encoder->writer->buffer_current += header_size;

            // Increase header count
            encoder->writer->segment_info_count++;
            assert(encoder->writer->segment_info_count < GPUJPEG_MAX_SEGMENT_INFO_HEADER_COUNT);
        }
    }

    // Begin scan header
    gpujpeg_writer_emit_marker(encoder->writer, GPUJPEG_MARKER_SOS);

    if ( encoder->coder.param.interleaved == 1 ) {
        // Length
        gpujpeg_writer_emit_2byte(encoder->writer, 6 + 2 * encoder->coder.param.comp_count);

        // Component count
        gpujpeg_writer_emit_byte(encoder->writer, encoder->coder.param.comp_count);

        // Components
        for ( int comp_index = 0; comp_index < encoder->coder.param.comp_count; comp_index++ ) {
            // Get component
            struct gpujpeg_component* component = &encoder->coder.component[comp_index];

            // Component ID
            gpujpeg_writer_emit_byte(encoder->writer, gpujpeg_writer_get_component_id(comp_index, encoder->coder.param.color_space_internal));

            // Component DC and AC entropy coding table indexes
            if ( component->type == GPUJPEG_COMPONENT_LUMINANCE ) {
                gpujpeg_writer_emit_byte(encoder->writer, 0);    // (0 << 4) | 0
            } else if ( component->type == GPUJPEG_COMPONENT_CHROMINANCE ) {
                gpujpeg_writer_emit_byte(encoder->writer, 0x11); // (1 << 4) | 1
            } else {
                assert(0);
            }
        }
    } else {
        // Component index
        int comp_index = scan_index;

        // Get component
        struct gpujpeg_component* component = &encoder->coder.component[comp_index];

        // Length
        gpujpeg_writer_emit_2byte(encoder->writer, 8);

        // Component count
        gpujpeg_writer_emit_byte(encoder->writer, 1);

        // Component index
        gpujpeg_writer_emit_byte(encoder->writer, gpujpeg_writer_get_component_id(comp_index, encoder->coder.param.color_space_internal));

        // Component DC and AC entropy coding table indexes
        if ( component->type == GPUJPEG_COMPONENT_LUMINANCE ) {
            gpujpeg_writer_emit_byte(encoder->writer, 0);    // (0 << 4) | 0
        } else if ( component->type == GPUJPEG_COMPONENT_CHROMINANCE ) {
            gpujpeg_writer_emit_byte(encoder->writer, 0x11); // (1 << 4) | 1
        } else {
            assert(0);
        }
    }

    gpujpeg_writer_emit_byte(encoder->writer, 0);    // Ss
    gpujpeg_writer_emit_byte(encoder->writer, 0x3F); // Se
    gpujpeg_writer_emit_byte(encoder->writer, 0);    // Ah/Al
}

/* vi: set expandtab sw=4 : */
