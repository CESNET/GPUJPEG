/**
 * @file
 * Copyright (c) 2025 CESNET, zájmové sdružení právnických osob
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

#include "gpujpeg_exif.h"

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "compat/endian.h"
#include "gpujpeg_common_internal.h"
#include "gpujpeg_encoder_internal.h"
#include "gpujpeg_marker.h"
#include "gpujpeg_writer.h"

enum exif_tag_type {
    ET_BYTE = 1,       ///< 8-bit unsigned integer
    ET_ASCII = 2,      ///< NULL-terminated ASCII string
    ET_SHORT = 3,      ///< 16-bit unsigned integer
    ET_LONG = 4,       ///< 32-bit unsigned integer
    ET_RATIONAL = 5,   ///< two LONGs, first LONG is numerator and second denominator
    ET_UNDEFINED = 7,  ///< An 8-bit byte that may take any value depending on the field definition.
    ET_SLONG = 9,      ///< 32-bit signed integer
    ET_SRATIONAL = 10, ///< two SLONGs - first numerator, second denominator
};

const struct exif_tag_type_info_t
{
    unsigned size;
} exif_tag_type_info[] = {[ET_BYTE] = {1},     [ET_ASCII] = {0},     [ET_SHORT] = {2}, [ET_LONG] = {4},
                          [ET_RATIONAL] = {8}, [ET_UNDEFINED] = {4}, [ET_SLONG] = {4}, [ET_SRATIONAL] = {8}};

enum exif_tiff_tag {
    // 0th IFD TIFF Tags
    ETIFF_XRESOLUTION,       ///< Image resolution in width direction (mandatory)
    ETIFF_YRESOLUTION,       ///< Image resolution in height direction (mandatory)
    ETIFF_RESOLUTION_UNIT,   ///< Unit of X and Y resolution (mandatory)
    ETIFF_SOFTWARE,          ///< Software used (optional)
    ETIFF_YCBCR_POSITIONING, ///< Y and C positioning (mandatory)
    ETIFF_EXIF_IFD_POINTER,  ///< EXIF tag (mandatory)
    // 0th SubIFD Exif Private Tags
    EEXIF_EXIF_VERSION,             ///< Exif version (mandatory)
    EEXIF_COMPONENTS_CONFIGURATION, ///< Meaning of each component (mandatory)
    EEXIF_FLASHPIX_VERSION,         ///< Supported Flashpix version (mandatory)
    EEXIF_COLOR_SPACE,              ///< Color space information (mandatory)
    EEXIF_PIXEL_X_DIMENSION,        ///< Valid image width (mandatory)
    EEXIF_PIXEL_Y_DIMENSION,        ///< Valid image height (mandatory)
};

const struct exif_tiff_tag_info_t {
    uint16_t id;
    enum exif_tag_type type;
} exif_tiff_tag_info[] = {
    [ETIFF_XRESOLUTION]       = {0x11A,  ET_RATIONAL},
    [ETIFF_YRESOLUTION]       = {0x11B,  ET_RATIONAL},
    [ETIFF_RESOLUTION_UNIT]   = {0x128,  ET_SHORT   },
    [ETIFF_SOFTWARE]          = {0x131,  ET_ASCII   },
    [ETIFF_YCBCR_POSITIONING] = {0x213,  ET_SHORT   },
    [ETIFF_EXIF_IFD_POINTER]  = {0x8769, ET_LONG    },
    // Exif SubIFD
    [EEXIF_EXIF_VERSION]             = {0x9000, ET_UNDEFINED },
    [EEXIF_COMPONENTS_CONFIGURATION] = {0x9101, ET_UNDEFINED },
    [EEXIF_FLASHPIX_VERSION]         = {0xA000, ET_UNDEFINED },
    [EEXIF_COLOR_SPACE]              = {0xA001, ET_SHORT },
    [EEXIF_PIXEL_X_DIMENSION]        = {0xA002, ET_SHORT }, // type can be also LONG
    [EEXIF_PIXEL_Y_DIMENSION]        = {0xA003, ET_SHORT }, // ditto
};

// misc constants
enum {
    ETIFF_CENTER = 1,
    ETIFF_SRGB = 1,
    ETIFF_INCHES = 2,
    NEXT_IFD_PTR_SZ = 4,
    IFD_ITEM_SZ = 12,
};

union value_u {
    uint32_t uvalue;
    const char* csvalue; // ET_STRING or ET_UNKNOWN
    struct
    {
        uint32_t uvalue_num;
        uint32_t uvalue_den;
    } urational;
};

static void
write_exif_emit_string_tag(struct gpujpeg_writer* writer, uint16_t tag, const char* str, const uint8_t* start,
                           uint8_t** end)
{
    const size_t count = strlen(str) + 1;
    gpujpeg_writer_emit_2byte(writer, tag);
    gpujpeg_writer_emit_2byte(writer, ET_ASCII);
    gpujpeg_writer_emit_4byte(writer, count);
    gpujpeg_writer_emit_4byte(writer, *end - start);;
    memcpy(*end, str, count);
    *end += count;
}

static void
write_exif_emit_4bytes_tag(struct gpujpeg_writer* writer, uint16_t tag, enum exif_tag_type type, uint32_t size,
                           uint32_t val)
{
    assert(size <= 4);
    gpujpeg_writer_emit_2byte(writer, tag);
    gpujpeg_writer_emit_2byte(writer, type);
    gpujpeg_writer_emit_4byte(writer, 1);

    for (unsigned i = size; i > 0; --i) { // Exif 4.6.2 - left aligned value; we use big endian
        gpujpeg_writer_emit_byte(writer, (val >> 8 * (i - 1)) & 0xFFU);
    }
    for (unsigned i = size; i < 4; ++i) {
        gpujpeg_writer_emit_byte(writer, 0);
    }
}

static void
write_exif_emit_lt_4b_tag(struct gpujpeg_writer* writer, uint16_t tag, enum exif_tag_type type, uint32_t size,
                           union value_u val, const uint8_t* start,
                           uint8_t** end)
{
    gpujpeg_writer_emit_2byte(writer, tag);
    gpujpeg_writer_emit_2byte(writer, type);
    gpujpeg_writer_emit_4byte(writer, 1);
    gpujpeg_writer_emit_4byte(writer, *end - start);;

    if (type == ET_RATIONAL) {
        assert(size == 8);
        uint32_t num = htobe32(val.urational.uvalue_num);
        uint32_t den = htobe32(val.urational.uvalue_den);
        memcpy(*end, &num, sizeof num);
        memcpy(*end + 4, &den, sizeof den);
    }
    else {
        abort();
    }

    *end += size;
}

static void
write_exif_tag(struct gpujpeg_writer* writer, enum exif_tiff_tag tag, union value_u val, const uint8_t* start,
               uint8_t** end)
{

    const struct exif_tiff_tag_info_t* t = &exif_tiff_tag_info[tag];
    const unsigned size = exif_tag_type_info[t->type].size;

    // size for string is computed
    assert(size > 0 || t->type == ET_ASCII);

    if ( t->type == ET_ASCII ) {
        write_exif_emit_string_tag(writer, t->id, val.csvalue, start, end);
        return;
    }
    if ( t->type == ET_UNDEFINED ) {
        assert(size == 4);
        gpujpeg_writer_emit_2byte(writer, t->id);
        gpujpeg_writer_emit_2byte(writer, t->type);
        gpujpeg_writer_emit_4byte(writer, 4); // count - we have all 4 B, unsure if defined otherwise for other
        memcpy(writer->buffer_current, val.csvalue, 4);
        writer->buffer_current += 4;
        return;
    }
    if (size <= 4) {
        write_exif_emit_4bytes_tag(writer, t->id, t->type, size, val.uvalue);
        return;
    }
    write_exif_emit_lt_4b_tag(writer, t->id, t->type, size, val, start, end);
}

struct tag_value
{
    enum exif_tiff_tag tag;
    union value_u value;
};

/**
 * @param tags  array of tags, should be ordered awcending according to exif_tiff_tag_info_t.id
 */
static void
gpujpeg_write_ifd(struct gpujpeg_writer* writer, const uint8_t* start, size_t count,
                  const struct tag_value tags[])
{
    enum {
        EXIF_IFD_NUM_SZ = 2,
    };
    uint8_t *end = writer->buffer_current + EXIF_IFD_NUM_SZ + (count * IFD_ITEM_SZ) + NEXT_IFD_PTR_SZ;
    gpujpeg_writer_emit_2byte(writer, count); // IFD Item Count

    for ( unsigned i = 0; i < count; ++i ) {
        const struct tag_value* info = &tags[i];
        union value_u value = info->value;
        if ( info->tag == ETIFF_EXIF_IFD_POINTER ) {
            value.uvalue = end - start;
        }
        write_exif_tag(writer, info->tag, value, start, &end);
    }
    gpujpeg_writer_emit_4byte(writer, 0); // Next IFD Offset (none)
    writer->buffer_current = end;         // jump after the section Value longer than 4Byte of 0th IFD
}

static void
gpujpeg_write_0th(struct gpujpeg_encoder* encoder, const uint8_t* start)
{
    const struct tag_value tags[] = {
        {ETIFF_XRESOLUTION,       {.urational = {72, 1}}  },
        {ETIFF_YRESOLUTION,       {.urational = {72, 1}}  },
        {ETIFF_RESOLUTION_UNIT,   {.uvalue = ETIFF_INCHES}},
        {ETIFF_SOFTWARE,          {.csvalue = "GPUJPEG"}  },
        {ETIFF_YCBCR_POSITIONING, {.uvalue = ETIFF_CENTER}}, // center
        {ETIFF_EXIF_IFD_POINTER,  {0}                     }, // value later; should be last
    };

    gpujpeg_write_ifd(encoder->writer, start, ARR_SIZE(tags), tags);
}

static void gpujpeg_write_exif_ifd(struct gpujpeg_encoder* encoder, const uint8_t *start)
{
    const struct tag_value tags[] = {
        {EEXIF_EXIF_VERSION,             {.csvalue = "0230"}                }, // 2.30
        {EEXIF_COMPONENTS_CONFIGURATION, {.csvalue = "\1\2\3\0"}            }, // YCbCr
        {EEXIF_FLASHPIX_VERSION,         {.csvalue = "0100"}                }, // "0100"
        {EEXIF_COLOR_SPACE,              {.uvalue = ETIFF_SRGB}             },
        {EEXIF_PIXEL_X_DIMENSION,        {encoder->coder.param_image.width} },
        {EEXIF_PIXEL_Y_DIMENSION,        {encoder->coder.param_image.height}},
    };
    gpujpeg_write_ifd(encoder->writer, start, ARR_SIZE(tags), tags);
}


/// writes EXIF APP1 marker
void
gpujpeg_writer_write_exif(struct gpujpeg_encoder* encoder)
{
    if ( encoder->coder.param.color_space_internal != GPUJPEG_YCBCR_BT601_256LVLS ) {
        WARN_MSG("[Exif] Color space %s currently not recorded, assumed %s (report)\n",
                 gpujpeg_color_space_get_name(encoder->coder.param.color_space_internal),
                 gpujpeg_color_space_get_name(GPUJPEG_YCBCR_BT601_256LVLS));
    }
    struct gpujpeg_writer* writer = encoder->writer;
    gpujpeg_writer_emit_marker(writer, GPUJPEG_MARKER_APP1);

    // Length - will be written later
    uint8_t *const length_p = writer->buffer_current;
    gpujpeg_writer_emit_2byte(writer, 0);

    // Identifier: 0x45786966
    gpujpeg_writer_emit_byte(writer, 'E');
    gpujpeg_writer_emit_byte(writer, 'x');
    gpujpeg_writer_emit_byte(writer, 'i');
    gpujpeg_writer_emit_byte(writer, 'f');
    gpujpeg_writer_emit_byte(writer, '\0');

    gpujpeg_writer_emit_byte(writer, 0); // pad

    // we will use big-endian (Motorola; 'II' would be Intel)
    const uint8_t* const start = writer->buffer_current;
    // 'MM' means big-endian (Motorola), 'ii' would be little (Intel)
    gpujpeg_writer_emit_byte(writer, 'M');
    gpujpeg_writer_emit_byte(writer, 'M');

    gpujpeg_writer_emit_2byte(writer, 0x002a); // TIFF header
    gpujpeg_writer_emit_4byte(writer, 0x08); // IFD offset - follows immediately

    gpujpeg_write_0th(encoder, start);
    gpujpeg_write_exif_ifd(encoder, start);

    // set the marker length
    size_t length = writer->buffer_current - length_p;
    length_p[0] = length >> 8;
    length_p[1] = length;
}

