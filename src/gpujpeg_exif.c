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
#include <ctype.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>    // for strftime

#ifdef _WIN32
#define strncasecmp _strnicmp
#endif

#include "compat/endian.h"
#include "gpujpeg_common_internal.h"
#include "gpujpeg_encoder_internal.h"
#include "gpujpeg_marker.h"
#include "gpujpeg_writer.h"

enum exif_tag_type {
    ET_NONE = 0,
    ET_BYTE = 1,       ///< 8-bit unsigned integer
    ET_ASCII = 2,      ///< NULL-terminated ASCII string
    ET_SHORT = 3,      ///< 16-bit unsigned integer
    ET_LONG = 4,       ///< 32-bit unsigned integer
    ET_RATIONAL = 5,   ///< two LONGs, first LONG is numerator and second denominator
    ET_UNDEFINED = 7,  ///< An 8-bit byte that may take any value depending on the field definition.
    ET_SLONG = 9,      ///< 32-bit signed integer
    ET_SRATIONAL = 10, ///< two SLONGs - first numerator, second denominator
    ET_END,
};

enum {
    T_NUMERIC = 1 << 0,
    T_UNSIGNED = 1 << 1,
};

static const struct exif_tag_type_info_t
{
    unsigned size;
    const char* name;
    unsigned type_flags;
} exif_tag_type_info[] = {
      [ET_BYTE] =      {1, "BYTE",      T_NUMERIC|T_UNSIGNED},
      [ET_ASCII] =     {0, "ASCII",     0                   },
      [ET_SHORT] =     {2, "SHORT",     T_NUMERIC|T_UNSIGNED},
      [ET_LONG] =      {4, "LONG",      T_NUMERIC|T_UNSIGNED},
      [ET_RATIONAL] =  {8, "RATIONAL",  T_UNSIGNED},
      [ET_UNDEFINED] = {4, "UNDEFINED", },
      [ET_SLONG] =     {4, "SLONG"    , T_NUMERIC},
      [ET_SRATIONAL] = {8, "SRATIONAL", 0}
};

enum exif_tiff_tag {
    // 0th IFD TIFF Tags
    ETIFF_ORIENTATION,       ///< Image resolution in width direction (recommended)
    ETIFF_XRESOLUTION,       ///< Image resolution in width direction (mandatory)
    ETIFF_YRESOLUTION,       ///< Image resolution in height direction (mandatory)
    ETIFF_RESOLUTION_UNIT,   ///< Unit of X and Y resolution (mandatory)
    ETIFF_SOFTWARE,          ///< Software used (optional)
    ETIFF_DATE_TIME,         ///< File change date and time (recommeneded)
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
    const char *name;
} exif_tiff_tag_info[] = {
    [ETIFF_ORIENTATION]       = {0x112,  ET_SHORT,    "Orientation"      },
    [ETIFF_XRESOLUTION]       = {0x11A,  ET_RATIONAL, "XResolution"      },
    [ETIFF_YRESOLUTION]       = {0x11B,  ET_RATIONAL, "YResolution"      },
    [ETIFF_RESOLUTION_UNIT]   = {0x128,  ET_SHORT,    "ResolutionUnit"   },
    [ETIFF_SOFTWARE]          = {0x131,  ET_ASCII,    "Sofware"          },
    [ETIFF_DATE_TIME]         = {0x132,  ET_ASCII,    "DateTime"         },
    [ETIFF_YCBCR_POSITIONING] = {0x213,  ET_SHORT,    "YCbCrPositioning" },
    [ETIFF_EXIF_IFD_POINTER]  = {0x8769, ET_LONG,     "Exif IFD Pointer"},
    // Exif SubIFD
    [EEXIF_EXIF_VERSION]             = {0x9000, ET_UNDEFINED, "ExifVersion"           },
    [EEXIF_COMPONENTS_CONFIGURATION] = {0x9101, ET_UNDEFINED, "ComponentConfiguration"},
    [EEXIF_FLASHPIX_VERSION]         = {0xA000, ET_UNDEFINED, "FlashPixVersion"       },
    [EEXIF_COLOR_SPACE]              = {0xA001, ET_SHORT,     "ColorSpace"            },
    [EEXIF_PIXEL_X_DIMENSION]        = {0xA002, ET_SHORT,     "PixelXDimension"       }, // type can be also LONG
    [EEXIF_PIXEL_Y_DIMENSION]        = {0xA003, ET_SHORT,     "PixelYDimension"       }, // ditto
};

// misc constants
enum {
    ETIFF_ORIENT_HORIZONTAL = 1, // normal
    ETIFF_CENTER = 1,
    ETIFF_SRGB = 1,
    ETIFF_INCHES = 2,
    NEXT_IFD_PTR_SZ = 4,
    IFD_ITEM_SZ = 12,
    EEXIF_FIRST = 0x827A, // (Exposure time) first tag id of Exif Private Tags
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
write_exif_tag(struct gpujpeg_writer* writer, enum exif_tag_type type, uint16_t tag_id, union value_u val,
               const uint8_t* start, uint8_t** end)
{
    assert(type < ET_END);
    const unsigned size = exif_tag_type_info[type].size;

    // size for string is computed
    assert(size > 0 || type == ET_ASCII);

    if ( type == ET_ASCII ) {
        write_exif_emit_string_tag(writer, tag_id, val.csvalue, start, end);
        return;
    }
    if ( type == ET_UNDEFINED ) {
        assert(size == 4);
        gpujpeg_writer_emit_2byte(writer, tag_id);
        gpujpeg_writer_emit_2byte(writer, type);
        gpujpeg_writer_emit_4byte(writer, 4); // count - we have all 4 B, unsure if defined otherwise for other
        memcpy(writer->buffer_current, val.csvalue, 4);
        writer->buffer_current += 4;
        return;
    }
    if (size <= 4) {
        write_exif_emit_4bytes_tag(writer, tag_id, type, size, val.uvalue);
        return;
    }
    write_exif_emit_lt_4b_tag(writer, tag_id, type, size, val, start, end);
}

struct tag_value
{
    enum exif_tiff_tag tag;
    union value_u value;
};

/// custom exif tag values
struct custom_tag_value
{
    uint16_t tag_id;
    enum exif_tag_type type;
    union value_u value;
};
enum { CT_TIFF, CT_EXIF, CT_NUM };
/// custom exif tags given by user
struct gpujpeg_exif_tags {
    struct custom_exif_tags
    {
        struct custom_tag_value *vals;
        size_t count;
    } tags[CT_NUM];
};

static int
ifd_sort(const void* a, const void* b)
{
    const uint8_t* aa = a;
    const uint8_t *bb = b;
    int a_tag_id = aa[0] << 8 | aa[1];
    int b_tag_id = bb[0] << 8 | bb[1];
    return a_tag_id - b_tag_id;
}

/**
 * @param tags  array of tags, should be ordered awcending according to exif_tiff_tag_info_t.id
 */
static void
gpujpeg_write_ifd(struct gpujpeg_writer* writer, const uint8_t* start, size_t count,
                  const struct tag_value tags[], const struct custom_exif_tags *custom_tags)
{
    enum {
        EXIF_IFD_NUM_SZ = 2,
    };
    size_t count_all = count + custom_tags->count;
    uint8_t* end = writer->buffer_current + EXIF_IFD_NUM_SZ + (count_all * IFD_ITEM_SZ) + NEXT_IFD_PTR_SZ;
    gpujpeg_writer_emit_2byte(writer, count_all); // IFD Item Count

    uint8_t *first_rec = writer->buffer_current;
    unsigned last_tag_id = 0;
    for ( unsigned i = 0; i < count; ++i ) {
        const struct tag_value* info = &tags[i];
        union value_u value = info->value;
        if ( info->tag == ETIFF_EXIF_IFD_POINTER ) {
            value.uvalue = end - start;
        }
        const struct exif_tiff_tag_info_t* t = &exif_tiff_tag_info[info->tag];
        assert(t->id >= last_tag_id);
        last_tag_id = t->id;
        write_exif_tag(writer, t->type, t->id, value, start, &end);
    }
    if ( custom_tags != NULL ) { // add user custom tags
        for ( unsigned i = 0; i < custom_tags->count; ++i ) {
            write_exif_tag(writer, custom_tags->vals[i].type, custom_tags->vals[i].tag_id, custom_tags->vals[i].value,
                           start, &end);
        }
        // ensure custom_tags are in-ordered
        qsort(first_rec, (writer->buffer_current - first_rec) / IFD_ITEM_SZ, IFD_ITEM_SZ, ifd_sort);
    }
    gpujpeg_writer_emit_4byte(writer, 0); // Next IFD Offset (none)
    writer->buffer_current = end;         // jump after the section Value longer than 4Byte of 0th IFD
}

/**
 * from tags remove the items that are overriden by custom_tags
 */
static size_t
remove_overriden(size_t count, struct tag_value tags[], const struct custom_exif_tags* custom_tags)
{
    for ( unsigned i = 0; i < custom_tags->count; ++i ) {
        for ( unsigned j = 0; j < count; ++j ) {
            if ( custom_tags->vals[i].tag_id == exif_tiff_tag_info[tags[j].tag].id ) {
                memmove(tags + j, tags + j + 1, (count - j - 1) * sizeof(tags[0]));
                count -= 1;
                break;
            }
        }
    }
    return count;
}

static void
gpujpeg_write_0th(struct gpujpeg_encoder* encoder, const uint8_t* start)
{
    char date_time[] = "    :  :     :  :  "; // unknown val by Exif 2.3
    time_t now = time(NULL);
    (void) strftime(date_time, sizeof date_time, "%Y:%m:%d %H:%M:%S", localtime(&now));
    struct tag_value tags[] = {
        {ETIFF_ORIENTATION,       {.uvalue = ETIFF_ORIENT_HORIZONTAL}},
        {ETIFF_XRESOLUTION,       {.urational = {72, 1}}             },
        {ETIFF_YRESOLUTION,       {.urational = {72, 1}}             },
        {ETIFF_RESOLUTION_UNIT,   {.uvalue = ETIFF_INCHES}           },
        {ETIFF_SOFTWARE,          {.csvalue = "GPUJPEG"}             },
        {ETIFF_DATE_TIME ,        {.csvalue = date_time}             },
        {ETIFF_YCBCR_POSITIONING, {.uvalue = ETIFF_CENTER}           },
        {ETIFF_EXIF_IFD_POINTER,  {0}                                }, // value will be set later
    };
    size_t tag_count = ARR_SIZE(tags);
    const struct custom_exif_tags* custom_tags = NULL;
    if (encoder->writer->exif_tags != NULL) {
        custom_tags = &encoder->writer->exif_tags->tags[CT_TIFF];
        tag_count = remove_overriden(tag_count, tags, custom_tags);
    }

    gpujpeg_write_ifd(encoder->writer, start, tag_count, tags, custom_tags);
}

static void gpujpeg_write_exif_ifd(struct gpujpeg_encoder* encoder, const uint8_t *start)
{
    struct tag_value tags[] = {
        {EEXIF_EXIF_VERSION,             {.csvalue = "0230"}                }, // 2.30
        {EEXIF_COMPONENTS_CONFIGURATION, {.csvalue = "\1\2\3\0"}            }, // YCbCr
        {EEXIF_FLASHPIX_VERSION,         {.csvalue = "0100"}                }, // "0100"
        {EEXIF_COLOR_SPACE,              {.uvalue = ETIFF_SRGB}             },
        {EEXIF_PIXEL_X_DIMENSION,        {encoder->coder.param_image.width} },
        {EEXIF_PIXEL_Y_DIMENSION,        {encoder->coder.param_image.height}},
    };
    size_t tag_count = ARR_SIZE(tags);
    const struct custom_exif_tags* custom_tags = NULL;
    if (encoder->writer->exif_tags != NULL) {
        custom_tags = &encoder->writer->exif_tags->tags[CT_EXIF];
        tag_count = remove_overriden(tag_count, tags, custom_tags);
    }

    gpujpeg_write_ifd(encoder->writer, start, ARR_SIZE(tags), tags, custom_tags);
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

static bool
get_numeric_tag_type(char** endptr, long* tag_id, enum exif_tag_type* type)
{
    *tag_id = strtol(*endptr, endptr, 0);
    if ( **endptr != ':' ) {
        ERROR_MSG("Error parsing Exif tag ID or missing type!\n");
        return false;
    }
    *endptr += 1;
    for ( unsigned i = ET_NONE + 1; i < ET_END; ++i ) {
        if ( exif_tag_type_info[i].name == NULL ) { // unset/invalid type
            continue;
        }
        size_t len = strlen(exif_tag_type_info[i].name);
        if ( strncasecmp(*endptr, exif_tag_type_info[i].name, len) == 0 ) {
            *type = i;
            *endptr += len;
            break;
        }
    }
    if ( type == ET_NONE ) {
        ERROR_MSG("Error parsing Exif tag type!\n");
        return false;
    }
    if ( **endptr != '=' ) {
        ERROR_MSG("Error parsing Exif - missing value!\n");
        return false;
    }
    return true;
}

/**
 * add user-provided Exif tag
 */
bool
gpujpeg_exif_add_tag(struct gpujpeg_exif_tags** exif_tags, const char* cfg)
{
    if (strcmp(cfg, "help") == 0) {
        printf("Exif value syntax:\n"
               "\t" GPUJPEG_ENC_OPT_EXIF_TAG "=<ID>:<type>=<value>\n"
               "\t" GPUJPEG_ENC_OPT_EXIF_TAG "=<name>=<value>\n"
               "\t\tname must be a tag name known to GPUJPEG\n");
        return false;
    }

    char *endptr = (char *) cfg;
    long tag_id = 0;
    enum exif_tag_type type = ET_NONE;

    if (isdigit(*endptr)) {
        if ( !get_numeric_tag_type(&endptr, &tag_id, &type) ) {
            return false;
        }
    } else {
        for (unsigned i = 0; i < ARR_SIZE(exif_tiff_tag_info); ++i) {
            size_t len = strlen(exif_tiff_tag_info[i].name);
            if ( strncasecmp(endptr, exif_tiff_tag_info[i].name, len) == 0 ) {
                tag_id = exif_tiff_tag_info[i].id;
                type = exif_tiff_tag_info[i].type;
                endptr += len;
            }
        }
        if (*endptr != '=') {
            ERROR_MSG("[Exif] Wrong tag name or missing value!\n");
            return false;
        }
    }

    unsigned numeric_unsigned = T_NUMERIC | T_UNSIGNED;
    if ( (exif_tag_type_info[type].type_flags & numeric_unsigned) != numeric_unsigned ) {
        ERROR_MSG("Only unsigned integers currently supported!\n");
        return false;
    }
    endptr += 1;
    unsigned long long val = strtoull(endptr, &endptr, 0);
    if (*endptr != '\0') {
        ERROR_MSG("Trainling data in Exif value!\n");
        return false;
    }

    if (*exif_tags == NULL) {
        *exif_tags = calloc(1, sizeof **exif_tags);
    }

    int table_idx =  tag_id < EEXIF_FIRST ? CT_TIFF : CT_EXIF;
    size_t new_size = (*exif_tags)->tags[table_idx].count += 1;
    (*exif_tags)->tags[table_idx].vals = realloc((*exif_tags)->tags[table_idx].vals,
            new_size * sizeof (*exif_tags)->tags[table_idx].vals[0]);
    (*exif_tags)->tags[table_idx].vals[new_size - 1].tag_id = tag_id;
    (*exif_tags)->tags[table_idx].vals[new_size - 1].type = type;
    (*exif_tags)->tags[table_idx].vals[new_size - 1].value.uvalue = val;

    return true;
}

void
gpujpeg_exif_tags_destroy(struct gpujpeg_exif_tags* exif_tags)
{
    if (exif_tags == NULL) {
        return;
    }
    free(exif_tags->tags[0].vals);
    free(exif_tags->tags[1].vals);
}
