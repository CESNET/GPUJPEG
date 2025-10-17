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

#include <assert.h>                    // for assert
#include <ctype.h>                     // for isdigit
#include <stddef.h>
#include <stdint.h>                    // for uint8_t, uint32_t, uint16_t
#include <stdio.h>                     // for printf
#include <stdlib.h>                    // for size_t, NULL, free, abort, calloc
#include <string.h>                    // for memcpy, strlen, memmove, strcmp
#include <time.h>                      // for strftime, time, time_t
// IWYU pragma: no_include <endian.h> # via compat/endian.h.

// for strncasecmp
#ifdef _WIN32
#define strncasecmp _strnicmp
#else
#include <strings.h>
#endif

#include "../libgpujpeg/gpujpeg_common.h"  // for gpujpeg_image_parameters, gpuj...
#include "../libgpujpeg/gpujpeg_encoder.h" // for GPUJPEG_ENC_OPT_EXIF_TAG
#include "../libgpujpeg/gpujpeg_type.h"    // for gpujpeg_color_space
#include "compat/endian.h"                 // IWYU pragma: keep for htobe32
#include "compat/time.h"                   // IWYU pragma: keep for localtime_s
#include "gpujpeg_common_internal.h"       // for gpujpeg_coder, ERROR_MSG, WARN...
#include "gpujpeg_marker.h"                // for gpujpeg_marker_code
#include "gpujpeg_util.h"                  // for ARR_SIZE
#include "gpujpeg_writer.h"                // for gpujpeg_writer, gpujpeg_writer...

#define MOD_NAME "[Exif] "

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
    T_NUMERIC = 1 << 0,    // single number in .uvalue
    T_UNSIGNED = 1 << 1,   // number (or pair if rational) is unsigned
    T_BYTE_ARRAY = 1 << 2, // data stored in .csvalue
    T_RATIONAL = 1 << 3,   // 2 items of .uvalue
};

static const struct exif_tag_type_info_t
{
    unsigned size;
    const char* name;
    unsigned type_flags;
} exif_tag_type_info[] = {
      [ET_BYTE] =      {1, "BYTE",      T_NUMERIC|T_UNSIGNED },
      [ET_ASCII] =     {1, "ASCII",     T_BYTE_ARRAY         },
      [ET_SHORT] =     {2, "SHORT",     T_NUMERIC|T_UNSIGNED },
      [ET_LONG] =      {4, "LONG",      T_NUMERIC|T_UNSIGNED },
      [ET_RATIONAL] =  {8, "RATIONAL",  T_UNSIGNED|T_RATIONAL},
      [ET_UNDEFINED] = {1, "UNDEFINED", T_BYTE_ARRAY         },
      [ET_SLONG] =     {4, "SLONG"    , T_NUMERIC            },
      [ET_SRATIONAL] = {8, "SRATIONAL", T_RATIONAL           },
};

enum exif_tiff_tag {
    TAG_NONE,
    // 0th IFD TIFF Tags
    ETIFF_ORIENTATION,       ///< Image resolution in width direction (recommended)
    ETIFF_XRESOLUTION,       ///< Image resolution in width direction (mandatory)
    ETIFF_YRESOLUTION,       ///< Image resolution in height direction (mandatory)
    ETIFF_RESOLUTION_UNIT,   ///< Unit of X and Y resolution (mandatory)
    ETIFF_SOFTWARE,          ///< Software used (optional)
    ETIFF_DATE_TIME,         ///< File change date and time (recommeneded)
    ETIFF_WHITE_POINT,       ///< White Point
    ETIFF_YCBCR_POSITIONING, ///< Y and C positioning (mandatory)
    ETIFF_EXIF_IFD_POINTER,  ///< EXIF tag (mandatory)
    // 0th SubIFD Exif Private Tags
    EEXIF_EXIF_VERSION,             ///< Exif version (mandatory)
    EEXIF_COMPONENTS_CONFIGURATION, ///< Meaning of each component (mandatory)
    EEXIF_FLASHPIX_VERSION,         ///< Supported Flashpix version (mandatory)
    EEXIF_COLOR_SPACE,              ///< Color space information (mandatory)
    EEXIF_PIXEL_X_DIMENSION,        ///< Valid image width (mandatory)
    EEXIF_PIXEL_Y_DIMENSION,        ///< Valid image height (mandatory)
    NUM_TAGS
};

const struct exif_tiff_tag_info_t {
    uint16_t id;
    enum exif_tag_type type;
    unsigned count;
    const char *name;
} exif_tiff_tag_info[] = {
    [TAG_NONE]                = {0,      0,           0,  "Unknown"         },
    [ETIFF_ORIENTATION]       = {0x112,  ET_SHORT,    1,  "Orientation"     },
    [ETIFF_XRESOLUTION]       = {0x11A,  ET_RATIONAL, 1,  "XResolution"     },
    [ETIFF_YRESOLUTION]       = {0x11B,  ET_RATIONAL, 1,  "YResolution"     },
    [ETIFF_RESOLUTION_UNIT]   = {0x128,  ET_SHORT,    1,  "ResolutionUnit"  },
    [ETIFF_SOFTWARE]          = {0x131,  ET_ASCII,    0,  "Sofware"         },
    [ETIFF_DATE_TIME]         = {0x132,  ET_ASCII,    20, "DateTime"        },
    [ETIFF_WHITE_POINT]       = {0x13E,  ET_RATIONAL, 2,  "WhitePoint"      },
    [ETIFF_YCBCR_POSITIONING] = {0x213,  ET_SHORT,    1,  "YCbCrPositioning"},
    [ETIFF_EXIF_IFD_POINTER]  = {0x8769, ET_LONG,     1,  "Exif IFD Pointer"},
    // Exif SubIFD
    [EEXIF_EXIF_VERSION]             = {0x9000, ET_UNDEFINED, 4, "ExifVersion"           },
    [EEXIF_COMPONENTS_CONFIGURATION] = {0x9101, ET_UNDEFINED, 4, "ComponentConfiguration"},
    [EEXIF_FLASHPIX_VERSION]         = {0xA000, ET_UNDEFINED, 4, "FlashPixVersion"       },
    [EEXIF_COLOR_SPACE]              = {0xA001, ET_SHORT,     1, "ColorSpace"            },
    [EEXIF_PIXEL_X_DIMENSION]        = {0xA002, ET_SHORT,     1, "PixelXDimension"       }, // type can be also LONG
    [EEXIF_PIXEL_Y_DIMENSION]        = {0xA003, ET_SHORT,     1, "PixelYDimension"       }, // ditto
};

// misc constants
enum {
    ETIFF_ORIENT_HORIZONTAL = 1, // normal
    ETIFF_CENTER = 1,
    ETIFF_SRGB = 1,
    ETIFF_INCHES = 2,
    NEXT_IFD_PTR_SZ = 4,
    IFD_ITEM_SZ = 12,
    DPI_DEFAULT = 72,
    EEXIF_FIRST = 0x827A, // (Exposure time) first tag id of Exif Private Tags
    TIFF_HDR_TAG = 0x002a,
};

////////////////////////////////////////////////////////////////////////////////
//                                   WRITER                                   //
////////////////////////////////////////////////////////////////////////////////
union value_u {
    const uint32_t *uvalue;
    const char* csvalue; // ET_STRING (must be 0-terminated) or ET_UNDEFINED
};

static void
write_exif_tag(struct gpujpeg_writer* writer, enum exif_tag_type type, uint16_t tag_id, unsigned count,
               union value_u val, const uint8_t* start, uint8_t** end)
{
    assert(type < ET_END);
    unsigned size = exif_tag_type_info[type].size;

    if ( type == ET_ASCII ) {
        if ( count == 0 ) {
            count = strlen(val.csvalue) + 1;
        }
        else {
            assert(count == strlen(val.csvalue) + 1);
        }
    }
    assert(count > 0);
    assert(size > 0);

    gpujpeg_writer_emit_2byte(writer, tag_id);
    gpujpeg_writer_emit_2byte(writer, type);
    gpujpeg_writer_emit_4byte(writer, count);

    // we actually store rational numbers as a pair
    if ( (exif_tag_type_info[type].type_flags & T_RATIONAL) != 0 ) {
        count *= 2;
        size /= 2;
    }

    const bool val_longer_than_4b = size * count > 4;
    uint8_t* return_pos = NULL;
    if ( val_longer_than_4b ) {
        gpujpeg_writer_emit_4byte(writer, *end - start);
        return_pos = writer->buffer_current;
        writer->buffer_current = *end;
    }

    if ( (exif_tag_type_info[type].type_flags & T_BYTE_ARRAY) != 0 ) {
        assert(size == 1);
        for ( unsigned i = 0; i < count; ++i ) {
            gpujpeg_writer_emit_byte(writer, val.csvalue[i]);
        }
    }
    else {
        for ( unsigned c = 0; c < count; ++c ) {
            for ( unsigned i = 0; i < size; ++i ) { // Exif 4.6.2 - left aligned value; we use big endian
                gpujpeg_writer_emit_byte(writer, (val.uvalue[c] >> 8 * (size - i - 1)) & 0xFFU);
            }
        }
    }

    if ( val_longer_than_4b ) {
        *end += (size_t)size * count;
        writer->buffer_current = return_pos;
    }
    else {
        // padding
        for ( unsigned i = size * count; i < 4; ++i ) {
            gpujpeg_writer_emit_byte(writer, 0);
        }
    }
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
    size_t val_count;
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
            value.uvalue = (uint32_t[]) {end - start};
        }
        const struct exif_tiff_tag_info_t* t = &exif_tiff_tag_info[info->tag];
        assert(t->id >= last_tag_id);
        last_tag_id = t->id;
        write_exif_tag(writer, t->type, t->id, exif_tiff_tag_info[info->tag].count, value, start, &end);
    }
    if ( custom_tags != NULL ) { // add user custom tags
        for ( unsigned i = 0; i < custom_tags->count; ++i ) {
            write_exif_tag(writer, custom_tags->vals[i].type, custom_tags->vals[i].tag_id,
                           custom_tags->vals[i].val_count, custom_tags->vals[i].value, start, &end);
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
remove_overriden(size_t count, struct tag_value tags[static count], const struct custom_exif_tags custom_tags[static 1])
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
gpujpeg_write_0th(struct gpujpeg_writer* writer, const uint8_t* start,
                   const struct custom_exif_tags* custom_tags)
{
    char date_time[] = "    :  :     :  :  "; // unknown val by Exif 2.3
    time_t now = time(NULL);
    struct tm buf;
    (void) strftime(date_time, sizeof date_time, "%Y:%m:%d %H:%M:%S", localtime_s(&now, &buf));
    struct tag_value tags[] = {
        {ETIFF_ORIENTATION,       {.uvalue = (uint32_t[]){ETIFF_ORIENT_HORIZONTAL}}},
        {ETIFF_XRESOLUTION,       {.uvalue = (uint32_t[]){DPI_DEFAULT, 1}}         },
        {ETIFF_YRESOLUTION,       {.uvalue = (uint32_t[]){DPI_DEFAULT, 1}}         },
        {ETIFF_RESOLUTION_UNIT,   {.uvalue = (uint32_t[]){ETIFF_INCHES}}           },
        {ETIFF_SOFTWARE,          {.csvalue = "GPUJPEG"}                           },
        {ETIFF_DATE_TIME ,        {.csvalue = date_time}                           },
        {ETIFF_YCBCR_POSITIONING, {.uvalue = (uint32_t[]){ETIFF_CENTER}}           },
        {ETIFF_EXIF_IFD_POINTER,  {0}                                }, // value will be set later
    };
    size_t tag_count = ARR_SIZE(tags);
    tag_count = remove_overriden(tag_count, tags, custom_tags);
    gpujpeg_write_ifd(writer, start, tag_count, tags, custom_tags);
}

static void
gpujpeg_write_exif_ifd(struct gpujpeg_writer* writer, const uint8_t* start,
                       const struct gpujpeg_image_parameters* param_image, const struct custom_exif_tags* custom_tags)
{
    struct tag_value tags[] = {
        {EEXIF_EXIF_VERSION,             {.csvalue = "0230"}                                        }, // 2.30
        {EEXIF_COMPONENTS_CONFIGURATION, {.csvalue = "\1\2\3\0"}                                    }, // YCbCr
        {EEXIF_FLASHPIX_VERSION,         {.csvalue = "0100"}                                        },
        {EEXIF_COLOR_SPACE,              {.uvalue = (uint32_t[]){ETIFF_SRGB}}                       },
        {EEXIF_PIXEL_X_DIMENSION,        {.uvalue = (uint32_t[]){param_image->width}} },
        {EEXIF_PIXEL_Y_DIMENSION,        {.uvalue = (uint32_t[]){param_image->height}}},
    };
    size_t tag_count = ARR_SIZE(tags);
    tag_count = remove_overriden(tag_count, tags, custom_tags);
    gpujpeg_write_ifd(writer, start, ARR_SIZE(tags), tags, custom_tags);
}

/// writes EXIF APP1 marker
void
gpujpeg_writer_write_exif(struct gpujpeg_writer* writer, const struct gpujpeg_parameters* param,
                          const struct gpujpeg_image_parameters* param_image,
                          const struct gpujpeg_exif_tags* custom_tags)
{
    if ( param->color_space_internal != GPUJPEG_YCBCR_BT601_256LVLS ) {
        WARN_MSG("[Exif] Color space %s currently not recorded, assumed %s (report)\n",
                 gpujpeg_color_space_get_name(param->color_space_internal),
                 gpujpeg_color_space_get_name(GPUJPEG_YCBCR_BT601_256LVLS));
    }
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

    gpujpeg_writer_emit_2byte(writer, TIFF_HDR_TAG); // TIFF header
    gpujpeg_writer_emit_4byte(writer, 0x08); // IFD offset - follows immediately

    const struct custom_exif_tags* tiff_tags = &(const struct custom_exif_tags){.count = 0};
    const struct custom_exif_tags* exif_tags = &(const struct custom_exif_tags){.count = 0};
    if ( custom_tags != NULL ) {
        tiff_tags = &custom_tags->tags[CT_TIFF];
        exif_tags = &custom_tags->tags[CT_EXIF];
    }

    gpujpeg_write_0th(writer, start, tiff_tags);
    gpujpeg_write_exif_ifd(writer, start, param_image, exif_tags);

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

static void
usage()
{
    printf("Exif value syntax:\n"
           "\t" GPUJPEG_ENC_OPT_EXIF_TAG "=<ID>:<type>=<value>\n"
           "\t" GPUJPEG_ENC_OPT_EXIF_TAG "=<name>=<value>\n"
           "\t\tname must be a tag name known to GPUJPEG\n");
    printf("\n");
    printf("If mulitple numeric values required, separate with a comma; rationals are in format num/den.\n");
    printf("UNDEFINED and ASCII should be raw strings.\n");
    printf("\n");
    printf("recognized tag name (type, count):\n");
    for ( unsigned i = 0; i < ARR_SIZE(exif_tiff_tag_info); ++i ) {
        printf("\t- %s (%s, %u)\n", exif_tiff_tag_info[i].name, exif_tag_type_info[exif_tiff_tag_info[i].type].name,
               exif_tiff_tag_info[i].count);
    }
}

/**
 * add user-provided Exif tag
 */
bool
gpujpeg_exif_add_tag(struct gpujpeg_exif_tags** exif_tags, const char* cfg)
{
    if (strcmp(cfg, "help") == 0) {
        usage();
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

    endptr += 1;
    void* val_alloc = NULL;
    size_t val_count = 0;
    if ( (exif_tag_type_info[type].type_flags & T_BYTE_ARRAY) != 0 ) {
        char* val_str = strdup(endptr);
        val_alloc = val_str;
        val_count = strlen(val_str);
        endptr += val_count;
        if (type == ET_ASCII) {
            val_count += 1; // include '\0'
        }
    }
    else {
        do {
            if ( *endptr == ',' ) {
                endptr += 1;
            }
            if ( (exif_tag_type_info[type].type_flags & T_NUMERIC) != 0U ) {
                unsigned long long val = strtoull(endptr, &endptr, 0);
                val_count += 1;
                uint32_t* val_a = realloc(val_alloc, val_count * sizeof *val_a);
                val_a[val_count - 1] = val;
                val_alloc = val_a;
            }
            else if ( (exif_tag_type_info[type].type_flags & T_RATIONAL) != 0U ) {
                unsigned long long num = strtoull(endptr, &endptr, 0);
                if ( *endptr != '/' ) {
                    ERROR_MSG("[Exif] Malformed rational, expected '/', got '%c'!\n", *endptr);
                }
                endptr += 1;
                unsigned long long den = strtoull(endptr, &endptr, 0);
                val_count += 1;
                uint32_t* val_a = realloc(val_alloc, val_count * 2 * sizeof *val_a);
                val_a[2 * (val_count - 1)] = num;
                val_a[(2 * (val_count - 1)) + 1] = den;
                val_alloc = val_a;
            }
        } while ( *endptr == ',' );
    }

    if ( *endptr != '\0' ) {
        free(val_alloc);
        ERROR_MSG("Trainling data in Exif value: %s\n", endptr);
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
    assert(val_alloc != NULL);
    (*exif_tags)->tags[table_idx].vals[new_size - 1].value.uvalue = val_alloc;
    (*exif_tags)->tags[table_idx].vals[new_size - 1].val_count = val_count;

    return true;
}

void
gpujpeg_exif_tags_destroy(struct gpujpeg_exif_tags* exif_tags)
{
    if ( exif_tags == NULL ) {
        return;
    }
    for ( unsigned i = 0; i < CT_NUM; ++i ) {
        for (unsigned j = 0; i < exif_tags->tags[i].count; ++i) {
            free((void *) exif_tags->tags[i].vals[j].value.uvalue);
        }
        free(exif_tags->tags[i].vals);
    }
    free(exif_tags);
}

////////////////////////////////////////////////////////////////////////////////
//                                   READER                                   //
////////////////////////////////////////////////////////////////////////////////
static enum exif_tiff_tag
get_tag_from_id(uint16_t tag_id)
{
    for ( unsigned i = TAG_NONE + 1; i < NUM_TAGS; ++i ) {
        if ( exif_tiff_tag_info[i].id == tag_id ) {
            return i;
        }
    }
    return TAG_NONE;
}

static uint8_t
read_byte(uint8_t** image)
{
    return *(*image)++;
}
static uint16_t
read_2byte_be(uint8_t** image) {
    uint16_t ret = (*image)[0] << 8 | (*image)[1];
    *image += 2;
    return ret;
}
static uint32_t
read_4byte_be(uint8_t** image) {
    uint32_t ret = (*image)[0] << 24 | (*image)[1] << 16 | (*image)[2] << 8 | (*image)[3];
    *image += 4;
    return ret;
}
static uint16_t
read_2byte_le(uint8_t** image) {
    uint16_t ret = (*image)[1] << 8 | (*image)[0];
    *image += 2;
    return ret;
}
static uint32_t
read_4byte_le(uint8_t** image) {
    uint32_t ret = (*image)[3] << 24 | (*image)[2] << 16 | (*image)[1] << 8 | (*image)[0];
    *image += 4;
    return ret;
}

static void
read_0th_ifd(uint8_t** image, const uint8_t* image_end, int verbose, uint16_t (*read_2byte)(uint8_t**),
             uint32_t (*read_4byte)(uint8_t**))
{
    if ( *image + 2 > image_end ) {
        WARN_MSG("Unexpected end of file!\n");
        return;
    }
    size_t num_interop = read_2byte(image);
    if ( *image + num_interop * IFD_ITEM_SZ > image_end ) {
        WARN_MSG(MOD_NAME "Insufficient space to hold %zu IFD0 items!\n", num_interop);
        return;
    }
    DEBUG_MSG(verbose, "Found %zu IFD0 items.\n", num_interop);

    for ( unsigned i = 0; i < num_interop; ++i ) {
        uint16_t tag_id = read_2byte(image);
        uint16_t type = read_2byte(image);
        uint32_t count = read_4byte(image);
        uint32_t val = read_4byte(image);
        unsigned size = 0;
        enum exif_tiff_tag tag = get_tag_from_id(tag_id);
        const char* type_name = "WRONG";
        if ( type < ET_END && exif_tag_type_info[type].name != NULL ) {
            type_name = exif_tag_type_info[type].name;
            if ( (exif_tag_type_info[type].type_flags & T_NUMERIC) != 0 ) {
                if (read_2byte == read_2byte_be) {
                    val >>= 8 * exif_tag_type_info[type].size;
                }
            }
            size = exif_tag_type_info[type].size;
        }
        DEBUG_MSG(verbose, MOD_NAME "Found IFD0 tag %s (%#x) type %s: count=%u, %s=%#x\n", exif_tiff_tag_info[tag].name,
                  tag_id, type_name, count, count * size <= 4 ? "value" : "offset", val);
        if ( tag == ETIFF_ORIENTATION && val != ETIFF_ORIENT_HORIZONTAL ) {
            WARN_MSG(MOD_NAME "Orientation %d not handled!\n", val);
        }
    }

    DEBUG_MSG(verbose, MOD_NAME "Skipping data after IFD0 marker (eg. Exif SubIFD)\n");
    // TODO: Exif Private Tags SubIFD
}

/**
 * parse the header
 *
 * Currently only the basic validity is cheecked. If verbosity is set to higher value,
 * the basic tags from 0th IFD are printed out (not Exif SubIFD).
 *
 * JPEG Orientation is checked and of not horizontal, warning is issued.
 */
void
gpujpeg_exif_parse(uint8_t** image, const uint8_t* image_end, int verbose)
{
#define HANDLE_ERROR(...)                                                                                              \
    WARN_MSG(__VA_ARGS__);                                                                                             \
    *image = image_start + length;                                                                                     \
    return

    enum {
        EXIF_HDR_MIN_LEN = 18, // with empty 0th IFD
    };
    assert(image_end - *image > 2);
    uint8_t *image_start = *image;
    uint16_t length = read_2byte_be(image);
    if (length > image_end - *image - 2) {
        HANDLE_ERROR("Unexpected end of file!\n");
    }
    if (length < EXIF_HDR_MIN_LEN) {
        HANDLE_ERROR("Insufficient Exif header length %u!\n", (unsigned)length);
    }
    uint8_t exif[5];
    for (int i = 0; i < 5; ++i) {
        exif[i] = read_byte(image);
    }
    assert(strncmp((char *) exif, "Exif", sizeof exif) == 0); // otherwise fn shouldn't be called
    read_byte(image); // drop (padding)

    uint8_t* const base = *image;

    uint16_t endian_tag = read_2byte_be(image);
    uint16_t (*read_2byte)(uint8_t **) = read_2byte_be;
    uint32_t (*read_4byte)(uint8_t **) = read_4byte_be;

    if ( endian_tag == ('I' << 8 | 'I') ) {
        DEBUG_MSG(verbose, "Little endian Exif detected.\n");
        read_2byte = read_2byte_le;
        read_4byte = read_4byte_le;
    }
    else if ( endian_tag == ('M' << 8 | 'M') ) {
        DEBUG_MSG(verbose, "Big endian Exif detected.\n");
    }
    else {
        HANDLE_ERROR("Unexpected endianity!\n");
    }
    uint16_t tiff_hdr = read_2byte(image);
    if (tiff_hdr != TIFF_HDR_TAG) {
        HANDLE_ERROR("Wrong TIFF tag, expected 0x%04x!\n", TIFF_HDR_TAG);
    }

    uint32_t offset = read_4byte(image); // 0th IFD offset
    *image = base + offset;
    read_0th_ifd(image, image_end, verbose, read_2byte, read_4byte);

    *image = image_start + length;
#undef HANDLE_ERROR
}
