/**
 * @file
 * Copyright (c) 2011-2023, CESNET z.s.p.o
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

#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(_MSC_VER)
    #include "gpujpeg_getopt_mingw.h"
#else
    #include <getopt.h>
#endif
#include "../libgpujpeg/gpujpeg.h"
#include "../libgpujpeg/gpujpeg_common.h"

#define USE_IF_NOT_NULL_ELSE(cond, alt_val) (cond) ? (cond) : (alt_val)

static void
print_help(void)
{
    printf("gpujpegtool [options] input.rgb output.jpg [input2.rgb output2.jpg ...]\n"
           "   -h, --help             print help\n"
           "   -v, --verbose          verbose output (multiply to increase verbosity - max 3) \n"
           "   -D, --device           set cuda device id (default 0)\n"
           "   -L, --device-list      list cuda devices\n"
           "\n");
    printf("   -s, --size             set input image size in pixels, e.g. 1920x1080\n"
           "   -f, --pixel-format     set input/output image pixel format, one of the\n"
           "                          following (example in parenthesis):\n");
    gpujpeg_print_pixel_formats();
    printf("\n"
           "   -c, --colorspace       set input/output image colorspace, e.g. rgb, ycbcr-jpeg (full\n"
           "                          range BT.601), ycbcr-bt601 (limited 601), ycbcr-bt709 (limited)\n"
           "\n");
    printf("   -q, --quality          set JPEG encoder quality level 0-100 (default 75)\n"
           "   -r, --restart          set JPEG encoder restart interval (default 8)\n"
           "   -S, --subsampled[=<s>] set JPEG encoder chroma subsampling in J:a:b[:A] format (default 4:2:0)\n"
           "   -i  --interleaved      set JPEG encoder to use interleaved stream\n"
           "   -g  --segment-info     set JPEG encoder to use segment info in stream\n"
           "                          for fast decoding\n"
           "\n");
    printf("   -e, --encode           perform JPEG encoding\n"
           "   -d, --decode           perform JPEG decoding\n"
           "   -C, --convert          convert input image to output image (change\n"
           "                          color space and/or sampling factor)\n"
           "   -R, --component-range  show samples range for each component in image\n"
           "\n");
    printf("   -n  --iterate          perform encoding/decoding in specified number of\n"
           "                          iterations for each image\n"
           "   -o  --use-opengl       use an OpenGL texture as input/output\n"
           "   -I  --info             print JPEG file info\n"
           "   -a  --alpha            encode/decode alpha channel (otherwise stripped)\n"
           "   -N  --native           create native JPEG (Adobe RGB for RGB, SPIFF for Y709;\n"
           "                                              may be incompatible with some decoders;\n"
           "                                              works also for decoding)\n");
    printf("recognized raw input/output file extensions: rgb, yuv, pnm... (use`gpujpegtool exts` for the full list)\n");
}

static void
print_gpujpeg_image_parameters(struct gpujpeg_image_parameters params_image, bool oneline,
                               const char* subsampling_details)
{
    const char *sep_str = oneline ? " " : "\n";
    const char *separator = ""; // first time empty
    if ( params_image.width ) {
        printf("%s%d", oneline ? "" : "width: ", params_image.width);
    }
    if ( params_image.height ) {
        printf("%s%d%s", oneline ? "x" : "\nheight: ", params_image.height, (separator = sep_str));
    }
    if ( params_image.pixel_format != GPUJPEG_PIXFMT_NONE ) {
        printf("%s%s",
               oneline ? "" : "internal representation: ", gpujpeg_pixel_format_get_name(params_image.pixel_format));
        if ( !oneline && subsampling_details != NULL ) {
            printf(" (%s)", subsampling_details);
        }
        printf("%s", (separator = sep_str));
    }
    if ( params_image.color_space ) {
        printf("%s%s", oneline ? "" : "color space: ", gpujpeg_color_space_get_name(params_image.color_space));
    }
    printf("\n");
}

static int print_image_info_jpeg(const char *filename, int verbose) {
    FILE *f = fopen(filename, "rb");
    if (!f) {
        perror("Cannot open");
        return 1;
    }
    fseek(f, 0L, SEEK_END);
    long int len = ftell(f);
    printf("size: %lu B\n", len);
    fseek(f, 0L, SEEK_SET);
    uint8_t *jpeg = malloc(len);
    size_t ret = fread(jpeg, len, 1, f);
    fclose(f);
    if (ret == 0) {
        fprintf(stderr, "Cannot read image contents.\n");
        return 1;
    }
    struct gpujpeg_image_parameters params_image = gpujpeg_default_image_parameters();
    params_image.pixel_format = GPUJPEG_PIXFMT_NONE;
    struct gpujpeg_parameters params =  gpujpeg_default_parameters();
    params.verbose = verbose;
    int segment_count = 0;
    if (gpujpeg_decoder_get_image_info(jpeg, len, &params_image, &params, &segment_count) == 0) {
        print_gpujpeg_image_parameters(params_image, false,
                                       gpujpeg_subsampling_get_name(params.comp_count, params.sampling_factor));
        printf("interleaved: %d\n", params.interleaved);
        if ( segment_count ) {
            printf("segment count: %d (DRI = %d)\n", segment_count, params.restart_interval);
        }
    }
    free(jpeg);

    return 0;
}

static int print_image_info(const char *filename, int verbose) {
    if (!filename) {
        fprintf(stderr, "Missing filename!\n");
        return 1;
    }
    printf("name: %s\n", filename);
    enum gpujpeg_image_file_format format = gpujpeg_image_get_file_format(filename);
    if (format == GPUJPEG_IMAGE_FILE_JPEG ) {
        return print_image_info_jpeg(filename, verbose);
    }
    struct gpujpeg_image_parameters param_image = { 0 };
    if (gpujpeg_image_get_properties(filename, &param_image, 1) != 0) {
        fprintf(stderr, "Error getting raw image %s info!\n", filename);
        return 1;
    }
    print_gpujpeg_image_parameters(param_image, false, NULL);
    return 0;
}

struct options
{
    gpujpeg_sampling_factor_t subsampling;
    bool native_file_format;
    bool keep_alpha;
};

static bool
adjust_params(struct gpujpeg_parameters* param, struct gpujpeg_image_parameters* param_image, const char* in,
              const char* out, bool encode, const struct options* opts)
{
    // if possible, read properties from file
    struct gpujpeg_image_parameters file_param_image = { 0, 0, GPUJPEG_NONE, GPUJPEG_PIXFMT_NONE, 0 };
    const char *raw_file = encode ? in : out;
    gpujpeg_image_get_properties(raw_file, &file_param_image, encode);
    param_image->width = USE_IF_NOT_NULL_ELSE(param_image->width, file_param_image.width);
    param_image->height = USE_IF_NOT_NULL_ELSE(param_image->height, file_param_image.height);
    if ( param_image->color_space == GPUJPEG_NONE ) {
        param_image->color_space = USE_IF_NOT_NULL_ELSE(file_param_image.color_space, GPUJPEG_RGB);
    }
    if ( param_image->pixel_format == GPUJPEG_PIXFMT_NONE ) {
        param_image->pixel_format = file_param_image.pixel_format;
        if (!encode && !opts->keep_alpha && file_param_image.pixel_format == GPUJPEG_PIXFMT_AUTODETECT) {
            param_image->pixel_format = GPUJPEG_PIXFMT_NO_ALPHA; // keep alpha only if requested
        }
    }
    if ( opts->keep_alpha && encode && param_image->pixel_format == GPUJPEG_4444_U8_P0123 ) {
        gpujpeg_parameters_chroma_subsampling(param, GPUJPEG_SUBSAMPLING_4444);
    }

    if (encode && (param_image->width <= 0 || param_image->height <= 0)) {
        fprintf(stderr, "Image dimensions must be set to nonzero values!\n");
        return false;
    }
    if (encode && param_image->pixel_format == GPUJPEG_PIXFMT_NONE) {
        fprintf(stderr, "Pixel format must be set!\n");
        return false;
    }

    return true;
}

static enum gpujpeg_pixel_format
parse_pixel_format(const char *arg)
{
    if (strcmp(arg, "help") == 0) {
        printf("Available pixel formats:\n");
        gpujpeg_print_pixel_formats();
        return GPUJPEG_PIXFMT_NONE;
    }
    const enum gpujpeg_pixel_format ret = gpujpeg_pixel_format_by_name(arg);
    if ( ret == GPUJPEG_PIXFMT_NONE ) {
        fprintf(stderr, "Unknown pixel format '%s'!\n", arg);
    }
    return ret;
}

#ifndef GIT_REV
#define GIT_REV "unknown"
#endif

int
main(int argc, char *argv[])
{
    printf("GPUJPEG rev %s\n", strlen(GIT_REV) > 0 ? GIT_REV : "unknown");

    int ret = EXIT_SUCCESS;
    // Default coder parameters
    struct gpujpeg_parameters param;
    gpujpeg_set_default_parameters(&param);
    param.perf_stats = 1;

    // Default image parameters
    struct gpujpeg_image_parameters param_image;
    gpujpeg_image_set_default_parameters(&param_image);

    // Original image parameters in conversion
    struct gpujpeg_image_parameters param_image_original;
    gpujpeg_image_set_default_parameters(&param_image_original);

    // Other parameters
    int device_id = 0;
    _Bool encode = 0;
    _Bool decode = 0;
    _Bool convert = 0;
    _Bool component_range = 0;
    int iterate = 1;
    _Bool use_opengl = 0;

    // Flags
    struct options opts = {.subsampling = GPUJPEG_SUBSAMPLING_UNKNOWN,
                           .native_file_format = false,
                           .keep_alpha = false};

    int rc;

    param_image.color_space = GPUJPEG_NONE;
    param_image.pixel_format = GPUJPEG_PIXFMT_NONE;
    param.restart_interval = RESTART_AUTO;

    // Parse command line
    struct option longopts[] = {
        {"alpha",                   no_argument,       0, 'a'},
        {"help",                    no_argument,       0, 'h'},
        {"verbose",                 optional_argument, 0, 'v'},
        {"device",                  required_argument, 0, 'D'},
        {"device-list",             no_argument,       0, 'L' },
        {"size",                    required_argument, 0, 's'},
        {"pixel-format",         required_argument, 0, 'f'},
        {"colorspace",              required_argument, 0, 'c'},
        {"quality",                 required_argument, 0, 'q'},
        {"restart",                 required_argument, 0, 'r'},
        {"segment-info",            optional_argument, 0, 'g' },
        {"subsampled",              optional_argument, 0, 'S' },
        {"interleaved",             optional_argument, 0, 'i'},
        {"encode",                  no_argument,       0, 'e'},
        {"decode",                  no_argument,       0, 'd'},
        {"convert",                 no_argument,       0, 'C' },
        {"component-range",         no_argument,       0,  'R' },
        {"iterate",                 required_argument, 0,  'n' },
        {"use-opengl",              no_argument,       0,  'o' },
        {"info",                    required_argument, 0,  'I' },
        {"native",                  no_argument,       0,  'N' },
        {0}
    };
    int ch = '\0';
    int optindex = 0;
    char* pos = 0;
    while ( (ch = getopt_long(argc, argv, "CLRahvD:s:f:c:q:r:g::i::edn:oI:NS::", longopts, &optindex)) != -1 ) {
        switch (ch) {
        case 'a':
            opts.keep_alpha = true;
            break;
        case 'h':
            print_help();
            return 0;
        case 'v':
            if (optarg) {
                param.verbose = atoi(optarg); // NOLINT(cert-err34-c): not important
            } else {
                param.verbose += 1;
            }
            break;
        case 's':
            pos = strstr(optarg, "x");
            if ( pos == NULL || pos == optarg ) {
                fprintf(stderr, "Incorrect image size '%s'! Use a format 'WxH'.\n", optarg);
                return -1;
            }
            param_image.width = atoi(optarg);
            param_image.height = atoi(pos + 1);
            break;
        case 'c':
            param_image.color_space = gpujpeg_color_space_by_name(optarg);
            if ( param_image.color_space == GPUJPEG_NONE ) {
                if ( strcmp(optarg, "help") == 0 ) {
                    return EXIT_SUCCESS;
                }
                fprintf(stderr, "Colorspace '%s' is not available!\n", optarg);
                return EXIT_FAILURE;
            }
            break;
        case 'f':
            param_image.pixel_format = parse_pixel_format(optarg);
            if (param_image.pixel_format == GPUJPEG_PIXFMT_NONE) {
                return 1;
            }
            break;
        case 'q':
            param.quality = atoi(optarg);
            if ( param.quality < 0 )
                param.quality = 0;
            if ( param.quality > 100 )
                param.quality = 100;
            break;
        case 'r':
            param.restart_interval = atoi(optarg);
            if ( param.restart_interval < RESTART_AUTO ) {
                fprintf(stderr, "Wrong restart interval '%s'!\n", optarg);
                return 1;
            }
            break;
        case 'g':
            if ( optarg == NULL || strcmp(optarg, "true") == 0 || atoi(optarg) )
                param.segment_info = 1;
            else
                param.segment_info = 0;
            break;
        case 'S':
            if (optarg == NULL) {
                opts.subsampling = GPUJPEG_SUBSAMPLING_420;
                break;
            }
            opts.subsampling = gpujpeg_subsampling_from_name(optarg);
            if ( opts.subsampling == GPUJPEG_SUBSAMPLING_UNKNOWN ) {
                fprintf(stderr, "Unknown subsampling '%s'!\n", optarg);
                return 1;
            }
            break;
        case 'L':
            return gpujpeg_print_devices_info() < 0 ? 1 : 0;
        case 'i':
            if ( optarg == NULL || strcmp(optarg, "true") == 0 || atoi(optarg) )
                param.interleaved = 1;
            else
                param.interleaved = 0;
            break;
        case 'e':
            encode = 1;
            break;
        case 'd':
            decode = 1;
            break;
        case 'D':
            device_id = atoi(optarg);
            break;
        case 'C':
            convert = 1;
            memcpy(&param_image_original, &param_image, sizeof(struct gpujpeg_image_parameters));
            break;
        case 'R':
            component_range = 1;
            break;
        case 'n':
            iterate = atoi(optarg);
            break;
        case 'o':
            use_opengl = 1;
            break;
        case 'I':
            return print_image_info(optarg, param.verbose);
        case 'N':
            opts.native_file_format = true;
            break;
        case '?':
            return -1;
        default:
            fprintf(stderr, "Unrecognized option '%c' (code 0%o)\n", ch, ch);
            print_help();
            return -1;
        }
    }
    argc -= optind;
    argv += optind;

    if ( opts.subsampling != GPUJPEG_SUBSAMPLING_UNKNOWN ) {
        gpujpeg_parameters_chroma_subsampling(&param, opts.subsampling);
    }

    // Show info about image samples range
    if ( component_range == 1 ) {
        // For each image
        for ( int index = 0; index < argc; index++ ) {
            gpujpeg_image_range_info(argv[index], param_image.width, param_image.height, param_image.pixel_format);
        }
        return 0;
    }

    if ( argv[0] != NULL && strcmp(argv[0], "exts") == 0 ) {
        gpujpeg_image_get_file_format(".help");
        return 0;
    }

    // Source image and target image must be presented
    if ( argc < 2 || argc % 2 != 0 ) {
        fprintf(stderr, "Please supply source and destination image filename(s)!\n");
        print_help();
        return -1;
    }

    // Init device
    int flags = GPUJPEG_VERBOSE;
    struct gpujpeg_opengl_context *gl_context = NULL;
    if ( use_opengl ) {
        flags |= GPUJPEG_OPENGL_INTEROPERABILITY;
        if ( gpujpeg_opengl_init(&gl_context) != 0 ) {
            fprintf(stderr, "Cannot initialize OpenGL context!\n");
            return -1;
        }
    }
    if ( gpujpeg_init_device(device_id, flags) != 0 )
        return -1;

    // Convert
    if ( convert == 1 ) {
        // Encode images
        for ( int index = 0; index < argc; index += 2 ) {
            // Get input and output image
            const char* input = argv[index];
            const char* output = argv[index + 1];
            // Perform conversion
            gpujpeg_image_convert(input, output, param_image_original, param_image);
        }
        return 0;
    }

    // Detect action if none is specified
    if ( encode == 0 && decode == 0 ) {
        enum gpujpeg_image_file_format input_format = gpujpeg_image_get_file_format(argv[0]);
        enum gpujpeg_image_file_format output_format = gpujpeg_image_get_file_format(argv[1]);
        if ( (input_format >= GPUJPEG_IMAGE_FILE_RAW) && output_format == GPUJPEG_IMAGE_FILE_JPEG ) {
            encode = 1;
        } else if ( input_format == GPUJPEG_IMAGE_FILE_JPEG && (output_format >= GPUJPEG_IMAGE_FILE_RAW) ) {
            decode = 1;
        } else {
            fprintf(stderr, "Action can't be recognized for specified images!\n");
            fprintf(stderr, "You must specify --encode or --decode option!\n");
            return -1;
        }
    }

    struct gpujpeg_parameters param_saved = param;
    struct gpujpeg_image_parameters param_image_saved = param_image;

    if ( encode == 1 ) {
        // Create OpenGL texture
        struct gpujpeg_opengl_texture* texture = NULL;
        if ( use_opengl ) {
            assert(param_image.pixel_format == GPUJPEG_444_U8_P012);
            int texture_id = gpujpeg_opengl_texture_create(param_image.width, param_image.height, NULL);
            assert(texture_id != 0);

            texture = gpujpeg_opengl_texture_register(texture_id, GPUJPEG_OPENGL_TEXTURE_READ);
            assert(texture != NULL);
        }

        // Create encoder
        struct gpujpeg_encoder* encoder = gpujpeg_encoder_create(NULL);
        if ( encoder == NULL ) {
            fprintf(stderr, "Failed to create encoder!\n");
            return -1;
        }

        // Encode images
        for ( int index = 0; index < argc; index += 2 ) {
            // Get and check input and output image
            const char* input = argv[index];
            char* output = argv[index + 1];
            enum gpujpeg_image_file_format input_format = gpujpeg_image_get_file_format(input);
            enum gpujpeg_image_file_format output_format = gpujpeg_image_get_file_format(output);
            if ( (input_format >= GPUJPEG_IMAGE_FILE_RAW) == 0 ) {
                fprintf(stderr, "[Warning] Encoder input file [%s] should be raw image (*.rgb, *.yuv, *.r, *.pnm)!\n", input);
                if ( input_format & GPUJPEG_IMAGE_FILE_JPEG ) {
                    return -1;
                }
            }
            if ( output_format != GPUJPEG_IMAGE_FILE_JPEG ) {
                fprintf(stderr, "[%s] Encoder output file [%s] should be JPEG image (*.jpg)!\n",
                        output_format == GPUJPEG_IMAGE_FILE_UNKNOWN ? "Warning" : "Error", output);
                if ( output_format != GPUJPEG_IMAGE_FILE_UNKNOWN ) {
                    ret = EXIT_FAILURE; continue;
                }
            }

            param = param_saved;
            param_image = param_image_saved;
            if ( !adjust_params(&param, &param_image, input, output, encode, &opts) ) {
                ret = EXIT_FAILURE; continue;
            }

            if ( opts.native_file_format ) {
                param.color_space_internal = param_image.color_space;
            }

            // Encode image
            double duration = gpujpeg_get_time();
            printf("\nEncoding Image [%s]: ", input);
            print_gpujpeg_image_parameters(param_image, true, NULL);

            // Load image
            size_t image_size = gpujpeg_image_calculate_size(&param_image);
            uint8_t* image = NULL;
            if ( gpujpeg_image_load_from_file(input, &image, &image_size) != 0 ) {
                fprintf(stderr, "Failed to load image [%s]!\n", argv[index]);
                ret = EXIT_FAILURE; continue;
            }

            duration = gpujpeg_get_time() - duration;
            printf("Load Image:          %10.2f ms\n", duration * 1000.0);

            // Prepare encoder input
            struct gpujpeg_encoder_input encoder_input;
            if ( use_opengl ) {
                gpujpeg_opengl_texture_set_data(texture->texture_id, image);
                gpujpeg_encoder_input_set_texture(&encoder_input, texture);
            } else {
                gpujpeg_encoder_input_set_image(&encoder_input, image);
            }

            // Encode image
            uint8_t* image_compressed = NULL;
            size_t image_compressed_size = 0;
            for ( int iteration = 0; iteration < iterate; iteration++ ) {
                if ( iterate > 1 ) {
                    printf("\nIteration #%d:\n", iteration + 1);
                }

                rc = gpujpeg_encoder_encode(encoder, &param, &param_image, &encoder_input, &image_compressed, &image_compressed_size);
                if ( rc != GPUJPEG_NOERR ) {
                    fprintf(stderr, "Failed to encode image [%s]!\n", argv[index]);
                    ret = EXIT_FAILURE; continue;
                }
            }

            duration = gpujpeg_get_time();

            // Save image
            if ( image_compressed == NULL ||
                 gpujpeg_image_save_to_file(output, image_compressed, image_compressed_size, &param_image) != 0 ) {
                fprintf(stderr, "Failed to save image [%s]!\n", argv[index]);
                ret = EXIT_FAILURE;
            }

            duration = gpujpeg_get_time() - duration;
            printf("Save Image:          %10.4f ms\n", duration * 1000.0);
            printf("Compressed Size:     %10.zu bytes [%s]\n", image_compressed_size, output);

            // Destroy image
            gpujpeg_image_destroy(image);
        }

        // Destroy OpenGL texture
        if ( use_opengl ) {
            int texture_id = texture->texture_id;
            gpujpeg_opengl_texture_unregister(texture);
            gpujpeg_opengl_texture_destroy(texture_id);
        }

        // Destroy encoder
        gpujpeg_encoder_destroy(encoder);
    }

    if ( decode == 1 ) {
        // Create OpenGL texture
        struct gpujpeg_opengl_texture* texture = NULL;
        if ( use_opengl ) {
            assert(param_image.pixel_format == GPUJPEG_444_U8_P012);
            int texture_id = gpujpeg_opengl_texture_create(param_image.width, param_image.height, NULL);
            assert(texture_id != 0);

            texture = gpujpeg_opengl_texture_register(texture_id, GPUJPEG_OPENGL_TEXTURE_WRITE);
            assert(texture != NULL);
        }

        // Create decoder
        struct gpujpeg_decoder* decoder = gpujpeg_decoder_create(NULL);
        if ( decoder == NULL ) {
            fprintf(stderr, "Failed to create decoder!\n");
            return -1;
        }

        // Init decoder if image size is filled
        if ( gpujpeg_decoder_init(decoder, &param, &param_image) != 0 ) {
            fprintf(stderr, "Failed to preinitialize decoder!\n");
            return -1;
        }

        // Decode images
        for ( int index = 0; index < argc; index += 2 ) {
            // Get and check input and output image
            const char* input = argv[index];
            char* output = argv[index + 1];
            if ( encode == 1 ) {
                static char buffer_output[255];
                if ( param_image.color_space != GPUJPEG_RGB ) {
                    sprintf(buffer_output, "%s.decoded.yuv", output);
                }
                else {
                    if ( param.comp_count == 1 ) {
                        sprintf(buffer_output, "%s.decoded.r", output);
                    } else {
                        sprintf(buffer_output, "%s.decoded.rgb", output);
                    }
                }
                input = output;
                output = buffer_output;
            }
            enum gpujpeg_image_file_format input_format = gpujpeg_image_get_file_format(input);
            enum gpujpeg_image_file_format output_format = gpujpeg_image_get_file_format(output);
            if ( input_format != GPUJPEG_IMAGE_FILE_JPEG ) {
                fprintf(stderr, "[Warning] Decoder input file [%s] should be JPEG image (*.jpg)!\n", input);
            }
            if ( (output_format >= GPUJPEG_IMAGE_FILE_RAW) == 0 ) {
                fprintf(stderr, "[Warning] Decoder output file [%s] should be raw image (*.rgb, *.yuv, *.r, *.pnm)!\n", output);
                if ( output_format & GPUJPEG_IMAGE_FILE_JPEG ) {
                    ret = EXIT_FAILURE; continue;
                }
            }

            param = param_saved;
            param_image = param_image_saved;
            adjust_params(&param, &param_image, input, output, encode, &opts);
            if ( opts.native_file_format ) {
                param_image.color_space = GPUJPEG_NONE;
            }

            gpujpeg_decoder_set_output_format(decoder, param_image.color_space, param_image.pixel_format);

            // Decode image
            double duration = gpujpeg_get_time();

            printf("\nDecoding Image [%s] to ", input);
            print_gpujpeg_image_parameters(param_image, true, NULL);

            // Load image
            size_t image_size = 0;
            uint8_t* image = NULL;
            if ( gpujpeg_image_load_from_file(input, &image, &image_size) != 0 ) {
                fprintf(stderr, "Failed to load image [%s]!\n", argv[index]);
                ret = EXIT_FAILURE; continue;
            }

            duration = gpujpeg_get_time() - duration;
            printf("Load Image:          %10.2f ms\n", duration * 1000.0);

            // Prepare decoder output buffer
            struct gpujpeg_decoder_output decoder_output;
            if ( use_opengl ) {
                gpujpeg_decoder_output_set_texture(&decoder_output, texture);
            } else {
                gpujpeg_decoder_output_set_default(&decoder_output);
            }

            for ( int iteration = 0; iteration < iterate; iteration++ ) {
                if ( iterate > 1 ) {
                    printf("\nIteration #%d:\n", iteration + 1);
                }

                // Decode image
                if ( (rc = gpujpeg_decoder_decode(decoder, image, image_size, &decoder_output)) != 0 ) {
                    if (rc == GPUJPEG_ERR_RESTART_CHANGE && param_image.width != 0 && param_image.height != 0) {
                        fprintf(stderr, "Hint: Do not enter image dimensions to avoid preinitialization or correctly specify restart interval.\n");
                    }
                    fprintf(stderr, "Failed to decode image [%s]!\n", argv[index]);
                    ret = EXIT_FAILURE; continue;
                }
            }

            uint8_t* data = NULL;
            size_t data_size = 0;
            if ( use_opengl ) {
                data = malloc(decoder_output.data_size);
                gpujpeg_opengl_texture_get_data(texture->texture_id, data, &data_size);
                assert(data != NULL && data_size != 0);
            } else {
                data = decoder_output.data;
                data_size = decoder_output.data_size;
            }

            duration = gpujpeg_get_time();

            // Save image
            if ( data == NULL ||
                 gpujpeg_image_save_to_file(output, data, data_size, &decoder_output.param_image) != 0 ) {
                fprintf(stderr, "Failed to save image [%s]!\n", output);
                ret = EXIT_FAILURE;
            }

            duration = gpujpeg_get_time() - duration;
            printf("Save Image:          %10.2f ms\n", duration * 1000.0);
            printf("Decompressed Size:   %10.zu bytes [%s] ", decoder_output.data_size, output);
            print_gpujpeg_image_parameters(decoder_output.param_image, true, NULL);

            if ( use_opengl ) {
                free(data);
            }

            // Destroy image
            gpujpeg_image_destroy(image);
        }

        // Destroy OpenGL texture
        if ( use_opengl ) {
            int texture_id = texture->texture_id;
            gpujpeg_opengl_texture_unregister(texture);
            gpujpeg_opengl_texture_destroy(texture_id);
        }

        // Destroy decoder
        gpujpeg_decoder_destroy(decoder);
    }

    if ( use_opengl ) {
        gpujpeg_opengl_destroy(gl_context);
    }

#ifdef DEBUG
    gpujpeg_device_reset(); // to allow "cuda-memcheck --leak-check full"
#endif // defined DEBUG

    return ret;
}

/* vim: set expandtab sw=4: */
