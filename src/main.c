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

#include <assert.h>
#include <libgpujpeg/gpujpeg.h>
#include <libgpujpeg/gpujpeg_common.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(_MSC_VER)
    #include "gpujpeg_getopt_mingw.h"
#else
    #include <getopt.h>
#endif

void
print_help()
{
    printf("gpujpeg [options] input.rgb output.jpg [input2.rgb output2.jpg ...]\n"
           "   -h, --help             print help\n"
           "   -v, --verbose          verbose output\n"
           "   -D, --device           set cuda device id (default 0)\n"
           "       --device-list      list cuda devices\n"
           "\n");
    printf("   -s, --size             set input image size in pixels, e.g. 1920x1080\n"
           "   -f, --pixel-format     set input/output image pixel format, one of the\n"
           "                          following:\n"
           "                          u8               444-u8-p0p1p2\n"
           "                          444-u8-p012      422-u8-p1020\n"
           "                          444-u8-p012a     422-u8-p0p1p2\n"
           "                          444-u8-p012z     420-u8-p0p1p2\n"
           "\n"
           "   -c, --colorspace       set input/output image colorspace, e.g. rgb,\n"
           "                          ycbcr, ycbcr-jpeg, ycbcr-bt601, ycbcr-bt709\n"
           "\n");
    printf("   -q, --quality          set JPEG encoder quality level 0-100 (default 75)\n"
           "   -r, --restart          set JPEG encoder restart interval (default 8)\n"
           "       --subsampled[=<s>] set JPEG encoder to use chroma subsampling (default 420)\n"
           "   -i  --interleaved      set JPEG encoder to use interleaved stream\n"
           "   -g  --segment-info     set JPEG encoder to use segment info in stream\n"
           "                          for fast decoding\n"
           "\n");
    printf("   -e, --encode           perform JPEG encoding\n"
           "   -d, --decode           perform JPEG decoding\n"
           "       --convert          convert input image to output image (change\n"
           "                          color space and/or sampling factor)\n"
           "       --component-range  show samples range for each component in image\n"
           "\n");
    printf("   -n  --iterate          perform encoding/decoding in specified number of\n"
           "                          iterations for each image\n"
           "   -o  --use-opengl       use an OpenGL texture as input/output\n"
           "   -I  --info             print JPEG file info\n"
           "   -R  --rgb              create RGB JPEG\n"
           "\n");
}

static int print_image_info(const char *filename) {
    if (!filename) {
        fprintf(stderr, "Missing filename!\n");
        return 1;
    }
    FILE *f = fopen(filename, "rb");
    if (!f) {
        perror("Cannot open");
        return 1;
    }
    fseek(f, 0L, SEEK_END);
    long int len = ftell(f);
    fseek(f, 0L, SEEK_SET);
    uint8_t *jpeg = malloc(len);
    size_t ret = fread(jpeg, len, 1, f);
    fclose(f);
    if (ret == 0) {
        fprintf(stderr, "Cannot read image contents.\n");
        return 1;
    }
    struct gpujpeg_image_parameters params;
    int segment_count = 0;
    memset(&params, 0, sizeof params);
    if (gpujpeg_decoder_get_image_info(jpeg, len, &params, &segment_count) == 0) {
        if (params.width) {
            printf("width: %d\n", params.width);
        }
        if (params.height) {
            printf("height: %d\n", params.height);
        }
        if (params.comp_count) {
            printf("component count: %d\n", params.comp_count);
        }
        if (params.color_space) {
            printf("color space: %s\n", gpujpeg_color_space_get_name(params.color_space));
        }
        if (params.pixel_format) {
            printf("internal representation: %s\n", gpujpeg_pixel_format_get_name(params.pixel_format));
        }
        if (segment_count) {
            printf("segment count: %d\n", segment_count);
        }
    }
    free(jpeg);

    return 0;
}

static void adjust_params(struct gpujpeg_parameters *param, struct gpujpeg_image_parameters *param_image,
        const char *in, const char *out, _Bool encode, _Bool chroma_subsampled, _Bool restart_interval_default) {
    // if possible, read properties from file
    struct gpujpeg_image_parameters file_param_image = { 0, 0, 0, GPUJPEG_NONE, GPUJPEG_PIXFMT_NONE };
    gpujpeg_image_get_properties(encode ? in : out, &file_param_image, encode);
    param_image->width = param_image->width == 0 ? file_param_image.width : param_image->width;
    param_image->height = param_image->height == 0 ? file_param_image.height : param_image->height;
    param_image->color_space = param_image->color_space == GPUJPEG_NONE ? file_param_image.color_space : param_image->color_space;
    if ( param_image->pixel_format == GPUJPEG_PIXFMT_NONE && file_param_image.pixel_format != GPUJPEG_PIXFMT_NONE ) {
        param_image->pixel_format = file_param_image.pixel_format;
        param_image->comp_count = gpujpeg_pixel_format_get_comp_count(param_image->pixel_format);
    }

    // Detect color spalce
    if ( param_image->color_space == GPUJPEG_NONE ) {
        if ( gpujpeg_image_get_file_format(encode ? in : out) == GPUJPEG_IMAGE_FILE_YUV ) {
            param_image->color_space = GPUJPEG_YCBCR_JPEG;
        } else {
            param_image->color_space = GPUJPEG_RGB;
        }
    }

    if ( param_image->pixel_format == GPUJPEG_PIXFMT_NONE ) {
        enum gpujpeg_image_file_format format = gpujpeg_image_get_file_format(encode ? in : out);
        switch (format) {
        case GPUJPEG_IMAGE_FILE_RGBA:
            param_image->pixel_format = GPUJPEG_444_U8_P012A;
            break;
        case GPUJPEG_IMAGE_FILE_I420:
            param_image->pixel_format = GPUJPEG_420_U8_P0P1P2;
            break;
        default:
            param_image->pixel_format = GPUJPEG_444_U8_P012;
        }
    }

    // Detect component count
    if ( gpujpeg_image_get_file_format(encode ? in : out) == GPUJPEG_IMAGE_FILE_GRAY && param_image->comp_count != 1 ) {
        param_image->pixel_format = GPUJPEG_U8;
        param_image->comp_count = 1;
    }

    // Adjust restart interval
    if ( restart_interval_default ) {
        // when chroma subsampling and interleaving is enabled, the restart interval should be smaller
        if ( chroma_subsampled == 1 && param->interleaved == 1 ) {
            param->restart_interval = 2;
        }
        else {
            // Adjust according to Mpix count
            double coefficient = ((double)param_image->width * param_image->height * param_image->comp_count) / (1000000.0 * 3.0);
            if ( coefficient < 1.0 ) {
                param->restart_interval = 4;
            } else if ( coefficient < 3.0 ) {
                param->restart_interval = 8;
            } else if ( coefficient < 9.0 ) {
                param->restart_interval = 10;
            } else {
                param->restart_interval = 12;
            }
        }
        if ( param->verbose ) {
            printf("\nAuto-adjusting restart interval to %d for better performance.\n", param->restart_interval);
        }
    }
}

#ifndef GIT_REV
#define GIT_REV "unknown"
#endif

int
main(int argc, char *argv[])
{
    printf("GPUJPEG rev %s\n", strlen(GIT_REV) > 0 ? GIT_REV : "unknown");

    // Default coder parameters
    struct gpujpeg_parameters param;
    gpujpeg_set_default_parameters(&param);

    // Default image parameters
    struct gpujpeg_image_parameters param_image;
    gpujpeg_image_set_default_parameters(&param_image);

    // Original image parameters in conversion
    struct gpujpeg_image_parameters param_image_original;
    gpujpeg_image_set_default_parameters(&param_image_original);

    // Other parameters
    int device_id = 0;
    int encode = 0;
    int decode = 0;
    int convert = 0;
    int component_range = 0;
    int iterate = 1;
    int use_opengl = 0;

    // Flags
    int restart_interval_default = 1;
    int chroma_subsampled = 0;

    int rc;

    param_image.color_space = GPUJPEG_NONE;
    param_image.pixel_format = GPUJPEG_PIXFMT_NONE;

    // Parse command line
    #define OPTION_DEVICE_INFO     1
    #define OPTION_SUBSAMPLED      2
    #define OPTION_CONVERT         3
    #define OPTION_COMPONENT_RANGE 4
    struct option longopts[] = {
        {"help",                    no_argument,       0, 'h'},
        {"verbose",                 no_argument,       0, 'v'},
        {"device",                  required_argument, 0, 'D'},
        {"device-list",             no_argument,       0,  OPTION_DEVICE_INFO },
        {"size",                    required_argument, 0, 's'},
        {"pixel-format",         required_argument, 0, 'f'},
        {"colorspace",              required_argument, 0, 'c'},
        {"quality",                 required_argument, 0, 'q'},
        {"restart",                 required_argument, 0, 'r'},
        {"segment-info",            optional_argument, 0, 'g' },
        {"subsampled",              optional_argument, 0,  OPTION_SUBSAMPLED },
        {"interleaved",             optional_argument, 0, 'i'},
        {"encode",                  no_argument,       0, 'e'},
        {"decode",                  no_argument,       0, 'd'},
        {"convert",                 no_argument,       0,  OPTION_CONVERT },
        {"component-range",         no_argument,       0,  OPTION_COMPONENT_RANGE },
        {"iterate",                 required_argument, 0,  'n' },
        {"use-opengl",              no_argument,       0,  'o' },
        {"info",                    required_argument, 0,  'I' },
        {"rgb",                     no_argument,       0,  'R' },
        0
    };
    int ch = '\0';
    int optindex = 0;
    char* pos = 0;
    while ( (ch = getopt_long(argc, argv, "hvD:s:C:f:c:q:r:g::i::edn:oI:R", longopts, &optindex)) != -1 ) {
        switch (ch) {
        case 'h':
            print_help();
            return 0;
        case 'v':
            param.verbose = 1;
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
            if ( strcmp(optarg, "rgb") == 0 )
                param_image.color_space = GPUJPEG_RGB;
            else if ( strcmp(optarg, "yuv") == 0 )
                param_image.color_space = GPUJPEG_YUV;
            else if ( strcmp(optarg, "ycbcr") == 0 )
                param_image.color_space = GPUJPEG_YCBCR;
            else if ( strcmp(optarg, "ycbcr-jpeg") == 0 )
                param_image.color_space = GPUJPEG_YCBCR_BT601_256LVLS;
            else if ( strcmp(optarg, "ycbcr-bt601") == 0 )
                param_image.color_space = GPUJPEG_YCBCR_BT601;
            else if ( strcmp(optarg, "ycbcr-bt709") == 0 )
                param_image.color_space = GPUJPEG_YCBCR_BT709;
            else
                fprintf(stderr, "Colorspace '%s' is not available!\n", optarg);
            break;
        case 'f':
            param_image.pixel_format = gpujpeg_pixel_format_by_name(optarg);
            if (param_image.pixel_format == GPUJPEG_PIXFMT_NONE) {
                fprintf(stderr, "Unknown pixel format '%s'!\n", optarg);
                return 1;
            }
            param_image.comp_count = gpujpeg_pixel_format_get_comp_count(param_image.pixel_format);
            chroma_subsampled = gpujpeg_pixel_format_is_subsampled(param_image.pixel_format);
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
            if ( param.restart_interval < 0 )
                param.restart_interval = 0;
            restart_interval_default = 0;
            break;
        case 'g':
            if ( optarg == NULL || strcmp(optarg, "true") == 0 || atoi(optarg) )
                param.segment_info = 1;
            else
                param.segment_info = 0;
            break;
        case OPTION_SUBSAMPLED:
            chroma_subsampled = 1;
            if ( optarg == NULL || strcmp(optarg, "420") == 0 )
                gpujpeg_parameters_chroma_subsampling_420(&param);
            else if ( strcmp(optarg, "422") == 0 )
                gpujpeg_parameters_chroma_subsampling_422(&param);
            else if ( strcmp(optarg, "444") == 0 )
                chroma_subsampled = 0;
            else {
                fprintf(stderr, "Unknown subsampling '%s'!\n", optarg);
                return 1;
            }
            break;
        case OPTION_DEVICE_INFO:
            gpujpeg_print_devices_info();
            return 0;
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
        case OPTION_CONVERT:
            convert = 1;
            memcpy(&param_image_original, &param_image, sizeof(struct gpujpeg_image_parameters));
            break;
        case OPTION_COMPONENT_RANGE:
            component_range = 1;
            break;
        case 'n':
            iterate = atoi(optarg);
            break;
        case 'o':
            use_opengl = 1;
            break;
        case 'I':
            return print_image_info(optarg);
        case 'R':
            param.color_space_internal = GPUJPEG_RGB;
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

    // Show info about image samples range
    if ( component_range == 1 ) {
        // For each image
        for ( int index = 0; index < argc; index++ ) {
            gpujpeg_image_range_info(argv[index], param_image.width, param_image.height, param_image.pixel_format);
        }
        return 0;
    }

    // Source image and target image must be presented
    if ( argc < 2 ) {
        fprintf(stderr, "Please supply source and destination image filename!\n");
        print_help();
        return -1;
    }

    // Init device
    int flags = GPUJPEG_VERBOSE;
    if ( use_opengl ) {
        flags |= GPUJPEG_OPENGL_INTEROPERABILITY;
        gpujpeg_opengl_init();
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
        if ( (input_format & GPUJPEG_IMAGE_FILE_RAW) && output_format == GPUJPEG_IMAGE_FILE_JPEG ) {
            encode = 1;
        } else if ( input_format == GPUJPEG_IMAGE_FILE_JPEG && (output_format & GPUJPEG_IMAGE_FILE_RAW) ) {
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
            const char* output = argv[index + 1];
            enum gpujpeg_image_file_format input_format = gpujpeg_image_get_file_format(input);
            enum gpujpeg_image_file_format output_format = gpujpeg_image_get_file_format(output);
            if ( (input_format & GPUJPEG_IMAGE_FILE_RAW) == 0 ) {
                fprintf(stderr, "[Warning] Encoder input file [%s] should be raw image (*.rgb, *.yuv, *.r, *.pnm)!\n", input);
                if ( input_format & GPUJPEG_IMAGE_FILE_JPEG ) {
                    return -1;
                }
            }
            if ( output_format != GPUJPEG_IMAGE_FILE_JPEG ) {
                fprintf(stderr, "Encoder output file [%s] should be JPEG image (*.jpg)!\n", output);
                return -1;
            }

            param = param_saved;
            param_image = param_image_saved;
            adjust_params(&param, &param_image, input, output, encode, chroma_subsampled, restart_interval_default);
            if (param_image.width <= 0 || param_image.height <= 0) {
                fprintf(stderr, "Image dimensions must be set to nonzero values!\n");
            }

            // Encode image
            double duration = gpujpeg_get_time();
            printf("\nEncoding Image [%s]\n", input);

            // Load image
            int image_size = gpujpeg_image_calculate_size(&param_image);
            uint8_t* image = NULL;
            if ( gpujpeg_image_load_from_file(input, &image, &image_size) != 0 ) {
                fprintf(stderr, "Failed to load image [%s]!\n", argv[index]);
                return -1;
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
            int image_compressed_size = 0;
            for ( int iteration = 0; iteration < iterate; iteration++ ) {
                if ( iterate > 1 ) {
                    printf("\nIteration #%d:\n", iteration + 1);
                }

                double duration = gpujpeg_get_time();

                rc = gpujpeg_encoder_encode(encoder, &param, &param_image, &encoder_input, &image_compressed, &image_compressed_size);
                if ( rc != GPUJPEG_NOERR ) {
                    if ( rc == GPUJPEG_ERR_WRONG_SUBSAMPLING ) {
                        fprintf(stderr, "Consider using '--subsampling' option!\n");
                    }
                    fprintf(stderr, "Failed to encode image [%s]!\n", argv[index]);
                    return -1;
                }

                duration = gpujpeg_get_time() - duration;
                struct gpujpeg_duration_stats stats;
                rc = gpujpeg_encoder_get_stats(encoder, &stats);

                if ( rc == 0 && param.verbose ) {
                    printf(" -Copy To Device:    %10.2f ms\n", stats.duration_memory_to);
                    if ( stats.duration_memory_map != 0.0 && stats.duration_memory_unmap != 0.0 ) {
                        printf(" -OpenGL Memory Map: %10.2f ms\n", stats.duration_memory_map);
                        printf(" -OpenGL Memory Unmap:%9.2f ms\n", stats.duration_memory_unmap);
                    }
                    printf(" -Preprocessing:     %10.2f ms\n", stats.duration_preprocessor);
                    printf(" -DCT & Quantization:%10.2f ms\n", stats.duration_dct_quantization);
                    printf(" -Huffman Encoder:   %10.2f ms\n", stats.duration_huffman_coder);
                    printf(" -Copy From Device:  %10.2f ms\n", stats.duration_memory_from);
                    printf(" -Stream Formatter:  %10.2f ms\n", stats.duration_stream);
                }
                printf("Encode Image GPU:    %10.2f ms (only in-GPU processing)\n", stats.duration_in_gpu);
                printf("Encode Image Bare:   %10.2f ms (without copy to/from GPU memory)\n", duration * 1000.0 - stats.duration_memory_to - stats.duration_memory_from);
                printf("Encode Image:        %10.2f ms\n", duration * 1000.0);
            }
            if ( iterate > 1 ) {
                printf("\n");
            }

            duration = gpujpeg_get_time();

            // Save image
            if ( gpujpeg_image_save_to_file(output, image_compressed, image_compressed_size, &param_image) != 0 ) {
                fprintf(stderr, "Failed to save image [%s]!\n", argv[index]);
                return -1;
            }

            duration = gpujpeg_get_time() - duration;
            printf("Save Image:          %10.2f ms\n", duration * 1000.0);
            printf("Compressed Size:     %10.d bytes [%s]\n", image_compressed_size, output);

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
        if ( param_image.width != 0 && param_image.height != 0 && param_image.pixel_format != GPUJPEG_PIXFMT_NONE && !restart_interval_default ) {
            if ( gpujpeg_decoder_init(decoder, &param, &param_image) != 0 ) {
                fprintf(stderr, "Failed to preinitialize decoder!\n");
                return -1;
            }
        }

        // Decode images
        for ( int index = 0; index < argc; index += 2 ) {
            // Get and check input and output image
            const char* input = argv[index];
            const char* output = argv[index + 1];
            if ( encode == 1 ) {
                static char buffer_output[255];
                if ( param_image.color_space != GPUJPEG_RGB ) {
                    sprintf(buffer_output, "%s.decoded.yuv", output);
                }
                else {
                    if ( param_image.comp_count == 1 ) {
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
                fprintf(stderr, "Decoder input file [%s] should be JPEG image (*.jpg)!\n", input);
                return -1;
            }
            if ( (output_format & GPUJPEG_IMAGE_FILE_RAW) == 0 ) {
                fprintf(stderr, "[Warning] Decoder output file [%s] should be raw image (*.rgb, *.yuv, *.r, *.pnm)!\n", output);
                if ( output_format & GPUJPEG_IMAGE_FILE_JPEG ) {
                    return -1;
                }
            }

            param = param_saved;
            param_image = param_image_saved;
            adjust_params(&param, &param_image, input, output, encode, chroma_subsampled, restart_interval_default);

            if ( param_image_saved.color_space == GPUJPEG_NONE ) {
                gpujpeg_decoder_set_output_format(decoder, param_image.color_space, param_image.pixel_format);
            }

            // Decode image
            double duration = gpujpeg_get_time();

            printf("\nDecoding Image [%s]\n", input);

            // Load image
            int image_size = 0;
            uint8_t* image = NULL;
            if ( gpujpeg_image_load_from_file(input, &image, &image_size) != 0 ) {
                fprintf(stderr, "Failed to load image [%s]!\n", argv[index]);
                return -1;
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

                double duration = gpujpeg_get_time();

                // Decode image
                if ( (rc = gpujpeg_decoder_decode(decoder, image, image_size, &decoder_output)) != 0 ) {
                    if (rc == GPUJPEG_ERR_RESTART_CHANGE && param_image.width != 0 && param_image.height != 0) {
                        fprintf(stderr, "Hint: Do not enter image dimensions to avoid preinitialization or correctly specify restart interval.\n");
                    }
                    fprintf(stderr, "Failed to decode image [%s]!\n", argv[index]);
                    return -1;
                }

                duration = gpujpeg_get_time() - duration;
                struct gpujpeg_duration_stats stats;
                rc = gpujpeg_decoder_get_stats(decoder, &stats);

                if ( rc == 0 && param.verbose ) {
                    printf(" -Stream Reader:     %10.2f ms\n", stats.duration_stream);
                    printf(" -Copy To Device:    %10.2f ms\n", stats.duration_memory_to);
                    printf(" -Huffman Decoder:   %10.2f ms\n", stats.duration_huffman_coder);
                    printf(" -DCT & Quantization:%10.2f ms\n", stats.duration_dct_quantization);
                    printf(" -Postprocessing:    %10.2f ms\n", stats.duration_preprocessor);
                    printf(" -Copy From Device:  %10.2f ms\n", stats.duration_memory_from);
                    if ( stats.duration_memory_map != 0.0 && stats.duration_memory_unmap != 0.0 ) {
                        printf(" -OpenGL Memory Map: %10.2f ms\n", stats.duration_memory_map);
                        printf(" -OpenGL Memory Unmap:%9.2f ms\n", stats.duration_memory_unmap);
                    }
                }
                printf("Decode Image GPU:    %10.2f ms (only in-GPU processing)\n", stats.duration_in_gpu);
                printf("Decode Image Bare:   %10.2f ms (without copy to/from GPU memory)\n", duration * 1000.0 - stats.duration_memory_to - stats.duration_memory_from);
                printf("Decode Image:        %10.2f ms\n", duration * 1000.0);
            }
            if ( iterate > 1 ) {
                printf("\n");
            }

            uint8_t* data = NULL;
            int data_size = 0;
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
            struct gpujpeg_image_parameters decoded_param_image;
            gpujpeg_decoder_get_image_info(image, image_size, &decoded_param_image, NULL);
            decoded_param_image.color_space = param_image.color_space;
            decoded_param_image.pixel_format = param_image.pixel_format;
            if ( gpujpeg_image_save_to_file(output, data, data_size, &decoded_param_image) != 0 ) {
                fprintf(stderr, "Failed to save image [%s]!\n", output);
                return -1;
            }

            duration = gpujpeg_get_time() - duration;
            printf("Save Image:          %10.2f ms\n", duration * 1000.0);
            printf("Decompressed Size:   %10.d bytes [%s]\n", decoder_output.data_size, output);

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

    return 0;
}

/* vim: set expandtab sw=4: */
