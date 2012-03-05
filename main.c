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

#include "libgpujpeg/gpujpeg.h"
#include "libgpujpeg/gpujpeg_util.h"
#include <getopt.h>

void
print_help() 
{
    printf(
        "gpujpeg [options] input.rgb output.jpg [input2.rgb output2.jpg ...]\n"
        "   -h, --help             print help\n"
        "   -v, --verbose          verbose output\n"
        "   -D, --device           set cuda device id (default 0)\n"
        "       --device-list      list cuda devices\n"
        "\n"
        "   -s, --size             set input image size in pixels, e.g. 1920x1080\n"
        "   -f, --sampling-factor  set input/output image sampling factor, e.g. 4:2:2\n"
        "   -c, --colorspace       set input/output image colorspace, e.g. rgb, yuv,\n"
        "                          ycbcr-jpeg, ycbcr-bt601, ycbcr-bt709\n"
        "\n"
        "   -q, --quality          set JPEG encoder quality level 0-100 (default 75)\n"
        "   -r, --restart          set JPEG encoder restart interval (default 8)\n"
        "       --subsampled       set JPEG encoder to use chroma subsampling\n"
        "   -i  --interleaved      set JPEG encoder to use interleaved stream\n"
        "       --segment-info     set JPEG encoder to use segment info in stream\n"
        "                          for fast decoding\n"
        "\n"
        "   -e, --encode           perform JPEG encoding\n"
        "   -d, --decode           perform JPEG decoding\n"
        "       --convert          scale sampling factor from input to output image\n"
        "       --component-range  show samples range for each component in image\n"
    );
}

int
main(int argc, char *argv[])
{       
    struct option longopts[] = {
        {"help",                    no_argument,       0, 'h'},
        {"verbose",                 no_argument,       0, 'v'},
        {"device",                  required_argument, 0, 'D'},
        {"device-list",             no_argument,       0,  3 },
        {"size",                    required_argument, 0, 's'},
        {"sampling-factor",         required_argument, 0, 'f'},
        {"colorspace",              required_argument, 0, 'c'},
        {"quality",                 required_argument, 0, 'q'},
        {"restart",                 required_argument, 0, 'r'},
        {"segment-info",            optional_argument, 0,  1 },
        {"subsampled",              optional_argument, 0,  2 },
        {"interleaved",             optional_argument, 0, 'i'},
        {"encode",                  no_argument,       0, 'e'},
        {"decode",                  no_argument,       0, 'd'},
        {"convert",                 no_argument,       0,  4 },
        {"component-range",         no_argument,       0,  5 },
        0
    };

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
    
    // Flags
    int restart_interval_default = 1;
    int chroma_subsampled = 0;
    
    // Parse command line
    char ch = '\0';
    int optindex = 0;
    char* pos = 0;
    while ( (ch = getopt_long(argc, argv, "hvs:q:r:ed", longopts, &optindex)) != -1 ) {
        switch (ch) {
        case 'h':
            print_help();
            return 0;
        case 'v':
            param.verbose = 1;
            break;
        case 's':
            param_image.width = atoi(optarg);
            pos = strstr(optarg, "x");
            if ( pos == NULL || param_image.width == 0 || (strlen(pos) >= strlen(optarg)) ) {
                print_help();
                return -1;
            }
            param_image.height = atoi(pos + 1);
            break;
        case 'c':
            if ( strcmp(optarg, "rgb") == 0 )
                param_image.color_space = GPUJPEG_RGB;
            else if ( strcmp(optarg, "yuv") == 0 )
                param_image.color_space = GPUJPEG_YUV;
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
            if ( strcmp(optarg, "4:4:4") == 0 )
                param_image.sampling_factor = GPUJPEG_4_4_4;
            else if ( strcmp(optarg, "4:2:2") == 0 )
                param_image.sampling_factor = GPUJPEG_4_2_2;
            else
                fprintf(stderr, "Sampling factor '%s' is not available!\n", optarg);
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
        case 1:
            if ( optarg == NULL || strcmp(optarg, "true") == 0 || atoi(optarg) )
                param.segment_info = 1;
            else
                param.segment_info = 0;
            break;
        case 2:
            gpujpeg_parameters_chroma_subsampling(&param);
            chroma_subsampled = 1;
            break;
        case 3:
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
        case 4:
            convert = 1;
            memcpy(&param_image_original, &param_image, sizeof(struct gpujpeg_image_parameters));
            break;
        case 5:
            component_range = 1;
            break;
        case '?':
            return -1;
        default:
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
            gpujpeg_image_range_info(argv[index], param_image.width, param_image.height, param_image.sampling_factor);
        }
        return 0;
    }

    // Source image and target image must be presented
    if ( argc < 2 ) {
        fprintf(stderr, "Please supply source and destination image filename!\n");
        print_help();
        return -1;
    }
    
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
    
    // Init device
    if ( gpujpeg_init_device(device_id, GPUJPEG_VERBOSE) != 0 )
        return -1;
    
    // Adjust restart interval (when chroma subsampling and interleaving is enabled and restart interval is not changed)
    if ( restart_interval_default == 1 && chroma_subsampled == 1 && param.interleaved == 1 ) {
        if ( param.verbose )
            printf("Auto-adjusting restart interval to 2 for better performance!\n");
        param.restart_interval = 2;
    }
    
    // Detect color spalce
    if ( gpujpeg_image_get_file_format(argv[0]) == GPUJPEG_IMAGE_FILE_YUV && param_image.color_space == GPUJPEG_RGB )
        param_image.color_space = GPUJPEG_YUV;
    
    if ( encode == 1 ) {    
        // Create encoder
        struct gpujpeg_encoder* encoder = gpujpeg_encoder_create(&param, &param_image);
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
                fprintf(stderr, "Encoder input file [%s] should be RGB image (*.rgb)!\n", input);
                return -1;
            }
            if ( output_format != GPUJPEG_IMAGE_FILE_JPEG ) {
                fprintf(stderr, "Encoder output file [%s] should be JPEG image (*.jpg)!\n", output);
                return -1;
            }                
            
            // Encode image
            GPUJPEG_TIMER_INIT();
            GPUJPEG_TIMER_START();
            
            printf("\nEncoding Image [%s]\n", input);
        
            // Load image
            int image_size = gpujpeg_image_calculate_size(&param_image);
            uint8_t* image = NULL;
            if ( gpujpeg_image_load_from_file(input, &image, &image_size) != 0 ) {
                fprintf(stderr, "Failed to load image [%s]!\n", argv[index]);
                return -1;
            }
            
            GPUJPEG_TIMER_STOP();
            printf("Load Image:          %10.2f ms\n", GPUJPEG_TIMER_DURATION());
            GPUJPEG_TIMER_START();
                
            // Encode image
            uint8_t* image_compressed = NULL;
            int image_compressed_size = 0;
            if ( gpujpeg_encoder_encode(encoder, image, &image_compressed, &image_compressed_size) != 0 ) {
                fprintf(stderr, "Failed to encode image [%s]!\n", argv[index]);
                return -1;
            }
            
            GPUJPEG_TIMER_STOP();
            float duration = GPUJPEG_TIMER_DURATION();
            if ( param.verbose ) {
                printf(" -Copy To Device:    %10.2f ms\n", encoder->coder.duration_memory_to);
                printf(" -Preprocessing:     %10.2f ms\n", encoder->coder.duration_preprocessor);
                printf(" -DCT & Quantization:%10.2f ms\n", encoder->coder.duration_dct_quantization);
                printf(" -Huffman Encoder:   %10.2f ms\n", encoder->coder.duration_huffman_coder);
                printf(" -Copy From Device:  %10.2f ms\n", encoder->coder.duration_memory_from);
                printf(" -Stream Formatter:  %10.2f ms\n", encoder->coder.duration_stream);
            }
            printf("Encode Image GPU:    %10.2f ms (only in-GPU processing)\n", encoder->coder.duration_in_gpu);
            printf("Encode Image Bare:   %10.2f ms (without copy to/from GPU memory)\n", duration - encoder->coder.duration_memory_to - encoder->coder.duration_memory_from);
            printf("Encode Image:        %10.2f ms\n", duration);
            GPUJPEG_TIMER_START();
            
            // Save image
            if ( gpujpeg_image_save_to_file(output, image_compressed, image_compressed_size) != 0 ) {
                fprintf(stderr, "Failed to save image [%s]!\n", argv[index]);
                return -1;
            }
            
            GPUJPEG_TIMER_STOP();
            printf("Save Image:          %10.2f ms\n", GPUJPEG_TIMER_DURATION());
            printf("Compressed Size:     %10.d bytes [%s]\n", image_compressed_size, output);
            
            // Destroy image
            gpujpeg_image_destroy(image);
        }
        
        // Destroy encoder
        gpujpeg_encoder_destroy(encoder);
    }
    
    if ( decode == 1 ) {    
        // Create decoder
        struct gpujpeg_decoder* decoder = gpujpeg_decoder_create();
        if ( decoder == NULL ) {
            fprintf(stderr, "Failed to create decoder!\n");
            return -1;
        }
        
        // Init decoder if image size is filled
        if ( param_image.width != 0 && param_image.height != 0 ) {
            if ( gpujpeg_decoder_init(decoder, &param, &param_image) != 0 ) {
                fprintf(stderr, "Failed to preinitialize decoder!\n");
                return -1;
            }
        } else {
            decoder->coder.param_image.color_space = param_image.color_space;
            decoder->coder.param_image.sampling_factor = param_image.sampling_factor;
        }

        // Decode images
        for ( int index = 0; index < argc; index += 2 ) {
            // Get and check input and output image
            const char* input = argv[index];
            const char* output = argv[index + 1];
            if ( encode == 1 ) {
                static char buffer_output[255];
                if ( param_image.color_space != GPUJPEG_RGB )
                    sprintf(buffer_output, "%s.decoded.yuv", output);
                else
                    sprintf(buffer_output, "%s.decoded.rgb", output);
                input = output;
                output = buffer_output;
            }
            enum gpujpeg_image_file_format input_format = gpujpeg_image_get_file_format(input);
            enum gpujpeg_image_file_format output_format = gpujpeg_image_get_file_format(output);
            if ( input_format != GPUJPEG_IMAGE_FILE_JPEG ) {
                fprintf(stderr, "Encoder input file [%s] should be JPEG image (*.jpg)!\n", input);
                return -1;
            }
            if ( (output_format & GPUJPEG_IMAGE_FILE_RAW) == 0 ) {
                fprintf(stderr, "Encoder output file [%s] should be RGB image (*.rgb)!\n", output);
                return -1;
            }
            
            // Decode image
            GPUJPEG_TIMER_INIT();
            GPUJPEG_TIMER_START();
            
            printf("\nDecoding Image [%s]\n", input);
        
            // Load image
            int image_size = 0;
            uint8_t* image = NULL;
            if ( gpujpeg_image_load_from_file(input, &image, &image_size) != 0 ) {
                fprintf(stderr, "Failed to load image [%s]!\n", argv[index]);
                return -1;
            }
            
            GPUJPEG_TIMER_STOP();
            printf("Load Image:          %10.2f ms\n", GPUJPEG_TIMER_DURATION());
            GPUJPEG_TIMER_START();
                
            // Prepare decoder output buffer
            struct gpujpeg_decoder_output decoder_output;
            gpujpeg_decoder_output_set_default(&decoder_output);
            
            // Decode image
            if ( gpujpeg_decoder_decode(decoder, image, image_size, &decoder_output) != 0 ) {
                fprintf(stderr, "Failed to decode image [%s]!\n", argv[index]);
                return -1;
            }
            
            GPUJPEG_TIMER_STOP();
            float duration = GPUJPEG_TIMER_DURATION();
            if ( param.verbose ) {
                printf(" -Stream Reader:     %10.2f ms\n", decoder->coder.duration_stream);
                printf(" -Copy To Device:    %10.2f ms\n", decoder->coder.duration_memory_to);
                printf(" -Huffman Decoder:   %10.2f ms\n", decoder->coder.duration_huffman_coder);
                printf(" -DCT & Quantization:%10.2f ms\n", decoder->coder.duration_dct_quantization);
                printf(" -Postprocessing:    %10.2f ms\n", decoder->coder.duration_preprocessor);
                printf(" -Copy From Device:  %10.2f ms\n", decoder->coder.duration_memory_from);
             }
             printf("Decode Image GPU:    %10.2f ms (only in-GPU processing)\n", decoder->coder.duration_in_gpu);
             printf("Decode Image Bare:   %10.2f ms (without copy to/from GPU memory)\n", duration - decoder->coder.duration_memory_to - decoder->coder.duration_memory_from);
             printf("Decode Image:        %10.2f ms\n", duration);
             GPUJPEG_TIMER_START();
            
            // Save image
            if ( gpujpeg_image_save_to_file(output, decoder_output.data, decoder_output.data_size) != 0 ) {
                fprintf(stderr, "Failed to save image [%s]!\n", argv[index]);
                return -1;
            }
            
            GPUJPEG_TIMER_STOP();
            printf("Save Image:          %10.2f ms\n", GPUJPEG_TIMER_DURATION());
            printf("Decompressed Size:   %10.d bytes [%s]\n", decoder_output.data_size, output);
            
            // Destroy image
            gpujpeg_image_destroy(image);
        }
        
        // Destroy decoder
        gpujpeg_decoder_destroy(decoder);
    }

    return 0;
}
