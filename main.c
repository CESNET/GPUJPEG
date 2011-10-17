/* 
 * Copyright (c) 2011, Martin Srom
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
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

#include "jpeg_common.h"
#include "jpeg_encoder.h"
#include "jpeg_decoder.h"
#include "jpeg_util.h"

#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <string.h>
#include <strings.h>

void
print_help() 
{
    printf(
        "jpeg_compress [options] img.rgb\n"
        "   -h, --help\t\tprint help\n"
        "   -s, --size\t\timage size in pixels, e.g. 1920x1080\n"
        "   -q, --quality\t\tquality level 1-100 (default 75)\n"
        "   -r, --restart\t\tset restart interval (default 8)\n"
        "   -e, --encode\t\tencode images\n"
        "   -d, --decode\t\tdecode images\n"
    );
}

int
main(int argc, char *argv[])
{       
    struct option longopts[] = {
        {"help",    no_argument,       0, 'h'},
        {"size",    required_argument, 0, 's'},
        {"quality", required_argument, 0, 'q'},
        {"restart", required_argument, 0, 'r'},
        {"encode",  no_argument,       0, 'e'},
        {"decode",  no_argument,       0, 'd'},
    };

    // Parameters
    int width = 0;
    int height = 0;
    int comp_count = 3;
    int quality = 75;
    int restart_interval = 8;
    int encode = 0;
    int decode = 0;
    
    // Parse command line
    char ch = '\0';
    int optindex = 0;
    char* pos = 0;
    while ( (ch = getopt_long(argc, argv, "hs:q:r:ed", longopts, &optindex)) != -1 ) {
        switch (ch) {
        case 'h':
            print_help();
            return 0;
        case 's':
            width = atoi(optarg);
            pos = strstr(optarg, "x");
            if ( pos == NULL || width == 0 || (strlen(pos) >= strlen(optarg)) ) {
                print_help();
                return -1;
            }
            height = atoi(pos + 1);
            break;
        case 'q':
            quality = atoi(optarg);
            if ( quality <= 0 )
                quality = 1;
            if ( quality > 100 )
                quality = 100;
            break;
        case 'r':
            restart_interval = atoi(optarg);
            if ( restart_interval < 0 )
                restart_interval = 0;
            break;
        case 'e':
            encode = 1;
            break;
        case 'd':
            decode = 1;
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
    
    // Source image and target image must be presented
    if ( argc < 2 ) {
        printf("Please supply source and destination image filename!\n");
        print_help();
        return -1;
    }
    
    // Detect action if none is specified
    if ( encode == 0 && decode == 0 ) {
        enum jpeg_image_file_format input_format = jpeg_image_get_file_format(argv[0]);
        enum jpeg_image_file_format output_format = jpeg_image_get_file_format(argv[1]);
        if ( input_format == IMAGE_FILE_RGB && output_format == IMAGE_FILE_JPEG ) {
            encode = 1;
        } else if ( input_format == IMAGE_FILE_JPEG && output_format == IMAGE_FILE_RGB ) {
            decode = 1;
        } else {
            fprintf(stderr, "Action can't be recognized for specified images!\n");
            fprintf(stderr, "You must specify --encode or --decode option!\n");
            return -1;
        }
    }
    
    // Init device
    jpeg_init_device(0);
    
    if ( encode == 1 ) {    
        // Create encoder
        struct jpeg_encoder* encoder = jpeg_encoder_create(width, height, comp_count, quality, restart_interval);
        if ( encoder == NULL ) {
            fprintf(stderr, "Failed to create encoder!\n");
            return -1;
        }
        
        // Encode images
        for ( int index = 0; index < argc; index += 2 ) {
            // Get and check input and output image
            const char* input = argv[index];
            const char* output = argv[index + 1];
            enum jpeg_image_file_format input_format = jpeg_image_get_file_format(input);
            enum jpeg_image_file_format output_format = jpeg_image_get_file_format(output);
            if ( input_format != IMAGE_FILE_RGB  ) {
                fprintf(stderr, "Encoder input file [%s] should be RGB image (*.rgb)!\n", input);
                return -1;
            }
            if ( output_format != IMAGE_FILE_JPEG ) {
                fprintf(stderr, "Encoder output file [%s] should be JPEG image (*.jpg)!\n", output);
                return -1;
            }
            
            // Encode image
            TIMER_INIT();
            TIMER_START();
        
            // Load image
            int image_size = width * height * 3;
            uint8_t* image = NULL;
            if ( jpeg_image_load_from_file(input, &image, &image_size) != 0 ) {
                fprintf(stderr, "Failed to load image [%s]!\n", argv[index]);
                return -1;
            }
            
            TIMER_STOP_PRINT("Load Image:        ");
            TIMER_START();
                
            // Encode image
            uint8_t* image_compressed = NULL;
            int image_compressed_size = 0;
            if ( jpeg_encoder_encode(encoder, image, &image_compressed, &image_compressed_size) != 0 ) {
                fprintf(stderr, "Failed to encode image [%s]!\n", argv[index]);
                return -1;
            }
            
            TIMER_STOP_PRINT("Encode Image:      ");
            TIMER_START();
            
            // Save image
            if ( jpeg_image_save_to_file(output, image_compressed, image_compressed_size) != 0 ) {
                fprintf(stderr, "Failed to save image [%s]!\n", argv[index]);
                return -1;
            }
            
            TIMER_STOP_PRINT("Save Image:        ");
            
            printf("Compressed Size: %d bytes\n", image_compressed_size);
            
            // Destroy image
            jpeg_image_destroy(image);
        }
        
        // Destroy encoder
        jpeg_encoder_destroy(encoder);
    }
    
    if ( decode == 1 ) {    
        // Create decoder
        struct jpeg_decoder* decoder = jpeg_decoder_create(width, height, comp_count);
        if ( decoder == NULL ) {
            fprintf(stderr, "Failed to create decoder!\n");
            return -1;
        }
        
        // Decode images
        for ( int index = 0; index < argc; index += 2 ) {
            // Get and check input and output image
            const char* input = argv[index];
            const char* output = argv[index + 1];
            if ( encode == 1 ) {
                static char buffer_output[255];
                sprintf(buffer_output, "decoded_%s", input);
                input = output;
                output = buffer_output;
            }
            enum jpeg_image_file_format input_format = jpeg_image_get_file_format(input);
            enum jpeg_image_file_format output_format = jpeg_image_get_file_format(output);
            if ( input_format != IMAGE_FILE_JPEG ) {
                fprintf(stderr, "Encoder input file [%s] should be JPEG image (*.jpg)!\n", input);
                return -1;
            }
            if ( output_format != IMAGE_FILE_RGB  ) {
                fprintf(stderr, "Encoder output file [%s] should be RGB image (*.rgb)!\n", output);
                return -1;
            }
            
            // Decode image
            TIMER_INIT();
            TIMER_START();
        
            // Load image
            int image_size = 0;
            uint8_t* image = NULL;
            if ( jpeg_image_load_from_file(input, &image, &image_size) != 0 ) {
                fprintf(stderr, "Failed to load image [%s]!\n", argv[index]);
                return -1;
            }
            
            TIMER_STOP_PRINT("Load Image:     ");
            TIMER_START();
                
            // Encode image
            uint8_t* image_decompressed = NULL;
            int image_decompressed_size = 0;
            if ( jpeg_decoder_decode(decoder, image, image_size, &image_decompressed, &image_decompressed_size) != 0 ) {
                fprintf(stderr, "Failed to decode image [%s]!\n", argv[index]);
                return -1;
            }
            
            TIMER_STOP_PRINT("Decode Image:   ");
            TIMER_START();
            
            // Save image
            if ( jpeg_image_save_to_file(output, image_decompressed, image_decompressed_size) != 0 ) {
                fprintf(stderr, "Failed to save image [%s]!\n", argv[index]);
                return -1;
            }
            
            TIMER_STOP_PRINT("Save Image:     ");
            
            printf("Decompressed Size: %d bytes\n", image_decompressed_size);
            
            // Destroy image
            jpeg_image_destroy(image);
        }
        
        // Destroy decoder
        jpeg_decoder_destroy(decoder);
    }
    
	return 0;
}
