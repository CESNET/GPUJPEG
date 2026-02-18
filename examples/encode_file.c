/**
 * @file encode_file.c
 *
 * Example showing GPUJPEG encode either raw RGB or raw grayscale image.
 *
 * Changelog:
 * - 2023-06-28 - initial implementation
 * - 2024-03-26 - updated to new API (ch_count omitted)
 * - 2026-02-18 - simplified (get the image props from img if possible), support for PNG etc.; rename to encode_file
 */
#include <libgpujpeg/gpujpeg_common.h>
#include <libgpujpeg/gpujpeg_encoder.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define strcasecmp _stricmp
#endif

static void usage(const char *progname) {
        printf("Usage:\n");
        printf("\t%s <image_name> [<width> <height>]\n", progname);
        printf("\nwidth and height must be specified if not deducible from file itself (RAW files)\n");
}

// to simplify deletion of allocated items on all paths
struct encode_data {
        char *out_filename;
        struct gpujpeg_encoder* encoder;
        uint8_t *input_image;
};

/// @retval 0 success; 1 failure; 2 show help
static int encode(int argc, char *argv[], struct encode_data *d)
{
        const char *ifname = argv[1];
        // enum gpujpeg_image_file_format ftype = gpujpeg_image_get_file_format(argv[1]);

        // set default encode parametrs, after calling, parameters can be tuned (eg. quality)
        struct gpujpeg_parameters param;
        gpujpeg_set_default_parameters(&param);

        // here we set image parameters
        struct gpujpeg_image_parameters param_image;
        gpujpeg_image_set_default_parameters(&param_image);
        gpujpeg_image_get_properties(ifname, &param_image, true);
        if (param_image.width == 0 && param_image.height == 0) {
                if (argc != 4) {
                        return 2;
                }
                param_image.width = atoi(argv[2]);
                param_image.height= atoi(argv[3]);
        }
        // create encoder
        if ((d->encoder = gpujpeg_encoder_create(0)) == NULL) {
                return 1;
        }

        // load image and set it as the encoder input buffer
        size_t input_image_size = 0;
        if (gpujpeg_image_load_from_file(ifname, &d->input_image, &input_image_size) != 0) {
                return 1;
        }
        struct gpujpeg_encoder_input encoder_input;
        gpujpeg_encoder_input_set_image(&encoder_input, d->input_image);

        // compress the image
        uint8_t *image_compressed = NULL;
        size_t image_compressed_size = 0;
        if (gpujpeg_encoder_encode(d->encoder, &param, &param_image, &encoder_input, &image_compressed, &image_compressed_size) != 0) {
                return 1;
        }

        // write the encoded image
        d->out_filename = malloc(strlen(ifname) + 1);
        strcpy(d->out_filename, ifname);
        strcpy(strrchr(d->out_filename, '.') + 1, "jpg");
        if (gpujpeg_image_save_to_file(d->out_filename, image_compressed, image_compressed_size, &param_image) != 0) {
                return 1;
        }

        return 0;
}

static void cleanup(const struct encode_data *d ) {
        free(d->out_filename);
        gpujpeg_image_destroy(d->input_image);
        if (d->encoder != NULL) {
            gpujpeg_encoder_destroy(d->encoder);
        }
}

int main(int argc, char *argv[]) {
        if (argc != 2 && argc != 4) {
                usage(argv[0]);
                return 2;
        }
        struct encode_data d = { 0 };
        int rc = encode(argc, argv, &d);
        if (rc == 2) {
                usage(argv[0]);
        } else if (rc == 0) {
                printf("Written: %s\n", d.out_filename);
                puts("Success\n");
        } else {
                puts("FAILURE\n");
        }
        cleanup(&d);

        return rc;
}
