/**
 * @file decode_to_file.c
 *
 * This demonstrates decoding input file to output file.
 *
 * Changelog:
 * - 2020-08-11 - initial implelemtation
 * - 2026-02-18 - rename, support for abitrary output format
 */
#include <libgpujpeg/gpujpeg_common.h>
#include <libgpujpeg/gpujpeg_decoder.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define strcasecmp _stricmp
#endif

static void usage(const char *progname) {
        printf("Usage:\n");
        printf("\t%s input.jpg [output.ext]\n", progname);
        printf(
            "\noptional output file should can use one of recognized extensions (eg. png); defaults to <input>.rgb\n");
}

static bool check_params(int argc, char *argv[]) {
        if ( (argc < 2 || argc > 3) || strrchr(argv[1], '.') == NULL ) {
                return false;
        }
        const char *ext = strrchr(argv[1], '.') + 1;
        return strcasecmp(ext, "jpg") == 0;
}

// to simplify deletion of allocated items on all paths
struct decode_data {
        char *out_filename;
        struct gpujpeg_decoder *decoder;
        uint8_t *input_image;
};

static int
decode(const char* input_filename, const char* output_filename, struct decode_data* d)
{
        // create decoder
        if ((d->decoder = gpujpeg_decoder_create(0)) == NULL) {
                return 1;
        }
        gpujpeg_decoder_set_output_format(d->decoder, GPUJPEG_RGB, GPUJPEG_444_U8_P012);

        // load image
        size_t input_image_size = 0;
        if (gpujpeg_image_load_from_file(input_filename, &d->input_image, &input_image_size) != 0) {
                return 1;
        }

        // set decoder default output destination
        struct gpujpeg_decoder_output decoder_output;
        gpujpeg_decoder_output_set_default(&decoder_output);

        // decompress the image
        if (gpujpeg_decoder_decode(d->decoder, d->input_image, input_image_size, &decoder_output) != 0) {
                return 1;
        }

        if (output_filename != nullptr) {
                d->out_filename = strdup(output_filename);
        } else { // create output file name with .rgb extension
            d->out_filename = malloc(strlen(input_filename) + 1);
            strcpy(d->out_filename, input_filename);
            strcpy(strrchr(d->out_filename, '.') + 1, "rgb");
        }

        // write the decoded image
        if ( gpujpeg_image_save_to_file(d->out_filename, decoder_output.data, decoder_output.data_size,
                                        &decoder_output.param_image) != 0 ) {
               return 1;
        }
        printf("File %s successfully written.\n", d->out_filename);

        return 0;
}

int main(int argc, char *argv[]) {
        if (!check_params(argc, argv)) {
                usage(argv[0]);
                return 1;
        }

        struct decode_data d = { 0 };
        int rc = decode(argv[1], argv[2], &d);

        free(d.out_filename);
        gpujpeg_image_destroy(d.input_image);
        if (d.decoder != NULL) {
            gpujpeg_decoder_destroy(d.decoder);
        }

        puts(rc == 0 ? "Success\n" : "FAILURE\n");

        return rc;
}

/* vim: set expandtab sw=4: */
