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
        printf("\t%s file.jpg\n", progname);
}

static bool check_params(int argc, char *argv[]) {
        if (argc != 2 || strrchr(argv[1], '.') == NULL) {
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

static int decode(const char *input_filename, struct decode_data *d)
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
        uint8_t *image_decompressed = NULL;
        size_t image_decompressed_size = 0;
        if (gpujpeg_decoder_decode(d->decoder, d->input_image, input_image_size, &decoder_output) != 0) {
                return 1;
        }

        // create output file name
        d->out_filename = malloc(strlen(input_filename) + 1);
        strcpy(d->out_filename, input_filename);
        strcpy(strrchr(d->out_filename, '.') + 1, "rgb");

        // write the decoded image
        if (gpujpeg_image_save_to_file(d->out_filename, decoder_output.data, decoder_output.data_size, NULL) != 0) {
                return 1;
        }

        return 0;
}

int main(int argc, char *argv[]) {
        if (!check_params(argc, argv)) {
                usage(argv[0]);
                return 1;
        }

        struct decode_data d = { 0 };
        int rc = decode(argv[1], &d);

        free(d.out_filename);
        gpujpeg_image_destroy(d.input_image);
        if (d.decoder != NULL) {
            gpujpeg_decoder_destroy(d.decoder);
        }

        puts(rc == 0 ? "Success\n" : "FAILURE\n");

        return rc;
}

/* vim: set expandtab sw=4: */
