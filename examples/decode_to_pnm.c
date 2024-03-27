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
        printf("\t%s file.jpg [file2.jpg...]\n", progname);
}

static bool check_params(int argc, char *argv[]) {
        if (argc < 2) {
                return false;
        }
        while ( *++argv != NULL ) {
            if ( strrchr(*argv, '.') == NULL ) {
                return false;
            }
            const char* ext = strrchr(*argv, '.') + 1;
            if ( strcasecmp(ext, "jpg") != 0 ) {
                return false;
            }
        }
        return true;
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
        if ( d->decoder == NULL ) {
            d->decoder = gpujpeg_decoder_create(0);
            if ( d->decoder == NULL ) {
                return 1;
            }
        }

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
        gpujpeg_image_destroy(d->input_image);
        d->input_image = NULL;

        // create output file name
        d->out_filename = realloc(d->out_filename, strlen(input_filename) + 1);
        strcpy(d->out_filename, input_filename);
        strcpy(strrchr(d->out_filename, '.') + 1, "pnm");

        // write the decoded image
        if ( gpujpeg_image_save_to_file(d->out_filename, decoder_output.data, decoder_output.data_size,
                                        &decoder_output.param_image) != 0 ) {
            return 1;
        }

        return 0;
}

int main(int argc, char *argv[]) {
        if (!check_params(argc, argv)) {
                usage(argv[0]);
                return 1;
        }

        int ret = EXIT_SUCCESS;

        struct decode_data d = {0};
        while ( *++argv != NULL ) {
            printf("decoding %s...\n", *argv);
            int rc = decode(*argv, &d);
            puts(rc == 0 ? "Success" : "FAILURE");
            ret += rc;
        }

        free(d.out_filename);
        gpujpeg_image_destroy(d.input_image);
        if ( d.decoder != NULL ) {
            gpujpeg_decoder_destroy(d.decoder);
        }

        return ret;
}

/* vim: set expandtab sw=4: */
