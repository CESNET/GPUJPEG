/**
 * @file encode_raw.c
 *
 * Example showing GPUJPEG encode either raw RGB or raw grayscale image.
 */
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
        printf("\t%s <width> <height> file.rgb|file.r\n", progname);
}

static enum gpujpeg_image_file_format get_filetype(int argc, char *argv[]) {
        if (argc != 4 || strrchr(argv[3], '.') == NULL) {
          return GPUJPEG_IMAGE_FILE_UNKNOWN;
        }
        const char *ext = strrchr(argv[3], '.') + 1;
        return strcasecmp(ext, "rgb") == 0 ? GPUJPEG_IMAGE_FILE_RGB
               : strcasecmp(ext, "r") == 0 ? GPUJPEG_IMAGE_FILE_GRAY
                                           : GPUJPEG_IMAGE_FILE_UNKNOWN;
}

// to simplify deletion of allocated items on all paths
struct encode_data {
        char *out_filename;
        struct gpujpeg_encoder* encoder;
        uint8_t *input_image;
};

static int encode(int width, int height, int ch_count, const char *input_filename, struct encode_data *d)
{
        // set default encode parametrs, after calling, parameters can be tuned (eg. quality)
        struct gpujpeg_parameters param;
        gpujpeg_set_default_parameters(&param);

        // here we set image parameters
        struct gpujpeg_image_parameters param_image;
        gpujpeg_image_set_default_parameters(&param_image);
        param_image.width = width;
        param_image.height = height;
        param_image.color_space =
            ch_count == 1 ? GPUJPEG_YCBCR_JPEG : GPUJPEG_RGB;
        param_image.pixel_format =
            ch_count == 1 ? GPUJPEG_U8 : GPUJPEG_444_U8_P012;

        // create encoder
        if ((d->encoder = gpujpeg_encoder_create(0)) == NULL) {
                return 1;
        }

        // load image and set it as the encoder input buffer
        size_t input_image_size = 0;
        if (gpujpeg_image_load_from_file(input_filename, &d->input_image, &input_image_size) != 0) {
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
        d->out_filename = malloc(strlen(input_filename) + 1);
        strcpy(d->out_filename, input_filename);
        strcpy(strrchr(d->out_filename, '.') + 1, "jpg");
        if (gpujpeg_image_save_to_file(d->out_filename, image_compressed, image_compressed_size, &param_image) != 0) {
                return 1;
        }

        return 0;
}

int main(int argc, char *argv[]) {
        enum gpujpeg_image_file_format ftype = get_filetype(argc, argv);
        int ch_count = 0;
        switch (ftype) {
        case GPUJPEG_IMAGE_FILE_RGB:
                ch_count = 3;
                break;
        case GPUJPEG_IMAGE_FILE_GRAY:
                ch_count = 1;
                break;
        default:
                usage(argv[0]);
                return 1;
        }

        struct encode_data d = { 0 };
        int rc = encode(atoi(argv[1]), atoi(argv[2]), ch_count, argv[3], &d);
        if (rc == 0) {
                printf("Written: %s\n", d.out_filename);
        }

        free(d.out_filename);
        gpujpeg_image_destroy(d.input_image);
        if (d.encoder != NULL) {
            gpujpeg_encoder_destroy(d.encoder);
        }

        puts(rc == 0 ? "Success\n" : "FAILURE\n");

        return rc;
}
