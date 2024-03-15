// Minimal encoding sample.
#include <assert.h>
#include <libgpujpeg/gpujpeg_encoder.h>
#include <stdio.h>
#include <stdlib.h>

#define OUT_FNAME "out.jpg"

int
main()
{
    struct gpujpeg_encoder *encoder = gpujpeg_encoder_create(0);
    assert(encoder != NULL);
    struct gpujpeg_parameters param;
    gpujpeg_set_default_parameters(&param);
    struct gpujpeg_image_parameters param_image;
    gpujpeg_image_set_default_parameters(&param_image);
    param_image.width = 640;
    param_image.height = 480;
    param_image.color_space = GPUJPEG_YCBCR_JPEG;
    param_image.comp_count = 1;
    param_image.pixel_format = GPUJPEG_U8;
    struct gpujpeg_encoder_input encoder_input;
    uint8_t *blank_buffer =
        calloc(1, gpujpeg_image_calculate_size(&param_image));
    gpujpeg_encoder_input_set_image(&encoder_input, blank_buffer);
    uint8_t *out = NULL;
    size_t len = 0;
    gpujpeg_encoder_encode(encoder, &param, &param_image, &encoder_input, &out,
                           &len);
    FILE *outf = fopen(OUT_FNAME, "wb");
    fwrite(out, len, 1, outf);
    printf("Ouput " OUT_FNAME " was written.\n");
    fclose(outf);
    free(blank_buffer);
    gpujpeg_encoder_destroy(encoder);
}
