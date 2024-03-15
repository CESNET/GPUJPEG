#include <assert.h>
#include <libgpujpeg/gpujpeg_decoder.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static uint8_t*
load_file(const char* fname, size_t* jpeg_len)
{
    FILE* in_file = fopen(fname, "rb");
    assert(in_file != NULL);
    fseek(in_file, 0, SEEK_END);
    *jpeg_len = ftell(in_file);
    uint8_t* image_data = malloc(*jpeg_len);
    fseek(in_file, 0, SEEK_SET);
    fread(image_data, *jpeg_len, 1, in_file);
    fclose(in_file);
    return image_data;
}

int
main(int argc, char* argv[])
{
    if ( argc <= 1 ) {
        printf("usage:\n%s <jpg>\n", argv[0]);
        return 1;
    }
    int ret = EXIT_SUCCESS;
    size_t jpeg_len = 0;
    uint8_t* jpeg_data = load_file(argv[1], &jpeg_len);
    struct gpujpeg_decoder* decoder = gpujpeg_decoder_create(0);
    assert(decoder != NULL);
    struct gpujpeg_decoder_output dec_output;
    gpujpeg_decoder_output_set_default(&dec_output);
    char fname[] = "out.XXX";
    if ( gpujpeg_decoder_decode(decoder, jpeg_data, jpeg_len, &dec_output) != 0 ||
         gpujpeg_image_save_to_file(fname, dec_output.data, dec_output.data_size, &dec_output.param_image) != 0 ) {
        fprintf(stderr, "decode or write failed!\n");
        ret = EXIT_FAILURE;
    }
    else {
        printf("Output written to %s\n", fname);
    }
    free(jpeg_data);
    gpujpeg_decoder_destroy(decoder);
    return ret;
}
