#include <assert.h>
#include <libgpujpeg/gpujpeg_decoder.h>
#include <libgpujpeg/gpujpeg_encoder.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Reuse one encoder and one decoder across JPEG layout and output format
 * changes and compare every result with freshly created coders.
 */

enum {
    REINIT_W = 256,
    REINIT_H = 256,
};

struct reinit_step
{
    gpujpeg_sampling_factor_t subsampling;
    int restart_interval;
    enum gpujpeg_color_space out_color_space;
    enum gpujpeg_pixel_format out_pixel_format;
    bool reuse_jpeg; ///< decode the previous JPEG again (only the output format changes)
};

static int
reinit_encode(struct gpujpeg_encoder* encoder, uint8_t* image, const struct reinit_step* step, uint8_t** out,
              size_t* out_len)
{
    struct gpujpeg_parameters param = gpujpeg_default_parameters();
    param.verbose = GPUJPEG_LL_QUIET;
    param.interleaved = 1;
    param.restart_interval = step->restart_interval;
    gpujpeg_parameters_chroma_subsampling(&param, step->subsampling);
    struct gpujpeg_image_parameters param_image = gpujpeg_default_image_parameters();
    param_image.width = REINIT_W;
    param_image.height = REINIT_H;
    struct gpujpeg_encoder_input input = gpujpeg_encoder_input_image(image);
    return gpujpeg_encoder_encode(encoder, &param, &param_image, &input, out, out_len);
}

static int
reinit_decode(struct gpujpeg_decoder* decoder, uint8_t* jpeg, size_t len, const struct reinit_step* step,
              struct gpujpeg_decoder_output* output)
{
    gpujpeg_decoder_set_output_format(decoder, step->out_color_space, step->out_pixel_format);
    gpujpeg_decoder_output_set_default(output);
    return gpujpeg_decoder_decode(decoder, jpeg, len, output);
}

static struct gpujpeg_decoder*
reinit_decoder_create(void)
{
    struct gpujpeg_decoder_init_parameters param = gpujpeg_decoder_default_init_parameters();
    param.verbose = GPUJPEG_LL_QUIET;
    return gpujpeg_decoder_create_with_params(&param);
}

static void
reinit_check(bool ok, unsigned step, const char* what)
{
    if ( !ok ) {
        fprintf(stderr, "step %u: reused %s differs from a fresh one!\n", step, what);
        abort();
    }
}

void
test_coder_reinit();

void
test_coder_reinit()
{
    printf("testing %s: ", __func__);
    if ( getenv("NO_CUDA_DEV") ) {
        printf("--\n");
        return;
    }
    static uint8_t image[REINIT_W * REINIT_H * 3];
    for ( unsigned i = 0; i < sizeof image; ++i ) {
        image[i] = (uint8_t)(i / 3 % REINIT_W + i / (3 * REINIT_W) * 2 + ((i * 2654435761U) >> 28));
    }
    const struct reinit_step steps[] = {
        {GPUJPEG_SUBSAMPLING_444, 8, GPUJPEG_RGB,        GPUJPEG_444_U8_P012,   false}, // GPU Huffman coder
        {GPUJPEG_SUBSAMPLING_420, 0, GPUJPEG_RGB,        GPUJPEG_444_U8_P012,   false}, // encoder subsampling change
        {GPUJPEG_SUBSAMPLING_444, 0, GPUJPEG_RGB,        GPUJPEG_444_U8_P012,   false}, // CPU Huffman buffer growth
        {GPUJPEG_SUBSAMPLING_420, 8, GPUJPEG_RGB,        GPUJPEG_444_U8_P012,   false},
        {GPUJPEG_SUBSAMPLING_420, 8, GPUJPEG_YCBCR_JPEG, GPUJPEG_420_U8_P0P1P2, true }, // output format change only
    };
    struct gpujpeg_encoder* encoder = gpujpeg_encoder_create(0);
    struct gpujpeg_decoder* decoder = reinit_decoder_create();
    assert(encoder != NULL && decoder != NULL);
    uint8_t* jpeg = NULL;
    size_t jpeg_len = 0;
    for ( unsigned i = 0; i < sizeof steps / sizeof steps[0]; ++i ) {
        const struct reinit_step* step = &steps[i];
        if ( !step->reuse_jpeg ) {
            struct gpujpeg_encoder* fresh = gpujpeg_encoder_create(0);
            uint8_t* out = NULL;
            size_t out_len = 0;
            reinit_check(reinit_encode(fresh, image, step, &out, &out_len) == 0, i, "encoder (fresh failed)");
            free(jpeg);
            jpeg = malloc(out_len);
            memcpy(jpeg, out, out_len);
            jpeg_len = out_len;
            gpujpeg_encoder_destroy(fresh);
            reinit_check(reinit_encode(encoder, image, step, &out, &out_len) == 0 && out_len == jpeg_len &&
                             memcmp(out, jpeg, jpeg_len) == 0,
                         i, "encoder");
        }
        struct gpujpeg_decoder* fresh = reinit_decoder_create();
        struct gpujpeg_decoder_output fresh_out;
        struct gpujpeg_decoder_output out;
        reinit_check(reinit_decode(fresh, jpeg, jpeg_len, step, &fresh_out) == 0, i, "decoder (fresh failed)");
        reinit_check(reinit_decode(decoder, jpeg, jpeg_len, step, &out) == 0 && out.data_size == fresh_out.data_size &&
                         memcmp(out.data, fresh_out.data, out.data_size) == 0,
                     i, "decoder");
        gpujpeg_decoder_destroy(fresh);
    }
    free(jpeg);
    gpujpeg_decoder_destroy(decoder);
    gpujpeg_encoder_destroy(encoder);
    printf("Ok\n");
}
