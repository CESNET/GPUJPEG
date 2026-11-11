#include <assert.h>
#include <cuda_runtime.h>
#include <libgpujpeg/gpujpeg_decoder.h>
#include <libgpujpeg/gpujpeg_encoder.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../../src/gpujpeg_decoder_internal.h"
#include "../../src/gpujpeg_encoder_internal.h"

/*
 * Compares the chroma filters with a CPU reference: the encoder's component planes after preprocessing and the
 * decoder's full-resolution output computed from its component planes. Input and output are in the JPEG color space,
 * so no color transform is involved.
 */

enum {
    CHROMA_W = 37,
    CHROMA_H = 21,
    CHROMA_PAD = 3, ///< input line padding in bytes
    CHROMA_PITCH = CHROMA_W * 3 + CHROMA_PAD,
};

#define CHROMA_CHECK(cond)                                                                                             \
    if ( !(cond) ) {                                                                                                   \
        fprintf(stderr, "%s:%d: chroma filter check '%s' failed!\n", __FILE__, __LINE__, #cond);                       \
        abort();                                                                                                       \
    }

static uint8_t*
chroma_plane(const struct gpujpeg_component* comp)
{
    size_t size = (size_t)comp->data_width * comp->height;
    uint8_t* plane = malloc(size);
    CHROMA_CHECK(cudaMemcpy(plane, comp->d_data, size, cudaMemcpyDeviceToHost) == cudaSuccess);
    return plane;
}

/// encoder reference: box average or point sample of channel ch for chroma sample (cx, cy), divisors h, v
static int
chroma_ref_down(const uint8_t* in, int ch, int cx, int cy, int h, int v, bool filter)
{
    if ( !filter ) {
        return in[cy * v * CHROMA_PITCH + cx * h * 3 + ch];
    }
    int sum = 0;
    for ( int dy = 0; dy < v; ++dy ) {
        for ( int dx = 0; dx < h; ++dx ) {
            int x = MIN(cx * h + dx, CHROMA_W - 1);
            int y = MIN(cy * v + dy, CHROMA_H - 1);
            sum += in[y * CHROMA_PITCH + x * 3 + ch];
        }
    }
    return (sum + h * v / 2) / (h * v);
}

/// decoder reference: libjpeg "fancy" upsampling (or replication) of plane p at pixel (x, y), divisors h, v
static int
chroma_ref_up(const uint8_t* p, const struct gpujpeg_component* comp, int x, int y, int h, int v, bool filter)
{
    const int pitch = comp->data_width;
    const int cx = x / h;
    const int cy = y / v;
    if ( !filter ) {
        return p[cy * pitch + cx];
    }
    const int nx = h == 2 ? (x % 2 ? MIN(cx + 1, comp->width - 1) : (cx > 0 ? cx - 1 : 0)) : cx;
    const int ny = v == 2 ? (y % 2 ? MIN(cy + 1, comp->height - 1) : (cy > 0 ? cy - 1 : 0)) : cy;
    if ( h == 2 && v == 2 ) { // weights 9 (nearest), 3, 3 and 1 (diagonal)
        return (9 * p[cy * pitch + cx] + 3 * p[cy * pitch + nx] + 3 * p[ny * pitch + cx] + p[ny * pitch + nx] +
                (x % 2 ? 7 : 8)) >> 4;
    }
    if ( h == 2 ) {
        return (3 * p[cy * pitch + cx] + p[cy * pitch + nx] + (x % 2 ? 2 : 1)) >> 2;
    }
    return (3 * p[cy * pitch + cx] + p[ny * pitch + cx] + (y % 2 ? 2 : 1)) >> 2;
}

/// @param interpolated whether the decoder interpolates this subsampling (precompiled 4:2:0, 4:2:2, 4:4:0 kernels)
/// @param filter       option value to set, -1 to keep the default (on)
static void
chroma_case(struct gpujpeg_encoder* encoder, struct gpujpeg_decoder* decoder, const uint8_t* in,
            gpujpeg_sampling_factor_t subsampling, bool interpolated, int filter)
{
    if ( filter >= 0 ) {
        const char* val = filter ? GPUJPEG_VAL_TRUE : GPUJPEG_VAL_FALSE;
        CHROMA_CHECK(gpujpeg_encoder_set_option(encoder, GPUJPEG_ENC_OPT_CHROMA_FILTER_BOOL, val) == GPUJPEG_NOERR);
        CHROMA_CHECK(gpujpeg_decoder_set_option(decoder, GPUJPEG_DEC_OPT_CHROMA_FILTER_BOOL, val) == GPUJPEG_NOERR);
    }
    filter = filter != 0;

    struct gpujpeg_parameters param = gpujpeg_default_parameters();
    param.verbose = GPUJPEG_LL_QUIET;
    param.interleaved = 1;
    gpujpeg_parameters_chroma_subsampling(&param, subsampling);
    struct gpujpeg_image_parameters param_image = gpujpeg_default_image_parameters();
    param_image.width = CHROMA_W;
    param_image.height = CHROMA_H;
    param_image.width_padding = CHROMA_PAD;
    param_image.color_space = GPUJPEG_YCBCR_JPEG;
    struct gpujpeg_encoder_input input = gpujpeg_encoder_input_image((uint8_t*)in);
    uint8_t* jpeg = NULL;
    size_t jpeg_len = 0;
    CHROMA_CHECK(gpujpeg_encoder_encode(encoder, &param, &param_image, &input, &jpeg, &jpeg_len) == 0);

    struct gpujpeg_coder* coder = &encoder->coder;
    for ( int c = 1; c < 3; ++c ) {
        const struct gpujpeg_component* comp = &coder->component[c];
        const int h = coder->sampling_factor.horizontal / comp->sampling_factor.horizontal;
        const int v = coder->sampling_factor.vertical / comp->sampling_factor.vertical;
        uint8_t* plane = chroma_plane(comp);
        for ( int cy = 0; cy < comp->height; ++cy ) {
            for ( int cx = 0; cx < comp->width; ++cx ) {
                CHROMA_CHECK(plane[cy * comp->data_width + cx] == chroma_ref_down(in, c, cx, cy, h, v, filter));
            }
        }
        free(plane);
    }

    gpujpeg_decoder_set_output_format(decoder, GPUJPEG_YCBCR_JPEG, GPUJPEG_444_U8_P012);
    struct gpujpeg_decoder_output output;
    gpujpeg_decoder_output_set_default(&output);
    CHROMA_CHECK(gpujpeg_decoder_decode(decoder, jpeg, jpeg_len, &output) == 0);
    CHROMA_CHECK(output.data_size == (size_t)CHROMA_W * CHROMA_H * 3);
    coder = &decoder->coder;
    for ( int c = 1; c < 3; ++c ) {
        const struct gpujpeg_component* comp = &coder->component[c];
        const int h = coder->sampling_factor.horizontal / comp->sampling_factor.horizontal;
        const int v = coder->sampling_factor.vertical / comp->sampling_factor.vertical;
        uint8_t* plane = chroma_plane(comp);
        for ( int y = 0; y < CHROMA_H; ++y ) {
            for ( int x = 0; x < CHROMA_W; ++x ) {
                CHROMA_CHECK(output.data[(y * CHROMA_W + x) * 3 + c] ==
                             chroma_ref_up(plane, comp, x, y, h, v, filter && interpolated));
            }
        }
        free(plane);
    }
}

static struct gpujpeg_decoder*
chroma_decoder_create(void)
{
    struct gpujpeg_decoder_init_parameters param = gpujpeg_decoder_default_init_parameters();
    param.verbose = GPUJPEG_LL_QUIET;
    return gpujpeg_decoder_create_with_params(&param);
}

void
test_chroma_filter();

void
test_chroma_filter()
{
    printf("testing %s: ", __func__);
    if ( getenv("NO_CUDA_DEV") ) {
        printf("--\n");
        return;
    }
    static uint8_t in[CHROMA_H * CHROMA_PITCH];
    for ( unsigned i = 0; i < sizeof in; ++i ) {
        in[i] = (uint8_t)((i * 2654435761U) >> 24);
    }
    const struct
    {
        gpujpeg_sampling_factor_t subsampling;
        bool interpolated;
    } cases[] = {
        {GPUJPEG_SUBSAMPLING_420, true },
        {GPUJPEG_SUBSAMPLING_422, true },
        {GPUJPEG_SUBSAMPLING_440, true },
        {GPUJPEG_SUBSAMPLING_411, false}, // dynamic kernels: encoder averages, decoder replicates
    };
    for ( unsigned i = 0; i < sizeof cases / sizeof cases[0]; ++i ) {
        struct gpujpeg_encoder* encoder = gpujpeg_encoder_create(0);
        struct gpujpeg_decoder* decoder = chroma_decoder_create();
        assert(encoder != NULL && decoder != NULL);
        // default (on), then the option changed between calls without a reinitialization
        for ( int filter = -1; filter < 2; ++filter ) {
            chroma_case(encoder, decoder, in, cases[i].subsampling, cases[i].interpolated, filter);
        }
        gpujpeg_decoder_destroy(decoder);
        gpujpeg_encoder_destroy(encoder);
    }
    printf("Ok\n");
}
