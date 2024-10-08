#include <assert.h>
#include <cuda_runtime.h>
#include <libgpujpeg/gpujpeg_encoder.h>
#include <stdio.h>
#include <stdlib.h>

static void
cpu(struct gpujpeg_encoder* encoder, struct gpujpeg_parameters* param, struct gpujpeg_image_parameters* param_image)
{

    struct gpujpeg_encoder_input encoder_input;
    uint8_t *blank_buffer = NULL;
    blank_buffer = malloc(gpujpeg_image_calculate_size(param_image));
    gpujpeg_encoder_input_set_image(&encoder_input, blank_buffer);
    uint8_t *out = NULL;
    size_t len = 0;
    gpujpeg_encoder_encode(encoder, param, param_image, &encoder_input, &out,
                           &len);
    free(blank_buffer);
}

static void
gpu(struct gpujpeg_encoder* encoder, struct gpujpeg_parameters* param, struct gpujpeg_image_parameters* param_image)
{

    struct gpujpeg_encoder_input encoder_input;
    uint8_t *blank_buffer = NULL;
    cudaMalloc((void **) &blank_buffer, gpujpeg_image_calculate_size(param_image));
    gpujpeg_encoder_input_set_gpu_image(&encoder_input, blank_buffer);
    uint8_t *out = NULL;
    size_t len = 0;
    gpujpeg_encoder_encode(encoder, param, param_image, &encoder_input, &out,
                           &len);
    cudaFree(blank_buffer);
}

static void
check_stats(struct gpujpeg_encoder* encoder)
{
    struct gpujpeg_duration_stats stats;
    int rc = gpujpeg_encoder_get_stats(encoder, &stats);
    if ( rc != 0 ) {
        fprintf(stderr, "gpujpeg_encoder_get_stats returned %d!\n", rc);
        abort();
    }
}

void
test_gh_95();

void
test_gh_95()
{
    printf("testing stat struct validity on CPU/GPU interleave: ");
    struct gpujpeg_encoder *encoder = gpujpeg_encoder_create(0);
    assert(encoder != NULL);
    struct gpujpeg_parameters param = gpujpeg_default_parameters();
    param.verbose = -1;
    param.perf_stats = 1;
    struct gpujpeg_image_parameters param_image = gpujpeg_default_image_parameters();
    param_image.width = 640;
    param_image.height = 480;
    param_image.color_space = GPUJPEG_YCBCR_JPEG;
    param_image.pixel_format = GPUJPEG_U8;

    cpu(encoder, &param, &param_image);
    check_stats(encoder);
    gpu(encoder, &param, &param_image);
    check_stats(encoder);
    cpu(encoder, &param, &param_image);
    check_stats(encoder);
    gpu(encoder, &param, &param_image);
    check_stats(encoder);
    cpu(encoder, &param, &param_image);
    check_stats(encoder);

    gpujpeg_encoder_destroy(encoder);
    printf("Ok\n");
}
