#include <assert.h>
#include <cuda_runtime.h>
#include <libgpujpeg/gpujpeg_common.h>
#include <libgpujpeg/gpujpeg_encoder.h>
#include "../../src/gpujpeg_common_internal.h"
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TEST_IMAGE_WIDTH 1920
#define TEST_IMAGE_HEIGHT 1080

#define ASSERT_STR_EQ(expected, actual) do { if (strcmp(expected, actual) != 0) { fprintf(stderr, "Assertion failed! Expected '%s', result '%s'\n", expected, actual); abort(); } } while(0)

static void subsampling_name_test() {
        struct {
                enum gpujpeg_pixel_format fmt;
                const char *exp_subs_name;
        } test_pairs[] = {
                { GPUJPEG_U8, "4:0:0" },
                { GPUJPEG_420_U8_P0P1P2, "4:2:0" },
                { GPUJPEG_422_U8_P1020, "4:2:2" },
                { GPUJPEG_444_U8_P0P1P2, "4:4:4" },
                { GPUJPEG_444_U8_P012A, "4:4:4:4" },
        };
        for (size_t i = 0; i < sizeof test_pairs / sizeof test_pairs[0]; ++i) {
                const char *name =
                        gpujpeg_subsampling_get_name(gpujpeg_pixel_format_get_comp_count(test_pairs[i].fmt), gpujpeg_get_subsampling(test_pairs[i].fmt));
                ASSERT_STR_EQ(test_pairs[i].exp_subs_name, name);
        }
}

/*
 * Test if we can encode GPU pointer as usual CPU pointer.
 */
static void encode_gpu_mem_as_cpu() {
        struct gpujpeg_encoder *encoder = gpujpeg_encoder_create(0);
        if (encoder == NULL) { // do not fail here if we do not have CUDA capable device - just skip this test
                return;
        }

        struct gpujpeg_parameters param;
        gpujpeg_set_default_parameters(&param);

        struct gpujpeg_image_parameters param_image;
        gpujpeg_image_set_default_parameters(&param_image);
        param_image.width = TEST_IMAGE_WIDTH;
        param_image.height = TEST_IMAGE_HEIGHT;
        param_image.comp_count = 3;
        param_image.color_space = GPUJPEG_YCBCR_BT709;
        param_image.pixel_format = GPUJPEG_420_U8_P0P1P2;

        uint8_t *image = NULL;
        size_t len = param_image.width * param_image.height * 3 / 2;
        if (cudaSuccess != cudaMalloc((void**) &image, len)) {
                abort();
        }
        cudaMemset(image, 0, len);

        struct gpujpeg_encoder_input encoder_input;
        gpujpeg_encoder_input_set_image(&encoder_input, image);

        uint8_t *image_compressed = NULL;
        int image_compressed_size = 0;

        if (gpujpeg_encoder_encode(encoder, &param, &param_image, &encoder_input, &image_compressed, &image_compressed_size) != 0) {
                abort();
        }

        cudaFree(image);
        gpujpeg_encoder_destroy(encoder);
}

int main() {
        subsampling_name_test();
        encode_gpu_mem_as_cpu();
        printf("PASSED\n");
}

