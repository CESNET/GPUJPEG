#include <cuda_runtime.h>
#include <libgpujpeg/gpujpeg_encoder.h>
#include <stdio.h>
#include <stdlib.h>
#include <threads.h>

#define WIDTH 1920
#define HEIGHT 1080

int iterations;

static int worker(void *input_image) {
        cudaSetDevice(0);
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        struct gpujpeg_encoder *encoder = gpujpeg_encoder_create(stream);
        if (!encoder) {
                return 1;
        }

        struct gpujpeg_parameters param;
        gpujpeg_set_default_parameters(&param);

        // here we set image parameters
        struct gpujpeg_image_parameters param_image;
        gpujpeg_image_set_default_parameters(&param_image);
        param_image.width = WIDTH;
        param_image.height = HEIGHT;
        param_image.comp_count = 3;
        param_image.color_space = GPUJPEG_RGB;
        struct gpujpeg_encoder_input encoder_input;
        gpujpeg_encoder_input_set_image(&encoder_input, input_image);

        for (int i = 0; i < iterations; ++i) {
                uint8_t *image_compressed = NULL;
                int image_compressed_size = 0;
                if (gpujpeg_encoder_encode(encoder, &param, &param_image, &encoder_input, &image_compressed, &image_compressed_size) != 0) {
                        return 1;
                }
        }
        gpujpeg_encoder_destroy(encoder);
        return 0;
}

int main(int argc, char *argv[]) {
        if (argc != 3) {
                printf("Usage: %s <iterations_per_thread> <threads>\n", argv[0]);
                return 1;
        }
        iterations = atoi(argv[1]);
        int threads = atoi(argv[2]);

        uint8_t *input_image = NULL;
        cudaMallocHost((void *) &input_image, WIDTH * HEIGHT * 3);
        for (int i = 0; i < WIDTH * HEIGHT * 3; ++i) {
            input_image[i] = i % 255;
        }

        thrd_t tid[threads];
        for (int i = 0; i < threads; ++i) {
                thrd_create(&tid[i], worker, input_image);
        }
        for (int i = 0; i < threads; ++i) {
                int ret = 0;
                thrd_join(tid[i], &ret);
                if (ret != 0) {
                        return 1;
                }
        }
        cudaFreeHost(input_image);
}

