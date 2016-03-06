#ifndef GPUJPEG_REFORMAT_H
#define GPUJPEG_REFORMAT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Add JPEG header(s) with segment offsets.
 *
 * @param in_image
 * @param in_image_size
 * @param out_image
 * @param out_image_size
 * @return 0 if succeeds, nonzero otherwise
 */
int
gpujpeg_reformat(uint8_t * in_image, int in_image_size, uint8_t ** out_image, int * out_image_size);

#ifdef __cplusplus
}
#endif

#endif // #ifndef GPUJPEG_REFORMAT_H
