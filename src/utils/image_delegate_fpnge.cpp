#include "image_delegate_fpnge.h"

#include <cstdio>

#include "../gpujpeg_common_internal.h"
#include "fpnge.h"

int
fpnge_save_delegate(const char* filename, const struct gpujpeg_image_parameters* param_image, const char* data)
{
    if ( param_image->pixel_format != GPUJPEG_U8 && param_image->color_space != GPUJPEG_RGB ) {
        ERROR_MSG("Wrong color space %s for PNG!\n", gpujpeg_color_space_get_name(param_image->color_space));
        return -1;
    }
    int comp_count = 0;
    switch (param_image->pixel_format) {
    case GPUJPEG_U8:
        comp_count = 1;
        break;
    case GPUJPEG_444_U8_P012:
        comp_count = 3;
        break;
    case GPUJPEG_4444_U8_P0123:
        comp_count = 4;
        break;
    default:
        ERROR_MSG(
                "Wrong pixel format %s for PNG! Only packed formats "
                "without subsampling are supported.\n",
                gpujpeg_pixel_format_get_name(param_image->pixel_format));
        return -1;
    }

    struct FPNGEOptions opts{};
    FPNGEFillOptions(&opts, 1, FPNGE_CICP_NONE);
    FILE* outf = fopen(filename, "wb");
    if ( outf == nullptr ) {
        perror("");
        return -1;
    }
    char* out = (char*)malloc(FPNGEOutputAllocSize(1, comp_count, param_image->width, param_image->height));
    assert(out != nullptr);
    size_t bytes = FPNGEEncode(1, comp_count, data, param_image->width, (size_t)param_image->width * comp_count,
                               param_image->height, out, &opts);
    int ret = fwrite(out, bytes, 1, outf) == 1 ? 0 : -1;
    free(out);
    fclose(outf);
    return ret;
}
