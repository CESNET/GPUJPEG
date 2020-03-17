#include <assert.h>
#include <libgpujpeg/gpujpeg_common.h>
#include <libgpujpeg/gpujpeg_common_internal.h>
#include <stdbool.h>
#include <string.h>

static void subsampling_name_test() {
        struct {
                enum gpujpeg_pixel_format fmt;
                const char *exp_subs_name;
        } test_pairs[] = {
                { GPUJPEG_U8, "4:0:0" },
                { GPUJPEG_420_U8_P0P1P2, "4:2:0" },
                { GPUJPEG_422_U8_P1020, "4:2:2" },
                { GPUJPEG_444_U8_P0P1P2, "4:4:4" },
        };
        for (size_t i = 0; i < sizeof test_pairs / sizeof test_pairs[0]; ++i) {
                const char *name =
                        gpujpeg_subsampling_get_name(gpujpeg_pixel_format_get_comp_count(test_pairs[i].fmt), gpujpeg_get_component_subsampling(test_pairs[i].fmt));
                assert(strcmp(name, test_pairs[i].exp_subs_name) == 0);
        }
}

int main() {
        subsampling_name_test();
        printf("PASSED\n");
}

