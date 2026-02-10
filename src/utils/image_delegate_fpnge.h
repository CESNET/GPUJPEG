#ifdef __cplusplus
extern "C" {
#endif

struct gpujpeg_image_parameters;
int
fpnge_save_delegate(const char* filename, const struct gpujpeg_image_parameters* param_image, const char* data);

#ifdef __cplusplus
}
#endif
