2025-10-22 - 0.27.10
----------

- provide orientation metadata by decoder
- SPIFF: read/write the orientation
- Exif: remove custom tag storing API

2025-10-20 - 0.27.9
----------

- add gpujpeg_decoder_get_image_info2 as a extensible replacement for
gpujpeg_decoder_get_image_info
- fetch header_type with gpujpeg_decoder_get_image_info2 + print with
gpujpegtool (`-I`)

2025-08-22 - 0.27.8
----------

- implement channel remapping via option in pre/postprocessor

2025-07-10 - 0.27.7
----------

- add encoder option for vertically flipped images

2025-07-08 - 0.27.6
----------

- deprecate gpujpeg_encoder_set_jpeg_header() in favor of opt API
- as the consequence, gpujpegtool is able to pass that opt (`-O enc_hdr=Adobe`)

2025-06-10 - 0.27.5
----------

- support for non-ASCII characters in filename (tool+_load/_save)

2025-05-23 - 0.27.4
----------

- added encoder option to provide the encoded JPEG in a pinned buffer

2025-05-19 - 0.27.3
----------

- added support for UYVY files (.uyvy extension)

2025-05-19 - 0.27.2
----------

- gpujpegtool: add -H/--fullhelp for addtional opts
- added gpujpeg_endoder_set_option with GPUJPEG_ENCODER_OPT_TGA_RLE
- added gpujpegtool option -O for the options

2025-05-14 - 0.27.1
----------

- gpujpeg_image_save_to_file() - take (const char *) for filename instead
of (char *)

2025-05-12 - 0.27.0
----------

- bug fixes (including build)
- pin lesser amount of RAM - improves performance if small amount of
images given in a batch; also reduces the physical memory used
- do not allocate pinned memory for quantizied data if Huffman is on GPU
- support reading/writing BMP and TGA (thanks to notings/stb)

2025-01-17 - 0.26.0
----------

This release brings mainly much faster start-up (and reinitialization) times
in Linux, which is significant especially when encoding small amount of images.
The speed-up for first frame is as much as 20x (eg. 43 ms for 10000x10000.tst
compared to original 810 ms). Subsequent images are still faster but this
narrows the gap. The improvement is especially noticable for big images.

### API changes
- \[changed\] log level version numbers (verbose + debug +1)
- \[added\] log levels symbolic names
- \[renamed\] GPUJPEG_VERBOSE to GPUJPEG_INIT_DEV_VERBOSE
- \[added\] struct gpujpeg_encoder_input init functions returning the struct
(like gpujpeg_encoder_input_gpu_image) to be able to directly initialize
the variable
- if verbosity is at least GPUJPEG_LL_STATUS, gpujpeg_parameters.perf_stats
doesn't need to be set to output coding duration

### Other
- new patterns (noise, blank) to test images (.tst extension)
- cmake - add hint to enable the architecture to native (to speed up startup)
- improved logging of JPEG reader in debug mode to inspect JPEG structure
- print also coded image size and properties (useful if autodeduced)
- report also (re)initialization duration (in verbose mode)

2024-11-11 - 0.25.6
----------
- gpujpeg_decoder_get_image_info(): fill also interleaved (+ print
with gpujpegtool)

2024-10-08 - 0.25.5
----------
- added gpujpeg_decoder_create_with_params() (+ struct
  gpujpeg_decoder_init_parameters, gpujpeg_decoder_default_init_parameters())
- added gpujpeg_decoder_init_parameters.ff_cs_itu601_is_709 flag to
  override colorspace to limited 709 if FFmpeg-specific comment CS=ITU601
  is present (as 709 is used by UltraGrid)

2024-10-07 - 0.25.4
----------
- deprecate gpujpeg_decoder_get_stats()/gpujpeg_decoder_get_stats()

2024-06-05 - 0.25.3
----------
- added gpujpeg_color_space_by_name
- added gpujpeg_print_pixel_format

2024-04-09 - 0.25.2
----------
- allowed setting subsampling 4:0:0 for color input image
- allowed passing GPUJPEG_SUBSAMPLING_UNKNOWN to
  gpujpeg_parameters_chroma_subsampling() to disable explicitly set
  subsampling and return to implicit (automatic) one on the encoder

2024-04-05 - 0.25.1
----------
- added gpujpeg_default[_image]_parameters() functions returning the
  structure instead of setting it through a pointer. This is better
  for conveninece to intitialize defined structs vars immediately.

2024-04-05 - 0.25.0
----------
- removed deprecated gpujpeg_parameters_chroma_subsampling_42{0,2}

2024-03-19 - 0.24.0
----------
- replaced GPUJPEG_444_U8_P012{A,Z} with GPUJPEG_4444_U8_P0123

2024-03-18 - 0.23.0
----------
- moved comp_count from gpujpeg_image_parameters to gpujpeg_parameters. Being
  in image parameter was from historic reasons but has been replaced by pixel
  format. The comp_count now represents rather count of components inside JPEG.

2024-03-15 - 0.22.0
----------
- support for ZLUDA
- updated API - provide more information about the decompressed image in
  gpujpeg_decoder_output, namely the image size

2023-06-02 - 0.21.0
----------
- support for larger images - until now, only pictures up to approx. 85 MP were
  supported, now it is something like 512 megapixels (pratical limit is the
  size of GPU RAM, however, 512 Mpix would currently require some 26.5 GB)
- updated API - all image sizes are now size\_t

2023-01-06 - 0.20.4
----------
- added support for Y4M files (only one image per file)
- removed implicit synchronization issues when multiple encoders are run in
  parallel; also removed cudaMemcpy for the decoder (but not tested if there
  are not any other issues)
- very slight performance improvements

2022-01-21 - 0.20.2
----------
- support for encoding alpha

2021-01-21 - 0.14.0
----------
- Support for encoding arbitrary images in planar formats

2021-01-20 - 0.13.0
----------
- Decoding of arbitrary images into planar formats

2020-10-02
----------
- Support for SPIFF files (decode, encode limited-range BT.709, BT.601)
- Support for FFmpeg limited-range BT.601 JPEGs

2020-05-12
----------
- Support for PNM and PAM files

2020-03-10
----------
- Command-line application is now supported in MSW

2020-01-21
----------
- Support for RGBA buffers
- Raw RGBA and I420 files supported

2019-12-02
----------
- Decoder compatibility with more (foreign) JPEGs

2019-06-18
----------
- Support for RGB JPEG images

2016-03-07
----------
- Support for decoding to OpenGL texture or CUDA buffer
- Encoding different image sizes
- Support for encoding of planar images (limited)

2013-03-19
----------
- New IDCT computing kernel 43 % faster

2012-08-02
----------
Changes:
-Added new parallel GPU Huffman encoder for compute capabilities >= 2.0
-Rewritten GPU Huffman decoder
-Rewritten GPU forward DCT, preserving precision of fixed point implementation 
 and gaining better performance than NPP implementation.
 (Both old implementations were thus removed.)
-Rewritten GPU encoder preprocessor to remove unnecessary operations
-Minor performance improvement in GPU encoder postprocessor

2012-03-29
----------
Changes:
-Optionally the encoder input or decoder output can be loaded/store
 from/to OpenGL texture.

2012-03-09
----------
Changes:
-Refactored color spaces transformations, added new color spaces
-Added new implementation of DCT/IDCT on CUDA, it is slightly slower
 than the NPP implementation, but the new IDCT doesn't cause color space 
 change. It can be turned on by Makefile option. By default is NPP 
 version used.

2012-02-24
----------
Changes:
-Slightly improved performance of encoder.
-Correction of segment info for large images

2012-02-21
----------
Changes:
-Added option to encoder for generating segment info into APP13 header that is used
 by decoder to perform fast stream parsing.
-Renamed --interleving option to --interleaved and --chroma-subsampling to --subsampled.

2012-02-16
----------
Changes:
-Added option for verbose output from console application.
-Library now stores all coder time durations inside coder structure and it
 can be accessed from userspace.
-Console application now prints GPU allocation info (when --verbose is used).
-Correction of encoding/decoding for large images (e.g. 4320p or 8K)

2012-02-07
----------
-Improved preprocessor and postprocessor performance.
-Added recovering from error inside JPEG stream to decoder.
-Library provides function for listing CUDA devices.
-Library provides option for decoder to output result right into OpenGL PBO resource
 which improves performance when the result should be displayed on the same GPU.
 -Correction of build warnings.
 -Correction of other errors.

2012-01-04
----------
Changes:
-Added option to libgpujpeg Makefile for moving gpu huffman coder tables into constant memory.
-Option is default set to true and makes better performance on older GPUs.

2012-01-03
----------
Changes:
-Added interleaved mode (optionally).
-Added subsampling (optionally) - better performance when used 
 (because of smaller amount of encoded/decoded data).

2011-12-07
----------
First public release of gpujpeg library and console 
application.
Features:
-Baseline JPEG, non-interleaved mode on CUDA-enabled GPUs.
-Performance - realtime HD and 4K encoding/decoding on NVIDIA GTX 580.
-Using CUDA and NPP library.
