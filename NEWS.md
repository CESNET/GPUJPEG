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
