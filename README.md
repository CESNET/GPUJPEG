GPUJPEG
=======

[![License](https://img.shields.io/badge/License-BSD_2--Clause-orange.svg)](https://opensource.org/licenses/BSD-2-Clause)
[![C/C++ CI](../../workflows/C%2FC%2B%2B%20CI/badge.svg)](../../actions)

![GPUJPEG Logo](logo.svg)

**JPEG** encoder and decoder library and console application for **NVIDIA GPUs**
for high-performance image encoding and decoding. The software runs also on
**AMD GPUs** using [ZLUDA](https://github.com/vosen/ZLUDA) (see
[ZLUDA.md](ZLUDA.md)).

This documents provides an introduction to the library and how to use it. You
can also look to [FAQ.md](FAQ.md) for _performance tuning_
and additional information. To see _latest changes_
you can display file [NEWS.md](NEWS.md).

Table of contents
-----------------
- [Authors](#authors)
- [Features](#features)
- [Overview](#overview)
- [Performance](#performance)
  * [Encoding](#encoding)
  * [Decoding](#decoding)
- [Quality](#quality)
- [Compile](#compile)
- [Usage](#usage)
  * [libgpujpeg library](#libgpujpeg-library)
    + [Encoding](#encoding-1)
    + [Decoding](#decoding-1)
  * [GPUJPEG console application](#gpujpeg-console-application)
- [Requirements](#requirements)
- [License](#license)
- [References](#references)

Authors
-------
 - Martin Srom, CESNET z.s.p.o
 - Jan Brothánek
 - Petr Holub
 - Martin Jirman
 - Jiri Matela
 - Martin Pulec
 - Lukáš Ručka

Features
--------

- uses NVIDIA CUDA platform
- baseline Huffman 8-bit coding
- use of **JFIF** file format by default, **Adobe** and **SPIFF** is supported as well (used by encoder
  if JPEG internal color space is not representable by JFIF - eg. limited range **YCbCr BT.709** or **RGB**)
- use of _restart markers_ that allow fast parallel encoding/decoding
- Encoder by default creates _non-interleaved_ stream, optionally it can produce
  an _interleaved_ stream (all components in one scan) or/and subsampled stream.
- support for _color transformations_ and coding RGB JPEG
- Decoder can decompress JPEG codestreams that can be generated by encoder. If scan
  contains restart flags, decoder can use parallelism for fast decoding.
-  _command-line_ tool with support for encoding/decoding **raw** images
as well as **BMP**, **TGA**, **PNM/PAM** or **Y4M**

Overview
--------

Encoding/Decoding of JPEG codestream is divided into following phases:

     Encoding:                       Decoding
     1) Input data loading           1) Input data loading
     2) Preprocessing                2) Parsing codestream
     3) Forward DCT  + Quantization  3) Huffman decoder
     4) Huffman encoder              4) Dequantization + Inverse DCT
     5) Formatting codestream        5) Postprocessing

and they are implemented on CPU or/and GPU as follows:
 - CPU:
    - Input data loading
    - Parsing codestream
    - Huffman encoder/decoder (when restart flags are disabled)
    - Output data formatting
 - GPU:
    - Preprocessing/Postprocessing (color component parsing,
      color transformation RGB <-> YCbCr)
    - Forward/Inverse DCT (discrete cosine transform)
    - De/Quantization
    - Huffman encoder/decoder (when restart flags are enabled)

Performance
-----------

Source 16K (DCI) image ([8], [9]) was cropped to _15360x8640+0+0_ (1920x1080
multiplied by 8 in both dimensions) and for lower resolutions downscaled.
Encoding was done with default values with input in RGB (quality **75**,
**non-interleaved**, rst 24-36, average from 99 measurements excluding first
iteration) with following command:

    gpujpegtool -v -e mediadivision_frame_<res>.pnm mediadivision_frame_<res>.jpg -n 100 [-q <Q>]

### Encoding

|      GPU \ resolution      | HD (2 Mpix) | 4K (8 Mpix) | 8K (33 Mpix) | 16K (132 Mpix) |
|----------------------------|-------------|-------------|--------------|----------------|
|          RTX 3080          |   0.54 ms   |   1.71 ms   |    6.20 ms   |    24.48 ms    |
|          RTX 2080 Ti       |   0.82 ms   |   2.89 ms   |   11.15 ms   |    46.23 ms    |
|          GTX 1060M         |   1.36 ms   |   4.55 ms   |   17.34 ms   |  _(low mem)_   |
|          GTX 580           |   2.38 ms   |   8.68 ms   |  _(low mem)_ |  _(low mem)_   |
| AMD Radeon RX 7600 [ZLUDA] |   0.88 ms   |   3.16 ms   |   13.09 ms   |    50.52 ms    |

**Note:** First iteration is slower because the initialization takes place and
lasts about _28.6 ms_ for 8K (_87.1 ms_ for 16K) with RTX 3080 (but the
overhead depends more on CPU than the GPU).

Further measurements were performed on _RTX 3080_ only:

|             quality              | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 |
|----------------------------------|----|----|----|----|----|----|----|----|----|-----|
| duration HD (ms)                 |0.48|0.49|0.50|0.51|0.51|0.53|0.54|0.57|0.60| 0.82|
| duration 4K (ms)                 |1.61|1.65|1.66|1.67|1.69|1.68|1.70|1.72|1.79| 2.44|
| duration 8K (ms)                 |6.02|6.04|6.09|6.14|6.12|6.17|6.21|6.24|6.47| 8.56|
| duration 8K (ms, w/o PCIe xfers) |2.13|2.14|2.18|2.24|2.23|2.25|2.28|2.33|2.50| 5.01|

<!-- Additional notes (applies also for decode):
 1. device needs to be set to maximum performance, otherwise powermanagement influences esp. PCIe transmits
 2. stream formatter is starting to be a significant performance factor, eg. 0.82 ms for 8K Q=75 (contained in last line)
 3. measurements were done without -DCMAKE_BUILD_TYPE=Release, should be measured with -->

### Decoding

Decoded images were those encoded in previous section, averaging has been done similarly by
taking 99 samples excluding the first one. Command used:

    gpujpegtool -v mediavision_frame_<res>.jpg output.pnm -n 100

|       GPU \ resolution     | HD (2 Mpix) | 4K (8 Mpix) | 8K (33 Mpix) | 16K (132 Mpix) |
|----------------------------|-------------|-------------|--------------|----------------|
|          RTX 3080          |   0.75 ms   |   1.94 ms   |    6.76 ms   |    31.50 ms    |
|          RTX 2080 Ti       |   1.02 ms   |   1.07 ms   |   11.29 ms   |    44.42 ms    |
|          GTX 1060M         |   1.68 ms   |   4.81 ms   |   17.56 ms   |  _(low mem)_   |
|          GTX 580           |   2.61 ms   |   7.96 ms   | _(low mem)_  |  _(low mem)_   |
| AMD Radeon RX 7600 [ZLUDA] |   1.00 ms   |   3.02 ms   |   11.25 ms   |    45.06 ms    |

**Note**: _(low mem)_ above means that the card didn't have sufficient memory to
encode or decode the picture.

Following measurements were performed on _RTX 3080_ only:

|              quality             | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 |
|----------------------------------|----|----|----|----|----|----|----|----|----|-----|
| duration HD (ms)                 |0.58|0.60|0.63|0.65|0.67|0.69|0.73|0.78|0.89| 1.58|
| duration 4K (ms)                 |1.77|1.80|1.83|1.84|1.87|1.89|1.92|1.95|2.11| 3.69|
| duration 8K (ms)                 |6.85|6.88|6.90|6.92|6.98|6.70|6.74|6.84|7.17|12.43|
| duration 8K (ms, w/o PCIe xfers) |2.14|2.18|2.21|2.24|2.27|2.29|2.34|2.42|2.71| 7.27|

Quality
-----------
Following tables summarizes encoding quality and file size using NVIDIA
GTX 580 for non-interleaved and non-subsampled stream with different quality
settings (PSNR and encoded size values are averages of encoding several
images, each of them multiple times):

| quality |  PSNR 4K¹|   size 4K  |  PSNR HD²|  size HD   |
|---------|----------|------------|----------|------------|
|   10    | 29.33 dB |  539.30 kB | 27.41 dB |  145.90 kB |
|   20    | 32.70 dB |  697.20 kB | 30.32 dB |  198.30 kB |
|   30    | 34.63 dB |  850.60 kB | 31.92 dB |  243.60 kB |
|   40    | 35.97 dB |  958.90 kB | 32.99 dB |  282.20 kB |
|   50    | 36.94 dB | 1073.30 kB | 33.82 dB |  319.10 kB |
|   60    | 37.96 dB | 1217.10 kB | 34.65 dB |  360.00 kB |
|   70    | 39.22 dB | 1399.20 kB | 35.71 dB |  422.10 kB |
|   80    | 40.67 dB | 1710.00 kB | 37.15 dB |  526.70 kB |
|   90    | 42.83 dB | 2441.40 kB | 39.84 dB |  768.40 kB |
|  100    | 47.09 dB | 7798.70 kB | 47.21 dB | 2499.60 kB |

<b><sup>1,2</sup></b> sizes 4096x2160 and 1920x1080

Compile
-------
To build console application check [Requirements](#requirements) and go
to `gpujpeg` directory (where [README.md](README.md) and [COPYING](COPYING)
files are placed) and run `cmake` command:

    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native -Bbuild .
    cmake --build build --config Release

In Linux, you can also use **autotools** to create a build recipe for
the library and the application or a plain old _Makefile.bkp_. However,
_cmake_ is recommended.

Usage
-----

### libgpujpeg library
To build _libgpujpeg_ library check [Compile](#compile).

To use library in your project you have to include library to your
sources and linked shared library object to your executable:

    #include <libgpujpeg/gpujpeg.h>

For simple library usage examples you look into subdirectory [examples](examples).

#### Encoding
For encoding by libgpujpeg library you have to declare two structures
and set proper values to them. The first is definition of encoding/decoding
parameters, and the second is structure with parameters of input image:

    struct gpujpeg_parameters param = gpujpeg_default_parameters();
    // you can adjust parameters:
    param.quality = 80; // (default value is 75)

    struct gpujpeg_image_parameters param_image = gpujpeg_default_image_parameter();
    param_image.width = 1920;
    param_image.height = 1080;
    param_image.color_space = GPUJPEG_RGB; // input colorspace (GPUJPEG_RGB
                                           // default), can be also
                                           // eg. GPUJPEG_YCBCR_JPEG
    param_image.pixel_format = GPUJPEG_444_U8_P012;
    // or eg. GPUJPEG_U8 for grayscale
    // (default value is GPUJPEG_444_U8_P012)

If you want to use subsampling in JPEG format call following function,
that will set default sampling factors (2x2 for Y, 1x1 for Cb and Cr):

    // Use 4:2:0 subsampling
    gpujpeg_parameters_chroma_subsampling(&param, GPUJPEG_SUBSAMPLING_420);

Or define sampling factors by hand:

    // User custom sampling factors
    gpujpeg_parameters_chroma_subsampling(&param, MK_SUBSAMPLING(4, 4, 1, 2, 2, 1, 0, 0));

Next you can initialize CUDA device by calling (if not called, default CUDA
device will be used):

    if ( gpujpeg_init_device(device_id, 0) )
        return -1;

where first parameters is CUDA device (e.g. `device_id = 0`) id and second
parameter is flag if init should be verbose (`0` or `GPUJPEG_INIT_DEV_VERBOSE`).
Next step is to create encoder:

    struct gpujpeg_encoder* encoder = gpujpeg_encoder_create(0);
    if ( encoder == NULL )
        return -1;

When creating encoder, library allocates all device buffers which will be
needed for image encoding and when you encode concrete image, they are
already allocated and encoder will used them for every image. Now we need
raw image data that we can encode by encoder, for example we can load it
from file:

    size_t image_size = 0;
    uint8_t* input_image = NULL;
    if ( gpujpeg_image_load_from_file("input_image.rgb", &input_image,
             &image_size) != 0 )
        return -1;

Next step is to encode uncompressed image data to JPEG compressed data
by encoder:

    struct gpujpeg_encoder_input encoder_input;
    gpujpeg_encoder_input_set_image(&encoder_input, input_image);

    uint8_t* image_compressed = NULL;
    int image_compressed_size = 0;
    if ( gpujpeg_encoder_encode(encoder, &encoder_input, &image_compressed,
             &image_compressed_size) != 0 )
        return -1;

Compressed data are placed in internal encoder buffer so we have to save
them somewhere else before we start encoding next image, for example we
can save them to file:

    if ( gpujpeg_image_save_to_file("output_image.jpg", image_compressed,
             image_compressed_size, NULL) != 0 )
        return -1;

Now we can load, encode and save next image or finish and move to clean up
encoder. Finally we have to clean up so destroy loaded image and destroy
the encoder.

    gpujpeg_image_destroy(input_image);
    gpujpeg_encoder_destroy(encoder);

#### Decoding
For decoding we don't need to initialize two structures of parameters.
We only have to initialize CUDA device if we haven't initialized it yet and
create decoder:

    if ( gpujpeg_init_device(device_id, 0) )
        return -1;
    
    struct gpujpeg_decoder* decoder = gpujpeg_decoder_create(0);
    if ( decoder == NULL )
        return -1;

Now we have **two options**. The first is to do nothing and decoder will
postpone buffer allocations to decoding first image where it determines
proper image size and all other parameters (recommended).  The second option
is to provide input image size and other parameters (reset interval, interleaving)
and the decoder will allocate all buffers and it is fully ready when encoding
even the first image:

    // you can skip this code below and let the decoder initialize automatically
    struct gpujpeg_parameters param;
    gpujpeg_set_default_parameters(&param);
    param.restart_interval = 16;
    param.interleaved = 1;
    
    struct gpujpeg_image_parameters param_image;
    gpujpeg_image_set_default_parameters(&param_image);
    param_image.width = 1920;
    param_image.height = 1080;
    param_image.color_space = GPUJPEG_RGB;
    param_image.pixel_format = GPUJPEG_444_U8_P012;
    
    // Pre initialize decoder before decoding
    gpujpeg_decoder_init(decoder, &param, &param_image);

If you didn't initialize the decoder by `gpujpeg_decoder_init` but want
to specify output image color space and subsampling factor, you can use
following code:

    gpujpeg_decoder_set_output_format(decoder, GPUJPEG_RGB,
                    GPUJPEG_444_U8_P012);
    // or eg. GPUJPEG_YCBCR_JPEG and GPUJPEG_422_U8_P1020

If not called, RGB or grayscale is output depending on JPEG channel count.

Next we have to load JPEG image data from file and decoded it to raw
image data:

    size_t image_size = 0;
    uint8_t* image = NULL;
    if ( gpujpeg_image_load_from_file("input_image.jpg", &image,
             &image_size) != 0 )
        return -1;
    
    struct gpujpeg_decoder_output decoder_output;
    gpujpeg_decoder_output_set_default(&decoder_output);
    if ( gpujpeg_decoder_decode(decoder, image, image_size,
             &decoder_output) != 0 )
        return -1;

Now we can save decoded raw image data to file and perform cleanup:

    if ( gpujpeg_image_save_to_file("output_image.pnm", decoder_output.data,
             decoder_output.data_size, &decoder_output.param_image) != 0 )
        return -1;
    
    gpujpeg_image_destroy(image);
    gpujpeg_decoder_destroy(decoder);

### GPUJPEG console application
The console application gpujpeg uses _libgpujpeg_ library to demonstrate
it's functions. To build console application check [Compile](#compile).

To encode image from raw RGB image file to JPEG image file use following
command:

    gpujpegtool --encode --size=WIDTHxHEIGHT --quality=QUALITY \
            INPUT_IMAGE.rgb OUTPUT_IMAGE.jpg

You must specify input image size by `--size=WIDTHxHEIGHT` parameter.
Optionally you can specify desired output quality by parameter
`--quality=QUALITY` which accepts values 0-100. Console application accepts
a few more parameters and you can list them by folling command:

    gpujpegtool --help

To decode image from JPEG image file to raw RGB image file use following
command:

    gpujpegtool --decode OUTPUT_IMAGE.jpg INPUT_IMAGE.rgb

You can also encode and decode image to test the console application:

    gpujpegtool --encode --decode --size=WIDTHxHEIGHT --quality=QUALITY \
            INPUT_IMAGE.rgb OUTPUT_IMAGE.jpg

Decoder will create new decoded file `OUTPUT_IMAGE.jpg.decoded.rgb` and do
not overwrite your `INPUT_IMAGE.rgb` file.

Console application is able to load raw RGB image file data from *.rgb
files and raw YUV and YUV422 data from *.yuv files. For YUV422 you must
specify *.yuv file and use `--sampling-factor=4:2:2` parameter.

All supported parameters for console application are following:

    --help
        Prints console application help
    --size=1920x1080
        Input image size in pixels, e.g. 1920x1080
    --pixel-format=444-u8-p012
        Input/output image pixel format ('u8', '444-u8-p012', '444-u8-p012z',
        '444-u8-p0p1p2', '422-u8-p1020', '422-u8-p0p1p2' or '420-u8-p0p1p2')
    --colorspace=rgb
        Input image colorspace (supported are 'rgb', 'yuv' and 'ycbcr-jpeg',
        where 'yuv' means YCbCr ITU-R BT.601), when *.yuv file is specified,
        instead of default 'rgb', automatically the colorspace 'yuv' is used
    --quality
        Set output quality level 0-100 (default 75)
    --restart=8
        Set restart interval for encoder, number of MCUs between
        restart markers
    --subsampled
        Produce chroma subsampled JPEG stream
    --interleaved
        Produce interleaved stream
    --encode
        Encode images
    --decode
        Decode images
    --device=0
        By using this parameter you can specify CUDA device id which will
        be used for encoding/decoding.

Restart interval is important for parallel huffman encoding and decoding.
When `--restart=N` is used (default is 8), the coder can process each
N MCUs independently, and so he can code each N MCUs in parallel. When
`--restart=0` is specified, restart interval is disabled and the coder
must use CPU version of huffman coder (because on GPU would run only one
thread, which is very slow).

The console application can encode/decode multiple images by following
command:

    gpujpegtool ARGUMENTS INPUT_IMAGE_1.rgb OUTPUT_IMAGE_1.jpg \
            INPUT_IMAGE_2.rgb OUTPUT_IMAGE_2.jpg ...

Requirements
------------
To be able to build and run libgpujpeg **library** and gpujpeg **console
application** you need:

1. NVIDIA **CUDA Toolkit**
2. **C/C++ compiler** + **CMake**
3. CUDA enabled **NVIDIA GPU** (cc >= 2.0; older may or may not work) with
   NVIDIA drivers _or_ AMD with [ZLUDA](https://github.com/vosen/ZLUDA)
   (see [ZLUDA.md](ZLUDA.md))
4. _optional_ **OpenGL** support:
   - _GLEW_, _OpenGL_ (usually present in Windows, may need headers installation in Linux)
   - _GLFW_ or _GLX_ (Linux only) for context creation
   - _GLUT_ for OpenGL tests

License
-------
- See file [COPYING](COPYING).
- This software contains source code provided by NVIDIA Corporation.
- This software source code is based on SiGenGPU \[[3]\].
- Thanks [notings/stb](https://github.com/nothings/stb) (BMP+TGA image
support, public domain).

References
----------
[1]: http://www.w3.org/Graphics/JPEG/itu-t81.pdf
[2]: http://www.ijg.org/
[3]: https://github.com/silicongenome/SiGenGPU
[4]: https://www.ecma-international.org/publications/files/ECMA-TR/ECMA%20TR-098.pdf
[5]: https://www.itu.int/rec/dologin_pub.asp?lang=e&id=T-REC-T.84-199607-I!!PDF-E&type=items
[6]: https://www.fileformat.info/format/spiff/egff.htm
[7]: https://docs.oracle.com/javase/8/docs/api/javax/imageio/metadata/doc-files/jpeg_metadata.html#optcolor
[8]: https://www.newsshooter.com/2019/07/25/beyond-imax-filming-with-a-gigantic-16k-200mp-sensor/
[9]: https://we.tl/t-mjlrZM99EB

1. [ITU-T Rec T.81][1]
2. [ILG][2]
3. [SiGenGPU][3] (currently defunct)
4. [ECMA TR/098 (JFIF)][4]
5. [ITU-T Rec T.84 (SPIFF)][5]
6. [SPIFF File Format Summary (FileFormat.Info)][6]
7. [Component ID registration][7]
