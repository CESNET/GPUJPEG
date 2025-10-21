# Frequently Asked Questions

- [Encoding/decoding is slow in first iteration](#encodingdecoding-is-slow-in-first-iteration)
- [What is a restart interval](#what-is-a-restart-interval)
- [Decoding is too slow](#decoding-is-too-slow)
- [Encoding different color spaces than full-range YCbCr BT.601](#encoding-different-color-spaces-than-full-range-ycbcr-bt601)
- [Optimizing encoding/decoding performance](#optimizing-encodingdecoding-performance)
- [Decoding (foreign) JPEG fails](#decoding-foreign-jpeg-fails)
- [Channel remapping - ARGB etc. encode/decode](#channel-remapping---argb-etc-encodedecode)
- [Encoding/decoding alpha channel](#encodingdecoding-alpha-channel)
   - [Alpha support in command-line application](#alpha-support-in-command-line-application)
   - [API for alpha](#api-for-alpha)
- [What are memory requirements for encoding/decoding](#what-are-memory-requirements-for-encodingdecoding)
- [How to preinitialize the decoder](#how-to-preinitialize-the-decoder)

## Encoding/decoding is slow in first iteration

Correct. This is because the there is initialization of GPUJPEG internal
structures, CUDA buffers, the initialization of GPU execution pipeline
as well as kernel compilation for actual device capability. The last
point can be eliminated by generating code for the particular device
during the compilation:

    cmake -DCMAKE_CUDA_ARCHITECTURES=native -DCMAKE_BUILD_TYPE=Release ...

(`all-major` or `all` will also work but the compilation will take longer)

Ideal use case for GPUJPEG is to run for many images (ideally equal-sized).

## What is a restart interval

A **restart interval** and related option in console application is a way
to increase paralelism to allow efficient Huffman encoding and decoding on
GPU. It is given by the number of MCU (minmum coded units, approximately
same as macroblocks) that can be encoded or decoded independently.

For the _encoder_, the restart interval is given as a member variable
`restart_interval` of `struct gpujpeg_parameters`. Higher value result
in smaller JPEG images but slower encoding. Good values are between 8
(default set by gpujpeg_set_default_parameters()) or 16. Disabling restart
intervals (setting value to 0) causes that the Huffman encoding/decoding
is done on CPU (while the rest is still performed by GPU). On larger
images, the restart interval can be a bit larger because there are more
MCUs. _gpujpegtool_ provides _-r_ option (if not set, a eligible
runtime-determined value is used).

For the _decoder_ the value cannot be changed by the decoder because it is an
attribute of the encoded JPEG.

## Decoding is too slow
It can be an effect of not using restart intervals in the JPEG (eg. by using
an encoder other than GPUJPEG). You can check number of segments with followng
command:

    gpujpeg -I image.jpg
    [...]
    Segment count: 129600 (DRI = 12)

The values in the order of magnitude in hundreds or thousands mean that the number
of segments should not be a problem.

You can also benchmark and find the potential bottleneck by running:

    gpujpeg -v -d image.jpg image.pnm
    [...]
     -Stream Reader:         543.33 ms
     -Copy To Device:          1.26 ms
     -Huffman Decoder:         1.27 ms
     -DCT & Quantization:      4.27 ms
     -Postprocessing:          2.89 ms
     -Copy From Device:        8.43 ms
    Decode Image GPU:         56.64 ms (only in-GPU processing)
    Decode Image Bare:       600.00 ms (without copy to/from GPU memory)
    Decode Image:            609.70 ms
    Save Image:              139.37 ms

which shows duration of individual decoding steps (use `-n <iter>` to see
duration of more iterations with the same image).

## Encoding different color spaces than full-range YCbCr BT.601
For compatibility reasons, GPUJPEG produces a full-range **YCbCr BT.601** with **JFIF**
header, color space conversions are performed by the encoder if needed. There is, however,
a possibility to encode also different color spaces like **RGB** or **YCbCr BT.709**
(limited range). Since JFIF supports **BT.601** YCbCr or grayscale only, **SPIFF** (for BT.709)
or **Adobe** (RGB) format is used in this case. Especially the first one is not widely used so
it may introduce some compatibility problems when not decoded by the _GPUJPEG_ decoder.

Usage in code is simple, just set `gpujpeg_parameters::internal_color_space` to required JPEG
internal representation.

This can be done also with the _command-line_ application (it just preserves input
color space, it cannot be changed by the app now). The relevant option is "**-N**"
(native):


    gpujpeg -s 1920x1080 -N -e image.rgb image.jpg

**Note:** **SPIFF** is not a widely adopted format of _JPEG_ files so is
hightly probable that the decoder other than GPUJPEG won't support the picture
and will ignore the color-space information.

## Optimizing encoding/decoding performance
To optimze encoding/decoding performance, following features can be tweaked (in order of importance):

1. restart intervals turned on and set to reasonable value (see autotuning in main.c)
2. enabling segment info (needs to be set on _encoder_, speeds up _decoder_)
3. avoiding color space conversions - setting `gpujpeg_parameters::internal_color_space` equal to
   `gpujpeg_image_parameters::color_space``
4. reducing quality - the lower quality, the lesser work for entropy encoder and decoder

Also a very helpful thing to use the image already in GPU memory is if possible to avoid costy
memory transfers.

It is also advisable to look at individual performance counters to see performance bottlenecks
(parameter `-v` for command-line application, see relevant code in _main.c_ to see how to use
in custom code).

## Decoding (foreign) JPEG fails
**GPUJPEG** is always capable of decoding the JPEG encoded by itself. As the standard allows
much options, including _progressive encoding_, _arithmetic coding_ etc., not all options
are supported. Basically a **baseline** **DCT-based** **Huffman-encoded** JPEGs are supported.
Few features of **extended** process are supported as well (4 Huffman tables). If the decoder
is incapable of decoding the above mentioned JPEG, you are encouraged to fill a bug report.

## Channel remapping - ARGB etc. encode/decode

Pixel formats with different channel order can be pre/postprocessed with
encoder or decoder option, eg.:

    gpujpegtool -s 1920x1080 -c rgb -f 4444-u8-p0123 -e in.gbra -O enc_opt_channel_remap=2103 out.jpg
    gpujpegtool -O dec_opt_channel_remap=210 in.jpg out.pnm

or in code:

    gpujpeg_encoder_set_option(encoder, GPUJPEG_ENC_OPT_CHANNEL_REMAP, "1230");

## Encoding/decoding alpha channel
Encoding is currently supported only for a single packed pixel format
444-u8-p012a (`GPUJPEG_444_U8_P012A`). Let us know if you'd need some other
like planar format.

### Alpha support in command-line application
To use with command line application, you'd need to use option `-a` for encode
and use a pixel format that suppors 4 channels eg. RGBA, some examples:

     gpujpeg -a -e -s 1920x1080 input.rgba output.jpg
     gpujpeg -a -e input.pam output.jpg
     gpujpeg -a -e input.yuv -f 444-u8-p102a output.jpg # YUVA

For decoding, you'd need to have 4-channel JPEG, no special tweaking is needed,
just using proper output pixel format, eg:

    gpujpeg -d input.jpg output.rgba
    gpujpeg -d input.jpg output.pam


**Note:** _GPUJPEG_ produces SPIFF headers for generated JPEG files - only a
few will recognize that but some do recognize component IDs ('R', 'G', 'B', 'A'
and 0x01, 0x02, 0x03, 0x04).

### API for alpha
#### Encode
Encoding alpha is quite simple, as indicated above, just set the pixel format `GPUJPEG_444_U8_P0123`
as `gpujpeg_image_parameters::pixel_format` and set subsampling to _4:4:4:4_ :
`gpujpeg_parameters_chroma_subsampling(param, GPUJPEG_SUBSAMPLING_4444);`.`

#### Decode
Select output pixel format either `GPUJPEG_444_U8_P0123` or
`GPUJPEG_PIXFMT_AUTODETECT` (RGB will be used if set to GPUJPEG_PIXFMT_NONE).

## What are memory requirements for encoding/decoding

Currently you can count about _20 bytes_ of **GPU memory** for every
_pixel and component_ for both encode and decode, eg. for 33 Mpix 4:4:4
frame it is _7680x4320x3x20=1901 MiB_. If the JPEG is 4:2:0 subsampled,
the memory requirements would be halfway.

You can check the amount of required GPU memory by running (adjust the
image format and parameters according to your needs):

    $ gpujpegtool -v -e 1920x1080.tst /dev/null   # or output file "nul" in MSW
    $ gpujpegtool -v -S -N -ai -r 16 -e 1920x1080.p_4444-u8-p0123.tst /dev/null
    ...
        Total GPU Memory Size:    102.6 MiB

The memory requirements may be excessive if dealing with really huge
images - let us know if there is a problem with this.

# How to preinitialize the decoder

Usually the GPUJPEG decoder full initialization is postponed to decoding
first image where it determines proper image size and all other parameters
(recommended).

If all image parameters (including restart interval and interleaving)
are known, the preinitialization may be performed immediately:

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

The `color_space` and `pixel_format` doesn't need to match the JPEG
properties, you can set eg.:

    param_image.color_space = GPUJPEG_YCBCR_JPEG;
    param_image.pixel_format = GPUJPEG_422_U8_P1020;

to enforce decoding to UYVY. Alternatively, you can set either color
space or pixel format to an _automatic_ value:

    param_image.color_space = GPUJPEG_NONE;
    param_image.pixel_format = GPUJPEG_PIXFMT_AUTODETECT;

In this case, the _color_space_ and/or _pixel_format_ will be set to
a format that is closest to internal JPEG representation. For the color
space this usually means _GPUJPEG_YCBCR_JPEG_ - 256 levels YCbCr BT.601.
Pixel format will be used to match the JPEG subsampling.
