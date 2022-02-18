# Frequently Asked Questions

- [What is an restart interval](#what-is-an-restart-interval)
- [Decoding is too slow](#decoding-is-too-slow)
- [Encoding different color spaces than full-range YCbCr BT.601](#encoding-different-color-spaces-than-full-range-ycbcr-bt601)
- [Optimizing encoding/decoding performance](#optimizing-encodingdecoding-performance)
- [Decoding (foreign) JPEG fails](#decoding-foreign-jpeg-fails)
- [Encoding/decoding alpha channel](#encodingdecoding-alpha-channel)
   - [Alpha support in command-line application](#alpha-support-in-command-line-application)
   - [API for alpha](#api-for-alpha)

## What is an restart interval
A **restart interval** and related option in UltraGrid is a way how to
increase paralelism to allow efficient Huffman encoding and decoding on GPU.
It is given by the number of MCU (minmum coded units, approximately same as
macroblocks) that can be encoded or decoded independently.

For the _encoder_ the restart interval is given as a option (enabled by default
with a runtime-determined value). Higher value result in smaller JPEG images
but slower encoding. Good values are around 10 – 8 or 16 are usually a good
bet. Disabling restart intervals (setting value to 0) causes that the Huffman
encoding/decoding is done on CPU (while the rest is still performed by GPU). On
larger images, the restart interval can be a bit larger because ther is more
MCUs.

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

which shows duration of individual decoding steps.

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
Encoding alpha is quite simple, as indicated above, just set the pixel format `GPUJPEG_444_U8_P012A`
as `gpujpeg_image_parameters::pixel_format` and `gpujpeg_image_parameters` to **4**.

#### Decode
Select output pixel format either `GPUJPEG_444_U8_P012A` or `GPUJPEG_NONE` (autodetect).
