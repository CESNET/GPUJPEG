# Frequently Asked Questions

- [What is an restart interval](#what-is-an-restart-interval)
- [Decoding is too slow](#decoding-is-too-slow)
- [Encoding different color spaces than full-range YCbCr BT.601](#encoding-different-color-spaces-than-full-range-ycbcr-bt601)
- [Optimizing encoding/decoding performance](#optimizing-encodingdecoding-performance)
- [Decoding (foreign) JPEG fails](#decoding-foreign-jpeg-fails)

## What is an restart interval
A **restart interval** and related option in UltraGrid is a way how to
increase paralelism to allow efficient Huffman encoding on GPU. It is
given in number of MCU (minmum coded units, approximately same as macroblocks)
that can be encoded independently.

Higher value result in smaller JPEG images but slower encoding. Good values
are around 10 – 8 or 16 are usually a good bet. Disabling restart intervals
(setting value to 0) causes that the Huffman encoding/decoding is done on CPU
(while the rest is still performed by GPU). On larger images, the restart
interval can be a bit larger because ther is more MCUs.

## Decoding is too slow
It can be an effect of not using restart intervals in the JPEG (eg. by using
an encoder other than GPUJPEG). You can check number of segments with followng
command:

    gpujpeg -I image.jpg

The values in the order of magnitude in hundreds or thousands mean that the number
of segments should not be a problem.

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

