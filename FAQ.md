# Frequently Asked Questions

- [What is an restart interval](#what-is-an-restart-interval)
- [Decoding is too slow](#decoding-is-too-slow)
- [Encoding different color spaces than full-range YCbCr BT.601](#encoding-different-color-spaces-than-full-range-ycbcr-bt601)

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

