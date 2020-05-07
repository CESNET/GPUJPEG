# Frequently Asked Questions

- [What is an restart interval](#what-is-an-restart-interval)
- [Decoding is too slow](#decoding-is-too-slow)

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


