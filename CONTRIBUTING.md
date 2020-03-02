Contributing to GPUJPEG
=======================
First of all, thanks for taking the time to contribute! :+1:

GPUJPEG is quite small project but there are few things that you can do
to help it to improve. Following section describe how can you help us.

Please note that we have only limited amount of resources so usually
feature requests cannot be fulfilled (depends on complexity of the task).
But if you want to take the development up, we'd willingly help you.

Testing
-------
Using and thus testing GPUJPEG is the most straightward way of
contributing because you may encounter a bug when using the library.

Please keep in mind that **encoder** should be able always to encode the
image. For the **decoder**, the situation is a bit different because
it supports only subset of features, see [Compatibility](#compatibility)
below to see which JPEG files are supported. If the decoder isn't
capable to decode a supported file, it should be considered a bug. The
decoder must be always capable of decoding the file encoded by _GPUJPEG_
encoder. Even if file is not supported you may issue a feature request.

Reporting bugs
--------------
You can open a bug on GitHub. GPUJPEG doesn't have its own mailing list,
you can, however, for a direct contact use a mailing list of a sister
project, [UltraGrid](https://github.com/CESNET/UltraGrid) which is
tightly personally linked.

Development
-----------
Contributing by your own code is definitely highly appreciated. There
are several fields that may you consider:
- adding more JPEG features (primarily to decoder, optionally also to
  encoder), including these:
    - Huffman extended profile features (eg. higher bit depths)
    - Exif decoder support
    - improving support for planar pixel formats (currently missing
      preprocessor - cannot use different subsampling than pixfmt and
      color format transformations)
- fixes
- performance improvements

Please note that your changes should not change default behavior or
remove features to be accepted unless agreed by GPUJPEG developers.

Feel free to contact us for support or advice.

Compatibility
-------------
Following files are produced by the encoder:
- baseline Huffman JPEG (with default Huffman tables)
- JFIF 1.01 (YCbCr or grayscale) or Adobe (RGB)
- using restart intervals

The _decoder_ shall be able to decode all following files:
- baseline Huffman JPEG
- JFIF up to 1.02 or using Adobe header (using RGB internally). Other
  generic JIF files may be also possible to decode.

Contacting us
-------------
You can contact us through our Matrix room
[@gpujpeg:matrix.org](https://matrix.to/#/!ppSneXxiHfvznPxTTN:matrix.org?via=matrix.org).

If you prefer an e-mail contact for your questions, remarks or ideas,
you can use a mailing list of our sister project
[UltraGrid](https://github.com/CESNET/UltraGrid).

