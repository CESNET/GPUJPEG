Contributing to GPUJPEG
=======================
First of all, thanks for taking the time to contribute! :+1:

GPUJPEG is quite small project but there are few things that you can do
to help it to improve. Following section describe how can you help us.

Please note that we have only limited amount of resources so usually
feature requests cannot be fulfilled (depends on complexity of the task).
But if you want to take the development up, we'd willingly help you.

Table of contents
-----------------
- [Testing](#testing)
- [Reporting bugs](#reporting-bugs)
- [Development](#development)
- [Compatibility](#compatibility)
- [Contacting us](#contacting-us)

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
encoder. Even if the file is not supported you may issue a feature request.

Reporting bugs
--------------
Primarily, you can [open an issue](https://github.com/CESNET/GPUJPEG/issues) on GitHub.
GPUJPEG doesn't have its own mailing list, for a direct contact
you can, however, use a mailing list of a sister
project, [UltraGrid](https://github.com/CESNET/UltraGrid) which is
tightly personally linked.

Please follow these rules when reporting the issue:
- use the _latest_ GPUJPEG code from GIT
- try to reproduce the issue either with the _sample console application_
   or provide a _minimal working example_ demonstrating the problem.
   Use debug verbosity level (for _console application_  option `-vv`).
- _do not alter_ GPUJPEG code for the report. If unnecessary, include the
   GPUJPEG patch in the bug report.
- provide relevant information to the problem, such as:
   1. actual error description (messages, error codes etc.), ideally including a context
   1. SW/HW environment
   1. compilation options
   1. GPUJPEG parameters
   1. console output of the library/application
   1. image for which the problem occurs (test pattern can be used, see below)

Also please look into [FAQ.md](FAQ.md) to see if the the issue isn't already solved.

### Test image pattern

_gpujpegtool_ application (in API implemented through
`gpujpeg_image_load_from_file()`) has capability to generate testing
input. If applicable, it is advised to report bugs against that
pattern (point 6 above). Usage is eg.:
```
gpujpegtool -e 1920x1080.tst out.jpg  # generates Full-HD RGB image gradient
gpujpegtool -e help.tst out.jpg       # to show usage (out.jpg not written)
```

Different patterns and/or properties like pixel format or color-space avaiable.


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
- JFIF 1.01 (YCbCr or grayscale), Adobe (RGB) or SPIFF (limited-range
  YCbCr BT.601 or BT.709)
- using restart intervals

The _decoder_ shall be able to decode all following files:
- baseline Huffman JPEG
- JFIF up to 1.02, Adobe header (using RGB internally) or SPIFF. Other
  generic JIF files may be also possible to decode.

Contacting us
-------------
You can post your questions, ideas or remarks to our
[GitHub Discussions](https://github.com/CESNET/GPUJPEG/discussions).

If you prefer an e-mail contact,
you can use a mailing list of our sister project
[UltraGrid](https://github.com/CESNET/UltraGrid).

