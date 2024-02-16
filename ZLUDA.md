ZLUDA support
=============

**AMD GPUs** can currently run native CUDA code using
[ZLUDA](https://github.com/vosen/ZLUDA), which includes GPUJPEG.

To enable the **CUDA** support on AMD devices, follow the instructions
described on the linked page. It is sufficient to download the
[binary build](https://github.com/vosen/ZLUDA/releases)
(at least as for the version 3) and use as described.

Below follow the platform specific notes.

Linux
-----

To run the CUDA code successfully, you will need to have:

1. AMD drivers
2. package _hip-runtime-amd_ (according to ZLUDA documentation, version
   5.7.x is currently required, not 6), which may be installed according
  [this instructions](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/)
3. _LD_LIBRARY_PATH_ environment variable containing path to ZLUDA library

Windows
-------

1. current AMD drivers
2. ZLUDA build
3. run the executable (eg. _gpujpegtool_) [with
   zluda.exe](https://github.com/vosen/ZLUDA?tab=readme-ov-file#windows)

**Note:** Windows antivirus sometimes tends to remove the zluda.exe
executable when extracted the downloaded ZIP. This can be easilly spotted
when there are just the DLLs extracted.
