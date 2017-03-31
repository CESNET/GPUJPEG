# see https://en.opensuse.org/openSUSE:Build_Service_Debian_builds#packageName.dsc
DEBTRANSFORM-TAR:	libgpujpeg-20170331.tar.bz2
DEBTRANSFORM-FILES-TAR:	debian.tar.gz
DEBTRANSFORM-SERIES:	debian-patches.series
Format: 1.0
Source: libgpujpeg
Binary: libgpujpeg
Architecture: any
Standards-Version: 3.9.6
Section: libs
Version: 20170331-1
Maintainer: 	Lukas Rucka <xrucka@fi.muni.cz>
#Build-Depends: 	debhelper (>= 7.0.50~), build-essential, make, autoconf, automake, libtool, cuda-core-7-0, cuda-command-line-tools-7-0, cuda-cudart-dev-7-0
Build-Depends: 	debhelper (>= 7.0.50~), build-essential, make, autoconf, automake, libtool, nvidia-cuda-toolkit (> 5.0)
