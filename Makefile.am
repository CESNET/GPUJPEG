AUTOMAKE_OPTIONS = subdir-objects
ACLOCAL_AMFLAGS=-I m4
CLEANFILES=

SUBDIRS = . # test/memcheck test/opengl_interop

CUDA_INSTALL_PATH=@CUDA_INSTALL_PATH@
NVCC=${CUDA_INSTALL_PATH}/bin/nvcc

NVCC_CFLAGS= -Xcompiler -fPIC @COMMON_CFLAGS@ \
	-gencode arch=compute_20,code=sm_20 \
	-gencode arch=compute_11,code=sm_11 \
	-gencode arch=compute_10,code=sm_10

SUFFIXES=.cu

pkgconfigdir = $(libdir)/pkgconfig
library_includedir=$(includedir)/libgpujpeg/
gpujpeg_libincludedir = $(libdir)/libgpujpeg

lib_LTLIBRARIES = libgpujpeg.la
bin_PROGRAMS = gpujpeg
pkgconfig_DATA = libgpujpeg.pc

library_include_HEADERS = libgpujpeg/*.h
nodist_gpujpeg_libinclude_HEADERS = config.h

gpujpeg_SOURCES = src/main.c 
gpujpeg_CFLAGS = -std=c99 @COMMON_CFLAGS@
gpujpeg_LDADD = libgpujpeg.la
gpujpeg_LDFLAGS = @GPUJPEG_LDFLAGS@

# gpu jpeg library sources
libgpujpeg_la_SOURCES = src/gpujpeg_common.c \
			src/gpujpeg_dct_cpu.c \
			src/gpujpeg_decoder.c \
			src/gpujpeg_encoder.c \
			src/gpujpeg_huffman_cpu_decoder.c \
			src/gpujpeg_huffman_cpu_encoder.c \
			src/gpujpeg_reader.c \
			src/gpujpeg_table.c \
			src/gpujpeg_writer.c

libgpujpeg_la_DEPENDENCIES = \
			src/gpujpeg_huffman_gpu_encoder.cu.o \
			src/gpujpeg_dct_gpu.cu.o \
			src/gpujpeg_preprocessor.cu.o \
			src/gpujpeg_huffman_gpu_decoder.cu.o

libgpujpeg_la_LIBADD = $(libgpujpeg_la_DEPENDENCIES)
libgpujpeg_la_LDFLAGS = -export-dynamic -version-info $(GPUJPEG_LIBRARY_VERSION) @GPUJPEG_LDFLAGS@
libgpujpeg_la_CFLAGS = -std=c99 @COMMON_CFLAGS@
#libgpujpeg_la_LINK = g++ -fPIC

check-TESTS: tests
tests:
	for testdir in `find ./test -type d` ; do \
		( cd $${testdir} ; make ) \
	done

# Pattern rule for compiling CUDA files
%.cu.o: %.cu
	$(NVCC) $(NVCC_CFLAGS) $(DEFAULT_INCLUDES) $(INCLUDES) -c $< -o $@

clean-local:
	rm -rf src/*.cu.lo

