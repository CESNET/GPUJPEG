AUTOMAKE_OPTIONS = subdir-objects no-dependencies

ACLOCAL_AMFLAGS=-I m4
CLEANFILES=

SUBDIRS = . # test/memcheck test/opengl_interop

CUDA_INSTALL_PATH=@CUDA_INSTALL_PATH@
NVCC=${CUDA_INSTALL_PATH}/bin/nvcc

NVCC_CFLAGS= -Xcompiler -fPIC -Xcompiler '@COMMON_CFLAGS@' \
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

libgpujpeg_la_DEPENDENCIES = @LIBGPUJPEG_CUDA_OBJS@

libgpujpeg_la_LIBADD = $(libgpujpeg_la_DEPENDENCIES)
libgpujpeg_la_LDFLAGS = -export-dynamic -version-info $(GPUJPEG_LIBRARY_VERSION) @GPUJPEG_LDFLAGS@
libgpujpeg_la_CFLAGS = -std=c99 @COMMON_CFLAGS@
#libgpujpeg_la_LINK = g++ -fPIC

check-TESTS: tests
tests:
	for testdir in `find ./test -type d` ; do \
		( cd $${testdir} ; make ) \
	done

static: $(libgpujpeg_la_DEPENDENCIES) all
	$(AR) ru .libs/libgpujpeg.a $(libgpujpeg_la_DEPENDENCIES)
	$(RANLIB) .libs/libgpujpeg.a

# Pattern rule for compiling CUDA files
%.cu.o: %.cu
	$(NVCC) $(NVCC_CFLAGS) $(DEFAULT_INCLUDES) $(INCLUDES) -c $< -o $@

build/universal/%.a: build/i386/%.cu.o build/x86_64/%.cu.o
	[ -d build/universal ] || mkdir -p build/universal
	lipo -create $? -output $@

build/i386/%.cu.o: src/%.cu
	[ -d build/i386 ] || mkdir -p build/i386
	$(NVCC) -m32 $(NVCC_CFLAGS) $(DEFAULT_INCLUDES) $(INCLUDES) -c $< -o $@

build/x86_64/%.cu.o: src/%.cu
	[ -d build/x86_64 ] || mkdir -p build/x86_64
	$(NVCC) -m64 $(NVCC_CFLAGS) $(DEFAULT_INCLUDES) $(INCLUDES) -c $< -o $@

clean-local:
	rm -rf src/*.cu.lo src/*.cu.o
	rm -rf build
