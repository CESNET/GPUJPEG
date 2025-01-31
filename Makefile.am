AUTOMAKE_OPTIONS = subdir-objects
AUTOHEDER = true
ACLOCAL_AMFLAGS=-I m4
CLEANFILES=

SUBDIRS = . # test/memcheck test/opengl_interop

CUDA_INSTALL_PATH=@CUDA_INSTALL_PATH@
CUDA_COMPILER=@CUDA_COMPILER@

CUDA_FLAGS=@CUDA_FLAGS@ -Xcompiler -fPIC -Xcompiler @COMMON_FLAGS@ @CUDA_COMPUTE_ARGS@ 

SUFFIXES=.cu

pkgconfigdir = $(libdir)/pkgconfig
library_includedir=$(includedir)/libgpujpeg/
gpujpeg_libincludedir = $(libdir)/libgpujpeg

EXTRA_DIST = AUTHORS CONTRIBUTING.md COPYING FAQ.md INSTALL NEWS.md README.md
bin_PROGRAMS = gpujpegtool
pkgconfig_DATA = libgpujpeg.pc

library_include_HEADERS = $(srcdir)/libgpujpeg/*.h libgpujpeg/gpujpeg_version.h
nodist_gpujpeg_libinclude_HEADERS = config.h

gpujpegtool_SOURCES = src/main.c dummy.cc
gpujpegtool_CFLAGS = -std=c11 @COMMON_FLAGS@
gpujpegtool_LDFLAGS = @GPUJPEG_LDFLAGS@

dummy.cc:
	echo 'static void dummy();' > $@

# gpu jpeg library sources
libgpujpeg_SRC = src/gpujpeg_common.c \
	     	src/gpujpeg_dct_cpu.c \
	     	src/gpujpeg_decoder.c \
	     	src/gpujpeg_encoder.c \
	     	src/gpujpeg_huffman_cpu_decoder.c \
	     	src/gpujpeg_huffman_cpu_encoder.c \
	     	src/gpujpeg_reader.c \
	     	src/gpujpeg_table.c \
	     	src/gpujpeg_writer.c \
	     	src/utils/image_delegate.c \
	     	src/utils/pam.c \
	     	src/utils/y4m.c

libgpujpeg_DEP = @LIBGPUJPEG_CUDA_OBJS@

libgpujpeg_LIB = $(libgpujpeg_DEP) -lm
libgpujpeg_LD = -export-dynamic -version-info $(GPUJPEG_LIBRARY_VERSION) @GPUJPEG_LDFLAGS@ @GPUJPEG_LIBS@
libgpujpeg_CFLAGS = -std=c11 -D_POSIX_C_SOURCE=200809L -fPIC @COMMON_FLAGS@
libgpujpeg_CXXFLAGS = -fPIC @COMMON_FLAGS@
#libgpujpeg_a_LINK = g++ -fPIC
if STATIC
lib_LIBRARIES = libgpujpeg.a
libgpujpeg_a_DEPENDENCIES = $(libgpujpeg_DEP)
libgpujpeg_a_SOURCES = $(libgpujpeg_SRC)
libgpujpeg_a_LIBADD = $(libgpujpeg_DEP)
libgpujpeg_a_CFLAGS = $(libgpujpeg_CFLAGS)
libgpujpeg_a_CXXFLAGS = $(libgpujpeg_CXXFLAGS)
gpujpegtool_LDADD = $(lib_LIBRARIES) @GPUJPEG_LIBS@
else
lib_LTLIBRARIES = libgpujpeg.la
libgpujpeg_la_DEPENDENCIES = $(libgpujpeg_DEP)
libgpujpeg_la_SOURCES = $(libgpujpeg_SRC)
libgpujpeg_la_LIBADD = $(libgpujpeg_LIB)
libgpujpeg_la_LDFLAGS = $(libgpujpeg_LD)
libgpujpeg_la_CFLAGS = $(libgpujpeg_CFLAGS)
libgpujpeg_la_CXXFLAGS = $(libgpujpeg_CXXFLAGS)
gpujpegtool_LDADD = $(lib_LTLIBRARIES)
endif

check-TESTS: tests
tests:
	for testdir in `find ./test -type d` ; do \
		( cd $${testdir} ; make ) \
	done

if DARWIN
static: $(libgpujpeg_la_DEPENDENCIES) all
	[ -d build/tmp/i386 ] || mkdir -p build/tmp/i386
	[ -d build/tmp/x86_64 ] || mkdir -p build/tmp/x86_64
	[ -d build/tmp/universal ] || mkdir -p build/tmp/universal
	for arch in i386 x86_64; do \
		lipo -thin $$arch  .libs/libgpujpeg.a  -output build/tmp/$$arch/libgpujpeg.a ; \
		cd build/tmp/$$arch; ar x libgpujpeg.a; rm libgpujpeg.a; cd -; \
		for file in $(libgpujpeg_la_DEPENDENCIES); do \
			BASENAME=$$(basename $$file) ; \
			lipo -thin $$arch  $$file  -output build/tmp/$$arch/$$BASENAME ; \
			ar ru build/tmp/$$arch/libgpujpeg.a build/tmp/$$arch/$$BASENAME ; \
		done ; \
	done
	for arch in i386 x86_64; do \
		$(AR) rcu build/tmp/$$(arch)/libgpujpeg.a build/tmp/$$(arch)/* ; \
	done
	$(RM) .libs/libgpujpeg.a
	lipo -create build/tmp/i386/libgpujpeg.a build/tmp/x86_64/libgpujpeg.a -output .libs/libgpujpeg.a
else
static: $(libgpujpeg_la_DEPENDENCIES) all
	$(AR) ru .libs/libgpujpeg.a $(libgpujpeg_la_DEPENDENCIES)
	$(RANLIB) .libs/libgpujpeg.a
endif

# Pattern rule for compiling CUDA files
%.cu.o: %.cu
	$(CUDA_COMPILER) $(CUDA_FLAGS) $(DEFAULT_INCLUDES) $(INCLUDES) -c $< -o $@

build/universal/%.o: build/i386/%.cu.o build/x86_64/%.cu.o
	[ -d build/universal ] || mkdir -p build/universal
	lipo -create $? -output $@

build/i386/%.cu.o: src/%.cu
	[ -d build/i386 ] || mkdir -p build/i386
	$(CUDA_COMPILER) -m32 $(CUDA_FLAGS) $(DEFAULT_INCLUDES) $(INCLUDES) -c $< -o $@

build/x86_64/%.cu.o: src/%.cu
	[ -d build/x86_64 ] || mkdir -p build/x86_64
	$(CUDA_COMPILER) -m64 $(CUDA_FLAGS) $(DEFAULT_INCLUDES) $(INCLUDES) -c $< -o $@

clean-local:
	$(RM) src/*.cu.lo src/*.cu.o
	$(RM) dummy.cc dummy.o

cscope:
	cscope -bv $(srcdir)/src/*cu $(srcdir)/src/*c $(srcdir)/libgpujpeg/*h

universal-ctags:
	ctags-universal $$(find $(srcdir)/src $(srcdir)/test \( -name '*.c' -o -name '*.cpp' -o -name '*.cu' -o -name '*.h' -o -name '*.hpp' -o -name '*.m' -o -name '*.mm' \))

# vim: set noexpandtabs:
