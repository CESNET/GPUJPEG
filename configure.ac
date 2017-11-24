AC_PREREQ([2.65])
AC_INIT([libgpujpeg], [0.0.2], [martin.srom@mail.muni.cz], [libgpujpeg], [https://sourceforge.net/p/gpujpeg/])
AC_CONFIG_MACRO_DIR([m4])
AC_CONFIG_SRCDIR([src/main.c])
AC_CONFIG_AUX_DIR([.])
AM_INIT_AUTOMAKE

AC_CONFIG_HEADERS([config.h])

GPUJPEG_MAJOR_VERSION=0
GPUJPEG_MINOR_VERSION=0
GPUJPEG_MICRO_VERSION=1

GPUJPEG_LIBRARY_VERSION=0:1:0

GPUJPEG_API_VERSION=0.1
AC_SUBST(GPUJPEG_API_VERSION)
AC_SUBST(GPUJPEG_LIBRARY_VERSION)

LT_PREREQ([2.2])
LT_INIT

AC_PROG_LIBTOOL
AM_PROG_LIBTOOL

# Checks for programs.
AC_PROG_CXX
AC_PROG_CC
AC_PROG_INSTALL
AM_PROG_CC_C_O

# debug
AC_ARG_ENABLE(debug, AC_HELP_STRING([--enable-debug], [compile with debug info]) )
if test x$enable_debug = x ; then
	enable_debug=no
fi

if test x$enable_debug = xyes ; then
	COMMON_FLAGS="$COMMON_FLAGS -g -D_DEBUG -O0"
	CUDA_FLAGS="$CUDA_FLAGS -G"
else
	COMMON_FLAGS="$COMMON_FLAGS -O2"
fi

# huffmann
AC_ARG_ENABLE(constant-tables, AC_HELP_STRING([--disable-constant-tables], [disable huffman coder tables in constant memory] ) )
if test x$enable_constant_tables = x ; then
	enable_constant_tables=yes
fi

if test x$enable_constant_tables = xyes ; then
	AC_DEFINE([GPUJPEG_HUFFMAN_CODER_TABLES_IN_CONSTANT], [1], [Build with huffmann coder tables in constant memory])
fi

# opengl
AC_ARG_ENABLE(opengl, AC_HELP_STRING([--enable-opengl], [enable opengl support]) )
if test x$enable_opengl = x ; then
	enable_opengl=no
fi

# OS X universal build
AC_ARG_ENABLE(osx-universal, AC_HELP_STRING([--enable-osx-universal], [enable OS X universal build] ) )
if test x$enable_osx_universal = x ; then
	enable_osx_universal=no
fi

if test $enable_opengl = yes ; then
	AC_CHECK_HEADER(GL/glew.h, FOUND_GLEW_H=yes)
	AC_CHECK_LIB(GLEW, glewIsSupported, FOUND_GLEW_L=yes)
	AC_CHECK_LIB(GL, glBindTexture, FOUND_GL_L=yes)

	if test "$FOUND_GLEW_L" = yes -a "$FOUND_GLEW_H" = yes -a "$FOUND_GL_L" = yes; then
		AC_DEFINE([GPUJPEG_USE_OPENGL], [1], [Build with OpenGL support])
		GPUJPEG_LIBS="$GPUJPEG_LIBS -lGLEW -lGL"
	else
		enable_opengl=no
	fi
fi


# CUDA
CUDA_COMPUTE_ARGS=""
CUDA_COMPILER=""

AC_ARG_WITH(cuda-compiler,
              AS_HELP_STRING([--with-cuda-compileri=compiler], [select compiler to compile cuda code with; suported: clang, nvcc; default: nvcc]),
              [CUDA_COMPILER="$withval"],
              [CUDA_COMPILER=nvcc])

case "$CUDA_COMPILER" in
        *nvcc*)
                CUDA_COMPUTE_ARGS="-gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_35,code=compute_35"
                ;;
        *clang*)
                CUDA_COMPUTE_ARGS="--cuda-gpu-arch=sm_30 --cuda-gpu-arch=sm_35"
                CUDA_FLAGS+=" --cuda-path=$CUDA_INSTALL_PATH"
                ;;
        *)
                AC_MSG_ERROR([Unsupported cuda compiler $CUDA_COMPILER])
                ;;
esac

AC_ARG_WITH(cuda,
	AC_HELP_STRING([--with-cuda=DIR], [specify cuda root]),
	CUDA_INSTALL_PATH=$withval
)
AC_PATH_PROG(CUDA, nvcc, nvcc_not_found, [${CUDA_INSTALL_PATH}/bin${PATH_SEPARATOR}$PATH${PATH_SEPARATOR}/opt/cuda/bin${PATH_SEPARATOR}/usr/local/cuda/bin])
if test x$CUDA = xnvcc_not_found ; then
	AC_MSG_FAILURE([CUDA is required to build the gpujpeg library!])
fi

CUDA=`dirname $CUDA`
CUDA=`dirname $CUDA`
CUDA_INSTALL_PATH=$CUDA

LBITS=`getconf LONG_BIT`
if test $LBITS = 64 -a `uname -s` != 'Darwin'; then
	CUDA_INSTALL_LIB="${CUDA_INSTALL_PATH}/lib64"
else
	CUDA_INSTALL_LIB="${CUDA_INSTALL_PATH}/lib"
fi

AM_CONDITIONAL([DARWIN], [test `uname -s` = Darwin])

if test $enable_osx_universal = yes; then
	CFLAGS="$CFLAGS -arch x86_64 -arch i386"
	CXXFLAGS="$CXXFLAGS -arch x86_64 -arch i386"
	LIBGPUJPEG_CUDA_OBJS=" \
		build/universal/gpujpeg_huffman_gpu_encoder.o \
		build/universal/gpujpeg_dct_gpu.o \
		build/universal/gpujpeg_preprocessor.o \
		build/universal/gpujpeg_huffman_gpu_decoder.o"
	CUDA_FLAGS="$CUDA_FLAGS -Xcompiler -Wno-error=unused-command-line-argument-hard-error-in-future"
else
	LIBGPUJPEG_CUDA_OBJS=" \
		src/gpujpeg_huffman_gpu_encoder.cu.o \
		src/gpujpeg_dct_gpu.cu.o \
		src/gpujpeg_preprocessor.cu.o \
		src/gpujpeg_huffman_gpu_decoder.cu.o"
fi

# CUDA host compiler
AC_ARG_WITH(cuda-host-compiler, 
	AC_HELP_STRING([--with-cuda-host-compiler=BINARY], [specify compiler used for building cuda host code]))
if test "x${with_cuda_host_compiler}" != "x" ; then
	AC_CHECK_PROG([cuda_host_cc], [${with_cuda_host_compiler}],[yes])
	if test "x${cuda_host_cc}" = "xyes" ; then
		CUDA_FLAGS+=" -ccbin ${with_cuda_host_compiler}"
	else
		AC_MSG_FAILURE([Binary ${with_cuda_host_compiler} given as cuda host compiler, yet not found!])
	fi
else
	with_cuda_host_compiler="default"
fi

GPUJPEG_LDFLAGS="$GPUJPEG_LDFLAGS -L${CUDA_INSTALL_LIB} -lcudart"
COMMON_FLAGS="$COMMON_FLAGS -I. -I${CUDA_INSTALL_PATH}/include"

AC_SUBST(CUDA_EXTRA_ARCH)
AC_SUBST(CUDA_INSTALL_PATH)
AC_SUBST(CUDA_INSTALL_LIB)
AC_SUBST(COMMON_FLAGS)
AC_SUBST(GPUJPEG_LIBS)
AC_SUBST(GPUJPEG_CFLAGS)
AC_SUBST(GPUJPEG_LDFLAGS)
AC_SUBST(GPUJPEG_LIBS)
AC_SUBST(LIBGPUJPEG_CUDA_OBJS)
AC_SUBST(CUDA_FLAGS)
AC_SUBST(CUDA_COMPILER)
AC_SUBST(CUDA_COMPUTE_ARGS)

AC_CONFIG_FILES([Makefile libgpujpeg.pc test/memcheck/Makefile test/opengl_interop/Makefile ])
AC_OUTPUT

AC_MSG_RESULT([
Configuration summary:

  Target ...................... $host
  Prefix ...................... $prefix
  Debug ....................... $debug

  Constant tables.............. $enable_constant_tables
  OpenGL ...................... $enable_opengl
  Host compiler ............... $with_cuda_host_compiler
  Cuda compiler ............... $CUDA_COMPILER

  CUDA root ................... $CUDA_INSTALL_PATH


])


