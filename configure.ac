AC_PREREQ([2.65])
# change version also in CMakeLists.txt
AC_INIT([libgpujpeg],[0.20.2],[https://github.com/CESNET/GPUJPEG/issues],[libgpujpeg],[https://github.com/CESNET/GPUJPEG])
AC_CONFIG_MACRO_DIR([m4])
AC_CONFIG_SRCDIR([src/main.c])
AC_CONFIG_AUX_DIR([.])
AM_INIT_AUTOMAKE([foreign])

AC_CONFIG_HEADERS([config.h libgpujpeg/gpujpeg_version.h])
COMMON_FLAGS="${COMMON_FLAGS:+$COMMON_FLAGS }-DHAVE_GPUJPEG_VERSION_H"

AC_DEFINE([GPUJPEG_VERSION_MAJOR], regexp(AC_PACKAGE_VERSION, [\([^.]*\)], [\1]), [GPUJPEG major version])
AC_DEFINE([GPUJPEG_VERSION_MINOR], regexp(AC_PACKAGE_VERSION, [[^.]*\.\([^.]*\)], [\1]), [GPUJPEG minor version])
AC_DEFINE([GPUJPEG_VERSION_PATCH], regexp(AC_PACKAGE_VERSION, [[^.]*\.[^.]*\.\([^.]*\)], [\1]), [GPUJPEG patch version])

GPUJPEG_LIBRARY_VERSION=0:1:0

AC_SUBST(GPUJPEG_LIBRARY_VERSION)

LT_PREREQ([2.2])
LT_INIT([disable-static])

# Checks for programs.
AC_PROG_CXX
AC_PROG_CC
AC_PROG_INSTALL
AM_PROG_CC_C_O

# debug
AC_ARG_ENABLE(debug, AS_HELP_STRING([--enable-debug], [compile with debug info]) )
if test x$enable_debug = x ; then
	enable_debug=no
fi

AM_CONDITIONAL([STATIC], [test x$enable_static = xyes])
if test "$enable_static" = yes && test "$enable_shared" = yes; then
	AC_MSG_ERROR([Currently only one of --enable-static/--enable-shared can be specified at once.])
fi

AC_ARG_VAR([CUDA_FLAGS], [Flags passed to nvcc])

if test x$enable_debug = xyes ; then
	COMMON_FLAGS="$COMMON_FLAGS -g -D_DEBUG -O0"
	CUDA_FLAGS="$CUDA_FLAGS -G"
else
	COMMON_FLAGS="$COMMON_FLAGS -O2"
fi

# huffmann
AC_ARG_ENABLE(huffman-gpu-const-tables, AS_HELP_STRING([--enable-huffman-gpu-const-tables],[enable Huffman GPU decoder tables in constant memory]) )
if test x$enable_huffman_gpu_const_tables = xyes ; then
	AC_DEFINE([HUFFMAN_GPU_CONST_TABLES], [1], [Build with huffmann coder tables in constant memory])
else
        enable_huffman_gpu_const_tables=no
fi

# opengl
AC_ARG_ENABLE(opengl, AS_HELP_STRING([--enable-opengl],[enable opengl support]) )
if test x$enable_opengl = x ; then
	enable_opengl=auto
fi

# OS X universal build
AC_ARG_ENABLE(osx-universal, AS_HELP_STRING([--enable-osx-universal],[enable OS X universal build]) )
if test x$enable_osx_universal = x ; then
	enable_osx_universal=no
fi

opengl=no
opengl_context=none
if test $enable_opengl != no ; then
	AC_CHECK_HEADER(GL/glew.h, FOUND_GLEW_H=yes)
	AC_CHECK_LIB(GLEW, glewIsSupported, FOUND_GLEW_L=yes)
	AC_CHECK_LIB(GL, glBindTexture, FOUND_GL_L=yes)
	AC_CHECK_LIB(X11, XCreateWindow, FOUND_X11_L=yes)
	AC_CHECK_HEADER(GL/glx.h, FOUND_GLX_H=yes)
	AC_CHECK_LIB(glfw, glfwCreateWindow, FOUND_GLFW_L=yes)
	AC_CHECK_HEADER(GLFW/glfw3.h, FOUND_GLFW_H=yes)

	opengl=yes
	if test "$FOUND_GLEW_L" != yes -o "$FOUND_GLEW_H" != yes -o "$FOUND_GL_L" != yes; then
		opengl=no
	elif test "$FOUND_X11_L" = yes -a "$FOUND_GLX_H" = yes; then
		AC_DEFINE([GPUJPEG_USE_GLX], [1], [Use GLX for OpenGL context creation])
		GPUJPEG_LIBS="${GPUJPEG_LIBS:+$GPUJPEG_LIBS }-lX11"
		opengl_context=glx
	elif test "$FOUND_GLFW_L" = yes -a "$FOUND_GLFW_H" = yes; then
		AC_DEFINE([GPUJPEG_USE_GLFW], [1], [Use GLFW for OpenGL context creation])
		GPUJPEG_LIBS="${GPUJPEG_LIBS:+$GPUJPEG_LIBS }-lGLFW"
		opengl_context=glfw
	fi

	if test $opengl = yes; then
		AC_DEFINE([GPUJPEG_USE_OPENGL], [1], [Build with OpenGL support])
		GPUJPEG_LIBS="${GPUJPEG_LIBS:+$GPUJPEG_LIBS }-lGLEW -lGL"
	fi

	if test $enable_opengl = yes -a $opengl = no; then
                AC_MSG_ERROR([OpenGL required but dependencies not found!])
	fi
fi


# CUDA
CUDA_COMPUTE_ARGS=""
CUDA_COMPILER=""

AC_ARG_WITH(cuda,
	AS_HELP_STRING([--with-cuda=DIR],[specify cuda root]),
	CUDA_INSTALL_PATH=$withval
)
AC_PATH_PROG(CUDA, nvcc, nvcc_not_found, [${CUDA_INSTALL_PATH}/bin${PATH_SEPARATOR}$PATH${PATH_SEPARATOR}/opt/cuda/bin${PATH_SEPARATOR}/usr/local/cuda/bin])
if test "x$CUDA" = xnvcc_not_found ; then
	AC_MSG_FAILURE([CUDA is required to build the gpujpeg library!])
fi
CUDA_COMPILER="$CUDA"

CUDA=`dirname "$CUDA"`
CUDA=`dirname "$CUDA"`
CUDA_INSTALL_PATH=$CUDA

LBITS=`getconf LONG_BIT`
if test $LBITS = 64 -a `uname -s` != 'Darwin'; then
	CUDA_INSTALL_LIB="${CUDA_INSTALL_PATH}/lib64"
elif test $LBITS = 64 -a `uname -o` = 'Msys'; then
	CUDA_INSTALL_LIB="${CUDA_INSTALL_PATH}/lib/x64"
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
		build/universal/gpujpeg_postprocessor.o \
		build/universal/gpujpeg_preprocessor.o \
		build/universal/gpujpeg_huffman_gpu_decoder.o"
	CUDA_FLAGS="$CUDA_FLAGS -Xcompiler -Wno-error=unused-command-line-argument-hard-error-in-future"
else
	LIBGPUJPEG_CUDA_OBJS=" \
		src/gpujpeg_huffman_gpu_encoder.cu.o \
		src/gpujpeg_dct_gpu.cu.o \
		src/gpujpeg_postprocessor.cu.o \
		src/gpujpeg_preprocessor.cu.o \
		src/gpujpeg_huffman_gpu_decoder.cu.o"
fi

AC_ARG_WITH(cuda-compiler,
              AS_HELP_STRING([--with-cuda-compiler=compiler], [select compiler to compile cuda code with; suported: clang, nvcc; default: nvcc autodetected from cuda]),
              [CUDA_COMPILER="$withval"])

case "$CUDA_COMPILER" in
        *nvcc*)
                ;;
        *clang*)
                CUDA_COMPUTE_ARGS="--cuda-gpu-arch=sm_30 --cuda-gpu-arch=sm_35"
                CUDA_FLAGS+=" --cuda-path=$CUDA_INSTALL_PATH"
                ;;
        *)
                AC_MSG_ERROR([Unsupported cuda compiler $CUDA_COMPILER])
                ;;
esac

# escape paths to remove potentially harmful spaces in paths on MSW
if test `uname -o` = 'Msys'; then
	CUDA_INSTALL_PATH="\"$CUDA_INSTALL_PATH\""
	CUDA_INSTALL_LIB="\"$CUDA_INSTALL_LIB\""
	CUDA_COMPILER="\"$CUDA_COMPILER\""
fi

# CUDA host compiler
AC_ARG_WITH(cuda-host-compiler, 
	AS_HELP_STRING([--with-cuda-host-compiler=BINARY],[specify compiler used for building cuda host code]))
if test "x${with_cuda_host_compiler}" != "x" ; then
	AC_CHECK_PROG([cuda_host_cc], [${with_cuda_host_compiler}],[yes])
	if test "x${cuda_host_cc}" = "xyes" -o -x "${with_cuda_host_compiler}"; then
		CUDA_FLAGS+=" -ccbin ${with_cuda_host_compiler}"
	else
		AC_MSG_FAILURE([Binary ${with_cuda_host_compiler} given as cuda host compiler, yet not found!])
	fi
else
	with_cuda_host_compiler="default"
fi

GPUJPEG_LDFLAGS="$GPUJPEG_LDFLAGS -L${CUDA_INSTALL_LIB}"
GPUJPEG_LIBS="$GPUJPEG_LIBS -lcudart"
COMMON_FLAGS="$COMMON_FLAGS -I. -I${CUDA_INSTALL_PATH}/include"

AC_SUBST(CUDA_EXTRA_ARCH)
AC_SUBST(CUDA_INSTALL_PATH)
AC_SUBST(CUDA_INSTALL_LIB)
AC_SUBST(COMMON_FLAGS)
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
  Debug ....................... $enable_debug
  Shared library .............. $enable_shared
  Static library .............. $enable_static

  Huffman const tables ........ $enable_huffman_gpu_const_tables
  OpenGL ...................... $opengl ($opengl_context)
  Host compiler ............... $with_cuda_host_compiler
  Cuda compiler ............... $CUDA_COMPILER

  CUDA root ................... $CUDA_INSTALL_PATH


])

# vi: set noexpandtab:
