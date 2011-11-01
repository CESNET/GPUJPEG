#
# COMMON MAKEFILE
#
# Included from all other makefiles
#
# Mandatory configuration:
#   TARGET = executable_name or module_name.a
#
# Optional configuration:
#   CFILES = file1.c file2.c ...
#   CXXFILES = file1.cpp file2.cpp ...
#   CUFILES = file1.cu file2.cu ...
#   MODULES = module1.a module2.a ..
#

# Configure
CUDA_INSTALL_PATH ?= /usr/local/cuda
CUDA_DISABLED = 0
DEBUG = 0
DEBUG_CUDA = 0

# DEBUG_CUDA implicates DEBUG
ifeq ($(DEBUG_CUDA),1)
    DEBUG = 1
endif

# Includes
INCLUDES := -I. -I$(CUDA_INSTALL_PATH)/include

# Common flags
COMMONFLAGS += $(INCLUDES) 
CFLAGS += $(COMMONFLAGS) -std=c99 
CXXFLAGS += $(COMMONFLAGS)
NVCCFLAGS += $(COMMONFLAGS) -gencode arch=compute_20,code=sm_20
#NVCCFLAGS += --ptxas-options="-v" -keep --opencc-options -LIST:source=on
#NVCCFLAGS += --maxrregcount 16
LDFLAGS += 

# Do 32bit vs. 64bit setup
LBITS := $(shell getconf LONG_BIT)
ifeq ($(LBITS),64)
    # 64bit
    LDFLAGS += -L$(CUDA_INSTALL_PATH)/lib64 -lcudart -lnpp
else
    # 32bit
    LDFLAGS += -L$(CUDA_INSTALL_PATH)/lib -lcudart
endif

# Warning flags (from cuda common.mk)
CXXWARN_FLAGS := \
	-W -Wall \
	-Wimplicit \
	-Wswitch \
	-Wformat \
	-Wchar-subscripts \
	-Wparentheses \
	-Wmultichar \
	-Wtrigraphs \
	-Wpointer-arith \
	-Wcast-align \
	-Wreturn-type \
	-Wno-unused-function \
	$(SPACE)
CWARN_FLAGS := $(CXXWARN_FLAGS) \
	-Wstrict-prototypes \
	-Wmissing-prototypes \
	-Wmissing-declarations \
	-Wnested-externs \
	-Wmain \
CFLAGS += $(CWARN_FLAGS)
CXXFLAGS += $(CXXWARN_FLAGS)
OPTFLAGS += -O2

# Debug/Release flags
ifeq ($(DEBUG),1)
    COMMONFLAGS += -g 
    CFLAGS      += -D_DEBUG
    CXXFLAGS    += -D_DEBUG
    NVCCFLAGS   += -D_DEBUG
    
else 
    COMMONFLAGS += $(OPTFLAGS)
    CFLAGS      += -fno-strict-aliasing
    CXXFLAGS    += -fno-strict-aliasing
    NVCCFLAGS   += --compiler-options -fno-strict-aliasing
endif

# Compilers
CC := gcc
CXX := g++
LINK := g++ -fPIC
NVCC := $(CUDA_INSTALL_PATH)/bin/nvcc

ifeq ($(CUDA_DISABLED),1)
    CFLAGS   += -D_CUDA_DISABLED
    CXXFLAGS += -D_CUDA_DISABLED
endif

# Generate Object Files
COBJS=$(CFILES:.c=.c.o)
CXXOBJS=$(CXXFILES:.cpp=.cpp.o)
CUOBJS=$(CUFILES:.cu=.cu.o)

# Set suffix for CUDA
.SUFFIXES: .cu

# Pattern rule for compiling C files
%.c.o: %.c 
	$(CC) $(CFLAGS) -c $< -o $@

# Pattern rule for compiling CPP files
%.cpp.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Pattern rule for compiling CUDA files
%.cu.o: %.cu
	if [ "$(CUDA_DISABLED)" = "1" ]; then \
	    touch _tmp.c; $(CC) $(CFLAGS) -c _tmp.c -o $@; rm _tmp.c; \
	else \
	    $(NVCC) $(NVCCFLAGS) -c $< -o $@; \
	fi

# Pattern rule for compiling modules
%.a.build:
	@cd `dirname $@` ; make

# Pattern rule for cleaning modules
%.a.clean:
	@cd `dirname $@` ; make clean

# Build modules and then target
build: $(MODULES:.a=.a.build) $(TARGET)

# Build target
ifeq ($(findstring .a,$(TARGET)),)
    # Build target as executable
    $(TARGET): $(COBJS) $(CXXOBJS) $(CUOBJS) $(MODULES)
		$(LINK) $(COBJS) $(CXXOBJS) $(CUOBJS) $(MODULES) $(LDFLAGS) -o $(TARGET);    
else 
    # Build target as static library
    ifeq ($(findstring .a,$(MODULES)),)
        # Build target as static library without modules
        $(TARGET): $(COBJS) $(CXXOBJS) $(CUOBJS)
			ar rcs $(TARGET) $(CXXOBJS) $(COBJS) $(CUOBJS);    
    else 
        # Build target as static library with modules 
        # (following lines extracts *.o from *.a and take care of same names and put everything to target static library)
        $(TARGET): $(COBJS) $(CXXOBJS) $(CUOBJS) $(MODULES)
			mkdir _tmp; cp $(MODULES) _tmp/;
			cd _tmp; for f in *.a; do mkdir _tmp; cd _tmp; ar x ../$${f}; for o in *.o; do cp $$o ../$${f}.$${o}; done; cd ..; rm -R _tmp; done;
			ar rcs $(TARGET) $(COBJS) $(CXXOBJS) $(CUOBJS) _tmp/*.o;
			rm -R _tmp;
    endif
endif

# Clean
clean: $(MODULES:.a=.a.clean)
	rm -f *.o $(TARGET)
	rm -f *.i *.ii 
	rm -f *.cudafe1.c *.cudafe1.cpp *.cudafe1.gpu *.cudafe1.stub.c
	rm -f *.cudafe2.c *.cudafe2.gpu *.cudafe2.stub.c
	rm -f *.fatbin *.fatbin.c *.ptx *.hash *.cubin *.cu.cpp
