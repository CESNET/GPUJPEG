#
# Copyright (c) 2011, CESNET z.s.p.o
# Copyright (c) 2011, Silicon Genome, LLC.
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#

# Target executable
TARGET := gpujpeg
# C files
CFILES := main.c

# CUDA install path
CUDA_INSTALL_PATH ?= /usr/local/cuda

# Compilers
CC := gcc
LINK := g++ -fPIC

# Common flags
COMMONFLAGS += -I. -I$(CUDA_INSTALL_PATH)/include -O2
# C flags
CFLAGS += $(COMMONFLAGS) -std=c99
# Linker flags
LDFLAGS += -Llibgpujpeg -lgpujpeg

# Build
build: $(TARGET) $(TARGET).sh

# Clean
clean:
	rm -f *.o $(TARGET) $(TARGET).sh
	@cd libgpujpeg; make clean

# Lists of object files
COBJS=$(CFILES:.c=.c.o)

# Build target
$(TARGET): $(COBJS) libgpujpeg/libgpujpeg.so.build
	$(LINK) $(COBJS) $(LDFLAGS) -o $(TARGET);    
    
# Build target run script
$(TARGET).sh:
	@printf "PATH=$$" > $(TARGET).sh
	@printf "(dirname $$" >> $(TARGET).sh
	@printf "0)\n" >> $(TARGET).sh
	@printf "LD_LIBRARY_PATH=$$" >> $(TARGET).sh
	@printf "PATH/libgpujpeg $$" >> $(TARGET).sh
	@printf "PATH/gpujpeg $$" >> $(TARGET).sh
	@printf "@\n" >> $(TARGET).sh
	@chmod a+x $(TARGET).sh

# Build gpujpeg library
libgpujpeg/libgpujpeg.so.build:
	@cd libgpujpeg; make
    
# Pattern rule for compiling C files
%.c.o: %.c 
	$(CC) $(CFLAGS) -c $< -o $@

# Set file dependencies
main.c.o: main.c
