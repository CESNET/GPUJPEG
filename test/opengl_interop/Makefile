TARGET := opengl_interop
CUDA_INSTALL_PATH ?= /usr/local/cuda
CC := gcc
NVCC := $(CUDA_INSTALL_PATH)/bin/nvcc
CCFLAGS := -O2 -I$(CUDA_INSTALL_PATH)/include
NVCCFLAGS := -O2 -gencode arch=compute_11,code=sm_11
OBJECTS := view.c.o image.cu.o

.SUFFIXES: .cu

$(TARGET): main.c $(OBJECTS)
	$(CC) $(CCFLAGS) -lglut -lGLU -lGLEW -lpthread -L$(CUDA_INSTALL_PATH)/lib64 -lcuda -lcudart $< $(OBJECTS) -o $@
    
build:
	$(TARGET)

clean:
	rm -f *.o $(TARGET)

view.c.o: view.c view.h
	$(CC) -c $(CCFLAGS) $< -o $@
    
image.cu.o: image.cu image.h
	$(NVCC) -c $(NVCCFLAGS) $< -o $@
    