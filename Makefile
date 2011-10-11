#
# JPEG COMPRESS MAKEFILE
#

# Configure
TARGET := jpeg_compress
CFILES := main.c jpeg_common.c jpeg_encoder.c jpeg_table.c
CUFILES := jpeg_preprocessor.cu
MODULES := 

# Include Common Makefile
include common.mk

main.c.o: main.c
jpeg_common.c.o: jpeg_common.c jpeg_common.h
jpeg_encoder.c.o: jpeg_encoder.c jpeg_encoder.h
jpeg_table.c.o: jpeg_table.c jpeg_table.h
jpeg_preprocessor.cu.o: jpeg_preprocessor.cu jpeg_preprocessor.h
