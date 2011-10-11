#
# JPEG COMPRESS MAKEFILE
#

# Configure
TARGET := jpeg_compress
CFILES := main.c jpeg_common.c jpeg_encoder.c jpeg_table.c
MODULES := 

# Include Common Makefile
include common.mk

main.c.o: main.c
jpeg_common.c.o: jpeg_common.c jpeg_common.h
jpeg_encoder.c.o: jpeg_encoder.c jpeg_encoder.h
jpeg_table.c.o: jpeg_table.c jpeg_table.h

