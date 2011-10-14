#
# JPEG COMPRESS MAKEFILE
#

# Configure
TARGET := jpeg_compress
CFILES := main.c jpeg_common.c jpeg_encoder.c jpeg_decoder.c jpeg_table.c jpeg_huffman_encoder.c \
          jpeg_huffman_decoder.c jpeg_writer.c jpeg_reader.c
CUFILES := jpeg_preprocessor.cu
MODULES := 

# Include Common Makefile
include common.mk

main.c.o: main.c
jpeg_common.c.o: jpeg_common.c jpeg_common.h
jpeg_encoder.c.o: jpeg_encoder.c jpeg_encoder.h
jpeg_decoder.c.o: jpeg_decoder.c jpeg_decoder.h
jpeg_table.c.o: jpeg_table.c jpeg_table.h
jpeg_preprocessor.cu.o: jpeg_preprocessor.cu jpeg_preprocessor.h
jpeg_huffman_encoder.c.o: jpeg_huffman_encoder.c jpeg_huffman_encoder.h
jpeg_huffman_decoder.c.o: jpeg_huffman_decoder.c jpeg_huffman_decoder.h
jpeg_writer.c.o: jpeg_writer.c jpeg_writer.h
jpeg_reader.c.o: jpeg_reader.c jpeg_reader.h
