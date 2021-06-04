LIBDIR ?= ../../build

all: run_tests
	LD_LIBRARY_PATH=$(LIBDIR) ./$<

run_tests: run_tests.c
	$(CC) -g -I../.. $< -L$(LIBDIR) -lgpujpeg -lcudart -o $@

