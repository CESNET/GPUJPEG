BUILD_DIR ?= .
LIBDIR ?= ../../build

all: run_tests
	LD_LIBRARY_PATH=$(LIBDIR) $(BUILD_DIR)/$<

run_tests: $(wildcard *.c)
	$(CC) -g -I../.. $^ -L$(LIBDIR) -lgpujpeg -lcudart -o $(BUILD_DIR)/$@

