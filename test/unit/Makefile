BUILD_DIR ?= .
LIBDIR ?= ../../build

all: run_tests
	cd $(BUILD_DIR);\
	LD_LIBRARY_PATH=$${LD_LIBRARY_PATH:+$$LD_LIBRARY_PATH:}$(LIBDIR) ./$<

run_tests: $(wildcard *.c)
	$(CC) -g -I../.. $^ -L$(LIBDIR) -lgpujpeg -lcudart -o $(BUILD_DIR)/$@

