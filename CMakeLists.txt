cmake_minimum_required(VERSION 2.8.11)
project(gpujpeg)

find_package(CUDA)

include_directories(${CMAKE_SOURCE_DIR})
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99")

file(GLOB H_FILES libgpujpeg/*.h)
file(GLOB CPP_FILES src/*.cpp src/*.cu src/*.c)
cuda_add_executable(gpujpeg ${H_FILES} ${CPP_FILES})
