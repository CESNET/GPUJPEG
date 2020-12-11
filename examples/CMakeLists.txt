cmake_minimum_required(VERSION 3.8.0 FATAL_ERROR)
project(gpujpeg-examples LANGUAGES C)

find_package(GPUJPEG REQUIRED)
find_package(CUDA)

add_executable(encode_pnm encode_pnm.c)
target_link_libraries(encode_pnm gpujpeg)

add_executable(encode_raw_rgb encode_raw_rgb.c)
target_link_libraries(encode_raw_rgb gpujpeg)

add_executable(decode_to_pnm decode_to_pnm.c)
target_link_libraries(decode_to_pnm gpujpeg)

add_executable(decode_to_raw_rgb decode_to_raw_rgb.c)
target_link_libraries(decode_to_raw_rgb gpujpeg)

if(CUDA_FOUND)
        add_executable(decode_to_cuda_pnm decode_to_cuda_pnm.c)
        target_link_libraries(decode_to_cuda_pnm gpujpeg ${CUDA_LIBRARIES})
endif()
