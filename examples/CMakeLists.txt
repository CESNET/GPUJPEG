cmake_minimum_required(VERSION 3.10.0 FATAL_ERROR)
project(gpujpeg-examples LANGUAGES C)

find_package(GPUJPEG REQUIRED)

add_executable(encode encode.c)
target_link_libraries(encode gpujpeg)

add_executable(encode_minimal encode_minimal.c)
target_link_libraries(encode_minimal gpujpeg)

add_executable(encode_file encode_file.c)
target_link_libraries(encode_file gpujpeg)

add_executable(decode_minimal decode_minimal.c)
target_link_libraries(decode_minimal gpujpeg)

add_executable(decode_to_pnm decode_to_pnm.c)
target_link_libraries(decode_to_pnm gpujpeg)

add_executable(decode_to_raw_rgb decode_to_raw_rgb.c)
target_link_libraries(decode_to_raw_rgb gpujpeg)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        find_package(CUDAToolkit)
        add_executable(decode_to_cuda_pnm decode_to_cuda_pnm.c)
        target_link_libraries(decode_to_cuda_pnm gpujpeg)
        target_link_libraries(decode_to_cuda_pnm CUDA::cudart)
endif()
