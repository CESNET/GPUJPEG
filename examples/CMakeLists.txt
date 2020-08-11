cmake_minimum_required(VERSION 3.8.0 FATAL_ERROR)
project(gpujpeg-examples LANGUAGES C CXX CUDA)

set(NEEDED_CXX_FEATURES cxx_delegating_constructors cxx_nullptr)

add_executable(encode_pnm encode_pnm.c)
target_compile_features(encode_pnm PRIVATE ${NEEDED_CXX_FEATURES})
target_link_libraries(encode_pnm gpujpeg)

add_executable(encode_raw_rgb encode_raw_rgb.c)
target_compile_features(encode_raw_rgb PRIVATE ${NEEDED_CXX_FEATURES})
target_link_libraries(encode_raw_rgb gpujpeg)

add_executable(decode_to_pnm decode_to_pnm.c)
target_compile_features(decode_to_pnm PRIVATE ${NEEDED_CXX_FEATURES})
target_link_libraries(decode_to_pnm gpujpeg)

add_executable(decode_to_raw_rgb decode_to_raw_rgb.c)
target_compile_features(decode_to_raw_rgb PRIVATE ${NEEDED_CXX_FEATURES})
target_link_libraries(decode_to_raw_rgb gpujpeg)
