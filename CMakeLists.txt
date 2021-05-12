cmake_minimum_required(VERSION 3.8.0 FATAL_ERROR)
# change version also in configure.ac
project(gpujpeg VERSION 0.17.0 LANGUAGES C CXX CUDA)

# options
set(BUILD_OPENGL AUTO CACHE STRING "Build with OpenGL support, options are: AUTO ON OFF")
set_property(CACHE BUILD_OPENGL PROPERTY STRINGS AUTO ON OFF)
option(BUILD_SHARED_LIBS "Build using shared libraries" ON)
option(CMAKE_CUDA_ARCHITECTURES "CUDA architectures to build for (needs CMake 3.18)" OFF)
option(DISABLE_CPP "Disable C++ code (currently image delegates)" OFF)
option(HUFFMAN_GPU_CONST_TABLES "Enable Huffman GPU decoder quick const tables" OFF)
set(OPENGL_CONTEXT AUTO CACHE STRING "Use specifiedn OpenGL context creation, options are: AUTO GLX GLFW NONE")
set_property(CACHE OPENGL_CONTEXT PROPERTY STRINGS AUTO GLX GLFW NONE)

set(CPACK_PACKAGE_INSTALL_DIRECTORY "gpujpeg") # may be removed when version will be explicitly defined
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/COPYING")
set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
include(CPack)

include(CTest)

execute_process(COMMAND git rev-parse --short HEAD
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        OUTPUT_VARIABLE GIT_REV
        OUTPUT_STRIP_TRAILING_WHITESPACE)
add_definitions(-DGIT_REV=\"${GIT_REV}\")

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/libgpujpeg/gpujpeg_version.h.cmakein
               ${CMAKE_CURRENT_BINARY_DIR}/libgpujpeg/gpujpeg_version.h @ONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/libgpujpeg.pc.cmakein
               ${CMAKE_CURRENT_BINARY_DIR}/libgpujpeg.pc @ONLY)
add_definitions("-DHAVE_GPUJPEG_VERSION_H")

# Include cmake modules in "cmake" directory (FindGLFW.cmake)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")

# Common settings
include_directories(${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

set(NEEDED_COMPILER_FEATURES c_static_assert cxx_delegating_constructors cxx_nullptr)

set(COMPILED_OPTIONS)

# allow passing <PackageName>_ROOT to find_package()
if(POLICY CMP0074)
    cmake_policy(SET CMP0074 NEW)
endif()

# Find OpenGL, GLEW, GLUT and GLFW
if(POLICY CMP0072)
    cmake_policy(SET CMP0072 NEW)
endif()
find_package(OpenGL)
find_package(GLEW)
find_package(GLUT)
find_package(GLFW)
find_package(X11)
if(NOT OPENGL_FOUND)
    set(OPENGL_ERROR "OpenGL library not found")
elseif(NOT GLEW_FOUND)
    set(OPENGL_ERROR "GLEW library not found")
endif()
if(NOT BUILD_OPENGL STREQUAL "OFF" AND NOT OPENGL_ERROR)
    message(STATUS "OpenGL: ${OPENGL_LIBRARIES} (${OPENGL_INCLUDE_DIR})")
    message(STATUS "GLEW: ${GLEW_LIBRARIES} (${GLEW_INCLUDE_DIRS})")
    message(STATUS "GLUT: ${GLUT_glut_LIBRARY} (${GLUT_INCLUDE_DIR})")
    message(STATUS "GLFW: ${GLFW_LIBRARIES} (${GLFW_INCLUDE_DIR})")
    message(STATUS "X11: ${X11_LIBRARIES} (${X11_INCLUDE_DIR})")

    # OpenGL executables are enabled
    set(GPUJPEG_OPENGL_ENABLED TRUE)

    # Build GPUJPEG library with OpenGL support
    add_definitions("-DGPUJPEG_USE_OPENGL")
    set(GPUJPEG_OPENGL_LIBRARIES)
    include_directories(${OPENGL_INCLUDE_DIR} ${GLEW_INCLUDE_DIRS})
    list(APPEND GPUJPEG_OPENGL_LIBRARIES ${GLEW_LIBRARIES})
    list(APPEND GPUJPEG_OPENGL_LIBRARIES ${OPENGL_LIBRARIES})
    if(GLUT_FOUND)
        include_directories(${GLUT_INCLUDE_DIR})
        list(APPEND GPUJPEG_OPENGL_LIBRARIES ${GLUT_glut_LIBRARY})
    endif()
    if(X11_FOUND AND (OPENGL_CONTEXT STREQUAL "AUTO" OR OPENGL_CONTEXT STREQUAL "GLX"))
        include_directories(${X11_INCLUDE_DIR})
        list(APPEND GPUJPEG_OPENGL_LIBRARIES ${X11_LIBRARIES})
        add_definitions("-DGPUJPEG_USE_GLX")
        list(APPEND COMPILED_OPTIONS "OpenGL(GLX)")
    elseif(GLFW_INCLUDE_DIR AND GLFW_FOUND AND NOT OPENGL_CONTEXT STREQUAL "NONE")
        include_directories(${GLFW_INCLUDE_DIR})
        list(APPEND GPUJPEG_OPENGL_LIBRARIES ${GLFW_LIBRARIES})
        add_definitions("-DGPUJPEG_USE_GLFW")
        list(APPEND COMPILED_OPTIONS "OpenGL(GLFW)")
    else()
        list(APPEND COMPILED_OPTIONS "OpenGL(nocontext)")
    endif()
elseif(BUILD_OPENGL STREQUAL "ON")
    message(FATAL_ERROR "OpenGL not found: ${OPENGL_ERROR}")
endif()

if(DISABLE_CPP)
    add_definitions("-DDISABLE_CPP")
    set(COMPILED_OPTIONS "${COMPILED_OPTIONS} No_cpp")
endif()

if(HUFFMAN_GPU_CONST_TABLES)
    add_definitions("-DHUFFMAN_GPU_CONST_TABLES")
    set(COMPILED_OPTIONS "${COMPILED_OPTIONS Huffman_const_tables")
endif()

if(DEFINED GLEW_USE_STATIC_LIBS)
    add_definitions("-DGLEW_STATIC")
endif()

message(STATUS "Configured options: ${COMPILED_OPTIONS}")

# GPUJPEG library
file(GLOB H_FILES libgpujpeg/*.h ${CMAKE_CURRENT_BINARY_DIR}/libgpujpeg/gpujpeg_version.h)
file(GLOB C_FILES src/*.c src/*.cu)
if(NOT DISABLE_CPP)
    file(GLOB_RECURSE CPP_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
endif()
list(REMOVE_ITEM C_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/main.c")
add_library(gpujpeg ${H_FILES} ${C_FILES} ${CPP_FILES})
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_compile_features(gpujpeg PRIVATE ${NEEDED_COMPILER_FEATURES})
set_property(TARGET gpujpeg PROPERTY C_STANDARD 99)
set_target_properties(gpujpeg PROPERTIES SOVERSION 0)
SET_TARGET_PROPERTIES(
  gpujpeg
PROPERTIES
  PUBLIC_HEADER "${H_FILES}"
)
if(GPUJPEG_OPENGL_ENABLED)
    target_link_libraries(gpujpeg ${GPUJPEG_OPENGL_LIBRARIES})
endif()

# GPUJPEG commmand-line application
add_executable(gpujpeg-bin src/main.c)
target_compile_features(gpujpeg-bin PRIVATE ${NEEDED_COMPILER_FEATURES})
set_property(TARGET gpujpeg-bin PROPERTY C_STANDARD 99)
target_link_libraries(gpujpeg-bin gpujpeg)
set_target_properties(gpujpeg-bin PROPERTIES OUTPUT_NAME "gpujpeg")

install(TARGETS gpujpeg gpujpeg-bin
        EXPORT GPUJPEGTargets
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        PUBLIC_HEADER DESTINATION include/libgpujpeg
        RUNTIME DESTINATION bin)

install(FILES AUTHORS CONTRIBUTING.md COPYING FAQ.md INSTALL NEWS.md README.md
        DESTINATION doc)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/libgpujpeg.pc DESTINATION share/pkgconfig)

# When OpenGL was found, include OpenGL executables
if(GPUJPEG_OPENGL_ENABLED AND GLUT_FOUND)

    # GPUJPEG decoder to OpenGL texture
    file(GLOB FILES test/decoder_gltex/*)
    add_executable(decoder_gltex ${FILES})
    target_compile_features(decoder_gltex PRIVATE ${NEEDED_COMPILER_FEATURES})
    set_property(TARGET decoder_gltex PROPERTY C_STANDARD 99)
    target_link_libraries(decoder_gltex gpujpeg)

    # OpenGL interoperability example (currently not working)
    #file(GLOB FILES test/opengl_interop/*.c test/opengl_interop/*.h test/opengl_interop/*.cu)
    #cuda_add_executable(opengl_interop ${FILES})
    #target_link_libraries(opengl_interop src)

endif()

# CMake config
include(CMakePackageConfigHelpers)
write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/GPUJPEGConfigVersion.cmake"
    VERSION ${Upstream_VERSION}
    COMPATIBILITY AnyNewerVersion)
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/GPUJPEGConfig.cmake"
    INSTALL_DESTINATION "lib/cmake/GPUJPEG"
    NO_CHECK_REQUIRED_COMPONENTS_MACRO
    NO_SET_AND_CHECK_MACRO)
install(EXPORT GPUJPEGTargets
    DESTINATION lib/cmake/GPUJPEG)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/GPUJPEGConfig.cmake ${CMAKE_CURRENT_BINARY_DIR}/GPUJPEGConfigVersion.cmake
    DESTINATION lib/cmake/GPUJPEG)

add_custom_target(cscope cscope -bv `find ${CMAKE_SOURCE_DIR}/libgpujpeg ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/test -type f \( -name '*.c' -o -name '*.cpp' -o -name '*.cu' -o -name '*.h' -o -name '*.hpp' -o -name '*.m' -o -name '*.mm' \)`)
add_custom_target(universal-ctags ctags-universal `find ${CMAKE_SOURCE_DIR}/libgpujpeg ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/test -type f \( -name '*.c' -o -name '*.cpp' -o -name '*.cu' -o -name '*.h' -o -name '*.hpp' -o -name '*.m' -o -name '*.mm' \)`)

add_test(NAME unittests COMMAND make -C test/unit LIBDIR=${CMAKE_BINARY_DIR}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME test-colors COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/colors/run_tests.sh ${CMAKE_CURRENT_BINARY_DIR}/gpujpeg)

# vim: sw=4
