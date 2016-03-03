cmake_minimum_required(VERSION 2.8.11)
project(gpujpeg)

# Include cmake modules in "cmake" directory (FindGLEW.cmake and FindGLUT.cmake)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")

# Common settings
include_directories(${CMAKE_SOURCE_DIR})
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99")

# Find CUDA
find_package(CUDA)
message(STATUS "CUDA: ${CUDA_VERSION}")

# Find OpenGL, GLEW and GLUT
find_package(OpenGL)
find_package(GLEW)
find_package(GLUT)
if(NOT OPENGL_FOUND)
    set(OPENGL_ERROR "OpenGL library not found")
elseif(NOT GLEW_FOUND)
    set(OPENGL_ERROR "GLEW library not found")
elseif(NOT GLUT_FOUND)
    set(OPENGL_ERROR "GLUT not found")
elseif(NOT GLEW_INCLUDE_DIR)
    set(OPENGL_ERROR "GLEW header files not found")
elseif(NOT GLUT_INCLUDE_DIR)
    set(OPENGL_ERROR "GLUT header files not found")
elseif(NOT OPENGL_INCLUDE_DIR)
    # On Windows the FindOpenGL.cmake doesn't work right for OPENGL_INCLUDE_DIR and thus reuse GLEW_INCLUDE_DIR
    if(MSVC)
        set(OPENGL_INCLUDE_DIR "${GLEW_INCLUDE_DIR}")
    else()
        set(OPENGL_ERROR "OpenGL header files not found")
    endif()
endif()
if(MSVC AND NOT GLEW_DLL)
    set(OPENGL_ERROR "GLEW DLL not found")
elseif(MSVC AND NOT GLUT_DLL)
    set(OPENGL_ERROR "GLUT DLL not found")
endif()
if(NOT OPENGL_ERROR)
    message(STATUS "OpenGL: ${OPENGL_gl_LIBRARY} (${OPENGL_INCLUDE_DIR})")
    message(STATUS "GLU: ${OPENGL_glu_LIBRARY}")
    message(STATUS "GLEW: ${GLEW_LIBRARIES} (${GLEW_INCLUDE_DIRS})")
    message(STATUS "GLUT: ${GLUT_glut_LIBRARY} (${GLUT_INCLUDE_DIR})")

    # OpenGL executables are enabled
    set(GPUJPEG_OPENGL_ENABLED TRUE)

    # Build GPUJPEG library with OpenGL support
    add_definitions("-DGPUJPEG_USE_OPENGL")
    include_directories("${OPENGL_INCLUDE_DIR}" "${GLEW_INCLUDE_DIRS}" "${GLUT_INCLUDE_DIR}")

    # Prepare list of OpenGL libraries for linking
    set(GPUJPEG_OPENGL_LIBRARIES)
    list(APPEND GPUJPEG_OPENGL_LIBRARIES ${GLEW_LIBRARIES})
    list(APPEND GPUJPEG_OPENGL_LIBRARIES ${GLUT_glut_LIBRARY})
    list(APPEND GPUJPEG_OPENGL_LIBRARIES ${OPENGL_LIBRARIES})
endif()

# GPUJPEG library
file(GLOB H_FILES libgpujpeg/*.h)
file(GLOB CPP_FILES src/*.cpp src/*.cu)
cuda_add_library(gpujpeg SHARED ${H_FILES} ${CPP_FILES})
if(GPUJPEG_OPENGL_ENABLED)
    target_link_libraries(gpujpeg ${GPUJPEG_OPENGL_LIBRARIES})
endif()

# GPUJPEG tester application
file(GLOB C_FILES src/main.c)
cuda_add_executable(tester ${C_FILES})
target_link_libraries(tester gpujpeg)

# When OpenGL was found, include OpenGL executables
if(GPUJPEG_OPENGL_ENABLED)

    # GPUJPEG decoder to OpenGL texture
    file(GLOB C_FILES test/decoder_gltex/main.c)
    cuda_add_executable(decoder_gltex ${C_FILES})
    target_link_libraries(decoder_gltex gpujpeg)

    # OpenGL interoperability example (currently not working)
    #file(GLOB FILES test/opengl_interop/*.c test/opengl_interop/*.h test/opengl_interop/*.cu)
    #cuda_add_executable(opengl_interop ${FILES})
    #target_link_libraries(opengl_interop src)

endif()
