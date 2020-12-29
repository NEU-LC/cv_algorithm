cmake_minimum_required(VERSION 2.8.8)
project(cv_core)


include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

# Enable compile optimizations
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fsee -fomit-frame-pointer -fno-signed-zeros -fno-math-errno -funroll-loops")

# Enable debug flags (use if you want to debug in gdb)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g3 -Wall -Wuninitialized -Wmaybe-uninitialized")

option(myMVG_BUILD_SHARED "Build My CV shared libs" ON)
option(myMVG_BUILD_TYPE "Build My CV as Debug." OFF)

# Set Build Path
if(myMVG_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug")
    message("CMAKE_BUILD_TYPE:${CMAKE_BUILD_TYPE}")
else()
    set(CMAKE_BUILD_TYPE "Release")
    message("CMAKE_BUILD_TYPE:${CMAKE_BUILD_TYPE}")
endif(myMVG_BUILD_TYPE)

set(EXECUTABLE_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/bin")
set(LIBRARY_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/lib")

# Set shared or static mode
if (myMVG_BUILD_SHARED)
  set(BUILD_SHARED_LIBS ON)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
else (myMVG_BUILD_SHARED)
  set(BUILD_SHARED_LIBS OFF)
endif()

# Include libraries
find_package(Eigen3 REQUIRED)
find_package(OpenCV 3 REQUIRED)
find_package(Boost REQUIRED COMPONENTS system filesystem thread date_time)

set(SIFTGPU_INCLUDE "/home/chao/ThirdParty/SiftGPU/src/SiftGPU")
set(SIFTGPU_LIBS "/home/chao/ThirdParty/SiftGPU/bin/libsiftgpu.so")
# Include our header files
include_directories(
        src
        ${EIGEN3_INCLUDE_DIR}
        ${Boost_INCLUDE_DIRS}
        ${SIFTGPU_INCLUDE}
)
# Set link libraries used by all binaries
list(APPEND thirdparty_libraries
        ${Boost_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${SIFTGPU_LIBS}
)


add_subdirectory(src)
add_subdirectory(test)