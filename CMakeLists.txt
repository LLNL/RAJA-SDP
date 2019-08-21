cmake_minimum_required(VERSION 3.14)
project(STP LANGUAGES CXX CUDA)

set(CMAKE_CUDA_FLAGS "${CUDA_NVCC_FLAGS} --std=c++11 --default-stream per-thread -DEXAMPLE=${EXAMPLE}")
option(ENABLE_TESTS Off)


message(STATUS "--------- Building CAMP ----------")
add_subdirectory(camp)

message(STATUS "--------- Building CHAI ----------")
#add_subdirectory(chai)
#include_directories(chai/src)

message(STATUS "--------- Building RAJA-SDP ----------")
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/build/")
add_subdirectory(src)
