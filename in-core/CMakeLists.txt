cmake_minimum_required(VERSION 3.10)
project(GEM LANGUAGES CXX CUDA)
# set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_CXX_COMPILER g++)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED True)
# set(CMAKE_BUILD_TYPE Debug)
# set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.4/bin/nvcc)
set(CMAKE_CUDA_LIBRsARY_ARCHITECTURES 86) # A6000
# set(CMAKE_CUDA_LIBRARY_ARCHITECTURES 75) # 2080Ti
# set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -lineinfo")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --maxrregcount=32")

set(cuda_arch "86")
# set(cuda_arch "75")

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(/usr/local/cuda-12.4/include)
link_directories(/usr/local/cuda-12.4/lib64)

add_subdirectory(./lib/utils)
add_subdirectory(./lib/graph)
add_subdirectory(./lib/io)
add_subdirectory(./lib/structure)
add_subdirectory(./lib/filter)
add_subdirectory(./lib/plan)
add_subdirectory(./lib/join)

add_subdirectory(./src)

add_compile_options(-w)  # Suppress all warnings