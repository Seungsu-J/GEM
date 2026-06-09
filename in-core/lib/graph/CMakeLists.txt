add_library(graph STATIC
  src/cpuGraph.cpp
  src/gpuGraph.cpp
)

# include_directories(${CMAKE_SOURCE_DIR}/graph/include)

target_link_libraries(graph
  PUBLIC
    utils
)

target_include_directories(graph
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  # $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

set_target_properties(graph PROPERTIES
    CUDA_ARCHITECTURES ${cuda_arch}
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

target_compile_options(graph PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
                       --generate-line-info
                       >)