add_library(io STATIC
  src/io.cu
)

target_link_libraries(io
  PUBLIC
    utils
    graph
)

target_include_directories(io
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    # ${CMAKE_SOURCE_DIR}/lib/graph/include
)

set_target_properties(io PROPERTIES
    CUDA_ARCHITECTURES ${cuda_arch}
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

target_compile_options(io PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
                       --generate-line-info
                       >)