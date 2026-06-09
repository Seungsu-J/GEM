add_library(utils STATIC
  src/cuda_helpers.cu
  src/globals.cu
)

target_include_directories(utils
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set_target_properties(utils PROPERTIES
    CUDA_ARCHITECTURES ${cuda_arch}
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

target_compile_options(utils PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
                       --generate-line-info
                       >)

target_link_libraries(utils
  PUBLIC
    cuda
    cudart
)