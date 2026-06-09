add_library(filter STATIC
  ${CMAKE_CURRENT_SOURCE_DIR}/src/filter.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/src/filter_kernel.cu
)

find_package(OpenMP REQUIRED)

target_link_libraries(filter
  PUBLIC
    utils
    graph
    structure
    OpenMP::OpenMP_CXX
)

target_include_directories(filter
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)



set_target_properties(filter PROPERTIES
    CUDA_ARCHITECTURES ${cuda_arch}
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

target_compile_options(filter
  PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
      --generate-line-info
      --ptxas-options=-v
      -Xcompiler=-fopenmp
    >
)
