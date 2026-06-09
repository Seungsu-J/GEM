add_library(structure STATIC
  src/lattice.cpp
  src/memory_manager.cu
  src/cuckooHash.cu
  src/unifiedTrie.cu
  src/construct_kernel.cu
  src/policy.cpp
  src/host_memory_manager.cu
  )

  find_package(OpenMP REQUIRED)

  target_link_libraries(structure
    PUBLIC
      utils
      graph
      OpenMP::OpenMP_CXX
  )

target_include_directories(structure
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

set_target_properties(structure PROPERTIES
    CUDA_ARCHITECTURES ${cuda_arch}
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

target_compile_options(structure PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
                       --generate-line-info
                       --extended-lambda
                       -Xcompiler=-fopenmp
                       >)
