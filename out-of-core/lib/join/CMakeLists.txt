add_library(join STATIC
  src/join_trie.cu
  src/join_trie_kernel.cu
  )

target_link_libraries(join
    PUBLIC
      utils
      graph
      structure
  )

target_include_directories(join
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

set_target_properties(join PROPERTIES
    CUDA_ARCHITECTURES ${cuda_arch}
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

target_compile_options(join PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
                       --generate-line-info
                       --ptxas-options=-v
                       >)