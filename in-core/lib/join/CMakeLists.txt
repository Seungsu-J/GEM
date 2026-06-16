add_library(join STATIC
  src/join_trie.cu
  src/host_select.cu
  src/direct_join.cu
  src/refer_join.cu
  src/join_helpers.cu
  src/join_context.cu
  src/join_trie_kernel.cu
  src/host_select_kernel.cu
  src/direct_join_kernel.cu
  src/refer_join_kernel.cu
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