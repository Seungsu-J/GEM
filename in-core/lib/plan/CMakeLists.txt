add_library(plan STATIC
  src/relax.cpp
)

target_link_libraries(plan
  PUBLIC
    utils
    graph
)

target_include_directories(plan
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set_target_properties(plan PROPERTIES
    CUDA_ARCHITECTURES ${cuda_arch}
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

target_compile_options(plan PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
                       --generate-line-info
                       >)