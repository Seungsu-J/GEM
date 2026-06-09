# This project uses column-major style to store intermeidate result tables. 

add_executable(GEM main.cu)

target_include_directories(GEM
  PUBLIC
  ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(GEM
  PUBLIC
    utils
    graph
    io
    structure
    plan
    filter
    join
)

set_target_properties(GEM PROPERTIES
    CUDA_ARCHITECTURES ${cuda_arch}
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

target_compile_options(GEM PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
                       --generate-line-info
                       >)