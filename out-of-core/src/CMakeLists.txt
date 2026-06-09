# This project uses column-major style to store intermeidate result tables. 

add_executable(FleSSM main.cu)

target_include_directories(FleSSM
  PUBLIC
  ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(FleSSM
  PUBLIC
    utils
    graph
    io
    structure
    plan
    filter
    join
)

set_target_properties(FleSSM PROPERTIES
    CUDA_ARCHITECTURES ${cuda_arch}
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

target_compile_options(FleSSM PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
                       --generate-line-info
                       >)