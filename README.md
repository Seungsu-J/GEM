# GEM
repo for ICDE research paper "GEM: GPU-Accelerated Edge-Centric Subgraph Matching on Large Graphs", under review

GEM supports both in-memory and out-of-core execution. For simplicity and convenient experimental study, codes for two modes are organized separately, with only the difference on filtering phase.

Compilation and execution scripts are the same for both.

# 1. Environment Requirements

- nvcc 12.0 or higher
- gcc 11.3 or higher
- C++ 17 or higher
- OpemMP (for out-of-core execution)

# 2. Compilation

Before the compilation, please set the proper compute capability number according to your GPU device. If not sure, please refer to: [CUDA GPU Compute Capability](https://developer.nvidia.com/cuda/gpus). 

Only two places need adjusting in the root `CMakeLists.txt` (`in-core/CMakeLists.txt` and `out-of-core/CMakeLists.txt`):

```CMake
set(CMAKE_CUDA_LIBRARY_ARCHITECTURES 86) # A6000
set(cuda_arch "86")
```

One can simply compile GEM by running:

```bash
rm -rf build
mkdir build
bash build.sh
```

Debug and Release version of executables will be built, and they will be in `./build/debug/GEM` and `./build/release/GEM` respectively.

# 3. Execution

One can run `GEM` by the command:

```bash
GEM -q <path/to/query_graph> -d <path/to/data_graph> --gpu <gpu_card_id> >> output_file.txt
```

or

```bash
GEM -q <path/to/query_graph_folder> -d <path/to/data_graph> --gpu <gpu_card_id> >> output_file.txt
```

`path/to/query_graph` or `path/to_query_graph_folder` and `path/to/data_graph` is required. 

`gpu_card_id` is 0 by default. 

If no output_file.txt is set, GEM uses standard output (terminal). 

In addition, one can simply conduct the experiments using the pre-written script `script.sh` with adjustments. 

# 4. Data Format

Both data graph file and query graph file use the same format in the file as below.

```
t #v #e
v <vid> <vlabel> <vdegree>
...
v <vid> <vlabel> <vdegree>
e <vid_src> <vid_dst>
...
e <vid_src> <vid_dst>
```

Note:
- #e is the number of undirected edges.
- vid grows from 0, contiguously
- vdegree is the degree of undirected edges
- vid_src should be less than vid_dst.
- the beginning `t`, `v`, `e` is a character for marking.

# 5. Report
