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

Only two places need adjusting:

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

Debug and Release version of executables will be built. 
