#ifndef JOIN_JOIN_HELPERS_H
#define JOIN_JOIN_HELPERS_H

#include "unifiedTrie.cuh"
#include "cpuGraph.h"
#include "gpuGraph.h"
#include "memory_manager.cuh"
#include "lattice.h"
#include "cuda_helpers.cuh"
#include "join_context.cuh"

#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/system/cuda/execution_policy.h>

#include "join_trie_kernel.cuh"

// extern uint32_t *d_dup_cols; // allocate once for reuse
// extern uint32_t num_dup_cols;
// extern uint32_t *h_dup_cols;

// // data_ptr, parent_ptr
// extern vtype **d_data_arrays;
// extern uint64_t **d_parent_arrays;

// // skip flag
// extern bool skip_final_materialization;

// extern cudaStream_t join_stream;

namespace
{
  struct PopcToCount
  {
    __host__ __device__ uint32_t operator()(uint32_t value) const
    {
#ifdef __CUDA_ARCH__
      return static_cast<uint32_t>(__popc(value));
#else
      return static_cast<uint32_t>(__builtin_popcount(value));
#endif
    }
  };
} // namespace

uint32_t ceilPowerOf2(uint32_t x);
uint32_t findClosestPowerOf2_Bitwise(uint32_t x);

#endif