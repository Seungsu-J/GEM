#include <cooperative_groups.h>

#include <cub/cub.cuh>

#include "cuda_helpers.cuh"
#include "globals.h"
#include "join_trie.cuh"
#include "join_trie_kernel.cuh"
#include "unifiedTrie.cuh"

// helper kernels.
__global__ void
gather_counts_kernel(
    uint32_t *d_count_src, uint64_t *d_indices, uint64_t num_indices, uint32_t *d_count_dst)
{
  const uint32_t tid = threadIdx.x;
  const uint32_t bid = blockIdx.x;
  const uint64_t idx = static_cast<uint64_t>(tid) + static_cast<uint64_t>(bid) * blockDim.x;
  const uint64_t grid_size = static_cast<uint64_t>(blockDim.x) * gridDim.x;

  for (uint64_t i = idx; i < num_indices; i += grid_size)
  {
    uint64_t src_idx = fast_load(d_indices + i);
    uint32_t count_value = fast_load(d_count_src + src_idx);
    d_count_dst[i] = count_value;
  }
  __syncwarp();
}

__global__ void
print_trie_kernel(
    // trie_inter
    vtype **vertex_arrays_, uint64_t **parent_pointers_,
    uint64_t num_rows, uint32_t num_cols,
    // for debug
    uint32_t *output_buffer, uint64_t *output_count)
{
  const uint32_t tid = threadIdx.x;
  const uint32_t bid = blockIdx.x;
  const uint32_t idx = tid + bid * blockDim.x;
  const uint32_t wid = tid >> 5;
  const uint32_t lid = tid & 31;
  const uint32_t wid_g = idx >> 5;
  const uint32_t grid_size = blockDim.x * gridDim.x;

  uint64_t my_row_idx = idx;
  uint32_t my_row_data[MAX_VQ];
  while (my_row_idx < num_rows)
  {
    uint64_t current_pos = my_row_idx;
    for (int col = num_cols - 1; col >= 1; --col)
    {
      my_row_data[col] = fast_load(vertex_arrays_[col] + current_pos);
      current_pos = fast_load(parent_pointers_[col] + current_pos);
    }
    my_row_data[0] = fast_load(vertex_arrays_[0] + current_pos); // loop peeling.

    // Write to output buffer
    unsigned long long out_base = atomicAdd((unsigned long long *)output_count, (unsigned long long)num_cols);
    for (uint32_t col = 0; col < num_cols; ++col)
    {
      output_buffer[out_base + col] = my_row_data[col];
    }

    my_row_idx += grid_size;
    __syncwarp();
  }
  __syncwarp();
}

__global__ void
naiive_count(uint32_t *d_bool_vec, uint32_t num_compressed_blocks, uint32_t *debug_mask_bits)
{
  const uint32_t tid = threadIdx.x;
  const uint32_t bid = blockIdx.x;
  const uint32_t idx = tid + bid * blockDim.x;
  const uint32_t grid_size = blockDim.x * gridDim.x;

  uint32_t local_count = 0;
  for (uint32_t i = idx; i < num_compressed_blocks; i += grid_size)
  {
    uint32_t block = fast_load(d_bool_vec + i);
    atomicAdd(debug_mask_bits, __popc(block));
  }
}

__global__ void
retrive_valid_parents_kernel(
    uint64_t *target_parent_array, uint64_t target_num_rows, uint32_t *frag_data_array, uint32_t *res)
{
  const uint32_t tid = threadIdx.x;
  const uint32_t bid = blockIdx.x;
  const uint64_t idx = static_cast<uint64_t>(tid) + static_cast<uint64_t>(bid) * blockDim.x;
  const uint64_t grid_size = static_cast<uint64_t>(blockDim.x) * gridDim.x;

  for (uint64_t row_idx = idx; row_idx < target_num_rows; row_idx += grid_size)
  {
    uint64_t parent_idx = fast_load(target_parent_array + row_idx);
    if (parent_idx != UINT64_MAX)
    {
      uint32_t parent_value = fast_load(frag_data_array + parent_idx);
      res[parent_idx] = parent_value;
    }
  }
  __syncwarp();
}

__global__ void
check_same_kernel(
    uint32_t *data_array_1, uint32_t *data_array_2, uint64_t num_rows, uint32_t *res)
{
  const uint32_t tid = threadIdx.x;
  const uint32_t bid = blockIdx.x;
  const uint64_t idx = static_cast<uint64_t>(tid) + static_cast<uint64_t>(bid) * blockDim.x;
  const uint64_t grid_size = static_cast<uint64_t>(blockDim.x) * gridDim.x;

  for (uint64_t row_idx = idx; row_idx < num_rows; row_idx += grid_size)
  {
    uint32_t value_1 = fast_load(data_array_1 + row_idx);
    uint32_t value_2 = fast_load(data_array_2 + row_idx);
    if (value_1 != value_2)
      atomicAdd(res, 1);
  }
  __syncwarp();
}

__global__ void
extract_edge_kernel(
    uint32_t *d_key_array, uint32_t *d_value_array,
    uint64_t num_rows, uint32_t num_cols,
    uint32_t key_col, uint32_t value_col,
    uint64_t **d_parent_arrays,
    uint32_t *d_res)
{
  const uint32_t tid = threadIdx.x;
  const uint32_t bid = blockIdx.x;
  const uint64_t idx = static_cast<uint64_t>(tid) + static_cast<uint64_t>(bid) * blockDim.x;
  const uint64_t grid_size = static_cast<uint64_t>(blockDim.x) * gridDim.x;
  for (uint64_t row_idx = idx; row_idx < num_rows; row_idx += grid_size)
  {
    uint64_t current_pos = row_idx;
    uint32_t key_value = UINT32_MAX;
    uint32_t edge_value = UINT32_MAX;
    for (int col = num_cols - 1; col >= 0; --col)
    {
      if (col == key_col)
        key_value = fast_load(d_key_array + current_pos);
      if (col == value_col)
        edge_value = fast_load(d_value_array + current_pos);

      if (col == 0)
        break;
      current_pos = fast_load(d_parent_arrays[col] + current_pos);
    }
    // write back
    d_res[row_idx * 2] = key_value;
    d_res[row_idx * 2 + 1] = edge_value;
  }
  __syncwarp();
}

__global__ void
convert_ull_to_uint32(
    uint32_t *d_count_for_each, uint32_t *d_count_for_each_32, uint32_t num_items)
{
  const uint32_t tid = threadIdx.x;
  const uint32_t bid = blockIdx.x;
  const uint32_t idx = tid + bid * blockDim.x;
  const uint32_t grid_size = blockDim.x * gridDim.x;
  for (uint32_t i = idx; i < num_items; i += grid_size)
  {
    d_count_for_each_32[i] = static_cast<uint32_t>(fast_load(d_count_for_each + i));
  }
  __syncwarp();
}

__global__ void
collect_res_for_each(
    uint32_t *d_scan_res, uint64_t num_rows, uint32_t scan_length, uint32_t *d_res_for_each)
{
  const uint32_t tid = threadIdx.x;
  const uint32_t bid = blockIdx.x;
  const uint64_t idx = static_cast<uint64_t>(tid) + static_cast<uint64_t>(bid) * blockDim.x;
  const uint64_t grid_size = static_cast<uint64_t>(blockDim.x) * gridDim.x;

  for (uint64_t row_idx = idx; row_idx < num_rows; row_idx += grid_size)
  {
    uint32_t val = fast_load(d_scan_res + row_idx * scan_length + (scan_length - 1));
    d_res_for_each[row_idx] = val;
  }
  __syncwarp();
}
