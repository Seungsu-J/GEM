#include <cooperative_groups.h>

#include <cub/cub.cuh>

#include "cuda_helpers.cuh"
#include "globals.h"
#include "join_trie.cuh"
#include "join_trie_kernel.cuh"
#include "unifiedTrie.cuh"

namespace
{
  constexpr uint32_t HYBRID_DIRECT_THRESHOLD = 4;  // <5 -> linear scan
  constexpr uint32_t HYBRID_BINARY_THRESHOLD = 32; // 5-32 -> binary search

  __device__ __forceinline__ bool linear_contains_small(
      const vtype *values, uint32_t len, vtype target)
  {
    for (uint32_t i = 0; i < len; ++i)
    {
      if (fast_load(values + i) == target)
        return true;
    }
    return false;
  }

  __device__ __forceinline__ bool binary_contains_medium(
      const vtype *values, uint32_t len, vtype target)
  {
    uint32_t pos = lower_bound(const_cast<vtype *>(values), len, target);
    if (pos == UINT32_MAX)
      return false;
    return fast_load(values + pos) == target;
  }

  __device__ __forceinline__ bool galloping_contains_large(
      const vtype *values, uint32_t len, vtype target)
  {
    if (len == 0)
      return false;

    vtype first = fast_load(values);
    if (first == target)
      return true;
    if (target < first)
      return false;

    uint32_t bound = 1;
    while (bound < len)
    {
      vtype probe = fast_load(values + bound);
      if (probe >= target)
        break;
      bound <<= 1;
    }

    uint32_t right = (bound < len) ? bound : (len - 1);
    uint32_t left = bound >> 1;
    if (right < left)
      right = left;

    uint32_t span = right + 1 - left;
    uint32_t rel = lower_bound(const_cast<vtype *>(values + left), span, target);
    if (rel == UINT32_MAX)
      return false;
    return fast_load(values + left + rel) == target;
  }

  __device__ __forceinline__ bool dispatch_hybrid_search(
      const vtype *values, uint32_t len, vtype target)
  {
    if (len == 0)
      return false;

    if (len <= HYBRID_DIRECT_THRESHOLD)
      return linear_contains_small(values, len, target);
    if (len <= HYBRID_BINARY_THRESHOLD)
      return binary_contains_medium(values, len, target);
    return galloping_contains_large(values, len, target);
  }
} // namespace

__global__ void
select_trie_kernel(
    // frag
    vtype *frag_key_data_, vtype *frag_value_data_,
    uint64_t **parent_pointers_, uint32_t num_cols,
    uint32_t key_col, uint32_t val_col,
    uint64_t frag_num_rows,
    // edge
    vtype *edge_key_array_, vtype *edge_value_array_, offtype *edge_offsets_,
    // hash
    uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx,
    // result
    uint32_t *bool_vec)
{
  const uint32_t tid = threadIdx.x;
  const uint32_t bid = blockIdx.x;
  const uint32_t idx = tid + bid * blockDim.x;
  const uint32_t wid = tid >> 5;
  const uint32_t lid = tid & 31;
  const uint32_t wid_g = idx >> 5;

  const uint32_t grid_size = blockDim.x * gridDim.x;

  uint64_t base_row_idx = wid_g << 5;
  uint64_t my_row_idx;
  bool matched;
  uint32_t pos, key_vertex, value_vertex;

  while (base_row_idx < frag_num_rows)
  {
    my_row_idx = base_row_idx + lid;
    matched = false;
    pos = UINT32_MAX;
    key_vertex = UINT32_MAX;
    value_vertex = UINT32_MAX;

    if (my_row_idx < frag_num_rows)
    {
      uint64_t current_pos = my_row_idx;
      for (int col = num_cols - 1; col >= 0; --col)
      {
        if (col == key_col)
          key_vertex = fast_load(frag_key_data_ + current_pos);
        if (col == val_col)
          value_vertex = fast_load(frag_value_data_ + current_pos);

        if (col == 0)
          break;
        current_pos = fast_load(parent_pointers_[col] + current_pos);
      }
    }
    __syncwarp();

    if (key_vertex != UINT32_MAX && value_vertex != UINT32_MAX)
      pos = hash_lookup(num_buckets, edge_offset, directed_edge_idx, key_vertex);
    __syncwarp();

    if (pos != UINT32_MAX)
    {
      uint32_t st, ed;
      fast_load_pair(edge_offsets_ + pos, st, ed);
      uint32_t res = UINT32_MAX;
      if (ed > st)
        res = lower_bound(edge_value_array_ + st, ed - st, value_vertex);
      if (res != UINT32_MAX && fast_load(edge_value_array_ + st + res) == value_vertex)
      {
        matched = true;
      }
    }

    uint32_t mask = __ballot_sync(FULL_MASK, matched);

    if (lid == 0)
    {
      bool_vec[base_row_idx >> 5] = mask;
    }
    __syncwarp();
    base_row_idx += grid_size;
  }
}

__global__ void select_trie_refer_kernel(
    // original data to refer to
    vtype *frag_key_data_, vtype *frag_value_data_,
    uint64_t **parent_pointers_, uint32_t num_cols,
    uint32_t key_col, uint32_t val_col,
    // selection mask
    uint32_t *select_bool_vec_, uint64_t frag_num_rows,
    // trie_edge
    vtype *edge_key_array_, vtype *edge_value_array_, offtype *edge_offsets_,
    // hash
    uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx,
    // result
    uint32_t *bool_vec)
{
  const uint32_t tid = threadIdx.x;
  const uint32_t bid = blockIdx.x;
  const uint32_t idx = tid + bid * blockDim.x;
  const uint32_t wid = tid >> 5;
  const uint32_t lid = tid & 31;
  const uint32_t wid_g = idx >> 5;

  const uint32_t grid_size = blockDim.x * gridDim.x;

  uint64_t base_row_idx = wid_g << 5;
  vtype key_vertex, value_vertex;
  bool matched;
  uint64_t my_row_idx;
  uint32_t pos;
  uint32_t row_selected;

  while (base_row_idx < frag_num_rows)
  {
    uint64_t tile_idx = base_row_idx >> 5;
    // Check if any rows in this warp are selected
    row_selected = fast_load(select_bool_vec_ + tile_idx);

    if (!row_selected)
    {
      if (lid == 0)
        bool_vec[tile_idx] = 0;
      __syncwarp();
      base_row_idx += grid_size;
      continue;
    }

    my_row_idx = base_row_idx + lid;
    matched = false;
    pos = UINT32_MAX;
    key_vertex = UINT32_MAX;
    value_vertex = UINT32_MAX;

    bool row_selected_bit = (row_selected & (1u << lid)) != 0u;
    bool lane_active = row_selected_bit && (my_row_idx < frag_num_rows);

    // Load the two vertices forming the edge
    if (lane_active)
    {
      uint64_t current_pos = my_row_idx;
      // Traverse up the parent pointers from lower_col to the target higher_col layer
      for (int col_id = num_cols - 1; col_id >= 0; --col_id)
      {
        if (col_id == key_col)
          key_vertex = fast_load(frag_key_data_ + current_pos);
        if (col_id == val_col)
          value_vertex = fast_load(frag_value_data_ + current_pos);
        if (col_id == 0)
          break;
        current_pos = fast_load(parent_pointers_[col_id] + current_pos);
      }
    }
    __syncwarp();

    // Search for the key in the edge hash table
    if (lane_active && key_vertex != UINT32_MAX)
      pos = hash_lookup(num_buckets, edge_offset, directed_edge_idx, key_vertex);
    __syncwarp();

    // If key found, search for the value in the associated value list
    if (lane_active && pos != UINT32_MAX && value_vertex != UINT32_MAX)
    {
      uint32_t st, ed;
      fast_load_pair(edge_offsets_ + pos, st, ed);
      uint32_t res = UINT32_MAX;
      if (ed > st)
        res = lower_bound(edge_value_array_ + st, ed - st, value_vertex);
      if (res != UINT32_MAX && fast_load(edge_value_array_ + st + res) == value_vertex)
      {
        matched = true;
      }
    }

    // Collect matches from all threads in the warp using ballot
    uint32_t mask = __ballot_sync(FULL_MASK, matched);

    if (lid == 0)
      bool_vec[tile_idx] = mask;
    __syncwarp();
    base_row_idx += grid_size;
  }
}

/**
 * ChunkSize: number of rows processed per warp in one iteration (should be <= 32)
 */
template <int ChunkSize>
__global__ void
join_count_kernel_nodup(
    // trie_inter
    vtype **vertex_arrays_, uint64_t **parent_pointers_,
    uint64_t num_rows, uint32_t num_cols, uint32_t join_col,
    // trie_edge
    vtype *edge_key_array_, vtype *edge_value_array_, offtype *edge_offsets_,
    // hash
    uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx,
    // mask
    uint32_t *d_num_res_for_each, uint32_t *d_st_array)
{
  const uint32_t tid = threadIdx.x;
  const uint32_t bid = blockIdx.x;
  const uint32_t idx = tid + bid * blockDim.x;
  const uint32_t wid = tid >> 5;
  const uint32_t lid = tid & 31;
  const uint32_t wid_g = idx >> 5;
  const uint32_t num_warps = (blockDim.x >> 5) * gridDim.x;

  // Local variables for each lane's assigned row (lanes 0-3)
  vtype local_key_mapped;
  uint32_t local_st, local_ed;
  uint32_t local_pos;
  uint64_t base_row;
  uint64_t my_row_idx;
  uint64_t current_pos;

  // Process rows in tiles of 4 (one memory transaction = 128 bytes = 4 rows * 32 bytes)

  const uint64_t num_tiles = (num_rows + ChunkSize - 1) / ChunkSize;

#ifndef NDEBUG
  // Debug: Only print from first warp once
  if (wid_g == 0 && lid == 0)
  {
    printf("DEBUG: num_cols=%u, join_col=%u\n", num_cols, join_col);
    printf("DEBUG: num_rows=%u\n", num_rows);
    for (uint32_t c = 0; c < num_cols; ++c)
    {
      printf("DEBUG: vertex_arrays_[%u]=%p, parent_pointers_[%u]=%p\n",
             c, vertex_arrays_[c], c, parent_pointers_[c]);
    }
  }
#endif

  for (uint64_t tile_id = wid_g; tile_id < num_tiles; tile_id += num_warps)
  {
    base_row = tile_id * ChunkSize;
    my_row_idx = base_row + lid;
    local_key_mapped = UINT32_MAX;
    local_pos = UINT32_MAX;
    local_st = UINT32_MAX;
    local_ed = UINT32_MAX;

    // Lanes 0~`ChunkSize-1` fetch data for their respective rows simultaneously
    if (lid < ChunkSize)
    {
      if (my_row_idx < num_rows)
      {
        current_pos = my_row_idx;

        for (int col_id = num_cols - 1; col_id > join_col; --col_id)
          current_pos = fast_load(parent_pointers_[col_id] + current_pos);
        local_key_mapped = fast_load(vertex_arrays_[join_col] + current_pos);

        // Perform hash search
        if (local_key_mapped != UINT32_MAX)
        {
          // printf("Thread %u: looking up key %u\n", idx, local_key_mapped);
          local_pos = hash_lookup(num_buckets, edge_offset, directed_edge_idx, local_key_mapped);
          if (local_pos != UINT32_MAX)
            fast_load_pair(edge_offsets_ + local_pos, local_st, local_ed);
        }
      }
    }
    __syncwarp();

    if (local_st != UINT32_MAX)
    {
      // printf("Thread %u: key %u found candidates [%u, %u)\n",
      //  idx, local_key_mapped, local_st, local_ed);
      uint32_t num_values = local_ed - local_st;
      d_num_res_for_each[my_row_idx] = num_values;
      d_st_array[my_row_idx] = local_st;
    }
    __syncwarp();
  }
}

__global__ void
join_count_kernel_dup(
    // trie_inter
    vtype **vertex_arrays_, uint64_t **parent_pointers_,
    uint64_t num_rows, uint32_t num_cols, uint32_t join_col,
    // trie_edge
    vtype *edge_key_array_, vtype *edge_value_array_, offtype *edge_offsets_,
    // hash
    uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx,
    // mask
    uint32_t *value_mask, uint32_t mask_length, uint32_t *d_st_array,
    // options
    int dup_check_type, uint32_t num_dup_cols, uint32_t *dup_cols)
{
  const uint32_t tid = threadIdx.x;
  const uint32_t bid = blockIdx.x;
  const uint32_t idx = tid + bid * blockDim.x;
  const uint32_t wid = tid >> 5;
  const uint32_t lid = tid & 31;
  const uint32_t wid_g = idx >> 5;
  const uint32_t num_warps = (blockDim.x >> 5) * gridDim.x;

  // Local variables for each lane's assigned row (lanes 0-3)
  vtype local_key_mapped;
  uint32_t local_st, local_ed;
  uint32_t local_pos;
  vtype local_row_values[MAX_VQ]; // Only used by lanes 0-3 when dup_check_type > 0
  vtype bcast_row_values[MAX_VQ];

  // Broadcast variables for current row being processed
  vtype bcast_key;
  uint32_t bcast_st, bcast_ed;

  // Process rows in tiles of 4 (one memory transaction = 128 bytes = 4 rows * 32 bytes)
  constexpr uint32_t ROWS_PER_TILE = 4;
  uint64_t num_tiles = (num_rows + ROWS_PER_TILE - 1) / ROWS_PER_TILE;

  for (uint64_t tile_id = wid_g; tile_id < num_tiles; tile_id += num_warps)
  {
    uint64_t base_row = tile_id * ROWS_PER_TILE;
    // Lanes 0-3 fetch data for their respective rows simultaneously
    if (lid < ROWS_PER_TILE)
    {
      uint64_t my_row_idx = base_row + lid;
      local_key_mapped = UINT32_MAX;
      local_pos = UINT32_MAX;
      local_st = UINT32_MAX;
      local_ed = UINT32_MAX;

      if (my_row_idx < num_rows)
      {
        uint64_t current_pos = my_row_idx;

        // Duplication check required: traverse from bottom to root, collecting all values
        // Store in local registers (lanes 0-3 only)
        for (int col_id = num_cols - 1; col_id >= 1; --col_id)
        {
          local_row_values[col_id] = fast_load(vertex_arrays_[col_id] + current_pos);
          current_pos = fast_load(parent_pointers_[col_id] + current_pos);
        }
        local_row_values[0] = fast_load(vertex_arrays_[0] + current_pos); // loop peeling.
        local_key_mapped = local_row_values[join_col];

        // Perform hash search

        local_pos = hash_lookup(num_buckets, edge_offset, directed_edge_idx, local_key_mapped);
        if (local_pos != UINT32_MAX)
        {
          fast_load_pair(edge_offsets_ + local_pos, local_st, local_ed);
          d_st_array[my_row_idx] = local_st;
        }
      }
    }
    __syncwarp();

    // Now all lanes process each of the 4 rows cooperatively
    for (uint32_t row_in_tile = 0; row_in_tile < ROWS_PER_TILE; ++row_in_tile)
    {
      uint64_t current_row = base_row + row_in_tile;
      if (current_row >= num_rows)
        break;

      // Broadcast row's data from lane row_in_tile to all lanes using shuffle
      bcast_st = __shfl_sync(FULL_MASK, local_st, row_in_tile);
      if (bcast_st == UINT32_MAX)
        continue; // No candidates for this key

      bcast_key = __shfl_sync(FULL_MASK, local_key_mapped, row_in_tile);
      bcast_ed = __shfl_sync(FULL_MASK, local_ed, row_in_tile);

      // Warp cooperatively searches through edge values
      bool found = false;
      uint32_t num_values = bcast_ed - bcast_st;
      uint32_t num_value_tiles = (num_values + 31) >> 5;

      // shuffle local row values to bcast_row_values
      if (dup_check_type == 1)
        for (int i = 0; i < num_dup_cols; ++i)
          bcast_row_values[i] = __shfl_sync(FULL_MASK, local_row_values[dup_cols[i]], row_in_tile);
      else
        for (int i = 0; i < num_cols; ++i)
          bcast_row_values[i] = __shfl_sync(FULL_MASK, local_row_values[i], row_in_tile);

      for (uint32_t value_tile = 0; value_tile < num_value_tiles; ++value_tile)
      {
        uint32_t value_idx = value_tile * warpSize + lid;
        found = false;

        if (value_idx < num_values)
        {
          vtype edge_value = fast_load(edge_value_array_ + bcast_st + value_idx);

          // Check for duplicates by broadcasting values from lane row_in_tile
          bool dup = false;
          if (dup_check_type == 1)
          {
            // Partial check: only broadcast needed columns
            for (uint32_t i = 0; i < num_dup_cols; ++i)
            {
              // vtype bcast_col_value = __shfl_sync(FULL_MASK, local_row_values[dup_cols[i]], row_in_tile);
              if (bcast_row_values[i] == edge_value)
              {
                dup = true;
                break;
              }
            }
          }
          else // if (dup_check_type == 2)
          {
            // Full check: broadcast all columns
            for (uint32_t i = 0; i < num_cols; ++i)
            {
              // vtype bcast_col_value = __shfl_sync(FULL_MASK, local_row_values[i], row_in_tile);
              if (bcast_row_values[i] == edge_value)
              {
                dup = true;
                break;
              }
            }
          }
          found = !dup;
        }

        // Collect votes from all lanes
        uint32_t found_mask = __ballot_sync(FULL_MASK, found);

        // Lane 0 writes the mask to global memory
        if (lid == 0)
          value_mask[current_row * mask_length + value_tile] = found_mask;
        __syncwarp();
      }
    }
  }
}

template <int ChunkSize>
__global__ void
join_write_kernel_nodup(
    // trie_inter
    uint64_t num_rows,
    // trie_edge
    vtype *edge_value_array_,
    uint32_t *d_num_res_for_each, uint32_t *d_st_array,
    uint32_t *d_res, uint64_t *d_offsets_of_, uint64_t *d_parent_res)
{
  const uint32_t tid = threadIdx.x;
  const uint32_t bid = blockIdx.x;
  const uint32_t idx = tid + bid * blockDim.x;
  const uint32_t wid = tid >> 5;
  const uint32_t lid = tid & 31;
  const uint32_t wid_g = idx >> 5;

  const uint32_t num_warps = (blockDim.x >> 5) * gridDim.x;

  // Process rows in tiles of 4 (one memory transaction = 128 bytes = 4 rows * 32 bytes)

  uint32_t num_tiles = (num_rows + ChunkSize - 1) / ChunkSize;

  uint32_t local_st, local_num_values, local_offset;
  uint32_t bcast_st, bcast_num_values, bcast_offset;

  for (uint64_t tile_id = wid_g; tile_id < num_tiles; tile_id += num_warps)
  {
    uint64_t base_row = tile_id * ChunkSize;
    uint64_t my_row = base_row + lid;

    // Lanes 0~`ChunkSize-1` fetch data for their respective rows simultaneously
    local_st = UINT32_MAX;
    local_num_values = 0;
    local_offset = UINT32_MAX;

    if (my_row < num_rows)
    {
      local_st = fast_load(d_st_array + my_row);
      local_num_values = fast_load(d_num_res_for_each + my_row);
      local_offset = fast_load(d_offsets_of_ + my_row);
    }
    __syncwarp();

    for (int i = 0; i < ChunkSize; ++i)
    {
      // Broadcast row's data from lane i to all lanes using shuffle
      bcast_st = __shfl_sync(FULL_MASK, local_st, i);
      if (bcast_st == UINT32_MAX)
        continue; // No candidates for this key
      bcast_num_values = __shfl_sync(FULL_MASK, local_num_values, i);
      bcast_offset = __shfl_sync(FULL_MASK, local_offset, i);

      warp_copy_segment<uint32_t, 2>(
          edge_value_array_ + bcast_st,
          d_res + bcast_offset,
          bcast_num_values, lid, FULL_MASK);
      __syncwarp();

      for (int ii = lid; ii < bcast_num_values; ii += warpSize)
      {
        d_parent_res[bcast_offset + ii] = base_row + i;
      }
      __syncwarp();
    }
  }
}

__global__ void
join_write_kernel_dup(
    // trie_inter
    uint64_t num_rows,
    // trie_edge
    vtype *edge_value_array_,
    uint32_t *value_mask, uint32_t mask_length,
    uint64_t *d_offsets_of_, uint32_t *d_st_array,
    uint32_t *d_res, uint64_t *d_parent_res)
{
  const uint32_t tid = threadIdx.x;
  const uint32_t bid = blockIdx.x;
  const uint32_t idx = tid + bid * blockDim.x;
  const uint32_t wid = tid >> 5;
  const uint32_t lid = tid & 31;
  const uint32_t wid_g = idx >> 5;
  const uint32_t num_warps = (blockDim.x >> 5) * gridDim.x;

  uint64_t row_id = wid_g;
  uint32_t my_tile, bcast_tile, tile_base;
  const uint32_t num_loops = (mask_length + warpSize - 1) / warpSize;

  __shared__ uint32_t warp_pos[WARP_PER_BLOCK];
  __shared__ uint32_t task_queue[WARP_PER_BLOCK][WARP_SIZE];
  __shared__ uint32_t task_word_idx[WARP_PER_BLOCK][WARP_SIZE];
  __shared__ uint32_t num_tasks[WARP_PER_BLOCK];

  while (row_id < num_rows)
  {
    uint64_t res_offset = fast_load(d_offsets_of_ + row_id);
    uint64_t num_res_row = fast_load(d_offsets_of_ + row_id + 1) - res_offset;
    if (!num_res_row)
    {
      row_id += num_warps;
      continue;
    }

    if (lid == 0)
      warp_pos[wid] = 0;
    __syncwarp();

    uint32_t base_offset = fast_load(d_st_array + row_id);

    for (uint32_t loop_id = 0; loop_id < num_loops; ++loop_id)
    {
      // tile_base = loop_id * warpSize;
      tile_base = loop_id << 5;
      uint32_t my_tile_idx = tile_base + lid;
      if (my_tile_idx < mask_length)
        my_tile = fast_load(value_mask + row_id * mask_length + my_tile_idx);
      else
        my_tile = 0;

      uint32_t has_task_ballot_mask = __ballot_sync(FULL_MASK, my_tile != 0);
      if (has_task_ballot_mask == 0)
        continue;
      if (my_tile)
      {
        uint32_t task_idx = __popc(has_task_ballot_mask & ((1u << lid) - 1));
        task_queue[wid][task_idx] = my_tile;
        task_word_idx[wid][task_idx] = my_tile_idx;
        if (task_idx == 0)
          num_tasks[wid] = __popc(has_task_ballot_mask);
      }
      __syncwarp();

      for (uint32_t task_id = 0; task_id < num_tasks[wid]; ++task_id)
      {
        bcast_tile = task_queue[wid][task_id];
        uint32_t word_base = task_word_idx[wid][task_id] << 5; // word index * 32
        if (bcast_tile & (1u << lid))
        {
          uint32_t index_in_tile = __popc(bcast_tile & ((1u << lid) - 1));
          uint32_t global_index = warp_pos[wid] + index_in_tile;
          uint32_t edge_value_idx = base_offset + word_base + lid;
          uint64_t res_idx = res_offset + global_index;
          d_res[res_idx] = fast_load(edge_value_array_ + edge_value_idx);
          d_parent_res[res_idx] = row_id;
          if (index_in_tile == 0)
            warp_pos[wid] += __popc(bcast_tile);
        }
        __syncwarp();
      }
    }
    row_id += num_warps;
  }
}

template <int ChunkSize>
__global__ void
join_count_refer_kernel_nodup(
    // data trie
    vtype **vertex_arrays_, uint64_t **parent_pointers_,
    uint64_t num_rows, uint32_t num_cols, uint32_t join_col,
    // mask
    uint32_t *select_bool_vec_,
    // trie_edge
    vtype *edge_key_array_, vtype *edge_value_array_, offtype *edge_offsets_,
    // hash
    uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx,
    // mask
    uint32_t *d_num_res_for_each, uint32_t *d_st_array)
{
  const uint32_t tid = threadIdx.x;
  const uint32_t bid = blockIdx.x;
  const uint32_t idx = tid + bid * blockDim.x;
  const uint32_t wid = tid >> 5;
  const uint32_t lid = tid & 31;
  const uint32_t wid_g = idx >> 5;
  const uint32_t num_warps = (blockDim.x >> 5) * gridDim.x;

  // Local variables for each lane's assigned row (lanes 0-3)
  vtype local_key_mapped;
  uint32_t local_st, local_ed;
  uint32_t local_pos;
  uint32_t base_row;
  uint64_t current_row;
  uint64_t current_pos;
  uint32_t current_tile_rank;

  // Process rows in tiles of 4 (one memory transaction = 128 bytes = 4 rows * 32 bytes)

  const uint64_t num_tiles = (num_rows + ChunkSize - 1) / ChunkSize;

#ifndef NDEBUG
  // Debug: Only print from first warp once
  if (wid_g == 0 && lid == 0)
  {
    printf("DEBUG: num_cols=%u, join_col=%u\n", num_cols, join_col);
    printf("DEBUG: num_rows=%llu\n", num_rows);
    for (uint32_t c = 0; c < num_cols; ++c)
    {
      printf("DEBUG: vertex_arrays_[%u]=%p, parent_pointers_[%u]=%p\n",
             c, vertex_arrays_[c], c, parent_pointers_[c]);
    }
  }
#endif

  for (uint64_t tile_id = wid_g; tile_id < num_tiles; tile_id += num_warps)
  {
    base_row = tile_id * ChunkSize;
    current_row = base_row + lid;
    current_tile_rank = current_row & 31;

    uint32_t tile = fast_load(select_bool_vec_ + (base_row >> 5));
    bool flag = current_row < num_rows && lid < ChunkSize && (tile & (1u << current_tile_rank));
    uint32_t ballot_mask = __ballot_sync(FULL_MASK, flag == true);
    if (ballot_mask == 0)
      continue;

    local_key_mapped = UINT32_MAX;
    local_pos = UINT32_MAX;
    local_st = UINT32_MAX;
    local_ed = UINT32_MAX;

    // Lanes 0~`ChunkSize-1` fetch data for their respective rows simultaneously
    if (flag)
    {
      current_pos = current_row;

      for (int col_id = num_cols - 1; col_id > join_col; --col_id)
        current_pos = fast_load(parent_pointers_[col_id] + current_pos);
      local_key_mapped = fast_load(vertex_arrays_[join_col] + current_pos);

      // Perform hash search
      if (local_key_mapped != UINT32_MAX)
      {
        // printf("Thread %u: looking up key %u\n", idx, local_key_mapped);
        local_pos = hash_lookup(num_buckets, edge_offset, directed_edge_idx, local_key_mapped);
        if (local_pos != UINT32_MAX)
          fast_load_pair(edge_offsets_ + local_pos, local_st, local_ed);
      }
    }
    __syncwarp();

    if (local_st != UINT32_MAX)
    {
      // printf("Thread %u: key %u found candidates [%u, %u)\n",
      //  idx, local_key_mapped, local_st, local_ed);
      uint32_t num_values = local_ed - local_st;
      d_num_res_for_each[current_row] = num_values;
      d_st_array[current_row] = local_st;
    }
    __syncwarp();
  }
}

template <int ChunkSize>
__global__ void
join_write_refer_kernel_nodup(
    // trie_inter
    uint64_t *valid_row_ids, uint64_t num_valid_rows,
    // trie_edge
    vtype *edge_value_array_,
    uint32_t *d_selected_count, uint32_t *d_st_array,
    uint32_t *d_res, uint64_t *d_offsets_of_, uint64_t *d_parent_res)
{
  const uint32_t tid = threadIdx.x;
  const uint32_t bid = blockIdx.x;
  const uint32_t idx = tid + bid * blockDim.x;
  const uint32_t wid = tid >> 5;
  const uint32_t lid = tid & 31;
  const uint32_t wid_g = idx >> 5;

  const uint32_t num_warps = (blockDim.x >> 5) * gridDim.x;

  // Process rows in tiles of 4 (one memory transaction = 128 bytes = 4 rows * 32 bytes)

  uint64_t num_tiles = (num_valid_rows + ChunkSize - 1) / ChunkSize;

  uint32_t local_st, local_num_values, local_offset;
  uint32_t bcast_st, bcast_num_values, bcast_offset;
  uint64_t my_row, bcast_row;

  for (uint64_t tile_id = wid_g; tile_id < num_tiles; tile_id += num_warps)
  {
    uint64_t base_row_idx = tile_id * ChunkSize;
    uint64_t my_row_idx = base_row_idx + lid;

    // Lanes 0~`ChunkSize-1` fetch data for their respective rows simultaneously
    local_st = UINT32_MAX;
    local_num_values = 0;
    local_offset = UINT32_MAX;
    // my_row = UINT64_MAX;

    if (my_row_idx < num_valid_rows)
    {
      my_row = fast_load(valid_row_ids + my_row_idx);
      local_st = fast_load(d_st_array + my_row);
      local_num_values = fast_load(d_selected_count + my_row_idx);
      local_offset = fast_load(d_offsets_of_ + my_row_idx);
    }
    __syncwarp();

    for (int i = 0; i < ChunkSize; ++i)
    {
      // Broadcast row's data from lane i to all lanes using shuffle
      bcast_st = __shfl_sync(FULL_MASK, local_st, i);
      if (bcast_st == UINT32_MAX)
        continue; // No candidates for this key
      bcast_num_values = __shfl_sync(FULL_MASK, local_num_values, i);
      bcast_offset = __shfl_sync(FULL_MASK, local_offset, i);
      bcast_row = __shfl_sync(FULL_MASK, my_row, i);

      warp_copy_segment<uint32_t, 2>(
          edge_value_array_ + bcast_st,
          d_res + bcast_offset,
          bcast_num_values, lid, FULL_MASK);
      __syncwarp();

      for (int ii = lid; ii < bcast_num_values; ii += warpSize)
      {
        d_parent_res[bcast_offset + ii] = bcast_row;
      }
      __syncwarp();
    }
  }
}

/**
 * join_refer_count_dup_kernel - Combines refer logic with duplication checking
 * Refers to root data through selection mask and checks for duplicates
 */
__global__ void
join_refer_count_dup_kernel(
    // data trie
    vtype **vertex_arrays_, uint64_t **parent_pointers_, uint32_t num_layers,
    // mask
    uint32_t *select_bool_vec_, uint64_t num_rows,
    uint32_t num_cols, uint32_t join_col,
    // trie_edge
    vtype *edge_key_array_, vtype *edge_value_array_, offtype *edge_offsets_,
    // hash
    uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx,
    // mask
    uint32_t *value_mask, uint32_t mask_length, uint32_t *d_st_array,
    // options
    int dup_check_type, uint32_t num_dup_cols, uint32_t *dup_cols)
{
  const uint32_t tid = threadIdx.x;
  const uint32_t bid = blockIdx.x;
  const uint32_t idx = tid + bid * blockDim.x;
  const uint32_t wid = tid >> 5;
  const uint32_t lid = tid & 31;
  const uint32_t wid_g = idx >> 5;
  const uint32_t grid_size = blockDim.x * gridDim.x;

  uint32_t row_selected;
  uint64_t base_row_idx = wid_g << 5;
  vtype current_row_values[MAX_VQ];
  vtype key_mapped;
  uint32_t pos, my_st, my_ed;

  while (base_row_idx < num_rows)
  {
    // Check if any rows in this warp's 32-row block are selected

    row_selected = select_bool_vec_[base_row_idx >> 5];

    if (!row_selected)
    {
      base_row_idx += grid_size;
      continue;
    }

    uint64_t my_row = base_row_idx + lid;
    key_mapped = UINT32_MAX;
    pos = UINT32_MAX;
    my_st = UINT32_MAX;
    my_ed = UINT32_MAX;

    // Check if this specific row is selected and extract data
    if (my_row < num_rows)
    {
      bool row_selected_bit = row_selected & (1u << lid);

      if (row_selected_bit)
      {
        uint64_t current_pos = my_row;

        // Traverse from bottom to root, collecting all values for dup check
        for (int col_id = num_cols - 1; col_id >= 1; --col_id)
        {
          current_row_values[col_id] = fast_load(vertex_arrays_[col_id] + current_pos);
          current_pos = fast_load(parent_pointers_[col_id] + current_pos);
        }
        current_row_values[0] = fast_load(vertex_arrays_[0] + current_pos);

        // Extract join key
        key_mapped = current_row_values[join_col];

        // Hash lookup
        if (key_mapped != UINT32_MAX)
        {
          pos = hash_lookup(num_buckets, edge_offset, directed_edge_idx, key_mapped);
          if (pos != UINT32_MAX)
          {
            fast_load_pair(edge_offsets_ + pos, my_st, my_ed);
            d_st_array[my_row] = my_st;
          }
        }
      }
    }
    __syncwarp();

    // Process each selected row in this warp cooperatively
    for (uint32_t lane = 0; lane < warpSize; ++lane)
    {
      uint32_t bcast_st = __shfl_sync(FULL_MASK, my_st, lane);
      if (bcast_st == UINT32_MAX)
        continue;

      uint32_t bcast_ed = __shfl_sync(FULL_MASK, my_ed, lane);
      uint32_t num_values = bcast_ed - bcast_st;
      uint32_t num_value_tiles = (num_values + 31) >> 5;

      // Broadcast row values for duplication checking
      vtype bcast_row_values[MAX_VQ];
      if (dup_check_type == 1)
        for (uint32_t i = 0; i < num_dup_cols; ++i)
          bcast_row_values[i] = __shfl_sync(FULL_MASK, current_row_values[dup_cols[i]], lane);
      else
        for (uint32_t i = 0; i < num_cols; ++i)
          bcast_row_values[i] = __shfl_sync(FULL_MASK, current_row_values[i], lane);

      // Process edge values in tiles
      for (uint32_t value_tile = 0; value_tile < num_value_tiles; ++value_tile)
      {
        uint32_t value_idx = value_tile * warpSize + lid;
        bool found = false;

        if (value_idx < num_values)
        {
          vtype edge_value = fast_load(edge_value_array_ + bcast_st + value_idx);

          // Check for duplicates
          bool dup = false;
          if (dup_check_type == 1)
          {
            for (uint32_t i = 0; i < num_dup_cols; ++i)
            {
              if (bcast_row_values[i] == edge_value)
              {
                dup = true;
                break;
              }
            }
          }
          else // dup_check_type == 2
          {
            for (uint32_t i = 0; i < num_cols; ++i)
            {
              if (bcast_row_values[i] == edge_value)
              {
                dup = true;
                break;
              }
            }
          }
          found = !dup;
        }

        uint32_t found_mask = __ballot_sync(FULL_MASK, found);
        if (lid == 0)
        {
          uint64_t row_idx = base_row_idx + lane;
          value_mask[row_idx * mask_length + value_tile] = found_mask;
        }
        __syncwarp();
      }
    }
    base_row_idx += grid_size;
  }
}

/**
 * join_refer_write_dup_kernel - Writes results from refer join with dup checking
 * Reads from value_mask and writes non-duplicate edge values to result
 */
__global__ void
join_refer_write_dup_kernel(
    // data trie
    vtype **vertex_arrays_, uint64_t **parent_pointers_, uint32_t num_layers,
    uint64_t num_rows, uint32_t num_cols, uint32_t join_col,
    // trie_edge
    vtype *edge_key_array_, vtype *edge_value_array_, offtype *edge_offsets_,
    // hash
    uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx,
    // mask and result
    uint32_t *value_mask, uint32_t mask_length, uint32_t *d_st_array,
    uint32_t *d_res,
    uint64_t *d_row_offsets,
    uint64_t *d_parent_res,
    // options
    int dup_check_type, uint32_t num_dup_cols, uint32_t *dup_cols)
{
  const uint32_t tid = threadIdx.x;
  const uint32_t bid = blockIdx.x;
  const uint32_t idx = tid + bid * blockDim.x;
  const uint32_t wid = tid >> 5;
  const uint32_t lid = tid & 31;
  const uint32_t wid_g = idx >> 5;
  const uint32_t num_warps = (blockDim.x >> 5) * gridDim.x;

  uint64_t row_id = wid_g;
  uint32_t my_tile, bcast_tile;
  uint32_t tile_base;
  const uint32_t num_loops = (mask_length + warpSize - 1) / warpSize;

  __shared__ uint32_t warp_pos[WARP_PER_BLOCK];
  __shared__ uint32_t task_queue[WARP_PER_BLOCK][WARP_SIZE];
  __shared__ uint32_t task_word_idx[WARP_PER_BLOCK][WARP_SIZE];
  __shared__ uint32_t num_tasks[WARP_PER_BLOCK];

  while (row_id < num_rows)
  {
    uint64_t num_res_row = d_row_offsets[row_id + 1] - d_row_offsets[row_id];
    if (!num_res_row)
    {
      row_id += num_warps;
      continue;
    }

    if (lid == 0)
      warp_pos[wid] = 0;
    __syncwarp();

    uint32_t base_offset = fast_load(d_st_array + row_id);
    uint64_t res_offset = fast_load(d_row_offsets + row_id);

    for (uint32_t loop_id = 0; loop_id < num_loops; ++loop_id)
    {
      tile_base = loop_id << 5;
      uint32_t my_tile_idx = tile_base + lid;
      if (my_tile_idx < mask_length)
        my_tile = fast_load(value_mask + row_id * mask_length + my_tile_idx);
      else
        my_tile = 0;

      uint32_t has_task_ballot_mask = __ballot_sync(FULL_MASK, my_tile != 0);
      if (has_task_ballot_mask == 0)
        continue;

      if (my_tile)
      {
        uint32_t task_idx = __popc(has_task_ballot_mask & ((1u << lid) - 1));
        task_queue[wid][task_idx] = my_tile;
        task_word_idx[wid][task_idx] = my_tile_idx;
        if (task_idx == 0)
          num_tasks[wid] = __popc(has_task_ballot_mask);
      }
      __syncwarp();

      for (uint32_t task_id = 0; task_id < num_tasks[wid]; ++task_id)
      {
        bcast_tile = task_queue[wid][task_id];
        uint32_t word_base = task_word_idx[wid][task_id] << 5;

        if (bcast_tile & (1u << lid))
        {
          uint32_t index_in_tile = __popc(bcast_tile & ((1u << lid) - 1));
          uint32_t global_index = warp_pos[wid] + index_in_tile;
          uint32_t edge_value_idx = base_offset + word_base + lid;
          uint64_t res_idx = res_offset + global_index;
          d_res[res_idx] = fast_load(edge_value_array_ + edge_value_idx);
          d_parent_res[res_idx] = row_id;

          if (index_in_tile == 0)
            warp_pos[wid] += __popc(bcast_tile);
        }
        __syncwarp();
      }
    }
    row_id += num_warps;
  }
}

template __global__ void join_count_kernel_nodup<1>(vtype **vertex_arrays_, uint64_t **parent_pointers_, uint64_t num_rows, uint32_t num_cols, uint32_t join_col, vtype *edge_key_array_, vtype *edge_value_array_, offtype *edge_offsets_, uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx, uint32_t *d_num_res_for_each, uint32_t *d_st_array);
template __global__ void join_count_kernel_nodup<2>(vtype **vertex_arrays_, uint64_t **parent_pointers_, uint64_t num_rows, uint32_t num_cols, uint32_t join_col, vtype *edge_key_array_, vtype *edge_value_array_, offtype *edge_offsets_, uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx, uint32_t *d_num_res_for_each, uint32_t *d_st_array);
template __global__ void join_count_kernel_nodup<4>(vtype **vertex_arrays_, uint64_t **parent_pointers_, uint64_t num_rows, uint32_t num_cols, uint32_t join_col, vtype *edge_key_array_, vtype *edge_value_array_, offtype *edge_offsets_, uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx, uint32_t *d_num_res_for_each, uint32_t *d_st_array);
template __global__ void join_count_kernel_nodup<8>(vtype **vertex_arrays_, uint64_t **parent_pointers_, uint64_t num_rows, uint32_t num_cols, uint32_t join_col, vtype *edge_key_array_, vtype *edge_value_array_, offtype *edge_offsets_, uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx, uint32_t *d_num_res_for_each, uint32_t *d_st_array);
template __global__ void join_count_kernel_nodup<16>(vtype **vertex_arrays_, uint64_t **parent_pointers_, uint64_t num_rows, uint32_t num_cols, uint32_t join_col, vtype *edge_key_array_, vtype *edge_value_array_, offtype *edge_offsets_, uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx, uint32_t *d_num_res_for_each, uint32_t *d_st_array);
template __global__ void join_count_kernel_nodup<32>(vtype **vertex_arrays_, uint64_t **parent_pointers_, uint64_t num_rows, uint32_t num_cols, uint32_t join_col, vtype *edge_key_array_, vtype *edge_value_array_, offtype *edge_offsets_, uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx, uint32_t *d_num_res_for_each, uint32_t *d_st_array);

template __global__ void join_write_kernel_nodup<1>(uint64_t num_rows, vtype *edge_value_array_, uint32_t *d_num_res_for_each, uint32_t *d_st_array, uint32_t *d_res, uint64_t *d_offsets_of_, uint64_t *d_parent_res);
template __global__ void join_write_kernel_nodup<2>(uint64_t num_rows, vtype *edge_value_array_, uint32_t *d_num_res_for_each, uint32_t *d_st_array, uint32_t *d_res, uint64_t *d_offsets_of_, uint64_t *d_parent_res);
template __global__ void join_write_kernel_nodup<4>(uint64_t num_rows, vtype *edge_value_array_, uint32_t *d_num_res_for_each, uint32_t *d_st_array, uint32_t *d_res, uint64_t *d_offsets_of_, uint64_t *d_parent_res);
template __global__ void join_write_kernel_nodup<8>(uint64_t num_rows, vtype *edge_value_array_, uint32_t *d_num_res_for_each, uint32_t *d_st_array, uint32_t *d_res, uint64_t *d_offsets_of_, uint64_t *d_parent_res);
template __global__ void join_write_kernel_nodup<16>(uint64_t num_rows, vtype *edge_value_array_, uint32_t *d_num_res_for_each, uint32_t *d_st_array, uint32_t *d_res, uint64_t *d_offsets_of_, uint64_t *d_parent_res);
template __global__ void join_write_kernel_nodup<32>(uint64_t num_rows, vtype *edge_value_array_, uint32_t *d_num_res_for_each, uint32_t *d_st_array, uint32_t *d_res, uint64_t *d_offsets_of_, uint64_t *d_parent_res);

template __global__ void join_count_refer_kernel_nodup<1>(vtype **vertex_arrays_, uint64_t **parent_pointers_, uint64_t num_rows, uint32_t num_cols, uint32_t join_col, uint32_t *select_bool_vec_, vtype *edge_key_array_, vtype *edge_value_array_, offtype *edge_offsets_, uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx, uint32_t *d_num_res_for_each, uint32_t *d_st_array);
template __global__ void join_count_refer_kernel_nodup<2>(vtype **vertex_arrays_, uint64_t **parent_pointers_, uint64_t num_rows, uint32_t num_cols, uint32_t join_col, uint32_t *select_bool_vec_, vtype *edge_key_array_, vtype *edge_value_array_, offtype *edge_offsets_, uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx, uint32_t *d_num_res_for_each, uint32_t *d_st_array);
template __global__ void join_count_refer_kernel_nodup<4>(vtype **vertex_arrays_, uint64_t **parent_pointers_, uint64_t num_rows, uint32_t num_cols, uint32_t join_col, uint32_t *select_bool_vec_, vtype *edge_key_array_, vtype *edge_value_array_, offtype *edge_offsets_, uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx, uint32_t *d_num_res_for_each, uint32_t *d_st_array);
template __global__ void join_count_refer_kernel_nodup<8>(vtype **vertex_arrays_, uint64_t **parent_pointers_, uint64_t num_rows, uint32_t num_cols, uint32_t join_col, uint32_t *select_bool_vec_, vtype *edge_key_array_, vtype *edge_value_array_, offtype *edge_offsets_, uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx, uint32_t *d_num_res_for_each, uint32_t *d_st_array);
template __global__ void join_count_refer_kernel_nodup<16>(vtype **vertex_arrays_, uint64_t **parent_pointers_, uint64_t num_rows, uint32_t num_cols, uint32_t join_col, uint32_t *select_bool_vec_, vtype *edge_key_array_, vtype *edge_value_array_, offtype *edge_offsets_, uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx, uint32_t *d_num_res_for_each, uint32_t *d_st_array);
template __global__ void join_count_refer_kernel_nodup<32>(vtype **vertex_arrays_, uint64_t **parent_pointers_, uint64_t num_rows, uint32_t num_cols, uint32_t join_col, uint32_t *select_bool_vec_, vtype *edge_key_array_, vtype *edge_value_array_, offtype *edge_offsets_, uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx, uint32_t *d_num_res_for_each, uint32_t *d_st_array);

template __global__ void join_write_refer_kernel_nodup<1>(uint64_t *valid_row_ids, uint64_t num_valid_rows, vtype *edge_value_array_, uint32_t *d_selected_count, uint32_t *d_st_array, uint32_t *d_res, uint64_t *d_offsets_of_, uint64_t *d_parent_res);
template __global__ void join_write_refer_kernel_nodup<2>(uint64_t *valid_row_ids, uint64_t num_valid_rows, vtype *edge_value_array_, uint32_t *d_selected_count, uint32_t *d_st_array, uint32_t *d_res, uint64_t *d_offsets_of_, uint64_t *d_parent_res);
template __global__ void join_write_refer_kernel_nodup<4>(uint64_t *valid_row_ids, uint64_t num_valid_rows, vtype *edge_value_array_, uint32_t *d_selected_count, uint32_t *d_st_array, uint32_t *d_res, uint64_t *d_offsets_of_, uint64_t *d_parent_res);
template __global__ void join_write_refer_kernel_nodup<8>(uint64_t *valid_row_ids, uint64_t num_valid_rows, vtype *edge_value_array_, uint32_t *d_selected_count, uint32_t *d_st_array, uint32_t *d_res, uint64_t *d_offsets_of_, uint64_t *d_parent_res);
template __global__ void join_write_refer_kernel_nodup<16>(uint64_t *valid_row_ids, uint64_t num_valid_rows, vtype *edge_value_array_, uint32_t *d_selected_count, uint32_t *d_st_array, uint32_t *d_res, uint64_t *d_offsets_of_, uint64_t *d_parent_res);
template __global__ void join_write_refer_kernel_nodup<32>(uint64_t *valid_row_ids, uint64_t num_valid_rows, vtype *edge_value_array_, uint32_t *d_selected_count, uint32_t *d_st_array, uint32_t *d_res, uint64_t *d_offsets_of_, uint64_t *d_parent_res);

// helper kernels.
__global__ void
gather_counts_kernel(
    uint32_t *d_count_src, uint64_t *d_indices, uint64_t num_indices, uint32_t *d_count_dst)
{
  const uint32_t tid = threadIdx.x;
  const uint32_t bid = blockIdx.x;
  const uint32_t idx = tid + bid * blockDim.x;
  const uint32_t grid_size = blockDim.x * gridDim.x;

  for (uint32_t i = idx; i < num_indices; i += grid_size)
  {
    uint64_t src_idx = fast_load(d_indices + i);
    uint32_t count_value = fast_load(d_count_src + src_idx);
    d_count_dst[i] = count_value;
  }
  __syncwarp();
}

// debug kernels
__global__ void
select_trie_kernel_debug(
    // frag
    vtype *frag_key_data_, vtype *frag_value_data_,
    uint64_t **parent_pointers_, uint32_t num_cols,
    uint32_t key_col, uint32_t val_col,
    uint64_t frag_num_rows,
    // edge
    vtype *edge_key_array_, vtype *edge_value_array_, offtype *edge_offsets_,
    // hash
    uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx,
    // result
    uint32_t *bool_vec, uint64_t *num_result, uint32_t *d_res_debug)
{
  const uint32_t tid = threadIdx.x;
  const uint32_t bid = blockIdx.x;
  const uint32_t idx = tid + bid * blockDim.x;
  const uint32_t wid = tid >> 5;
  const uint32_t lid = tid & 31;
  const uint32_t wid_g = idx >> 5;

  const uint32_t grid_size = blockDim.x * gridDim.x;

  uint32_t base_row_idx = wid_g << 5;
  uint32_t my_row_idx;
  bool matched;
  uint32_t pos, key_vertex, value_vertex;

  while (base_row_idx < frag_num_rows)
  {
    my_row_idx = base_row_idx + lid;
    matched = false;
    pos = UINT32_MAX;
    key_vertex = UINT32_MAX;
    value_vertex = UINT32_MAX;

    if (my_row_idx < frag_num_rows)
    {
      uint32_t current_pos = my_row_idx;
      for (int col = num_cols - 1; col >= 0; --col)
      {
        if (col == key_col)
          key_vertex = fast_load(frag_key_data_ + current_pos);
        if (col == val_col)
          value_vertex = fast_load(frag_value_data_ + current_pos);

        if (col == 0)
          break;
        current_pos = fast_load(parent_pointers_[col] + current_pos);
      }
      // d_res_debug[my_row_idx * 2] = key_vertex;
      // d_res_debug[my_row_idx * 2 + 1] = value_vertex;
    }
    __syncwarp();

    if (key_vertex != UINT32_MAX && value_vertex != UINT32_MAX)
      pos = hash_lookup(num_buckets, edge_offset, directed_edge_idx, key_vertex);
    __syncwarp();

    if (pos != UINT32_MAX)
    {
      // d_res_debug[my_row_idx * 2 + 0] = 1; // all passed
      uint32_t st, ed;
      fast_load_pair(edge_offsets_ + pos, st, ed);
      d_res_debug[my_row_idx * 2 + 0] = ed - st; // range size
      uint32_t res = UINT32_MAX;
      if (ed > st)
        res = lower_bound(edge_value_array_ + st, ed - st, value_vertex);
      if (res != UINT32_MAX && fast_load(edge_value_array_ + st + res) == value_vertex)
      {
        matched = true;
        d_res_debug[my_row_idx * 2 + 1] = 1;
      }
    }

    uint32_t mask = __ballot_sync(FULL_MASK, matched);

    if (lid == 0)
    {
      atomicAdd((unsigned long long *)num_result, (unsigned long long)__popc(mask));
      bool_vec[base_row_idx >> 5] = mask;
    }
    __syncwarp();
    base_row_idx += grid_size;
  }
}

__global__ void
join_write_kernel_dup_debug(
    // trie_inter
    uint64_t num_rows,
    // trie_edge
    vtype *edge_value_array_,
    uint32_t *value_mask, uint32_t mask_length,
    uint64_t *d_offsets_of_, uint32_t *d_st_array,
    uint32_t *d_res, uint64_t *d_parent_res,
    // for debug
    uint64_t *num_processed, uint64_t num_total_res)
{
  const uint32_t tid = threadIdx.x;
  const uint32_t bid = blockIdx.x;
  const uint32_t idx = tid + bid * blockDim.x;
  const uint32_t wid = tid >> 5;
  const uint32_t lid = tid & 31;
  const uint32_t wid_g = idx >> 5;
  const uint32_t num_warps = (blockDim.x >> 5) * gridDim.x;

  uint64_t row_id = wid_g;
  uint32_t my_tile, bcast_tile, tile_base;
  const uint32_t num_loops = (mask_length + warpSize - 1) / warpSize;

  __shared__ uint64_t warp_pos[WARP_PER_BLOCK];
  __shared__ uint32_t task_queue[WARP_PER_BLOCK][WARP_SIZE];
  __shared__ uint32_t task_word_idx[WARP_PER_BLOCK][WARP_SIZE];
  __shared__ uint32_t num_tasks[WARP_PER_BLOCK];

  while (row_id < num_rows)
  {
    uint64_t res_offset = fast_load(d_offsets_of_ + row_id);
    uint64_t num_res_row = fast_load(d_offsets_of_ + row_id + 1) - res_offset;
    if (!num_res_row)
    {
      row_id += num_warps;
      continue;
    }

    if (lid == 0)
      warp_pos[wid] = 0;
    __syncwarp();

    uint32_t base_offset = fast_load(d_st_array + row_id);

    for (uint32_t loop_id = 0; loop_id < num_loops; ++loop_id)
    {
      // tile_base = loop_id * warpSize;
      tile_base = loop_id << 5;
      uint32_t my_tile_idx = tile_base + lid;
      if (my_tile_idx < mask_length)
        my_tile = fast_load(value_mask + row_id * mask_length + my_tile_idx);
      else
        my_tile = 0;

      uint32_t has_task_ballot_mask = __ballot_sync(FULL_MASK, my_tile != 0);
      if (has_task_ballot_mask == 0)
        continue;
      if (my_tile)
      {
        uint32_t task_idx = __popc(has_task_ballot_mask & ((1u << lid) - 1));
        task_queue[wid][task_idx] = my_tile;
        task_word_idx[wid][task_idx] = my_tile_idx;
        if (task_idx == 0)
          num_tasks[wid] = __popc(has_task_ballot_mask);
      }
      __syncwarp();

      for (uint32_t task_id = 0; task_id < num_tasks[wid]; ++task_id)
      {
        bcast_tile = task_queue[wid][task_id];
        uint32_t word_base = task_word_idx[wid][task_id] << 5; // word index * 32
        if (bcast_tile & (1u << lid))
        {
          uint32_t index_in_tile = __popc(bcast_tile & ((1u << lid) - 1));
          uint64_t global_index = warp_pos[wid] + index_in_tile;
          uint32_t edge_value_idx = base_offset + word_base + lid;
          uint64_t res_idx = res_offset + global_index;
          if (res_idx >= num_total_res)
          {
            // printf("Error: res_idx %llu out of bounds %llu\n", res_idx, num_total_res);
          }
          else
          {
            d_res[res_idx] = fast_load(edge_value_array_ + edge_value_idx);
            d_parent_res[res_idx] = row_id;
          }
          if (index_in_tile == 0)
          {
            warp_pos[wid] += __popc(bcast_tile);
          }
        }
        __syncwarp();
      }
    }
    if (lid == 0)
    {
      atomicAdd((unsigned long long *)num_processed, (unsigned long long)warp_pos[wid]);
    }
    __syncwarp();
    row_id += num_warps;
  }
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
    uint32_t current_pos = my_row_idx;
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
  const uint32_t idx = tid + bid * blockDim.x;
  const uint32_t grid_size = blockDim.x * gridDim.x;

  for (uint32_t row_idx = idx; row_idx < target_num_rows; row_idx += grid_size)
  {
    uint32_t parent_idx = fast_load(target_parent_array + row_idx);
    if (parent_idx != UINT32_MAX)
    {
      uint32_t parent_value = fast_load(frag_data_array + parent_idx);
      res[parent_idx] = parent_value;
    }
  }
  __syncwarp();
}

__global__ void
check_same_kernel(
    uint32_t *data_array_1, uint32_t *data_array_2, uint32_t num_rows, uint32_t *res)
{
  const uint32_t tid = threadIdx.x;
  const uint32_t bid = blockIdx.x;
  const uint32_t idx = tid + bid * blockDim.x;
  const uint32_t grid_size = blockDim.x * gridDim.x;

  for (uint32_t row_idx = idx; row_idx < num_rows; row_idx += grid_size)
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
    uint32_t num_rows, uint32_t num_cols,
    uint32_t key_col, uint32_t value_col,
    uint64_t **d_parent_arrays,
    uint32_t *d_res)
{
  const uint32_t tid = threadIdx.x;
  const uint32_t bid = blockIdx.x;
  const uint32_t idx = tid + bid * blockDim.x;
  const uint32_t grid_size = blockDim.x * gridDim.x;
  for (uint32_t row_idx = idx; row_idx < num_rows; row_idx += grid_size)
  {
    uint32_t current_pos = row_idx;
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
    uint32_t *d_scan_res, uint32_t num_rows, uint32_t scan_length, uint32_t *d_res_for_each)
{
  const uint32_t tid = threadIdx.x;
  const uint32_t bid = blockIdx.x;
  const uint32_t idx = tid + bid * blockDim.x;
  const uint32_t grid_size = blockDim.x * gridDim.x;

  for (uint32_t row_idx = idx; row_idx < num_rows; row_idx += grid_size)
  {
    uint32_t val = fast_load(d_scan_res + row_idx * scan_length + (scan_length - 1));
    d_res_for_each[row_idx] = val;
  }
  __syncwarp();
}
