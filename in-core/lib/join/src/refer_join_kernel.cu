#include "refer_join_kernel.cuh"
#include "cuda_helpers.cuh"
#include "join_context.cuh"
#include "unifiedTrie.cuh"

template <typename T>
__device__ __forceinline__ T load_joininfo(const JoinInfo<T> &info, uint64_t global_idx)
{
  if (info.array_ptr == nullptr)
    return static_cast<T>(~0ull);

  uint64_t local_idx = info.is_partial ? (global_idx - info.offset) : global_idx;
  return fast_load(info.array_ptr + local_idx);
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
    vtype *edge_key_array_, vtype *edge_value_array_, uint32_t *edge_offsets_,
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
join_count_refer_kernel_nodup_pipeline_info(
    // data trie
    JoinInfo<uint32_t> *data_info_arrays_, JoinInfo<uint64_t> *parent_info_arrays_,
    // selection mask (windowed)
    uint32_t *select_bool_vec_,
    uint64_t pipeline_off_st, uint64_t pipeline_num_rows,
    uint32_t num_cols, uint32_t join_col,
    // trie_edge
    vtype *edge_key_array_, vtype *edge_value_array_, uint32_t *edge_offsets_,
    // hash
    uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx,
    // result
    uint32_t *d_num_res_for_each, uint32_t *d_st_array)
{
  const uint32_t tid = threadIdx.x;
  const uint32_t bid = blockIdx.x;
  const uint32_t idx = tid + bid * blockDim.x;
  const uint32_t lid = tid & 31;
  const uint32_t wid_g = idx >> 5;
  const uint32_t num_warps = (blockDim.x >> 5) * gridDim.x;

  vtype local_key_mapped;
  uint32_t local_st, local_ed;
  uint32_t local_pos;
  uint64_t base_row;
  uint64_t local_row;
  uint64_t global_row;
  uint64_t current_pos;
  uint32_t local_rank;

  const uint64_t num_tiles = (pipeline_num_rows + ChunkSize - 1) / ChunkSize;

  for (uint64_t tile_id = wid_g; tile_id < num_tiles; tile_id += num_warps)
  {
    base_row = tile_id * ChunkSize;
    local_row = base_row + lid;
    global_row = pipeline_off_st + local_row;
    local_rank = static_cast<uint32_t>(local_row & 31);

    uint32_t tile = fast_load(select_bool_vec_ + (base_row >> 5));
    bool lane_active = local_row < pipeline_num_rows && lid < ChunkSize && (tile & (1u << local_rank));
    uint32_t ballot_mask = __ballot_sync(FULL_MASK, lane_active);
    if (ballot_mask == 0)
      continue;

    local_key_mapped = UINT32_MAX;
    local_pos = UINT32_MAX;
    local_st = UINT32_MAX;
    local_ed = UINT32_MAX;

    if (lane_active)
    {
      current_pos = global_row;
      for (int col_id = static_cast<int>(num_cols) - 1; col_id > static_cast<int>(join_col); --col_id)
        current_pos = load_joininfo<uint64_t>(parent_info_arrays_[col_id], current_pos);
      local_key_mapped = load_joininfo<uint32_t>(data_info_arrays_[join_col], current_pos);

      if (local_key_mapped != UINT32_MAX)
      {
        local_pos = hash_lookup(num_buckets, edge_offset, directed_edge_idx, local_key_mapped);
        if (local_pos != UINT32_MAX)
          fast_load_pair(edge_offsets_ + local_pos, local_st, local_ed);
      }
    }
    __syncwarp();

    if (local_st != UINT32_MAX)
    {
      uint32_t num_values = local_ed - local_st;
      d_num_res_for_each[global_row] = num_values;
      d_st_array[global_row] = local_st;
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

__global__ void
join_write_refer_kernel_nodup_pipeline(
    // trie_inter
    uint64_t frag_offset, uint64_t num_valid_rows_pipeline,
    uint64_t *valid_row_ids,
    // trie_edge
    vtype *edge_value_array_,
    uint32_t *d_selected_count, uint32_t *d_st_array,
    // offsets and output
    uint64_t *d_offsets_of_,
    uint32_t *d_res, uint64_t *d_parent_res,
    // options
    uint64_t result_offset)
{
  const uint32_t tid = threadIdx.x;
  const uint32_t idx = tid + blockIdx.x * blockDim.x;
  const uint32_t lid = tid & 31;
  const uint32_t wid_g = idx >> 5;
  const uint32_t num_warps = (blockDim.x >> 5) * gridDim.x;

  uint64_t row_idx = frag_offset + wid_g;
  while (row_idx < frag_offset + num_valid_rows_pipeline)
  {
    uint64_t row_id = fast_load(valid_row_ids + row_idx);
    uint32_t base_offset = fast_load(d_st_array + row_id);
    uint32_t num_values = fast_load(d_selected_count + row_idx);
    uint64_t res_offset = fast_load(d_offsets_of_ + row_idx);
    uint64_t local_offset = res_offset - result_offset;

    for (uint64_t i = lid; i < num_values; i += warpSize)
    {
      uint64_t out_idx = local_offset + i;
      d_res[out_idx] = fast_load(edge_value_array_ + base_offset + i);
      d_parent_res[out_idx] = row_id;
    }
    row_idx += num_warps;
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
    vtype *edge_key_array_, vtype *edge_value_array_, uint32_t *edge_offsets_,
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

__global__ void
join_refer_count_dup_kernel_2(
    // data trie
    vtype **vertex_arrays_, uint64_t **parent_pointers_,
    // mask
    uint32_t *select_bool_vec_, uint64_t num_rows,
    uint32_t num_cols, uint32_t join_col,
    // trie_edge
    vtype *edge_value_array_, uint32_t *edge_offsets_,
    // hash
    uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx,
    // mask
    uint32_t *value_mask, uint32_t mask_length, uint32_t *d_st_array,
    // options
    int dup_check_type, uint32_t num_dup_cols, uint32_t *dup_cols, uint32_t up_to_col)
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
        for (int col_id = num_cols - 1; col_id > up_to_col; --col_id)
        {
          if (vertex_arrays_[col_id] == nullptr)
            current_row_values[col_id] = UINT32_MAX;
          else
            current_row_values[col_id] = fast_load(vertex_arrays_[col_id] + current_pos);
          current_pos = fast_load(parent_pointers_[col_id] + current_pos);
        }
        if (vertex_arrays_[up_to_col] == nullptr)
          current_row_values[up_to_col] = UINT32_MAX;
        else
          current_row_values[up_to_col] = fast_load(vertex_arrays_[up_to_col] + current_pos);

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

__global__ void
join_refer_count_dup_kernel_2_pipeline_info(
    // data trie
    JoinInfo<uint32_t> *data_info_arrays_, JoinInfo<uint64_t> *parent_info_arrays_,
    // selection mask (windowed)
    uint32_t *select_bool_vec_,
    uint64_t pipeline_off_st, uint64_t pipeline_num_rows,
    uint32_t num_cols, uint32_t join_col,
    // trie_edge
    vtype *edge_value_array_, uint32_t *edge_offsets_,
    // hash
    uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx,
    // mask
    uint32_t *value_mask, uint32_t mask_length, uint32_t *d_st_array,
    // options
    int dup_check_type, uint32_t num_dup_cols, uint32_t *dup_cols, uint32_t up_to_col)
{
  const uint32_t tid = threadIdx.x;
  const uint32_t bid = blockIdx.x;
  const uint32_t idx = tid + bid * blockDim.x;
  const uint32_t lid = tid & 31;
  const uint32_t wid_g = idx >> 5;
  const uint32_t grid_size = blockDim.x * gridDim.x;

  uint64_t base_row_idx = pipeline_off_st + (static_cast<uint64_t>(wid_g) << 5);
  vtype current_row_values[MAX_VQ];
  vtype key_mapped;
  uint32_t pos, my_st, my_ed;

  while (base_row_idx < pipeline_off_st + pipeline_num_rows)
  {
    uint64_t tile_idx = (base_row_idx - pipeline_off_st) >> 5;
    uint32_t row_selected = fast_load(select_bool_vec_ + tile_idx);

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

    if (my_row < pipeline_off_st + pipeline_num_rows)
    {
      bool row_selected_bit = (row_selected & (1u << lid)) != 0u;
      if (row_selected_bit)
      {
        uint64_t current_pos = my_row;
        for (int col_id = static_cast<int>(num_cols) - 1; col_id > static_cast<int>(up_to_col); --col_id)
        {
          current_row_values[col_id] = load_joininfo<uint32_t>(data_info_arrays_[col_id], current_pos);
          current_pos = load_joininfo<uint64_t>(parent_info_arrays_[col_id], current_pos);
        }
        current_row_values[up_to_col] = load_joininfo<uint32_t>(data_info_arrays_[up_to_col], current_pos);

        key_mapped = current_row_values[join_col];

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

    for (uint32_t lane = 0; lane < warpSize; ++lane)
    {
      uint32_t bcast_st = __shfl_sync(FULL_MASK, my_st, lane);
      if (bcast_st == UINT32_MAX)
        continue;

      uint32_t bcast_ed = __shfl_sync(FULL_MASK, my_ed, lane);
      uint32_t num_values = bcast_ed - bcast_st;
      uint32_t num_value_tiles = (num_values + 31) >> 5;

      vtype bcast_row_values[MAX_VQ];
      if (dup_check_type == 1)
      {
        for (uint32_t i = 0; i < num_dup_cols; ++i)
          bcast_row_values[i] = __shfl_sync(FULL_MASK, current_row_values[dup_cols[i]], lane);
      }
      else
      {
        for (uint32_t i = 0; i < num_cols; ++i)
          bcast_row_values[i] = __shfl_sync(FULL_MASK, current_row_values[i], lane);
      }

      for (uint32_t value_tile = 0; value_tile < num_value_tiles; ++value_tile)
      {
        uint32_t value_idx = value_tile * warpSize + lid;
        bool found = false;

        if (value_idx < num_values)
        {
          vtype edge_value = fast_load(edge_value_array_ + bcast_st + value_idx);

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

__global__ void
join_refer_count_dup_nomask_kernel_2(
    // data trie
    vtype **vertex_arrays_, uint64_t **parent_pointers_,
    // mask
    uint32_t *select_bool_vec_, uint64_t num_rows,
    uint32_t num_cols, uint32_t join_col,
    // trie_edge
    vtype *edge_value_array_, uint32_t *edge_offsets_,
    // hash
    uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx,
    // outputs
    uint32_t *d_num_res_for_each, uint32_t *d_st_array,
    // options
    int dup_check_type, uint32_t num_dup_cols, uint32_t *dup_cols, uint32_t up_to_col)
{
  const uint32_t tid = threadIdx.x;
  const uint32_t bid = blockIdx.x;
  const uint32_t idx = tid + bid * blockDim.x;
  const uint32_t lid = tid & 31;
  const uint32_t wid_g = idx >> 5;
  const uint32_t grid_size = blockDim.x * gridDim.x;

  uint32_t row_selected;
  uint64_t base_row_idx = static_cast<uint64_t>(wid_g) << 5;
  vtype current_row_values[MAX_VQ];
  uint32_t pos, my_st, my_ed;

  while (base_row_idx < num_rows)
  {
    row_selected = select_bool_vec_[base_row_idx >> 5];
    if (!row_selected)
    {
      base_row_idx += grid_size;
      continue;
    }

    uint64_t my_row = base_row_idx + lid;
    vtype key_mapped = UINT32_MAX;
    pos = UINT32_MAX;
    my_st = UINT32_MAX;
    my_ed = UINT32_MAX;

    if (my_row < num_rows)
    {
      bool row_selected_bit = (row_selected & (1u << lid)) != 0u;
      if (row_selected_bit)
      {
        uint64_t current_pos = my_row;
        for (int col_id = static_cast<int>(num_cols) - 1; col_id > static_cast<int>(up_to_col); --col_id)
        {
          if (vertex_arrays_[col_id] == nullptr)
            current_row_values[col_id] = UINT32_MAX;
          else
            current_row_values[col_id] = fast_load(vertex_arrays_[col_id] + current_pos);
          current_pos = fast_load(parent_pointers_[col_id] + current_pos);
        }
        if (vertex_arrays_[up_to_col] == nullptr)
          current_row_values[up_to_col] = UINT32_MAX;
        else
          current_row_values[up_to_col] = fast_load(vertex_arrays_[up_to_col] + current_pos);

        key_mapped = current_row_values[join_col];
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

    for (uint32_t lane = 0; lane < warpSize; ++lane)
    {
      uint32_t bcast_st = __shfl_sync(FULL_MASK, my_st, lane);
      if (bcast_st == UINT32_MAX)
        continue;

      uint32_t bcast_ed = __shfl_sync(FULL_MASK, my_ed, lane);
      uint32_t num_values = bcast_ed - bcast_st;
      uint32_t num_value_tiles = (num_values + 31) >> 5;

      vtype bcast_row_values[MAX_VQ];
      if (dup_check_type == 1)
      {
        for (uint32_t i = 0; i < num_dup_cols; ++i)
          bcast_row_values[i] = __shfl_sync(FULL_MASK, current_row_values[dup_cols[i]], lane);
      }
      else
      {
        for (uint32_t i = 0; i < num_cols; ++i)
          bcast_row_values[i] = __shfl_sync(FULL_MASK, current_row_values[i], lane);
      }

      uint32_t row_count = 0;
      for (uint32_t value_tile = 0; value_tile < num_value_tiles; ++value_tile)
      {
        uint32_t value_idx = value_tile * warpSize + lid;
        bool found = false;

        if (value_idx < num_values)
        {
          vtype edge_value = fast_load(edge_value_array_ + bcast_st + value_idx);

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
          else
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
          row_count += __popc(found_mask);
        __syncwarp();
      }

      if (lid == 0)
      {
        uint64_t row_idx = base_row_idx + lane;
        d_num_res_for_each[row_idx] = row_count;
      }
      __syncwarp();
    }

    base_row_idx += grid_size;
  }
}

__global__ void
join_refer_count_dup_nomask_kernel_2_pipeline_info(
    // data trie
    JoinInfo<uint32_t> *data_info_arrays_, JoinInfo<uint64_t> *parent_info_arrays_,
    // selection mask (windowed)
    uint32_t *select_bool_vec_,
    uint64_t pipeline_off_st, uint64_t pipeline_num_rows,
    uint32_t num_cols, uint32_t join_col,
    // trie_edge
    vtype *edge_value_array_, uint32_t *edge_offsets_,
    // hash
    uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx,
    // outputs
    uint32_t *d_num_res_for_each, uint32_t *d_st_array,
    // options
    int dup_check_type, uint32_t num_dup_cols, uint32_t *dup_cols, uint32_t up_to_col)
{
  const uint32_t tid = threadIdx.x;
  const uint32_t bid = blockIdx.x;
  const uint32_t idx = tid + bid * blockDim.x;
  const uint32_t lid = tid & 31;
  const uint32_t wid_g = idx >> 5;
  const uint32_t grid_size = blockDim.x * gridDim.x;

  uint64_t base_row_idx = pipeline_off_st + (static_cast<uint64_t>(wid_g) << 5);
  vtype current_row_values[MAX_VQ];
  uint32_t pos, my_st, my_ed;

  while (base_row_idx < pipeline_off_st + pipeline_num_rows)
  {
    uint64_t tile_idx = (base_row_idx - pipeline_off_st) >> 5;
    uint32_t row_selected = fast_load(select_bool_vec_ + tile_idx);

    if (!row_selected)
    {
      base_row_idx += grid_size;
      continue;
    }

    uint64_t my_row = base_row_idx + lid;
    vtype key_mapped = UINT32_MAX;
    pos = UINT32_MAX;
    my_st = UINT32_MAX;
    my_ed = UINT32_MAX;

    if (my_row < pipeline_off_st + pipeline_num_rows)
    {
      bool row_selected_bit = (row_selected & (1u << lid)) != 0u;
      if (row_selected_bit)
      {
        uint64_t current_pos = my_row;
        for (int col_id = static_cast<int>(num_cols) - 1; col_id > static_cast<int>(up_to_col); --col_id)
        {
          current_row_values[col_id] = load_joininfo<uint32_t>(data_info_arrays_[col_id], current_pos);
          current_pos = load_joininfo<uint64_t>(parent_info_arrays_[col_id], current_pos);
        }
        current_row_values[up_to_col] = load_joininfo<uint32_t>(data_info_arrays_[up_to_col], current_pos);

        key_mapped = current_row_values[join_col];
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

    for (uint32_t lane = 0; lane < warpSize; ++lane)
    {
      uint32_t bcast_st = __shfl_sync(FULL_MASK, my_st, lane);
      if (bcast_st == UINT32_MAX)
        continue;

      uint32_t bcast_ed = __shfl_sync(FULL_MASK, my_ed, lane);
      uint32_t num_values = bcast_ed - bcast_st;
      uint32_t num_value_tiles = (num_values + 31) >> 5;

      vtype bcast_row_values[MAX_VQ];
      if (dup_check_type == 1)
      {
        for (uint32_t i = 0; i < num_dup_cols; ++i)
          bcast_row_values[i] = __shfl_sync(FULL_MASK, current_row_values[dup_cols[i]], lane);
      }
      else
      {
        for (uint32_t i = 0; i < num_cols; ++i)
          bcast_row_values[i] = __shfl_sync(FULL_MASK, current_row_values[i], lane);
      }

      uint32_t row_count = 0;
      for (uint32_t value_tile = 0; value_tile < num_value_tiles; ++value_tile)
      {
        uint32_t value_idx = value_tile * warpSize + lid;
        bool found = false;

        if (value_idx < num_values)
        {
          vtype edge_value = fast_load(edge_value_array_ + bcast_st + value_idx);

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
          else
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
          row_count += __popc(found_mask);
        __syncwarp();
      }

      if (lid == 0)
      {
        uint64_t row_idx = base_row_idx + lane;
        d_num_res_for_each[row_idx] = row_count;
      }
      __syncwarp();
    }

    base_row_idx += grid_size;
  }
}

__global__ void
join_refer_write_dup_nomask_pipeline(
    // row range
    uint64_t frag_offset, uint64_t num_frag_rows_pipeline,
    // data trie
    vtype **vertex_arrays_, uint64_t **parent_pointers_,
    uint32_t num_cols, uint32_t join_col,
    // trie_edge
    vtype *edge_value_array_, uint32_t *edge_offsets_,
    // hash
    uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx,
    // offsets and output
    uint64_t *d_row_offsets,
    uint32_t *d_res, uint64_t *d_parent_res, uint64_t result_offset,
    // options
    int dup_check_type, uint32_t num_dup_cols, uint32_t *dup_cols, uint32_t up_to_col)
{
  const uint32_t tid = threadIdx.x;
  const uint32_t bid = blockIdx.x;
  const uint32_t idx = tid + bid * blockDim.x;
  const uint32_t lid = tid & 31;
  const uint32_t wid_g = idx >> 5;
  const uint32_t grid_size = blockDim.x * gridDim.x;

  uint64_t base_row_idx = frag_offset + (static_cast<uint64_t>(wid_g) << 5);
  vtype current_row_values[MAX_VQ];
  uint32_t pos, my_st, my_ed;
  uint64_t my_res_offset;

  while (base_row_idx < frag_offset + num_frag_rows_pipeline)
  {
    uint64_t my_row = base_row_idx + lid;
    vtype key_mapped = UINT32_MAX;
    pos = UINT32_MAX;
    my_st = UINT32_MAX;
    my_ed = UINT32_MAX;
    my_res_offset = 0;

    if (my_row < frag_offset + num_frag_rows_pipeline)
    {
      uint64_t res_offset = fast_load(d_row_offsets + my_row);
      uint64_t num_res_row = fast_load(d_row_offsets + my_row + 1) - res_offset;
      if (num_res_row)
      {
        my_res_offset = res_offset;

        uint64_t current_pos = my_row;
        for (int col_id = static_cast<int>(num_cols) - 1; col_id > static_cast<int>(up_to_col); --col_id)
        {
          if (vertex_arrays_[col_id] == nullptr)
            current_row_values[col_id] = UINT32_MAX;
          else
            current_row_values[col_id] = fast_load(vertex_arrays_[col_id] + current_pos);
          current_pos = fast_load(parent_pointers_[col_id] + current_pos);
        }
        if (vertex_arrays_[up_to_col] == nullptr)
          current_row_values[up_to_col] = UINT32_MAX;
        else
          current_row_values[up_to_col] = fast_load(vertex_arrays_[up_to_col] + current_pos);

        key_mapped = current_row_values[join_col];
        if (key_mapped != UINT32_MAX)
        {
          pos = hash_lookup(num_buckets, edge_offset, directed_edge_idx, key_mapped);
          if (pos != UINT32_MAX)
            fast_load_pair(edge_offsets_ + pos, my_st, my_ed);
        }
      }
    }
    __syncwarp();

    for (uint32_t lane = 0; lane < warpSize; ++lane)
    {
      uint32_t bcast_st = __shfl_sync(FULL_MASK, my_st, lane);
      if (bcast_st == UINT32_MAX)
        continue;

      uint64_t row_idx = base_row_idx + lane;
      uint32_t bcast_ed = __shfl_sync(FULL_MASK, my_ed, lane);
      uint32_t res_offset_lo = __shfl_sync(FULL_MASK, static_cast<uint32_t>(my_res_offset), lane);
      uint32_t res_offset_hi = __shfl_sync(FULL_MASK, static_cast<uint32_t>(my_res_offset >> 32), lane);
      uint64_t bcast_res_offset = (static_cast<uint64_t>(res_offset_hi) << 32) | res_offset_lo;
      uint32_t num_values = bcast_ed - bcast_st;
      uint32_t num_value_tiles = (num_values + 31) >> 5;

      vtype bcast_row_values[MAX_VQ];
      if (dup_check_type == 1)
      {
        for (uint32_t i = 0; i < num_dup_cols; ++i)
          bcast_row_values[i] = __shfl_sync(FULL_MASK, current_row_values[dup_cols[i]], lane);
      }
      else
      {
        for (uint32_t i = 0; i < num_cols; ++i)
          bcast_row_values[i] = __shfl_sync(FULL_MASK, current_row_values[i], lane);
      }

      uint32_t warp_pos = 0;
      for (uint32_t value_tile = 0; value_tile < num_value_tiles; ++value_tile)
      {
        uint32_t value_idx = value_tile * warpSize + lid;
        bool found = false;
        vtype edge_value = UINT32_MAX;

        if (value_idx < num_values)
        {
          edge_value = fast_load(edge_value_array_ + bcast_st + value_idx);

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
          else
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
        if (found_mask)
        {
          uint32_t base_index = __shfl_sync(FULL_MASK, warp_pos, 0);
          if (found)
          {
            uint32_t index_in_tile = __popc(found_mask & ((1u << lid) - 1));
            uint64_t res_idx = bcast_res_offset + base_index + index_in_tile - result_offset;
            d_res[res_idx] = edge_value;
            d_parent_res[res_idx] = row_idx;
          }
          if (lid == 0)
            warp_pos += __popc(found_mask);
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
    vtype *edge_key_array_, vtype *edge_value_array_, uint32_t *edge_offsets_,
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

template __global__ void join_count_refer_kernel_nodup<1>(vtype **vertex_arrays_, uint64_t **parent_pointers_, uint64_t num_rows, uint32_t num_cols, uint32_t join_col, uint32_t *select_bool_vec_, vtype *edge_key_array_, vtype *edge_value_array_, uint32_t *edge_offsets_, uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx, uint32_t *d_num_res_for_each, uint32_t *d_st_array);
template __global__ void join_count_refer_kernel_nodup<2>(vtype **vertex_arrays_, uint64_t **parent_pointers_, uint64_t num_rows, uint32_t num_cols, uint32_t join_col, uint32_t *select_bool_vec_, vtype *edge_key_array_, vtype *edge_value_array_, uint32_t *edge_offsets_, uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx, uint32_t *d_num_res_for_each, uint32_t *d_st_array);
template __global__ void join_count_refer_kernel_nodup<4>(vtype **vertex_arrays_, uint64_t **parent_pointers_, uint64_t num_rows, uint32_t num_cols, uint32_t join_col, uint32_t *select_bool_vec_, vtype *edge_key_array_, vtype *edge_value_array_, uint32_t *edge_offsets_, uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx, uint32_t *d_num_res_for_each, uint32_t *d_st_array);
template __global__ void join_count_refer_kernel_nodup<8>(vtype **vertex_arrays_, uint64_t **parent_pointers_, uint64_t num_rows, uint32_t num_cols, uint32_t join_col, uint32_t *select_bool_vec_, vtype *edge_key_array_, vtype *edge_value_array_, uint32_t *edge_offsets_, uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx, uint32_t *d_num_res_for_each, uint32_t *d_st_array);
template __global__ void join_count_refer_kernel_nodup<16>(vtype **vertex_arrays_, uint64_t **parent_pointers_, uint64_t num_rows, uint32_t num_cols, uint32_t join_col, uint32_t *select_bool_vec_, vtype *edge_key_array_, vtype *edge_value_array_, uint32_t *edge_offsets_, uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx, uint32_t *d_num_res_for_each, uint32_t *d_st_array);
template __global__ void join_count_refer_kernel_nodup<32>(vtype **vertex_arrays_, uint64_t **parent_pointers_, uint64_t num_rows, uint32_t num_cols, uint32_t join_col, uint32_t *select_bool_vec_, vtype *edge_key_array_, vtype *edge_value_array_, uint32_t *edge_offsets_, uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx, uint32_t *d_num_res_for_each, uint32_t *d_st_array);

template __global__ void join_count_refer_kernel_nodup_pipeline_info<1>(JoinInfo<uint32_t> *data_info_arrays_, JoinInfo<uint64_t> *parent_info_arrays_, uint32_t *select_bool_vec_, uint64_t pipeline_off_st, uint64_t pipeline_num_rows, uint32_t num_cols, uint32_t join_col, vtype *edge_key_array_, vtype *edge_value_array_, uint32_t *edge_offsets_, uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx, uint32_t *d_num_res_for_each, uint32_t *d_st_array);
template __global__ void join_count_refer_kernel_nodup_pipeline_info<2>(JoinInfo<uint32_t> *data_info_arrays_, JoinInfo<uint64_t> *parent_info_arrays_, uint32_t *select_bool_vec_, uint64_t pipeline_off_st, uint64_t pipeline_num_rows, uint32_t num_cols, uint32_t join_col, vtype *edge_key_array_, vtype *edge_value_array_, uint32_t *edge_offsets_, uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx, uint32_t *d_num_res_for_each, uint32_t *d_st_array);
template __global__ void join_count_refer_kernel_nodup_pipeline_info<4>(JoinInfo<uint32_t> *data_info_arrays_, JoinInfo<uint64_t> *parent_info_arrays_, uint32_t *select_bool_vec_, uint64_t pipeline_off_st, uint64_t pipeline_num_rows, uint32_t num_cols, uint32_t join_col, vtype *edge_key_array_, vtype *edge_value_array_, uint32_t *edge_offsets_, uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx, uint32_t *d_num_res_for_each, uint32_t *d_st_array);
template __global__ void join_count_refer_kernel_nodup_pipeline_info<8>(JoinInfo<uint32_t> *data_info_arrays_, JoinInfo<uint64_t> *parent_info_arrays_, uint32_t *select_bool_vec_, uint64_t pipeline_off_st, uint64_t pipeline_num_rows, uint32_t num_cols, uint32_t join_col, vtype *edge_key_array_, vtype *edge_value_array_, uint32_t *edge_offsets_, uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx, uint32_t *d_num_res_for_each, uint32_t *d_st_array);
template __global__ void join_count_refer_kernel_nodup_pipeline_info<16>(JoinInfo<uint32_t> *data_info_arrays_, JoinInfo<uint64_t> *parent_info_arrays_, uint32_t *select_bool_vec_, uint64_t pipeline_off_st, uint64_t pipeline_num_rows, uint32_t num_cols, uint32_t join_col, vtype *edge_key_array_, vtype *edge_value_array_, uint32_t *edge_offsets_, uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx, uint32_t *d_num_res_for_each, uint32_t *d_st_array);
template __global__ void join_count_refer_kernel_nodup_pipeline_info<32>(JoinInfo<uint32_t> *data_info_arrays_, JoinInfo<uint64_t> *parent_info_arrays_, uint32_t *select_bool_vec_, uint64_t pipeline_off_st, uint64_t pipeline_num_rows, uint32_t num_cols, uint32_t join_col, vtype *edge_key_array_, vtype *edge_value_array_, uint32_t *edge_offsets_, uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx, uint32_t *d_num_res_for_each, uint32_t *d_st_array);

template __global__ void join_write_refer_kernel_nodup<1>(uint64_t *valid_row_ids, uint64_t num_valid_rows, vtype *edge_value_array_, uint32_t *d_selected_count, uint32_t *d_st_array, uint32_t *d_res, uint64_t *d_offsets_of_, uint64_t *d_parent_res);
template __global__ void join_write_refer_kernel_nodup<2>(uint64_t *valid_row_ids, uint64_t num_valid_rows, vtype *edge_value_array_, uint32_t *d_selected_count, uint32_t *d_st_array, uint32_t *d_res, uint64_t *d_offsets_of_, uint64_t *d_parent_res);
template __global__ void join_write_refer_kernel_nodup<4>(uint64_t *valid_row_ids, uint64_t num_valid_rows, vtype *edge_value_array_, uint32_t *d_selected_count, uint32_t *d_st_array, uint32_t *d_res, uint64_t *d_offsets_of_, uint64_t *d_parent_res);
template __global__ void join_write_refer_kernel_nodup<8>(uint64_t *valid_row_ids, uint64_t num_valid_rows, vtype *edge_value_array_, uint32_t *d_selected_count, uint32_t *d_st_array, uint32_t *d_res, uint64_t *d_offsets_of_, uint64_t *d_parent_res);
template __global__ void join_write_refer_kernel_nodup<16>(uint64_t *valid_row_ids, uint64_t num_valid_rows, vtype *edge_value_array_, uint32_t *d_selected_count, uint32_t *d_st_array, uint32_t *d_res, uint64_t *d_offsets_of_, uint64_t *d_parent_res);
template __global__ void join_write_refer_kernel_nodup<32>(uint64_t *valid_row_ids, uint64_t num_valid_rows, vtype *edge_value_array_, uint32_t *d_selected_count, uint32_t *d_st_array, uint32_t *d_res, uint64_t *d_offsets_of_, uint64_t *d_parent_res);
