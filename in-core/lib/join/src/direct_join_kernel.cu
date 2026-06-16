#include "direct_join_kernel.cuh"
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

template <int ChunkSize>
__global__ void
join_count_kernel_nodup_pipeline_info(
    // trie_inter
    JoinInfo<uint32_t> *data_info_arrays_, JoinInfo<uint64_t> *parent_info_arrays_,
    uint64_t pipeline_off_st, uint64_t pipeline_num_rows,
    uint32_t num_cols, uint32_t join_col,
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
  const uint32_t lid = tid & 31;
  const uint32_t wid_g = idx >> 5;
  const uint32_t num_warps = (blockDim.x >> 5) * gridDim.x;

  vtype local_key_mapped;
  uint32_t local_st, local_ed;
  uint32_t local_pos;
  uint64_t base_row;
  uint64_t local_row_idx;
  uint64_t global_row_idx;
  uint64_t current_pos;

  const uint64_t num_tiles = (pipeline_num_rows + ChunkSize - 1) / ChunkSize;

  for (uint64_t tile_id = wid_g; tile_id < num_tiles; tile_id += num_warps)
  {
    base_row = tile_id * ChunkSize;
    local_row_idx = base_row + lid;
    global_row_idx = pipeline_off_st + local_row_idx;

    local_key_mapped = UINT32_MAX;
    local_pos = UINT32_MAX;
    local_st = UINT32_MAX;
    local_ed = UINT32_MAX;

    if (lid < ChunkSize)
    {
      if (local_row_idx < pipeline_num_rows)
      {
        current_pos = global_row_idx;

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
    }
    __syncwarp();

    if (local_st != UINT32_MAX)
    {
      uint32_t num_values = local_ed - local_st;
      d_num_res_for_each[global_row_idx] = num_values;
      d_st_array[global_row_idx] = local_st;
    }
    __syncwarp();
  }
}

template <int ChunkSize>
__global__ void
join_count_kernel_nodup_pipeline_local(
    // trie_inter (JoinInfo supports partial staging)
    JoinInfo<uint32_t> *data_info_arrays_, JoinInfo<uint64_t> *parent_info_arrays_,
    uint64_t pipeline_off_st, uint64_t pipeline_num_rows,
    uint32_t num_cols, uint32_t join_col,
    // trie_edge
    vtype *edge_key_array_, vtype *edge_value_array_, uint32_t *edge_offsets_,
    // hash
    uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx,
    // outputs (chunk-local)
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
  uint64_t local_row_idx;
  uint64_t global_row_idx;
  uint64_t current_pos;

  const uint64_t num_tiles = (pipeline_num_rows + ChunkSize - 1) / ChunkSize;

  for (uint64_t tile_id = wid_g; tile_id < num_tiles; tile_id += num_warps)
  {
    base_row = tile_id * ChunkSize;
    local_row_idx = base_row + lid;
    global_row_idx = pipeline_off_st + local_row_idx;

    local_key_mapped = UINT32_MAX;
    local_pos = UINT32_MAX;
    local_st = UINT32_MAX;
    local_ed = UINT32_MAX;

    if (lid < ChunkSize)
    {
      if (local_row_idx < pipeline_num_rows)
      {
        current_pos = global_row_idx;

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
    }
    __syncwarp();

    if (local_st != UINT32_MAX)
    {
      uint32_t num_values = local_ed - local_st;
      d_num_res_for_each[local_row_idx] = num_values;
      d_st_array[local_row_idx] = local_st;
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

__global__ void
join_count_kernel_dup_pipeline_info(
    // trie_inter
    JoinInfo<uint32_t> *data_info_arrays_, JoinInfo<uint64_t> *parent_info_arrays_,
    uint64_t pipeline_off_st, uint64_t pipeline_num_rows,
    uint32_t num_cols, uint32_t join_col,
    // trie_edge
    vtype *edge_key_array_, vtype *edge_value_array_, uint32_t *edge_offsets_,
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
  const uint32_t num_warps = (blockDim.x >> 5) * gridDim.x;

  vtype local_key_mapped;
  uint32_t local_st, local_ed;
  uint32_t local_pos;
  vtype local_row_values[MAX_VQ];
  vtype bcast_row_values[MAX_VQ];

  vtype bcast_key;
  uint32_t bcast_st, bcast_ed;

  constexpr uint32_t ROWS_PER_TILE = 4;
  uint64_t num_tiles = (pipeline_num_rows + ROWS_PER_TILE - 1) / ROWS_PER_TILE;

  for (uint64_t tile_id = wid_g; tile_id < num_tiles; tile_id += num_warps)
  {
    uint64_t base_row_local = tile_id * ROWS_PER_TILE;
    if (lid < ROWS_PER_TILE)
    {
      uint64_t local_row = base_row_local + lid;
      uint64_t global_row = pipeline_off_st + local_row;
      local_key_mapped = UINT32_MAX;
      local_pos = UINT32_MAX;
      local_st = UINT32_MAX;
      local_ed = UINT32_MAX;

      if (local_row < pipeline_num_rows)
      {
        uint64_t current_pos = global_row;

        for (int col_id = static_cast<int>(num_cols) - 1; col_id > static_cast<int>(up_to_col); --col_id)
        {
          local_row_values[col_id] = load_joininfo<uint32_t>(data_info_arrays_[col_id], current_pos);
          current_pos = load_joininfo<uint64_t>(parent_info_arrays_[col_id], current_pos);
        }
        local_row_values[up_to_col] = load_joininfo<uint32_t>(data_info_arrays_[up_to_col], current_pos);
        local_key_mapped = local_row_values[join_col];

        local_pos = hash_lookup(num_buckets, edge_offset, directed_edge_idx, local_key_mapped);
        if (local_pos != UINT32_MAX)
        {
          fast_load_pair(edge_offsets_ + local_pos, local_st, local_ed);
          d_st_array[global_row] = local_st;
        }
      }
    }
    __syncwarp();

    for (uint32_t row_in_tile = 0; row_in_tile < ROWS_PER_TILE; ++row_in_tile)
    {
      uint64_t local_row = base_row_local + row_in_tile;
      if (local_row >= pipeline_num_rows)
        break;

      uint64_t global_row = pipeline_off_st + local_row;

      bcast_st = __shfl_sync(FULL_MASK, local_st, row_in_tile);
      if (bcast_st == UINT32_MAX)
        continue;

      bcast_key = __shfl_sync(FULL_MASK, local_key_mapped, row_in_tile);
      bcast_ed = __shfl_sync(FULL_MASK, local_ed, row_in_tile);

      bool found = false;
      uint32_t num_values = bcast_ed - bcast_st;
      uint32_t num_value_tiles = (num_values + 31) >> 5;

      if (dup_check_type == 1)
        for (int i = 0; i < static_cast<int>(num_dup_cols); ++i)
          bcast_row_values[i] = __shfl_sync(FULL_MASK, local_row_values[dup_cols[i]], row_in_tile);
      else
        for (int i = 0; i < static_cast<int>(num_cols); ++i)
          bcast_row_values[i] = __shfl_sync(FULL_MASK, local_row_values[i], row_in_tile);

      for (uint32_t value_tile = 0; value_tile < num_value_tiles; ++value_tile)
      {
        uint32_t value_idx = value_tile * warpSize + lid;
        found = false;

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
          value_mask[global_row * mask_length + value_tile] = found_mask;
        __syncwarp();
      }
    }
  }
}

// dup aware
__global__ void
join_count_kernel_dup_2(
    // trie_inter
    vtype **vertex_arrays_, uint64_t **parent_pointers_,
    uint64_t num_rows, uint32_t num_cols, uint32_t join_col,
    // trie_edge
    vtype *edge_key_array_, vtype *edge_value_array_, uint32_t *edge_offsets_,
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
        for (int col_id = num_cols - 1; col_id > (int)up_to_col; --col_id)
        {
          if (vertex_arrays_[col_id] == nullptr)
            local_row_values[col_id] = UINT32_MAX;
          else
            local_row_values[col_id] = fast_load(vertex_arrays_[col_id] + current_pos);
          current_pos = fast_load(parent_pointers_[col_id] + current_pos);
        }
        if (vertex_arrays_[up_to_col] == nullptr)
          local_row_values[up_to_col] = UINT32_MAX;
        else
          local_row_values[up_to_col] = fast_load(vertex_arrays_[up_to_col] + current_pos); // loop peeling.
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

  uint32_t local_st, local_num_values;
  uint32_t bcast_st, bcast_num_values;
  uint64_t local_offset, bcast_offset;

  for (uint64_t tile_id = wid_g; tile_id < num_tiles; tile_id += num_warps)
  {
    uint64_t base_row = tile_id * ChunkSize;
    uint64_t my_row = base_row + lid;

    // Lanes 0~`ChunkSize-1` fetch data for their respective rows simultaneously
    local_st = UINT32_MAX;
    local_num_values = 0;
    local_offset = 0;

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
      uint32_t bcast_offset_lo = __shfl_sync(FULL_MASK, static_cast<uint32_t>(local_offset), i);
      uint32_t bcast_offset_hi = __shfl_sync(FULL_MASK, static_cast<uint32_t>(local_offset >> 32), i);
      bcast_offset = (static_cast<uint64_t>(bcast_offset_hi) << 32) | bcast_offset_lo;

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

__global__ void
join_write_kernel_dup_pipeline(
    // trie_inter
    uint64_t frag_offset, uint64_t num_frag_rows_pipeline,
    // trie_edge
    vtype *edge_value_array_,
    uint32_t *value_mask, uint32_t mask_length,
    uint64_t *d_offsets_of_, uint32_t *d_st_array, // these two resident in GPU.
    uint32_t *d_res, uint64_t *d_parent_res, uint64_t result_offset)
{
  const uint32_t tid = threadIdx.x;
  const uint32_t bid = blockIdx.x;
  const uint32_t idx = tid + bid * blockDim.x;
  const uint32_t wid = tid >> 5;
  const uint32_t lid = tid & 31;
  const uint32_t wid_g = idx >> 5;
  const uint32_t num_warps = (blockDim.x >> 5) * gridDim.x;

  uint64_t row_id = wid_g + frag_offset;
  uint32_t my_tile, bcast_tile, tile_base;
  const uint32_t num_loops = (mask_length + warpSize - 1) / warpSize;

  __shared__ uint32_t warp_pos[WARP_PER_BLOCK];
  __shared__ uint32_t task_queue[WARP_PER_BLOCK][WARP_SIZE];
  __shared__ uint32_t task_word_idx[WARP_PER_BLOCK][WARP_SIZE];
  __shared__ uint32_t num_tasks[WARP_PER_BLOCK];

  while (row_id < frag_offset + num_frag_rows_pipeline)
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
          uint64_t res_idx = res_offset + global_index - result_offset;
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

__global__ void
join_write_kernel_nodup_pipeline(
    // trie_inter
    uint64_t frag_offset, uint64_t num_frag_rows_pipeline,
    // trie_edge
    vtype *edge_value_array_,
    uint32_t *d_st_array, uint64_t *d_offsets_of_,
    uint32_t *d_res, uint64_t *d_parent_res, uint64_t result_offset)
{
  const uint32_t tid = threadIdx.x;
  const uint32_t idx = tid + blockIdx.x * blockDim.x;
  const uint32_t lid = tid & 31;
  const uint32_t wid_g = idx >> 5;
  const uint32_t num_warps = (blockDim.x >> 5) * gridDim.x;

  uint64_t row_id = frag_offset + wid_g;

  while (row_id < frag_offset + num_frag_rows_pipeline)
  {
    uint64_t res_offset = fast_load(d_offsets_of_ + row_id);
    uint64_t num_res_row = fast_load(d_offsets_of_ + row_id + 1) - res_offset;
    if (num_res_row)
    {
      uint32_t base_offset = fast_load(d_st_array + row_id);
      for (uint64_t i = lid; i < num_res_row; i += warpSize)
      {
        uint64_t res_idx = res_offset - result_offset + i;
        d_res[res_idx] = fast_load(edge_value_array_ + base_offset + i);
        d_parent_res[res_idx] = row_id;
      }
    }
    row_id += num_warps;
  }
}

__global__ void
join_write_kernel_nodup_pipeline_local(
    // row range (global row ids)
    uint64_t frag_offset, uint64_t num_frag_rows_pipeline,
    // trie_edge
    vtype *edge_value_array_,
    // chunk-local st and offsets for this fragment range
    uint32_t *d_st_array, uint64_t *d_row_offsets, uint64_t base_result_offset,
    // output buffers (0-based for this fragment range)
    uint32_t *d_res, uint64_t *d_parent_res)
{
  const uint32_t tid = threadIdx.x;
  const uint32_t idx = tid + blockIdx.x * blockDim.x;
  const uint32_t lid = tid & 31;
  const uint32_t wid_g = idx >> 5;
  const uint32_t num_warps = (blockDim.x >> 5) * gridDim.x;

  uint64_t row_id = frag_offset + wid_g;

  while (row_id < frag_offset + num_frag_rows_pipeline)
  {
    uint64_t local_row = row_id - frag_offset;
    uint64_t res_offset = fast_load(d_row_offsets + local_row) - base_result_offset;
    uint64_t num_res_row = fast_load(d_row_offsets + local_row + 1) - fast_load(d_row_offsets + local_row);
    if (num_res_row)
    {
      uint32_t base_edge_offset = fast_load(d_st_array + local_row);
      for (uint64_t i = lid; i < num_res_row; i += warpSize)
      {
        uint64_t res_idx = res_offset + i;
        d_res[res_idx] = fast_load(edge_value_array_ + base_edge_offset + i);
        d_parent_res[res_idx] = row_id;
      }
    }
    row_id += num_warps;
  }
}

__global__ void
join_count_kernel_dup_nomask_2(
    // trie_inter
    vtype **vertex_arrays_, uint64_t **parent_pointers_,
    uint64_t num_rows, uint32_t num_cols, uint32_t join_col,
    // trie_edge
    vtype *edge_key_array_, vtype *edge_value_array_, uint32_t *edge_offsets_,
    // hash
    uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx,
    // outputs
    uint32_t *d_num_res_for_each, uint32_t *d_st_array,
    // options
    int dup_check_type, uint32_t num_dup_cols, uint32_t *dup_cols, uint32_t up_to_col)
{
  const uint32_t tid = threadIdx.x;
  const uint32_t idx = tid + blockIdx.x * blockDim.x;
  const uint32_t lid = tid & 31;
  const uint32_t wid_g = idx >> 5;
  const uint32_t num_warps = (blockDim.x >> 5) * gridDim.x;

  vtype local_key_mapped;
  uint32_t local_st, local_ed;
  uint32_t local_pos;
  vtype local_row_values[MAX_VQ];
  vtype bcast_row_values[MAX_VQ];

  constexpr uint32_t ROWS_PER_TILE = 4;
  uint64_t num_tiles = (num_rows + ROWS_PER_TILE - 1) / ROWS_PER_TILE;

  for (uint64_t tile_id = wid_g; tile_id < num_tiles; tile_id += num_warps)
  {
    uint64_t base_row = tile_id * ROWS_PER_TILE;
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

        for (int col_id = static_cast<int>(num_cols) - 1; col_id > static_cast<int>(up_to_col); --col_id)
        {
          if (vertex_arrays_[col_id] == nullptr)
            local_row_values[col_id] = UINT32_MAX;
          else
            local_row_values[col_id] = fast_load(vertex_arrays_[col_id] + current_pos);
          current_pos = fast_load(parent_pointers_[col_id] + current_pos);
        }
        if (vertex_arrays_[up_to_col] == nullptr)
          local_row_values[up_to_col] = UINT32_MAX;
        else
          local_row_values[up_to_col] = fast_load(vertex_arrays_[up_to_col] + current_pos);

        local_key_mapped = local_row_values[join_col];

        local_pos = hash_lookup(num_buckets, edge_offset, directed_edge_idx, local_key_mapped);
        if (local_pos != UINT32_MAX)
        {
          fast_load_pair(edge_offsets_ + local_pos, local_st, local_ed);
          d_st_array[my_row_idx] = local_st;
        }
      }
    }
    __syncwarp();

    for (uint32_t row_in_tile = 0; row_in_tile < ROWS_PER_TILE; ++row_in_tile)
    {
      uint64_t current_row = base_row + row_in_tile;
      if (current_row >= num_rows)
        break;

      uint32_t bcast_st = __shfl_sync(FULL_MASK, local_st, row_in_tile);
      if (bcast_st == UINT32_MAX)
      {
        if (lid == 0)
          d_num_res_for_each[current_row] = 0;
        __syncwarp();
        continue;
      }

      uint32_t bcast_ed = __shfl_sync(FULL_MASK, local_ed, row_in_tile);
      uint32_t num_values = bcast_ed - bcast_st;
      uint32_t num_value_tiles = (num_values + 31) >> 5;
      uint32_t row_count = 0;

      if (dup_check_type == 1)
      {
        for (uint32_t i = 0; i < num_dup_cols; ++i)
          bcast_row_values[i] = __shfl_sync(FULL_MASK, local_row_values[dup_cols[i]], row_in_tile);
      }
      else
      {
        for (uint32_t i = 0; i < num_cols; ++i)
          bcast_row_values[i] = __shfl_sync(FULL_MASK, local_row_values[i], row_in_tile);
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
        d_num_res_for_each[current_row] = row_count;
      __syncwarp();
    }
  }
}

__global__ void
join_write_kernel_dup_nomask_pipeline(
    // row range
    uint64_t frag_offset, uint64_t num_frag_rows_pipeline,
    // trie_inter
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
  const uint32_t idx = tid + blockIdx.x * blockDim.x;
  const uint32_t lid = tid & 31;
  const uint32_t wid_g = idx >> 5;
  const uint32_t num_warps = (blockDim.x >> 5) * gridDim.x;
  const uint32_t grid_size = blockDim.x * gridDim.x;

  uint64_t base_row_idx = frag_offset + (static_cast<uint64_t>(wid_g) << 5);
  vtype current_row_values[MAX_VQ];
  uint32_t my_st = UINT32_MAX, my_ed = UINT32_MAX;
  uint64_t my_res_offset = 0;

  while (base_row_idx < frag_offset + num_frag_rows_pipeline)
  {
    uint64_t my_row = base_row_idx + lid;
    my_st = UINT32_MAX;
    my_ed = UINT32_MAX;
    my_res_offset = 0;

    if (my_row < frag_offset + num_frag_rows_pipeline)
    {
      uint64_t res_offset = fast_load(d_row_offsets + my_row);
      uint64_t next_offset = fast_load(d_row_offsets + my_row + 1);
      uint64_t num_res_row = next_offset - res_offset;
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

        vtype key_mapped = current_row_values[join_col];
        if (key_mapped != UINT32_MAX)
        {
          uint32_t pos = hash_lookup(num_buckets, edge_offset, directed_edge_idx, key_mapped);
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

__global__ void
join_count_kernel_dup_nomask_pipeline_local_info(
    // trie_inter (JoinInfo supports partial staging)
    JoinInfo<uint32_t> *data_info_arrays_, JoinInfo<uint64_t> *parent_info_arrays_,
    uint64_t pipeline_off_st, uint64_t pipeline_num_rows,
    uint32_t num_cols, uint32_t join_col,
    // trie_edge
    vtype *edge_value_array_, uint32_t *edge_offsets_,
    // hash
    uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx,
    // outputs (chunk-local)
    uint32_t *d_num_res_for_each,
    // options
    int dup_check_type, uint32_t num_dup_cols, uint32_t *dup_cols, uint32_t up_to_col)
{
  const uint32_t tid = threadIdx.x;
  const uint32_t idx = tid + blockIdx.x * blockDim.x;
  const uint32_t lid = tid & 31;
  const uint32_t wid_g = idx >> 5;
  const uint32_t num_warps = (blockDim.x >> 5) * gridDim.x;
  const uint32_t grid_size = blockDim.x * gridDim.x;

  uint64_t base_row_idx = pipeline_off_st + (static_cast<uint64_t>(wid_g) << 5);
  const uint64_t end_row = pipeline_off_st + pipeline_num_rows;

  vtype current_row_values[MAX_VQ];
  uint32_t my_st = UINT32_MAX, my_ed = UINT32_MAX;

  while (base_row_idx < end_row)
  {
    uint64_t my_row = base_row_idx + lid;
    my_st = UINT32_MAX;
    my_ed = UINT32_MAX;

    if (my_row < end_row)
    {
      uint64_t current_pos = my_row;
      for (int col_id = static_cast<int>(num_cols) - 1; col_id > static_cast<int>(up_to_col); --col_id)
      {
        current_row_values[col_id] = load_joininfo<uint32_t>(data_info_arrays_[col_id], current_pos);
        current_pos = load_joininfo<uint64_t>(parent_info_arrays_[col_id], current_pos);
      }
      current_row_values[up_to_col] = load_joininfo<uint32_t>(data_info_arrays_[up_to_col], current_pos);

      vtype key_mapped = current_row_values[join_col];
      if (key_mapped != UINT32_MAX)
      {
        uint32_t pos = hash_lookup(num_buckets, edge_offset, directed_edge_idx, key_mapped);
        if (pos != UINT32_MAX)
          fast_load_pair(edge_offsets_ + pos, my_st, my_ed);
      }
    }
    __syncwarp();

    for (uint32_t lane = 0; lane < warpSize; ++lane)
    {
      uint64_t row_idx = base_row_idx + lane;
      if (row_idx >= end_row)
        break;

      uint32_t bcast_st = __shfl_sync(FULL_MASK, my_st, lane);
      if (bcast_st == UINT32_MAX)
      {
        if (lid == 0)
          d_num_res_for_each[row_idx - pipeline_off_st] = 0;
        __syncwarp();
        continue;
      }

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
        if (lid == 0)
          row_count += __popc(found_mask);
        __syncwarp();
      }

      if (lid == 0)
        d_num_res_for_each[row_idx - pipeline_off_st] = row_count;
      __syncwarp();
    }

    base_row_idx += grid_size;
  }
}

__global__ void
join_write_kernel_dup_nomask_pipeline_local_info(
    // row range (global row ids)
    uint64_t frag_offset, uint64_t num_frag_rows_pipeline,
    // trie_inter (JoinInfo supports partial staging)
    JoinInfo<uint32_t> *data_info_arrays_, JoinInfo<uint64_t> *parent_info_arrays_,
    uint32_t num_cols, uint32_t join_col,
    // trie_edge
    vtype *edge_value_array_, uint32_t *edge_offsets_,
    // hash
    uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx,
    // chunk-local offsets and output (0-based)
    uint64_t *d_row_offsets, uint64_t base_result_offset,
    uint32_t *d_res, uint64_t *d_parent_res,
    // options
    int dup_check_type, uint32_t num_dup_cols, uint32_t *dup_cols, uint32_t up_to_col)
{
  const uint32_t tid = threadIdx.x;
  const uint32_t idx = tid + blockIdx.x * blockDim.x;
  const uint32_t lid = tid & 31;
  const uint32_t wid_g = idx >> 5;
  const uint32_t num_warps = (blockDim.x >> 5) * gridDim.x;
  const uint32_t grid_size = blockDim.x * gridDim.x;

  uint64_t base_row_idx = frag_offset + (static_cast<uint64_t>(wid_g) << 5);
  const uint64_t end_row = frag_offset + num_frag_rows_pipeline;

  vtype current_row_values[MAX_VQ];
  uint32_t my_st = UINT32_MAX, my_ed = UINT32_MAX;
  uint64_t my_res_offset = 0;

  while (base_row_idx < end_row)
  {
    uint64_t my_row = base_row_idx + lid;
    my_st = UINT32_MAX;
    my_ed = UINT32_MAX;
    my_res_offset = 0;

    if (my_row < end_row)
    {
      uint64_t local_row = my_row - frag_offset;
      uint64_t off0 = fast_load(d_row_offsets + local_row);
      uint64_t off1 = fast_load(d_row_offsets + local_row + 1);
      uint64_t num_res_row = off1 - off0;
      if (num_res_row)
      {
        my_res_offset = off0 - base_result_offset;

        uint64_t current_pos = my_row;
        for (int col_id = static_cast<int>(num_cols) - 1; col_id > static_cast<int>(up_to_col); --col_id)
        {
          current_row_values[col_id] = load_joininfo<uint32_t>(data_info_arrays_[col_id], current_pos);
          current_pos = load_joininfo<uint64_t>(parent_info_arrays_[col_id], current_pos);
        }
        current_row_values[up_to_col] = load_joininfo<uint32_t>(data_info_arrays_[up_to_col], current_pos);

        vtype key_mapped = current_row_values[join_col];
        if (key_mapped != UINT32_MAX)
        {
          uint32_t pos = hash_lookup(num_buckets, edge_offset, directed_edge_idx, key_mapped);
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
      if (row_idx >= end_row)
        break;

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
            uint64_t res_idx = bcast_res_offset + base_index + index_in_tile;
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

template __global__ void join_count_kernel_nodup<1>(vtype **vertex_arrays_, uint64_t **parent_pointers_, uint64_t num_rows, uint32_t num_cols, uint32_t join_col, vtype *edge_key_array_, vtype *edge_value_array_, uint32_t *edge_offsets_, uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx, uint32_t *d_num_res_for_each, uint32_t *d_st_array);
template __global__ void join_count_kernel_nodup<2>(vtype **vertex_arrays_, uint64_t **parent_pointers_, uint64_t num_rows, uint32_t num_cols, uint32_t join_col, vtype *edge_key_array_, vtype *edge_value_array_, uint32_t *edge_offsets_, uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx, uint32_t *d_num_res_for_each, uint32_t *d_st_array);
template __global__ void join_count_kernel_nodup<4>(vtype **vertex_arrays_, uint64_t **parent_pointers_, uint64_t num_rows, uint32_t num_cols, uint32_t join_col, vtype *edge_key_array_, vtype *edge_value_array_, uint32_t *edge_offsets_, uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx, uint32_t *d_num_res_for_each, uint32_t *d_st_array);
template __global__ void join_count_kernel_nodup<8>(vtype **vertex_arrays_, uint64_t **parent_pointers_, uint64_t num_rows, uint32_t num_cols, uint32_t join_col, vtype *edge_key_array_, vtype *edge_value_array_, uint32_t *edge_offsets_, uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx, uint32_t *d_num_res_for_each, uint32_t *d_st_array);
template __global__ void join_count_kernel_nodup<16>(vtype **vertex_arrays_, uint64_t **parent_pointers_, uint64_t num_rows, uint32_t num_cols, uint32_t join_col, vtype *edge_key_array_, vtype *edge_value_array_, uint32_t *edge_offsets_, uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx, uint32_t *d_num_res_for_each, uint32_t *d_st_array);
template __global__ void join_count_kernel_nodup<32>(vtype **vertex_arrays_, uint64_t **parent_pointers_, uint64_t num_rows, uint32_t num_cols, uint32_t join_col, vtype *edge_key_array_, vtype *edge_value_array_, uint32_t *edge_offsets_, uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx, uint32_t *d_num_res_for_each, uint32_t *d_st_array);

template __global__ void join_count_kernel_nodup_pipeline_info<1>(JoinInfo<uint32_t> *data_info_arrays_, JoinInfo<uint64_t> *parent_info_arrays_, uint64_t pipeline_off_st, uint64_t pipeline_num_rows, uint32_t num_cols, uint32_t join_col, vtype *edge_key_array_, vtype *edge_value_array_, uint32_t *edge_offsets_, uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx, uint32_t *d_num_res_for_each, uint32_t *d_st_array);
template __global__ void join_count_kernel_nodup_pipeline_info<2>(JoinInfo<uint32_t> *data_info_arrays_, JoinInfo<uint64_t> *parent_info_arrays_, uint64_t pipeline_off_st, uint64_t pipeline_num_rows, uint32_t num_cols, uint32_t join_col, vtype *edge_key_array_, vtype *edge_value_array_, uint32_t *edge_offsets_, uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx, uint32_t *d_num_res_for_each, uint32_t *d_st_array);
template __global__ void join_count_kernel_nodup_pipeline_info<4>(JoinInfo<uint32_t> *data_info_arrays_, JoinInfo<uint64_t> *parent_info_arrays_, uint64_t pipeline_off_st, uint64_t pipeline_num_rows, uint32_t num_cols, uint32_t join_col, vtype *edge_key_array_, vtype *edge_value_array_, uint32_t *edge_offsets_, uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx, uint32_t *d_num_res_for_each, uint32_t *d_st_array);
template __global__ void join_count_kernel_nodup_pipeline_info<8>(JoinInfo<uint32_t> *data_info_arrays_, JoinInfo<uint64_t> *parent_info_arrays_, uint64_t pipeline_off_st, uint64_t pipeline_num_rows, uint32_t num_cols, uint32_t join_col, vtype *edge_key_array_, vtype *edge_value_array_, uint32_t *edge_offsets_, uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx, uint32_t *d_num_res_for_each, uint32_t *d_st_array);
template __global__ void join_count_kernel_nodup_pipeline_info<16>(JoinInfo<uint32_t> *data_info_arrays_, JoinInfo<uint64_t> *parent_info_arrays_, uint64_t pipeline_off_st, uint64_t pipeline_num_rows, uint32_t num_cols, uint32_t join_col, vtype *edge_key_array_, vtype *edge_value_array_, uint32_t *edge_offsets_, uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx, uint32_t *d_num_res_for_each, uint32_t *d_st_array);
template __global__ void join_count_kernel_nodup_pipeline_info<32>(JoinInfo<uint32_t> *data_info_arrays_, JoinInfo<uint64_t> *parent_info_arrays_, uint64_t pipeline_off_st, uint64_t pipeline_num_rows, uint32_t num_cols, uint32_t join_col, vtype *edge_key_array_, vtype *edge_value_array_, uint32_t *edge_offsets_, uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx, uint32_t *d_num_res_for_each, uint32_t *d_st_array);

template __global__ void join_count_kernel_nodup_pipeline_local<1>(JoinInfo<uint32_t> *data_info_arrays_, JoinInfo<uint64_t> *parent_info_arrays_, uint64_t pipeline_off_st, uint64_t pipeline_num_rows, uint32_t num_cols, uint32_t join_col, vtype *edge_key_array_, vtype *edge_value_array_, uint32_t *edge_offsets_, uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx, uint32_t *d_num_res_for_each, uint32_t *d_st_array);
template __global__ void join_count_kernel_nodup_pipeline_local<2>(JoinInfo<uint32_t> *data_info_arrays_, JoinInfo<uint64_t> *parent_info_arrays_, uint64_t pipeline_off_st, uint64_t pipeline_num_rows, uint32_t num_cols, uint32_t join_col, vtype *edge_key_array_, vtype *edge_value_array_, uint32_t *edge_offsets_, uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx, uint32_t *d_num_res_for_each, uint32_t *d_st_array);
template __global__ void join_count_kernel_nodup_pipeline_local<4>(JoinInfo<uint32_t> *data_info_arrays_, JoinInfo<uint64_t> *parent_info_arrays_, uint64_t pipeline_off_st, uint64_t pipeline_num_rows, uint32_t num_cols, uint32_t join_col, vtype *edge_key_array_, vtype *edge_value_array_, uint32_t *edge_offsets_, uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx, uint32_t *d_num_res_for_each, uint32_t *d_st_array);
template __global__ void join_count_kernel_nodup_pipeline_local<8>(JoinInfo<uint32_t> *data_info_arrays_, JoinInfo<uint64_t> *parent_info_arrays_, uint64_t pipeline_off_st, uint64_t pipeline_num_rows, uint32_t num_cols, uint32_t join_col, vtype *edge_key_array_, vtype *edge_value_array_, uint32_t *edge_offsets_, uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx, uint32_t *d_num_res_for_each, uint32_t *d_st_array);
template __global__ void join_count_kernel_nodup_pipeline_local<16>(JoinInfo<uint32_t> *data_info_arrays_, JoinInfo<uint64_t> *parent_info_arrays_, uint64_t pipeline_off_st, uint64_t pipeline_num_rows, uint32_t num_cols, uint32_t join_col, vtype *edge_key_array_, vtype *edge_value_array_, uint32_t *edge_offsets_, uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx, uint32_t *d_num_res_for_each, uint32_t *d_st_array);
template __global__ void join_count_kernel_nodup_pipeline_local<32>(JoinInfo<uint32_t> *data_info_arrays_, JoinInfo<uint64_t> *parent_info_arrays_, uint64_t pipeline_off_st, uint64_t pipeline_num_rows, uint32_t num_cols, uint32_t join_col, vtype *edge_key_array_, vtype *edge_value_array_, uint32_t *edge_offsets_, uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx, uint32_t *d_num_res_for_each, uint32_t *d_st_array);

template __global__ void join_write_kernel_nodup<1>(uint64_t num_rows, vtype *edge_value_array_, uint32_t *d_num_res_for_each, uint32_t *d_st_array, uint32_t *d_res, uint64_t *d_offsets_of_, uint64_t *d_parent_res);
template __global__ void join_write_kernel_nodup<2>(uint64_t num_rows, vtype *edge_value_array_, uint32_t *d_num_res_for_each, uint32_t *d_st_array, uint32_t *d_res, uint64_t *d_offsets_of_, uint64_t *d_parent_res);
template __global__ void join_write_kernel_nodup<4>(uint64_t num_rows, vtype *edge_value_array_, uint32_t *d_num_res_for_each, uint32_t *d_st_array, uint32_t *d_res, uint64_t *d_offsets_of_, uint64_t *d_parent_res);
template __global__ void join_write_kernel_nodup<8>(uint64_t num_rows, vtype *edge_value_array_, uint32_t *d_num_res_for_each, uint32_t *d_st_array, uint32_t *d_res, uint64_t *d_offsets_of_, uint64_t *d_parent_res);
template __global__ void join_write_kernel_nodup<16>(uint64_t num_rows, vtype *edge_value_array_, uint32_t *d_num_res_for_each, uint32_t *d_st_array, uint32_t *d_res, uint64_t *d_offsets_of_, uint64_t *d_parent_res);
template __global__ void join_write_kernel_nodup<32>(uint64_t num_rows, vtype *edge_value_array_, uint32_t *d_num_res_for_each, uint32_t *d_st_array, uint32_t *d_res, uint64_t *d_offsets_of_, uint64_t *d_parent_res);
