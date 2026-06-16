#include "host_select_kernel.cuh"
#include "cuda_helpers.cuh"
#include "unifiedTrie.cuh"

template <typename T>
__device__ __forceinline__ T load_joininfo(const JoinInfo<T> &info, uint64_t global_idx)
{
  if (info.array_ptr == nullptr)
    return static_cast<T>(~0ull);

  uint64_t local_idx = info.is_partial ? (global_idx - info.offset) : global_idx;
  return fast_load(info.array_ptr + local_idx);
}

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
  uint32_t col_upper_bound = min(key_col, val_col);

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
      for (int col = num_cols - 1; col >= col_upper_bound; --col)
      {
        if (col == key_col)
          key_vertex = fast_load(frag_key_data_ + current_pos);
        if (col == val_col)
          value_vertex = fast_load(frag_value_data_ + current_pos);

        if (col == col_upper_bound)
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

__global__ void
select_trie_kernel_pipeline(
    // frag
    vtype *frag_key_data_, vtype *frag_value_data_,
    uint64_t **parent_pointers_, uint32_t num_cols,
    uint32_t key_col, uint32_t val_col,
    uint64_t pipeline_off_st, uint64_t pipeline_num_rows,
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

  uint64_t base_row_idx = pipeline_off_st + (wid_g << 5);
  uint64_t my_row_idx;
  bool matched;
  uint32_t pos, key_vertex, value_vertex;
  uint32_t col_upper_bound = min(key_col, val_col);

  while (base_row_idx < pipeline_off_st + pipeline_num_rows)
  {
    my_row_idx = base_row_idx + lid;
    matched = false;
    pos = UINT32_MAX;
    key_vertex = UINT32_MAX;
    value_vertex = UINT32_MAX;

    if (my_row_idx < pipeline_off_st + pipeline_num_rows)
    {
      uint64_t current_pos = my_row_idx;
      for (int col = num_cols - 1; col >= col_upper_bound; --col)
      {
        if (col == key_col)
          key_vertex = fast_load(frag_key_data_ + current_pos);
        if (col == val_col)
          value_vertex = fast_load(frag_value_data_ + current_pos);

        if (col == col_upper_bound)
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
      bool_vec[(base_row_idx - pipeline_off_st) >> 5] = mask;
    }
    __syncwarp();
    base_row_idx += grid_size;
  }
}

__global__ void
select_trie_kernel_pipeline_info(
    // frag (key/value column data)
    JoinInfo<uint32_t> frag_key_info, JoinInfo<uint32_t> frag_value_info,
    // parents per column
    JoinInfo<uint64_t> *parent_info_arrays, uint32_t num_cols,
    uint32_t key_col, uint32_t val_col,
    uint64_t pipeline_off_st, uint64_t pipeline_num_rows,
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

  uint64_t base_row_idx = pipeline_off_st + (wid_g << 5);
  uint64_t my_row_idx;
  bool matched;
  uint32_t pos, key_vertex, value_vertex;
  uint32_t col_upper_bound = min(key_col, val_col);

  while (base_row_idx < pipeline_off_st + pipeline_num_rows)
  {
    my_row_idx = base_row_idx + lid;
    matched = false;
    pos = UINT32_MAX;
    key_vertex = UINT32_MAX;
    value_vertex = UINT32_MAX;

    if (my_row_idx < pipeline_off_st + pipeline_num_rows)
    {
      uint64_t current_pos = my_row_idx;
      for (int col = num_cols - 1; col >= col_upper_bound; --col)
      {
        if (col == static_cast<int>(key_col))
          key_vertex = load_joininfo<uint32_t>(frag_key_info, current_pos);
        if (col == static_cast<int>(val_col))
          value_vertex = load_joininfo<uint32_t>(frag_value_info, current_pos);

        if (col == static_cast<int>(col_upper_bound))
          break;
        current_pos = load_joininfo<uint64_t>(parent_info_arrays[col], current_pos);
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
        matched = true;
    }

    uint32_t mask = __ballot_sync(FULL_MASK, matched);

    if (lid == 0)
      bool_vec[(base_row_idx - pipeline_off_st) >> 5] = mask;
    __syncwarp();
    base_row_idx += grid_size;
  }
}

__global__ void host_atomicAdd(uint64_t *addr, uint64_t *val)
{
  atomicAdd((unsigned long long *)addr, (unsigned long long)*val);
}

__global__ void select_trie_refer_kernel(
    // original data to refer to
    vtype *frag_key_data_, vtype *frag_value_data_,
    uint64_t **parent_pointers_, uint32_t num_cols,
    uint32_t key_col, uint32_t val_col,
    // selection mask
    uint32_t *select_bool_vec_, uint64_t frag_num_rows,
    // trie_edge
    vtype *edge_key_array_, vtype *edge_value_array_, uint32_t *edge_offsets_,
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
  uint32_t col_upper_bound = min(key_col, val_col);

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
      for (int col_id = num_cols - 1; col_id >= col_upper_bound; --col_id)
      {
        if (col_id == key_col)
          key_vertex = fast_load(frag_key_data_ + current_pos);
        if (col_id == val_col)
          value_vertex = fast_load(frag_value_data_ + current_pos);
        if (col_id == col_upper_bound)
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

__global__ void select_trie_refer_kernel_pipeline(
    // original data to refer to
    vtype *frag_key_data_, vtype *frag_value_data_,
    uint64_t **parent_pointers_, uint32_t num_cols,
    uint32_t key_col, uint32_t val_col,
    // selection mask
    uint32_t *select_bool_vec_,
    uint64_t pipeline_off_st, uint64_t pipeline_num_rows,
    // trie_edge
    vtype *edge_key_array_, vtype *edge_value_array_, uint32_t *edge_offsets_,
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

  uint64_t base_row_idx = pipeline_off_st + (wid_g << 5);
  vtype key_vertex, value_vertex;
  bool matched;
  uint64_t my_row_idx;
  uint32_t pos;
  uint32_t row_selected;
  uint32_t col_upper_bound = min(key_col, val_col);

  while (base_row_idx < pipeline_off_st + pipeline_num_rows)
  {
    uint64_t tile_idx = (base_row_idx - pipeline_off_st) >> 5;
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
    bool lane_active = row_selected_bit && (my_row_idx < pipeline_off_st + pipeline_num_rows);

    // Load the two vertices forming the edge
    if (lane_active)
    {
      uint64_t current_pos = my_row_idx;
      // Traverse up the parent pointers from lower_col to the target higher_col layer
      for (int col_id = num_cols - 1; col_id >= col_upper_bound; --col_id)
      {
        if (col_id == key_col)
          key_vertex = fast_load(frag_key_data_ + current_pos);
        if (col_id == val_col)
          value_vertex = fast_load(frag_value_data_ + current_pos);
        if (col_id == col_upper_bound)
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

__global__ void select_trie_refer_kernel_pipeline_info(
    // original data to refer to (key/value column data)
    JoinInfo<uint32_t> frag_key_info, JoinInfo<uint32_t> frag_value_info,
    // parents per column
    JoinInfo<uint64_t> *parent_info_arrays, uint32_t num_cols,
    uint32_t key_col, uint32_t val_col,
    // selection mask (already windowed to this pipeline tile)
    uint32_t *select_bool_vec_,
    uint64_t pipeline_off_st, uint64_t pipeline_num_rows,
    // trie_edge
    vtype *edge_key_array_, vtype *edge_value_array_, uint32_t *edge_offsets_,
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

  uint64_t base_row_idx = pipeline_off_st + (wid_g << 5);
  vtype key_vertex, value_vertex;
  bool matched;
  uint64_t my_row_idx;
  uint32_t pos;
  uint32_t row_selected;
  uint32_t col_upper_bound = min(key_col, val_col);

  while (base_row_idx < pipeline_off_st + pipeline_num_rows)
  {
    uint64_t tile_idx = (base_row_idx - pipeline_off_st) >> 5;
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
    bool lane_active = row_selected_bit && (my_row_idx < pipeline_off_st + pipeline_num_rows);

    if (lane_active)
    {
      uint64_t current_pos = my_row_idx;
      for (int col = num_cols - 1; col >= col_upper_bound; --col)
      {
        if (col == static_cast<int>(key_col))
          key_vertex = load_joininfo<uint32_t>(frag_key_info, current_pos);
        if (col == static_cast<int>(val_col))
          value_vertex = load_joininfo<uint32_t>(frag_value_info, current_pos);

        if (col == static_cast<int>(col_upper_bound))
          break;
        current_pos = load_joininfo<uint64_t>(parent_info_arrays[col], current_pos);
      }
    }
    __syncwarp();

    if (lane_active && key_vertex != UINT32_MAX)
      pos = hash_lookup(num_buckets, edge_offset, directed_edge_idx, key_vertex);
    __syncwarp();

    if (lane_active && pos != UINT32_MAX && value_vertex != UINT32_MAX)
    {
      uint32_t st, ed;
      fast_load_pair(edge_offsets_ + pos, st, ed);
      uint32_t res = UINT32_MAX;
      if (ed > st)
        res = lower_bound(edge_value_array_ + st, ed - st, value_vertex);
      if (res != UINT32_MAX && fast_load(edge_value_array_ + st + res) == value_vertex)
        matched = true;
    }

    uint32_t mask = __ballot_sync(FULL_MASK, matched);

    if (lid == 0)
      bool_vec[tile_idx] = mask;
    __syncwarp();
    base_row_idx += grid_size;
  }
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
    vtype *edge_key_array_, vtype *edge_value_array_, uint32_t *edge_offsets_,
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

  const uint64_t grid_size = static_cast<uint64_t>(blockDim.x) * gridDim.x;

  uint64_t base_row_idx = static_cast<uint64_t>(wid_g) << 5;
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
      bool_vec[static_cast<size_t>(base_row_idx >> 5)] = mask;
    }
    __syncwarp();
    base_row_idx += grid_size;
  }
}
