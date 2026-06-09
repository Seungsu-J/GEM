#include "construct_kernel.cuh"
#include "cuda_helpers.cuh"

template <typename T>
__device__ __forceinline__ void merge_path_search_device(
    const T *a, uint32_t a_len,
    const T *b, uint32_t b_len,
    uint32_t diag,
    uint32_t &a_idx,
    uint32_t &b_idx)
{
  uint32_t total = a_len + b_len;
  if (diag > total)
    diag = total;

  uint32_t low = 0;
  if (diag > b_len)
    low = diag - b_len;
  uint32_t high = diag;
  if (high > a_len)
    high = a_len;

  while (low < high)
  {
    uint32_t mid = (low + high) >> 1;
    uint32_t y = diag - mid;
    bool move_high = false;
    bool move_low = false;

    if (mid > 0 && y < b_len)
    {
      T a_left = fast_load(a + mid - 1);
      T b_val = fast_load(b + y);
      if (a_left > b_val)
        move_high = true;
    }

    if (y > 0 && mid < a_len)
    {
      T b_left = fast_load(b + y - 1);
      T a_val = fast_load(a + mid);
      if (b_left >= a_val)
        move_low = true;
    }

    if (move_high)
    {
      high = mid;
    }
    else if (move_low)
    {
      low = mid + 1;
    }
    else
    {
      low = mid;
      break;
    }
  }

  a_idx = low;
  if (a_idx > a_len)
    a_idx = a_len;

  b_idx = diag - a_idx;
  if (b_idx > b_len)
  {
    b_idx = b_len;
    a_idx = diag - b_idx;
  }
}

__global__ void
exclusive_scan_on_mask(
    uint32_t *bitmap_mask_, uint32_t mask_length,
    uint32_t num_rows,
    uint32_t *scan_result)
{
  const uint32_t tid = threadIdx.x;
  const uint32_t bid = blockIdx.x;
  const uint32_t num_warps = (blockDim.x + warpSize - 1) / warpSize;
  const uint32_t warp_id = tid >> 5;
  const uint32_t lane = tid & (warpSize - 1);
  const uint32_t warp_base = warp_id * warpSize;
  const uint32_t threads_in_block = blockDim.x;
  const uint32_t remaining_threads =
      (threads_in_block > warp_base) ? (threads_in_block - warp_base) : 0u;
  const uint32_t warp_threads =
      (remaining_threads >= warpSize) ? warpSize : remaining_threads;
  const uint32_t warp_mask =
      (warp_threads == warpSize) ? FULL_MASK : ((1u << warp_threads) - 1u);
  const uint32_t warp0_threads =
      (threads_in_block >= warpSize) ? warpSize : threads_in_block;
  const uint32_t warp0_mask =
      (warp0_threads == warpSize) ? FULL_MASK : ((1u << warp0_threads) - 1u);
  const uint32_t last_lane = warp_threads - 1;
  const uint32_t scan_stride = mask_length + 1;

  __shared__ uint32_t warp_totals[WARP_PER_BLOCK];
  __shared__ uint32_t warp_prefix[WARP_PER_BLOCK];
  __shared__ uint32_t row_carry;

  for (uint32_t row = bid; row < num_rows; row += gridDim.x)
  {
    if (tid == 0)
      row_carry = 0;
    __syncthreads();

    const uint32_t mask_row_offset = row * mask_length;
    const uint32_t scan_row_offset = row * scan_stride;

    for (uint32_t col_base = 0; col_base < mask_length; col_base += blockDim.x)
    {
      const uint32_t col = col_base + tid;
      uint32_t tile_count = 0;
      if (col < mask_length)
      {
        uint32_t tile_mask = fast_load(bitmap_mask_ + mask_row_offset + col);
        tile_count = __popc(tile_mask);
      }

      uint32_t prefix = tile_count;
      for (int offset = 1; offset < warpSize; offset <<= 1)
      {
        uint32_t val = __shfl_up_sync(warp_mask, prefix, offset);
        if (lane >= static_cast<uint32_t>(offset))
          prefix += val;
      }
      const uint32_t warp_exclusive = prefix - tile_count;
      const uint32_t warp_total = __shfl_sync(warp_mask, prefix, last_lane);

      if (lane == last_lane)
        warp_totals[warp_id] = warp_total;
      __syncthreads();

      if (warp_id == 0)
      {
        uint32_t warp_val = (lane < num_warps) ? warp_totals[lane] : 0;
        uint32_t warp_scan = warp_val;
        for (int offset = 1; offset < warpSize; offset <<= 1)
        {
          uint32_t val = __shfl_up_sync(warp0_mask, warp_scan, offset);
          if (lane >= static_cast<uint32_t>(offset))
            warp_scan += val;
        }
        if (lane < num_warps)
          warp_prefix[lane] = warp_scan - warp_val;
      }
      __syncthreads();

      const uint32_t chunk_base = row_carry + warp_prefix[warp_id];
      if (col < mask_length)
        scan_result[scan_row_offset + col] = chunk_base + warp_exclusive;
      __syncthreads();

      if (tid == 0)
      {
        uint32_t chunk_total = 0;
        if (num_warps != 0)
          chunk_total = warp_prefix[num_warps - 1] + warp_totals[num_warps - 1];
        row_carry += chunk_total;
      }
      __syncthreads();
    }

    if (tid == 0)
      scan_result[scan_row_offset + mask_length] = row_carry;
    __syncthreads();
  }
}

__global__ void
first_join_count_new(
    // data graph
    offtype *offsets_, vtype *nbrs_, degtype *degree_,
    uint32_t *bitmap_mask_, uint32_t mask_length,
    vtype u, vtype u_nbr,
    numtype num_can_u, numtype num_can_u_nbr,
    vtype *d_u_candidate_vs_)
{
  const uint32_t tid = threadIdx.x;
  const uint32_t bid = blockIdx.x;
  const uint32_t idx = tid + bid * blockDim.x;
  const uint32_t wid = tid >> 5;
  const uint32_t lid = tid & 31;
  const uint32_t wid_g = idx >> 5;
  const uint32_t num_warps = (blockDim.x >> 5) * gridDim.x;

  extern __shared__ vtype s_candidates[];

  vtype *s_can_u = s_candidates;
  vtype *s_can_u_nbr = s_candidates + num_can_u;

  // load to shared memory
  for (uint32_t i = tid; i < num_can_u; i += blockDim.x)
    s_can_u[i] = fast_load(d_u_candidate_vs_ + u * C_MAX_L_FREQ + i);
  for (uint32_t i = tid; i < num_can_u_nbr; i += blockDim.x)
    s_can_u_nbr[i] = fast_load(d_u_candidate_vs_ + u_nbr * C_MAX_L_FREQ + i);
  __syncthreads();

  uint32_t local_v;
  offtype local_off_st;
  uint32_t local_v_degree;
  uint32_t shared_v;
  offtype shared_off_st;
  uint32_t shared_v_degree;

  uint32_t num_can_u_tiles = (num_can_u + warpSize - 1) / warpSize;
  uint32_t num_can_u_nbr_tiles = (num_can_u_nbr + warpSize - 1) / warpSize;
  uint32_t total_tiles = num_can_u_tiles * num_can_u_nbr_tiles;
  for (int task_id = wid_g; task_id < total_tiles; task_id += num_warps)
  {
    uint32_t u_tile_id = task_id / num_can_u_nbr_tiles;
    uint32_t u_nbr_tile_id = task_id % num_can_u_nbr_tiles;
    uint32_t v_off = u_tile_id * warpSize + lid;
    uint32_t u_nbr_off = u_nbr_tile_id * warpSize + lid;
    if (v_off < num_can_u) [[likely]]
    {
      local_v = s_can_u[v_off];
      // local_v = fast_load(s_can_u + v_off);
      local_off_st = fast_load(offsets_ + local_v);
      local_v_degree = fast_load(degree_ + local_v);
    }
    else
      local_v = UINT32_MAX;
    __syncwarp();

    for (int cur_task_id = 0; cur_task_id < warpSize; ++cur_task_id)
    {
      // data preparing
      shared_v = __shfl_sync(FULL_MASK, local_v, cur_task_id);
      if (shared_v == UINT32_MAX)
        break; // early exit
      shared_off_st = __shfl_sync(FULL_MASK, local_off_st, cur_task_id);
      shared_v_degree = __shfl_sync(FULL_MASK, local_v_degree, cur_task_id);
      // if (lid == 0)
      // warp_count[wid] = 0;
      // __syncwarp();

      // computing
      vtype v_nbr = UINT32_MAX;
      if (u_nbr_off < num_can_u_nbr)
        v_nbr = s_can_u_nbr[u_nbr_off];
      __syncwarp();
      bool found = false;
      if (v_nbr != shared_v)
        found = binary_search(nbrs_ + shared_off_st, shared_v_degree, v_nbr);
      __syncwarp();

      uint32_t found_mask = __ballot_sync(FULL_MASK, found);
      if (lid == cur_task_id)
      {
        // atomicAdd(&num_res_for_each[v_off], __popc(found_mask));
        bitmap_mask_[v_off * mask_length + u_nbr_tile_id] = found_mask;
      }
      __syncwarp();
      // if (found_mask && lid == 0)
      //   warp_count[wid] += __popc(found_mask);
    }
  }
}

__global__ void
first_join_count_global_workpool_32_32(
    // data graph
    offtype *offsets_, vtype *nbrs_, degtype *degree_,
    uint32_t *bitmap_mask_, uint32_t mask_length,
    vtype u, vtype u_nbr,
    numtype num_can_u, numtype num_can_u_nbr,
    vtype *d_u_candidate_vs_, uint32_t *workpool)
{
  const uint32_t tid = threadIdx.x;
  const uint32_t bid = blockIdx.x;
  const uint32_t idx = tid + bid * blockDim.x;
  const uint32_t wid = tid >> 5;
  const uint32_t lid = tid & 31;
  const uint32_t wid_g = idx >> 5;
  const uint32_t num_warps = (blockDim.x >> 5) * gridDim.x;

  // extern __shared__ vtype s_candidates[];

  vtype *s_can_u = d_u_candidate_vs_ + u * C_MAX_L_FREQ;
  vtype *s_can_u_nbr = d_u_candidate_vs_ + u_nbr * C_MAX_L_FREQ;

  uint32_t local_v;
  offtype local_off_st;
  uint32_t local_v_degree;
  uint32_t shared_v;
  offtype shared_off_st;
  uint32_t shared_v_degree;

  uint32_t num_can_u_tiles = (num_can_u + warpSize - 1) / warpSize;
  uint32_t num_can_u_nbr_tiles = (num_can_u_nbr + warpSize - 1) / warpSize;
  uint32_t total_tiles = num_can_u_tiles * num_can_u_nbr_tiles;
  // for (int task_id = wid_g; task_id < total_tiles; task_id += num_warps)
  uint32_t task_id;
  while (1)
  {
    if (lid == 0)
      task_id = atomicAdd(workpool, 1);
    task_id = __shfl_sync(FULL_MASK, task_id, 0);
    if (task_id >= total_tiles)
      break;
    uint32_t u_tile_id = task_id / num_can_u_nbr_tiles;
    uint32_t u_nbr_tile_id = task_id % num_can_u_nbr_tiles;
    uint32_t v_off = u_tile_id * warpSize + lid;
    uint32_t u_nbr_off = u_nbr_tile_id * warpSize + lid;
    if (v_off < num_can_u) [[likely]]
    {
      local_v = fast_load(s_can_u + v_off);
      local_off_st = fast_load(offsets_ + local_v);
      local_v_degree = fast_load(degree_ + local_v);
    }
    else
      local_v = UINT32_MAX;
    __syncwarp();

    for (int cur_task_id = 0; cur_task_id < warpSize; ++cur_task_id)
    {
      // data preparing
      shared_v = __shfl_sync(FULL_MASK, local_v, cur_task_id);
      if (shared_v == UINT32_MAX)
        break; // early exit
      shared_off_st = __shfl_sync(FULL_MASK, local_off_st, cur_task_id);
      shared_v_degree = __shfl_sync(FULL_MASK, local_v_degree, cur_task_id);
      // if (lid == 0)
      // warp_count[wid] = 0;
      // __syncwarp();

      // computing
      vtype v_nbr = UINT32_MAX;
      if (u_nbr_off < num_can_u_nbr)
        v_nbr = fast_load(s_can_u_nbr + u_nbr_off);
      __syncwarp();
      bool found = false;
      if (v_nbr != shared_v)
        found = binary_search(nbrs_ + shared_off_st, shared_v_degree, v_nbr);
      __syncwarp();

      uint32_t found_mask = __ballot_sync(FULL_MASK, found);
      if (lid == cur_task_id)
      {
        // atomicAdd(&num_res_for_each[v_off], __popc(found_mask));
        bitmap_mask_[v_off * mask_length + u_nbr_tile_id] = found_mask;
      }
      __syncwarp();
      // if (found_mask && lid == 0)
      //   warp_count[wid] += __popc(found_mask);
    }
  }
}

__global__ void
first_join_count_global_32_32(
    // data graph
    offtype *offsets_, vtype *nbrs_, degtype *degree_,
    uint32_t *bitmap_mask_, uint32_t mask_length,
    vtype u, vtype u_nbr,
    numtype num_can_u, numtype num_can_u_nbr,
    vtype *d_u_candidate_vs_)
{
  const uint32_t tid = threadIdx.x;
  const uint32_t bid = blockIdx.x;
  const uint32_t idx = tid + bid * blockDim.x;
  const uint32_t wid = tid >> 5;
  const uint32_t lid = tid & 31;
  const uint32_t wid_g = idx >> 5;
  const uint32_t num_warps = (blockDim.x >> 5) * gridDim.x;

  // extern __shared__ vtype s_candidates[];

  vtype *s_can_u = d_u_candidate_vs_ + u * C_MAX_L_FREQ;
  vtype *s_can_u_nbr = d_u_candidate_vs_ + u_nbr * C_MAX_L_FREQ;

  uint32_t local_v;
  offtype local_off_st;
  uint32_t local_v_degree;
  uint32_t shared_v;
  offtype shared_off_st;
  uint32_t shared_v_degree;

  uint32_t num_can_u_tiles = (num_can_u + warpSize - 1) / warpSize;
  uint32_t num_can_u_nbr_tiles = (num_can_u_nbr + warpSize - 1) / warpSize;
  uint32_t total_tiles = num_can_u_tiles * num_can_u_nbr_tiles;
  for (int task_id = wid_g; task_id < total_tiles; task_id += num_warps)
  {
    uint32_t u_tile_id = task_id / num_can_u_nbr_tiles;
    uint32_t u_nbr_tile_id = task_id % num_can_u_nbr_tiles;
    uint32_t v_off = u_tile_id * warpSize + lid;
    uint32_t u_nbr_off = u_nbr_tile_id * warpSize + lid;
    if (v_off < num_can_u) [[likely]]
    {
      local_v = fast_load(s_can_u + v_off);
      local_off_st = fast_load(offsets_ + local_v);
      local_v_degree = fast_load(degree_ + local_v);
    }
    else
      local_v = UINT32_MAX;
    __syncwarp();

    for (int cur_task_id = 0; cur_task_id < warpSize; ++cur_task_id)
    {
      // data preparing
      shared_v = __shfl_sync(FULL_MASK, local_v, cur_task_id);
      if (shared_v == UINT32_MAX)
        break; // early exit
      shared_off_st = __shfl_sync(FULL_MASK, local_off_st, cur_task_id);
      shared_v_degree = __shfl_sync(FULL_MASK, local_v_degree, cur_task_id);
      // if (lid == 0)
      // warp_count[wid] = 0;
      // __syncwarp();

      // computing
      vtype v_nbr = UINT32_MAX;
      if (u_nbr_off < num_can_u_nbr)
        v_nbr = fast_load(s_can_u_nbr + u_nbr_off);
      __syncwarp();
      bool found = false;
      if (v_nbr != shared_v)
        found = binary_search(nbrs_ + shared_off_st, shared_v_degree, v_nbr);
      __syncwarp();

      uint32_t found_mask = __ballot_sync(FULL_MASK, found);
      if (lid == cur_task_id)
      {
        // atomicAdd(&num_res_for_each[v_off], __popc(found_mask));
        bitmap_mask_[v_off * mask_length + u_nbr_tile_id] = found_mask;
      }
      __syncwarp();
      // if (found_mask && lid == 0)
      //   warp_count[wid] += __popc(found_mask);
    }
  }
}

__global__ void
first_join_count_global_workpool_1_32(
    // data graph
    offtype *offsets_, vtype *nbrs_, degtype *degree_,
    uint32_t *bitmap_mask_, uint32_t mask_length,
    vtype u, vtype u_nbr,
    numtype num_can_u, numtype num_can_u_nbr,
    vtype *d_u_candidate_vs_, uint32_t *workpool)
{
  const uint32_t tid = threadIdx.x;
  const uint32_t bid = blockIdx.x;
  const uint32_t idx = tid + bid * blockDim.x;
  const uint32_t wid = tid >> 5;
  const uint32_t lid = tid & 31;
  const uint32_t wid_g = idx >> 5;
  const uint32_t num_warps = (blockDim.x >> 5) * gridDim.x;

  // extern __shared__ vtype s_candidates[];

  vtype *s_can_u = d_u_candidate_vs_ + u * C_MAX_L_FREQ;
  vtype *s_can_u_nbr = d_u_candidate_vs_ + u_nbr * C_MAX_L_FREQ;

  uint32_t local_v;
  offtype local_off_st;
  uint32_t local_v_degree;

  uint32_t num_can_u_tiles = num_can_u;
  uint32_t num_can_u_nbr_tiles = (num_can_u_nbr + warpSize - 1) / warpSize;
  uint32_t total_tiles = num_can_u_tiles * num_can_u_nbr_tiles;

  uint32_t base_task_id, task_id;
  while (1)
  {
    // if (lid == 0)
    //   task_id = atomicAdd(workpool, 1);
    // task_id = __shfl_sync(FULL_MASK, task_id, 0);
    // if (task_id >= total_tiles)
    //   break;
    if (lid == 0)
      base_task_id = atomicAdd(workpool, warpSize);
    base_task_id = __shfl_sync(FULL_MASK, base_task_id, 0);
    for (task_id = base_task_id; task_id < min(base_task_id + warpSize, total_tiles); ++task_id)
    {
      uint32_t u_tile_id = task_id / num_can_u_nbr_tiles;
      uint32_t u_nbr_tile_id = task_id % num_can_u_nbr_tiles;
      uint32_t v_off = u_tile_id;
      uint32_t u_nbr_off = u_nbr_tile_id * warpSize + lid;
      if (v_off < num_can_u) [[likely]]
      {
        local_v = fast_load(s_can_u + v_off);
        local_off_st = fast_load(offsets_ + local_v);
        local_v_degree = fast_load(degree_ + local_v);
      }
      else
        local_v = UINT32_MAX;
      __syncwarp();

      // if (lid == 0)
      // warp_count[wid] = 0;
      // __syncwarp();

      // computing
      vtype v_nbr = UINT32_MAX;
      if (u_nbr_off < num_can_u_nbr)
        v_nbr = fast_load(s_can_u_nbr + u_nbr_off);
      __syncwarp();
      bool found = false;
      if (v_nbr != local_v)
        found = binary_search(nbrs_ + local_off_st, local_v_degree, v_nbr);
      __syncwarp();

      uint32_t found_mask = __ballot_sync(FULL_MASK, found);
      if (lid == 0)
      {
        // atomicAdd(&num_res_for_each[v_off], __popc(found_mask));
        bitmap_mask_[v_off * mask_length + u_nbr_tile_id] = found_mask;
      }
      __syncwarp();
    }
    // if (found_mask && lid == 0)
    //   warp_count[wid] += __popc(found_mask);
  }
}

__global__ void
first_join_count_global_1_32(
    // data graph
    offtype *offsets_, vtype *nbrs_, degtype *degree_,
    uint32_t *bitmap_mask_, uint32_t mask_length,
    vtype u, vtype u_nbr,
    numtype num_can_u, numtype num_can_u_nbr,
    vtype *d_u_candidate_vs_)
{
  const uint32_t tid = threadIdx.x;
  const uint32_t bid = blockIdx.x;
  const uint32_t idx = tid + bid * blockDim.x;
  const uint32_t wid = tid >> 5;
  const uint32_t lid = tid & 31;
  const uint32_t wid_g = idx >> 5;
  const uint32_t num_warps = (blockDim.x >> 5) * gridDim.x;

  // extern __shared__ vtype s_candidates[];

  vtype *s_can_u = d_u_candidate_vs_ + u * C_MAX_L_FREQ;
  vtype *s_can_u_nbr = d_u_candidate_vs_ + u_nbr * C_MAX_L_FREQ;

  uint32_t local_v;
  offtype local_off_st;
  uint32_t local_v_degree;

  uint32_t num_can_u_tiles = num_can_u;
  uint32_t num_can_u_nbr_tiles = (num_can_u_nbr + warpSize - 1) / warpSize;
  uint32_t total_tiles = num_can_u_tiles * num_can_u_nbr_tiles;
  for (int task_id = wid_g; task_id < total_tiles; task_id += num_warps)
  {
    uint32_t u_tile_id = task_id / num_can_u_nbr_tiles;
    uint32_t u_nbr_tile_id = task_id % num_can_u_nbr_tiles;
    uint32_t v_off = u_tile_id;
    uint32_t u_nbr_off = u_nbr_tile_id * warpSize + lid;
    if (v_off < num_can_u) [[likely]]
    {
      local_v = fast_load(s_can_u + v_off);
      local_off_st = fast_load(offsets_ + local_v);
      local_v_degree = fast_load(degree_ + local_v);
    }
    else
      local_v = UINT32_MAX;
    __syncwarp();

    // if (lid == 0)
    // warp_count[wid] = 0;
    // __syncwarp();

    // computing
    vtype v_nbr = UINT32_MAX;
    if (u_nbr_off < num_can_u_nbr)
      v_nbr = fast_load(s_can_u_nbr + u_nbr_off);
    __syncwarp();
    bool found = false;
    if (v_nbr != local_v)
      found = binary_search(nbrs_ + local_off_st, local_v_degree, v_nbr);
    __syncwarp();

    uint32_t found_mask = __ballot_sync(FULL_MASK, found);
    if (lid == 0)
    {
      // atomicAdd(&num_res_for_each[v_off], __popc(found_mask));
      bitmap_mask_[v_off * mask_length + u_nbr_tile_id] = found_mask;
    }
    __syncwarp();
    // if (found_mask && lid == 0)
    //   warp_count[wid] += __popc(found_mask);
  }
}

__global__ void
first_join_count_global_1_1(
    // data graph
    offtype *offsets_, vtype *nbrs_, degtype *degree_,
    uint32_t *bitmap_mask_, uint32_t mask_length,
    vtype u, vtype u_nbr,
    numtype num_can_u, numtype num_can_u_nbr,
    vtype *d_u_candidate_vs_)
{
  const uint32_t tid = threadIdx.x;
  const uint32_t bid = blockIdx.x;
  const uint32_t idx = tid + bid * blockDim.x;
  const uint32_t wid = tid >> 5;
  const uint32_t lid = tid & 31;
  const uint32_t wid_g = idx >> 5;
  const uint32_t num_warps = (blockDim.x >> 5) * gridDim.x;

  // extern __shared__ vtype s_candidates[];

  vtype *s_can_u = d_u_candidate_vs_ + u * C_MAX_L_FREQ;
  vtype *s_can_u_nbr = d_u_candidate_vs_ + u_nbr * C_MAX_L_FREQ;

  uint32_t local_v;
  offtype local_off_st;
  uint32_t local_v_degree;

  uint32_t num_can_u_tiles = num_can_u;
  uint32_t num_can_u_nbr_tiles = num_can_u_nbr;
  uint32_t total_tiles = num_can_u_tiles * num_can_u_nbr_tiles;
  for (int task_id = wid_g; task_id < total_tiles; task_id += num_warps)
  {
    uint32_t u_tile_id = task_id / num_can_u_nbr_tiles;
    uint32_t u_nbr_tile_id = task_id % num_can_u_nbr_tiles;
    uint32_t v_off = u_tile_id;
    uint32_t u_nbr_off = u_nbr_tile_id;
    if (v_off < num_can_u) [[likely]]
    {
      local_v = fast_load(s_can_u + v_off);
      local_off_st = fast_load(offsets_ + local_v);
      local_v_degree = fast_load(degree_ + local_v);
    }
    else
      local_v = UINT32_MAX;
    __syncwarp();

    // if (lid == 0)
    // warp_count[wid] = 0;
    // __syncwarp();

    // computing
    vtype v_nbr = UINT32_MAX;
    if (u_nbr_off < num_can_u_nbr)
      v_nbr = fast_load(s_can_u_nbr + u_nbr_off);
    __syncwarp();
    bool found = false;
    if (v_nbr != local_v)
      found = binary_search(nbrs_ + local_off_st, local_v_degree, v_nbr);
    __syncwarp();

    // uint32_t found_mask = __ballot_sync(FULL_MASK, found);
    if (lid == 0 && found)
    {
      // atomicAdd(&num_res_for_each[v_off], __popc(found_mask));
      // bitmap_mask_[v_off * mask_length + u_nbr_tile_id] = found_mask;
      atomicOr(bitmap_mask_ + v_off * mask_length + (u_nbr_tile_id >> 5), (1u << (u_nbr_off & 31)));
    }
    __syncwarp();
    // if (found_mask && lid == 0)
    //   warp_count[wid] += __popc(found_mask);
  }
}

__global__ void
first_join_count_global_para_serial(
    // data graph
    offtype *offsets_, vtype *nbrs_, degtype *degree_,
    uint32_t *bitmap_mask_, uint32_t mask_length,
    vtype u, vtype u_nbr,
    numtype num_can_u, numtype num_can_u_nbr,
    vtype *d_u_candidate_vs_)
{
  const uint32_t tid = threadIdx.x;
  const uint32_t bid = blockIdx.x;
  const uint32_t idx = tid + bid * blockDim.x;
  const uint32_t wid = tid >> 5;
  const uint32_t lid = tid & 31;
  const uint32_t wid_g = idx >> 5;
  const uint32_t num_warps = (blockDim.x >> 5) * gridDim.x;

  // extern __shared__ vtype s_candidates[];
  using WarpReduce = cub::WarpReduce<uint32_t>;
  __shared__ typename WarpReduce::TempStorage warp_reduce_storage[WARP_PER_BLOCK];

  vtype *s_can_u = d_u_candidate_vs_ + u * C_MAX_L_FREQ;
  vtype *s_can_u_nbr = d_u_candidate_vs_ + u_nbr * C_MAX_L_FREQ;

  uint32_t local_v;
  offtype local_off_st;
  uint32_t local_v_degree;

  uint32_t num_can_u_tiles = num_can_u;
  uint32_t num_can_u_nbr_tiles = (num_can_u_nbr + warpSize - 1) / warpSize;
  uint32_t total_tiles = num_can_u_tiles * num_can_u_nbr_tiles;
  for (int task_id = wid_g; task_id < total_tiles; task_id += num_warps)
  {
    uint32_t u_tile_id = task_id / num_can_u_nbr_tiles;
    uint32_t u_nbr_tile_id = task_id % num_can_u_nbr_tiles;
    uint32_t v_off = u_tile_id;
    uint32_t u_nbr_off = u_nbr_tile_id * warpSize + lid;
    if (v_off < num_can_u) [[likely]]
    {
      local_v = fast_load(s_can_u + v_off);
      local_off_st = fast_load(offsets_ + local_v);
      local_v_degree = fast_load(degree_ + local_v);
    }
    else
      local_v = UINT32_MAX;
    __syncwarp();

    // if (lid == 0)
    // warp_count[wid] = 0;
    // __syncwarp();

    // computing
    vtype v_nbr = UINT32_MAX;
    if (u_nbr_off < num_can_u_nbr)
      v_nbr = fast_load(s_can_u_nbr + u_nbr_off);
    __syncwarp();
    bool found = false;

    uint32_t min_val = __shfl_sync(FULL_MASK, v_nbr, 0);
    uint32_t max_lane = 31, max_val;
    while (max_lane > 0)
    {
      max_val = __shfl_sync(FULL_MASK, v_nbr, max_lane);
      if (max_val != UINT32_MAX)
        break;
      --max_lane;
    }

    for (int off = local_off_st; off < local_off_st + local_v_degree; off += warpSize)
    {
      const int my_idx = off + lid;
      const vtype nbr = my_idx < local_off_st + local_v_degree ? fast_load(nbrs_ + my_idx) : UINT32_MAX;
      const uint32_t remaining = (local_off_st + local_v_degree) - off;
      const uint32_t valid = remaining > warpSize ? warpSize : remaining;

      const vtype min_u_nbr = __shfl_sync(FULL_MASK, nbr, 0);
      const vtype max_u_nbr = __shfl_sync(FULL_MASK, nbr, valid - 1);
      if (max_val < min_u_nbr || min_val > max_u_nbr)
      {
        __syncwarp();
        continue;
      }

      for (int src = 0; src < valid; ++src)
      {
        const vtype bcast_nbr = __shfl_sync(FULL_MASK, nbr, src);
        if (bcast_nbr == v_nbr)
        {
          found = true;
        }
      }
      __syncwarp();
    }
    found = found && v_nbr != local_v;
    // if (v_nbr != local_v)
    // found = binary_search(nbrs_ + local_off_st, local_v_degree, v_nbr);
    // __syncwarp();

    uint32_t found_mask = __ballot_sync(FULL_MASK, found);
    if (lid == 0)
    {
      // atomicAdd(&num_res_for_each[v_off], __popc(found_mask));
      bitmap_mask_[v_off * mask_length + u_nbr_tile_id] = found_mask;
    }
    __syncwarp();
    // if (found_mask && lid == 0)
    //   warp_count[wid] += __popc(found_mask);
  }
}

// not work.
__global__ void
first_join_count_global_merge_path(
    // data graph
    offtype *offsets_, vtype *nbrs_, degtype *degree_,
    uint32_t *bitmap_mask_, uint32_t mask_length,
    vtype u, vtype u_nbr,
    numtype num_can_u, numtype num_can_u_nbr,
    vtype *d_u_candidate_vs_)
{
  const uint32_t tid = threadIdx.x;
  const uint32_t bid = blockIdx.x;
  const uint32_t idx = tid + bid * blockDim.x;
  const uint32_t lid = tid & 31;
  const uint32_t wid_g = idx >> 5;
  const uint32_t num_warps = (blockDim.x >> 5) * gridDim.x;

  vtype *s_can_u = d_u_candidate_vs_ + u * C_MAX_L_FREQ;
  vtype *s_can_u_nbr = d_u_candidate_vs_ + u_nbr * C_MAX_L_FREQ;

  uint32_t local_v;
  offtype local_off_st;
  uint32_t local_v_degree;

  uint32_t num_can_u_tiles = (num_can_u + warpSize - 1) / warpSize;
  uint32_t num_can_u_nbr_tiles = (num_can_u_nbr + warpSize - 1) / warpSize;
  uint32_t total_tiles = num_can_u_tiles * num_can_u_nbr_tiles;

  for (uint32_t task_id = wid_g; task_id < total_tiles; task_id += num_warps)
  {
    uint32_t u_tile_id = task_id / num_can_u_nbr_tiles;
    uint32_t u_nbr_tile_id = task_id % num_can_u_nbr_tiles;
    uint32_t v_off = u_tile_id * warpSize + lid;
    uint32_t u_nbr_tile_base = u_nbr_tile_id * warpSize;
    uint32_t tile_len = 0;
    if (u_nbr_tile_base < num_can_u_nbr)
    {
      tile_len = num_can_u_nbr - u_nbr_tile_base;
      if (tile_len > warpSize)
        tile_len = warpSize;
    }
    const vtype *tile_vals = s_can_u_nbr + u_nbr_tile_base;
    uint32_t lane_tile_val = UINT32_MAX;
    if (lid < tile_len)
      lane_tile_val = fast_load(tile_vals + lid);

    if (v_off < num_can_u) [[likely]]
    {
      local_v = fast_load(s_can_u + v_off);
      local_off_st = fast_load(offsets_ + local_v);
      local_v_degree = fast_load(degree_ + local_v);
    }
    else
    {
      local_v = UINT32_MAX;
      local_off_st = 0;
      local_v_degree = 0;
    }
    __syncwarp();

    for (int cur_task_id = 0; cur_task_id < warpSize; ++cur_task_id)
    {
      vtype shared_v = __shfl_sync(FULL_MASK, local_v, cur_task_id);
      if (shared_v == UINT32_MAX)
        break;
      offtype shared_off_st = __shfl_sync(FULL_MASK, local_off_st, cur_task_id);
      uint32_t shared_degree = __shfl_sync(FULL_MASK, local_v_degree, cur_task_id);
      uint32_t shared_v_off = u_tile_id * warpSize + cur_task_id;

      if (tile_len == 0 || shared_degree == 0)
      {
        if (lid == cur_task_id)
          bitmap_mask_[shared_v_off * mask_length + u_nbr_tile_id] = 0;
        __syncwarp();
        continue;
      }

      const vtype *nbr_list = nbrs_ + shared_off_st;
      uint32_t total = shared_degree + tile_len;
      uint32_t diag_start = (static_cast<uint64_t>(lid) * total) / warpSize;
      uint32_t diag_end = (static_cast<uint64_t>(lid + 1) * total) / warpSize;
      if (diag_start > total)
        diag_start = total;
      if (diag_end > total)
        diag_end = total;

      uint32_t lane_mask = 0;
      if (diag_start != diag_end)
      {
        uint32_t a_start, b_start, a_end, b_end;
        merge_path_search_device(nbr_list, shared_degree, tile_vals, tile_len,
                                 diag_start, a_start, b_start);
        merge_path_search_device(nbr_list, shared_degree, tile_vals, tile_len,
                                 diag_end, a_end, b_end);

        uint32_t a_idx = a_start;
        uint32_t b_idx = b_start;
        while (a_idx < a_end && b_idx < b_end)
        {
          vtype a_val = fast_load(nbr_list + a_idx);
          vtype b_val = __shfl_sync(FULL_MASK, lane_tile_val, b_idx);
          if (a_val == b_val)
          {
            lane_mask |= (1u << b_idx);
            ++a_idx;
            ++b_idx;
          }
          else if (a_val < b_val)
          {
            ++a_idx;
          }
          else
          {
            ++b_idx;
          }
        }
        while (b_idx < b_end)
        {
          vtype b_val = __shfl_sync(FULL_MASK, lane_tile_val, b_idx);
          bool found = false;
          if (a_idx < shared_degree)
          {
            found = (fast_load(nbr_list + a_idx) == b_val);
          }
          if (!found && a_idx > 0)
          {
            found = (fast_load(nbr_list + a_idx - 1) == b_val);
          }
          if (found)
            lane_mask |= (1u << b_idx);
          ++b_idx;
        }
      }

      for (int offset = 16; offset > 0; offset >>= 1)
        lane_mask |= __shfl_xor_sync(FULL_MASK, lane_mask, offset);

      if (lid == 0)
        bitmap_mask_[shared_v_off * mask_length + u_nbr_tile_id] = lane_mask;
    }
  }
}

__global__ void
first_join_write_new(
    offtype *offsets_, vtype *nbrs_, degtype *degree_,
    uint32_t *bitmap_mask_, uint32_t mask_length,
    vtype u, vtype u_nbr,
    numtype num_can_u, numtype num_can_u_nbr,
    vtype *d_u_candidate_vs_,
    vtype *d_res, offtype *d_offsets_each_row, uint32_t *d_offsets_each_tile, numtype scan_res_length,
    uint64_t *d_parent_res)
{
  const uint32_t tid = threadIdx.x;
  const uint32_t bid = blockIdx.x;
  const uint32_t idx = tid + bid * blockDim.x;
  const uint32_t wid = tid >> 5;
  const uint32_t lid = tid & 31;
  const uint32_t wid_g = idx >> 5;
  const uint32_t num_warps = (blockDim.x >> 5) * gridDim.x;

  extern __shared__ vtype s_candidates[];

  vtype *s_can_u = s_candidates;
  vtype *s_can_u_nbr = s_candidates + num_can_u;

  // load data to shared memory
  for (int i = tid; i < num_can_u; i += blockDim.x)
    s_can_u[i] = fast_load(d_u_candidate_vs_ + u * C_MAX_L_FREQ + i);
  for (int i = tid; i < num_can_u_nbr; i += blockDim.x)
    s_can_u_nbr[i] = fast_load(d_u_candidate_vs_ + u_nbr * C_MAX_L_FREQ + i);
  __syncthreads();

  uint32_t local_v;
  offtype local_off_base, local_off_end;
  uint32_t local_count;
  uint32_t shared_v;
  offtype shared_off_base, shared_off_end;
  uint32_t shared_count;
  uint32_t local_tile_off_base, local_tile_off_end, local_tile_count;
  uint32_t shared_tile_off_base, shared_tile_count;
  // bool local_flag, shared_flag;

  uint32_t num_can_u_tiles = (num_can_u + warpSize - 1) / warpSize;
  uint32_t num_can_u_nbr_tiles = (num_can_u_nbr + warpSize - 1) / warpSize;
  uint32_t total_tiles = num_can_u_tiles * num_can_u_nbr_tiles;

  // one warp, one task. (32 can(u)->iterate, 32 can(u_nbr)->one tile)
  for (int task_id = wid_g; task_id < total_tiles; task_id += num_warps)
  {
    uint32_t u_tile_id = task_id / num_can_u_nbr_tiles;
    uint32_t u_nbr_tile_id = task_id % num_can_u_nbr_tiles;
    uint32_t v_off = u_tile_id * warpSize + lid;
    uint32_t u_nbr_off = u_nbr_tile_id * warpSize + lid;

    if (v_off < num_can_u) [[likely]]
    {
      local_v = s_can_u[v_off];
      // local_v = fast_load(s_can_u + v_off);
      fast_load_pair(d_offsets_each_row + v_off, local_off_base, local_off_end);
      fast_load_pair(d_offsets_each_tile + v_off * scan_res_length + u_nbr_tile_id,
                     local_tile_off_base, local_tile_off_end);
      local_count = local_off_end - local_off_base;
      local_tile_count = local_tile_off_end - local_tile_off_base;
    }
    else
      local_v = UINT32_MAX;

    for (int cur_task_id = 0; cur_task_id < warpSize; ++cur_task_id)
    {
      shared_v = __shfl_sync(FULL_MASK, local_v, cur_task_id);
      if (shared_v == UINT32_MAX)
        break; // early exit

      uint32_t shared_v_off = v_off - lid + cur_task_id;
      uint32_t partial_mask = bitmap_mask_[shared_v_off * mask_length + u_nbr_tile_id];
      if (partial_mask == 0)
        continue;

      shared_count = __shfl_sync(FULL_MASK, local_count, cur_task_id);
      if (shared_count == 0)
        continue;
      shared_off_base = __shfl_sync(FULL_MASK, local_off_base, cur_task_id);
      shared_tile_off_base = __shfl_sync(FULL_MASK, local_tile_off_base, cur_task_id);
      shared_tile_count = __shfl_sync(FULL_MASK, local_tile_count, cur_task_id);

      bool set_flag = partial_mask & (1u << lid);
      uint32_t num_found = __popc(partial_mask);

      if (set_flag)
      {
        uint32_t my_rank = __popc(partial_mask & ((1u << lid) - 1));
        offtype write_pos = shared_off_base + shared_tile_off_base + my_rank;
        vtype v_nbr = s_can_u_nbr[u_nbr_off];
        d_res[write_pos] = v_nbr;
        d_parent_res[write_pos] = shared_v_off;
      }
      __syncwarp();
    }
  }
}

__global__ void
first_join_write_global_32_32(
    offtype *offsets_, vtype *nbrs_, degtype *degree_,
    uint32_t *bitmap_mask_, uint32_t mask_length,
    vtype u, vtype u_nbr,
    numtype num_can_u, numtype num_can_u_nbr,
    vtype *d_u_candidate_vs_,
    vtype *d_res, offtype *d_offsets_each_row, uint32_t *d_offsets_each_tile, numtype scan_res_length,
    uint64_t *d_parent_res)
{
  const uint32_t tid = threadIdx.x;
  const uint32_t bid = blockIdx.x;
  const uint32_t idx = tid + bid * blockDim.x;
  const uint32_t wid = tid >> 5;
  const uint32_t lid = tid & 31;
  const uint32_t wid_g = idx >> 5;
  const uint32_t num_warps = (blockDim.x >> 5) * gridDim.x;

  // extern __shared__ vtype s_candidates[];

  vtype *s_can_u = d_u_candidate_vs_ + u * C_MAX_L_FREQ;
  vtype *s_can_u_nbr = d_u_candidate_vs_ + u_nbr * C_MAX_L_FREQ;

  uint32_t local_v;
  offtype local_off_base, local_off_end;
  uint32_t local_count;
  uint32_t shared_v;
  offtype shared_off_base, shared_off_end;
  uint32_t shared_count;
  uint32_t local_tile_off_base, local_tile_off_end, local_tile_count;
  uint32_t shared_tile_off_base, shared_tile_count;
  // bool local_flag, shared_flag;

  uint32_t num_can_u_tiles = (num_can_u + warpSize - 1) / warpSize;
  uint32_t num_can_u_nbr_tiles = (num_can_u_nbr + warpSize - 1) / warpSize;
  uint32_t total_tiles = num_can_u_tiles * num_can_u_nbr_tiles;

  // one warp, one task. (32 can(u)->iterate, 32 can(u_nbr)->one tile)
  for (int task_id = wid_g; task_id < total_tiles; task_id += num_warps)
  {
    uint32_t u_tile_id = task_id / num_can_u_nbr_tiles;
    uint32_t u_nbr_tile_id = task_id % num_can_u_nbr_tiles;
    uint32_t v_off = u_tile_id * warpSize + lid;
    uint32_t u_nbr_off = u_nbr_tile_id * warpSize + lid;

    if (v_off < num_can_u) [[likely]]
    {
      local_v = fast_load(s_can_u + v_off);
      // local_v = fast_load(s_can_u + v_off);
      fast_load_pair(d_offsets_each_row + v_off, local_off_base, local_off_end);
      fast_load_pair(d_offsets_each_tile + v_off * scan_res_length + u_nbr_tile_id,
                     local_tile_off_base, local_tile_off_end);
      local_count = local_off_end - local_off_base;
      local_tile_count = local_tile_off_end - local_tile_off_base;
    }
    else
      local_v = UINT32_MAX;

    for (int cur_task_id = 0; cur_task_id < warpSize; ++cur_task_id)
    {
      shared_v = __shfl_sync(FULL_MASK, local_v, cur_task_id);
      if (shared_v == UINT32_MAX)
        break; // early exit

      uint32_t shared_v_off = v_off - lid + cur_task_id;
      uint32_t partial_mask = bitmap_mask_[shared_v_off * mask_length + u_nbr_tile_id];
      if (partial_mask == 0)
        continue;

      shared_count = __shfl_sync(FULL_MASK, local_count, cur_task_id);
      if (shared_count == 0)
        continue;
      shared_off_base = __shfl_sync(FULL_MASK, local_off_base, cur_task_id);
      shared_tile_off_base = __shfl_sync(FULL_MASK, local_tile_off_base, cur_task_id);
      shared_tile_count = __shfl_sync(FULL_MASK, local_tile_count, cur_task_id);

      bool set_flag = partial_mask & (1u << lid);
      uint32_t num_found = __popc(partial_mask);

      if (set_flag)
      {
        uint32_t my_rank = __popc(partial_mask & ((1u << lid) - 1));
        offtype write_pos = shared_off_base + shared_tile_off_base + my_rank;
        vtype v_nbr = fast_load(s_can_u_nbr + u_nbr_off);
        d_res[write_pos] = v_nbr;
        d_parent_res[write_pos] = shared_v_off;
      }
      __syncwarp();
    }
  }
}

__global__ void
first_join_write_global_1_32(
    offtype *offsets_, vtype *nbrs_, degtype *degree_,
    uint32_t *bitmap_mask_, uint32_t mask_length,
    vtype u, vtype u_nbr,
    numtype num_can_u, numtype num_can_u_nbr,
    vtype *d_u_candidate_vs_,
    vtype *d_res, offtype *d_offsets_each_row, uint32_t *d_offsets_each_tile, numtype scan_res_length,
    uint64_t *d_parent_res)
{
  const uint32_t tid = threadIdx.x;
  const uint32_t bid = blockIdx.x;
  const uint32_t idx = tid + bid * blockDim.x;
  const uint32_t wid = tid >> 5;
  const uint32_t lid = tid & 31;
  const uint32_t wid_g = idx >> 5;
  const uint32_t num_warps = (blockDim.x >> 5) * gridDim.x;

  // extern __shared__ vtype s_candidates[];

  vtype *s_can_u = d_u_candidate_vs_ + u * C_MAX_L_FREQ;
  vtype *s_can_u_nbr = d_u_candidate_vs_ + u_nbr * C_MAX_L_FREQ;

  uint32_t local_v;
  offtype local_off_base, local_off_end;
  uint32_t local_count;
  uint32_t local_tile_off_base, local_tile_off_end, local_tile_count;
  // bool local_flag, shared_flag;

  uint32_t num_can_u_tiles = num_can_u;
  uint32_t num_can_u_nbr_tiles = (num_can_u_nbr + warpSize - 1) / warpSize;
  uint32_t total_tiles = num_can_u_tiles * num_can_u_nbr_tiles;

  // one warp, one task. (1 can(u)->iterate, 32 can(u_nbr)->one tile)
  for (int task_id = wid_g; task_id < total_tiles; task_id += num_warps)
  {
    uint32_t u_tile_id = task_id / num_can_u_nbr_tiles;
    uint32_t u_nbr_tile_id = task_id % num_can_u_nbr_tiles;
    uint32_t v_off = u_tile_id;
    uint32_t u_nbr_off = u_nbr_tile_id * warpSize + lid;

    if (v_off < num_can_u) [[likely]]
    {
      local_v = fast_load(s_can_u + v_off);
      // local_v = fast_load(s_can_u + v_off);
      fast_load_pair(d_offsets_each_row + v_off, local_off_base, local_off_end);
      fast_load_pair(d_offsets_each_tile + v_off * scan_res_length + u_nbr_tile_id,
                     local_tile_off_base, local_tile_off_end);
      local_count = local_off_end - local_off_base;
      local_tile_count = local_tile_off_end - local_tile_off_base;
    }
    else
      local_v = UINT32_MAX;

    uint32_t partial_mask = bitmap_mask_[v_off * mask_length + u_nbr_tile_id];
    if (partial_mask == 0)
      continue;

    if (local_count == 0)
      continue;

    bool set_flag = partial_mask & (1u << lid);
    uint32_t num_found = __popc(partial_mask);

    if (set_flag)
    {
      uint32_t my_rank = __popc(partial_mask & ((1u << lid) - 1));
      offtype write_pos = local_off_base + local_tile_off_base + my_rank;
      vtype v_nbr = fast_load(s_can_u_nbr + u_nbr_off);
      d_res[write_pos] = v_nbr;
      d_parent_res[write_pos] = v_off;
    }
    __syncwarp();
  }
}

__global__ void
first_join_count_bitmap_task1(
    // data graph
    offtype *offsets_, vtype *nbrs_, degtype *degree_,
    uint32_t *d_bitmap_u_nbr, // bitmap from filter.
    vtype u, numtype num_can_u, vtype *d_u_candidate_vs_,
    uint32_t *neighbor_mask_, uint32_t *neighbor_mask_offsets_, uint32_t *workpool)
{
  const uint32_t tid = threadIdx.x;
  const uint32_t bid = blockIdx.x;
  const uint32_t idx = tid + bid * blockDim.x;
  const uint32_t wid = tid >> 5;
  const uint32_t lid = tid & 31;
  const uint32_t wid_g = idx >> 5;

  // vtype *s_can_u_nbr = d_u_candidate_vs_ + u_nbr * C_MAX_L_FREQ;

  uint32_t v_off = wid_g; // idx in can(u)
  vtype local_v;          // can(u)[task_id]
  offtype nbr_off_st, nbr_off_ed, my_off;
  vtype v_nbr;
  bool found = false;
  while (v_off < num_can_u)
  {
    local_v = fast_load(d_u_candidate_vs_ + v_off);
    fast_load_pair(offsets_ + local_v, nbr_off_st, nbr_off_ed);

    for (uint32_t base_off = nbr_off_st; base_off < nbr_off_ed; base_off += warpSize)
    {
      found = false;
      my_off = base_off + lid;
      if (my_off < nbr_off_ed)
      {
        v_nbr = fast_load(nbrs_ + my_off);
        found = (d_bitmap_u_nbr[v_nbr >> 5] & (1u << (v_nbr & 31)));
      }
      uint32_t found_mask = __ballot_sync(FULL_MASK, found);
      if (found_mask && lid == 0)
        neighbor_mask_[neighbor_mask_offsets_[local_v] + (base_off - nbr_off_st) / 32] = found_mask;
      __syncwarp();
    }

    // final
    if (lid == 0)
      v_off = atomicAdd(workpool, 1u);
    v_off = __shfl_sync(FULL_MASK, v_off, 0);
  }
}

__global__ void
first_join_write_bitmap_task1(
    offtype *offsets_, vtype *nbrs_, degtype *degree_,
    uint32_t *neighbor_mask_, uint32_t *neighbor_mask_offsets_, uint32_t *scan_result,
    vtype u, numtype num_can_u, vtype *d_u_candidate_vs_,
    vtype *d_res, offtype *d_offsets_each_row, uint64_t *d_parent_res,
    uint32_t *workpool)
{
  const uint32_t tid = threadIdx.x;
  const uint32_t bid = blockIdx.x;
  const uint32_t idx = tid + bid * blockDim.x;
  const uint32_t wid = tid >> 5;
  const uint32_t lid = tid & 31;
  const uint32_t wid_g = idx >> 5;

  uint32_t task_id = wid_g; // idx in can(u)
  uint32_t local_v;         // can(u)[task_id]
  offtype local_off_st, offset_end;
  while (task_id < num_can_u)
  {
    local_v = fast_load(d_u_candidate_vs_ + task_id);
    fast_load_pair(offsets_ + local_v, local_off_st, offset_end);

    for (uint32_t base_off = local_off_st; base_off < offset_end; base_off += warpSize)
    {
      uint32_t mask_idx = neighbor_mask_offsets_[local_v] + (base_off - local_off_st) / 32;
      uint32_t mask = fast_load(neighbor_mask_ + mask_idx);
      if (mask == 0)
        continue; // Skip if no valid neighbors in this iteration
      if (lid == 0)
        neighbor_mask_[mask_idx] = 0; // reset for next use
      __syncwarp();

      uint32_t my_off = base_off + lid;
      vtype v_nbr = UINT32_MAX;
      bool found = false;

      if (my_off < offset_end)
      {
        v_nbr = fast_load(nbrs_ + my_off);
        // Get the bitmap mask offset for this neighbor position
        found = (mask & (1u << lid));
      }

      // Aggregate found flags across the warp
      uint32_t found_mask = __ballot_sync(FULL_MASK, found);

      // Compute the base write offset from scan_result
      uint32_t write_offset_base = fast_load(scan_result + mask_idx);

      if (found)
      {
        // Compute rank within the warp
        uint32_t my_rank = __popc(found_mask & ((1u << lid) - 1));
        uint32_t write_pos = write_offset_base + my_rank;

        // Write the result and parent
        d_res[write_pos] = v_nbr;
        d_parent_res[write_pos] = task_id;
      }
      __syncwarp();
    }

    // final
    if (lid == 0)
      task_id = atomicAdd(workpool, 1);
    task_id = __shfl_sync(FULL_MASK, task_id, 0);
  }
}

__global__ void
getMax(
    vtype *d_u_candidate_vs_, uint32_t num_can_u,
    uint32_t *scan_output, uint32_t *neighbor_mask_offsets_,
    uint32_t *dst, uint32_t *d_edge_offsets)
{
  const uint32_t tid = threadIdx.x;
  const uint32_t bid = blockIdx.x;
  const uint32_t idx = tid + bid * blockDim.x;
  const uint32_t wid = tid >> 5;
  const uint32_t lid = tid & 31;
  const uint32_t wid_g = idx >> 5;
  const uint32_t grid_size = blockDim.x * gridDim.x;

  using BlockReduce = cub::BlockReduce<uint32_t, BLOCK_DIM>;
  __shared__ typename BlockReduce::TempStorage block_reduce_storage;

  uint32_t max_val = 0;

  for (int base_v_off = wid_g << 5; base_v_off < num_can_u; base_v_off += grid_size)
  {
    int my_v_off = base_v_off + lid;
    if (my_v_off < num_can_u)
    {
      uint32_t my_vertex = fast_load(d_u_candidate_vs_ + my_v_off);
      uint32_t my_neighbor_mask_off = fast_load(neighbor_mask_offsets_ + my_vertex);
      uint32_t my_neighbor_mask_off_next = fast_load(neighbor_mask_offsets_ + my_vertex + 1);
      uint32_t my_offset = fast_load(scan_output + my_neighbor_mask_off);
      d_edge_offsets[my_v_off] = my_offset;
      uint32_t my_count = fast_load(scan_output + my_neighbor_mask_off_next) -
                          my_offset;
      max_val = max(max_val, my_count);
    }
  }
  __syncthreads();
  uint32_t block_max = BlockReduce(block_reduce_storage).Reduce(max_val, cub::Max());
  if (tid == 0)
    atomicMax(dst, block_max);
}

// launch by <<<1,1>>>
__global__ void add_last_for_exclusive_sum_div32ceil(
    uint32_t *org_data_array, uint32_t *scan_output_array, uint32_t org_length, uint32_t *dst)
{
  dst[0] =
      (fast_load(org_data_array + org_length - 1) + 31) / 32 +
      fast_load(scan_output_array + org_length - 1);
}

// launch by <<<1,1>>>
__global__ void add_last_for_exclusive_sum_popc(
    uint32_t *org_data_array, uint32_t *scan_output_array, uint32_t org_length, uint32_t *dst)
{
  dst[0] =
      __popc(fast_load(org_data_array + org_length - 1)) +
      fast_load(scan_output_array + org_length - 1);
}