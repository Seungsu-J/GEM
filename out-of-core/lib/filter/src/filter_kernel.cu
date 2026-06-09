#include "filter.h"
#include "filter_kernel.cuh"
#include "cuda_helpers.cuh"
#include "globals.h"

#include <cstring>
#include <cstdio>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/pipeline>

// Optimized atomic operations for common cases
__device__ __forceinline__ uint32_t fast_atomic_add(uint32_t *address, uint32_t val)
{
  // Check if all threads in warp target the same address
  auto active = cooperative_groups::coalesced_threads();

  // Get address from first active thread (cast properly for 64-bit pointers)
  uintptr_t first_addr_int = active.shfl(reinterpret_cast<uintptr_t>(address), 0);
  uint32_t *first_addr = reinterpret_cast<uint32_t *>(first_addr_int);
  bool same_address = (address == first_addr);

  // Use ballot to check if all threads target same address
  uint32_t same_addr_mask = active.ballot(same_address);

  if (same_addr_mask == active.ballot(true))
  {
    // All active threads target same address - use optimized path
    uint32_t total = cooperative_groups::reduce(active, val, cooperative_groups::plus<uint32_t>());

    uint32_t result = 0;
    if (active.thread_rank() == 0)
    {
      result = atomicAdd(address, total);
    }
    result = active.shfl(result, 0);

    // Calculate this thread's contribution offset
    uint32_t offset = 0;
    for (int i = 0; i < active.thread_rank(); ++i)
    {
      offset += active.shfl(val, i);
    }

    return result + offset;
  }
  else
  {
    // Different addresses - fall back to regular atomic
    return atomicAdd(address, val);
  }
}

// Performance monitoring structure
struct KernelStats
{
  uint64_t total_operations;
  uint64_t memory_accesses;
  uint32_t warp_efficiency;
};

__global__ void
buildQueryNLCSSM_kernel(
    degtype *query_out_degrees_,
    vltype *query_vLabels_,
    offtype *query_offsets_,
    vtype *query_nbrs_,
    uint32_t *d_query_nlc_table_)
{
  // Shared memory for frequently accessed query data
  __shared__ vltype s_query_vLabels[MAX_VQ];
  __shared__ offtype s_query_offsets[MAX_VQ + 1];

  // Optimized thread indexing with register optimization
  const uint32_t tid = threadIdx.x;
  const uint32_t bid = blockIdx.x;
  const uint32_t idx = tid + bid * blockDim.x;
  const uint32_t lid = tid & 31;   // lane ID within warp
  const uint32_t wid = tid >> 5;   // warp ID within block
  const uint32_t wid_g = idx >> 5; // global warp ID

  if (bid * WARP_PER_BLOCK >= C_NUM_VQ)
    return;
  // Collaborative loading of query data into shared memory
  if (wid == 0 && lid < C_NUM_VQ)
    s_query_vLabels[lid] = fast_load(&query_vLabels_[lid]);
  if (wid == 1 && lid <= C_NUM_VQ)
    s_query_offsets[lid] = fast_load(&query_offsets_[lid]);
  __syncthreads();

  // Early exit with coalesced check
  if (wid_g >= C_NUM_VQ)
    return;

  const offtype off_st = s_query_offsets[wid_g];
  const offtype off_end = s_query_offsets[wid_g + 1];

  /* no query vertex has more than 32 neighbors*/
  // for (offtype off = off_st + lid; off < off_end; off += warpSize)
  // {
  // auto group = cooperative_groups::coalesced_threads();
  uint32_t off = off_st + lid;
  if (off < off_end)
  {
    const vtype v_nbr = fast_load(&query_nbrs_[off]);
    const vltype v_nbr_label = s_query_vLabels[v_nbr]; // Fast shared memory access

    atomicAdd(&d_query_nlc_table_[wid_g * C_NUM_VLQ + v_nbr_label], 1);
  }
  __syncwarp();
  // fast_atomic_add(&d_query_nlc_table_[u * C_NUM_VLQ + v_nbr_label], 1);

  // group.sync();
  // }
}

// one warp - 32 data vertices.
__global__ void
LDfilterSSM_kernel_workpool(
    degtype *query_out_degrees_,
    vltype *query_vLabels_,
    offtype *d_offsets_,
    vtype *d_nbrs_,
    vltype *d_v_labels_,
    degtype *d_v_degrees_,
    uint32_t *d_query_nlc_table_,
    uint32_t *d_bitmap, uint32_t bitmap_pitch,
    uint32_t *workpool)
{
  const uint32_t tid = threadIdx.x;
  const uint32_t bid = blockIdx.x;
  const uint32_t idx = tid + bid * blockDim.x;
  const uint32_t wid = tid >> 5;
  const uint32_t lid = tid & 31;
  const uint32_t wid_g = idx >> 5;

  // Optimized shared memory layout for better bank conflict avoidance
  __shared__ vltype s_q_vlabels[MAX_VQ];
  __shared__ degtype s_q_degs[MAX_VQ];
  __shared__ numtype s_q_nlc_table[MAX_VQ][MAX_VQ];
  __shared__ uint32_t s_bitmap[MAX_VQ][WARP_PER_BLOCK];

  // Each warp processes 32 vertices, each thread handles one vertex's NLC table
  __shared__ uint32_t s_d_nlc_table[WARP_PER_BLOCK][32][MAX_VQ];

  __shared__ uint32_t num_tasks[WARP_PER_BLOCK];
  __shared__ uint32_t task_u[WARP_PER_BLOCK][MAX_VQ * 32];
  __shared__ uint32_t task_v_off_32[WARP_PER_BLOCK][MAX_VQ * 32];
  __shared__ uint32_t warp_bit[WARP_PER_BLOCK][MAX_VQ];

  // Stride between iterations (all warps process 32 vertices each iteration)
  // constexpr uint32_t warp_stride = GRID_DIM << 3 << 5; // GRID_DIM * WARP_PER_BLOCK * 32

  vtype v_nbr_off_st, v_nbr_off_ed;
  vtype bcast_v;
  vltype data_vlabel = 0;
  degtype data_deg = 0;

  // Optimized shared memory loading with vectorized access
  if (tid < C_NUM_VQ)
  {
    s_q_vlabels[tid] = fast_load(query_vLabels_ + tid);
    s_q_degs[tid] = fast_load(query_out_degrees_ + tid);
  }

  for (int i = tid; i < C_NUM_VQ * C_NUM_VLQ; i += blockDim.x)
  {
    const int u_idx = i / C_NUM_VLQ;
    const int vlabel_idx = i % C_NUM_VLQ;
    s_q_nlc_table[u_idx][vlabel_idx] = fast_load(d_query_nlc_table_ + i);
  }
  __syncthreads();

  uint32_t v_off_base = wid_g << 5;

  // Main processing loop - each warp processes 32 vertices, each thread handles 1 vertex
  // for (uint32_t v_off_base = wid_g << 5; v_off_base < C_NUM_VD; v_off_base += warp_stride)
  while (v_off_base < C_NUM_VD)
  {
    // Initialize warp_bit for all query vertices
    if (lid < C_NUM_VQ)
      warp_bit[wid][lid] = 0;
    if (lid == 0)
      num_tasks[wid] = 0;
    __syncwarp();

    for (uint32_t v_seq = 0; v_seq < 32; ++v_seq)
    {
      bcast_v = v_off_base + v_seq;
      if (bcast_v >= C_NUM_VD)
        break;

      data_vlabel = fast_load(d_v_labels_ + bcast_v);
      data_deg = fast_load(d_v_degrees_ + bcast_v);

      bool pass = false;
      if (lid < C_NUM_VQ)
        pass = (s_q_vlabels[lid] == data_vlabel) &&
               (s_q_degs[lid] <= data_deg + C_THRESHOLD);

      uint32_t data_v_pass = __ballot_sync(FULL_MASK, pass);
      if (!data_v_pass)
        continue;

      if (lid < MAX_VLQ)
        s_d_nlc_table[wid][v_seq][lid] = 0;
      __syncwarp();

      fast_load_pair(d_offsets_ + bcast_v, v_nbr_off_st, v_nbr_off_ed);

      for (offtype nbr_off = v_nbr_off_st + lid; nbr_off < v_nbr_off_ed; nbr_off += warpSize)
      {
        const vtype v_nbr = fast_load(d_nbrs_ + nbr_off);
        const vltype v_nbr_label = fast_load(d_v_labels_ + v_nbr);
        if (v_nbr_label < C_NUM_VLQ)
          atomicAdd(&s_d_nlc_table[wid][v_seq][v_nbr_label], 1);
      }
      __syncwarp();

      if (pass)
      {
        uint32_t my_rank = __popc(data_v_pass & ((1u << lid) - 1));
        uint32_t my_pos = num_tasks[wid] + my_rank;
        task_u[wid][my_pos] = lid;
        task_v_off_32[wid][my_pos] = v_seq;
      }
      __syncwarp();
      if (lid == 0)
        num_tasks[wid] += __popc(data_v_pass);
      __syncwarp();
    }

    for (int task_id = lid; task_id < num_tasks[wid]; task_id += warpSize)
    {
      auto group = cooperative_groups::coalesced_threads();
      vtype u = task_u[wid][task_id];
      vtype v_seq = task_v_off_32[wid][task_id];
      uint32_t total_diff = 0;
      // #pragma unroll
      for (int lbl = 0; lbl < C_NUM_VLQ; ++lbl)
      {
        const uint32_t q_count = s_q_nlc_table[u][lbl];
        const uint32_t d_count = s_d_nlc_table[wid][v_seq][lbl];
        const uint32_t diff = (q_count > d_count) ? (q_count - d_count) : 0;
        total_diff += diff;
      }
      if (total_diff <= C_THRESHOLD)
      {
        atomicOr(&warp_bit[wid][u], (1u << v_seq));
      }
      group.sync();
    }
    __syncwarp();

    if (num_tasks[wid] && lid < C_NUM_VQ && warp_bit[wid][lid])
      d_bitmap[lid * bitmap_pitch + (v_off_base >> 5)] = warp_bit[wid][lid];
    // atomicOr(&d_bitmap[lid * bitmap_pitch + (v_off_base >> 5)], warp_bit[wid][lid]);
    __syncwarp();

    if (lid == 0)
      v_off_base = atomicAdd(workpool, 32);
    v_off_base = __shfl_sync(FULL_MASK, v_off_base, 0);
  }
}

// one warp - 32 data vertices.
__global__ void
LDfilterSSM_kernel_sequence(
    degtype *query_out_degrees_,
    vltype *query_vLabels_,
    offtype *d_offsets_,
    vtype *d_nbrs_,
    vltype *d_v_labels_,
    degtype *d_v_degrees_,
    uint32_t *d_query_nlc_table_,
    uint32_t *d_bitmap, uint32_t bitmap_pitch)
{
  const uint32_t tid = threadIdx.x;
  const uint32_t bid = blockIdx.x;
  const uint32_t idx = tid + bid * blockDim.x;
  const uint32_t wid = tid >> 5;
  const uint32_t lid = tid & 31;
  const uint32_t wid_g = idx >> 5;

  // Optimized shared memory layout for better bank conflict avoidance
  __shared__ vltype s_q_vlabels[MAX_VQ];
  __shared__ degtype s_q_degs[MAX_VQ];
  __shared__ numtype s_q_nlc_table[MAX_VQ][MAX_VQ];
  __shared__ uint32_t s_bitmap[MAX_VQ][WARP_PER_BLOCK];

  // Each warp processes 32 vertices, each thread handles one vertex's NLC table
  __shared__ uint32_t s_d_nlc_table[WARP_PER_BLOCK][32][MAX_VQ];

  __shared__ uint32_t num_tasks[WARP_PER_BLOCK];
  __shared__ uint32_t task_u[WARP_PER_BLOCK][MAX_VQ * 32];
  __shared__ uint32_t task_v_off_32[WARP_PER_BLOCK][MAX_VQ * 32];
  __shared__ uint32_t warp_bit[WARP_PER_BLOCK][MAX_VQ];

  // Stride between iterations (all warps process 32 vertices each iteration)
  constexpr uint32_t warp_stride = GRID_DIM << 3 << 5; // GRID_DIM * WARP_PER_BLOCK * 32

  vtype v_nbr_off_st, v_nbr_off_ed;
  vtype bcast_v;
  vltype data_vlabel = 0;
  degtype data_deg = 0;

  // Optimized shared memory loading with vectorized access
  if (tid < C_NUM_VQ)
  {
    s_q_vlabels[tid] = fast_load(query_vLabels_ + tid);
    s_q_degs[tid] = fast_load(query_out_degrees_ + tid);
  }

  for (int i = tid; i < C_NUM_VQ * C_NUM_VLQ; i += blockDim.x)
  {
    const int u_idx = i / C_NUM_VLQ;
    const int vlabel_idx = i % C_NUM_VLQ;
    s_q_nlc_table[u_idx][vlabel_idx] = fast_load(d_query_nlc_table_ + i);
  }
  __syncthreads();

  // Main processing loop - each warp processes 32 vertices, each thread handles 1 vertex
  for (uint32_t v_off_base = wid_g << 5; v_off_base < C_NUM_VD; v_off_base += warp_stride)
  {
    // Initialize warp_bit for all query vertices
    if (lid < C_NUM_VQ)
      warp_bit[wid][lid] = 0;
    if (lid == 0)
      num_tasks[wid] = 0;
    __syncwarp();

    for (uint32_t v_seq = 0; v_seq < 32; ++v_seq)
    {
      bcast_v = v_off_base + v_seq;
      if (bcast_v >= C_NUM_VD)
        break;

      data_vlabel = fast_load(d_v_labels_ + bcast_v);
      data_deg = fast_load(d_v_degrees_ + bcast_v);

      bool pass = false;
      if (lid < C_NUM_VQ)
        pass = (s_q_vlabels[lid] == data_vlabel) &&
               (s_q_degs[lid] <= data_deg + C_THRESHOLD);

      uint32_t data_v_pass = __ballot_sync(FULL_MASK, pass);
      if (!data_v_pass)
        continue;

      if (lid < MAX_VLQ)
        s_d_nlc_table[wid][v_seq][lid] = 0;
      __syncwarp();

      fast_load_pair(d_offsets_ + bcast_v, v_nbr_off_st, v_nbr_off_ed);

      for (offtype nbr_off = v_nbr_off_st + lid; nbr_off < v_nbr_off_ed; nbr_off += warpSize)
      {
        const vtype v_nbr = fast_load(d_nbrs_ + nbr_off);
        const vltype v_nbr_label = fast_load(d_v_labels_ + v_nbr);
        if (v_nbr_label < C_NUM_VLQ)
          atomicAdd(&s_d_nlc_table[wid][v_seq][v_nbr_label], 1);
      }
      __syncwarp();

      if (pass)
      {
        uint32_t my_rank = __popc(data_v_pass & ((1u << lid) - 1));
        uint32_t my_pos = num_tasks[wid] + my_rank;
        task_u[wid][my_pos] = lid;
        task_v_off_32[wid][my_pos] = v_seq;
      }
      __syncwarp();
      if (lid == 0)
        num_tasks[wid] += __popc(data_v_pass);
      __syncwarp();
    }

    for (int task_id = lid; task_id < num_tasks[wid]; task_id += warpSize)
    {
      auto group = cooperative_groups::coalesced_threads();
      vtype u = task_u[wid][task_id];
      vtype v_seq = task_v_off_32[wid][task_id];
      uint32_t total_diff = 0;
      // #pragma unroll
      for (int lbl = 0; lbl < C_NUM_VLQ; ++lbl)
      {
        const uint32_t q_count = s_q_nlc_table[u][lbl];
        const uint32_t d_count = s_d_nlc_table[wid][v_seq][lbl];
        const uint32_t diff = (q_count > d_count) ? (q_count - d_count) : 0;
        total_diff += diff;
      }
      if (total_diff <= C_THRESHOLD)
      {
        atomicOr(&warp_bit[wid][u], (1u << v_seq));
      }
      group.sync();
    }
    __syncwarp();

    if (num_tasks[wid] && lid < C_NUM_VQ && warp_bit[wid][lid])
      d_bitmap[lid * bitmap_pitch + (v_off_base >> 5)] = warp_bit[wid][lid];
    // atomicOr(&d_bitmap[lid * bitmap_pitch + (v_off_base >> 5)], warp_bit[wid][lid]);
    __syncwarp();
  }
}

__global__ void
LDfilterSSM_kernel_workpool_4(
    degtype *query_out_degrees_,
    vltype *query_vLabels_,
    offtype *d_offsets_,
    vtype *d_nbrs_,
    vltype *d_v_labels_,
    degtype *d_v_degrees_,
    uint32_t *d_query_nlc_table_,
    uint32_t *d_bitmap, uint32_t bitmap_pitch, uint32_t *workpool)
{
  const uint32_t tid = threadIdx.x;
  const uint32_t bid = blockIdx.x;
  const uint32_t idx = tid + bid * blockDim.x;
  const uint32_t wid = tid >> 5;
  const uint32_t lid = tid & 31;
  const uint32_t wid_g = idx >> 5;

  // Optimized shared memory layout for better bank conflict avoidance
  __shared__ vltype s_q_vlabels[MAX_VQ];
  __shared__ degtype s_q_degs[MAX_VQ];
  __shared__ uint32_t s_bitmap[MAX_VQ][WARP_PER_BLOCK];
  __shared__ numtype s_q_nlc_table[MAX_VQ][MAX_VLQ];

  __shared__ uint32_t s_d_nlc_table[WARP_PER_BLOCK][4][MAX_VLQ];

  __shared__ uint32_t num_tasks[WARP_PER_BLOCK];
  __shared__ uint32_t task_u[WARP_PER_BLOCK][MAX_VQ * 4];
  __shared__ uint32_t task_v_off_4[WARP_PER_BLOCK][MAX_VQ * 4];
  __shared__ uint32_t warp_bit[WARP_PER_BLOCK][MAX_VQ];
  __shared__ uint32_t block_bit[WARP_PER_BLOCK][MAX_VQ];

  // Stride between iterations (all warps process 4 vertices each iteration)
  // const uint32_t warp_stride = gridDim.x * WARP_PER_BLOCK * 4;

  uint32_t v_off_base = wid_g * 4;

  vtype v_nbr_off_st, v_nbr_off_ed;
  vtype bcast_v;
  vltype data_vlabel = 0;
  degtype data_deg = 0;

  // Optimized shared memory loading with vectorized access
  if (tid < C_NUM_VQ)
  {
    s_q_vlabels[tid] = fast_load(query_vLabels_ + tid);
    s_q_degs[tid] = fast_load(query_out_degrees_ + tid);
  }

  for (int i = tid; i < C_NUM_VQ * C_NUM_VLQ; i += blockDim.x)
  {
    const int u_idx = i / C_NUM_VLQ;
    const int vlabel_idx = i % C_NUM_VLQ;
    s_q_nlc_table[u_idx][vlabel_idx] = fast_load(d_query_nlc_table_ + i);
  }
  __syncthreads();

  // Main processing loop - each warp processes 4 vertices, then jumps by warp_stride
  // for (uint32_t v_off_base = wid_g * 4; v_off_base < C_NUM_VD; v_off_base += warp_stride)
  while (v_off_base < C_NUM_VD)
  {
    // Initialize warp_bit for all query vertices
    if (lid < C_NUM_VQ)
      warp_bit[wid][lid] = 0;
    if (lid == 0)
      num_tasks[wid] = 0;
    __syncwarp();

    // Process each of the 4 vertices sequentially to reuse shared NLC table
    for (uint32_t v_seq = 0; v_seq < 4; ++v_seq)
    {
      bcast_v = v_off_base + v_seq;

      if (bcast_v >= C_NUM_VD)
        break;

      // Load vertex properties with broadcast (no shared memory needed)
      data_vlabel = fast_load(d_v_labels_ + bcast_v);
      data_deg = fast_load(d_v_degrees_ + bcast_v);

      // early check
      bool pass = false;
      if (lid < C_NUM_VQ)
        pass =
            (s_q_vlabels[lid] == data_vlabel) &&
            (s_q_degs[lid] <= data_deg + C_THRESHOLD);

      uint32_t data_v_pass = __ballot_sync(FULL_MASK, pass);
      if (!data_v_pass)
        continue;

      // Initialize shared NLC table for this data vertex
      if (lid < MAX_VLQ)
        s_d_nlc_table[wid][v_seq][lid] = 0;
      __syncwarp();

      fast_load_pair(&d_offsets_[bcast_v], v_nbr_off_st, v_nbr_off_ed);

      // Build NLC table - all threads collaborate
      for (offtype nbr_off = v_nbr_off_st + lid; nbr_off < v_nbr_off_ed; nbr_off += warpSize)
      {
        const vtype v_nbr = fast_load(d_nbrs_ + nbr_off);
        const vltype v_nbr_label = fast_load(d_v_labels_ + v_nbr);
        if (v_nbr_label < C_NUM_VLQ)
          atomicAdd(&s_d_nlc_table[wid][v_seq][v_nbr_label], 1);
      }
      __syncwarp();

      if (pass)
      {
        uint32_t my_rank = __popc(data_v_pass & ((1u << lid) - 1));
        uint32_t my_pos = num_tasks[wid] + my_rank;
        task_u[wid][my_pos] = lid;
        task_v_off_4[wid][my_pos] = v_seq;
      }
      __syncwarp();
      if (lid == 0)
        num_tasks[wid] += __popc(data_v_pass);
      __syncwarp();
    }

    for (int task_id = lid; task_id < num_tasks[wid]; task_id += warpSize)
    {
      auto group = cooperative_groups::coalesced_threads();
      vtype u = task_u[wid][task_id];
      vtype v_seq = task_v_off_4[wid][task_id];
      uint32_t total_diff = 0;
      // #pragma unroll
      for (int lbl = 0; lbl < C_NUM_VLQ; ++lbl)
      {
        const uint32_t q_count = s_q_nlc_table[u][lbl];
        const uint32_t d_count = s_d_nlc_table[wid][v_seq][lbl];
        const uint32_t diff = (q_count > d_count) ? (q_count - d_count) : 0;
        total_diff += diff;
      }
      if (total_diff <= C_THRESHOLD)
      {
        atomicOr(&warp_bit[wid][u], (1u << v_seq << (wid * 4)));
        // atomicOr(&warp_bit[wid][u], (1u << ((v_off_base | v_seq) & 31)));
      }
      group.sync();
    }
    __syncwarp();

    if (num_tasks[wid] && lid < C_NUM_VQ && warp_bit[wid][lid])
      atomicOr(&d_bitmap[lid * bitmap_pitch + (v_off_base >> 5)], warp_bit[wid][lid]);
    __syncwarp();

    if (lid == 0)
      v_off_base = atomicAdd(workpool, 4);
    v_off_base = __shfl_sync(FULL_MASK, v_off_base, 0);
  }
}

__global__ void
LDfilterSSM_kernel_4(
    degtype *query_out_degrees_,
    vltype *query_vLabels_,
    offtype *d_offsets_,
    vtype *d_nbrs_,
    vltype *d_v_labels_,
    degtype *d_v_degrees_,
    uint32_t *d_query_nlc_table_,
    uint32_t *d_bitmap, uint32_t bitmap_pitch)
{
  const uint32_t tid = threadIdx.x;
  const uint32_t bid = blockIdx.x;
  const uint32_t idx = tid + bid * blockDim.x;
  const uint32_t wid = tid >> 5;
  const uint32_t lid = tid & 31;
  const uint32_t wid_g = idx >> 5;

  // Optimized shared memory layout for better bank conflict avoidance
  __shared__ vltype s_q_vlabels[MAX_VQ];
  __shared__ degtype s_q_degs[MAX_VQ];
  __shared__ uint32_t s_bitmap[MAX_VQ][WARP_PER_BLOCK];
  __shared__ numtype s_q_nlc_table[MAX_VQ][MAX_VLQ];

  __shared__ uint32_t s_d_nlc_table[WARP_PER_BLOCK][4][MAX_VLQ];

  __shared__ uint32_t num_tasks[WARP_PER_BLOCK];
  __shared__ uint32_t task_u[WARP_PER_BLOCK][MAX_VQ * 4];
  __shared__ uint32_t task_v_off_4[WARP_PER_BLOCK][MAX_VQ * 4];
  __shared__ uint32_t warp_bit[WARP_PER_BLOCK][MAX_VQ];
  __shared__ uint32_t block_bit[WARP_PER_BLOCK][MAX_VQ];

  // Stride between iterations (all warps process 4 vertices each iteration)
  // const uint32_t warp_stride = gridDim.x * WARP_PER_BLOCK * 4;
  constexpr uint32_t warp_stride = GRID_DIM << 3 << 2; // WARP_PER_BLOCK = 8, each warp 4 vertices.

  vtype v_nbr_off_st, v_nbr_off_ed;
  vtype bcast_v;
  vltype data_vlabel = 0;
  degtype data_deg = 0;

  // Optimized shared memory loading with vectorized access
  if (tid < C_NUM_VQ)
  {
    s_q_vlabels[tid] = fast_load(query_vLabels_ + tid);
    s_q_degs[tid] = fast_load(query_out_degrees_ + tid);
  }

  for (int i = tid; i < C_NUM_VQ * C_NUM_VLQ; i += blockDim.x)
  {
    const int u_idx = i / C_NUM_VLQ;
    const int vlabel_idx = i % C_NUM_VLQ;
    s_q_nlc_table[u_idx][vlabel_idx] = fast_load(d_query_nlc_table_ + i);
  }
  __syncthreads();

  // Main processing loop - each warp processes 4 vertices, then jumps by warp_stride
  for (uint32_t v_off_base = wid_g * 4; v_off_base < C_NUM_VD; v_off_base += warp_stride)
  {
    // Initialize warp_bit for all query vertices
    if (lid < C_NUM_VQ)
      warp_bit[wid][lid] = 0;
    if (lid == 0)
      num_tasks[wid] = 0;
    __syncwarp();

    // Process each of the 4 vertices sequentially to reuse shared NLC table
    for (uint32_t v_seq = 0; v_seq < 4; ++v_seq)
    {
      bcast_v = v_off_base + v_seq;

      if (bcast_v >= C_NUM_VD)
        break;

      // Load vertex properties with broadcast (no shared memory needed)
      data_vlabel = fast_load(d_v_labels_ + bcast_v);
      data_deg = fast_load(d_v_degrees_ + bcast_v);

      // early check
      bool pass = false;
      if (lid < C_NUM_VQ)
        pass =
            (s_q_vlabels[lid] == data_vlabel) &&
            (s_q_degs[lid] <= data_deg + C_THRESHOLD);

      uint32_t data_v_pass = __ballot_sync(FULL_MASK, pass);
      if (!data_v_pass)
        continue;

      // Initialize shared NLC table for this data vertex
      if (lid < MAX_VLQ)
        s_d_nlc_table[wid][v_seq][lid] = 0;
      __syncwarp();

      fast_load_pair(&d_offsets_[bcast_v], v_nbr_off_st, v_nbr_off_ed);

      // Build NLC table - all threads collaborate
      for (offtype nbr_off = v_nbr_off_st + lid; nbr_off < v_nbr_off_ed; nbr_off += warpSize)
      {
        const vtype v_nbr = fast_load(d_nbrs_ + nbr_off);
        const vltype v_nbr_label = fast_load(d_v_labels_ + v_nbr);
        if (v_nbr_label < C_NUM_VLQ)
          atomicAdd(&s_d_nlc_table[wid][v_seq][v_nbr_label], 1);
      }
      __syncwarp();

      if (pass)
      {
        uint32_t my_rank = __popc(data_v_pass & ((1u << lid) - 1));
        uint32_t my_pos = num_tasks[wid] + my_rank;
        task_u[wid][my_pos] = lid;
        task_v_off_4[wid][my_pos] = v_seq;
      }
      __syncwarp();
      if (lid == 0)
        num_tasks[wid] += __popc(data_v_pass);
      __syncwarp();
    }

    for (int task_id = lid; task_id < num_tasks[wid]; task_id += warpSize)
    {
      auto group = cooperative_groups::coalesced_threads();
      vtype u = task_u[wid][task_id];
      vtype v_seq = task_v_off_4[wid][task_id];
      uint32_t total_diff = 0;
      // #pragma unroll
      for (int lbl = 0; lbl < C_NUM_VLQ; ++lbl)
      {
        const uint32_t q_count = s_q_nlc_table[u][lbl];
        const uint32_t d_count = s_d_nlc_table[wid][v_seq][lbl];
        const uint32_t diff = (q_count > d_count) ? (q_count - d_count) : 0;
        total_diff += diff;
      }
      if (total_diff <= C_THRESHOLD)
      {
        atomicOr(&warp_bit[wid][u], (1u << v_seq << (wid * 4)));
        // atomicOr(&warp_bit[wid][u], (1u << ((v_off_base | v_seq) & 31)));
      }
      group.sync();
    }
    __syncwarp();

    if (num_tasks[wid] && lid < C_NUM_VQ && warp_bit[wid][lid])
      atomicOr(&d_bitmap[lid * bitmap_pitch + (v_off_base >> 5)], warp_bit[wid][lid]);
    __syncwarp();
  }
}

__global__ void
write_results(
    uint32_t *d_bitmap, uint32_t bitmap_pitch,
    uint32_t *d_offset_for_each,
    vtype *d_u_candidate_vs_, numtype *d_num_u_candidate_vs_)
{
  const uint32_t tid = threadIdx.x;
  const uint32_t bid = blockIdx.x;
  const uint32_t wid = tid >> 5;
  const uint32_t lid = tid & 31;
  const uint32_t wid_g = (bid * WARP_PER_BLOCK + wid);

  // Assuming bitmap_pitch is padded to be divisible by 4
  // This eliminates all boundary checking!
  const uint32_t tasks_per_u = bitmap_pitch >> 2; // bitmap_pitch / 4
  const uint32_t total_tasks = C_NUM_VQ * tasks_per_u;
  const uint32_t warp_stride = GRID_DIM * WARP_PER_BLOCK;

  // Shared memory buffer for coalesced writes
  __shared__ uint32_t count[WARP_PER_BLOCK];
  __shared__ uint32_t buffer[WARP_PER_BLOCK][128];

  for (uint32_t task_id = wid_g; task_id < total_tasks; task_id += warp_stride)
  {
    if (lid == 0)
      count[wid] = 0;
    __syncwarp();

    const uint32_t u = task_id / tasks_per_u;
    const uint32_t block_group = task_id % tasks_per_u;
    const uint32_t base_block = block_group << 2; // block_group * 4

    // Load 4 bitmap blocks - always valid due to padding
    const uint4 bitmap_val4 = fast_load_uint4(&d_bitmap[u * bitmap_pitch + base_block]);
    const uint4 offset_val4 = fast_load_uint4(&d_offset_for_each[u * bitmap_pitch + base_block]);

    const uint32_t bitmap_vals[4] = {bitmap_val4.x, bitmap_val4.y, bitmap_val4.z, bitmap_val4.w};

// Accumulate vertex IDs into shared buffer
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
      const uint32_t block_idx = base_block + i;
      const uint32_t bitmap = bitmap_vals[i];

      if (bitmap & (1u << lid))
      {
        const uint32_t rank_in_block = __popc(bitmap & ((1u << lid) - 1));
        const uint32_t buf_pos = count[wid] + rank_in_block;
        const uint32_t vertex_id = (block_idx << 5) + lid; // block_idx * 32 + lid
        buffer[wid][buf_pos] = vertex_id;
      }

      if (lid == 0)
        count[wid] += __popc(bitmap);
      __syncwarp();
    }

    // Coalesced write from buffer to global memory
    const uint32_t base_offset = offset_val4.x;
    for (uint32_t i = lid; i < count[wid]; i += WARP_SIZE)
    {
      const uint32_t global_pos = u * C_MAX_L_FREQ + base_offset + i;
      const uint32_t vertex_id = buffer[wid][i];
      // Only check if vertex_id is valid (in case we loaded padded zeros)
      if (vertex_id < C_NUM_VD)
        d_u_candidate_vs_[global_pos] = vertex_id;
    }
    __syncwarp();
  }
}

__global__ void
add_kernel(
    uint32_t *d_bitmap,
    uint32_t *d_offset_for_each,
    uint32_t bitmap_pitch,
    uint32_t *d_num_u_candidate_vs_)
{
  const uint32_t tid = threadIdx.x;
  const uint32_t wid_g = (tid + blockIdx.x * blockDim.x) >> 5;
  const uint32_t lid = tid & 31;
  if (wid_g < C_NUM_VQ && lid == 0)
  {
    uint32_t last_offset = __ldg(d_offset_for_each + wid_g * bitmap_pitch + bitmap_pitch - 1);
    uint32_t last_value = __ldg(d_bitmap + wid_g * bitmap_pitch + bitmap_pitch - 1);
    uint32_t res = last_offset + __popc(last_value);
    d_num_u_candidate_vs_[wid_g] = res;
  }
  __syncwarp();
}