#include "cuckooHash.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cooperative_groups.h>

__global__ void buildHashKeys(
    const uint32_t *in,        // Can(u)
    const uint32_t in_size,    // |Can(u)|
    uint32_t *keys0,           // i-th edge, key[0]
    uint32_t *keys1,           // i-th edge, key[1]
    const uint32_t num_bucket, // i-th edge num buckets
    const uint32_t C0,
    const uint32_t C1,
    const uint32_t C2,
    const uint32_t C3,
    uint32_t *progress,
    uint32_t *success // progess+2;
)
{
  // a warp inserts at most 4 keys to the hash table at a time
  __shared__ uint32_t v[WARP_PER_BLOCK][4];
  __shared__ uint32_t bucket_index[WARP_PER_BLOCK][4];
  __shared__ uint32_t table_index[WARP_PER_BLOCK][4];

  __shared__ uint32_t idx_start[WARP_PER_BLOCK];
  __shared__ uint32_t nloops[WARP_PER_BLOCK];

  __shared__ uint32_t *keys[2];
  __shared__ uint32_t C[2][2];

  if (threadIdx.x == 0)
  {
    keys[0] = keys0;
    keys[1] = keys1;
    C[0][0] = C0;
    C[0][1] = C1;
    C[1][0] = C2;
    C[1][1] = C3;
  }
  __syncthreads();

  uint32_t warp_id = threadIdx.x / WARP_SIZE;
  uint32_t lane_id = threadIdx.x % WARP_SIZE;
  uint32_t mask = 0xff << (lane_id / 8) * 8; // divide a warp into four parts. 32/8=4

  uint32_t pre_value, result, leader, elem_idx = lane_id / 8;

  while (true)
  {
    if (lane_id == 0)
      idx_start[warp_id] = atomicAdd(progress, 4u); // a warp inserts 4 elems.
    __syncwarp();

    if (idx_start[warp_id] > in_size || *success != 0u)
      break;

    if (lane_id < 4)
    {
      v[warp_id][lane_id] = idx_start[warp_id] + lane_id < in_size ? in[idx_start[warp_id] + lane_id] : UINT32_MAX; // take a candidate v from Can(u), if exceeds, then take uintmax(a null vertex).
      table_index[warp_id][lane_id] = 0u;
    }
    if (lane_id == 0)
      nloops[warp_id] = 0u;
    __syncwarp();

    while (nloops[warp_id] < MAX_CUCKOO_LOOP)
    {
      if (*success != 0u) // if succeeded.
        break;

      // result is a mask. if the res_mask is 0, it means all 4 vertices are processed.
      result = __ballot_sync(0xffffffff, v[warp_id][elem_idx] != UINT32_MAX);
      if (result == 0u)
        break;

      if (lane_id < 4 && v[warp_id][lane_id] != UINT32_MAX)
      {
        bucket_index[warp_id][lane_id] =
            (C[table_index[warp_id][lane_id]][0] ^ v[warp_id][lane_id] + C[table_index[warp_id][lane_id]][1]) % num_bucket; // the same as `hashSearch()`, getting the hash_value.
      }
      __syncwarp();

      if (v[warp_id][elem_idx] != UINT32_MAX) // if v is not processed.
      {
        pre_value = keys[table_index[warp_id][elem_idx]][bucket_index[warp_id][elem_idx] * 8 + lane_id % 8];
        // keys[0/1][index]
        // 2 tables. so table_index[wid][elem_id] returns 0 or 1.
        // bucket_index[wid][elem_idx] returns the bid of v.
        // 8 elements in each bucket. so multiply 8 to get the global start offset of the bucket.
        // plus lid%8 to get the global offset in the whole hashed array.

        // pre_value is a key. a key is a v in G.
        // the `value` for the key is the `position` of v in the `hashed array`.

        result = __ballot_sync(mask, pre_value == UINT32_MAX);
        // mask is different for each part of the warp.
        // may cause undefined behaviour.
        // Here, cooperative group should be used.
        // Not all 4 parts participate the computing.
        // may cause a hault.

        while (result != 0)
        {
          leader = __ffs(result) - 1;
          if (lane_id == leader)
          {
            // atomicCAS: if same, then assign; else no change. whatever it does, return old value.
            if (atomicCAS(keys[table_index[warp_id][elem_idx]] + bucket_index[warp_id][elem_idx] * 8 + lane_id % 8, pre_value, v[warp_id][elem_idx]) == pre_value)
              v[warp_id][elem_idx] = UINT32_MAX;
          }
          __syncwarp(mask);
          if (v[warp_id][elem_idx] == UINT32_MAX) // converged. done.
            break;
          pre_value = keys[table_index[warp_id][elem_idx]][bucket_index[warp_id][elem_idx] * 8 + lane_id % 8];
          result = __ballot_sync(mask, pre_value == UINT32_MAX);
        }
        if (v[warp_id][elem_idx] != UINT32_MAX) // not succeeded.
        {
          leader = v[warp_id][elem_idx] % BUCKET_DIM;
          if (lane_id % BUCKET_DIM == leader)
          {
            pre_value = atomicExch(keys[table_index[warp_id][elem_idx]] + bucket_index[warp_id][elem_idx] * 8 + lane_id % 8, v[warp_id][elem_idx]);
            v[warp_id][elem_idx] = pre_value;
            table_index[warp_id][elem_idx] = (table_index[warp_id][elem_idx] + 1) % 2;
          }
          __syncwarp(mask);
        }
      }
      __syncwarp();
      if (lane_id == 0)
        nloops[warp_id]++;
      __syncwarp();
    }
    if (*success != 0u) // if success.
    {
      break;
    }
    if (nloops[warp_id] >= MAX_CUCKOO_LOOP)
    {
      atomicAdd(success, 1u);
      break;
    }
  }
}

__global__ void buildHashValuesCount(
    const offtype *d_offsets_, const vtype *d_neighbors_,
    const uint32_t *flags_second_level, // bitmap
    const uint32_t num_flags,           // size of one row.
    uint32_t *progress,
    uint32_t *hash_keys,
    uint32_t *hash_values,
    const uint32_t num_bucket)
{
  __shared__ uint32_t v[WARP_PER_BLOCK][WARP_SIZE];
  __shared__ uint32_t idx[WARP_PER_BLOCK][WARP_SIZE];
  __shared__ uint32_t s_hash_values[WARP_PER_BLOCK][WARP_SIZE];

  __shared__ uint32_t table_start[WARP_PER_BLOCK];
  __shared__ uint8_t queue_size[WARP_PER_BLOCK];

  uint32_t warp_id = threadIdx.x / WARP_SIZE;
  uint32_t lane_id = threadIdx.x % WARP_SIZE;
  uint32_t off1, off2, off, v_other;

  while (true)
  {
    if (lane_id == 0)
    {
      table_start[warp_id] = atomicAdd(progress, 32u);
    }
    __syncwarp();

    if (table_start[warp_id] >= num_bucket * BUCKET_DIM)
      break;

    if (lane_id == 0)
      queue_size[warp_id] = 0u;
    s_hash_values[warp_id][lane_id] = 0u;
    __syncwarp();

    // write vertices and their index to shared memory
    if (
        table_start[warp_id] + lane_id < num_bucket * BUCKET_DIM &&
        hash_keys[table_start[warp_id] + lane_id] != UINT32_MAX)
    {
      auto group = cooperative_groups::coalesced_threads();
      auto rank = group.thread_rank();
      idx[warp_id][rank] = lane_id;
      v[warp_id][rank] = hash_keys[table_start[warp_id] + lane_id];
      if (rank == 0)
        queue_size[warp_id] = group.size();
    }
    __syncwarp();

    // process each vertex
    for (uint32_t i = 0; i < queue_size[warp_id]; i++)
    {
      off1 = d_offsets_[v[warp_id][i]];
      off2 = d_offsets_[v[warp_id][i] + 1];
      for (off = off1 + lane_id; off < off2; off += WARP_SIZE)
      {
        v_other = d_neighbors_[off];

        if ((flags_second_level[v_other / 32] & (1 << (v_other % 32))) > 0)
        {
          atomicAdd(&s_hash_values[warp_id][idx[warp_id][i]], 1u);
        }
      }
      __syncwarp();
    }
    if (lane_id < queue_size[warp_id])
    {
      hash_values[(table_start[warp_id] + idx[warp_id][lane_id]) * 2] = s_hash_values[warp_id][idx[warp_id][lane_id]];
      // if a vertex does not have any neighbor, remove the vertex from the hash table
      if (s_hash_values[warp_id][idx[warp_id][lane_id]] == 0)
      {
        hash_keys[table_start[warp_id] + idx[warp_id][lane_id]] = UINT32_MAX;
      }
    }
    __syncwarp();
  }
}

__global__ void buildHashValuesWrite(
    const offtype *d_offsets_, const vtype *d_neighbors_,
    const uint32_t *flags_second_level,
    const uint32_t num_flags,
    uint32_t *progress,
    uint32_t *hash_keys,
    uint32_t *hash_values,
    const uint32_t num_bucket,
    uint32_t *neighbors)
{
  __shared__ uint32_t v[WARP_PER_BLOCK][WARP_SIZE];
  __shared__ uint32_t write_pos[WARP_PER_BLOCK][WARP_SIZE];

  __shared__ uint32_t table_start[WARP_PER_BLOCK];
  __shared__ uint8_t queue_size[WARP_PER_BLOCK];

  uint32_t warp_id = threadIdx.x / WARP_SIZE;
  uint32_t lane_id = threadIdx.x % WARP_SIZE;
  uint32_t off1, off2, off, v_other;

  while (true)
  {
    if (lane_id == 0)
    {
      table_start[warp_id] = atomicAdd(progress, 32u);
    }
    __syncwarp();

    if (table_start[warp_id] >= num_bucket * BUCKET_DIM)
      break;

    if (lane_id == 0)
    {
      queue_size[warp_id] = 0u;
    }
    __syncwarp();

    // compute the number of neighbors of each vertex from hash_values
    if (table_start[warp_id] + lane_id < num_bucket * BUCKET_DIM)
    {
      hash_values[(table_start[warp_id] + lane_id) * 2 + 1] -= hash_values[(table_start[warp_id] + lane_id) * 2];
    }
    __syncwarp();

    // write vertices and their index to shared memory
    if (
        table_start[warp_id] + lane_id < num_bucket * BUCKET_DIM &&
        hash_keys[table_start[warp_id] + lane_id] != UINT32_MAX)
    {
      auto group = cooperative_groups::coalesced_threads();
      auto rank = group.thread_rank();
      v[warp_id][rank] = hash_keys[table_start[warp_id] + lane_id];
      write_pos[warp_id][rank] = hash_values[(table_start[warp_id] + lane_id) * 2];
      if (rank == 0)
      {
        queue_size[warp_id] = group.size();
      }
    }
    __syncwarp();

    // process each vertex
    for (uint32_t i = 0; i < queue_size[warp_id]; i++)
    {
      off1 = d_offsets_[v[warp_id][i]];
      off2 = d_offsets_[v[warp_id][i] + 1];
      for (off = off1 + lane_id; off < off2; off += WARP_SIZE)
      {
        v_other = d_neighbors_[off];

        if ((flags_second_level[v_other / 32] & (1 << (v_other % 32))) > 0)
        {
          auto group = cooperative_groups::coalesced_threads();
          auto rank = group.thread_rank();
          neighbors[write_pos[warp_id][i] + rank] = v_other;
          if (rank == 0)
          {
            write_pos[warp_id][i] += group.size();
          }
        }
        __syncwarp(__activemask());
      }
      __syncwarp();
    }
  }
}

__global__ void create_bitmap_from_candidates(
    const vtype *candidates,
    const uint32_t num_candidates,
    const uint32_t max_vertex_id,
    uint32_t *bitmap,
    const uint32_t num_flags)
{
  // Only set bits for candidate vertices.
  // IMPORTANT: Ensure caller clears 'bitmap' prior to invocation to avoid races.
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  for (uint32_t i = tid; i < num_candidates; i += gridDim.x * blockDim.x)
  {
    vtype vertex = candidates[i];
    if (vertex < max_vertex_id)
    {
      uint32_t word_idx = vertex >> 5;
      uint32_t bit_idx = vertex & 31;
      atomicOr(&bitmap[word_idx], 1u << bit_idx);
    }
  }
}
