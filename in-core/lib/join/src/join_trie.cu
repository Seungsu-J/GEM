#include "join_trie.cuh"
#include "join_trie_kernel.cuh"

#include "globals.h"
#include "lattice.h"
#include "memory_manager.h"
#include "unifiedTrie.cuh"

#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/system/cuda/execution_policy.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <set>
#include <stack>
#include <unordered_set>
#include <vector>
#include <numeric>
#include <cstdio>

/**
 * Global Variables:
 * 1. duplication informations
 * 2. data_ptr and parent_ptr
 * 3. flag_ if skip final materialization
 */

// dup info
uint32_t *d_dup_cols; // allocate once for reuse
uint32_t num_dup_cols;
uint32_t *h_dup_cols;

// data_ptr, parent_ptr
vtype **d_data_arrays;
uint64_t **d_parent_arrays;

// skip flag
bool skip_final_materialization = false;

cudaStream_t join_stream;

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

uint32_t ceilPowerOf2(uint32_t x)
{
  if (x <= 1)
    return 1;

  // 1. 为了处理 x 本身就是 2 的 n 次幂的情况
  // (例如 x=16)，我们先减 1。
  // x=16 -> 15 (0...01111) -> 16
  // x=17 -> 16 (0...10000) -> 32
  x--;

  // 2. 将最高位的 1 向右"涂抹"，填满所有低位
  // 假设 x = 100 (0110 0100)
  // x-- -> 99 (0110 0011)
  x |= x >> 1;  // 0111 0011
  x |= x >> 2;  // 0111 1111
  x |= x >> 4;  // 0111 1111
  x |= x >> 8;  // 0111 1111
  x |= x >> 16; // 0111 1111  (这是 127)
  // x |= x >> 32; // 对于 64 位整数，这一步是必要的

  // 3. 最后加 1，得到 128 (1000 0000)
  x++;

  return x;
}

uint32_t findClosestPowerOf2_Bitwise(uint32_t x)
{
  if (x <= 1)
    return 1; // 2^0 = 1

  // 1. 找到 "天花板"
  uint32_t high_pow = ceilPowerOf2(x); // 例如 x=100, high_pow=128

  // 2. 找到 "地板"
  // (注意：如果 high_pow 是 1, low_pow 会是 0，但这被 x<=1 捕获了)
  uint32_t low_pow = high_pow >> 1; // 128 >> 1 = 64
  // 3. 比较距离 (与方法一相同)
  if ((x - low_pow) < (high_pow - x))
    return low_pow;
  else
    return high_pow;
}

// void select_hybrid(
//     Lattice *lat, UnifiedTrieManager *utm, MemoryManager *mem_mgr,
//     HashTable *ht, cpuGraph *hq,
//     UnifiedTrie *frag_trie, UnifiedTrie *probe_trie, UnifiedTrie *target_trie,
//     uint32_t col_key, uint32_t col_value)
// {
// }

void host_select(
    cpuGraph *hq,
    UnifiedTrie *frag_trie, UnifiedTrie *probe_trie, UnifiedTrie *target_trie,
    uint32_t col_key, uint32_t col_value)
{
  auto lat = &Lattice::getInstance();
  auto utm = &UnifiedTrieManager::getInstance();
  auto mem_mgr = &MemoryManager::getInstance();
  auto ht = &HashTable::getInstance();

  // already made sure col_key < col_value.
  bool edge_forward = frag_trie->mapped_query_us_[col_key] == probe_trie->edge_vertices[0];
#ifndef NDEBUG
  std::cout << "See query us: " << std::endl;
  std::cout << "  probe trie us: " << probe_trie->edge_vertices[0] << ", " << probe_trie->edge_vertices[1] << std::endl;
  for (int i = 0; i < frag_trie->num_cols; ++i)
  {
    std::cout << "  frag trie us[" << i << "]: " << frag_trie->mapped_query_us_[i] << std::endl;
  }
  std::cout << "Edge direction: " << (edge_forward ? "forward" : "backward") << std::endl;
  std::cout << "col_key: " << col_key << ", col_value: " << col_value << std::endl;
#endif

  etype edge_idx = probe_trie->et._Find_first();
  etype directed_edge_idx = edge_idx * 2 + (edge_forward ? 0 : 1);

#ifndef NDEBUG
  std::cout << "directed_edge_idx: " << directed_edge_idx << std::endl;
#endif

  vtype key_u = probe_trie->edge_vertices[0];
  vtype value_u = probe_trie->edge_vertices[1];
  if (!edge_forward)
    std::swap(key_u, value_u);

  // Get device pointers for fragment data
  vtype *d_key_array = utm->getTrie(frag_trie->column_ids[col_key])->data->d_data;
  vtype *d_value_array = utm->getTrie(frag_trie->column_ids[col_value])->data->d_data;

#ifndef NDEBUG
  std::cout << "key_u = " << key_u << ", value_u = " << value_u << std::endl;
  std::cout << "d_key_array: " << frag_trie->column_ids[col_key] << ", d_value_array: " << frag_trie->column_ids[col_value] << std::endl;
  std::cout << "size_d_key_array: " << utm->getTrie(frag_trie->column_ids[col_key])->data->num_rows << std::endl;
  std::cout << "size_d_value_array: " << utm->getTrie(frag_trie->column_ids[col_value])->data->num_rows << std::endl;
  std::cout << "addr_d_key_array: " << d_key_array << ", addr_d_value_array: " << d_value_array << std::endl;
#endif

  // Allocate target mask
  // uint32_t num_compressed_blocks = (frag_trie->data->num_rows + 31) >> 5;
  // target_trie->allocateData(num_rows, true); // is_mask = true
  // target_trie->data->init(frag_trie->data->num_rows, true);
  // target_trie->data->allocate();

#ifndef NDEBUG
  // uint32_t num_compressed_blocks = (frag_trie->data->num_rows + 31) >> 5;
  std::cout << "num_compressed_blocks: " << ((frag_trie->data->num_rows + 31) >> 5) << std::endl;
  std::cout << "should be = " << (frag_trie->data->num_rows + 31) / 32 << std::endl;
#endif

  // Get edge trie offsets and data
  offtype *d_edge_offsets = ht->d_offsets_ + ht->edge_offset_starts_[directed_edge_idx];

#ifndef NDEBUG
  std::cout << "Edge trie offsets at: " << ht->edge_offset_starts_[directed_edge_idx] << std::endl;
#endif

  vtype *d_edge_values = nullptr;
  vtype *d_edge_keys = utm->getTrie(lat->get_vertex_node_id(key_u))->data->d_data;
  if (edge_forward)
    d_edge_values = probe_trie->data->d_data;
  else
    d_edge_values = probe_trie->r_data->d_data;

  // Check if fragment has a mask parent (reference chain)
  if (frag_trie->mask_parent_id == UINT32_MAX)
  {
    // std::cout << "Join on expansion chain." << std::endl;
    utm->linkMask(target_trie->id, frag_trie->id);
    uint32_t *d_bool_vec = target_trie->data->d_data;

    uint64_t num_compressed_blocks = target_trie->data->num_compressed_blocks;
    cuchk(cudaMemsetAsync(d_bool_vec, 0, sizeof(uint32_t) * num_compressed_blocks, join_stream));
    auto d_parent_pointers = utm->get_parent_arrays(frag_trie->id);
    // if (edge_forward == false) // reversed edge parent pointer
    // {
    // d_parent_pointers[1] = utm->getTrie(probe_trie->id)->data->d_parents;
    // }
    cuchk(cudaMemcpyAsync(d_parent_arrays, d_parent_pointers.data(), sizeof(uint64_t *) * d_parent_pointers.size(), cudaMemcpyHostToDevice, join_stream));
    uint64_t num_rows = frag_trie->num_results;

#ifndef NDEBUG
    std::cout << "preparing select, num_rows: " << num_rows << std::endl;
#endif

    select_trie_kernel<<<GRID_DIM, BLOCK_DIM, 0, join_stream>>>(
        // frag
        d_key_array, d_value_array,
        d_parent_arrays, frag_trie->num_cols, col_key, col_value,
        num_rows,
        // edge
        d_edge_keys, d_edge_values, d_edge_offsets,
        // hash
        ht->h_num_buckets_[directed_edge_idx], ht->h_hash_table_offs_[directed_edge_idx], directed_edge_idx,
        // result
        d_bool_vec);
    cuchk(cudaStreamSynchronize(join_stream));

#ifndef NDEBUG
    uint64_t *d_debug_res;
    cuchk(cudaMallocAsync((void **)&d_debug_res, sizeof(uint64_t), join_stream));
    cuchk(cudaMemsetAsync(d_debug_res, 0, sizeof(uint64_t), join_stream));
    std::cout << "See parameters: " << std::endl;
    std::cout << "d_key_array: " << d_key_array << std::endl;
    std::cout << "d_value_array: " << d_value_array << std::endl;
    std::cout << "d_parent_arrays: " << d_parent_arrays << std::endl;
    std::cout << "frag_trie->num_cols: " << frag_trie->num_cols << std::endl;
    std::cout << "col_key: " << col_key << ", col_value: " << col_value << std::endl;
    std::cout << "num_rows: " << num_rows << std::endl;
    std::cout << "d_edge_keys: " << d_edge_keys << std::endl;
    std::cout << "d_edge_values: " << d_edge_values << std::endl;
    std::cout << "d_edge_offsets: " << d_edge_offsets << std::endl;
    std::cout << "num_buckets: " << ht->h_num_buckets_[directed_edge_idx] << std::endl;
    std::cout << "hash_table_offs: " << ht->h_hash_table_offs_[directed_edge_idx] << std::endl;
    std::cout << "directed_edge_idx: " << directed_edge_idx << std::endl;
    std::cout << "d_bool_vec: " << d_bool_vec << std::endl;

    uint32_t *d_res_debug;
    cuchk(cudaMallocAsync((void **)&d_res_debug, sizeof(uint32_t) * num_rows * 2, join_stream));
    cuchk(cudaMemsetAsync(d_res_debug, 0xFF, sizeof(uint32_t) * num_rows * 2, join_stream));
    extract_edge_kernel<<<GRID_DIM, BLOCK_DIM, 0, join_stream>>>(
        d_key_array, d_value_array,
        num_rows, frag_trie->num_cols,
        col_key, col_value,
        d_parent_arrays,
        d_res_debug);
    uint32_t *h_res_debug = new uint32_t[num_rows * 2];
    cuchk(cudaMemcpyAsync(h_res_debug, d_res_debug, sizeof(uint32_t) * num_rows * 2, cudaMemcpyDeviceToHost, join_stream));
    std::ofstream fout("runtime_data/extract_edge_debug.txt");
    for (uint32_t i = 0; i < num_rows; ++i)
    {
      fout << h_res_debug[i * 2] << " " << h_res_debug[i * 2 + 1] << std::endl;
    }
    fout.close();
    delete[] h_res_debug;
    // cuchk(cudaFree(d_res_debug));
    cuchk(cudaMemsetAsync(d_res_debug, 0, sizeof(uint32_t) * num_rows * 2, join_stream));

    select_trie_kernel_debug<<<GRID_DIM, BLOCK_DIM, 0, join_stream>>>(
        // frag
        d_key_array, d_value_array,
        d_parent_arrays, frag_trie->num_cols, col_key, col_value,
        num_rows,
        // edge
        d_edge_keys, d_edge_values, d_edge_offsets,
        // hash
        ht->h_num_buckets_[directed_edge_idx], ht->h_hash_table_offs_[directed_edge_idx], directed_edge_idx,
        // result
        d_bool_vec, d_debug_res, d_res_debug);

    h_res_debug = new uint32_t[num_rows * 2];
    cuchk(cudaMemcpyAsync(h_res_debug, d_res_debug, sizeof(uint32_t) * num_rows * 2, cudaMemcpyDeviceToHost, join_stream));
    fout.open("runtime_data/extract_edge_in_kernel.txt");
    for (uint32_t i = 0; i < num_rows; ++i)
    {
      fout << h_res_debug[i * 2] << " " << h_res_debug[i * 2 + 1] << std::endl;
    }
    fout.close();
    delete[] h_res_debug;

    uint32_t *h_debug_res = new uint32_t;
    cuchk(cudaMemcpyAsync(h_debug_res, d_debug_res, sizeof(uint32_t), cudaMemcpyDeviceToHost, join_stream));
    std::cout << "Debug res: " << *h_debug_res << std::endl;
    delete h_debug_res;

    uint32_t *h_bool_vec = new uint32_t[num_compressed_blocks];
    cuchk(cudaMemcpyAsync(h_bool_vec, d_bool_vec, sizeof(uint32_t) * num_compressed_blocks, cudaMemcpyDeviceToHost, join_stream));
    uint32_t host_result = 0;
    for (int i = 0; i < num_compressed_blocks; ++i)
    {
      host_result += __builtin_popcount(h_bool_vec[i]);
    }
    std::cout << "Debug res (host count): " << host_result << std::endl;
    delete[] h_bool_vec;
    cuchk(cudaFreeAsync(d_debug_res, join_stream));
#endif
  }
  else
  {
    // std::cout << "Join on mask chain." << std::endl;
    utm->linkMask(target_trie->id, frag_trie->id);
    uint32_t *d_bool_vec = target_trie->data->d_data;

    uint64_t num_compressed_blocks = target_trie->data->num_compressed_blocks;
    cuchk(cudaMemsetAsync(d_bool_vec, 0, sizeof(uint32_t) * num_compressed_blocks, join_stream));

    // Handle reference chain - find the root data trie (climb through all mask layers)
    UnifiedTrie *root_data = frag_trie;
    do
    {
      root_data = utm->getTrie(root_data->mask_parent_id);
    } while (root_data->mask_parent_id != UINT32_MAX);

    // Get original data arrays from root_data
    vtype *d_original_key_array = utm->getTrie(root_data->column_ids[col_key])->data->d_data;
    vtype *d_original_value_array = utm->getTrie(root_data->column_ids[col_value])->data->d_data;
    auto d_parent_pointers = utm->get_parent_arrays(root_data->id);
    cuchk(cudaMemcpyAsync(d_parent_arrays, d_parent_pointers.data(), sizeof(uint64_t *) * d_parent_pointers.size(), cudaMemcpyHostToDevice, join_stream));
    // uint32_t **d_original_parent_pointers = utm->parent_addr_list[root_data->id].data();

    // Get selection mask from fragment
    uint32_t *d_select_bool_vec = frag_trie->data->d_data;
    uint64_t num_rows = root_data->num_results;

    select_trie_refer_kernel<<<GRID_DIM, BLOCK_DIM, 0, join_stream>>>(
        // original data to refer to
        d_original_key_array, d_original_value_array,
        d_parent_arrays, root_data->num_cols, col_key, col_value,
        // selection mask
        d_select_bool_vec, num_rows,
        // trie_edge
        d_edge_keys, d_edge_values, d_edge_offsets,
        // hash
        ht->h_num_buckets_[directed_edge_idx], ht->h_hash_table_offs_[directed_edge_idx], directed_edge_idx,
        // result
        d_bool_vec);
    cuchk(cudaStreamSynchronize(join_stream));
  }
  uint32_t *d_bool_vec = target_trie->data->d_data;
  uint64_t num_compressed_blocks = target_trie->data->num_compressed_blocks;

#ifndef NDEBUG
  uint32_t *debug_mask_bits;
  cuchk(cudaMallocAsync((void **)&debug_mask_bits, sizeof(uint32_t), join_stream));
  cuchk(cudaMemsetAsync(debug_mask_bits, 0, sizeof(uint32_t), join_stream));
  // uint32_t *d_bool_vec = target_trie->data->d_data;
  naiive_count<<<GRID_DIM, BLOCK_DIM, 0, join_stream>>>(target_trie->data->d_data, num_compressed_blocks, debug_mask_bits);
  uint32_t h_debug_mask_bits;
  cuchk(cudaMemcpyAsync(&h_debug_mask_bits, debug_mask_bits, sizeof(uint32_t), cudaMemcpyDeviceToHost, join_stream));
  std::cout << "Debug: number of selected results (naive count): " << h_debug_mask_bits << std::endl;
  cuchk(cudaFreeAsync(debug_mask_bits, join_stream));
#endif
  // Count the number of set bits using CUB

  uint64_t h_num_selected = 0;
  using PopcIterator = cub::TransformInputIterator<uint32_t, PopcToCount, uint32_t *>;
  PopcIterator popc_iter(d_bool_vec, PopcToCount{});
  // auto popc_iter = thrust::make_transform_iterator(d_bool_vec, PopcToCount{});

  size_t temp_storage_bytes = 0;
  void *d_temp_storage = nullptr;
  uint64_t *d_result;
  cuchk(cudaMallocAsync((void **)&d_result, sizeof(uint64_t), join_stream));
  cuchk(cudaMemsetAsync(d_result, 0, sizeof(uint64_t), join_stream));

  cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, popc_iter,
                         d_result, num_compressed_blocks, join_stream);
  cuchk(cudaMallocAsync(&d_temp_storage, temp_storage_bytes, join_stream));
  cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, popc_iter,
                         d_result, num_compressed_blocks, join_stream);

  // Copy result back
  cuchk(cudaMemcpyAsync(&h_num_selected, d_result, sizeof(uint64_t),
                        cudaMemcpyDeviceToHost, join_stream));
  cuchk(cudaStreamSynchronize(join_stream));

  // Update target properties
  target_trie->updateNumResults(h_num_selected);
  // target_trie->data->num_rows = h_num_selected;

  // Cleanup
  cuchk(cudaFreeAsync(d_temp_storage, join_stream));
  cuchk(cudaFreeAsync(d_result, join_stream));
  if (skip_final_materialization)
    cuchk(cudaFreeAsync(target_trie->data->d_data, join_stream)); // free mask data if skip final materialization
  cuchk(cudaStreamSynchronize(join_stream));

#ifndef NDEBUG
  std::cout << "select: " << h_num_selected << " results selected" << std::endl;
#endif
}

void direct_join(
    cpuGraph *hq,
    UnifiedTrie *frag_trie, UnifiedTrie *probe_trie, UnifiedTrie *target_trie,
    uint32_t join_col, bool is_forward)
{
  Lattice *lat = &Lattice::getInstance();
  UnifiedTrieManager *utm = &UnifiedTrieManager::getInstance();
  MemoryManager *mem_mgr = &MemoryManager::getInstance();
  HashTable *ht = &HashTable::getInstance();

  DataBlock *edge_data_block; // for edge value layer
  vtype *d_edge_keys;         // edge key layer
  if (is_forward)
  {
    d_edge_keys = utm->getTrie(lat->get_vertex_node_id(probe_trie->edge_vertices[0]))->data->d_data;
    edge_data_block = probe_trie->data;
  }
  else
  {
    d_edge_keys = utm->getTrie(lat->get_vertex_node_id(probe_trie->edge_vertices[1]))->data->d_data;
    edge_data_block = probe_trie->r_data;
  }

  vtype new_u = is_forward ? probe_trie->edge_vertices[1] : probe_trie->edge_vertices[0];
  utm->linkExpansion(target_trie->id, frag_trie->id, 0, new_u);

  // Get fragment data pointers
  uint64_t num_rows_frag = frag_trie->num_results;
#ifndef NDEBUG
  std::cout << "Fragment trie num rows: " << num_rows_frag << std::endl;
#endif

  // Get edge trie information
  etype edge_idx = probe_trie->et._Find_first();
  etype directed_edge_idx = edge_idx * 2 + (is_forward ? 0 : 1);
  offtype *d_edge_offsets = ht->d_offsets_ + ht->edge_offset_starts_[directed_edge_idx];

  // Allocate counting arrays
  uint64_t h_num_res = 0;

  auto data_arrays = utm->get_data_arrays(frag_trie->id);
  auto parent_arrays = utm->get_parent_arrays(frag_trie->id);

#ifndef NDEBUG
  std::cout << data_arrays.size() << " data arrays for frag_trie " << frag_trie->id << std::endl;
  std::cout << data_arrays[0] << " first data array ptr " << std::endl;
#endif

  cuchk(cudaMemcpyAsync(d_data_arrays, data_arrays.data(), sizeof(vtype *) * data_arrays.size(),
                        cudaMemcpyHostToDevice, join_stream));
  cuchk(cudaMemcpyAsync(d_parent_arrays, parent_arrays.data(), sizeof(uint64_t *) * parent_arrays.size(),
                        cudaMemcpyHostToDevice, join_stream));

  int dup_check_type = 0;
  num_dup_cols = 0;

  vltype new_u_label = hq->vLabels_[new_u];
  for (int i = 0; i < frag_trie->num_cols; ++i)
    if (new_u_label == hq->vLabels_[frag_trie->mapped_query_us_[i]])
      h_dup_cols[num_dup_cols++] = i;
  if (num_dup_cols == frag_trie->num_cols)
    dup_check_type = 2;
  else if (num_dup_cols > 0)
    dup_check_type = 1;
  if (num_dup_cols > 0)
    cuchk(cudaMemcpyAsync(d_dup_cols, h_dup_cols, sizeof(uint32_t) * num_dup_cols, cudaMemcpyHostToDevice, join_stream));

  uint32_t *d_num_res_for_each;
  cuchk(cudaMallocAsync((void **)&d_num_res_for_each, sizeof(uint32_t) * (num_rows_frag + 1), join_stream));
  cuchk(cudaMemsetAsync(d_num_res_for_each, 0, sizeof(uint32_t) * (num_rows_frag + 1), join_stream));

  uint32_t *d_st_array;
  cuchk(cudaMallocAsync((void **)&d_st_array, sizeof(uint32_t) * num_rows_frag, join_stream));
  cuchk(cudaMemsetAsync(d_st_array, 0xFFFFFFFF, sizeof(uint32_t) * num_rows_frag, join_stream));
  if (dup_check_type == 0)
  {
    // std::cout << "Join without duplication check." << std::endl;
    uint32_t task_per_warp;
    if (num_rows_frag >= UINT32_MAX)
      task_per_warp = 32;
    else
    {
      uint32_t avg = num_rows_frag / (GRID_DIM * BLOCK_DIM / 32);
      task_per_warp = findClosestPowerOf2_Bitwise(avg);
      task_per_warp = std::min(task_per_warp, 32u);
    }

    assert(__builtin_popcount(task_per_warp) == 1);

#ifndef NDEBUG
    std::cout << "task_per_warp: " << task_per_warp << std::endl;
    std::cout << "num_rows_frag: " << num_rows_frag << std::endl;
#endif

    join_count_nodup_dispatcher(
        // trie_inter
        d_data_arrays, d_parent_arrays,
        num_rows_frag, frag_trie->num_cols, join_col,
        // trie_edge
        d_edge_keys, edge_data_block->d_data, d_edge_offsets,
        // hash
        ht->h_num_buckets_[directed_edge_idx],
        ht->h_hash_table_offs_[directed_edge_idx],
        directed_edge_idx,
        // mask
        d_num_res_for_each,
        d_st_array,
        task_per_warp);
    cuchk(cudaStreamSynchronize(join_stream));

#ifndef NDEBUG
    uint32_t *h_num_res_array = new uint32_t[num_rows_frag];
    cuchk(cudaMemcpyAsync(h_num_res_array, d_num_res_for_each, sizeof(uint32_t) * num_rows_frag, cudaMemcpyDeviceToHost, join_stream));
    cuchk(cudaStreamSynchronize(join_stream));
    uint32_t total_count = 0;
    for (uint32_t i = 0; i < num_rows_frag; ++i)
    {
      total_count += h_num_res_array[i];
    }
    std::cout << "Debug: total count from d_num_res_for_each: " << total_count << std::endl;
    delete[] h_num_res_array;
#endif

    uint64_t *d_row_offsets = nullptr;
    cuchk(cudaMallocAsync((void **)&d_row_offsets, sizeof(uint64_t) * (num_rows_frag + 1), join_stream));
    // exclusive scan on d_num_res_for_each to write position

    size_t temp_storage_bytes = 0;
    void *d_temp_storage = nullptr;

    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
                                  d_num_res_for_each, d_row_offsets,
                                  num_rows_frag + 1, join_stream);
    cuchk(cudaMallocAsync(&d_temp_storage, temp_storage_bytes, join_stream));
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
                                  d_num_res_for_each, d_row_offsets,
                                  num_rows_frag + 1, join_stream);
    // Copy total count from d_row_offsets[num_rows] to host
    cuchk(cudaMemcpyAsync(&h_num_res, d_row_offsets + num_rows_frag,
                          sizeof(uint64_t),
                          cudaMemcpyDeviceToHost, join_stream));

    // cuchk(cudaMallocAsync((void **)&target_trie->data->d_data, sizeof(vtype) * h_num_res, join_stream));
    if (target_trie->data)
      delete target_trie->data;
    target_trie->data = new DataBlock();
    cuchk(cudaStreamSynchronize(join_stream));
    target_trie->data->init(h_num_res, false);

    if (skip_final_materialization == false)
    {
      target_trie->data->allocate();

      // TODO: alternative: cub::DeviceMemcpy::Batched().
      // if time allows, implement cub::DeviceMemcpy::Batched() and compare the time consumed.

      join_write_nodup_dispatcher(
          num_rows_frag, edge_data_block->d_data,
          d_num_res_for_each, d_st_array, target_trie->data->d_data,
          d_row_offsets, task_per_warp, target_trie->data->d_parents);
    }

    if (d_row_offsets)
      cuchk(cudaFreeAsync(d_row_offsets, join_stream));
    cuchk(cudaFreeAsync(d_temp_storage, join_stream));
    cuchk(cudaStreamSynchronize(join_stream));
  }
  else
  {
    // std::cout << "Join with duplication check" << std::endl;
    // mask for avoiding 2nd binary search on edge values
    uint32_t *d_mask;
    uint32_t mask_length = ht->max_values_per_edge_[directed_edge_idx];
    // Convert to number of 32-bit words (round up) and align to 32 words (128 bytes) for coalescing
    mask_length = ((mask_length + 31) / 32 + 31) / 32 * 32;
    cuchk(cudaMallocAsync((void **)&d_mask, sizeof(uint32_t) * num_rows_frag * mask_length, join_stream));
    cuchk(cudaMemsetAsync(d_mask, 0, sizeof(uint32_t) * num_rows_frag * mask_length, join_stream));
    // Count phase - placeholder kernel

    join_count_kernel_dup<<<GRID_DIM, BLOCK_DIM, 0, join_stream>>>(
        d_data_arrays, d_parent_arrays, num_rows_frag, frag_trie->num_cols, join_col,
        d_edge_keys, edge_data_block->d_data, d_edge_offsets,
        ht->h_num_buckets_[directed_edge_idx],
        ht->h_hash_table_offs_[directed_edge_idx],
        directed_edge_idx,
        d_mask, mask_length, d_st_array,
        dup_check_type, num_dup_cols, d_dup_cols);

    uint64_t *d_row_offsets = nullptr;
    cuchk(cudaMallocAsync((void **)&d_row_offsets, sizeof(uint64_t) * (num_rows_frag + 1), join_stream));

    // Exclusive prefix sum to compute write offsets for each row
    // Input: d_num_res_for_each[i] = count for row i
    // Output: d_row_offsets[i] = exclusive scan (starting offset for row i)
    //         d_row_offsets[num_rows] = total count
    size_t temp_storage_bytes = 0;
    void *d_temp_storage = nullptr;

    // Create segment offsets: [0, mask_length, 2*mask_length, ..., num_rows*mask_length]
    uint32_t *d_segment_offsets = nullptr;
    cuchk(cudaMallocAsync((void **)&d_segment_offsets, sizeof(uint32_t) * (num_rows_frag + 1), join_stream));

    thrust::device_ptr<uint32_t> offsets_ptr(d_segment_offsets);
    // Use stream-bound execution policy to utilize the memory pool associated with join_stream
    thrust::sequence(thrust::cuda::par.on(join_stream),
                     offsets_ptr, offsets_ptr + num_rows_frag + 1, 0u, (uint32_t)mask_length);

    // Create a transform iterator that applies __popc to each mask tile
    auto popcount_iterator = thrust::make_transform_iterator(d_mask, PopcToCount{});

    // Use CUB DeviceSegmentedReduce to sum popcounts per row
    cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes,
                                    popcount_iterator, d_num_res_for_each,
                                    num_rows_frag, d_segment_offsets, d_segment_offsets + 1, join_stream);
    cuchk(cudaMallocAsync(&d_temp_storage, temp_storage_bytes, join_stream));
    cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes,
                                    popcount_iterator, d_num_res_for_each,
                                    num_rows_frag, d_segment_offsets, d_segment_offsets + 1, join_stream);
    // Free temporary arrays
    cuchk(cudaFreeAsync(d_segment_offsets, join_stream));
    cuchk(cudaFreeAsync(d_temp_storage, join_stream));
    d_temp_storage = nullptr;
    temp_storage_bytes = 0;

    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
                                  d_num_res_for_each, d_row_offsets,
                                  num_rows_frag + 1, join_stream);
    cuchk(cudaMallocAsync(&d_temp_storage, temp_storage_bytes, join_stream));
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
                                  d_num_res_for_each, d_row_offsets,
                                  num_rows_frag + 1, join_stream);
    // Copy total count from d_row_offsets[num_rows] to host
    cuchk(cudaMemcpyAsync(&h_num_res, d_row_offsets + num_rows_frag,
                          sizeof(uint64_t),
                          cudaMemcpyDeviceToHost, join_stream));

    // Free temporary storage
    cuchk(cudaFreeAsync(d_temp_storage, join_stream));

    if (target_trie->data)
      delete target_trie->data;
    target_trie->data = new DataBlock();
    cuchk(cudaStreamSynchronize(join_stream));
    target_trie->data->init(h_num_res, false);
    if (skip_final_materialization == false)
    {
      target_trie->data->allocate();
      // cuchk(cudaMemset(target_trie->data->d_data, 0xFF, sizeof(vtype) * h_num_res));
      // cuchk(cudaMemset(target_trie->data->d_parents, 0xFF, sizeof(uint32_t) * h_num_res));
      // cuchk(cudaMalloc((void **)&target_trie->data->d_data, sizeof(vtype) * h_num_res));

      join_write_kernel_dup<<<GRID_DIM, BLOCK_DIM, 0, join_stream>>>(
          num_rows_frag, edge_data_block->d_data,
          d_mask, mask_length, d_num_res_for_each, d_row_offsets, d_st_array,
          target_trie->data->d_data, target_trie->data->d_parents);
    }
#ifndef NDEBUG
    uint64_t *d_num_processed;
    cuchk(cudaMallocAsync((void **)&d_num_processed, sizeof(uint64_t), join_stream));
    cuchk(cudaMemsetAsync(d_num_processed, 0, sizeof(uint64_t), join_stream));
    join_write_kernel_dup_debug<<<GRID_DIM, BLOCK_DIM, 0, join_stream>>>(
        num_rows_frag, edge_data_block->d_data,
        d_mask, mask_length, d_num_res_for_each, d_row_offsets, d_st_array,
        target_trie->data->d_data, target_trie->data->d_parents, d_num_processed);
    uint32_t h_num_processed = 0;
    cuchk(cudaMemcpyAsync(&h_num_processed, d_num_processed, sizeof(uint32_t), cudaMemcpyDeviceToHost, join_stream));
    cuchk(cudaStreamSynchronize(join_stream));
    std::cout << "Num processed: " << h_num_processed << std::endl;
    cuchk(cudaFreeAsync(d_num_processed, join_stream));

    // check if parent array is set correctly.
    std::cout << "Checking parent array correctness..." << std::endl;
    uint32_t *d_res;
    cuchk(cudaMallocAsync((void **)&d_res, sizeof(uint32_t) * frag_trie->num_results, join_stream));
    uint32_t *d_no_valid;
    cuchk(cudaMallocAsync((void **)&d_no_valid, sizeof(uint32_t), join_stream));
    cuchk(cudaMemsetAsync(d_no_valid, 0, sizeof(uint32_t), join_stream));
    retrive_valid_parents_kernel<<<GRID_DIM, BLOCK_DIM, 0, join_stream>>>(
        target_trie->data->d_parents, h_num_res, frag_trie->data->d_data, d_res);
    check_same_kernel<<<GRID_DIM, BLOCK_DIM, 0, join_stream>>>(
        d_res, frag_trie->data->d_data, frag_trie->num_results, d_no_valid);
    uint32_t h_no_valid;
    cuchk(cudaMemcpyAsync(&h_no_valid, d_no_valid, sizeof(uint32_t), cudaMemcpyDeviceToHost, join_stream));
    cuchk(cudaStreamSynchronize(join_stream));
    std::cout << h_no_valid << " parents have no children in the result." << std::endl;
    cuchk(cudaFreeAsync(d_res, join_stream));
    cuchk(cudaFreeAsync(d_no_valid, join_stream));
#endif
    if (d_row_offsets)
      cuchk(cudaFreeAsync(d_row_offsets, join_stream));
    cuchk(cudaFreeAsync(d_mask, join_stream));
    cuchk(cudaStreamSynchronize(join_stream));
  }
  // Update target properties
  cuchk(cudaStreamSynchronize(join_stream));
  target_trie->updateNumResults(h_num_res);
  target_trie->data->num_rows = h_num_res;

#ifndef NDEBUG
  std::cout << "direct_join: " << h_num_res << " results produced" << std::endl;

  // check
  vtype *h_data = new vtype[h_num_res];
  cuchk(cudaMemcpyAsync(h_data, target_trie->data->d_data, sizeof(vtype) * h_num_res, cudaMemcpyDeviceToHost, join_stream));
  cuchk(cudaStreamSynchronize(join_stream));
  uint32_t count = 0;
  for (int i = 0; i < h_num_res; ++i)
  {
    if (h_data[i] != UINT32_MAX)
      ++count;
  }
  std::cout << "Valid results count: " << count << std::endl;
#endif

  // Cleanup
  cuchk(cudaFreeAsync(d_st_array, join_stream));
  cuchk(cudaFreeAsync(d_num_res_for_each, join_stream));
  cuchk(cudaStreamSynchronize(join_stream));
  // cuchk(cudaFree(dev_space));
  // cuchk(cudaFree(d_data_arrays));
  // cuchk(cudaFree(d_parent_arrays));
}

void refer_join(
    cpuGraph *hq,
    UnifiedTrie *frag_trie, UnifiedTrie *probe_trie, UnifiedTrie *target_trie,
    uint32_t col_key, bool is_forward)
{
  auto lat = &Lattice::getInstance();
  auto utm = &UnifiedTrieManager::getInstance();
  auto mem_mgr = &MemoryManager::getInstance();
  auto ht = &HashTable::getInstance();

  DataBlock *edge_data_block; // for edge value layer
  vtype *d_edge_keys;
  if (is_forward)
  {
    d_edge_keys = utm->getTrie(lat->get_vertex_node_id(probe_trie->edge_vertices[0]))->data->d_data;
    edge_data_block = probe_trie->data;
  }
  else
  {
    d_edge_keys = utm->getTrie(lat->get_vertex_node_id(probe_trie->edge_vertices[1]))->data->d_data;
    edge_data_block = probe_trie->r_data;
  }
  vtype *d_edge_values = edge_data_block->d_data;
  // Determine join direction and new vertex

  vtype new_u = is_forward ? probe_trie->edge_vertices[1] : probe_trie->edge_vertices[0];
  etype edge_idx = probe_trie->et._Find_first();
  etype directed_edge_idx = edge_idx * 2 + (is_forward ? 0 : 1);
  utm->linkExpansion(target_trie->id, frag_trie->id, 0, new_u);

  // Find the root data trie in the reference chain
  UnifiedTrie *root_data = frag_trie;
  do
  {
    root_data = utm->getTrie(root_data->mask_parent_id);
  } while (root_data->mask_parent_id != UINT32_MAX);

  // Get selection mask from fragment's first layer
  uint32_t *d_select_mask = frag_trie->data->d_data; // Mask data
  uint64_t num_rows = root_data->num_results;

  // Get edge trie information
  offtype *d_edge_offsets = ht->d_offsets_ + ht->edge_offset_starts_[directed_edge_idx];

  // Allocate counting arrays
  uint64_t *h_num_res = new uint64_t;

  auto d_data_pointers = utm->get_data_arrays(root_data->id);
  auto d_parent_pointers = utm->get_parent_arrays(root_data->id);
  cuchk(cudaMemcpyAsync(d_data_arrays, d_data_pointers.data(), sizeof(vtype *) * d_data_pointers.size(),
                        cudaMemcpyHostToDevice, join_stream));
  cuchk(cudaMemcpyAsync(d_parent_arrays, d_parent_pointers.data(), sizeof(uint64_t *) * d_parent_pointers.size(),
                        cudaMemcpyHostToDevice, join_stream));

  uint32_t *d_num_res_for_each;
  cuchk(cudaMallocAsync((void **)&d_num_res_for_each, sizeof(uint32_t) * num_rows, join_stream));
  cuchk(cudaMemsetAsync(d_num_res_for_each, 0, sizeof(uint32_t) * num_rows, join_stream));

  // Duplicate check setup
  int dup_check_type = 0;
  num_dup_cols = 0;

  vltype new_u_label = hq->vLabels_[new_u];
  for (int i = 0; i < root_data->num_cols; ++i)
    if (new_u_label == hq->vLabels_[root_data->mapped_query_us_[i]])
      h_dup_cols[num_dup_cols++] = i;
  if (num_dup_cols == root_data->num_cols)
    dup_check_type = 2;
  else if (num_dup_cols > 0)
    dup_check_type = 1;
  if (dup_check_type > 0)
    cuchk(cudaMemcpyAsync(d_dup_cols, h_dup_cols, sizeof(uint32_t) * num_dup_cols, cudaMemcpyHostToDevice, join_stream));

  // Count phase
  // uint32_t *d_num_res_total;
  // cuchk(cudaMalloc((void **)&d_num_res_total, sizeof(uint32_t)));
  // cuchk(cudaMemset(d_num_res_total, 0, sizeof(uint32_t)));

  uint32_t *d_st_array;
  cuchk(cudaMallocAsync((void **)&d_st_array, sizeof(uint32_t) * num_rows, join_stream));
  cuchk(cudaMemsetAsync(d_st_array, 0xFFFFFFFF, sizeof(uint32_t) * num_rows, join_stream));

  if (dup_check_type == 0)
  {
    uint32_t task_per_warp;
    if (num_rows >= UINT32_MAX)
      task_per_warp = 32;
    else
    {
      uint32_t avg = num_rows / (GRID_DIM * BLOCK_DIM / 32);
      task_per_warp = findClosestPowerOf2_Bitwise(avg);
      task_per_warp = std::min(task_per_warp, 32u);
    }

    assert(__builtin_popcount(task_per_warp) == 1);

    join_count_refer_no_dup_dispatcher(
        // trie_inter
        d_data_arrays, d_parent_arrays,
        num_rows, root_data->num_cols, col_key,
        d_select_mask,
        // trie_edge
        d_edge_keys, d_edge_values, d_edge_offsets,
        // hash
        ht->h_num_buckets_[directed_edge_idx],
        ht->h_hash_table_offs_[directed_edge_idx],
        directed_edge_idx,
        // result
        d_num_res_for_each,
        d_st_array,
        task_per_warp);

    // Manually allocate memory to use the memory pool
    uint64_t *d_indices_raw;
    cuchk(cudaMallocAsync((void **)&d_indices_raw, sizeof(uint64_t) * num_rows, join_stream));
    thrust::device_ptr<uint64_t> d_indices(d_indices_raw);
    thrust::sequence(thrust::cuda::par.on(join_stream),
                     d_indices, d_indices + num_rows);

    uint64_t *d_selected_row_ids;
    uint64_t *d_num_selected;
    cuchk(cudaMallocAsync((void **)&d_selected_row_ids, sizeof(uint64_t) * num_rows, join_stream));
    cuchk(cudaMallocAsync((void **)&d_num_selected, sizeof(uint64_t), join_stream));

    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    CountPositive pred{d_num_res_for_each};

    cub::DeviceSelect::If(
        d_temp_storage, temp_storage_bytes,
        d_indices_raw, d_selected_row_ids,
        d_num_selected,
        num_rows,
        pred, join_stream);
    cuchk(cudaMallocAsync(&d_temp_storage, temp_storage_bytes, join_stream));
    cub::DeviceSelect::If(
        d_temp_storage, temp_storage_bytes,
        d_indices_raw, d_selected_row_ids,
        d_num_selected,
        num_rows,
        pred, join_stream);

    cuchk(cudaFreeAsync(d_temp_storage, join_stream));
    cuchk(cudaFreeAsync(d_indices_raw, join_stream)); // release the memory

    uint64_t h_num_selected = 0;
    cuchk(cudaMemcpyAsync(&h_num_selected, d_num_selected, sizeof(uint64_t), cudaMemcpyDeviceToHost, join_stream));
    cuchk(cudaStreamSynchronize(join_stream));

    uint64_t *d_row_offsets = nullptr;
    cuchk(cudaMallocAsync((void **)&d_row_offsets, sizeof(uint64_t) * (h_num_selected + 1), join_stream));
    cuchk(cudaMemsetAsync(d_row_offsets, 0, sizeof(uint64_t) * (h_num_selected + 1), join_stream));
    uint32_t *d_selected_count;
    cuchk(cudaMallocAsync((void **)&d_selected_count, sizeof(uint32_t) * (h_num_selected + 1), join_stream));
    cuchk(cudaMemsetAsync(d_selected_count + h_num_selected, 0, sizeof(uint32_t), join_stream));
    // Gather counts for selected rows
    gather_counts_kernel<<<GRID_DIM, BLOCK_DIM, 0, join_stream>>>(
        d_num_res_for_each,
        d_selected_row_ids,
        h_num_selected,
        d_selected_count);

    d_temp_storage = nullptr;
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
                                  d_selected_count, d_row_offsets,
                                  h_num_selected + 1, join_stream);
    cuchk(cudaMallocAsync(&d_temp_storage, temp_storage_bytes, join_stream));
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
                                  d_selected_count, d_row_offsets,
                                  h_num_selected + 1, join_stream);
    cuchk(cudaMemcpyAsync(h_num_res, d_row_offsets + h_num_selected,
                          sizeof(uint64_t),
                          cudaMemcpyDeviceToHost, join_stream));
    cuchk(cudaStreamSynchronize(join_stream));
    // Allocate result memory
    if (target_trie->data)
      delete target_trie->data;
    target_trie->data = new DataBlock();
    target_trie->data->init(*h_num_res, false);
    if (skip_final_materialization == false)
    {
      target_trie->data->allocate();

      join_write_refer_no_dup_dispatcher(
          d_selected_row_ids, h_num_selected,
          d_edge_values,
          d_selected_count, d_st_array,
          target_trie->data->d_data, d_row_offsets, target_trie->data->d_parents, task_per_warp);
    }

    if (d_row_offsets)
      cuchk(cudaFreeAsync(d_row_offsets, join_stream));
    cuchk(cudaFreeAsync(d_temp_storage, join_stream));
    cuchk(cudaFreeAsync(d_selected_row_ids, join_stream));
    cuchk(cudaFreeAsync(d_selected_count, join_stream));
    cuchk(cudaFreeAsync(d_num_selected, join_stream));
  }
  else
  {
    // std::cout << "Join with duplication check." << std::endl;
    uint32_t *d_mask;
    uint32_t mask_length = ht->max_values_per_edge_[directed_edge_idx];
    // Convert to number of 32-bit words (round up) and align to 32 words (128 bytes) for coalescing
    mask_length = ((mask_length + 31) / 32 + 31) / 32 * 32;
    cuchk(cudaMallocAsync((void **)&d_mask, sizeof(uint32_t) * num_rows * mask_length, join_stream));
    cuchk(cudaMemsetAsync(d_mask, 0, sizeof(uint32_t) * num_rows * mask_length, join_stream));

    join_refer_count_dup_kernel<<<GRID_DIM, BLOCK_DIM, 0, join_stream>>>(
        d_data_arrays, d_parent_arrays, root_data->num_layers,
        d_select_mask, num_rows, root_data->num_cols, col_key,
        d_edge_keys, d_edge_values, d_edge_offsets,
        ht->h_num_buckets_[directed_edge_idx],
        ht->h_hash_table_offs_[directed_edge_idx],
        directed_edge_idx,
        d_mask, mask_length, d_st_array,
        dup_check_type, num_dup_cols, d_dup_cols);

    uint64_t *d_row_offsets = nullptr;
    cuchk(cudaMallocAsync((void **)&d_row_offsets, sizeof(uint64_t) * (num_rows + 1), join_stream));

    size_t temp_storage_bytes = 0;
    void *d_temp_storage = nullptr;

    // Create segment offsets: [0, mask_length, 2*mask_length, ..., num_rows*mask_length]
    uint32_t *d_segment_offsets = nullptr;
    cuchk(cudaMallocAsync((void **)&d_segment_offsets, sizeof(uint32_t) * (num_rows + 1), join_stream));
    thrust::device_ptr<uint32_t> offsets_ptr(d_segment_offsets);
    // Use stream-bound execution policy - cudaStreamDefault since this is in a synchronous path
    thrust::sequence(thrust::cuda::par.on(join_stream),
                     offsets_ptr, offsets_ptr + num_rows + 1, (uint32_t)0u, mask_length);
    // Create a transform iterator that applies __popc to each mask tile
    auto popcount_iterator = thrust::make_transform_iterator(d_mask, PopcToCount{});
    // Use CUB DeviceSegmentedReduce to sum popcounts per row
    cuchk(cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes,
                                          popcount_iterator, d_num_res_for_each,
                                          num_rows, d_segment_offsets, d_segment_offsets + 1, join_stream));
    cuchk(cudaMallocAsync(&d_temp_storage, temp_storage_bytes, join_stream));
    cuchk(cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes,
                                          popcount_iterator, d_num_res_for_each,
                                          num_rows, d_segment_offsets, d_segment_offsets + 1, join_stream));
    cuchk(cudaFreeAsync(d_segment_offsets, join_stream));
    cuchk(cudaFreeAsync(d_temp_storage, join_stream));
    d_temp_storage = nullptr;
    temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
                                  d_num_res_for_each, d_row_offsets,
                                  num_rows + 1, join_stream);
    cuchk(cudaMallocAsync(&d_temp_storage, temp_storage_bytes, join_stream));
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
                                  d_num_res_for_each, d_row_offsets,
                                  num_rows + 1, join_stream);
    // Copy total count from d_row_offsets[num_rows] to host
    cuchk(cudaMemcpyAsync(h_num_res, d_row_offsets + num_rows,
                          sizeof(uint64_t),
                          cudaMemcpyDeviceToHost, join_stream));
    // Free temporary storage
    cuchk(cudaFreeAsync(d_temp_storage, join_stream));
    cuchk(cudaStreamSynchronize(join_stream));
    d_temp_storage = nullptr;
    temp_storage_bytes = 0;
    // Allocate result memory
    target_trie->data = new DataBlock();
    target_trie->data->init(*h_num_res, false);
    if (skip_final_materialization == false)
    {
      target_trie->data->allocate();

      join_refer_write_dup_kernel<<<GRID_DIM, BLOCK_DIM, 0, join_stream>>>(
          d_data_arrays, d_parent_arrays, root_data->num_layers, num_rows, root_data->num_cols, col_key,
          d_edge_keys, d_edge_values, d_edge_offsets,
          ht->h_num_buckets_[directed_edge_idx],
          ht->h_hash_table_offs_[directed_edge_idx],
          directed_edge_idx,
          d_mask, mask_length, d_st_array,
          target_trie->data->d_data,
          d_row_offsets,
          target_trie->data->d_parents,
          dup_check_type, num_dup_cols, d_dup_cols);
      cuchk(cudaStreamSynchronize(join_stream));
    }

    if (d_row_offsets)
      cuchk(cudaFreeAsync(d_row_offsets, join_stream));
    cuchk(cudaFreeAsync(d_mask, join_stream));
    cuchk(cudaFreeAsync(d_st_array, join_stream));
    cuchk(cudaStreamSynchronize(join_stream));
  }
  // Update target properties
  cuchk(cudaStreamSynchronize(join_stream));
  target_trie->updateNumResults(*h_num_res);
  target_trie->data->num_rows = *h_num_res;

#ifndef NDEBUG
  std::cout << "refer_join: " << *h_num_res << " results produced" << std::endl;
#endif

  // Cleanup
  delete h_num_res;
  // cuchk(cudaFreeHost(h_num_res));
  // cuchk(cudaFree(d_num_res_total));
  cuchk(cudaFreeAsync(d_num_res_for_each, join_stream));
  cuchk(cudaStreamSynchronize(join_stream));
}

void process(
    cpuGraph *hq, cpuGraph *hg, gpuGraph *dg,
    UnifiedTrie *frag_trie, UnifiedTrie *probe_trie, UnifiedTrie *target_trie)
{
  // Find which columns in frag_trie contain the probe_trie vertices
  bool found[2] = {false, false};
  uint32_t col[2] = {UINT32_MAX, UINT32_MAX};

  bool is_subset = false;

  for (int i = 0; i < frag_trie->num_cols; ++i)
  {
    for (int j = 0; j < probe_trie->num_cols; ++j) // j = 0 or 1.
    {
      if (frag_trie->mapped_query_us_[i] == probe_trie->edge_vertices[j])
      {
        found[j] = true;
        col[j] = i;
      }
    }
    if (is_subset = found[0] && found[1])
      break;
  }

  if (is_subset)
  {
    // Subset selection: both probe vertices are in fragment
    // non-tree edge must be bidirectionally valid.
    // std::cout << "Performing subset selection join." << std::endl;
#ifndef NDEBUG
    // std::cout << "Performing subset selection join." << std::endl;
#endif
    uint32_t col_key = std::min(col[0], col[1]);
    uint32_t col_value = std::max(col[0], col[1]);
    host_select(hq, frag_trie, probe_trie, target_trie, col_key, col_value);
  }
  else if (frag_trie->mask_parent_id != UINT32_MAX)
  {
    // std::cout << "Performing reference join." << std::endl;
#ifndef NDEBUG
    // std::cout << "Performing reference join." << std::endl;
#endif
    // Reference join: fragment has a mask parent
    target_trie->num_cols++;
    uint32_t join_col = found[0] ? col[0] : col[1];
    bool is_forward = found[0];
    refer_join(hq, frag_trie, probe_trie, target_trie, join_col, is_forward);
  }
  else
  {
    // std::cout << "Performing direct join." << std::endl;
#ifndef NDEBUG
    // std::cout << "Performing direct join." << std::endl;
#endif
    // Direct join: fragment has no mask parent
    target_trie->num_cols++;
    uint32_t join_col = found[0] ? col[0] : col[1];
    bool is_forward = found[0];
    direct_join(hq, frag_trie, probe_trie, target_trie, join_col, is_forward);
  }
}

void join(
    cpuGraph *hq, cpuGraph *hg, gpuGraph *dg)
{
  auto lat = &Lattice::getInstance();
  auto utm = &UnifiedTrieManager::getInstance();
  auto mem_mgr = &MemoryManager::getInstance();
  auto ht = &HashTable::getInstance();

  // stream init
  cuchk(cudaStreamCreate(&join_stream));

  // build __constant__ lookup table (only once.)
  // Allocate device memory for the HashLookupTables structure
  HashLookupTables *d_lookup;
  cuchk(cudaMallocAsync((void **)&d_lookup, sizeof(HashLookupTables), join_stream));

  // Create host-side structure and populate with device pointers from HashTable
  HashLookupTables h_lookup_stack;
  h_lookup_stack.hash_constants = ht->d_hash_constants_;
  for (int i = 0; i < NUM_TABLES; ++i)
  {
    h_lookup_stack.keys[i] = ht->d_keys_[i];
    h_lookup_stack.pos[i] = ht->d_pos_in_org[i];
  }

  // Copy the structure (containing device pointers) from host to device
  cuchk(cudaMemcpyAsync(d_lookup, &h_lookup_stack, sizeof(HashLookupTables), cudaMemcpyHostToDevice, join_stream));

  // Set the __constant__ pointer to point to the device structure
  cuchk(cudaMemcpyToSymbolAsync(lookup, &d_lookup, sizeof(HashLookupTables *), 0, cudaMemcpyHostToDevice, join_stream));

  std::stack<int> computation_stack;
  uint32_t computation_count = 0;
  uint32_t cache_hits = 0;

  // global info
  cuchk(cudaMallocAsync((void **)&d_dup_cols, sizeof(uint32_t) * NUM_VQ, join_stream));
  h_dup_cols = new uint32_t[NUM_VQ];

  cuchk(cudaMallocAsync((void **)&d_data_arrays, sizeof(vtype *) * MAX_VQ, join_stream));
  cuchk(cudaMallocAsync((void **)&d_parent_arrays, sizeof(uint64_t *) * MAX_VQ, join_stream));
  uint32_t debug_iteration = 0;

  while (!lat->uncomputed_rq_ids.empty())
  {
    int cur_trie_id = -1;

    if (computation_stack.empty())
    {
      cur_trie_id = utm->find_best_edge_candidate();
      if (cur_trie_id == UINT32_MAX)
        break;
      computation_stack.push(cur_trie_id);
    }

    while (!computation_stack.empty())
    {
      cur_trie_id = computation_stack.top();
      computation_stack.pop();

      skip_final_materialization = false;

      if (computation_stack.empty())
      {
        if (lat->uncomputed_reachable_nodes[cur_trie_id].size() < lat->uncomputed_rq_ids.size())
          cur_trie_id = utm->find_best_edge_candidate();
      }

      UnifiedTrie *cur_trie = utm->getTrie(cur_trie_id);

      // mem_mgr->record_access(cur_trie_id);
      // if (!mem_mgr->is_intermediate_on_gpu(cur_trie_id))
      // {
      //   if (!mem_mgr->load_intermediate_to_gpu(cur_trie_id, lat, utm))
      //   {
      //     std::cerr << "Failed to load trie " << cur_trie_id << " to GPU memory." << std::endl;
      //     return;
      //   }
      // }
      // else
      //   ++cache_hits;

      if (lat->uncomputed_reachable_rq[cur_trie_id].empty())
        continue;

      uint32_t probe_trie_id = utm->find_best_edge_candidate(cur_trie->et);
      if (probe_trie_id == UINT32_MAX)
        continue;

      UnifiedTrie *probe_trie = utm->getTrie(probe_trie_id);
      if (probe_trie == nullptr)
        continue;

      ettype target_trie_et = cur_trie->et | probe_trie->et;
      uint32_t target_trie_id = lat->et2id[target_trie_et];
      UnifiedTrie *target_trie = utm->getTrie(target_trie_id);
      if (target_trie == nullptr)
      {
        target_trie = utm->createTrie(
            target_trie_id, target_trie_et,
            cur_trie->num_layers + 1, cur_trie->num_cols);
      }
      else
      {
        computation_stack.push(cur_trie_id);
        computation_stack.push(target_trie_id);
      }

#ifndef NDEBUG
      std::cout << "processing " << cur_trie_id << " and " << probe_trie_id
                << " to produce " << target_trie_id << std::endl;
#endif
      // mem_mgr->pin_intermediate(cur_trie_id);
      // mem_mgr->pin_intermediate(probe_trie_id);
      // mem_mgr->pin_intermediate(target_trie_id);
      // mem_mgr->load_intermediate_to_gpu(probe_trie_id, lat, utm);

      skip_final_materialization = (lat->uncomputed_reachable_rq[target_trie_id].size() == 0);

      process(hq, hg, dg, cur_trie, probe_trie, target_trie);

      lat->mark_computed(target_trie_id);
      computation_count++;

      lat->update_contribution_on_computation(target_trie_et);

      // mem_mgr->unpin_intermediate(cur_trie_id);
      // mem_mgr->unpin_intermediate(probe_trie_id);
      // if (lat->uncomputed_reachable_rq[target_trie_id].empty())
      //   mem_mgr->unpin_intermediate(target_trie_id);

      computation_stack.push(cur_trie_id);
      computation_stack.push(target_trie_id);

#ifndef NDEBUG
      debug_iteration++;
      std::ofstream fout("debug_file" + std::to_string(debug_iteration - 1) + ".txt");
      fout << "Debug iteration: " << debug_iteration << std::endl;
      fout << cur_trie_id << " " << probe_trie_id << " -> " << target_trie_id << std::endl;
      fout << "result: " << target_trie->num_results << " rows." << std::endl;

      if (debug_iteration != 3 && debug_iteration != 4)
      {
        uint32_t *d_output_buffer;
        uint64_t *d_output_count;
        cuchk(cudaMallocAsync((void **)&d_output_buffer, sizeof(uint32_t) * target_trie->num_results * target_trie->num_cols, join_stream));
        cuchk(cudaMallocAsync((void **)&d_output_count, sizeof(uint64_t), join_stream));
        cuchk(cudaMemsetAsync(d_output_count, 0, sizeof(uint64_t), join_stream));

        auto data_arrays = utm->get_data_arrays(target_trie->id);
        auto parent_arrays = utm->get_parent_arrays(target_trie->id);
        cuchk(cudaMemcpyAsync(d_data_arrays, data_arrays.data(), sizeof(vtype *) * data_arrays.size(),
                              cudaMemcpyHostToDevice, join_stream));
        cuchk(cudaMemcpyAsync(d_parent_arrays, parent_arrays.data(), sizeof(uint64_t *) * parent_arrays.size(),
                              cudaMemcpyHostToDevice, join_stream));

        print_trie_kernel<<<GRID_DIM, BLOCK_DIM, 0, join_stream>>>(
            d_data_arrays, d_parent_arrays,
            target_trie->num_results, target_trie->num_cols,
            d_output_buffer, d_output_count);

        uint32_t *h_output_buffer = new uint32_t[target_trie->num_results * target_trie->num_cols];
        cuchk(cudaMemcpyAsync(h_output_buffer, d_output_buffer,
                              sizeof(uint32_t) * target_trie->num_results * target_trie->num_cols,
                              cudaMemcpyDeviceToHost, join_stream));
        for (int i = 0; i < target_trie->num_results; ++i)
        {
          for (int j = 0; j < target_trie->num_cols; ++j)
          {
            fout << h_output_buffer[i * target_trie->num_cols + j] << " ";
          }
          fout << std::endl;
        }
        cuchk(cudaFreeAsync(d_output_buffer, join_stream));
        cuchk(cudaFreeAsync(d_output_count, join_stream));
        delete[] h_output_buffer;
      }
      fout.close();
#endif
    }
  }
  cuchk(cudaFreeAsync(d_dup_cols, join_stream));
  d_dup_cols = nullptr;
  delete[] h_dup_cols;
  // cuchk(cudaFree(dev_space));
  cuchk(cudaFreeAsync(d_data_arrays, join_stream));
  cuchk(cudaFreeAsync(d_parent_arrays, join_stream));
  cuchk(cudaFreeAsync(d_lookup, join_stream));
  cuchk(cudaStreamSynchronize(join_stream));
  cuchk(cudaStreamDestroy(join_stream));
}

void join_count_nodup_dispatcher(
    // trie_inter
    vtype **vertex_arrays_, uint64_t **parent_pointers_,
    uint64_t num_rows, uint32_t num_cols, uint32_t join_col,
    // trie_edge
    vtype *edge_key_array_, vtype *edge_value_array_, uint32_t *edge_offsets_,
    // hash
    uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx,
    // mask
    uint32_t *d_num_res_for_each, uint32_t *d_st_array, uint32_t task_per_warp)
{
  switch (task_per_warp)
  {
  case 1:
    join_count_kernel_nodup<1><<<GRID_DIM, BLOCK_DIM, 0, join_stream>>>(
        vertex_arrays_, parent_pointers_, num_rows, num_cols, join_col,
        edge_key_array_, edge_value_array_, edge_offsets_,
        num_buckets, edge_offset, directed_edge_idx, d_num_res_for_each, d_st_array);
    break;
  case 2:
    join_count_kernel_nodup<2><<<GRID_DIM, BLOCK_DIM, 0, join_stream>>>(
        vertex_arrays_, parent_pointers_, num_rows, num_cols, join_col,
        edge_key_array_, edge_value_array_, edge_offsets_,
        num_buckets, edge_offset, directed_edge_idx,
        d_num_res_for_each, d_st_array);
    break;
  case 4:
    join_count_kernel_nodup<4><<<GRID_DIM, BLOCK_DIM, 0, join_stream>>>(
        vertex_arrays_, parent_pointers_, num_rows, num_cols, join_col,
        edge_key_array_, edge_value_array_, edge_offsets_,
        num_buckets, edge_offset, directed_edge_idx,
        d_num_res_for_each, d_st_array);
    break;
  case 8:
    join_count_kernel_nodup<8><<<GRID_DIM, BLOCK_DIM, 0, join_stream>>>(
        vertex_arrays_, parent_pointers_, num_rows, num_cols, join_col,
        edge_key_array_, edge_value_array_, edge_offsets_,
        num_buckets, edge_offset, directed_edge_idx,
        d_num_res_for_each, d_st_array);
    break;
  case 16:
    join_count_kernel_nodup<16><<<GRID_DIM, BLOCK_DIM, 0, join_stream>>>(
        vertex_arrays_, parent_pointers_, num_rows, num_cols, join_col,
        edge_key_array_, edge_value_array_, edge_offsets_,
        num_buckets, edge_offset, directed_edge_idx,
        d_num_res_for_each, d_st_array);
    break;
  case 32:
    join_count_kernel_nodup<32><<<GRID_DIM, BLOCK_DIM, 0, join_stream>>>(
        vertex_arrays_, parent_pointers_, num_rows, num_cols, join_col,
        edge_key_array_, edge_value_array_, edge_offsets_,
        num_buckets, edge_offset, directed_edge_idx,
        d_num_res_for_each, d_st_array);
    break;
  default:
    std::cerr << "Unsupported task_per_warp: " << task_per_warp << std::endl;
    exit(EXIT_FAILURE);
  }
}

void join_write_nodup_dispatcher(
    uint64_t num_rows,
    // trie_edge
    vtype *edge_value_array_,
    uint32_t *d_num_res_for_each, uint32_t *d_st_array,
    uint32_t *d_res, uint64_t *d_offsets_of_, uint32_t num_tasks_per_warp, uint64_t *d_parent_res)
{
  switch (num_tasks_per_warp)
  {
  case 1:
    join_write_kernel_nodup<1><<<GRID_DIM, BLOCK_DIM, 0, join_stream>>>(
        num_rows,
        edge_value_array_,
        d_num_res_for_each, d_st_array,
        d_res, d_offsets_of_, d_parent_res);
    break;
  case 2:
    join_write_kernel_nodup<2><<<GRID_DIM, BLOCK_DIM, 0, join_stream>>>(
        num_rows,
        edge_value_array_,
        d_num_res_for_each, d_st_array,
        d_res, d_offsets_of_, d_parent_res);
    break;
  case 4:
    join_write_kernel_nodup<4><<<GRID_DIM, BLOCK_DIM, 0, join_stream>>>(
        num_rows,
        edge_value_array_,
        d_num_res_for_each, d_st_array,
        d_res, d_offsets_of_, d_parent_res);
    break;
  case 8:
    join_write_kernel_nodup<8><<<GRID_DIM, BLOCK_DIM, 0, join_stream>>>(
        num_rows,
        edge_value_array_,
        d_num_res_for_each, d_st_array,
        d_res, d_offsets_of_, d_parent_res);
    break;
  case 16:
    join_write_kernel_nodup<16><<<GRID_DIM, BLOCK_DIM, 0, join_stream>>>(
        num_rows,
        edge_value_array_,
        d_num_res_for_each, d_st_array,
        d_res, d_offsets_of_, d_parent_res);
    break;
  case 32:
    join_write_kernel_nodup<32><<<GRID_DIM, BLOCK_DIM, 0, join_stream>>>(
        num_rows,
        edge_value_array_,
        d_num_res_for_each, d_st_array,
        d_res, d_offsets_of_, d_parent_res);
    break;
  default:
    std::cerr << "Unsupported task_per_warp: " << num_tasks_per_warp << std::endl;
    exit(EXIT_FAILURE);
  }
  cuchk(cudaStreamSynchronize(join_stream));
}

void join_count_refer_no_dup_dispatcher(
    vtype **vertex_arrays, uint64_t **parent_pointers,
    uint64_t num_rows, uint32_t num_cols, uint32_t join_col,
    uint32_t *select_bool_vec,
    vtype *edge_key_array, vtype *edge_value_array, uint32_t *edge_offsets,
    uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx,
    uint32_t *d_num_res_for_each, uint32_t *d_st_array, uint32_t task_per_warp)
{
  switch (task_per_warp)
  {
  case 1:
    join_count_refer_kernel_nodup<1><<<GRID_DIM, BLOCK_DIM, 0, join_stream>>>(
        vertex_arrays, parent_pointers,
        num_rows, num_cols, join_col,
        select_bool_vec,
        edge_key_array, edge_value_array, edge_offsets,
        num_buckets, edge_offset, directed_edge_idx,
        d_num_res_for_each, d_st_array);
    break;
  case 2:
    join_count_refer_kernel_nodup<2><<<GRID_DIM, BLOCK_DIM, 0, join_stream>>>(
        vertex_arrays, parent_pointers,
        num_rows, num_cols, join_col,
        select_bool_vec,
        edge_key_array, edge_value_array, edge_offsets,
        num_buckets, edge_offset, directed_edge_idx,
        d_num_res_for_each, d_st_array);
    break;
  case 4:
    join_count_refer_kernel_nodup<4><<<GRID_DIM, BLOCK_DIM, 0, join_stream>>>(
        vertex_arrays, parent_pointers,
        num_rows, num_cols, join_col,
        select_bool_vec,
        edge_key_array, edge_value_array, edge_offsets,
        num_buckets, edge_offset, directed_edge_idx,
        d_num_res_for_each, d_st_array);
    break;
  case 8:
    join_count_refer_kernel_nodup<8><<<GRID_DIM, BLOCK_DIM, 0, join_stream>>>(
        vertex_arrays, parent_pointers,
        num_rows, num_cols, join_col,
        select_bool_vec,
        edge_key_array, edge_value_array, edge_offsets,
        num_buckets, edge_offset, directed_edge_idx,
        d_num_res_for_each, d_st_array);
    break;
  case 16:
    join_count_refer_kernel_nodup<16><<<GRID_DIM, BLOCK_DIM, 0, join_stream>>>(
        vertex_arrays, parent_pointers,
        num_rows, num_cols, join_col,
        select_bool_vec,
        edge_key_array, edge_value_array, edge_offsets,
        num_buckets, edge_offset, directed_edge_idx,
        d_num_res_for_each, d_st_array);
    break;
  case 32:
    join_count_refer_kernel_nodup<32><<<GRID_DIM, BLOCK_DIM, 0, join_stream>>>(
        vertex_arrays, parent_pointers,
        num_rows, num_cols, join_col,
        select_bool_vec,
        edge_key_array, edge_value_array, edge_offsets,
        num_buckets, edge_offset, directed_edge_idx,
        d_num_res_for_each, d_st_array);
    break;
  default:
    std::cerr << "Unsupported task_per_warp: " << task_per_warp << std::endl;
    exit(EXIT_FAILURE);
  }
}

void join_write_refer_no_dup_dispatcher(
    uint64_t *valid_row_ids, uint64_t num_valid_rows,
    vtype *edge_value_array_,
    uint32_t *d_selected_count, uint32_t *d_st_array,
    uint32_t *d_res, uint64_t *d_row_offsets, uint64_t *d_parent_res, uint32_t task_per_warp)
{
  switch (task_per_warp)
  {
  case 1:
    join_write_refer_kernel_nodup<1><<<GRID_DIM, BLOCK_DIM, 0, join_stream>>>(
        valid_row_ids, num_valid_rows,
        edge_value_array_,
        d_selected_count, d_st_array,
        d_res, d_row_offsets, d_parent_res);
    break;
  case 2:
    join_write_refer_kernel_nodup<2><<<GRID_DIM, BLOCK_DIM, 0, join_stream>>>(
        valid_row_ids, num_valid_rows,
        edge_value_array_,
        d_selected_count, d_st_array,
        d_res, d_row_offsets, d_parent_res);
    break;
  case 4:
    join_write_refer_kernel_nodup<4><<<GRID_DIM, BLOCK_DIM, 0, join_stream>>>(
        valid_row_ids, num_valid_rows,
        edge_value_array_,
        d_selected_count, d_st_array,
        d_res, d_row_offsets, d_parent_res);
    break;
  case 8:
    join_write_refer_kernel_nodup<8><<<GRID_DIM, BLOCK_DIM, 0, join_stream>>>(
        valid_row_ids, num_valid_rows,
        edge_value_array_,
        d_selected_count, d_st_array,
        d_res, d_row_offsets, d_parent_res);
    break;
  case 16:
    join_write_refer_kernel_nodup<16><<<GRID_DIM, BLOCK_DIM, 0, join_stream>>>(
        valid_row_ids, num_valid_rows,
        edge_value_array_,
        d_selected_count, d_st_array,
        d_res, d_row_offsets, d_parent_res);
    break;
  case 32:
    join_write_refer_kernel_nodup<32><<<GRID_DIM, BLOCK_DIM, 0, join_stream>>>(
        valid_row_ids, num_valid_rows,
        edge_value_array_,
        d_selected_count, d_st_array,
        d_res, d_row_offsets, d_parent_res);
    break;
  default:
    std::cerr << "Unsupported task_per_warp: " << task_per_warp << std::endl;
    exit(EXIT_FAILURE);
  }
  cuchk(cudaStreamSynchronize(join_stream));
}