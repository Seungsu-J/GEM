#include "direct_join.cuh"
#include "direct_join_kernel.cuh"
#include "host_memory_manager.cuh"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
  [[noreturn]] void fssmExit(int code, const char *file, int line)
  {
    throw std::runtime_error(std::string("exit(") + std::to_string(code) + ") at " +
                             (file != nullptr ? file : "<unknown>") + ":" + std::to_string(line));
  }

  void resetTargetTrieData(UnifiedTrie *trie, uint64_t num_rows, bool is_mask)
  {
    if (trie == nullptr)
      throw std::runtime_error("resetTargetTrieData: target trie is null.");

    if (trie->data != nullptr || trie->r_data != nullptr)
    {
      trie->deallocateData();
    }

    trie->data = new DataBlock();
    trie->data->init(num_rows, is_mask);
  }
} // namespace

#define exit(code) fssmExit((code), __FILE__, __LINE__)

template <typename OffsetT>
uint64_t find_frag_range(const OffsetT *h_row_offsets, uint64_t left, uint64_t right, uint64_t target_rows_limit);

namespace
{
  void direct_join_without_duplication_streaming();
  void direct_join_with_duplication_streaming_nomask();
}

void direct_join_without_duplication()
{
  auto &mem_mgr = MemoryManager::getInstance();
  auto &ht = HashTable::getInstance();
  auto &jc = JoinContext::getInstance();
  // std::cout << "Performing Direct Join without duplication check." << std::endl;
#ifndef NDEBUG
  std::cout << "Performing Direct Join without duplication check." << std::endl;
#endif

  jc.direct_join_without_duplication_context();

  join_count_nodup_dispatcher(
      // trie_inter
      jc.d_data_arrays_, jc.d_parent_arrays_,
      jc.num_frag_rows, jc.frag_trie->num_cols, jc.col_key,
      // trie_edge
      jc.d_edge_keys_, jc.d_edge_values_, jc.d_edge_offsets_,
      // hash
      ht.h_num_buckets_[jc.directed_edge_idx],
      ht.h_hash_table_offs_[jc.directed_edge_idx],
      jc.directed_edge_idx,
      // mask
      jc.d_num_res_for_each,
      jc.d_st_array,
      jc.task_per_warp);
  mem_mgr.unpin_climb_data();
  mem_mgr.unpin_climb_parents();

  uint64_t *d_row_offsets = nullptr;
  if (!mem_mgr.allocate_temporal(d_row_offsets, sizeof(uint64_t) * (jc.num_frag_rows + 1)))
  {
    std::cout << "direct_join_without_duplication: falling back to streaming join." << std::endl;
    mem_mgr.deallocate_temporal(jc.d_num_res_for_each);
    mem_mgr.deallocate_temporal(jc.d_st_array);
    direct_join_without_duplication_streaming();
    return;
  }
  // exclusive scan on d_num_res_for_each to write position

  size_t temp_storage_bytes = 0;
  void *d_temp_storage = nullptr;

  cub::DeviceScan::ExclusiveScan(d_temp_storage, temp_storage_bytes,
                                 jc.d_num_res_for_each, d_row_offsets,
                                 cub::Sum(), static_cast<uint64_t>(0), jc.num_frag_rows + 1, jc.join_stream);
  if (!mem_mgr.allocate_temporal(d_temp_storage, temp_storage_bytes))
  {
    std::cout << "direct_join_without_duplication: falling back to streaming join." << std::endl;
    mem_mgr.deallocate_temporal(d_row_offsets);
    mem_mgr.deallocate_temporal(jc.d_num_res_for_each);
    mem_mgr.deallocate_temporal(jc.d_st_array);
    direct_join_without_duplication_streaming();
    return;
  }
  cub::DeviceScan::ExclusiveScan(d_temp_storage, temp_storage_bytes,
                                 jc.d_num_res_for_each, d_row_offsets,
                                 cub::Sum(), static_cast<uint64_t>(0), jc.num_frag_rows + 1, jc.join_stream);

  // Copy total count from d_row_offsets[num_rows] to host
  uint64_t h_num_res = 0;
  cuchk(cudaMemcpyAsync(&h_num_res, d_row_offsets + jc.num_frag_rows,
                        sizeof(uint64_t),
                        cudaMemcpyDeviceToHost, jc.join_stream));
  mem_mgr.deallocate_temporal(d_temp_storage);

  cuchk(cudaStreamSynchronize(jc.join_stream));
  jc.h_num_res[0] = h_num_res;
  resetTargetTrieData(jc.target_trie, jc.h_num_res[0], false);
#ifndef NDEBUG
  std::cout << "Direct Join produced " << jc.h_num_res[0] << " results." << std::endl;
#endif

  if (jc.skip_final_materialization == false)
  {
    // target_trie->data->allocate();
    bool allocate_success = mem_mgr.allocate_trie(jc.target_trie->id);

    // TODO: alternative: cub::DeviceMemcpy::Batched().
    // if time allows, implement cub::DeviceMemcpy::Batched() and compare the time consumed.

    if (allocate_success)
    {
      join_write_nodup_dispatcher(
          jc.num_frag_rows, jc.d_edge_values_,
          jc.d_num_res_for_each, jc.d_st_array, jc.target_trie->data->d_data,
          d_row_offsets, jc.task_per_warp, jc.target_trie->data->d_parents);
    }
    else // if (!allocate_success)
    {
      direct_join_without_duplication_pipeline_write(d_row_offsets);
    }
  }
  mem_mgr.deallocate_temporal(jc.d_num_res_for_each);

  if (d_row_offsets)
    mem_mgr.deallocate_temporal(d_row_offsets);
  cuchk(cudaStreamSynchronize(jc.join_stream));
}

void direct_join_without_duplication_debug()
{
  direct_join_without_duplication();
}

void direct_join_with_duplication()
{
  // std::cout << "Performing Direct Join with duplication check" << std::endl;

  auto &mem_mgr = MemoryManager::getInstance();
  auto &ht = HashTable::getInstance();
  auto &jc = JoinContext::getInstance();

  jc.direct_join_with_duplication_context();
  if (jc.d_mask == nullptr)
  {
    std::cout << "Falling back to on-the-fly duplication check (no value mask)." << std::endl;

    mem_mgr.deallocate_temporal(jc.d_num_res_for_each);
    mem_mgr.deallocate_temporal(jc.d_st_array);
    direct_join_with_duplication_streaming_nomask();
    return;
  }

  join_count_kernel_dup_2<<<GRID_DIM, BLOCK_DIM, 0, jc.join_stream>>>(
      jc.d_data_arrays_, jc.d_parent_arrays_, jc.num_frag_rows, jc.frag_trie->num_cols, jc.col_key,
      jc.d_edge_keys_, jc.d_edge_values_, jc.d_edge_offsets_,
      ht.h_num_buckets_[jc.directed_edge_idx],
      ht.h_hash_table_offs_[jc.directed_edge_idx],
      jc.directed_edge_idx,
      jc.d_mask, jc.mask_length, jc.d_st_array,
      jc.dup_check_type, jc.num_dup_cols, jc.d_dup_cols, jc.up_to_col);
  mem_mgr.unpin_climb_data();
  mem_mgr.unpin_climb_parents();

  uint64_t *d_row_offsets = nullptr;
  if (!mem_mgr.allocate_temporal(d_row_offsets, sizeof(uint64_t) * (jc.num_frag_rows + 1)))
  {
    std::cout << "direct_join_with_duplication: falling back to streaming join." << std::endl;
    mem_mgr.deallocate_temporal(jc.d_mask);
    mem_mgr.deallocate_temporal(jc.d_num_res_for_each);
    mem_mgr.deallocate_temporal(jc.d_st_array);
    direct_join_with_duplication_streaming_nomask();
    return;
  }
  cuchk(cudaMemsetAsync(d_row_offsets, 0, sizeof(uint64_t) * (jc.num_frag_rows + 1), jc.join_stream));

  // Exclusive prefix sum to compute write offsets for each row
  // Input: d_num_res_for_each[i] = count for row i
  // Output: d_row_offsets[i] = exclusive scan (starting offset for row i)
  //         d_row_offsets[num_rows] = total count
  size_t temp_storage_bytes = 0;
  void *d_temp_storage = nullptr;

  // Create segment offsets: [0, mask_length, 2*mask_length, ..., num_rows*mask_length]
  // Use 64-bit offsets to avoid overflow when num_rows_frag * mask_length exceeds 2^32.
  uint64_t *d_segment_offsets = nullptr;
  if (!mem_mgr.allocate_temporal(d_segment_offsets, sizeof(uint64_t) * (jc.num_frag_rows + 1)))
  {
    std::cout << "direct_join_with_duplication: falling back to streaming join." << std::endl;
    mem_mgr.deallocate_temporal(d_row_offsets);
    mem_mgr.deallocate_temporal(jc.d_mask);
    mem_mgr.deallocate_temporal(jc.d_num_res_for_each);
    mem_mgr.deallocate_temporal(jc.d_st_array);
    direct_join_with_duplication_streaming_nomask();
    return;
  }

  thrust::device_ptr<uint64_t> offsets_ptr(d_segment_offsets);
  // Use stream-bound execution policy to utilize the memory pool associated withjc.join_stream
  thrust::sequence(thrust::cuda::par.on(jc.join_stream),
                   offsets_ptr, offsets_ptr + jc.num_frag_rows + 1,
                   static_cast<uint64_t>(0),
                   static_cast<uint64_t>(jc.mask_length));

  // Create a transform iterator that applies __popc to each mask tile
  auto popcount_iterator = thrust::make_transform_iterator(jc.d_mask, PopcToCount{});

  // Use CUB DeviceSegmentedReduce to sum popcounts per row
  cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes,
                                  popcount_iterator, jc.d_num_res_for_each,
                                  jc.num_frag_rows, d_segment_offsets, d_segment_offsets + 1, jc.join_stream);
  if (!mem_mgr.allocate_temporal(d_temp_storage, temp_storage_bytes))
  {
    std::cout << "direct_join_with_duplication: falling back to streaming join." << std::endl;
    mem_mgr.deallocate_temporal(d_segment_offsets);
    mem_mgr.deallocate_temporal(d_row_offsets);
    mem_mgr.deallocate_temporal(jc.d_mask);
    mem_mgr.deallocate_temporal(jc.d_num_res_for_each);
    mem_mgr.deallocate_temporal(jc.d_st_array);
    direct_join_with_duplication_streaming_nomask();
    return;
  }
  cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes,
                                  popcount_iterator, jc.d_num_res_for_each,
                                  jc.num_frag_rows, d_segment_offsets, d_segment_offsets + 1, jc.join_stream);
  // Free temporary arrays
  mem_mgr.deallocate_temporal(d_segment_offsets);
  mem_mgr.deallocate_temporal(d_temp_storage);
  d_temp_storage = nullptr;
  temp_storage_bytes = 0;

  cub::DeviceScan::ExclusiveScan(d_temp_storage, temp_storage_bytes,
                                 jc.d_num_res_for_each, d_row_offsets,
                                 cub::Sum(), (uint64_t)0, jc.num_frag_rows + 1, jc.join_stream);
  if (!mem_mgr.allocate_temporal(d_temp_storage, temp_storage_bytes))
  {
    std::cout << "direct_join_with_duplication: falling back to streaming join." << std::endl;
    mem_mgr.deallocate_temporal(d_row_offsets);
    mem_mgr.deallocate_temporal(jc.d_mask);
    mem_mgr.deallocate_temporal(jc.d_num_res_for_each);
    mem_mgr.deallocate_temporal(jc.d_st_array);
    direct_join_with_duplication_streaming_nomask();
    return;
  }
  cub::DeviceScan::ExclusiveScan(d_temp_storage, temp_storage_bytes,
                                 jc.d_num_res_for_each, d_row_offsets,
                                 cub::Sum(), (uint64_t)0, jc.num_frag_rows + 1, jc.join_stream);

  mem_mgr.deallocate_temporal(d_temp_storage);
  mem_mgr.deallocate_temporal(jc.d_num_res_for_each);
  // Copy total count from d_row_offsets[num_rows] to host
  cuchk(cudaMemcpyAsync(jc.h_num_res, d_row_offsets + jc.num_frag_rows,
                        sizeof(uint64_t),
                        cudaMemcpyDeviceToHost, jc.join_stream));

  cuchk(cudaStreamSynchronize(jc.join_stream));
  resetTargetTrieData(jc.target_trie, *(jc.h_num_res), false);

#ifndef NDEBUG
  std::cout << "Direct Join produced " << *(jc.h_num_res) << " results." << std::endl;
#endif

  if (jc.skip_final_materialization == false)
  {
    bool alloc_success = mem_mgr.allocate_trie(jc.target_trie->id);
    // jc.target_trie->data->allocate();

    // cuchk(cudaMemset(jc.target_trie->data->d_data, 0xFF, sizeof(vtype) * h_num_res));
    // cuchk(cudaMemset(jc.target_trie->data->d_parents, 0xFF, sizeof(uint32_t) * h_num_res));
    // cuchk(cudaMalloc((void **)&jc.target_trie->data->d_data, sizeof(vtype) * h_num_res));

    if (alloc_success)
    {
      join_write_kernel_dup<<<GRID_DIM, BLOCK_DIM, 0, jc.join_stream>>>(
          jc.num_frag_rows, jc.d_edge_values_,
          jc.d_mask, jc.mask_length, d_row_offsets, jc.d_st_array,
          jc.target_trie->data->d_data, jc.target_trie->data->d_parents);
      cuchk(cudaStreamSynchronize(jc.join_stream));
    }
    else // if (!alloc_success)
    {
      direct_join_with_duplication_pipeline_write(d_row_offsets);
    }
  }
  mem_mgr.deallocate_temporal(d_row_offsets);
  mem_mgr.deallocate_temporal(jc.d_mask);

  cuchk(cudaStreamSynchronize(jc.join_stream));
}

namespace
{
  struct ColumnRange
  {
    uint64_t start = 0;
    uint64_t end = 0; // exclusive
  };

  uint64_t load_parent_host_or_device(
      uint32_t col_trie_id, uint64_t row_idx, cudaStream_t stream)
  {
    auto &mem_mgr = MemoryManager::getInstance();
    auto &utm = UnifiedTrieManager::getInstance();

    const int block_idx = static_cast<int>(col_trie_id) * 2 + 1;
    auto &block = mem_mgr.memory_blocks[block_idx];
    if (block.host_ptr != nullptr)
    {
      auto *h_parents = static_cast<uint64_t *>(block.host_ptr);
      return h_parents[row_idx];
    }

    UnifiedTrie *col_trie = utm.getTrie(col_trie_id);
    uint64_t value = UINT64_MAX;
    if (col_trie && col_trie->data && col_trie->data->d_parents)
    {
      cuchk(cudaMemcpyAsync(&value, col_trie->data->d_parents + row_idx, sizeof(uint64_t),
                            cudaMemcpyDeviceToHost, stream));
      cuchk(cudaStreamSynchronize(stream));
      return value;
    }
    if (col_trie && col_trie->r_data && col_trie->r_data->d_parents)
    {
      cuchk(cudaMemcpyAsync(&value, col_trie->r_data->d_parents + row_idx, sizeof(uint64_t),
                            cudaMemcpyDeviceToHost, stream));
      cuchk(cudaStreamSynchronize(stream));
      return value;
    }

    std::cerr << "load_parent_host_or_device: missing parents for trie " << col_trie_id << std::endl;
    exit(EXIT_FAILURE);
  }

  void compute_column_ranges(
      UnifiedTrie *trie,
      uint32_t col_upper_bound,
      uint64_t leaf_start,
      uint64_t leaf_end,
      std::vector<ColumnRange> &ranges,
      cudaStream_t stream)
  {
    const int num_cols = trie->num_cols;
    ranges.assign(num_cols, ColumnRange{});
    ranges[num_cols - 1] = ColumnRange{leaf_start, leaf_end};

    for (int col = num_cols - 1; col > static_cast<int>(col_upper_bound); --col)
    {
      ColumnRange cur = ranges[col];
      if (cur.end <= cur.start)
      {
        ranges[col - 1] = ColumnRange{0, 0};
        continue;
      }

      uint32_t col_trie_id = trie->column_ids[col];
      uint64_t parent_start = load_parent_host_or_device(col_trie_id, cur.start, stream);
      uint64_t parent_end = load_parent_host_or_device(col_trie_id, cur.end - 1, stream);
      uint64_t st = std::min(parent_start, parent_end);
      uint64_t ed = std::max(parent_start, parent_end) + 1;
      ranges[col - 1] = ColumnRange{st, ed};
    }
  }

  void join_count_nodup_pipeline_dispatcher(
      JoinInfo<uint32_t> *data_info_arrays_, JoinInfo<uint64_t> *parent_info_arrays_,
      uint64_t pipeline_off_st, uint64_t pipeline_num_rows,
      uint32_t num_cols, uint32_t join_col,
      vtype *edge_key_array_, vtype *edge_value_array_, uint32_t *edge_offsets_,
      uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx,
      uint32_t *d_num_res_for_each, uint32_t *d_st_array,
      uint32_t task_per_warp, cudaStream_t stream)
  {
    switch (task_per_warp)
    {
    case 1:
      join_count_kernel_nodup_pipeline_info<1><<<GRID_DIM, BLOCK_DIM, 0, stream>>>(
          data_info_arrays_, parent_info_arrays_,
          pipeline_off_st, pipeline_num_rows,
          num_cols, join_col,
          edge_key_array_, edge_value_array_, edge_offsets_,
          num_buckets, edge_offset, directed_edge_idx,
          d_num_res_for_each, d_st_array);
      break;
    case 2:
      join_count_kernel_nodup_pipeline_info<2><<<GRID_DIM, BLOCK_DIM, 0, stream>>>(
          data_info_arrays_, parent_info_arrays_,
          pipeline_off_st, pipeline_num_rows,
          num_cols, join_col,
          edge_key_array_, edge_value_array_, edge_offsets_,
          num_buckets, edge_offset, directed_edge_idx,
          d_num_res_for_each, d_st_array);
      break;
    case 4:
      join_count_kernel_nodup_pipeline_info<4><<<GRID_DIM, BLOCK_DIM, 0, stream>>>(
          data_info_arrays_, parent_info_arrays_,
          pipeline_off_st, pipeline_num_rows,
          num_cols, join_col,
          edge_key_array_, edge_value_array_, edge_offsets_,
          num_buckets, edge_offset, directed_edge_idx,
          d_num_res_for_each, d_st_array);
      break;
    case 8:
      join_count_kernel_nodup_pipeline_info<8><<<GRID_DIM, BLOCK_DIM, 0, stream>>>(
          data_info_arrays_, parent_info_arrays_,
          pipeline_off_st, pipeline_num_rows,
          num_cols, join_col,
          edge_key_array_, edge_value_array_, edge_offsets_,
          num_buckets, edge_offset, directed_edge_idx,
          d_num_res_for_each, d_st_array);
      break;
    case 16:
      join_count_kernel_nodup_pipeline_info<16><<<GRID_DIM, BLOCK_DIM, 0, stream>>>(
          data_info_arrays_, parent_info_arrays_,
          pipeline_off_st, pipeline_num_rows,
          num_cols, join_col,
          edge_key_array_, edge_value_array_, edge_offsets_,
          num_buckets, edge_offset, directed_edge_idx,
          d_num_res_for_each, d_st_array);
      break;
    case 32:
      join_count_kernel_nodup_pipeline_info<32><<<GRID_DIM, BLOCK_DIM, 0, stream>>>(
          data_info_arrays_, parent_info_arrays_,
          pipeline_off_st, pipeline_num_rows,
          num_cols, join_col,
          edge_key_array_, edge_value_array_, edge_offsets_,
          num_buckets, edge_offset, directed_edge_idx,
          d_num_res_for_each, d_st_array);
      break;
    default:
      std::cerr << "Unsupported task_per_warp: " << task_per_warp << std::endl;
      exit(EXIT_FAILURE);
    }
  }

  void join_count_nodup_pipeline_local_dispatcher(
      JoinInfo<uint32_t> *data_info_arrays_, JoinInfo<uint64_t> *parent_info_arrays_,
      uint64_t pipeline_off_st, uint64_t pipeline_num_rows,
      uint32_t num_cols, uint32_t join_col,
      vtype *edge_key_array_, vtype *edge_value_array_, uint32_t *edge_offsets_,
      uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx,
      uint32_t *d_num_res_for_each, uint32_t *d_st_array,
      uint32_t task_per_warp, cudaStream_t stream)
  {
    switch (task_per_warp)
    {
    case 1:
      join_count_kernel_nodup_pipeline_local<1><<<GRID_DIM, BLOCK_DIM, 0, stream>>>(
          data_info_arrays_, parent_info_arrays_,
          pipeline_off_st, pipeline_num_rows,
          num_cols, join_col,
          edge_key_array_, edge_value_array_, edge_offsets_,
          num_buckets, edge_offset, directed_edge_idx,
          d_num_res_for_each, d_st_array);
      break;
    case 2:
      join_count_kernel_nodup_pipeline_local<2><<<GRID_DIM, BLOCK_DIM, 0, stream>>>(
          data_info_arrays_, parent_info_arrays_,
          pipeline_off_st, pipeline_num_rows,
          num_cols, join_col,
          edge_key_array_, edge_value_array_, edge_offsets_,
          num_buckets, edge_offset, directed_edge_idx,
          d_num_res_for_each, d_st_array);
      break;
    case 4:
      join_count_kernel_nodup_pipeline_local<4><<<GRID_DIM, BLOCK_DIM, 0, stream>>>(
          data_info_arrays_, parent_info_arrays_,
          pipeline_off_st, pipeline_num_rows,
          num_cols, join_col,
          edge_key_array_, edge_value_array_, edge_offsets_,
          num_buckets, edge_offset, directed_edge_idx,
          d_num_res_for_each, d_st_array);
      break;
    case 8:
      join_count_kernel_nodup_pipeline_local<8><<<GRID_DIM, BLOCK_DIM, 0, stream>>>(
          data_info_arrays_, parent_info_arrays_,
          pipeline_off_st, pipeline_num_rows,
          num_cols, join_col,
          edge_key_array_, edge_value_array_, edge_offsets_,
          num_buckets, edge_offset, directed_edge_idx,
          d_num_res_for_each, d_st_array);
      break;
    case 16:
      join_count_kernel_nodup_pipeline_local<16><<<GRID_DIM, BLOCK_DIM, 0, stream>>>(
          data_info_arrays_, parent_info_arrays_,
          pipeline_off_st, pipeline_num_rows,
          num_cols, join_col,
          edge_key_array_, edge_value_array_, edge_offsets_,
          num_buckets, edge_offset, directed_edge_idx,
          d_num_res_for_each, d_st_array);
      break;
    case 32:
      join_count_kernel_nodup_pipeline_local<32><<<GRID_DIM, BLOCK_DIM, 0, stream>>>(
          data_info_arrays_, parent_info_arrays_,
          pipeline_off_st, pipeline_num_rows,
          num_cols, join_col,
          edge_key_array_, edge_value_array_, edge_offsets_,
          num_buckets, edge_offset, directed_edge_idx,
          d_num_res_for_each, d_st_array);
      break;
    default:
      std::cerr << "Unsupported task_per_warp: " << task_per_warp << std::endl;
      exit(EXIT_FAILURE);
    }
  }

  struct ChunkStagingBuffers
  {
    std::vector<uint64_t *> parent_buffers;
    std::vector<uint32_t *> data_buffers;
  };

  void cleanup_chunk_staging_buffers(ChunkStagingBuffers &buffers)
  {
    auto &mem_mgr = MemoryManager::getInstance();
    for (auto *ptr : buffers.parent_buffers)
    {
      uint64_t *tmp = ptr;
      mem_mgr.deallocate_temporal(tmp);
    }
    for (auto *ptr : buffers.data_buffers)
    {
      uint32_t *tmp = ptr;
      mem_mgr.deallocate_temporal(tmp);
    }
    buffers.parent_buffers.clear();
    buffers.data_buffers.clear();
  }

  bool prepare_joininfo_chunk(
      uint64_t leaf_start,
      uint64_t leaf_end,
      std::vector<JoinInfo<uint32_t>> &h_data_infos,
      std::vector<JoinInfo<uint64_t>> &h_parent_infos,
      ChunkStagingBuffers &buffers,
      cudaStream_t stream)
  {
    auto &mem_mgr = MemoryManager::getInstance();
    auto &jc = JoinContext::getInstance();

    buffers.parent_buffers.clear();
    buffers.data_buffers.clear();

    std::vector<ColumnRange> ranges;
    compute_column_ranges(jc.frag_trie, jc.up_to_col, leaf_start, leaf_end, ranges, jc.join_stream);

    for (uint32_t col : jc.partial_parent_cols)
    {
      uint32_t col_trie_id = jc.frag_trie->column_ids[col];
      const int block_idx = static_cast<int>(col_trie_id) * 2 + 1;
      auto &block = mem_mgr.memory_blocks[block_idx];
      if (block.host_ptr == nullptr)
      {
        std::cerr << "prepare_joininfo_chunk: missing host parents for trie " << col_trie_id << std::endl;
        cleanup_chunk_staging_buffers(buffers);
        return false;
      }
      auto *h_parents = static_cast<uint64_t *>(block.host_ptr);
      ColumnRange r = ranges[col];
      uint64_t len = r.end - r.start;

      if (len == 0)
      {
        h_parent_infos[col].array_ptr = nullptr;
        h_parent_infos[col].offset = 0;
        h_parent_infos[col].is_partial = 1;
        continue;
      }

      uint64_t *d_buf = nullptr;
      if (!mem_mgr.allocate_temporal(d_buf, sizeof(uint64_t) * len))
      {
        cleanup_chunk_staging_buffers(buffers);
        return false;
      }
      buffers.parent_buffers.push_back(d_buf);
      cuchk(cudaMemcpyAsync(d_buf, h_parents + r.start, sizeof(uint64_t) * len,
                            cudaMemcpyHostToDevice, stream));

      h_parent_infos[col].array_ptr = d_buf;
      h_parent_infos[col].offset = r.start;
      h_parent_infos[col].is_partial = 1;
    }

    for (uint32_t col : jc.partial_data_cols)
    {
      uint32_t col_trie_id = jc.frag_trie->column_ids[col];
      const int block_idx = static_cast<int>(col_trie_id) * 2;
      auto &block = mem_mgr.memory_blocks[block_idx];
      if (block.host_ptr == nullptr)
      {
        std::cerr << "prepare_joininfo_chunk: missing host data for trie " << col_trie_id << std::endl;
        cleanup_chunk_staging_buffers(buffers);
        return false;
      }
      auto *h_data = static_cast<uint32_t *>(block.host_ptr);
      ColumnRange r = ranges[col];
      uint64_t len = r.end - r.start;

      if (len == 0)
      {
        h_data_infos[col].array_ptr = nullptr;
        h_data_infos[col].offset = 0;
        h_data_infos[col].is_partial = 1;
        continue;
      }

      uint32_t *d_buf = nullptr;
      if (!mem_mgr.allocate_temporal(d_buf, sizeof(uint32_t) * len))
      {
        cleanup_chunk_staging_buffers(buffers);
        return false;
      }
      buffers.data_buffers.push_back(d_buf);
      cuchk(cudaMemcpyAsync(d_buf, h_data + r.start, sizeof(uint32_t) * len,
                            cudaMemcpyHostToDevice, stream));

      h_data_infos[col].array_ptr = d_buf;
      h_data_infos[col].offset = r.start;
      h_data_infos[col].is_partial = 1;
    }

    cuchk(cudaMemcpyAsync(
        jc.d_data_info_arrays_, h_data_infos.data(),
        sizeof(JoinInfo<uint32_t>) * h_data_infos.size(),
        cudaMemcpyHostToDevice, stream));
    cuchk(cudaMemcpyAsync(
        jc.d_parent_info_arrays_, h_parent_infos.data(),
        sizeof(JoinInfo<uint64_t>) * h_parent_infos.size(),
        cudaMemcpyHostToDevice, stream));

    return true;
  }

  bool allocate_output_staging_buffers(
      uint64_t &capacity_rows,
      uint32_t *&d_data_buffer,
      uint64_t *&d_parent_buffer)
  {
    auto &mem_mgr = MemoryManager::getInstance();
    d_data_buffer = nullptr;
    d_parent_buffer = nullptr;

    while (capacity_rows > 0)
    {
      uint32_t *data = nullptr;
      uint64_t *parents = nullptr;
      const size_t data_bytes = sizeof(uint32_t) * static_cast<size_t>(capacity_rows);
      const size_t parent_bytes = sizeof(uint64_t) * static_cast<size_t>(capacity_rows);

      if (mem_mgr.allocate_temporal(data, data_bytes) && mem_mgr.allocate_temporal(parents, parent_bytes))
      {
        d_data_buffer = data;
        d_parent_buffer = parents;
        return true;
      }

      if (data != nullptr)
        mem_mgr.deallocate_temporal(data);
      if (parents != nullptr)
        mem_mgr.deallocate_temporal(parents);

      capacity_rows >>= 1;
    }

    return false;
  }

  void direct_join_without_duplication_streaming()
  {
    std::cout << "Entering direct join without duplication streaming pipeline." << std::endl;

    auto &mem_mgr = MemoryManager::getInstance();
    auto &host_mgr = HostMemoryManager::getInstance();
    auto &ht = HashTable::getInstance();
    auto &jc = JoinContext::getInstance();

    get_data_arrays_pipeline(jc.frag_trie->id, jc.pin_cols, jc.h_data_info_arrays_);
    get_parent_arrays_pipeline(jc.frag_trie->id, jc.up_to_col, jc.h_parent_info_arrays_);

    // Base JoinInfo arrays copied to device (will be overwritten per chunk if partial staging is needed).
    cuchk(cudaMemcpyAsync(
        jc.d_data_info_arrays_, jc.h_data_info_arrays_.data(),
        sizeof(JoinInfo<uint32_t>) * jc.h_data_info_arrays_.size(),
        cudaMemcpyHostToDevice, jc.join_stream));
    cuchk(cudaMemcpyAsync(
        jc.d_parent_info_arrays_, jc.h_parent_info_arrays_.data(),
        sizeof(JoinInfo<uint64_t>) * jc.h_parent_info_arrays_.size(),
        cudaMemcpyHostToDevice, jc.join_stream));

    jc.direct_join_without_duplication_context();

    auto align_rows = [](uint64_t rows)
    {
      rows = (rows / 32) * 32;
      return rows ? rows : 32;
    };

    uint64_t chunk_rows = std::min<uint64_t>(jc.num_frag_rows, align_rows(1ull << 20));
    if (chunk_rows == 0)
      chunk_rows = 32;

    uint64_t total_res = 0;
    ChunkStagingBuffers staging_buffers;
    std::vector<JoinInfo<uint32_t>> h_data_infos;
    std::vector<JoinInfo<uint64_t>> h_parent_infos;

    // Pass 1: count total output size
    for (uint64_t cur = 0; cur < jc.num_frag_rows;)
    {
      uint64_t n = std::min<uint64_t>(chunk_rows, jc.num_frag_rows - cur);
      if (n == 0)
        break;
      n = std::min<uint64_t>(n, jc.num_frag_rows - cur);
      if (n == 0)
        break;
      if (n < 32)
        n = jc.num_frag_rows - cur;
      if (n == 0)
        break;

      bool chunk_ok = false;
      while (!chunk_ok)
      {
        n = std::min<uint64_t>(n, jc.num_frag_rows - cur);
        n = align_rows(n);
        if (n > jc.num_frag_rows - cur)
          n = jc.num_frag_rows - cur;
        if (n == 0)
          break;

        h_data_infos = jc.h_data_info_arrays_;
        h_parent_infos = jc.h_parent_info_arrays_;

        if (!prepare_joininfo_chunk(cur, cur + n, h_data_infos, h_parent_infos, staging_buffers, jc.join_stream))
        {
          if (n <= 32)
            break;
          cleanup_chunk_staging_buffers(staging_buffers);
          n >>= 1;
          continue;
        }

        uint32_t *d_counts = nullptr;
        uint32_t *d_st = nullptr;
        uint64_t *d_offsets = nullptr;
        void *d_scan_tmp = nullptr;
        size_t scan_tmp_bytes = 0;

        if (!mem_mgr.allocate_temporal(d_counts, sizeof(uint32_t) * static_cast<size_t>(n + 1)) ||
            !mem_mgr.allocate_temporal(d_st, sizeof(uint32_t) * static_cast<size_t>(n)) ||
            !mem_mgr.allocate_temporal(d_offsets, sizeof(uint64_t) * static_cast<size_t>(n + 1)))
        {
          if (d_counts)
            mem_mgr.deallocate_temporal(d_counts);
          if (d_st)
            mem_mgr.deallocate_temporal(d_st);
          if (d_offsets)
            mem_mgr.deallocate_temporal(d_offsets);
          if (n <= 32)
            break;
          cleanup_chunk_staging_buffers(staging_buffers);
          n >>= 1;
          continue;
        }

        cuchk(cudaMemsetAsync(d_counts, 0, sizeof(uint32_t) * static_cast<size_t>(n + 1), jc.join_stream));
        cuchk(cudaMemsetAsync(d_st, 0xFF, sizeof(uint32_t) * static_cast<size_t>(n), jc.join_stream));
        cuchk(cudaMemsetAsync(d_offsets, 0, sizeof(uint64_t) * static_cast<size_t>(n + 1), jc.join_stream));

        join_count_nodup_pipeline_local_dispatcher(
            jc.d_data_info_arrays_, jc.d_parent_info_arrays_,
            cur, n,
            jc.frag_trie->num_cols, jc.col_key,
            jc.d_edge_keys_, jc.d_edge_values_, jc.d_edge_offsets_,
            ht.h_num_buckets_[jc.directed_edge_idx],
            ht.h_hash_table_offs_[jc.directed_edge_idx],
            jc.directed_edge_idx,
            d_counts, d_st,
            jc.task_per_warp, jc.join_stream);

        cub::DeviceScan::ExclusiveScan(d_scan_tmp, scan_tmp_bytes,
                                       d_counts, d_offsets,
                                       cub::Sum(), static_cast<uint64_t>(0), n + 1, jc.join_stream);
        if (!mem_mgr.allocate_temporal(d_scan_tmp, scan_tmp_bytes))
        {
          mem_mgr.deallocate_temporal(d_counts);
          mem_mgr.deallocate_temporal(d_st);
          mem_mgr.deallocate_temporal(d_offsets);
          if (n <= 32)
            break;
          cleanup_chunk_staging_buffers(staging_buffers);
          n >>= 1;
          continue;
        }
        cub::DeviceScan::ExclusiveScan(d_scan_tmp, scan_tmp_bytes,
                                       d_counts, d_offsets,
                                       cub::Sum(), static_cast<uint64_t>(0), n + 1, jc.join_stream);

        uint64_t chunk_total = 0;
        cuchk(cudaMemcpyAsync(&chunk_total, d_offsets + n, sizeof(uint64_t),
                              cudaMemcpyDeviceToHost, jc.join_stream));
        cuchk(cudaStreamSynchronize(jc.join_stream));
        total_res += chunk_total;

        mem_mgr.deallocate_temporal(d_scan_tmp);
        mem_mgr.deallocate_temporal(d_counts);
        mem_mgr.deallocate_temporal(d_st);
        mem_mgr.deallocate_temporal(d_offsets);
        cleanup_chunk_staging_buffers(staging_buffers);

        chunk_ok = true;
      }

      if (!chunk_ok)
      {
        std::cerr << "direct_join_without_duplication_streaming: failed to count chunk starting at " << cur << std::endl;
        exit(EXIT_FAILURE);
      }

      cur += n;
    }

    jc.h_num_res[0] = total_res;
    resetTargetTrieData(jc.target_trie, jc.h_num_res[0], false);

    if (jc.skip_final_materialization)
    {
      mem_mgr.unpin_climb_data();
      mem_mgr.unpin_climb_parents();
      cuchk(cudaStreamSynchronize(jc.join_stream));
      return;
    }

    bool output_on_device = mem_mgr.allocate_trie(jc.target_trie->id);
	    if (!output_on_device)
	    {
	      mem_mgr.p_type = ProcessType::PIPELINE;

	      uint32_t *h_data_ptr = static_cast<uint32_t *>(host_mgr.allocateManagedBytes(
	          sizeof(uint32_t) * jc.h_num_res[0],
	          sizeof(uint32_t)));
	      if (h_data_ptr == nullptr)
	      {
	        std::cerr << "direct_join_without_duplication_streaming: host allocation failed for data." << std::endl;
	        exit(EXIT_FAILURE);
	      }
	      uint64_t *h_parents_ptr = static_cast<uint64_t *>(host_mgr.allocateManagedBytes(
	          sizeof(uint64_t) * jc.h_num_res[0],
	          sizeof(uint64_t)));
	      if (h_parents_ptr == nullptr)
	      {
	        std::cerr << "direct_join_without_duplication_streaming: host allocation failed for parents." << std::endl;
	        exit(EXIT_FAILURE);
	      }

      auto &data_mem_block = mem_mgr.memory_blocks[jc.target_trie->id * 2];
      data_mem_block.host_ptr = h_data_ptr;
      data_mem_block.device_ptr = nullptr;
      data_mem_block.size_in_bytes = sizeof(uint32_t) * jc.h_num_res[0];
      auto &parent_mem_block = mem_mgr.memory_blocks[jc.target_trie->id * 2 + 1];
      parent_mem_block.host_ptr = h_parents_ptr;
      parent_mem_block.device_ptr = nullptr;
      parent_mem_block.size_in_bytes = sizeof(uint64_t) * jc.h_num_res[0];

      // Allocate reusable device staging buffers for output.
      size_t free_mem = mem_mgr.stats.available_gpu_memory / 4;
      uint64_t staging_capacity = static_cast<uint64_t>(free_mem / (sizeof(uint32_t) + sizeof(uint64_t)));
      staging_capacity = std::max<uint64_t>(staging_capacity, 1);

	      uint32_t *d_out_data = nullptr;
	      uint64_t *d_out_parents = nullptr;
	      if (!allocate_output_staging_buffers(staging_capacity, d_out_data, d_out_parents))
	      {
	        std::cerr << "direct_join_without_duplication_streaming: failed to allocate output staging buffers." << std::endl;
	        exit(EXIT_FAILURE);
	      }

	      uint64_t *h_offsets = static_cast<uint64_t *>(host_mgr.allocateManagedBytes(
	          sizeof(uint64_t) * static_cast<size_t>(chunk_rows + 1),
	          sizeof(uint64_t)));
	      if (h_offsets == nullptr)
	      {
	        std::cerr << "direct_join_without_duplication_streaming: host allocation failed for offsets." << std::endl;
	        exit(EXIT_FAILURE);
	      }

      uint64_t global_out_offset = 0;

      for (uint64_t cur = 0; cur < jc.num_frag_rows;)
      {
        uint64_t n = std::min<uint64_t>(chunk_rows, jc.num_frag_rows - cur);
        if (n == 0)
          break;
        if (n < 32)
          n = jc.num_frag_rows - cur;
        n = std::min<uint64_t>(n, jc.num_frag_rows - cur);
        if (n == 0)
          break;

        h_data_infos = jc.h_data_info_arrays_;
        h_parent_infos = jc.h_parent_info_arrays_;
        if (!prepare_joininfo_chunk(cur, cur + n, h_data_infos, h_parent_infos, staging_buffers, jc.join_stream))
        {
          std::cerr << "direct_join_without_duplication_streaming: failed to prepare chunk at " << cur << std::endl;
          exit(EXIT_FAILURE);
        }

        uint32_t *d_counts = nullptr;
        uint32_t *d_st = nullptr;
        uint64_t *d_offsets = nullptr;
        void *d_scan_tmp = nullptr;
        size_t scan_tmp_bytes = 0;

        if (!mem_mgr.allocate_temporal(d_counts, sizeof(uint32_t) * static_cast<size_t>(n + 1)) ||
            !mem_mgr.allocate_temporal(d_st, sizeof(uint32_t) * static_cast<size_t>(n)) ||
            !mem_mgr.allocate_temporal(d_offsets, sizeof(uint64_t) * static_cast<size_t>(n + 1)))
        {
          std::cerr << "direct_join_without_duplication_streaming: failed to allocate chunk temporals." << std::endl;
          exit(EXIT_FAILURE);
        }

        cuchk(cudaMemsetAsync(d_counts, 0, sizeof(uint32_t) * static_cast<size_t>(n + 1), jc.join_stream));
        cuchk(cudaMemsetAsync(d_st, 0xFF, sizeof(uint32_t) * static_cast<size_t>(n), jc.join_stream));

        join_count_nodup_pipeline_local_dispatcher(
            jc.d_data_info_arrays_, jc.d_parent_info_arrays_,
            cur, n,
            jc.frag_trie->num_cols, jc.col_key,
            jc.d_edge_keys_, jc.d_edge_values_, jc.d_edge_offsets_,
            ht.h_num_buckets_[jc.directed_edge_idx],
            ht.h_hash_table_offs_[jc.directed_edge_idx],
            jc.directed_edge_idx,
            d_counts, d_st,
            jc.task_per_warp, jc.join_stream);

        cub::DeviceScan::ExclusiveScan(d_scan_tmp, scan_tmp_bytes,
                                       d_counts, d_offsets,
                                       cub::Sum(), static_cast<uint64_t>(0), n + 1, jc.join_stream);
        if (!mem_mgr.allocate_temporal(d_scan_tmp, scan_tmp_bytes))
        {
          std::cerr << "direct_join_without_duplication_streaming: failed to allocate scan temp storage." << std::endl;
          exit(EXIT_FAILURE);
        }
        cub::DeviceScan::ExclusiveScan(d_scan_tmp, scan_tmp_bytes,
                                       d_counts, d_offsets,
                                       cub::Sum(), static_cast<uint64_t>(0), n + 1, jc.join_stream);

        cuchk(cudaMemcpyAsync(h_offsets, d_offsets, sizeof(uint64_t) * static_cast<size_t>(n + 1),
                              cudaMemcpyDeviceToHost, jc.join_stream));
        cuchk(cudaStreamSynchronize(jc.join_stream));

        uint64_t chunk_total = h_offsets[n];
        uint64_t row_start = 0;
        while (row_start < n)
        {
          uint64_t row_end = find_frag_range(h_offsets, row_start, n, staging_capacity);
          if (row_end == row_start)
            row_end = row_start + 1;
          uint64_t num_rows_sub = row_end - row_start;
          uint64_t num_results_sub = h_offsets[row_end] - h_offsets[row_start];
          if (num_results_sub == 0)
          {
            row_start = row_end;
            continue;
          }

          uint32_t *d_write_data = d_out_data;
          uint64_t *d_write_parents = d_out_parents;
          uint32_t *d_big_data = nullptr;
          uint64_t *d_big_parents = nullptr;
          if (num_results_sub > staging_capacity)
          {
            if (!mem_mgr.allocate_temporal(d_big_data, sizeof(uint32_t) * static_cast<size_t>(num_results_sub)) ||
                !mem_mgr.allocate_temporal(d_big_parents, sizeof(uint64_t) * static_cast<size_t>(num_results_sub)))
            {
              std::cerr << "direct_join_without_duplication_streaming: failed to allocate oversized staging buffers (" << num_results_sub << ")." << std::endl;
              exit(EXIT_FAILURE);
            }
            d_write_data = d_big_data;
            d_write_parents = d_big_parents;
          }

          join_write_kernel_nodup_pipeline_local<<<GRID_DIM, BLOCK_DIM, 0, jc.join_stream>>>(
              cur + row_start, num_rows_sub,
              jc.d_edge_values_,
              d_st + row_start, d_offsets + row_start, h_offsets[row_start],
              d_write_data, d_write_parents);

          cuchk(cudaMemcpyAsync(h_data_ptr + global_out_offset + h_offsets[row_start], d_write_data,
                                sizeof(uint32_t) * static_cast<size_t>(num_results_sub),
                                cudaMemcpyDeviceToHost, jc.join_stream));
          cuchk(cudaMemcpyAsync(h_parents_ptr + global_out_offset + h_offsets[row_start], d_write_parents,
                                sizeof(uint64_t) * static_cast<size_t>(num_results_sub),
                                cudaMemcpyDeviceToHost, jc.join_stream));
          if (d_big_data)
            mem_mgr.deallocate_temporal(d_big_data);
          if (d_big_parents)
            mem_mgr.deallocate_temporal(d_big_parents);

          row_start = row_end;
        }

        cuchk(cudaStreamSynchronize(jc.join_stream));
        global_out_offset += chunk_total;
        cur += n;

        mem_mgr.deallocate_temporal(d_scan_tmp);
        mem_mgr.deallocate_temporal(d_counts);
        mem_mgr.deallocate_temporal(d_st);
        mem_mgr.deallocate_temporal(d_offsets);
        cleanup_chunk_staging_buffers(staging_buffers);
      }

	      {
	        void *h_offsets_void = h_offsets;
	        host_mgr.deallocateManaged(h_offsets_void);
	        h_offsets = nullptr;
	      }
	      mem_mgr.deallocate_temporal(d_out_data);
	      mem_mgr.deallocate_temporal(d_out_parents);
	    }
    else
    {
      mem_mgr.p_type = ProcessType::DIRECT;

      uint64_t global_out_offset = 0;
      for (uint64_t cur = 0; cur < jc.num_frag_rows;)
      {
        uint64_t n = std::min<uint64_t>(chunk_rows, jc.num_frag_rows - cur);
        if (n == 0)
          break;
        if (n < 32)
          n = jc.num_frag_rows - cur;
        n = std::min<uint64_t>(n, jc.num_frag_rows - cur);

        h_data_infos = jc.h_data_info_arrays_;
        h_parent_infos = jc.h_parent_info_arrays_;
        if (!prepare_joininfo_chunk(cur, cur + n, h_data_infos, h_parent_infos, staging_buffers, jc.join_stream))
        {
          std::cerr << "direct_join_without_duplication_streaming: failed to prepare chunk at " << cur << std::endl;
          exit(EXIT_FAILURE);
        }

        uint32_t *d_counts = nullptr;
        uint32_t *d_st = nullptr;
        uint64_t *d_offsets = nullptr;
        void *d_scan_tmp = nullptr;
        size_t scan_tmp_bytes = 0;

        if (!mem_mgr.allocate_temporal(d_counts, sizeof(uint32_t) * static_cast<size_t>(n + 1)) ||
            !mem_mgr.allocate_temporal(d_st, sizeof(uint32_t) * static_cast<size_t>(n)) ||
            !mem_mgr.allocate_temporal(d_offsets, sizeof(uint64_t) * static_cast<size_t>(n + 1)))
        {
          std::cerr << "direct_join_without_duplication_streaming: failed to allocate chunk temporals." << std::endl;
          exit(EXIT_FAILURE);
        }

        cuchk(cudaMemsetAsync(d_counts, 0, sizeof(uint32_t) * static_cast<size_t>(n + 1), jc.join_stream));
        cuchk(cudaMemsetAsync(d_st, 0xFF, sizeof(uint32_t) * static_cast<size_t>(n), jc.join_stream));

        join_count_nodup_pipeline_local_dispatcher(
            jc.d_data_info_arrays_, jc.d_parent_info_arrays_,
            cur, n,
            jc.frag_trie->num_cols, jc.col_key,
            jc.d_edge_keys_, jc.d_edge_values_, jc.d_edge_offsets_,
            ht.h_num_buckets_[jc.directed_edge_idx],
            ht.h_hash_table_offs_[jc.directed_edge_idx],
            jc.directed_edge_idx,
            d_counts, d_st,
            jc.task_per_warp, jc.join_stream);

        cub::DeviceScan::ExclusiveScan(d_scan_tmp, scan_tmp_bytes,
                                       d_counts, d_offsets,
                                       cub::Sum(), static_cast<uint64_t>(0), n + 1, jc.join_stream);
        if (!mem_mgr.allocate_temporal(d_scan_tmp, scan_tmp_bytes))
        {
          std::cerr << "direct_join_without_duplication_streaming: failed to allocate scan temp storage." << std::endl;
          exit(EXIT_FAILURE);
        }
        cub::DeviceScan::ExclusiveScan(d_scan_tmp, scan_tmp_bytes,
                                       d_counts, d_offsets,
                                       cub::Sum(), static_cast<uint64_t>(0), n + 1, jc.join_stream);

        uint64_t chunk_total = 0;
        cuchk(cudaMemcpyAsync(&chunk_total, d_offsets + n, sizeof(uint64_t),
                              cudaMemcpyDeviceToHost, jc.join_stream));
        cuchk(cudaStreamSynchronize(jc.join_stream));

        join_write_kernel_nodup_pipeline_local<<<GRID_DIM, BLOCK_DIM, 0, jc.join_stream>>>(
            cur, n,
            jc.d_edge_values_,
            d_st, d_offsets, 0,
            jc.target_trie->data->d_data + global_out_offset,
            jc.target_trie->data->d_parents + global_out_offset);
        cuchk(cudaStreamSynchronize(jc.join_stream));

        global_out_offset += chunk_total;
        cur += n;

        mem_mgr.deallocate_temporal(d_scan_tmp);
        mem_mgr.deallocate_temporal(d_counts);
        mem_mgr.deallocate_temporal(d_st);
        mem_mgr.deallocate_temporal(d_offsets);
        cleanup_chunk_staging_buffers(staging_buffers);
      }
    }

    mem_mgr.unpin_climb_data();
    mem_mgr.unpin_climb_parents();
    cuchk(cudaStreamSynchronize(jc.join_stream));
  }
} // namespace

void direct_join_without_duplication_pipeline()
{
  std::cout << "Entering direct join without duplication pipeline." << std::endl;

  auto &mem_mgr = MemoryManager::getInstance();
  auto &ht = HashTable::getInstance();
  auto &jc = JoinContext::getInstance();
  auto &host_mgr = HostMemoryManager::getInstance();

  get_data_arrays_pipeline(jc.frag_trie->id, jc.pin_cols, jc.h_data_info_arrays_);
  get_parent_arrays_pipeline(jc.frag_trie->id, jc.up_to_col, jc.h_parent_info_arrays_);

  cuchk(cudaMemcpyAsync(
      jc.d_data_info_arrays_, jc.h_data_info_arrays_.data(),
      sizeof(JoinInfo<uint32_t>) * jc.h_data_info_arrays_.size(),
      cudaMemcpyHostToDevice, jc.join_stream));
  cuchk(cudaMemcpyAsync(
      jc.d_parent_info_arrays_, jc.h_parent_info_arrays_.data(),
      sizeof(JoinInfo<uint64_t>) * jc.h_parent_info_arrays_.size(),
      cudaMemcpyHostToDevice, jc.join_stream));

  if (!mem_mgr.allocate_temporal(jc.d_num_res_for_each, sizeof(uint32_t) * (jc.num_frag_rows + 1)) ||
      !mem_mgr.allocate_temporal(jc.d_st_array, sizeof(uint32_t) * jc.num_frag_rows))
  {
    mem_mgr.unpin_climb_data();
    mem_mgr.unpin_climb_parents();
    if (jc.d_num_res_for_each)
      mem_mgr.deallocate_temporal(jc.d_num_res_for_each);
    if (jc.d_st_array)
      mem_mgr.deallocate_temporal(jc.d_st_array);
    direct_join_without_duplication_streaming();
    return;
  }
  cuchk(cudaMemsetAsync(jc.d_num_res_for_each, 0, sizeof(uint32_t) * (jc.num_frag_rows + 1), jc.join_stream));
  cuchk(cudaMemsetAsync(jc.d_st_array, 0xFFFFFFFF, sizeof(uint32_t) * jc.num_frag_rows, jc.join_stream));

  jc.direct_join_without_duplication_context();

  const int nStreams = JoinContext::nStreams;
  const size_t num_partial_parents = jc.partial_parent_cols.size();
  const size_t num_partial_data = jc.partial_data_cols.size();

  uint64_t rows_per_stream = jc.num_frag_rows;
  if (num_partial_parents + num_partial_data > 0)
  {
    size_t bytes_per_row = num_partial_parents * sizeof(uint64_t) + num_partial_data * sizeof(uint32_t);
    bytes_per_row = std::max<size_t>(bytes_per_row, 1);
    size_t budget_bytes = mem_mgr.stats.available_gpu_memory / 2 / nStreams;
    rows_per_stream = std::max<uint64_t>(budget_bytes / bytes_per_row, 32);
    rows_per_stream = std::min<uint64_t>(rows_per_stream, jc.num_frag_rows);
    rows_per_stream = (rows_per_stream / 32) * 32;
  }
  if (rows_per_stream == 0)
    rows_per_stream = 32;

  uint64_t *d_partial_parent_base = nullptr;
  if (num_partial_parents > 0)
  {
    mem_mgr.allocate_temporal(d_partial_parent_base,
                              sizeof(uint64_t) * rows_per_stream * num_partial_parents * nStreams);
  }

  uint32_t *d_partial_data_base = nullptr;
  if (num_partial_data > 0)
  {
    mem_mgr.allocate_temporal(d_partial_data_base,
                              sizeof(uint32_t) * rows_per_stream * num_partial_data * nStreams);
  }

  void *d_parent_info_arrays_base_void = nullptr;
  JoinInfo<uint64_t> *d_parent_info_arrays_base = nullptr;
  JoinInfo<uint64_t> *h_parent_info_arrays[nStreams];
  for (int i = 0; i < nStreams; ++i)
    h_parent_info_arrays[i] = nullptr;

  void *d_data_info_arrays_base_void = nullptr;
  JoinInfo<uint32_t> *d_data_info_arrays_base = nullptr;
  JoinInfo<uint32_t> *h_data_info_arrays[nStreams];
  for (int i = 0; i < nStreams; ++i)
    h_data_info_arrays[i] = nullptr;

  const size_t parent_info_count = jc.h_parent_info_arrays_.size();
  const size_t data_info_count = jc.h_data_info_arrays_.size();
  const bool need_parent_updates = (num_partial_parents > 0);
  const bool need_data_updates = (num_partial_data > 0);

  if (need_parent_updates)
  {
    mem_mgr.allocate_temporal(d_parent_info_arrays_base_void,
                              sizeof(JoinInfo<uint64_t>) * parent_info_count * nStreams);
    d_parent_info_arrays_base = static_cast<JoinInfo<uint64_t> *>(d_parent_info_arrays_base_void);
    for (int i = 0; i < nStreams; ++i)
    {
      h_parent_info_arrays[i] = static_cast<JoinInfo<uint64_t> *>(host_mgr.allocateManagedBytes(
          sizeof(JoinInfo<uint64_t>) * parent_info_count,
          alignof(JoinInfo<uint64_t>)));
      if (h_parent_info_arrays[i] == nullptr)
      {
        std::cerr << "direct_join_without_duplication_pipeline: host allocation failed for parent JoinInfo buffers." << std::endl;
        exit(EXIT_FAILURE);
      }
    }
  }

  if (need_data_updates)
  {
    mem_mgr.allocate_temporal(d_data_info_arrays_base_void,
                              sizeof(JoinInfo<uint32_t>) * data_info_count * nStreams);
    d_data_info_arrays_base = static_cast<JoinInfo<uint32_t> *>(d_data_info_arrays_base_void);
    for (int i = 0; i < nStreams; ++i)
    {
      h_data_info_arrays[i] = static_cast<JoinInfo<uint32_t> *>(host_mgr.allocateManagedBytes(
          sizeof(JoinInfo<uint32_t>) * data_info_count,
          alignof(JoinInfo<uint32_t>)));
      if (h_data_info_arrays[i] == nullptr)
      {
        std::cerr << "direct_join_without_duplication_pipeline: host allocation failed for data JoinInfo buffers." << std::endl;
        exit(EXIT_FAILURE);
      }
    }
  }

  cudaEvent_t init_event;
  cuchk(cudaEventCreateWithFlags(&init_event, cudaEventDisableTiming));
  cuchk(cudaEventRecord(init_event, jc.join_stream));
  for (int i = 0; i < nStreams; ++i)
    cuchk(cudaStreamWaitEvent(jc.pipeline_streams[i], init_event, 0));
  cuchk(cudaEventDestroy(init_event));

  cudaEvent_t done_events[nStreams];
  for (int i = 0; i < nStreams; ++i)
  {
    cuchk(cudaEventCreateWithFlags(&done_events[i], cudaEventDisableTiming));
    cuchk(cudaEventRecord(done_events[i], jc.pipeline_streams[i]));
  }

  std::vector<ColumnRange> ranges;
  uint64_t cur_offset = 0;
  while (cur_offset < jc.num_frag_rows)
  {
    for (int stream_id = 0; stream_id < nStreams && cur_offset < jc.num_frag_rows; ++stream_id)
    {
      cuchk(cudaEventSynchronize(done_events[stream_id]));

      uint64_t num_rows_this_iter = std::min(rows_per_stream, jc.num_frag_rows - cur_offset);
      cudaStream_t stream = jc.pipeline_streams[stream_id];

      compute_column_ranges(jc.frag_trie, jc.up_to_col, cur_offset, cur_offset + num_rows_this_iter, ranges, jc.join_stream);

      JoinInfo<uint64_t> *d_parent_info_ptr = jc.d_parent_info_arrays_;
      if (need_parent_updates)
      {
        JoinInfo<uint64_t> *h_parent_infos = h_parent_info_arrays[stream_id];
        for (size_t col = 0; col < parent_info_count; ++col)
          h_parent_infos[col] = jc.h_parent_info_arrays_[col];

        for (size_t i = 0; i < num_partial_parents; ++i)
        {
          uint32_t col = jc.partial_parent_cols[i];
          uint32_t col_trie_id = jc.frag_trie->column_ids[col];
          const int block_idx = static_cast<int>(col_trie_id) * 2 + 1;
          auto &block = mem_mgr.memory_blocks[block_idx];
          if (block.host_ptr == nullptr)
          {
            std::cerr << "direct_join_without_duplication_pipeline: missing host parents for trie " << col_trie_id << std::endl;
            exit(EXIT_FAILURE);
          }
          auto *h_parents = static_cast<uint64_t *>(block.host_ptr);
          ColumnRange r = ranges[col];
          uint64_t len = r.end - r.start;

          uint64_t *d_parent_buf = d_partial_parent_base +
                                  (static_cast<size_t>(stream_id) * num_partial_parents + i) * rows_per_stream;
          if (len > 0)
          {
            cuchk(cudaMemcpyAsync(d_parent_buf, h_parents + r.start, sizeof(uint64_t) * len,
                                  cudaMemcpyHostToDevice, stream));
            h_parent_infos[col].array_ptr = d_parent_buf;
            h_parent_infos[col].offset = r.start;
            h_parent_infos[col].is_partial = 1;
          }
          else
          {
            h_parent_infos[col].array_ptr = nullptr;
            h_parent_infos[col].offset = 0;
            h_parent_infos[col].is_partial = 1;
          }
        }

        d_parent_info_ptr = d_parent_info_arrays_base + parent_info_count * static_cast<size_t>(stream_id);
        cuchk(cudaMemcpyAsync(
            d_parent_info_ptr, h_parent_infos,
            sizeof(JoinInfo<uint64_t>) * parent_info_count,
            cudaMemcpyHostToDevice, stream));
      }

      JoinInfo<uint32_t> *d_data_info_ptr = jc.d_data_info_arrays_;
      if (need_data_updates)
      {
        JoinInfo<uint32_t> *h_data_infos = h_data_info_arrays[stream_id];
        for (size_t col = 0; col < data_info_count; ++col)
          h_data_infos[col] = jc.h_data_info_arrays_[col];

        for (size_t i = 0; i < num_partial_data; ++i)
        {
          uint32_t col = jc.partial_data_cols[i];
          uint32_t col_trie_id = jc.frag_trie->column_ids[col];
          const int block_idx = static_cast<int>(col_trie_id) * 2;
          auto &block = mem_mgr.memory_blocks[block_idx];
          if (block.host_ptr == nullptr)
          {
            std::cerr << "direct_join_without_duplication_pipeline: missing host data for trie " << col_trie_id << std::endl;
            exit(EXIT_FAILURE);
          }
          auto *h_data = static_cast<uint32_t *>(block.host_ptr);
          ColumnRange r = ranges[col];
          uint64_t len = r.end - r.start;

          uint32_t *d_data_buf = d_partial_data_base +
                                 (static_cast<size_t>(stream_id) * num_partial_data + i) * rows_per_stream;
          if (len > 0)
          {
            cuchk(cudaMemcpyAsync(d_data_buf, h_data + r.start, sizeof(uint32_t) * len,
                                  cudaMemcpyHostToDevice, stream));
            h_data_infos[col].array_ptr = d_data_buf;
            h_data_infos[col].offset = r.start;
            h_data_infos[col].is_partial = 1;
          }
          else
          {
            h_data_infos[col].array_ptr = nullptr;
            h_data_infos[col].offset = 0;
            h_data_infos[col].is_partial = 1;
          }
        }

        d_data_info_ptr = d_data_info_arrays_base + data_info_count * static_cast<size_t>(stream_id);
        cuchk(cudaMemcpyAsync(
            d_data_info_ptr, h_data_infos,
            sizeof(JoinInfo<uint32_t>) * data_info_count,
            cudaMemcpyHostToDevice, stream));
      }

      join_count_nodup_pipeline_dispatcher(
          d_data_info_ptr, d_parent_info_ptr,
          cur_offset, num_rows_this_iter,
          jc.frag_trie->num_cols, jc.col_key,
          jc.d_edge_keys_, jc.d_edge_values_, jc.d_edge_offsets_,
          ht.h_num_buckets_[jc.directed_edge_idx],
          ht.h_hash_table_offs_[jc.directed_edge_idx],
          jc.directed_edge_idx,
          jc.d_num_res_for_each,
          jc.d_st_array,
          jc.task_per_warp,
          stream);

      cuchk(cudaEventRecord(done_events[stream_id], stream));
      cur_offset += num_rows_this_iter;
    }
  }

  for (int i = 0; i < nStreams; ++i)
  {
    cuchk(cudaEventSynchronize(done_events[i]));
    cuchk(cudaEventDestroy(done_events[i]));
  }

  if (need_parent_updates)
  {
    for (int i = 0; i < nStreams; ++i)
    {
      void *tmp = h_parent_info_arrays[i];
      host_mgr.deallocateManaged(tmp);
      h_parent_info_arrays[i] = nullptr;
    }
    mem_mgr.deallocate_temporal(d_parent_info_arrays_base_void);
  }
  if (need_data_updates)
  {
    for (int i = 0; i < nStreams; ++i)
    {
      void *tmp = h_data_info_arrays[i];
      host_mgr.deallocateManaged(tmp);
      h_data_info_arrays[i] = nullptr;
    }
    mem_mgr.deallocate_temporal(d_data_info_arrays_base_void);
  }
  if (d_partial_parent_base != nullptr)
    mem_mgr.deallocate_temporal(d_partial_parent_base);
  if (d_partial_data_base != nullptr)
    mem_mgr.deallocate_temporal(d_partial_data_base);

  mem_mgr.unpin_climb_data();
  mem_mgr.unpin_climb_parents();

  uint64_t *d_row_offsets = nullptr;
  if (!mem_mgr.allocate_temporal(d_row_offsets, sizeof(uint64_t) * (jc.num_frag_rows + 1)))
  {
    std::cout << "direct_join_without_duplication_pipeline: falling back to streaming join." << std::endl;
    mem_mgr.deallocate_temporal(jc.d_num_res_for_each);
    mem_mgr.deallocate_temporal(jc.d_st_array);
    direct_join_without_duplication_streaming();
    return;
  }

  size_t temp_storage_bytes = 0;
  void *d_temp_storage = nullptr;

  cub::DeviceScan::ExclusiveScan(d_temp_storage, temp_storage_bytes,
                                 jc.d_num_res_for_each, d_row_offsets,
                                 cub::Sum(), static_cast<uint64_t>(0), jc.num_frag_rows + 1, jc.join_stream);
  if (!mem_mgr.allocate_temporal(d_temp_storage, temp_storage_bytes))
  {
    std::cout << "direct_join_without_duplication_pipeline: falling back to streaming join." << std::endl;
    mem_mgr.deallocate_temporal(d_row_offsets);
    mem_mgr.deallocate_temporal(jc.d_num_res_for_each);
    mem_mgr.deallocate_temporal(jc.d_st_array);
    direct_join_without_duplication_streaming();
    return;
  }
  cub::DeviceScan::ExclusiveScan(d_temp_storage, temp_storage_bytes,
                                 jc.d_num_res_for_each, d_row_offsets,
                                 cub::Sum(), static_cast<uint64_t>(0), jc.num_frag_rows + 1, jc.join_stream);

  uint64_t h_num_res = 0;
  cuchk(cudaMemcpyAsync(&h_num_res, d_row_offsets + jc.num_frag_rows,
                        sizeof(uint64_t),
                        cudaMemcpyDeviceToHost, jc.join_stream));
  mem_mgr.deallocate_temporal(d_temp_storage);

  cuchk(cudaStreamSynchronize(jc.join_stream));
  jc.h_num_res[0] = h_num_res;
  resetTargetTrieData(jc.target_trie, jc.h_num_res[0], false);

  if (jc.skip_final_materialization == false)
  {
    bool allocate_success = mem_mgr.allocate_trie(jc.target_trie->id);

    if (allocate_success)
    {
      join_write_nodup_dispatcher(
          jc.num_frag_rows, jc.d_edge_values_,
          jc.d_num_res_for_each, jc.d_st_array, jc.target_trie->data->d_data,
          d_row_offsets, jc.task_per_warp, jc.target_trie->data->d_parents);
    }
    else
    {
      direct_join_without_duplication_pipeline_write(d_row_offsets);
    }
  }

  mem_mgr.deallocate_temporal(jc.d_num_res_for_each);
  mem_mgr.deallocate_temporal(d_row_offsets);
  cuchk(cudaStreamSynchronize(jc.join_stream));
}

namespace
{
  void direct_join_with_duplication_streaming_nomask()
  {
    std::cout << "Entering direct join with duplication nomask streaming pipeline." << std::endl;

    auto &mem_mgr = MemoryManager::getInstance();
    auto &host_mgr = HostMemoryManager::getInstance();
    auto &ht = HashTable::getInstance();
    auto &jc = JoinContext::getInstance();

    get_data_arrays_pipeline(jc.frag_trie->id, jc.pin_cols, jc.h_data_info_arrays_);
    get_parent_arrays_pipeline(jc.frag_trie->id, jc.up_to_col, jc.h_parent_info_arrays_);

    cuchk(cudaMemcpyAsync(
        jc.d_data_info_arrays_, jc.h_data_info_arrays_.data(),
        sizeof(JoinInfo<uint32_t>) * jc.h_data_info_arrays_.size(),
        cudaMemcpyHostToDevice, jc.join_stream));
    cuchk(cudaMemcpyAsync(
        jc.d_parent_info_arrays_, jc.h_parent_info_arrays_.data(),
        sizeof(JoinInfo<uint64_t>) * jc.h_parent_info_arrays_.size(),
        cudaMemcpyHostToDevice, jc.join_stream));

    auto align_rows = [](uint64_t rows)
    {
      rows = (rows / 32) * 32;
      return rows ? rows : 32;
    };

    uint64_t chunk_rows = std::min<uint64_t>(jc.num_frag_rows, align_rows(1ull << 20));
    if (chunk_rows == 0)
      chunk_rows = 32;

    uint64_t total_res = 0;
    ChunkStagingBuffers staging_buffers;
    std::vector<JoinInfo<uint32_t>> h_data_infos;
    std::vector<JoinInfo<uint64_t>> h_parent_infos;

    // Pass 1: count total output size
    for (uint64_t cur = 0; cur < jc.num_frag_rows;)
    {
      uint64_t n = std::min<uint64_t>(chunk_rows, jc.num_frag_rows - cur);
      if (n == 0)
        break;

      bool chunk_ok = false;
      while (!chunk_ok)
      {
        n = std::min<uint64_t>(n, jc.num_frag_rows - cur);
        n = align_rows(n);
        if (n > jc.num_frag_rows - cur)
          n = jc.num_frag_rows - cur;
        if (n == 0)
          break;

        h_data_infos = jc.h_data_info_arrays_;
        h_parent_infos = jc.h_parent_info_arrays_;
        if (!prepare_joininfo_chunk(cur, cur + n, h_data_infos, h_parent_infos, staging_buffers, jc.join_stream))
        {
          if (n <= 32)
            break;
          n >>= 1;
          continue;
        }

        uint32_t *d_counts = nullptr;
        uint64_t *d_offsets = nullptr;
        void *d_scan_tmp = nullptr;
        size_t scan_tmp_bytes = 0;

        if (!mem_mgr.allocate_temporal(d_counts, sizeof(uint32_t) * static_cast<size_t>(n + 1)) ||
            !mem_mgr.allocate_temporal(d_offsets, sizeof(uint64_t) * static_cast<size_t>(n + 1)))
        {
          if (d_counts)
            mem_mgr.deallocate_temporal(d_counts);
          if (d_offsets)
            mem_mgr.deallocate_temporal(d_offsets);
          if (n <= 32)
            break;
          cleanup_chunk_staging_buffers(staging_buffers);
          n >>= 1;
          continue;
        }

        cuchk(cudaMemsetAsync(d_counts, 0, sizeof(uint32_t) * static_cast<size_t>(n + 1), jc.join_stream));

        join_count_kernel_dup_nomask_pipeline_local_info<<<GRID_DIM, BLOCK_DIM, 0, jc.join_stream>>>(
            jc.d_data_info_arrays_, jc.d_parent_info_arrays_,
            cur, n,
            jc.frag_trie->num_cols, jc.col_key,
            jc.d_edge_values_, jc.d_edge_offsets_,
            ht.h_num_buckets_[jc.directed_edge_idx],
            ht.h_hash_table_offs_[jc.directed_edge_idx],
            jc.directed_edge_idx,
            d_counts,
            jc.dup_check_type, jc.num_dup_cols, jc.d_dup_cols, jc.up_to_col);

        cub::DeviceScan::ExclusiveScan(d_scan_tmp, scan_tmp_bytes,
                                       d_counts, d_offsets,
                                       cub::Sum(), static_cast<uint64_t>(0), n + 1, jc.join_stream);
        if (!mem_mgr.allocate_temporal(d_scan_tmp, scan_tmp_bytes))
        {
          mem_mgr.deallocate_temporal(d_counts);
          mem_mgr.deallocate_temporal(d_offsets);
          if (n <= 32)
            break;
          cleanup_chunk_staging_buffers(staging_buffers);
          n >>= 1;
          continue;
        }
        cub::DeviceScan::ExclusiveScan(d_scan_tmp, scan_tmp_bytes,
                                       d_counts, d_offsets,
                                       cub::Sum(), static_cast<uint64_t>(0), n + 1, jc.join_stream);

        uint64_t chunk_total = 0;
        cuchk(cudaMemcpyAsync(&chunk_total, d_offsets + n, sizeof(uint64_t),
                              cudaMemcpyDeviceToHost, jc.join_stream));
        cuchk(cudaStreamSynchronize(jc.join_stream));
        total_res += chunk_total;

        mem_mgr.deallocate_temporal(d_scan_tmp);
        mem_mgr.deallocate_temporal(d_counts);
        mem_mgr.deallocate_temporal(d_offsets);
        cleanup_chunk_staging_buffers(staging_buffers);

        chunk_ok = true;
      }

      if (!chunk_ok)
      {
        std::cerr << "direct_join_with_duplication_streaming_nomask: failed to count chunk starting at " << cur << std::endl;
        exit(EXIT_FAILURE);
      }
      cur += n;
    }

    jc.h_num_res[0] = total_res;
    resetTargetTrieData(jc.target_trie, jc.h_num_res[0], false);

    if (jc.skip_final_materialization)
    {
      mem_mgr.unpin_climb_data();
      mem_mgr.unpin_climb_parents();
      cuchk(cudaStreamSynchronize(jc.join_stream));
      return;
    }

    bool output_on_device = mem_mgr.allocate_trie(jc.target_trie->id);
	    if (!output_on_device)
	    {
	      mem_mgr.p_type = ProcessType::PIPELINE;

	      uint32_t *h_data_ptr = static_cast<uint32_t *>(host_mgr.allocateManagedBytes(
	          sizeof(uint32_t) * jc.h_num_res[0],
	          sizeof(uint32_t)));
	      if (h_data_ptr == nullptr)
	      {
	        std::cerr << "direct_join_with_duplication_streaming_nomask: host allocation failed for data." << std::endl;
	        exit(EXIT_FAILURE);
	      }
	      uint64_t *h_parents_ptr = static_cast<uint64_t *>(host_mgr.allocateManagedBytes(
	          sizeof(uint64_t) * jc.h_num_res[0],
	          sizeof(uint64_t)));
	      if (h_parents_ptr == nullptr)
	      {
	        std::cerr << "direct_join_with_duplication_streaming_nomask: host allocation failed for parents." << std::endl;
	        exit(EXIT_FAILURE);
	      }

      auto &data_mem_block = mem_mgr.memory_blocks[jc.target_trie->id * 2];
      data_mem_block.host_ptr = h_data_ptr;
      data_mem_block.device_ptr = nullptr;
      data_mem_block.size_in_bytes = sizeof(uint32_t) * jc.h_num_res[0];
      auto &parent_mem_block = mem_mgr.memory_blocks[jc.target_trie->id * 2 + 1];
      parent_mem_block.host_ptr = h_parents_ptr;
      parent_mem_block.device_ptr = nullptr;
      parent_mem_block.size_in_bytes = sizeof(uint64_t) * jc.h_num_res[0];

      size_t free_mem = mem_mgr.stats.available_gpu_memory / 4;
      uint64_t staging_capacity = static_cast<uint64_t>(free_mem / (sizeof(uint32_t) + sizeof(uint64_t)));
      staging_capacity = std::max<uint64_t>(staging_capacity, 1);

	      uint32_t *d_out_data = nullptr;
	      uint64_t *d_out_parents = nullptr;
	      if (!allocate_output_staging_buffers(staging_capacity, d_out_data, d_out_parents))
	      {
	        std::cerr << "direct_join_with_duplication_streaming_nomask: failed to allocate output staging buffers." << std::endl;
	        exit(EXIT_FAILURE);
	      }

	      uint64_t *h_offsets = static_cast<uint64_t *>(host_mgr.allocateManagedBytes(
	          sizeof(uint64_t) * static_cast<size_t>(chunk_rows + 1),
	          sizeof(uint64_t)));
	      if (h_offsets == nullptr)
	      {
	        std::cerr << "direct_join_with_duplication_streaming_nomask: host allocation failed for offsets." << std::endl;
	        exit(EXIT_FAILURE);
	      }

      uint64_t global_out_offset = 0;
      for (uint64_t cur = 0; cur < jc.num_frag_rows;)
      {
        uint64_t n = std::min<uint64_t>(chunk_rows, jc.num_frag_rows - cur);
        if (n == 0)
          break;
        if (n < 32)
          n = jc.num_frag_rows - cur;
        n = std::min<uint64_t>(n, jc.num_frag_rows - cur);

        h_data_infos = jc.h_data_info_arrays_;
        h_parent_infos = jc.h_parent_info_arrays_;
        if (!prepare_joininfo_chunk(cur, cur + n, h_data_infos, h_parent_infos, staging_buffers, jc.join_stream))
        {
          std::cerr << "direct_join_with_duplication_streaming_nomask: failed to prepare chunk at " << cur << std::endl;
          exit(EXIT_FAILURE);
        }

        uint32_t *d_counts = nullptr;
        uint64_t *d_offsets = nullptr;
        void *d_scan_tmp = nullptr;
        size_t scan_tmp_bytes = 0;

        if (!mem_mgr.allocate_temporal(d_counts, sizeof(uint32_t) * static_cast<size_t>(n + 1)) ||
            !mem_mgr.allocate_temporal(d_offsets, sizeof(uint64_t) * static_cast<size_t>(n + 1)))
        {
          std::cerr << "direct_join_with_duplication_streaming_nomask: failed to allocate chunk temporals." << std::endl;
          exit(EXIT_FAILURE);
        }

        cuchk(cudaMemsetAsync(d_counts, 0, sizeof(uint32_t) * static_cast<size_t>(n + 1), jc.join_stream));

        join_count_kernel_dup_nomask_pipeline_local_info<<<GRID_DIM, BLOCK_DIM, 0, jc.join_stream>>>(
            jc.d_data_info_arrays_, jc.d_parent_info_arrays_,
            cur, n,
            jc.frag_trie->num_cols, jc.col_key,
            jc.d_edge_values_, jc.d_edge_offsets_,
            ht.h_num_buckets_[jc.directed_edge_idx],
            ht.h_hash_table_offs_[jc.directed_edge_idx],
            jc.directed_edge_idx,
            d_counts,
            jc.dup_check_type, jc.num_dup_cols, jc.d_dup_cols, jc.up_to_col);

        cub::DeviceScan::ExclusiveScan(d_scan_tmp, scan_tmp_bytes,
                                       d_counts, d_offsets,
                                       cub::Sum(), static_cast<uint64_t>(0), n + 1, jc.join_stream);
        if (!mem_mgr.allocate_temporal(d_scan_tmp, scan_tmp_bytes))
        {
          std::cerr << "direct_join_with_duplication_streaming_nomask: failed to allocate scan temp storage." << std::endl;
          exit(EXIT_FAILURE);
        }
        cub::DeviceScan::ExclusiveScan(d_scan_tmp, scan_tmp_bytes,
                                       d_counts, d_offsets,
                                       cub::Sum(), static_cast<uint64_t>(0), n + 1, jc.join_stream);

        cuchk(cudaMemcpyAsync(h_offsets, d_offsets, sizeof(uint64_t) * static_cast<size_t>(n + 1),
                              cudaMemcpyDeviceToHost, jc.join_stream));
        cuchk(cudaStreamSynchronize(jc.join_stream));

        uint64_t chunk_total = h_offsets[n];
        uint64_t row_start = 0;
        while (row_start < n)
        {
          uint64_t row_end = find_frag_range(h_offsets, row_start, n, staging_capacity);
          if (row_end == row_start)
            row_end = row_start + 1;
          uint64_t num_rows_sub = row_end - row_start;
          uint64_t num_results_sub = h_offsets[row_end] - h_offsets[row_start];
          if (num_results_sub == 0)
          {
            row_start = row_end;
            continue;
          }

          uint32_t *d_write_data = d_out_data;
          uint64_t *d_write_parents = d_out_parents;
          uint32_t *d_big_data = nullptr;
          uint64_t *d_big_parents = nullptr;
          if (num_results_sub > staging_capacity)
          {
            if (!mem_mgr.allocate_temporal(d_big_data, sizeof(uint32_t) * static_cast<size_t>(num_results_sub)) ||
                !mem_mgr.allocate_temporal(d_big_parents, sizeof(uint64_t) * static_cast<size_t>(num_results_sub)))
            {
              std::cerr << "direct_join_with_duplication_streaming_nomask: failed to allocate oversized staging buffers (" << num_results_sub << ")." << std::endl;
              exit(EXIT_FAILURE);
            }
            d_write_data = d_big_data;
            d_write_parents = d_big_parents;
          }

          join_write_kernel_dup_nomask_pipeline_local_info<<<GRID_DIM, BLOCK_DIM, 0, jc.join_stream>>>(
              cur + row_start, num_rows_sub,
              jc.d_data_info_arrays_, jc.d_parent_info_arrays_,
              jc.frag_trie->num_cols, jc.col_key,
              jc.d_edge_values_, jc.d_edge_offsets_,
              ht.h_num_buckets_[jc.directed_edge_idx],
              ht.h_hash_table_offs_[jc.directed_edge_idx],
              jc.directed_edge_idx,
              d_offsets + row_start, h_offsets[row_start],
              d_write_data, d_write_parents,
              jc.dup_check_type, jc.num_dup_cols, jc.d_dup_cols, jc.up_to_col);

          cuchk(cudaMemcpyAsync(h_data_ptr + global_out_offset + h_offsets[row_start], d_write_data,
                                sizeof(uint32_t) * static_cast<size_t>(num_results_sub),
                                cudaMemcpyDeviceToHost, jc.join_stream));
          cuchk(cudaMemcpyAsync(h_parents_ptr + global_out_offset + h_offsets[row_start], d_write_parents,
                                sizeof(uint64_t) * static_cast<size_t>(num_results_sub),
                                cudaMemcpyDeviceToHost, jc.join_stream));

          if (d_big_data)
            mem_mgr.deallocate_temporal(d_big_data);
          if (d_big_parents)
            mem_mgr.deallocate_temporal(d_big_parents);

          row_start = row_end;
        }

        cuchk(cudaStreamSynchronize(jc.join_stream));
        global_out_offset += chunk_total;
        cur += n;

        mem_mgr.deallocate_temporal(d_scan_tmp);
        mem_mgr.deallocate_temporal(d_counts);
        mem_mgr.deallocate_temporal(d_offsets);
        cleanup_chunk_staging_buffers(staging_buffers);
      }

	      {
	        void *h_offsets_void = h_offsets;
	        host_mgr.deallocateManaged(h_offsets_void);
	        h_offsets = nullptr;
	      }
	      mem_mgr.deallocate_temporal(d_out_data);
	      mem_mgr.deallocate_temporal(d_out_parents);
	    }
    else
    {
      mem_mgr.p_type = ProcessType::DIRECT;

      uint64_t global_out_offset = 0;
      for (uint64_t cur = 0; cur < jc.num_frag_rows;)
      {
        uint64_t n = std::min<uint64_t>(chunk_rows, jc.num_frag_rows - cur);
        if (n == 0)
          break;
        if (n < 32)
          n = jc.num_frag_rows - cur;
        n = std::min<uint64_t>(n, jc.num_frag_rows - cur);

        h_data_infos = jc.h_data_info_arrays_;
        h_parent_infos = jc.h_parent_info_arrays_;
        if (!prepare_joininfo_chunk(cur, cur + n, h_data_infos, h_parent_infos, staging_buffers, jc.join_stream))
        {
          std::cerr << "direct_join_with_duplication_streaming_nomask: failed to prepare chunk at " << cur << std::endl;
          exit(EXIT_FAILURE);
        }

        uint32_t *d_counts = nullptr;
        uint64_t *d_offsets = nullptr;
        void *d_scan_tmp = nullptr;
        size_t scan_tmp_bytes = 0;

        if (!mem_mgr.allocate_temporal(d_counts, sizeof(uint32_t) * static_cast<size_t>(n + 1)) ||
            !mem_mgr.allocate_temporal(d_offsets, sizeof(uint64_t) * static_cast<size_t>(n + 1)))
        {
          std::cerr << "direct_join_with_duplication_streaming_nomask: failed to allocate chunk temporals." << std::endl;
          exit(EXIT_FAILURE);
        }

        cuchk(cudaMemsetAsync(d_counts, 0, sizeof(uint32_t) * static_cast<size_t>(n + 1), jc.join_stream));

        join_count_kernel_dup_nomask_pipeline_local_info<<<GRID_DIM, BLOCK_DIM, 0, jc.join_stream>>>(
            jc.d_data_info_arrays_, jc.d_parent_info_arrays_,
            cur, n,
            jc.frag_trie->num_cols, jc.col_key,
            jc.d_edge_values_, jc.d_edge_offsets_,
            ht.h_num_buckets_[jc.directed_edge_idx],
            ht.h_hash_table_offs_[jc.directed_edge_idx],
            jc.directed_edge_idx,
            d_counts,
            jc.dup_check_type, jc.num_dup_cols, jc.d_dup_cols, jc.up_to_col);

        cub::DeviceScan::ExclusiveScan(d_scan_tmp, scan_tmp_bytes,
                                       d_counts, d_offsets,
                                       cub::Sum(), static_cast<uint64_t>(0), n + 1, jc.join_stream);
        if (!mem_mgr.allocate_temporal(d_scan_tmp, scan_tmp_bytes))
        {
          std::cerr << "direct_join_with_duplication_streaming_nomask: failed to allocate scan temp storage." << std::endl;
          exit(EXIT_FAILURE);
        }
        cub::DeviceScan::ExclusiveScan(d_scan_tmp, scan_tmp_bytes,
                                       d_counts, d_offsets,
                                       cub::Sum(), static_cast<uint64_t>(0), n + 1, jc.join_stream);

        uint64_t chunk_total = 0;
        cuchk(cudaMemcpyAsync(&chunk_total, d_offsets + n, sizeof(uint64_t),
                              cudaMemcpyDeviceToHost, jc.join_stream));
        cuchk(cudaStreamSynchronize(jc.join_stream));

        join_write_kernel_dup_nomask_pipeline_local_info<<<GRID_DIM, BLOCK_DIM, 0, jc.join_stream>>>(
            cur, n,
            jc.d_data_info_arrays_, jc.d_parent_info_arrays_,
            jc.frag_trie->num_cols, jc.col_key,
            jc.d_edge_values_, jc.d_edge_offsets_,
            ht.h_num_buckets_[jc.directed_edge_idx],
            ht.h_hash_table_offs_[jc.directed_edge_idx],
            jc.directed_edge_idx,
            d_offsets, 0,
            jc.target_trie->data->d_data + global_out_offset,
            jc.target_trie->data->d_parents + global_out_offset,
            jc.dup_check_type, jc.num_dup_cols, jc.d_dup_cols, jc.up_to_col);
        cuchk(cudaStreamSynchronize(jc.join_stream));

        global_out_offset += chunk_total;
        cur += n;

        mem_mgr.deallocate_temporal(d_scan_tmp);
        mem_mgr.deallocate_temporal(d_counts);
        mem_mgr.deallocate_temporal(d_offsets);
        cleanup_chunk_staging_buffers(staging_buffers);
      }
    }

    mem_mgr.unpin_climb_data();
    mem_mgr.unpin_climb_parents();
    cuchk(cudaStreamSynchronize(jc.join_stream));
  }
} // namespace

void direct_join_with_duplication_pipeline()
{
  std::cout << "Entering direct join with duplication pipeline." << std::endl;

  auto &mem_mgr = MemoryManager::getInstance();
  auto &ht = HashTable::getInstance();
  auto &jc = JoinContext::getInstance();
  auto &host_mgr = HostMemoryManager::getInstance();

  get_data_arrays_pipeline(jc.frag_trie->id, jc.pin_cols, jc.h_data_info_arrays_);
  get_parent_arrays_pipeline(jc.frag_trie->id, jc.up_to_col, jc.h_parent_info_arrays_);

  cuchk(cudaMemcpyAsync(
      jc.d_data_info_arrays_, jc.h_data_info_arrays_.data(),
      sizeof(JoinInfo<uint32_t>) * jc.h_data_info_arrays_.size(),
      cudaMemcpyHostToDevice, jc.join_stream));
  cuchk(cudaMemcpyAsync(
      jc.d_parent_info_arrays_, jc.h_parent_info_arrays_.data(),
      sizeof(JoinInfo<uint64_t>) * jc.h_parent_info_arrays_.size(),
      cudaMemcpyHostToDevice, jc.join_stream));

  jc.direct_join_with_duplication_context();
  if (jc.d_mask == nullptr)
  {
    mem_mgr.unpin_climb_data();
    mem_mgr.unpin_climb_parents();
    direct_join_with_duplication_streaming_nomask();
    return;
  }

  if (!mem_mgr.allocate_temporal(jc.d_num_res_for_each, sizeof(uint32_t) * (jc.num_frag_rows + 1)) ||
      !mem_mgr.allocate_temporal(jc.d_st_array, sizeof(uint32_t) * jc.num_frag_rows))
  {
    mem_mgr.unpin_climb_data();
    mem_mgr.unpin_climb_parents();
    mem_mgr.deallocate_temporal(jc.d_mask);
    if (jc.d_num_res_for_each)
      mem_mgr.deallocate_temporal(jc.d_num_res_for_each);
    if (jc.d_st_array)
      mem_mgr.deallocate_temporal(jc.d_st_array);
    direct_join_with_duplication_streaming_nomask();
    return;
  }
  cuchk(cudaMemsetAsync(jc.d_num_res_for_each, 0, sizeof(uint32_t) * (jc.num_frag_rows + 1), jc.join_stream));
  cuchk(cudaMemsetAsync(jc.d_st_array, 0xFFFFFFFF, sizeof(uint32_t) * jc.num_frag_rows, jc.join_stream));

  const int nStreams = JoinContext::nStreams;
  const size_t num_partial_parents = jc.partial_parent_cols.size();
  const size_t num_partial_data = jc.partial_data_cols.size();

  uint64_t rows_per_stream = jc.num_frag_rows;
  if (num_partial_parents + num_partial_data > 0)
  {
    size_t bytes_per_row = num_partial_parents * sizeof(uint64_t) + num_partial_data * sizeof(uint32_t);
    bytes_per_row = std::max<size_t>(bytes_per_row, 1);
    size_t budget_bytes = mem_mgr.stats.available_gpu_memory / 2 / nStreams;
    rows_per_stream = std::max<uint64_t>(budget_bytes / bytes_per_row, 32);
    rows_per_stream = std::min<uint64_t>(rows_per_stream, jc.num_frag_rows);
    rows_per_stream = (rows_per_stream / 32) * 32;
  }
  if (rows_per_stream == 0)
    rows_per_stream = 32;

  uint64_t *d_partial_parent_base = nullptr;
  if (num_partial_parents > 0)
  {
    mem_mgr.allocate_temporal(d_partial_parent_base,
                              sizeof(uint64_t) * rows_per_stream * num_partial_parents * nStreams);
  }

  uint32_t *d_partial_data_base = nullptr;
  if (num_partial_data > 0)
  {
    mem_mgr.allocate_temporal(d_partial_data_base,
                              sizeof(uint32_t) * rows_per_stream * num_partial_data * nStreams);
  }

  void *d_parent_info_arrays_base_void = nullptr;
  JoinInfo<uint64_t> *d_parent_info_arrays_base = nullptr;
  JoinInfo<uint64_t> *h_parent_info_arrays[nStreams];
  for (int i = 0; i < nStreams; ++i)
    h_parent_info_arrays[i] = nullptr;

  void *d_data_info_arrays_base_void = nullptr;
  JoinInfo<uint32_t> *d_data_info_arrays_base = nullptr;
  JoinInfo<uint32_t> *h_data_info_arrays[nStreams];
  for (int i = 0; i < nStreams; ++i)
    h_data_info_arrays[i] = nullptr;

  const size_t parent_info_count = jc.h_parent_info_arrays_.size();
  const size_t data_info_count = jc.h_data_info_arrays_.size();
  const bool need_parent_updates = (num_partial_parents > 0);
  const bool need_data_updates = (num_partial_data > 0);

  if (need_parent_updates)
  {
    mem_mgr.allocate_temporal(d_parent_info_arrays_base_void,
                              sizeof(JoinInfo<uint64_t>) * parent_info_count * nStreams);
    d_parent_info_arrays_base = static_cast<JoinInfo<uint64_t> *>(d_parent_info_arrays_base_void);
    for (int i = 0; i < nStreams; ++i)
    {
      h_parent_info_arrays[i] = static_cast<JoinInfo<uint64_t> *>(host_mgr.allocateManagedBytes(
          sizeof(JoinInfo<uint64_t>) * parent_info_count,
          alignof(JoinInfo<uint64_t>)));
      if (h_parent_info_arrays[i] == nullptr)
      {
        std::cerr << "direct_join_with_duplication_pipeline: host allocation failed for parent JoinInfo buffers." << std::endl;
        exit(EXIT_FAILURE);
      }
    }
  }

  if (need_data_updates)
  {
    mem_mgr.allocate_temporal(d_data_info_arrays_base_void,
                              sizeof(JoinInfo<uint32_t>) * data_info_count * nStreams);
    d_data_info_arrays_base = static_cast<JoinInfo<uint32_t> *>(d_data_info_arrays_base_void);
    for (int i = 0; i < nStreams; ++i)
    {
      h_data_info_arrays[i] = static_cast<JoinInfo<uint32_t> *>(host_mgr.allocateManagedBytes(
          sizeof(JoinInfo<uint32_t>) * data_info_count,
          alignof(JoinInfo<uint32_t>)));
      if (h_data_info_arrays[i] == nullptr)
      {
        std::cerr << "direct_join_with_duplication_pipeline: host allocation failed for data JoinInfo buffers." << std::endl;
        exit(EXIT_FAILURE);
      }
    }
  }

  cudaEvent_t init_event;
  cuchk(cudaEventCreateWithFlags(&init_event, cudaEventDisableTiming));
  cuchk(cudaEventRecord(init_event, jc.join_stream));
  for (int i = 0; i < nStreams; ++i)
    cuchk(cudaStreamWaitEvent(jc.pipeline_streams[i], init_event, 0));
  cuchk(cudaEventDestroy(init_event));

  cudaEvent_t done_events[nStreams];
  for (int i = 0; i < nStreams; ++i)
  {
    cuchk(cudaEventCreateWithFlags(&done_events[i], cudaEventDisableTiming));
    cuchk(cudaEventRecord(done_events[i], jc.pipeline_streams[i]));
  }

  std::vector<ColumnRange> ranges;
  uint64_t cur_offset = 0;
  while (cur_offset < jc.num_frag_rows)
  {
    for (int stream_id = 0; stream_id < nStreams && cur_offset < jc.num_frag_rows; ++stream_id)
    {
      cuchk(cudaEventSynchronize(done_events[stream_id]));

      uint64_t num_rows_this_iter = std::min(rows_per_stream, jc.num_frag_rows - cur_offset);
      cudaStream_t stream = jc.pipeline_streams[stream_id];

      compute_column_ranges(jc.frag_trie, jc.up_to_col, cur_offset, cur_offset + num_rows_this_iter, ranges, jc.join_stream);

      JoinInfo<uint64_t> *d_parent_info_ptr = jc.d_parent_info_arrays_;
      if (need_parent_updates)
      {
        JoinInfo<uint64_t> *h_parent_infos = h_parent_info_arrays[stream_id];
        for (size_t col = 0; col < parent_info_count; ++col)
          h_parent_infos[col] = jc.h_parent_info_arrays_[col];

        for (size_t i = 0; i < num_partial_parents; ++i)
        {
          uint32_t col = jc.partial_parent_cols[i];
          uint32_t col_trie_id = jc.frag_trie->column_ids[col];
          const int block_idx = static_cast<int>(col_trie_id) * 2 + 1;
          auto &block = mem_mgr.memory_blocks[block_idx];
          if (block.host_ptr == nullptr)
          {
            std::cerr << "direct_join_with_duplication_pipeline: missing host parents for trie " << col_trie_id << std::endl;
            exit(EXIT_FAILURE);
          }
          auto *h_parents = static_cast<uint64_t *>(block.host_ptr);
          ColumnRange r = ranges[col];
          uint64_t len = r.end - r.start;

          uint64_t *d_parent_buf = d_partial_parent_base +
                                  (static_cast<size_t>(stream_id) * num_partial_parents + i) * rows_per_stream;
          if (len > 0)
          {
            cuchk(cudaMemcpyAsync(d_parent_buf, h_parents + r.start, sizeof(uint64_t) * len,
                                  cudaMemcpyHostToDevice, stream));
            h_parent_infos[col].array_ptr = d_parent_buf;
            h_parent_infos[col].offset = r.start;
            h_parent_infos[col].is_partial = 1;
          }
          else
          {
            h_parent_infos[col].array_ptr = nullptr;
            h_parent_infos[col].offset = 0;
            h_parent_infos[col].is_partial = 1;
          }
        }

        d_parent_info_ptr = d_parent_info_arrays_base + parent_info_count * static_cast<size_t>(stream_id);
        cuchk(cudaMemcpyAsync(
            d_parent_info_ptr, h_parent_infos,
            sizeof(JoinInfo<uint64_t>) * parent_info_count,
            cudaMemcpyHostToDevice, stream));
      }

      JoinInfo<uint32_t> *d_data_info_ptr = jc.d_data_info_arrays_;
      if (need_data_updates)
      {
        JoinInfo<uint32_t> *h_data_infos = h_data_info_arrays[stream_id];
        for (size_t col = 0; col < data_info_count; ++col)
          h_data_infos[col] = jc.h_data_info_arrays_[col];

        for (size_t i = 0; i < num_partial_data; ++i)
        {
          uint32_t col = jc.partial_data_cols[i];
          uint32_t col_trie_id = jc.frag_trie->column_ids[col];
          const int block_idx = static_cast<int>(col_trie_id) * 2;
          auto &block = mem_mgr.memory_blocks[block_idx];
          if (block.host_ptr == nullptr)
          {
            std::cerr << "direct_join_with_duplication_pipeline: missing host data for trie " << col_trie_id << std::endl;
            exit(EXIT_FAILURE);
          }
          auto *h_data = static_cast<uint32_t *>(block.host_ptr);
          ColumnRange r = ranges[col];
          uint64_t len = r.end - r.start;

          uint32_t *d_data_buf = d_partial_data_base +
                                 (static_cast<size_t>(stream_id) * num_partial_data + i) * rows_per_stream;
          if (len > 0)
          {
            cuchk(cudaMemcpyAsync(d_data_buf, h_data + r.start, sizeof(uint32_t) * len,
                                  cudaMemcpyHostToDevice, stream));
            h_data_infos[col].array_ptr = d_data_buf;
            h_data_infos[col].offset = r.start;
            h_data_infos[col].is_partial = 1;
          }
          else
          {
            h_data_infos[col].array_ptr = nullptr;
            h_data_infos[col].offset = 0;
            h_data_infos[col].is_partial = 1;
          }
        }

        d_data_info_ptr = d_data_info_arrays_base + data_info_count * static_cast<size_t>(stream_id);
        cuchk(cudaMemcpyAsync(
            d_data_info_ptr, h_data_infos,
            sizeof(JoinInfo<uint32_t>) * data_info_count,
            cudaMemcpyHostToDevice, stream));
      }

      join_count_kernel_dup_pipeline_info<<<GRID_DIM, BLOCK_DIM, 0, stream>>>(
          d_data_info_ptr, d_parent_info_ptr,
          cur_offset, num_rows_this_iter,
          jc.frag_trie->num_cols, jc.col_key,
          jc.d_edge_keys_, jc.d_edge_values_, jc.d_edge_offsets_,
          ht.h_num_buckets_[jc.directed_edge_idx],
          ht.h_hash_table_offs_[jc.directed_edge_idx],
          jc.directed_edge_idx,
          jc.d_mask, jc.mask_length, jc.d_st_array,
          jc.dup_check_type, jc.num_dup_cols, jc.d_dup_cols, jc.up_to_col);

      cuchk(cudaEventRecord(done_events[stream_id], stream));
      cur_offset += num_rows_this_iter;
    }
  }

  for (int i = 0; i < nStreams; ++i)
  {
    cuchk(cudaEventSynchronize(done_events[i]));
    cuchk(cudaEventDestroy(done_events[i]));
  }

  if (need_parent_updates)
  {
    for (int i = 0; i < nStreams; ++i)
    {
      void *tmp = h_parent_info_arrays[i];
      host_mgr.deallocateManaged(tmp);
      h_parent_info_arrays[i] = nullptr;
    }
    mem_mgr.deallocate_temporal(d_parent_info_arrays_base_void);
  }
  if (need_data_updates)
  {
    for (int i = 0; i < nStreams; ++i)
    {
      void *tmp = h_data_info_arrays[i];
      host_mgr.deallocateManaged(tmp);
      h_data_info_arrays[i] = nullptr;
    }
    mem_mgr.deallocate_temporal(d_data_info_arrays_base_void);
  }
  if (d_partial_parent_base != nullptr)
    mem_mgr.deallocate_temporal(d_partial_parent_base);
  if (d_partial_data_base != nullptr)
    mem_mgr.deallocate_temporal(d_partial_data_base);

  mem_mgr.unpin_climb_data();
  mem_mgr.unpin_climb_parents();

  uint64_t *d_row_offsets = nullptr;
  if (!mem_mgr.allocate_temporal(d_row_offsets, sizeof(uint64_t) * (jc.num_frag_rows + 1)))
  {
    std::cout << "direct_join_with_duplication_pipeline: falling back to streaming join." << std::endl;
    mem_mgr.deallocate_temporal(jc.d_mask);
    mem_mgr.deallocate_temporal(jc.d_num_res_for_each);
    mem_mgr.deallocate_temporal(jc.d_st_array);
    direct_join_with_duplication_streaming_nomask();
    return;
  }
  cuchk(cudaMemsetAsync(d_row_offsets, 0, sizeof(uint64_t) * (jc.num_frag_rows + 1), jc.join_stream));

  size_t temp_storage_bytes = 0;
  void *d_temp_storage = nullptr;

  uint64_t *d_segment_offsets = nullptr;
  if (!mem_mgr.allocate_temporal(d_segment_offsets, sizeof(uint64_t) * (jc.num_frag_rows + 1)))
  {
    std::cout << "direct_join_with_duplication_pipeline: falling back to streaming join." << std::endl;
    mem_mgr.deallocate_temporal(d_row_offsets);
    mem_mgr.deallocate_temporal(jc.d_mask);
    mem_mgr.deallocate_temporal(jc.d_num_res_for_each);
    mem_mgr.deallocate_temporal(jc.d_st_array);
    direct_join_with_duplication_streaming_nomask();
    return;
  }

  thrust::device_ptr<uint64_t> offsets_ptr(d_segment_offsets);
  thrust::sequence(thrust::cuda::par.on(jc.join_stream),
                   offsets_ptr, offsets_ptr + jc.num_frag_rows + 1,
                   static_cast<uint64_t>(0),
                   static_cast<uint64_t>(jc.mask_length));

  auto popcount_iterator = thrust::make_transform_iterator(jc.d_mask, PopcToCount{});

  cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes,
                                  popcount_iterator, jc.d_num_res_for_each,
                                  jc.num_frag_rows, d_segment_offsets, d_segment_offsets + 1, jc.join_stream);
  if (!mem_mgr.allocate_temporal(d_temp_storage, temp_storage_bytes))
  {
    std::cout << "direct_join_with_duplication_pipeline: falling back to streaming join." << std::endl;
    mem_mgr.deallocate_temporal(d_segment_offsets);
    mem_mgr.deallocate_temporal(d_row_offsets);
    mem_mgr.deallocate_temporal(jc.d_mask);
    mem_mgr.deallocate_temporal(jc.d_num_res_for_each);
    mem_mgr.deallocate_temporal(jc.d_st_array);
    direct_join_with_duplication_streaming_nomask();
    return;
  }
  cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes,
                                  popcount_iterator, jc.d_num_res_for_each,
                                  jc.num_frag_rows, d_segment_offsets, d_segment_offsets + 1, jc.join_stream);
  mem_mgr.deallocate_temporal(d_segment_offsets);
  mem_mgr.deallocate_temporal(d_temp_storage);
  d_temp_storage = nullptr;
  temp_storage_bytes = 0;

  cub::DeviceScan::ExclusiveScan(d_temp_storage, temp_storage_bytes,
                                 jc.d_num_res_for_each, d_row_offsets,
                                 cub::Sum(), (uint64_t)0, jc.num_frag_rows + 1, jc.join_stream);
  if (!mem_mgr.allocate_temporal(d_temp_storage, temp_storage_bytes))
  {
    std::cout << "direct_join_with_duplication_pipeline: falling back to streaming join." << std::endl;
    mem_mgr.deallocate_temporal(d_row_offsets);
    mem_mgr.deallocate_temporal(jc.d_mask);
    mem_mgr.deallocate_temporal(jc.d_num_res_for_each);
    mem_mgr.deallocate_temporal(jc.d_st_array);
    direct_join_with_duplication_streaming_nomask();
    return;
  }
  cub::DeviceScan::ExclusiveScan(d_temp_storage, temp_storage_bytes,
                                 jc.d_num_res_for_each, d_row_offsets,
                                 cub::Sum(), (uint64_t)0, jc.num_frag_rows + 1, jc.join_stream);

  mem_mgr.deallocate_temporal(d_temp_storage);
  mem_mgr.deallocate_temporal(jc.d_num_res_for_each);

  cuchk(cudaMemcpyAsync(jc.h_num_res, d_row_offsets + jc.num_frag_rows,
                        sizeof(uint64_t),
                        cudaMemcpyDeviceToHost, jc.join_stream));

  cuchk(cudaStreamSynchronize(jc.join_stream));
  resetTargetTrieData(jc.target_trie, *(jc.h_num_res), false);

  if (jc.skip_final_materialization == false)
  {
    bool alloc_success = mem_mgr.allocate_trie(jc.target_trie->id);
    if (alloc_success)
    {
      join_write_kernel_dup<<<GRID_DIM, BLOCK_DIM, 0, jc.join_stream>>>(
          jc.num_frag_rows, jc.d_edge_values_,
          jc.d_mask, jc.mask_length, d_row_offsets, jc.d_st_array,
          jc.target_trie->data->d_data, jc.target_trie->data->d_parents);
      cuchk(cudaStreamSynchronize(jc.join_stream));
    }
    else
    {
      direct_join_with_duplication_pipeline_write(d_row_offsets);
    }
  }

  mem_mgr.deallocate_temporal(d_row_offsets);
  mem_mgr.deallocate_temporal(jc.d_mask);
  cuchk(cudaStreamSynchronize(jc.join_stream));
}

// return exclusive right boundary (one past the last fragment row to process)
template <typename OffsetT>
uint64_t find_frag_range(const OffsetT *h_row_offsets, uint64_t left, uint64_t right, uint64_t target_rows_limit)
{
  // Binary search to find the largest 'end' such that:
  // h_row_offsets[end] - h_row_offsets[left] <= target_rows_limit
  // Fragment range [left, end) produces exactly h_row_offsets[end] - h_row_offsets[left] results

  uint64_t l = left, r = right;
  while (l < r)
  {
    uint64_t mid = l + (r - l + 1) / 2; // Use upper mid to avoid infinite loop when l = r-1
    uint64_t target_rows = static_cast<uint64_t>(h_row_offsets[mid]) - static_cast<uint64_t>(h_row_offsets[left]);
    if (target_rows <= target_rows_limit)
    {
      l = mid; // mid is valid, try larger
    }
    else
    {
      r = mid - 1; // mid exceeds, try smaller
    }
  }

  // Now l is the largest index where h_row_offsets[l] - h_row_offsets[left] <= target_rows_limit
  // Fragment range [left, l) produces h_row_offsets[l] - h_row_offsets[left] results

  // However, due to duplicates (zero-result rows), we should skip forward past all duplicates
  // to find the actual boundary where results end
  while (l < right && static_cast<uint64_t>(h_row_offsets[l]) == static_cast<uint64_t>(h_row_offsets[l + 1]))
  {
    l++;
  }

  // Return exclusive end (l represents the index one past the last row to include)
  return l;
}

namespace
{
bool allocate_pipeline_write_buffers(
    uint64_t &num_target_rows_each_stream,
    uint32_t *&d_data_buffer,
    uint64_t *&d_parents_buffer)
{
  auto &mem_mgr = MemoryManager::getInstance();
  d_data_buffer = nullptr;
  d_parents_buffer = nullptr;

  while (num_target_rows_each_stream > 0)
  {
    uint32_t *data = nullptr;
    uint64_t *parents = nullptr;
    const size_t nStreams = JoinContext::nStreams;
    const size_t data_bytes = sizeof(uint32_t) * static_cast<size_t>(num_target_rows_each_stream) * nStreams;
    const size_t parent_bytes = sizeof(uint64_t) * static_cast<size_t>(num_target_rows_each_stream) * nStreams;

    if (mem_mgr.allocate_temporal(data, data_bytes) && mem_mgr.allocate_temporal(parents, parent_bytes))
    {
      d_data_buffer = data;
      d_parents_buffer = parents;
      return true;
    }

    if (data != nullptr)
      mem_mgr.deallocate_temporal(data);
    if (parents != nullptr)
      mem_mgr.deallocate_temporal(parents);

    num_target_rows_each_stream >>= 1;
  }

  return false;
}
} // namespace

void direct_join_with_duplication_pipeline_write(uint64_t *d_row_offsets)
{
  std::cout << "Entering direct join with duplication pipeline write." << std::endl;
  auto &utm = UnifiedTrieManager::getInstance();
  auto &mem_mgr = MemoryManager::getInstance();
  auto &host_mgr = HostMemoryManager::getInstance();
  auto &ht = HashTable::getInstance();
  auto &jc = JoinContext::getInstance();

  mem_mgr.p_type = ProcessType::PIPELINE;

  uint32_t *h_data_ptr = static_cast<uint32_t *>(host_mgr.allocateManagedBytes(
      sizeof(uint32_t) * jc.h_num_res[0],
      sizeof(uint32_t)));
  if (h_data_ptr == nullptr)
  {
    std::cerr << "Host memory allocation failed for direct join pipeline write." << std::endl;
    exit(EXIT_FAILURE);
  }
  uint64_t *h_parents_ptr = static_cast<uint64_t *>(host_mgr.allocateManagedBytes(
      sizeof(uint64_t) * jc.h_num_res[0],
      sizeof(uint64_t)));
  if (h_parents_ptr == nullptr)
  {
    std::cerr << "Host memory allocation failed for direct join pipeline write." << std::endl;
    exit(EXIT_FAILURE);
  }

  auto &data_mem_block = mem_mgr.memory_blocks[jc.target_trie->id * 2];
  data_mem_block.host_ptr = (void *)h_data_ptr;
  data_mem_block.device_ptr = nullptr;
  data_mem_block.size_in_bytes = sizeof(uint32_t) * jc.h_num_res[0];
  auto &parent_mem_block = mem_mgr.memory_blocks[jc.target_trie->id * 2 + 1];
  parent_mem_block.host_ptr = (void *)h_parents_ptr;
  parent_mem_block.device_ptr = nullptr;
  parent_mem_block.size_in_bytes = sizeof(uint64_t) * jc.h_num_res[0];

  size_t free_mem_total = std::max(
      mem_mgr.stats.available_gpu_memory / 100 * 95,       // 95% of available memory
      mem_mgr.stats.gpu_memory_limit - 100 * 1024 * 1024); // leave 100MB margin
  free_mem_total = mem_mgr.stats.available_gpu_memory / 100 * 90;
  size_t free_mem_each_stream = free_mem_total / JoinContext::nStreams;
  uint64_t num_target_rows_each_stream = free_mem_each_stream / (sizeof(uint32_t) + sizeof(uint64_t));
  uint64_t frag_offset = 0, result_offset = 0;

  uint64_t *h_row_offsets = static_cast<uint64_t *>(host_mgr.allocateBytes(
      sizeof(uint64_t) * (jc.num_frag_rows + 1),
      sizeof(uint64_t)));
  if (h_row_offsets == nullptr)
  {
    std::cerr << "Host memory allocation failed for direct join pipeline offsets." << std::endl;
    exit(EXIT_FAILURE);
  }
  cuchk(cudaMemcpyAsync(h_row_offsets, d_row_offsets,
                        sizeof(uint64_t) * (jc.num_frag_rows + 1),
                        cudaMemcpyDeviceToHost, jc.join_stream));
  // cuchk(cudaStreamSynchronize(jc.join_stream));

  if (num_target_rows_each_stream == 0)
  {
    std::cerr << "direct_join_with_duplication_pipeline_write: insufficient GPU memory to pipeline results." << std::endl;
    return;
  }

  // allocate buffer
  uint32_t *d_data_buffer = nullptr;
  uint64_t *d_parents_buffer = nullptr;
  if (!allocate_pipeline_write_buffers(num_target_rows_each_stream, d_data_buffer, d_parents_buffer))
  {
    std::cerr << "direct_join_with_duplication_pipeline_write: failed to allocate GPU buffers for pipelining." << std::endl;
    exit(EXIT_FAILURE);
  }
  uint32_t *d_data_ptrs[JoinContext::nStreams] = {
      d_data_buffer,
      d_data_buffer + num_target_rows_each_stream,
      d_data_buffer + 2 * num_target_rows_each_stream};
  uint64_t *d_parents_ptrs[JoinContext::nStreams] = {
      d_parents_buffer,
      d_parents_buffer + num_target_rows_each_stream,
      d_parents_buffer + 2 * num_target_rows_each_stream};

  cuchk(cudaStreamSynchronize(jc.join_stream));

  while (result_offset < jc.h_num_res[0])
  {
    for (int i = 0; i < JoinContext::nStreams && result_offset < jc.h_num_res[0]; ++i)
    {
      uint64_t num_frag_rows_this_iter = 0;
      uint64_t num_target_rows_this_iter = 0;

      // perform binary search on h_row_offsets to find how many fragment rows can be processed
      uint64_t frag_left = frag_offset;
      uint64_t frag_right = find_frag_range(h_row_offsets, frag_left, jc.num_frag_rows, num_target_rows_each_stream);
      // frag_right is exclusive end, so fragment range is [frag_left, frag_right)
      num_frag_rows_this_iter = frag_right - frag_left;

      join_write_kernel_dup_pipeline<<<GRID_DIM, BLOCK_DIM, 0, jc.pipeline_streams[i]>>>(
          // trie_inter
          frag_left, num_frag_rows_this_iter,
          // trie_edge
          jc.d_edge_values_,
          jc.d_mask, jc.mask_length,
          d_row_offsets, jc.d_st_array,
          // output buffers
          d_data_ptrs[i], d_parents_ptrs[i],
          result_offset);

      // Calculate actual number of result rows produced by [frag_left, frag_right)
      num_target_rows_this_iter = h_row_offsets[frag_right] - h_row_offsets[frag_left];

      // copy back to host
      cuchk(cudaMemcpyAsync(h_data_ptr + result_offset, d_data_ptrs[i],
                            sizeof(uint32_t) * num_target_rows_this_iter,
                            cudaMemcpyDeviceToHost, jc.pipeline_streams[i]));
      cuchk(cudaMemcpyAsync(h_parents_ptr + result_offset, d_parents_ptrs[i],
                            sizeof(uint64_t) * num_target_rows_this_iter,
                            cudaMemcpyDeviceToHost, jc.pipeline_streams[i]));
      frag_offset += num_frag_rows_this_iter;
      result_offset += num_target_rows_this_iter;
    }
  }
  // synchronize all streams
  for (int i = 0; i < JoinContext::nStreams; ++i)
    cuchk(cudaStreamSynchronize(jc.pipeline_streams[i]));
  mem_mgr.deallocate_temporal(d_data_buffer);
  mem_mgr.deallocate_temporal(d_parents_buffer);
  cuchk(cudaStreamSynchronize(jc.join_stream));
}

void direct_join_with_duplication_nomask_pipeline_write(uint64_t *d_row_offsets)
{
  std::cout << "Entering direct join with duplication nomask pipeline write." << std::endl;
  auto &mem_mgr = MemoryManager::getInstance();
  auto &host_mgr = HostMemoryManager::getInstance();
  auto &ht = HashTable::getInstance();
  auto &jc = JoinContext::getInstance();

  mem_mgr.p_type = ProcessType::PIPELINE;

  uint32_t *h_data_ptr = static_cast<uint32_t *>(host_mgr.allocateManagedBytes(
      sizeof(uint32_t) * jc.h_num_res[0],
      sizeof(uint32_t)));
  if (h_data_ptr == nullptr)
  {
    std::cerr << "Host memory allocation failed for direct join pipeline write." << std::endl;
    exit(EXIT_FAILURE);
  }
  uint64_t *h_parents_ptr = static_cast<uint64_t *>(host_mgr.allocateManagedBytes(
      sizeof(uint64_t) * jc.h_num_res[0],
      sizeof(uint64_t)));
  if (h_parents_ptr == nullptr)
  {
    std::cerr << "Host memory allocation failed for direct join pipeline write." << std::endl;
    exit(EXIT_FAILURE);
  }

  auto &data_mem_block = mem_mgr.memory_blocks[jc.target_trie->id * 2];
  data_mem_block.host_ptr = (void *)h_data_ptr;
  data_mem_block.device_ptr = nullptr;
  data_mem_block.size_in_bytes = sizeof(uint32_t) * jc.h_num_res[0];
  auto &parent_mem_block = mem_mgr.memory_blocks[jc.target_trie->id * 2 + 1];
  parent_mem_block.host_ptr = (void *)h_parents_ptr;
  parent_mem_block.device_ptr = nullptr;
  parent_mem_block.size_in_bytes = sizeof(uint64_t) * jc.h_num_res[0];

  size_t free_mem_total = mem_mgr.stats.available_gpu_memory / 100 * 90;
  size_t free_mem_each_stream = free_mem_total / JoinContext::nStreams;
  uint64_t num_target_rows_each_stream = free_mem_each_stream / (sizeof(uint32_t) + sizeof(uint64_t));
  uint64_t frag_offset = 0, result_offset = 0;

  uint64_t *h_row_offsets = static_cast<uint64_t *>(host_mgr.allocateBytes(
      sizeof(uint64_t) * (jc.num_frag_rows + 1),
      sizeof(uint64_t)));
  if (h_row_offsets == nullptr)
  {
    std::cerr << "Host memory allocation failed for direct join pipeline offsets." << std::endl;
    exit(EXIT_FAILURE);
  }
  cuchk(cudaMemcpyAsync(h_row_offsets, d_row_offsets,
                        sizeof(uint64_t) * (jc.num_frag_rows + 1),
                        cudaMemcpyDeviceToHost, jc.join_stream));
  cuchk(cudaStreamSynchronize(jc.join_stream));

  if (num_target_rows_each_stream == 0)
  {
    std::cerr << "direct_join_with_duplication_nomask_pipeline_write: insufficient GPU memory to pipeline results." << std::endl;
    return;
  }

  uint32_t *d_data_buffer = nullptr;
  uint64_t *d_parents_buffer = nullptr;
  if (!allocate_pipeline_write_buffers(num_target_rows_each_stream, d_data_buffer, d_parents_buffer))
  {
    std::cerr << "direct_join_with_duplication_nomask_pipeline_write: failed to allocate GPU buffers for pipelining." << std::endl;
    exit(EXIT_FAILURE);
  }
  uint32_t *d_data_ptrs[JoinContext::nStreams] = {
      d_data_buffer,
      d_data_buffer + num_target_rows_each_stream,
      d_data_buffer + 2 * num_target_rows_each_stream};
  uint64_t *d_parents_ptrs[JoinContext::nStreams] = {
      d_parents_buffer,
      d_parents_buffer + num_target_rows_each_stream,
      d_parents_buffer + 2 * num_target_rows_each_stream};

  while (result_offset < jc.h_num_res[0])
  {
    for (int i = 0; i < JoinContext::nStreams && result_offset < jc.h_num_res[0]; ++i)
    {
      uint64_t frag_left = frag_offset;
      uint64_t frag_right = find_frag_range(h_row_offsets, frag_left, jc.num_frag_rows, num_target_rows_each_stream);
      uint64_t num_frag_rows_this_iter = frag_right - frag_left;
      uint64_t num_target_rows_this_iter = h_row_offsets[frag_right] - h_row_offsets[frag_left];
      if (num_target_rows_this_iter == 0)
      {
        frag_offset += num_frag_rows_this_iter;
        continue;
      }

      join_write_kernel_dup_nomask_pipeline<<<GRID_DIM, BLOCK_DIM, 0, jc.pipeline_streams[i]>>>(
          frag_left, num_frag_rows_this_iter,
          jc.d_data_arrays_, jc.d_parent_arrays_, jc.frag_trie->num_cols, jc.col_key,
          jc.d_edge_values_, jc.d_edge_offsets_,
          ht.h_num_buckets_[jc.directed_edge_idx],
          ht.h_hash_table_offs_[jc.directed_edge_idx],
          jc.directed_edge_idx,
          d_row_offsets,
          d_data_ptrs[i], d_parents_ptrs[i], result_offset,
          jc.dup_check_type, jc.num_dup_cols, jc.d_dup_cols, jc.up_to_col);

      cuchk(cudaMemcpyAsync(h_data_ptr + result_offset, d_data_ptrs[i],
                            sizeof(uint32_t) * num_target_rows_this_iter,
                            cudaMemcpyDeviceToHost, jc.pipeline_streams[i]));
      cuchk(cudaMemcpyAsync(h_parents_ptr + result_offset, d_parents_ptrs[i],
                            sizeof(uint64_t) * num_target_rows_this_iter,
                            cudaMemcpyDeviceToHost, jc.pipeline_streams[i]));
      frag_offset += num_frag_rows_this_iter;
      result_offset += num_target_rows_this_iter;
    }
  }

  for (int i = 0; i < JoinContext::nStreams; ++i)
    cuchk(cudaStreamSynchronize(jc.pipeline_streams[i]));
  mem_mgr.deallocate_temporal(d_data_buffer);
  mem_mgr.deallocate_temporal(d_parents_buffer);
  cuchk(cudaStreamSynchronize(jc.join_stream));
}

void direct_join_without_duplication_pipeline_write(uint64_t *d_row_offsets)
{
  std::cout << "Entering direct join without duplication pipeline write." << std::endl;
  auto &utm = UnifiedTrieManager::getInstance();
  auto &mem_mgr = MemoryManager::getInstance();
  auto &host_mgr = HostMemoryManager::getInstance();
  auto &jc = JoinContext::getInstance();

  mem_mgr.p_type = ProcessType::PIPELINE;

  uint32_t *h_data_ptr = static_cast<uint32_t *>(host_mgr.allocateManagedBytes(
      sizeof(uint32_t) * jc.h_num_res[0],
      sizeof(uint32_t)));
  if (h_data_ptr == nullptr)
  {
    std::cerr << "Host memory allocation failed for direct join pipeline write." << std::endl;
    exit(EXIT_FAILURE);
  }
  uint64_t *h_parents_ptr = static_cast<uint64_t *>(host_mgr.allocateManagedBytes(
      sizeof(uint64_t) * jc.h_num_res[0],
      sizeof(uint64_t)));
  if (h_parents_ptr == nullptr)
  {
    std::cerr << "Host memory allocation failed for direct join pipeline write." << std::endl;
    exit(EXIT_FAILURE);
  }

  auto &data_mem_block = mem_mgr.memory_blocks[jc.target_trie->id * 2];
  data_mem_block.host_ptr = (void *)h_data_ptr;
  data_mem_block.device_ptr = nullptr;
  data_mem_block.size_in_bytes = sizeof(uint32_t) * jc.h_num_res[0];
  auto &parent_mem_block = mem_mgr.memory_blocks[jc.target_trie->id * 2 + 1];
  parent_mem_block.host_ptr = (void *)h_parents_ptr;
  parent_mem_block.device_ptr = nullptr;
  parent_mem_block.size_in_bytes = sizeof(uint64_t) * jc.h_num_res[0];

  size_t free_mem_total = mem_mgr.stats.available_gpu_memory / 100 * 90;
  size_t free_mem_each_stream = free_mem_total / JoinContext::nStreams;
  uint64_t num_target_rows_each_stream = free_mem_each_stream / (sizeof(uint32_t) + sizeof(uint64_t));
  uint64_t frag_offset = 0, result_offset = 0;

  uint64_t *h_row_offsets = static_cast<uint64_t *>(host_mgr.allocateBytes(
      sizeof(uint64_t) * (jc.num_frag_rows + 1),
      sizeof(uint64_t)));
  if (h_row_offsets == nullptr)
  {
    std::cerr << "Host memory allocation failed for direct join pipeline offsets." << std::endl;
    exit(EXIT_FAILURE);
  }
  cuchk(cudaMemcpyAsync(h_row_offsets, d_row_offsets,
                        sizeof(uint64_t) * (jc.num_frag_rows + 1),
                        cudaMemcpyDeviceToHost, jc.join_stream));
  cuchk(cudaStreamSynchronize(jc.join_stream));

  if (num_target_rows_each_stream == 0)
  {
    std::cerr << "direct_join_without_duplication_pipeline_write: insufficient GPU memory to pipeline results." << std::endl;
    return;
  }

  // allocate buffer
  uint32_t *d_data_buffer = nullptr;
  uint64_t *d_parents_buffer = nullptr;
  if (!allocate_pipeline_write_buffers(num_target_rows_each_stream, d_data_buffer, d_parents_buffer))
  {
    std::cerr << "direct_join_without_duplication_pipeline_write: failed to allocate GPU buffers for pipelining." << std::endl;
    exit(EXIT_FAILURE);
  }
  uint32_t *d_data_ptrs[JoinContext::nStreams] = {
      d_data_buffer,
      d_data_buffer + num_target_rows_each_stream,
      d_data_buffer + 2 * num_target_rows_each_stream};
  uint64_t *d_parents_ptrs[JoinContext::nStreams] = {
      d_parents_buffer,
      d_parents_buffer + num_target_rows_each_stream,
      d_parents_buffer + 2 * num_target_rows_each_stream};

  while (result_offset < jc.h_num_res[0])
  {
    for (int i = 0; i < JoinContext::nStreams && result_offset < jc.h_num_res[0]; ++i)
    {
      uint64_t num_frag_rows_this_iter = 0;
      uint64_t num_target_rows_this_iter = 0;

      // perform binary search on h_row_offsets to find how many fragment rows can be processed
      uint64_t frag_left = frag_offset;
      uint64_t frag_right = find_frag_range(h_row_offsets, frag_left, jc.num_frag_rows, num_target_rows_each_stream);
      num_frag_rows_this_iter = frag_right - frag_left;
      num_target_rows_this_iter = h_row_offsets[frag_right] - h_row_offsets[frag_left];
      if (num_target_rows_this_iter == 0)
      {
        frag_offset += num_frag_rows_this_iter;
        continue;
      }

      join_write_kernel_nodup_pipeline<<<GRID_DIM, BLOCK_DIM, 0, jc.pipeline_streams[i]>>>(
          // trie_inter
          frag_left, num_frag_rows_this_iter,
          // trie_edge
          jc.d_edge_values_,
          jc.d_st_array, d_row_offsets,
          // output buffers
          d_data_ptrs[i], d_parents_ptrs[i],
          result_offset);

      // copy back to host
      cuchk(cudaMemcpyAsync(h_data_ptr + result_offset, d_data_ptrs[i],
                            sizeof(uint32_t) * num_target_rows_this_iter,
                            cudaMemcpyDeviceToHost, jc.pipeline_streams[i]));
      cuchk(cudaMemcpyAsync(h_parents_ptr + result_offset, d_parents_ptrs[i],
                            sizeof(uint64_t) * num_target_rows_this_iter,
                            cudaMemcpyDeviceToHost, jc.pipeline_streams[i]));
      frag_offset += num_frag_rows_this_iter;
      result_offset += num_target_rows_this_iter;
    }
  }
  // synchronize all streams
  for (int i = 0; i < JoinContext::nStreams; ++i)
    cuchk(cudaStreamSynchronize(jc.pipeline_streams[i]));
  mem_mgr.deallocate_temporal(d_data_buffer);
  mem_mgr.deallocate_temporal(d_parents_buffer);
  cuchk(cudaStreamSynchronize(jc.join_stream));
}

void direct_join_with_duplication_debug()
{
  // std::cout << "Performing Direct Join with duplication check" << std::endl;

  auto &mem_mgr = MemoryManager::getInstance();
  auto &ht = HashTable::getInstance();
  auto &jc = JoinContext::getInstance();

  jc.direct_join_with_duplication_context();

  join_count_kernel_dup_2<<<GRID_DIM, BLOCK_DIM, 0, jc.join_stream>>>(
      jc.d_data_arrays_, jc.d_parent_arrays_, jc.num_frag_rows, jc.frag_trie->num_cols, jc.col_key,
      jc.d_edge_keys_, jc.d_edge_values_, jc.d_edge_offsets_,
      ht.h_num_buckets_[jc.directed_edge_idx],
      ht.h_hash_table_offs_[jc.directed_edge_idx],
      jc.directed_edge_idx,
      jc.d_mask, jc.mask_length, jc.d_st_array,
      jc.dup_check_type, jc.num_dup_cols, jc.d_dup_cols, jc.up_to_col);
  mem_mgr.unpin_climb_data();
  mem_mgr.unpin_climb_parents();

  uint64_t *d_row_offsets = nullptr;
  mem_mgr.allocate_temporal(d_row_offsets, sizeof(uint64_t) * (jc.num_frag_rows + 1));
  cuchk(cudaMemsetAsync(d_row_offsets, 0, sizeof(uint64_t) * (jc.num_frag_rows + 1), jc.join_stream));

  // Exclusive prefix sum to compute write offsets for each row
  // Input: d_num_res_for_each[i] = count for row i
  // Output: d_row_offsets[i] = exclusive scan (starting offset for row i)
  //         d_row_offsets[num_rows] = total count
  size_t temp_storage_bytes = 0;
  void *d_temp_storage = nullptr;

  // Create segment offsets: [0, mask_length, 2*mask_length, ..., num_rows*mask_length]
  // Use 64-bit offsets to avoid overflow when num_rows_frag * mask_length exceeds 2^32.
  uint64_t *d_segment_offsets = nullptr;
  mem_mgr.allocate_temporal(d_segment_offsets, sizeof(uint64_t) * (jc.num_frag_rows + 1));

  thrust::device_ptr<uint64_t> offsets_ptr(d_segment_offsets);
  // Use stream-bound execution policy to utilize the memory pool associated withjc.join_stream
  thrust::sequence(thrust::cuda::par.on(jc.join_stream),
                   offsets_ptr, offsets_ptr + jc.num_frag_rows + 1,
                   static_cast<uint64_t>(0),
                   static_cast<uint64_t>(jc.mask_length));

  // Create a transform iterator that applies __popc to each mask tile
  auto popcount_iterator = thrust::make_transform_iterator(jc.d_mask, PopcToCount{});

  // Use CUB DeviceSegmentedReduce to sum popcounts per row
  cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes,
                                  popcount_iterator, jc.d_num_res_for_each,
                                  jc.num_frag_rows, d_segment_offsets, d_segment_offsets + 1, jc.join_stream);
  mem_mgr.allocate_temporal(d_temp_storage, temp_storage_bytes);
  cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes,
                                  popcount_iterator, jc.d_num_res_for_each,
                                  jc.num_frag_rows, d_segment_offsets, d_segment_offsets + 1, jc.join_stream);
  // Free temporary arrays
  mem_mgr.deallocate_temporal(d_segment_offsets);
  mem_mgr.deallocate_temporal(d_temp_storage);
  d_temp_storage = nullptr;
  temp_storage_bytes = 0;

  cub::DeviceScan::ExclusiveScan(d_temp_storage, temp_storage_bytes,
                                 jc.d_num_res_for_each, d_row_offsets,
                                 cub::Sum(), (uint64_t)0, jc.num_frag_rows + 1, jc.join_stream);
  mem_mgr.allocate_temporal(d_temp_storage, temp_storage_bytes);
  cub::DeviceScan::ExclusiveScan(d_temp_storage, temp_storage_bytes,
                                 jc.d_num_res_for_each, d_row_offsets,
                                 cub::Sum(), (uint64_t)0, jc.num_frag_rows + 1, jc.join_stream);

  mem_mgr.deallocate_temporal(d_temp_storage);
  mem_mgr.deallocate_temporal(jc.d_num_res_for_each);
  // Copy total count from d_row_offsets[num_rows] to host
  cuchk(cudaMemcpyAsync(jc.h_num_res, d_row_offsets + jc.num_frag_rows,
                        sizeof(uint64_t),
                        cudaMemcpyDeviceToHost, jc.join_stream));

  cuchk(cudaStreamSynchronize(jc.join_stream));
  resetTargetTrieData(jc.target_trie, *(jc.h_num_res), false);

#ifndef NDEBUG
  std::cout << "Direct Join produced " << *(jc.h_num_res) << " results." << std::endl;
#endif

  if (jc.skip_final_materialization == false)
  {
    bool alloc_success = mem_mgr.allocate_trie(jc.target_trie->id);
    // jc.target_trie->data->allocate();

    // cuchk(cudaMemset(jc.target_trie->data->d_data, 0xFF, sizeof(vtype) * h_num_res));
    // cuchk(cudaMemset(jc.target_trie->data->d_parents, 0xFF, sizeof(uint32_t) * h_num_res));
    // cuchk(cudaMalloc((void **)&jc.target_trie->data->d_data, sizeof(vtype) * h_num_res));

    if (alloc_success)
    {
      join_write_kernel_dup<<<GRID_DIM, BLOCK_DIM, 0, jc.join_stream>>>(
          jc.num_frag_rows, jc.d_edge_values_,
          jc.d_mask, jc.mask_length, d_row_offsets, jc.d_st_array,
          jc.target_trie->data->d_data, jc.target_trie->data->d_parents);
      cuchk(cudaStreamSynchronize(jc.join_stream));
    }
    else // if (!alloc_success)
    {
      direct_join_with_duplication_pipeline_write(d_row_offsets);
    }
  }
  mem_mgr.deallocate_temporal(d_row_offsets);
  mem_mgr.deallocate_temporal(jc.d_mask);

  cuchk(cudaStreamSynchronize(jc.join_stream));
}

void direct_join()
{
  auto &mem_mgr = MemoryManager::getInstance();
  auto &jc = JoinContext::getInstance();

  bool context_ready = jc.direct_join_general_context();
  if (!context_ready)
  {
    if (jc.dup_check_type == 0)
      direct_join_without_duplication_pipeline();
    else
      direct_join_with_duplication_pipeline();
  }
  else if (jc.dup_check_type == 0)
  {
#ifndef NDEBUG
    direct_join_without_duplication_debug();
#else
    direct_join_without_duplication();
#endif
  }
  else
  {
#ifndef NDEBUG
    direct_join_with_duplication_debug();
#else
    direct_join_with_duplication();
#endif
  }
  // Update target properties
  mem_mgr.deallocate_temporal(jc.d_st_array);

  jc.target_trie->updateNumResults(jc.h_num_res[0]);
  cuchk(cudaStreamSynchronize(jc.join_stream));
  // target_trie->data->num_rows = h_num_res;

#ifndef NDEBUG
  std::cout << "direct_join: " << jc.h_num_res[0] << " results produced" << std::endl;
#endif
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
  auto &jc = JoinContext::getInstance();
  switch (task_per_warp)
  {
  case 1:
    join_count_kernel_nodup<1><<<GRID_DIM, BLOCK_DIM, 0, jc.join_stream>>>(
        vertex_arrays_, parent_pointers_, num_rows, num_cols, join_col,
        edge_key_array_, edge_value_array_, edge_offsets_,
        num_buckets, edge_offset, directed_edge_idx, d_num_res_for_each, d_st_array);
    break;
  case 2:
    join_count_kernel_nodup<2><<<GRID_DIM, BLOCK_DIM, 0, jc.join_stream>>>(
        vertex_arrays_, parent_pointers_, num_rows, num_cols, join_col,
        edge_key_array_, edge_value_array_, edge_offsets_,
        num_buckets, edge_offset, directed_edge_idx,
        d_num_res_for_each, d_st_array);
    break;
  case 4:
    join_count_kernel_nodup<4><<<GRID_DIM, BLOCK_DIM, 0, jc.join_stream>>>(
        vertex_arrays_, parent_pointers_, num_rows, num_cols, join_col,
        edge_key_array_, edge_value_array_, edge_offsets_,
        num_buckets, edge_offset, directed_edge_idx,
        d_num_res_for_each, d_st_array);
    break;
  case 8:
    join_count_kernel_nodup<8><<<GRID_DIM, BLOCK_DIM, 0, jc.join_stream>>>(
        vertex_arrays_, parent_pointers_, num_rows, num_cols, join_col,
        edge_key_array_, edge_value_array_, edge_offsets_,
        num_buckets, edge_offset, directed_edge_idx,
        d_num_res_for_each, d_st_array);
    break;
  case 16:
    join_count_kernel_nodup<16><<<GRID_DIM, BLOCK_DIM, 0, jc.join_stream>>>(
        vertex_arrays_, parent_pointers_, num_rows, num_cols, join_col,
        edge_key_array_, edge_value_array_, edge_offsets_,
        num_buckets, edge_offset, directed_edge_idx,
        d_num_res_for_each, d_st_array);
    break;
  case 32:
    join_count_kernel_nodup<32><<<GRID_DIM, BLOCK_DIM, 0, jc.join_stream>>>(
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
  auto &jc = JoinContext::getInstance();
  switch (num_tasks_per_warp)
  {
  case 1:
    join_write_kernel_nodup<1><<<GRID_DIM, BLOCK_DIM, 0, jc.join_stream>>>(
        num_rows,
        edge_value_array_,
        d_num_res_for_each, d_st_array,
        d_res, d_offsets_of_, d_parent_res);
    break;
  case 2:
    join_write_kernel_nodup<2><<<GRID_DIM, BLOCK_DIM, 0, jc.join_stream>>>(
        num_rows,
        edge_value_array_,
        d_num_res_for_each, d_st_array,
        d_res, d_offsets_of_, d_parent_res);
    break;
  case 4:
    join_write_kernel_nodup<4><<<GRID_DIM, BLOCK_DIM, 0, jc.join_stream>>>(
        num_rows,
        edge_value_array_,
        d_num_res_for_each, d_st_array,
        d_res, d_offsets_of_, d_parent_res);
    break;
  case 8:
    join_write_kernel_nodup<8><<<GRID_DIM, BLOCK_DIM, 0, jc.join_stream>>>(
        num_rows,
        edge_value_array_,
        d_num_res_for_each, d_st_array,
        d_res, d_offsets_of_, d_parent_res);
    break;
  case 16:
    join_write_kernel_nodup<16><<<GRID_DIM, BLOCK_DIM, 0, jc.join_stream>>>(
        num_rows,
        edge_value_array_,
        d_num_res_for_each, d_st_array,
        d_res, d_offsets_of_, d_parent_res);
    break;
  case 32:
    join_write_kernel_nodup<32><<<GRID_DIM, BLOCK_DIM, 0, jc.join_stream>>>(
        num_rows,
        edge_value_array_,
        d_num_res_for_each, d_st_array,
        d_res, d_offsets_of_, d_parent_res);
    break;
  default:
    std::cerr << "Unsupported task_per_warp: " << num_tasks_per_warp << std::endl;
    exit(EXIT_FAILURE);
  }
  cuchk(cudaStreamSynchronize(jc.join_stream));
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
