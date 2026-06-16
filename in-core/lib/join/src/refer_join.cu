#include "direct_join_kernel.cuh"
#include "host_memory_manager.cuh"
#include "refer_join.cuh"
#include "refer_join_kernel.cuh"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

void join_refer_write_dup_nomask_pipeline_write(uint64_t *d_row_offsets);

namespace {
[[noreturn]] void fssmExit(int code, const char *file, int line) {
  throw std::runtime_error(std::string("exit(") + std::to_string(code) +
                           ") at " + (file != nullptr ? file : "<unknown>") +
                           ":" + std::to_string(line));
}

void resetTargetTrieData(UnifiedTrie *trie, uint64_t num_rows, bool is_mask) {
  if (trie == nullptr)
    throw std::runtime_error("resetTargetTrieData: target trie is null.");

  if (trie->data != nullptr || trie->r_data != nullptr) {
    trie->deallocateData();
  }

  trie->data = new DataBlock();
  trie->data->init(num_rows, is_mask);
}
} // namespace

#define exit(code) fssmExit((code), __FILE__, __LINE__)

namespace {
struct ColumnRange {
  uint64_t start = 0;
  uint64_t end = 0; // exclusive
};

uint64_t load_parent_host_or_device(uint32_t col_trie_id, uint64_t row_idx,
                                    cudaStream_t stream) {
  auto &mem_mgr = MemoryManager::getInstance();
  auto &utm = UnifiedTrieManager::getInstance();

  const int block_idx = static_cast<int>(col_trie_id) * 2 + 1;
  auto &block = mem_mgr.memory_blocks[block_idx];
  if (block.host_ptr != nullptr) {
    auto *h_parents = static_cast<uint64_t *>(block.host_ptr);
    return h_parents[row_idx];
  }

  UnifiedTrie *col_trie = utm.getTrie(col_trie_id);
  uint64_t value = UINT64_MAX;
  if (col_trie && col_trie->data && col_trie->data->d_parents) {
    cuchk(cudaMemcpyAsync(&value, col_trie->data->d_parents + row_idx,
                          sizeof(uint64_t), cudaMemcpyDeviceToHost, stream));
    cuchk(cudaStreamSynchronize(stream));
    return value;
  }
  if (col_trie && col_trie->r_data && col_trie->r_data->d_parents) {
    cuchk(cudaMemcpyAsync(&value, col_trie->r_data->d_parents + row_idx,
                          sizeof(uint64_t), cudaMemcpyDeviceToHost, stream));
    cuchk(cudaStreamSynchronize(stream));
    return value;
  }

  std::cerr << "load_parent_host_or_device: missing parents for trie "
            << col_trie_id << std::endl;
  exit(EXIT_FAILURE);
}

void compute_column_ranges(UnifiedTrie *trie, uint32_t col_upper_bound,
                           uint64_t leaf_start, uint64_t leaf_end,
                           std::vector<ColumnRange> &ranges,
                           cudaStream_t stream) {
  const int num_cols = trie->num_cols;
  ranges.assign(num_cols, ColumnRange{});
  ranges[num_cols - 1] = ColumnRange{leaf_start, leaf_end};

  for (int col = num_cols - 1; col > static_cast<int>(col_upper_bound); --col) {
    ColumnRange cur = ranges[col];
    if (cur.end <= cur.start) {
      ranges[col - 1] = ColumnRange{0, 0};
      continue;
    }

    uint32_t col_trie_id = trie->column_ids[col];
    uint64_t parent_start =
        load_parent_host_or_device(col_trie_id, cur.start, stream);
    uint64_t parent_end =
        load_parent_host_or_device(col_trie_id, cur.end - 1, stream);
    uint64_t st = std::min(parent_start, parent_end);
    uint64_t ed = std::max(parent_start, parent_end) + 1;
    ranges[col - 1] = ColumnRange{st, ed};
  }
}

bool allocate_pipeline_write_buffers(uint64_t &num_target_rows_each_stream,
                                     uint32_t *&d_data_buffer,
                                     uint64_t *&d_parents_buffer) {
  auto &mem_mgr = MemoryManager::getInstance();
  d_data_buffer = nullptr;
  d_parents_buffer = nullptr;

  while (num_target_rows_each_stream > 0) {
    uint32_t *data = nullptr;
    uint64_t *parents = nullptr;
    const size_t nStreams = JoinContext::nStreams;
    const size_t data_bytes = sizeof(uint32_t) *
                              static_cast<size_t>(num_target_rows_each_stream) *
                              nStreams;
    const size_t parent_bytes =
        sizeof(uint64_t) * static_cast<size_t>(num_target_rows_each_stream) *
        nStreams;

    if (mem_mgr.allocate_temporal(data, data_bytes) &&
        mem_mgr.allocate_temporal(parents, parent_bytes)) {
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

void join_count_refer_nodup_pipeline_dispatcher(
    JoinInfo<uint32_t> *data_info_arrays_,
    JoinInfo<uint64_t> *parent_info_arrays_, uint32_t *select_bool_vec_,
    uint64_t pipeline_off_st, uint64_t pipeline_num_rows, uint32_t num_cols,
    uint32_t join_col, vtype *edge_key_array_, vtype *edge_value_array_,
    uint32_t *edge_offsets_, uint32_t num_buckets, uint32_t edge_offset,
    uint32_t directed_edge_idx, uint32_t *d_num_res_for_each,
    uint32_t *d_st_array, uint32_t task_per_warp, cudaStream_t stream) {
  switch (task_per_warp) {
  case 1:
    join_count_refer_kernel_nodup_pipeline_info<1>
        <<<GRID_DIM, BLOCK_DIM, 0, stream>>>(
            data_info_arrays_, parent_info_arrays_, select_bool_vec_,
            pipeline_off_st, pipeline_num_rows, num_cols, join_col,
            edge_key_array_, edge_value_array_, edge_offsets_, num_buckets,
            edge_offset, directed_edge_idx, d_num_res_for_each, d_st_array);
    break;
  case 2:
    join_count_refer_kernel_nodup_pipeline_info<2>
        <<<GRID_DIM, BLOCK_DIM, 0, stream>>>(
            data_info_arrays_, parent_info_arrays_, select_bool_vec_,
            pipeline_off_st, pipeline_num_rows, num_cols, join_col,
            edge_key_array_, edge_value_array_, edge_offsets_, num_buckets,
            edge_offset, directed_edge_idx, d_num_res_for_each, d_st_array);
    break;
  case 4:
    join_count_refer_kernel_nodup_pipeline_info<4>
        <<<GRID_DIM, BLOCK_DIM, 0, stream>>>(
            data_info_arrays_, parent_info_arrays_, select_bool_vec_,
            pipeline_off_st, pipeline_num_rows, num_cols, join_col,
            edge_key_array_, edge_value_array_, edge_offsets_, num_buckets,
            edge_offset, directed_edge_idx, d_num_res_for_each, d_st_array);
    break;
  case 8:
    join_count_refer_kernel_nodup_pipeline_info<8>
        <<<GRID_DIM, BLOCK_DIM, 0, stream>>>(
            data_info_arrays_, parent_info_arrays_, select_bool_vec_,
            pipeline_off_st, pipeline_num_rows, num_cols, join_col,
            edge_key_array_, edge_value_array_, edge_offsets_, num_buckets,
            edge_offset, directed_edge_idx, d_num_res_for_each, d_st_array);
    break;
  case 16:
    join_count_refer_kernel_nodup_pipeline_info<16>
        <<<GRID_DIM, BLOCK_DIM, 0, stream>>>(
            data_info_arrays_, parent_info_arrays_, select_bool_vec_,
            pipeline_off_st, pipeline_num_rows, num_cols, join_col,
            edge_key_array_, edge_value_array_, edge_offsets_, num_buckets,
            edge_offset, directed_edge_idx, d_num_res_for_each, d_st_array);
    break;
  case 32:
    join_count_refer_kernel_nodup_pipeline_info<32>
        <<<GRID_DIM, BLOCK_DIM, 0, stream>>>(
            data_info_arrays_, parent_info_arrays_, select_bool_vec_,
            pipeline_off_st, pipeline_num_rows, num_cols, join_col,
            edge_key_array_, edge_value_array_, edge_offsets_, num_buckets,
            edge_offset, directed_edge_idx, d_num_res_for_each, d_st_array);
    break;
  default:
    std::cerr << "Unsupported task_per_warp: " << task_per_warp << std::endl;
    exit(EXIT_FAILURE);
  }
}
} // namespace

void refer_join_without_duplication_pipeline() {
  std::cout << "Entering refer join without duplication pipeline." << std::endl;

  auto &mem_mgr = MemoryManager::getInstance();
  auto &ht = HashTable::getInstance();
  auto &jc = JoinContext::getInstance();
  auto &host_mgr = HostMemoryManager::getInstance();

  get_data_arrays_pipeline(jc.root_trie->id, jc.pin_cols,
                           jc.h_data_info_arrays_);
  get_parent_arrays_pipeline(jc.root_trie->id, jc.up_to_col,
                             jc.h_parent_info_arrays_);

  cuchk(cudaMemcpyAsync(jc.d_data_info_arrays_, jc.h_data_info_arrays_.data(),
                        sizeof(JoinInfo<uint32_t>) *
                            jc.h_data_info_arrays_.size(),
                        cudaMemcpyHostToDevice, jc.join_stream));
  cuchk(cudaMemcpyAsync(
      jc.d_parent_info_arrays_, jc.h_parent_info_arrays_.data(),
      sizeof(JoinInfo<uint64_t>) * jc.h_parent_info_arrays_.size(),
      cudaMemcpyHostToDevice, jc.join_stream));

  mem_mgr.allocate_temporal(jc.d_num_res_for_each,
                            sizeof(uint32_t) * (jc.num_frag_rows + 1));
  cuchk(cudaMemsetAsync(jc.d_num_res_for_each, 0,
                        sizeof(uint32_t) * (jc.num_frag_rows + 1),
                        jc.join_stream));
  mem_mgr.allocate_temporal(jc.d_st_array, sizeof(uint32_t) * jc.num_frag_rows);
  cuchk(cudaMemsetAsync(jc.d_st_array, 0xFFFFFFFF,
                        sizeof(uint32_t) * jc.num_frag_rows, jc.join_stream));

  jc.refer_join_without_duplication_context();

  const bool select_mask_on_device = (jc.select_bool_mask_ != nullptr);
  uint32_t *host_select_mask = nullptr;
  if (!select_mask_on_device) {
    const int mask_block_id = jc.frag_trie->id * 2;
    host_select_mask =
        static_cast<uint32_t *>(mem_mgr.memory_blocks[mask_block_id].host_ptr);
    if (host_select_mask == nullptr) {
      std::cerr << "refer_join_without_duplication_pipeline: missing host "
                   "select mask for trie "
                << jc.frag_trie->id << std::endl;
      exit(EXIT_FAILURE);
    }
  }

  const int nStreams = JoinContext::nStreams;
  const size_t num_partial_parents = jc.partial_parent_cols.size();
  const size_t num_partial_data = jc.partial_data_cols.size();

  uint64_t rows_per_stream = jc.num_frag_rows;
  {
    size_t bytes_per_row = num_partial_parents * sizeof(uint64_t) +
                           num_partial_data * sizeof(uint32_t);
    if (!select_mask_on_device)
      bytes_per_row += 1;
    bytes_per_row = std::max<size_t>(bytes_per_row, 1);
    size_t budget_bytes = mem_mgr.stats.available_gpu_memory / 2 / nStreams;
    rows_per_stream = std::max<uint64_t>(budget_bytes / bytes_per_row, 32);
    rows_per_stream = std::min<uint64_t>(rows_per_stream, jc.num_frag_rows);
    if (rows_per_stream >= 32)
      rows_per_stream = (rows_per_stream / 32) * 32;
  }
  if (rows_per_stream == 0)
    rows_per_stream = std::min<uint64_t>(jc.num_frag_rows, 32);

  const uint64_t num_tiles_per_stream = (rows_per_stream + 31) >> 5;
  uint32_t *d_select_mask_tile_base = nullptr;
  if (!select_mask_on_device)
    mem_mgr.allocate_temporal(d_select_mask_tile_base,
                              sizeof(uint32_t) * num_tiles_per_stream *
                                  nStreams);

  uint64_t *d_partial_parent_base = nullptr;
  if (num_partial_parents > 0) {
    mem_mgr.allocate_temporal(d_partial_parent_base,
                              sizeof(uint64_t) * rows_per_stream *
                                  num_partial_parents * nStreams);
  }

  uint32_t *d_partial_data_base = nullptr;
  if (num_partial_data > 0) {
    mem_mgr.allocate_temporal(d_partial_data_base,
                              sizeof(uint32_t) * rows_per_stream *
                                  num_partial_data * nStreams);
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

  if (need_parent_updates) {
    mem_mgr.allocate_temporal(d_parent_info_arrays_base_void,
                              sizeof(JoinInfo<uint64_t>) * parent_info_count *
                                  nStreams);
    d_parent_info_arrays_base =
        static_cast<JoinInfo<uint64_t> *>(d_parent_info_arrays_base_void);
    for (int i = 0; i < nStreams; ++i) {
      h_parent_info_arrays[i] =
          static_cast<JoinInfo<uint64_t> *>(host_mgr.allocateManagedBytes(
              sizeof(JoinInfo<uint64_t>) * parent_info_count,
              alignof(JoinInfo<uint64_t>)));
      if (h_parent_info_arrays[i] == nullptr) {
        std::cerr << "refer_join_without_duplication_pipeline: host allocation "
                     "failed for parent JoinInfo buffers."
                  << std::endl;
        exit(EXIT_FAILURE);
      }
    }
  }

  if (need_data_updates) {
    mem_mgr.allocate_temporal(d_data_info_arrays_base_void,
                              sizeof(JoinInfo<uint32_t>) * data_info_count *
                                  nStreams);
    d_data_info_arrays_base =
        static_cast<JoinInfo<uint32_t> *>(d_data_info_arrays_base_void);
    for (int i = 0; i < nStreams; ++i) {
      h_data_info_arrays[i] =
          static_cast<JoinInfo<uint32_t> *>(host_mgr.allocateManagedBytes(
              sizeof(JoinInfo<uint32_t>) * data_info_count,
              alignof(JoinInfo<uint32_t>)));
      if (h_data_info_arrays[i] == nullptr) {
        std::cerr << "refer_join_without_duplication_pipeline: host allocation "
                     "failed for data JoinInfo buffers."
                  << std::endl;
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
  for (int i = 0; i < nStreams; ++i) {
    cuchk(cudaEventCreateWithFlags(&done_events[i], cudaEventDisableTiming));
    cuchk(cudaEventRecord(done_events[i], jc.pipeline_streams[i]));
  }

  std::vector<ColumnRange> ranges;
  uint64_t cur_offset = 0;
  while (cur_offset < jc.num_frag_rows) {
    for (int stream_id = 0;
         stream_id < nStreams && cur_offset < jc.num_frag_rows; ++stream_id) {
      cuchk(cudaEventSynchronize(done_events[stream_id]));

      uint64_t num_rows_this_iter =
          std::min(rows_per_stream, jc.num_frag_rows - cur_offset);
      uint64_t num_tiles_this_iter = (num_rows_this_iter + 31) >> 5;
      cudaStream_t stream = jc.pipeline_streams[stream_id];

      uint32_t *d_select_mask_ptr = nullptr;
      if (select_mask_on_device) {
        d_select_mask_ptr = jc.select_bool_mask_ + (cur_offset >> 5);
      } else {
        d_select_mask_ptr =
            d_select_mask_tile_base +
            num_tiles_per_stream * static_cast<size_t>(stream_id);
        cuchk(cudaMemcpyAsync(d_select_mask_ptr,
                              host_select_mask + (cur_offset >> 5),
                              sizeof(uint32_t) * num_tiles_this_iter,
                              cudaMemcpyHostToDevice, stream));
      }

      compute_column_ranges(jc.root_trie, jc.up_to_col, cur_offset,
                            cur_offset + num_rows_this_iter, ranges,
                            jc.join_stream);

      JoinInfo<uint64_t> *d_parent_info_ptr = jc.d_parent_info_arrays_;
      if (need_parent_updates) {
        JoinInfo<uint64_t> *h_parent_infos = h_parent_info_arrays[stream_id];
        for (size_t col = 0; col < parent_info_count; ++col)
          h_parent_infos[col] = jc.h_parent_info_arrays_[col];

        for (size_t i = 0; i < num_partial_parents; ++i) {
          uint32_t col = jc.partial_parent_cols[i];
          uint32_t col_trie_id = jc.root_trie->column_ids[col];
          const int block_idx = static_cast<int>(col_trie_id) * 2 + 1;
          auto &block = mem_mgr.memory_blocks[block_idx];
          if (block.host_ptr == nullptr) {
            std::cerr << "refer_join_without_duplication_pipeline: missing "
                         "host parents for trie "
                      << col_trie_id << std::endl;
            exit(EXIT_FAILURE);
          }
          auto *h_parents = static_cast<uint64_t *>(block.host_ptr);
          ColumnRange r = ranges[col];
          uint64_t len = r.end - r.start;

          uint64_t *d_parent_buf =
              d_partial_parent_base +
              (static_cast<size_t>(stream_id) * num_partial_parents + i) *
                  rows_per_stream;
          if (len > 0) {
            cuchk(cudaMemcpyAsync(d_parent_buf, h_parents + r.start,
                                  sizeof(uint64_t) * len,
                                  cudaMemcpyHostToDevice, stream));
            h_parent_infos[col].array_ptr = d_parent_buf;
            h_parent_infos[col].offset = r.start;
            h_parent_infos[col].is_partial = 1;
          } else {
            h_parent_infos[col].array_ptr = nullptr;
            h_parent_infos[col].offset = 0;
            h_parent_infos[col].is_partial = 1;
          }
        }

        d_parent_info_ptr = d_parent_info_arrays_base +
                            parent_info_count * static_cast<size_t>(stream_id);
        cuchk(cudaMemcpyAsync(d_parent_info_ptr, h_parent_infos,
                              sizeof(JoinInfo<uint64_t>) * parent_info_count,
                              cudaMemcpyHostToDevice, stream));
      }

      JoinInfo<uint32_t> *d_data_info_ptr = jc.d_data_info_arrays_;
      if (need_data_updates) {
        JoinInfo<uint32_t> *h_data_infos = h_data_info_arrays[stream_id];
        for (size_t col = 0; col < data_info_count; ++col)
          h_data_infos[col] = jc.h_data_info_arrays_[col];

        for (size_t i = 0; i < num_partial_data; ++i) {
          uint32_t col = jc.partial_data_cols[i];
          uint32_t col_trie_id = jc.root_trie->column_ids[col];
          const int block_idx = static_cast<int>(col_trie_id) * 2;
          auto &block = mem_mgr.memory_blocks[block_idx];
          if (block.host_ptr == nullptr) {
            std::cerr << "refer_join_without_duplication_pipeline: missing "
                         "host data for trie "
                      << col_trie_id << std::endl;
            exit(EXIT_FAILURE);
          }
          auto *h_data = static_cast<uint32_t *>(block.host_ptr);
          ColumnRange r = ranges[col];
          uint64_t len = r.end - r.start;

          uint32_t *d_data_buf =
              d_partial_data_base +
              (static_cast<size_t>(stream_id) * num_partial_data + i) *
                  rows_per_stream;
          if (len > 0) {
            cuchk(cudaMemcpyAsync(d_data_buf, h_data + r.start,
                                  sizeof(uint32_t) * len,
                                  cudaMemcpyHostToDevice, stream));
            h_data_infos[col].array_ptr = d_data_buf;
            h_data_infos[col].offset = r.start;
            h_data_infos[col].is_partial = 1;
          } else {
            h_data_infos[col].array_ptr = nullptr;
            h_data_infos[col].offset = 0;
            h_data_infos[col].is_partial = 1;
          }
        }

        d_data_info_ptr = d_data_info_arrays_base +
                          data_info_count * static_cast<size_t>(stream_id);
        cuchk(cudaMemcpyAsync(d_data_info_ptr, h_data_infos,
                              sizeof(JoinInfo<uint32_t>) * data_info_count,
                              cudaMemcpyHostToDevice, stream));
      }

      join_count_refer_nodup_pipeline_dispatcher(
          d_data_info_ptr, d_parent_info_ptr, d_select_mask_ptr, cur_offset,
          num_rows_this_iter, jc.root_trie->num_cols, jc.col_key,
          jc.d_edge_keys_, jc.d_edge_values_, jc.d_edge_offsets_,
          ht.h_num_buckets_[jc.directed_edge_idx],
          ht.h_hash_table_offs_[jc.directed_edge_idx], jc.directed_edge_idx,
          jc.d_num_res_for_each, jc.d_st_array, jc.task_per_warp, stream);

      cuchk(cudaEventRecord(done_events[stream_id], stream));
      cur_offset += num_rows_this_iter;
    }
  }

  for (int i = 0; i < nStreams; ++i) {
    cuchk(cudaEventSynchronize(done_events[i]));
    cuchk(cudaEventDestroy(done_events[i]));
  }

  if (need_parent_updates) {
    for (int i = 0; i < nStreams; ++i) {
      void *tmp = h_parent_info_arrays[i];
      host_mgr.deallocateManaged(tmp);
      h_parent_info_arrays[i] = nullptr;
    }
    mem_mgr.deallocate_temporal(d_parent_info_arrays_base_void);
  }
  if (need_data_updates) {
    for (int i = 0; i < nStreams; ++i) {
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
  if (d_select_mask_tile_base != nullptr)
    mem_mgr.deallocate_temporal(d_select_mask_tile_base);

  mem_mgr.unpin_climb_data();
  mem_mgr.unpin_climb_parents();

  // Manually allocate memory to use the memory pool
  uint64_t *d_indices_raw;
  mem_mgr.allocate_temporal(d_indices_raw, sizeof(uint64_t) * jc.num_frag_rows);
  thrust::device_ptr<uint64_t> d_indices(d_indices_raw);
  thrust::sequence(thrust::cuda::par.on(jc.join_stream), d_indices,
                   d_indices + jc.num_frag_rows);

  uint64_t *d_selected_row_ids;
  uint64_t *d_num_selected;
  mem_mgr.allocate_temporal(d_selected_row_ids,
                            sizeof(uint64_t) * jc.num_frag_rows);
  mem_mgr.allocate_temporal(d_num_selected, sizeof(uint64_t));

  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;

  CountPositive pred{jc.d_num_res_for_each};

  cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, d_indices_raw,
                        d_selected_row_ids, d_num_selected, jc.num_frag_rows,
                        pred, jc.join_stream);
  mem_mgr.allocate_temporal(d_temp_storage, temp_storage_bytes);
  cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, d_indices_raw,
                        d_selected_row_ids, d_num_selected, jc.num_frag_rows,
                        pred, jc.join_stream);

  mem_mgr.deallocate_temporal(d_temp_storage);
  mem_mgr.deallocate_temporal(d_indices_raw);

  uint64_t h_num_selected = 0;
  cuchk(cudaMemcpyAsync(&h_num_selected, d_num_selected, sizeof(uint64_t),
                        cudaMemcpyDeviceToHost, jc.join_stream));
  cuchk(cudaStreamSynchronize(jc.join_stream));

  uint64_t *d_row_offsets = nullptr;
  mem_mgr.allocate_temporal(d_row_offsets,
                            sizeof(uint64_t) * (h_num_selected + 1));
  cuchk(cudaMemsetAsync(d_row_offsets, 0,
                        sizeof(uint64_t) * (h_num_selected + 1),
                        jc.join_stream));

  uint32_t *d_selected_count;
  mem_mgr.allocate_temporal(d_selected_count,
                            sizeof(uint32_t) * (h_num_selected + 1));
  cuchk(cudaMemsetAsync(d_selected_count + h_num_selected, 0, sizeof(uint32_t),
                        jc.join_stream));
  mem_mgr.deallocate_temporal(d_num_selected);

  // Gather counts for selected rows
  gather_counts_kernel<<<GRID_DIM, BLOCK_DIM, 0, jc.join_stream>>>(
      jc.d_num_res_for_each, d_selected_row_ids, h_num_selected,
      d_selected_count);
  mem_mgr.deallocate_temporal(jc.d_num_res_for_each);
  d_temp_storage = nullptr;

  cub::DeviceScan::ExclusiveScan(
      d_temp_storage, temp_storage_bytes, d_selected_count, d_row_offsets,
      cub::Sum(), (uint64_t)0, h_num_selected + 1, jc.join_stream);
  mem_mgr.allocate_temporal(d_temp_storage, temp_storage_bytes);
  cub::DeviceScan::ExclusiveScan(
      d_temp_storage, temp_storage_bytes, d_selected_count, d_row_offsets,
      cub::Sum(), (uint64_t)0, h_num_selected + 1, jc.join_stream);
  mem_mgr.deallocate_temporal(d_temp_storage);
  cuchk(cudaMemcpyAsync(jc.h_num_res, d_row_offsets + h_num_selected,
                        sizeof(uint64_t), cudaMemcpyDeviceToHost,
                        jc.join_stream));
  cuchk(cudaStreamSynchronize(jc.join_stream));
  // Allocate result memory
  resetTargetTrieData(jc.target_trie, *jc.h_num_res, false);
  if (jc.skip_final_materialization == false) {
    bool allocate_success = mem_mgr.allocate_trie(jc.target_trie->id);

    if (allocate_success) {
      join_write_refer_no_dup_dispatcher(
          d_selected_row_ids, h_num_selected, jc.d_edge_values_,
          d_selected_count, jc.d_st_array, jc.target_trie->data->d_data,
          d_row_offsets, jc.target_trie->data->d_parents, jc.task_per_warp);
    } else {
      join_refer_write_nodup_pipeline(d_selected_row_ids, h_num_selected,
                                      d_selected_count, d_row_offsets);
    }
  }

  if (d_row_offsets)
    mem_mgr.deallocate_temporal(d_row_offsets);
  mem_mgr.deallocate_temporal(d_selected_row_ids);
  mem_mgr.deallocate_temporal(d_selected_count);
  mem_mgr.deallocate_temporal(jc.d_st_array);
}

void refer_join_with_duplication_pipeline() {
  std::cout << "Entering refer join with duplication pipeline." << std::endl;

  auto &mem_mgr = MemoryManager::getInstance();
  auto &ht = HashTable::getInstance();
  auto &jc = JoinContext::getInstance();
  auto &host_mgr = HostMemoryManager::getInstance();

  get_data_arrays_pipeline(jc.root_trie->id, jc.pin_cols,
                           jc.h_data_info_arrays_);
  get_parent_arrays_pipeline(jc.root_trie->id, jc.up_to_col,
                             jc.h_parent_info_arrays_);

  cuchk(cudaMemcpyAsync(jc.d_data_info_arrays_, jc.h_data_info_arrays_.data(),
                        sizeof(JoinInfo<uint32_t>) *
                            jc.h_data_info_arrays_.size(),
                        cudaMemcpyHostToDevice, jc.join_stream));
  cuchk(cudaMemcpyAsync(
      jc.d_parent_info_arrays_, jc.h_parent_info_arrays_.data(),
      sizeof(JoinInfo<uint64_t>) * jc.h_parent_info_arrays_.size(),
      cudaMemcpyHostToDevice, jc.join_stream));

  if (!mem_mgr.allocate_temporal(jc.d_num_res_for_each,
                                 sizeof(uint32_t) * (jc.num_frag_rows + 1))) {
    std::cerr << "refer_join_with_duplication_pipeline: failed to allocate "
                 "d_num_res_for_each."
              << std::endl;
    exit(EXIT_FAILURE);
  }
  cuchk(cudaMemsetAsync(jc.d_num_res_for_each, 0,
                        sizeof(uint32_t) * (jc.num_frag_rows + 1),
                        jc.join_stream));
  if (!mem_mgr.allocate_temporal(jc.d_st_array,
                                 sizeof(uint32_t) * jc.num_frag_rows)) {
    mem_mgr.deallocate_temporal(jc.d_num_res_for_each);
    std::cerr << "refer_join_with_duplication_pipeline: failed to allocate "
                 "d_st_array."
              << std::endl;
    exit(EXIT_FAILURE);
  }
  cuchk(cudaMemsetAsync(jc.d_st_array, 0xFFFFFFFF,
                        sizeof(uint32_t) * jc.num_frag_rows, jc.join_stream));

  jc.refer_join_with_duplication_context();
  if (jc.d_mask == nullptr) {
    std::cerr << "refer_join_with_duplication_pipeline: insufficient GPU "
                 "memory for duplication mask."
              << std::endl;
    exit(EXIT_FAILURE);
  }

  const bool select_mask_on_device = (jc.select_bool_mask_ != nullptr);
  uint32_t *host_select_mask = nullptr;
  if (!select_mask_on_device) {
    const int mask_block_id = jc.frag_trie->id * 2;
    host_select_mask =
        static_cast<uint32_t *>(mem_mgr.memory_blocks[mask_block_id].host_ptr);
    if (host_select_mask == nullptr) {
      std::cerr << "refer_join_with_duplication_pipeline: missing host select "
                   "mask for trie "
                << jc.frag_trie->id << std::endl;
      exit(EXIT_FAILURE);
    }
  }

  const int nStreams = JoinContext::nStreams;
  const size_t num_partial_parents = jc.partial_parent_cols.size();
  const size_t num_partial_data = jc.partial_data_cols.size();

  uint64_t rows_per_stream = jc.num_frag_rows;
  {
    size_t bytes_per_row = num_partial_parents * sizeof(uint64_t) +
                           num_partial_data * sizeof(uint32_t);
    if (!select_mask_on_device)
      bytes_per_row += 1;
    bytes_per_row = std::max<size_t>(bytes_per_row, 1);
    size_t budget_bytes = mem_mgr.stats.available_gpu_memory / 2 / nStreams;
    rows_per_stream = std::max<uint64_t>(budget_bytes / bytes_per_row, 32);
    rows_per_stream = std::min<uint64_t>(rows_per_stream, jc.num_frag_rows);
    if (rows_per_stream >= 32)
      rows_per_stream = (rows_per_stream / 32) * 32;
  }
  if (rows_per_stream == 0)
    rows_per_stream = std::min<uint64_t>(jc.num_frag_rows, 32);

  const uint64_t num_tiles_per_stream = (rows_per_stream + 31) >> 5;
  uint32_t *d_select_mask_tile_base = nullptr;
  if (!select_mask_on_device)
    mem_mgr.allocate_temporal(d_select_mask_tile_base,
                              sizeof(uint32_t) * num_tiles_per_stream *
                                  nStreams);

  uint64_t *d_partial_parent_base = nullptr;
  if (num_partial_parents > 0) {
    mem_mgr.allocate_temporal(d_partial_parent_base,
                              sizeof(uint64_t) * rows_per_stream *
                                  num_partial_parents * nStreams);
  }

  uint32_t *d_partial_data_base = nullptr;
  if (num_partial_data > 0) {
    mem_mgr.allocate_temporal(d_partial_data_base,
                              sizeof(uint32_t) * rows_per_stream *
                                  num_partial_data * nStreams);
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

  if (need_parent_updates) {
    mem_mgr.allocate_temporal(d_parent_info_arrays_base_void,
                              sizeof(JoinInfo<uint64_t>) * parent_info_count *
                                  nStreams);
    d_parent_info_arrays_base =
        static_cast<JoinInfo<uint64_t> *>(d_parent_info_arrays_base_void);
    for (int i = 0; i < nStreams; ++i) {
      h_parent_info_arrays[i] =
          static_cast<JoinInfo<uint64_t> *>(host_mgr.allocateManagedBytes(
              sizeof(JoinInfo<uint64_t>) * parent_info_count,
              alignof(JoinInfo<uint64_t>)));
      if (h_parent_info_arrays[i] == nullptr) {
        std::cerr << "refer_join_with_duplication_pipeline: host allocation "
                     "failed for parent JoinInfo buffers."
                  << std::endl;
        exit(EXIT_FAILURE);
      }
    }
  }

  if (need_data_updates) {
    mem_mgr.allocate_temporal(d_data_info_arrays_base_void,
                              sizeof(JoinInfo<uint32_t>) * data_info_count *
                                  nStreams);
    d_data_info_arrays_base =
        static_cast<JoinInfo<uint32_t> *>(d_data_info_arrays_base_void);
    for (int i = 0; i < nStreams; ++i) {
      h_data_info_arrays[i] =
          static_cast<JoinInfo<uint32_t> *>(host_mgr.allocateManagedBytes(
              sizeof(JoinInfo<uint32_t>) * data_info_count,
              alignof(JoinInfo<uint32_t>)));
      if (h_data_info_arrays[i] == nullptr) {
        std::cerr << "refer_join_with_duplication_pipeline: host allocation "
                     "failed for data JoinInfo buffers."
                  << std::endl;
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
  for (int i = 0; i < nStreams; ++i) {
    cuchk(cudaEventCreateWithFlags(&done_events[i], cudaEventDisableTiming));
    cuchk(cudaEventRecord(done_events[i], jc.pipeline_streams[i]));
  }

  std::vector<ColumnRange> ranges;
  uint64_t cur_offset = 0;
  while (cur_offset < jc.num_frag_rows) {
    for (int stream_id = 0;
         stream_id < nStreams && cur_offset < jc.num_frag_rows; ++stream_id) {
      cuchk(cudaEventSynchronize(done_events[stream_id]));

      uint64_t num_rows_this_iter =
          std::min(rows_per_stream, jc.num_frag_rows - cur_offset);
      uint64_t num_tiles_this_iter = (num_rows_this_iter + 31) >> 5;
      cudaStream_t stream = jc.pipeline_streams[stream_id];

      uint32_t *d_select_mask_ptr = nullptr;
      if (select_mask_on_device) {
        d_select_mask_ptr = jc.select_bool_mask_ + (cur_offset >> 5);
      } else {
        d_select_mask_ptr =
            d_select_mask_tile_base +
            num_tiles_per_stream * static_cast<size_t>(stream_id);
        cuchk(cudaMemcpyAsync(d_select_mask_ptr,
                              host_select_mask + (cur_offset >> 5),
                              sizeof(uint32_t) * num_tiles_this_iter,
                              cudaMemcpyHostToDevice, stream));
      }

      compute_column_ranges(jc.root_trie, jc.up_to_col, cur_offset,
                            cur_offset + num_rows_this_iter, ranges,
                            jc.join_stream);

      JoinInfo<uint64_t> *d_parent_info_ptr = jc.d_parent_info_arrays_;
      if (need_parent_updates) {
        JoinInfo<uint64_t> *h_parent_infos = h_parent_info_arrays[stream_id];
        for (size_t col = 0; col < parent_info_count; ++col)
          h_parent_infos[col] = jc.h_parent_info_arrays_[col];

        for (size_t i = 0; i < num_partial_parents; ++i) {
          uint32_t col = jc.partial_parent_cols[i];
          uint32_t col_trie_id = jc.root_trie->column_ids[col];
          const int block_idx = static_cast<int>(col_trie_id) * 2 + 1;
          auto &block = mem_mgr.memory_blocks[block_idx];
          if (block.host_ptr == nullptr) {
            std::cerr << "refer_join_with_duplication_pipeline: missing host "
                         "parents for trie "
                      << col_trie_id << std::endl;
            exit(EXIT_FAILURE);
          }
          auto *h_parents = static_cast<uint64_t *>(block.host_ptr);
          ColumnRange r = ranges[col];
          uint64_t len = r.end - r.start;

          uint64_t *d_parent_buf =
              d_partial_parent_base +
              (static_cast<size_t>(stream_id) * num_partial_parents + i) *
                  rows_per_stream;
          if (len > 0) {
            cuchk(cudaMemcpyAsync(d_parent_buf, h_parents + r.start,
                                  sizeof(uint64_t) * len,
                                  cudaMemcpyHostToDevice, stream));
            h_parent_infos[col].array_ptr = d_parent_buf;
            h_parent_infos[col].offset = r.start;
            h_parent_infos[col].is_partial = 1;
          } else {
            h_parent_infos[col].array_ptr = nullptr;
            h_parent_infos[col].offset = 0;
            h_parent_infos[col].is_partial = 1;
          }
        }

        d_parent_info_ptr = d_parent_info_arrays_base +
                            parent_info_count * static_cast<size_t>(stream_id);
        cuchk(cudaMemcpyAsync(d_parent_info_ptr, h_parent_infos,
                              sizeof(JoinInfo<uint64_t>) * parent_info_count,
                              cudaMemcpyHostToDevice, stream));
      }

      JoinInfo<uint32_t> *d_data_info_ptr = jc.d_data_info_arrays_;
      if (need_data_updates) {
        JoinInfo<uint32_t> *h_data_infos = h_data_info_arrays[stream_id];
        for (size_t col = 0; col < data_info_count; ++col)
          h_data_infos[col] = jc.h_data_info_arrays_[col];

        for (size_t i = 0; i < num_partial_data; ++i) {
          uint32_t col = jc.partial_data_cols[i];
          uint32_t col_trie_id = jc.root_trie->column_ids[col];
          const int block_idx = static_cast<int>(col_trie_id) * 2;
          auto &block = mem_mgr.memory_blocks[block_idx];
          if (block.host_ptr == nullptr) {
            std::cerr << "refer_join_with_duplication_pipeline: missing host "
                         "data for trie "
                      << col_trie_id << std::endl;
            exit(EXIT_FAILURE);
          }
          auto *h_data = static_cast<uint32_t *>(block.host_ptr);
          ColumnRange r = ranges[col];
          uint64_t len = r.end - r.start;

          uint32_t *d_data_buf =
              d_partial_data_base +
              (static_cast<size_t>(stream_id) * num_partial_data + i) *
                  rows_per_stream;
          if (len > 0) {
            cuchk(cudaMemcpyAsync(d_data_buf, h_data + r.start,
                                  sizeof(uint32_t) * len,
                                  cudaMemcpyHostToDevice, stream));
            h_data_infos[col].array_ptr = d_data_buf;
            h_data_infos[col].offset = r.start;
            h_data_infos[col].is_partial = 1;
          } else {
            h_data_infos[col].array_ptr = nullptr;
            h_data_infos[col].offset = 0;
            h_data_infos[col].is_partial = 1;
          }
        }

        d_data_info_ptr = d_data_info_arrays_base +
                          data_info_count * static_cast<size_t>(stream_id);
        cuchk(cudaMemcpyAsync(d_data_info_ptr, h_data_infos,
                              sizeof(JoinInfo<uint32_t>) * data_info_count,
                              cudaMemcpyHostToDevice, stream));
      }

      join_refer_count_dup_kernel_2_pipeline_info<<<GRID_DIM, BLOCK_DIM, 0,
                                                    stream>>>(
          d_data_info_ptr, d_parent_info_ptr, d_select_mask_ptr, cur_offset,
          num_rows_this_iter, jc.root_trie->num_cols, jc.col_key,
          jc.d_edge_values_, jc.d_edge_offsets_,
          ht.h_num_buckets_[jc.directed_edge_idx],
          ht.h_hash_table_offs_[jc.directed_edge_idx], jc.directed_edge_idx,
          jc.d_mask, jc.mask_length, jc.d_st_array, jc.dup_check_type,
          jc.num_dup_cols, jc.d_dup_cols, jc.up_to_col);

      cuchk(cudaEventRecord(done_events[stream_id], stream));
      cur_offset += num_rows_this_iter;
    }
  }

  for (int i = 0; i < nStreams; ++i) {
    cuchk(cudaEventSynchronize(done_events[i]));
    cuchk(cudaEventDestroy(done_events[i]));
  }

  if (need_parent_updates) {
    for (int i = 0; i < nStreams; ++i) {
      void *tmp = h_parent_info_arrays[i];
      host_mgr.deallocateManaged(tmp);
      h_parent_info_arrays[i] = nullptr;
    }
    mem_mgr.deallocate_temporal(d_parent_info_arrays_base_void);
  }
  if (need_data_updates) {
    for (int i = 0; i < nStreams; ++i) {
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
  if (d_select_mask_tile_base != nullptr)
    mem_mgr.deallocate_temporal(d_select_mask_tile_base);

  mem_mgr.unpin_climb_data();
  mem_mgr.unpin_climb_parents();

  uint64_t *d_row_offsets = nullptr;
  mem_mgr.allocate_temporal(d_row_offsets,
                            sizeof(uint64_t) * (jc.num_frag_rows + 1));

  size_t temp_storage_bytes = 0;
  void *d_temp_storage = nullptr;

  uint64_t *d_segment_offsets = nullptr;
  mem_mgr.allocate_temporal(d_segment_offsets,
                            sizeof(uint64_t) * (jc.num_frag_rows + 1));
  thrust::device_ptr<uint64_t> offsets_ptr(d_segment_offsets);
  thrust::sequence(thrust::cuda::par.on(jc.join_stream), offsets_ptr,
                   offsets_ptr + jc.num_frag_rows + 1, static_cast<uint64_t>(0),
                   static_cast<uint64_t>(jc.mask_length));
  auto popcount_iterator =
      thrust::make_transform_iterator(jc.d_mask, PopcToCount{});
  cuchk(cub::DeviceSegmentedReduce::Sum(
      d_temp_storage, temp_storage_bytes, popcount_iterator,
      jc.d_num_res_for_each, jc.num_frag_rows, d_segment_offsets,
      d_segment_offsets + 1, jc.join_stream));
  mem_mgr.allocate_temporal(d_temp_storage, temp_storage_bytes);
  cuchk(cub::DeviceSegmentedReduce::Sum(
      d_temp_storage, temp_storage_bytes, popcount_iterator,
      jc.d_num_res_for_each, jc.num_frag_rows, d_segment_offsets,
      d_segment_offsets + 1, jc.join_stream));
  mem_mgr.deallocate_temporal(d_segment_offsets);
  mem_mgr.deallocate_temporal(d_temp_storage);
  d_temp_storage = nullptr;
  temp_storage_bytes = 0;

  cub::DeviceScan::ExclusiveScan(
      d_temp_storage, temp_storage_bytes, jc.d_num_res_for_each, d_row_offsets,
      cub::Sum(), (uint64_t)0, jc.num_frag_rows + 1, jc.join_stream);
  mem_mgr.allocate_temporal(d_temp_storage, temp_storage_bytes);
  cub::DeviceScan::ExclusiveScan(
      d_temp_storage, temp_storage_bytes, jc.d_num_res_for_each, d_row_offsets,
      cub::Sum(), (uint64_t)0, jc.num_frag_rows + 1, jc.join_stream);
  mem_mgr.deallocate_temporal(jc.d_num_res_for_each);
  cuchk(cudaMemcpyAsync(jc.h_num_res, d_row_offsets + jc.num_frag_rows,
                        sizeof(uint64_t), cudaMemcpyDeviceToHost,
                        jc.join_stream));
  mem_mgr.deallocate_temporal(d_temp_storage);
  d_temp_storage = nullptr;
  temp_storage_bytes = 0;

  cuchk(cudaStreamSynchronize(jc.join_stream));
  resetTargetTrieData(jc.target_trie, *jc.h_num_res, false);
  if (jc.skip_final_materialization == false) {
    bool allocate_success = mem_mgr.allocate_trie(jc.target_trie->id);

    if (allocate_success) {
      join_refer_write_dup_kernel<<<GRID_DIM, BLOCK_DIM, 0, jc.join_stream>>>(
          nullptr, nullptr, jc.root_trie->num_layers, jc.num_frag_rows,
          jc.root_trie->num_cols, jc.col_key, jc.d_edge_keys_,
          jc.d_edge_values_, jc.d_edge_offsets_,
          ht.h_num_buckets_[jc.directed_edge_idx],
          ht.h_hash_table_offs_[jc.directed_edge_idx], jc.directed_edge_idx,
          jc.d_mask, jc.mask_length, jc.d_st_array,
          jc.target_trie->data->d_data, d_row_offsets,
          jc.target_trie->data->d_parents, jc.dup_check_type, jc.num_dup_cols,
          jc.d_dup_cols);
      cuchk(cudaStreamSynchronize(jc.join_stream));
    } else {
      join_refer_write_dup_pipeline(d_row_offsets);
    }
  }

  if (d_row_offsets)
    mem_mgr.deallocate_temporal(d_row_offsets);
  mem_mgr.deallocate_temporal(jc.d_mask);
  mem_mgr.deallocate_temporal(jc.d_st_array);
}

void refer_join_without_duplication() {
  // std::cout << "Performing Refer Join without duplication check." <<
  // std::endl;
#ifndef NDEBUG
  std::cout << "Performing Refer Join without duplication check." << std::endl;
#endif

  auto lat = &Lattice::getInstance();
  auto utm = &UnifiedTrieManager::getInstance();
  auto mem_mgr = &MemoryManager::getInstance();
  auto ht = &HashTable::getInstance();
  auto &jc = JoinContext::getInstance();

  jc.refer_join_without_duplication_context();

  uint64_t *d_indices_raw = nullptr;
  uint64_t *d_selected_row_ids = nullptr;
  uint64_t *d_num_selected = nullptr;
  void *d_temp_storage = nullptr;
  uint64_t *d_row_offsets = nullptr;
  uint32_t *d_selected_count = nullptr;

  auto cleanupOnFailure = [&]() noexcept {
    try {
      if (d_temp_storage != nullptr)
        mem_mgr->deallocate_temporal(d_temp_storage);
    } catch (...) {
    }
    try {
      if (d_indices_raw != nullptr)
        mem_mgr->deallocate_temporal(d_indices_raw);
    } catch (...) {
    }
    try {
      if (d_row_offsets != nullptr)
        mem_mgr->deallocate_temporal(d_row_offsets);
    } catch (...) {
    }
    try {
      if (d_selected_row_ids != nullptr)
        mem_mgr->deallocate_temporal(d_selected_row_ids);
    } catch (...) {
    }
    try {
      if (d_selected_count != nullptr)
        mem_mgr->deallocate_temporal(d_selected_count);
    } catch (...) {
    }
    try {
      if (d_num_selected != nullptr)
        mem_mgr->deallocate_temporal(d_num_selected);
    } catch (...) {
    }
    try {
      if (jc.d_num_res_for_each != nullptr)
        mem_mgr->deallocate_temporal(jc.d_num_res_for_each);
    } catch (...) {
    }
    try {
      if (jc.d_st_array != nullptr)
        mem_mgr->deallocate_temporal(jc.d_st_array);
    } catch (...) {
    }
  };

  try {
    join_count_refer_no_dup_dispatcher(
        // trie_inter
        jc.d_data_arrays_, jc.d_parent_arrays_, jc.num_frag_rows,
        jc.root_trie->num_cols, jc.col_key, jc.select_bool_mask_,
        // trie_edge
        jc.d_edge_keys_, jc.d_edge_values_, jc.d_edge_offsets_,
        // hash
        ht->h_num_buckets_[jc.directed_edge_idx],
        ht->h_hash_table_offs_[jc.directed_edge_idx], jc.directed_edge_idx,
        // result
        jc.d_num_res_for_each, jc.d_st_array, jc.task_per_warp);

    mem_mgr->unpin_climb_data();
    mem_mgr->unpin_climb_parents();
    // Manually allocate memory to use the memory pool
    if (!mem_mgr->allocate_temporal(d_indices_raw,
                                    sizeof(uint64_t) * jc.num_frag_rows)) {
      throw std::runtime_error(
          "refer_join_without_duplication: failed to allocate indices buffer.");
    }
    thrust::device_ptr<uint64_t> d_indices(d_indices_raw);
    thrust::sequence(thrust::cuda::par.on(jc.join_stream), d_indices,
                     d_indices + jc.num_frag_rows);

    if (!mem_mgr->allocate_temporal(d_selected_row_ids,
                                    sizeof(uint64_t) * jc.num_frag_rows)) {
      throw std::runtime_error("refer_join_without_duplication: failed to "
                               "allocate selected row ids buffer.");
    }
    if (!mem_mgr->allocate_temporal(d_num_selected, sizeof(uint64_t))) {
      throw std::runtime_error("refer_join_without_duplication: failed to "
                               "allocate selected row count buffer.");
    }

    size_t temp_storage_bytes = 0;

    CountPositive pred{jc.d_num_res_for_each};

    cuchk(cub::DeviceSelect::If(
        d_temp_storage, temp_storage_bytes, d_indices_raw, d_selected_row_ids,
        d_num_selected, jc.num_frag_rows, pred, jc.join_stream));
    if (!mem_mgr->allocate_temporal(d_temp_storage, temp_storage_bytes)) {
      throw std::runtime_error("refer_join_without_duplication: failed to "
                               "allocate device select temp storage.");
    }
    cuchk(cub::DeviceSelect::If(
        d_temp_storage, temp_storage_bytes, d_indices_raw, d_selected_row_ids,
        d_num_selected, jc.num_frag_rows, pred, jc.join_stream));

    mem_mgr->deallocate_temporal(d_temp_storage);
    mem_mgr->deallocate_temporal(d_indices_raw);

    uint64_t h_num_selected = 0;
    cuchk(cudaMemcpyAsync(&h_num_selected, d_num_selected, sizeof(uint64_t),
                          cudaMemcpyDeviceToHost, jc.join_stream));
    cuchk(cudaStreamSynchronize(jc.join_stream));

    if (!mem_mgr->allocate_temporal(d_row_offsets,
                                    sizeof(uint64_t) * (h_num_selected + 1))) {
      throw std::runtime_error("refer_join_without_duplication: failed to "
                               "allocate row offsets buffer.");
    }
    cuchk(cudaMemsetAsync(d_row_offsets, 0,
                          sizeof(uint64_t) * (h_num_selected + 1),
                          jc.join_stream));

    if (!mem_mgr->allocate_temporal(d_selected_count,
                                    sizeof(uint32_t) * (h_num_selected + 1))) {
      throw std::runtime_error("refer_join_without_duplication: failed to "
                               "allocate selected count buffer.");
    }
    cuchk(cudaMemsetAsync(d_selected_count + h_num_selected, 0,
                          sizeof(uint32_t), jc.join_stream));
    mem_mgr->deallocate_temporal(d_num_selected);

    // Gather counts for selected rows
    gather_counts_kernel<<<GRID_DIM, BLOCK_DIM, 0, jc.join_stream>>>(
        jc.d_num_res_for_each, d_selected_row_ids, h_num_selected,
        d_selected_count);
    cuchk(cudaGetLastError());
    mem_mgr->deallocate_temporal(jc.d_num_res_for_each);
    d_temp_storage = nullptr;

    cuchk(cub::DeviceScan::ExclusiveScan(d_temp_storage, temp_storage_bytes,
                                         d_selected_count, d_row_offsets,
                                         cub::Sum(), static_cast<uint64_t>(0),
                                         h_num_selected + 1, jc.join_stream));
    if (!mem_mgr->allocate_temporal(d_temp_storage, temp_storage_bytes)) {
      throw std::runtime_error("refer_join_without_duplication: failed to "
                               "allocate scan temp storage.");
    }
    cuchk(cub::DeviceScan::ExclusiveScan(d_temp_storage, temp_storage_bytes,
                                         d_selected_count, d_row_offsets,
                                         cub::Sum(), static_cast<uint64_t>(0),
                                         h_num_selected + 1, jc.join_stream));
    mem_mgr->deallocate_temporal(d_temp_storage);
    cuchk(cudaMemcpyAsync(jc.h_num_res, d_row_offsets + h_num_selected,
                          sizeof(uint64_t), cudaMemcpyDeviceToHost,
                          jc.join_stream));
    cuchk(cudaStreamSynchronize(jc.join_stream));
    // Allocate result memory
    resetTargetTrieData(jc.target_trie, *jc.h_num_res, false);
    if (jc.skip_final_materialization == false) {
      bool allocate_success = mem_mgr->allocate_trie(jc.target_trie->id);
      // target_trie->data->allocate();

      if (allocate_success) {
        join_write_refer_no_dup_dispatcher(
            d_selected_row_ids, h_num_selected, jc.d_edge_values_,
            d_selected_count, jc.d_st_array, jc.target_trie->data->d_data,
            d_row_offsets, jc.target_trie->data->d_parents, jc.task_per_warp);
      } else // if (!allocate_success)
      {
        join_refer_write_nodup_pipeline(d_selected_row_ids, h_num_selected,
                                        d_selected_count, d_row_offsets);
      }
    }

    if (d_row_offsets)
      mem_mgr->deallocate_temporal(d_row_offsets);
    mem_mgr->deallocate_temporal(d_selected_row_ids);
    mem_mgr->deallocate_temporal(d_selected_count);
    mem_mgr->deallocate_temporal(jc.d_st_array);
  } catch (...) {
    cleanupOnFailure();
    throw;
  }
}
void refer_join_without_duplication_debug() {
  refer_join_without_duplication();
}

void refer_join_with_duplication() {
  // std::cout << "Performing Refer Join with duplication check." << std::endl;
#ifndef NDEBUG
  std::cout << "Performing Refer Join with duplication check." << std::endl;
#endif

  auto &lat = Lattice::getInstance();
  auto &utm = UnifiedTrieManager::getInstance();
  auto &mem_mgr = MemoryManager::getInstance();
  auto &ht = HashTable::getInstance();
  auto &jc = JoinContext::getInstance();

  jc.refer_join_with_duplication_context();

  if (jc.d_mask == nullptr) {
    std::cout << "Falling back to on-the-fly duplication check (no value mask)."
              << std::endl;

    join_refer_count_dup_nomask_kernel_2<<<GRID_DIM, BLOCK_DIM, 0,
                                           jc.join_stream>>>(
        jc.d_data_arrays_, jc.d_parent_arrays_, jc.select_bool_mask_,
        jc.num_frag_rows, jc.root_trie->num_cols, jc.col_key, jc.d_edge_values_,
        jc.d_edge_offsets_, ht.h_num_buckets_[jc.directed_edge_idx],
        ht.h_hash_table_offs_[jc.directed_edge_idx], jc.directed_edge_idx,
        jc.d_num_res_for_each, jc.d_st_array, jc.dup_check_type,
        jc.num_dup_cols, jc.d_dup_cols, jc.up_to_col);
    mem_mgr.unpin_climb_data();
    mem_mgr.unpin_climb_parents();

    uint64_t *d_row_offsets = nullptr;
    if (!mem_mgr.allocate_temporal(d_row_offsets,
                                   sizeof(uint64_t) * (jc.num_frag_rows + 1))) {
      std::cerr
          << "refer_join_with_duplication: failed to allocate d_row_offsets."
          << std::endl;
      exit(EXIT_FAILURE);
    }

    size_t temp_storage_bytes = 0;
    void *d_temp_storage = nullptr;
    cub::DeviceScan::ExclusiveScan(d_temp_storage, temp_storage_bytes,
                                   jc.d_num_res_for_each, d_row_offsets,
                                   cub::Sum(), (uint64_t)0,
                                   jc.num_frag_rows + 1, jc.join_stream);
    if (!mem_mgr.allocate_temporal(d_temp_storage, temp_storage_bytes)) {
      std::cerr << "refer_join_with_duplication: failed to allocate scan temp "
                   "storage."
                << std::endl;
      exit(EXIT_FAILURE);
    }
    cub::DeviceScan::ExclusiveScan(d_temp_storage, temp_storage_bytes,
                                   jc.d_num_res_for_each, d_row_offsets,
                                   cub::Sum(), (uint64_t)0,
                                   jc.num_frag_rows + 1, jc.join_stream);
    mem_mgr.deallocate_temporal(jc.d_num_res_for_each);

    cuchk(cudaMemcpyAsync(jc.h_num_res, d_row_offsets + jc.num_frag_rows,
                          sizeof(uint64_t), cudaMemcpyDeviceToHost,
                          jc.join_stream));
    mem_mgr.deallocate_temporal(d_temp_storage);
    d_temp_storage = nullptr;

    cuchk(cudaStreamSynchronize(jc.join_stream));
    resetTargetTrieData(jc.target_trie, *jc.h_num_res, false);
    if (jc.skip_final_materialization == false) {
      bool allocate_success = mem_mgr.allocate_trie(jc.target_trie->id);

      if (allocate_success) {
        join_refer_write_dup_nomask_pipeline<<<GRID_DIM, BLOCK_DIM, 0,
                                               jc.join_stream>>>(
            0, jc.num_frag_rows, jc.d_data_arrays_, jc.d_parent_arrays_,
            jc.root_trie->num_cols, jc.col_key, jc.d_edge_values_,
            jc.d_edge_offsets_, ht.h_num_buckets_[jc.directed_edge_idx],
            ht.h_hash_table_offs_[jc.directed_edge_idx], jc.directed_edge_idx,
            d_row_offsets, jc.target_trie->data->d_data,
            jc.target_trie->data->d_parents, 0, jc.dup_check_type,
            jc.num_dup_cols, jc.d_dup_cols, jc.up_to_col);
        cuchk(cudaStreamSynchronize(jc.join_stream));
      } else {
        join_refer_write_dup_nomask_pipeline_write(d_row_offsets);
      }
    }

    if (d_row_offsets)
      mem_mgr.deallocate_temporal(d_row_offsets);
    mem_mgr.deallocate_temporal(jc.d_st_array);
    return;
  }

  join_refer_count_dup_kernel_2<<<GRID_DIM, BLOCK_DIM, 0, jc.join_stream>>>(
      jc.d_data_arrays_, jc.d_parent_arrays_, jc.select_bool_mask_,
      jc.num_frag_rows, jc.root_trie->num_cols, jc.col_key, jc.d_edge_values_,
      jc.d_edge_offsets_, ht.h_num_buckets_[jc.directed_edge_idx],
      ht.h_hash_table_offs_[jc.directed_edge_idx], jc.directed_edge_idx,
      jc.d_mask, jc.mask_length, jc.d_st_array, jc.dup_check_type,
      jc.num_dup_cols, jc.d_dup_cols, jc.up_to_col);
  mem_mgr.unpin_climb_data();
  mem_mgr.unpin_climb_parents();

  uint64_t *d_row_offsets = nullptr;
  mem_mgr.allocate_temporal(d_row_offsets,
                            sizeof(uint64_t) * (jc.num_frag_rows + 1));

  size_t temp_storage_bytes = 0;
  void *d_temp_storage = nullptr;

  // Create segment offsets: [0, mask_length, 2*mask_length, ...,
  // num_rows*mask_length] Use 64-bit offsets to avoid overflow when num_rows *
  // mask_length exceeds 2^32.
  uint64_t *d_segment_offsets = nullptr;
  mem_mgr.allocate_temporal(d_segment_offsets,
                            sizeof(uint64_t) * (jc.num_frag_rows + 1));
  thrust::device_ptr<uint64_t> offsets_ptr(d_segment_offsets);
  // Use stream-bound execution policy - cudaStreamDefault since this is in a
  // synchronous path
  thrust::sequence(thrust::cuda::par.on(jc.join_stream), offsets_ptr,
                   offsets_ptr + jc.num_frag_rows + 1, static_cast<uint64_t>(0),
                   static_cast<uint64_t>(jc.mask_length));
  // Create a transform iterator that applies __popc to each mask tile
  auto popcount_iterator =
      thrust::make_transform_iterator(jc.d_mask, PopcToCount{});
  // Use CUB DeviceSegmentedReduce to sum popcounts per row
  cuchk(cub::DeviceSegmentedReduce::Sum(
      d_temp_storage, temp_storage_bytes, popcount_iterator,
      jc.d_num_res_for_each, jc.num_frag_rows, d_segment_offsets,
      d_segment_offsets + 1, jc.join_stream));
  mem_mgr.allocate_temporal(d_temp_storage, temp_storage_bytes);
  cuchk(cub::DeviceSegmentedReduce::Sum(
      d_temp_storage, temp_storage_bytes, popcount_iterator,
      jc.d_num_res_for_each, jc.num_frag_rows, d_segment_offsets,
      d_segment_offsets + 1, jc.join_stream));
  mem_mgr.deallocate_temporal(d_segment_offsets);
  mem_mgr.deallocate_temporal(d_temp_storage);
  d_temp_storage = nullptr;
  temp_storage_bytes = 0;

  cub::DeviceScan::ExclusiveScan(
      d_temp_storage, temp_storage_bytes, jc.d_num_res_for_each, d_row_offsets,
      cub::Sum(), (uint64_t)0, jc.num_frag_rows + 1, jc.join_stream);
  mem_mgr.allocate_temporal(d_temp_storage, temp_storage_bytes);
  cub::DeviceScan::ExclusiveScan(
      d_temp_storage, temp_storage_bytes, jc.d_num_res_for_each, d_row_offsets,
      cub::Sum(), (uint64_t)0, jc.num_frag_rows + 1, jc.join_stream);
  mem_mgr.deallocate_temporal(jc.d_num_res_for_each);
  // Copy total count from d_row_offsets[num_rows] to host
  cuchk(cudaMemcpyAsync(jc.h_num_res, d_row_offsets + jc.num_frag_rows,
                        sizeof(uint64_t), cudaMemcpyDeviceToHost,
                        jc.join_stream));
  // Free temporary storage
  mem_mgr.deallocate_temporal(d_temp_storage);
  d_temp_storage = nullptr;
  temp_storage_bytes = 0;
  // Allocate result memory
  cuchk(cudaStreamSynchronize(jc.join_stream));
  resetTargetTrieData(jc.target_trie, *jc.h_num_res, false);
  if (jc.skip_final_materialization == false) {
    bool allocate_success = mem_mgr.allocate_trie(jc.target_trie->id);
    // target_trie->data->allocate();

    if (allocate_success) {
      join_refer_write_dup_kernel<<<GRID_DIM, BLOCK_DIM, 0, jc.join_stream>>>(
          nullptr, nullptr, jc.root_trie->num_layers, jc.num_frag_rows,
          jc.root_trie->num_cols, jc.col_key, jc.d_edge_keys_,
          jc.d_edge_values_, jc.d_edge_offsets_,
          ht.h_num_buckets_[jc.directed_edge_idx],
          ht.h_hash_table_offs_[jc.directed_edge_idx], jc.directed_edge_idx,
          jc.d_mask, jc.mask_length, jc.d_st_array,
          jc.target_trie->data->d_data, d_row_offsets,
          jc.target_trie->data->d_parents, jc.dup_check_type, jc.num_dup_cols,
          jc.d_dup_cols);
      cuchk(cudaStreamSynchronize(jc.join_stream));
    } else // if (!allocate_success)
    {
      join_refer_write_dup_pipeline(d_row_offsets);
    }
  }

  if (d_row_offsets)
    mem_mgr.deallocate_temporal(d_row_offsets);
  mem_mgr.deallocate_temporal(jc.d_mask);
  mem_mgr.deallocate_temporal(jc.d_st_array);
}

void refer_join_with_duplication_debug() { refer_join_with_duplication(); }

void refer_join() {
  auto lat = &Lattice::getInstance();
  auto utm = &UnifiedTrieManager::getInstance();
  auto mem_mgr = &MemoryManager::getInstance();
  auto ht = &HashTable::getInstance();
  auto &jc = JoinContext::getInstance();

  bool context_ready = jc.refer_join_general_context();
  if (!context_ready) {
    if (jc.dup_check_type == 0)
      refer_join_without_duplication_pipeline();
    else
      refer_join_with_duplication_pipeline();
  } else if (jc.dup_check_type == 0) {
#ifndef NDEBUG
    refer_join_without_duplication_debug();
#else
    refer_join_without_duplication();
#endif
  } else {
#ifndef NDEBUG
    refer_join_with_duplication_debug();
#else
    refer_join_with_duplication();
#endif
  }
  // Update target properties
  cuchk(cudaStreamSynchronize(jc.join_stream));
  jc.target_trie->updateNumResults(*jc.h_num_res);
  jc.target_trie->data->num_rows = *jc.h_num_res;

#ifndef NDEBUG
  std::cout << "refer_join: " << *jc.h_num_res << " results produced"
            << std::endl;
#endif

  // Cleanup
  cuchk(cudaStreamSynchronize(jc.join_stream));
}

void join_count_refer_no_dup_dispatcher(
    vtype **vertex_arrays, uint64_t **parent_pointers, uint64_t num_rows,
    uint32_t num_cols, uint32_t join_col, uint32_t *select_bool_vec,
    vtype *edge_key_array, vtype *edge_value_array, uint32_t *edge_offsets,
    uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx,
    uint32_t *d_num_res_for_each, uint32_t *d_st_array,
    uint32_t task_per_warp) {
  auto &jc = JoinContext::getInstance();
  switch (task_per_warp) {
  case 1:
    join_count_refer_kernel_nodup<1>
        <<<GRID_DIM, BLOCK_DIM, 0, jc.join_stream>>>(
            vertex_arrays, parent_pointers, num_rows, num_cols, join_col,
            select_bool_vec, edge_key_array, edge_value_array, edge_offsets,
            num_buckets, edge_offset, directed_edge_idx, d_num_res_for_each,
            d_st_array);
    break;
  case 2:
    join_count_refer_kernel_nodup<2>
        <<<GRID_DIM, BLOCK_DIM, 0, jc.join_stream>>>(
            vertex_arrays, parent_pointers, num_rows, num_cols, join_col,
            select_bool_vec, edge_key_array, edge_value_array, edge_offsets,
            num_buckets, edge_offset, directed_edge_idx, d_num_res_for_each,
            d_st_array);
    break;
  case 4:
    join_count_refer_kernel_nodup<4>
        <<<GRID_DIM, BLOCK_DIM, 0, jc.join_stream>>>(
            vertex_arrays, parent_pointers, num_rows, num_cols, join_col,
            select_bool_vec, edge_key_array, edge_value_array, edge_offsets,
            num_buckets, edge_offset, directed_edge_idx, d_num_res_for_each,
            d_st_array);
    break;
  case 8:
    join_count_refer_kernel_nodup<8>
        <<<GRID_DIM, BLOCK_DIM, 0, jc.join_stream>>>(
            vertex_arrays, parent_pointers, num_rows, num_cols, join_col,
            select_bool_vec, edge_key_array, edge_value_array, edge_offsets,
            num_buckets, edge_offset, directed_edge_idx, d_num_res_for_each,
            d_st_array);
    break;
  case 16:
    join_count_refer_kernel_nodup<16>
        <<<GRID_DIM, BLOCK_DIM, 0, jc.join_stream>>>(
            vertex_arrays, parent_pointers, num_rows, num_cols, join_col,
            select_bool_vec, edge_key_array, edge_value_array, edge_offsets,
            num_buckets, edge_offset, directed_edge_idx, d_num_res_for_each,
            d_st_array);
    break;
  case 32:
    join_count_refer_kernel_nodup<32>
        <<<GRID_DIM, BLOCK_DIM, 0, jc.join_stream>>>(
            vertex_arrays, parent_pointers, num_rows, num_cols, join_col,
            select_bool_vec, edge_key_array, edge_value_array, edge_offsets,
            num_buckets, edge_offset, directed_edge_idx, d_num_res_for_each,
            d_st_array);
    break;
  default:
    std::cerr << "Unsupported task_per_warp: " << task_per_warp << std::endl;
    exit(EXIT_FAILURE);
  }
}

void join_write_refer_no_dup_dispatcher(
    uint64_t *valid_row_ids, uint64_t num_valid_rows, vtype *edge_value_array_,
    uint32_t *d_selected_count, uint32_t *d_st_array, uint32_t *d_res,
    uint64_t *d_row_offsets, uint64_t *d_parent_res, uint32_t task_per_warp) {
  auto &jc = JoinContext::getInstance();
  switch (task_per_warp) {
  case 1:
    join_write_refer_kernel_nodup<1>
        <<<GRID_DIM, BLOCK_DIM, 0, jc.join_stream>>>(
            valid_row_ids, num_valid_rows, edge_value_array_, d_selected_count,
            d_st_array, d_res, d_row_offsets, d_parent_res);
    break;
  case 2:
    join_write_refer_kernel_nodup<2>
        <<<GRID_DIM, BLOCK_DIM, 0, jc.join_stream>>>(
            valid_row_ids, num_valid_rows, edge_value_array_, d_selected_count,
            d_st_array, d_res, d_row_offsets, d_parent_res);
    break;
  case 4:
    join_write_refer_kernel_nodup<4>
        <<<GRID_DIM, BLOCK_DIM, 0, jc.join_stream>>>(
            valid_row_ids, num_valid_rows, edge_value_array_, d_selected_count,
            d_st_array, d_res, d_row_offsets, d_parent_res);
    break;
  case 8:
    join_write_refer_kernel_nodup<8>
        <<<GRID_DIM, BLOCK_DIM, 0, jc.join_stream>>>(
            valid_row_ids, num_valid_rows, edge_value_array_, d_selected_count,
            d_st_array, d_res, d_row_offsets, d_parent_res);
    break;
  case 16:
    join_write_refer_kernel_nodup<16>
        <<<GRID_DIM, BLOCK_DIM, 0, jc.join_stream>>>(
            valid_row_ids, num_valid_rows, edge_value_array_, d_selected_count,
            d_st_array, d_res, d_row_offsets, d_parent_res);
    break;
  case 32:
    join_write_refer_kernel_nodup<32>
        <<<GRID_DIM, BLOCK_DIM, 0, jc.join_stream>>>(
            valid_row_ids, num_valid_rows, edge_value_array_, d_selected_count,
            d_st_array, d_res, d_row_offsets, d_parent_res);
    break;
  default:
    std::cerr << "Unsupported task_per_warp: " << task_per_warp << std::endl;
    exit(EXIT_FAILURE);
  }
  cuchk(cudaStreamSynchronize(jc.join_stream));
}

// return exclusive right boundary (one past the last fragment row to process)
static uint64_t find_frag_range(uint64_t *h_row_offsets, uint64_t left,
                                uint64_t right, uint64_t target_rows_limit) {
  // Binary search to find the largest 'end' such that:
  // h_row_offsets[end] - h_row_offsets[left] <= target_rows_limit
  // Fragment range [left, end) produces exactly h_row_offsets[end] -
  // h_row_offsets[left] results

  uint64_t l = left, r = right;
  while (l < r) {
    uint64_t mid =
        l +
        (r - l + 1) / 2; // Use upper mid to avoid infinite loop when l = r-1
    uint64_t target_rows = h_row_offsets[mid] - h_row_offsets[left];
    if (target_rows <= target_rows_limit) {
      l = mid; // mid is valid, try larger
    } else {
      r = mid - 1; // mid exceeds, try smaller
    }
  }

  // However, due to duplicates (zero-result rows), skip forward past all
  // duplicates
  while (l < right && h_row_offsets[l] == h_row_offsets[l + 1]) {
    l++;
  }

  // Return exclusive end
  return l;
}

void join_refer_write_dup_nomask_pipeline_write(uint64_t *d_row_offsets) {
  std::cout << "Entering refer join with duplication nomask pipeline write."
            << std::endl;
  auto &mem_mgr = MemoryManager::getInstance();
  auto &host_mgr = HostMemoryManager::getInstance();
  auto &ht = HashTable::getInstance();
  auto &jc = JoinContext::getInstance();

  mem_mgr.p_type = ProcessType::PIPELINE;

  uint32_t *h_data_ptr = static_cast<uint32_t *>(host_mgr.allocateManagedBytes(
      sizeof(uint32_t) * jc.h_num_res[0], sizeof(uint32_t)));
  if (h_data_ptr == nullptr) {
    std::cerr << "Host memory allocation failed for refer join pipeline write."
              << std::endl;
    exit(EXIT_FAILURE);
  }
  uint64_t *h_parents_ptr =
      static_cast<uint64_t *>(host_mgr.allocateManagedBytes(
          sizeof(uint64_t) * jc.h_num_res[0], sizeof(uint64_t)));
  if (h_parents_ptr == nullptr) {
    std::cerr << "Host memory allocation failed for refer join pipeline write."
              << std::endl;
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
  uint64_t num_target_rows_each_stream =
      free_mem_each_stream / (sizeof(uint32_t) + sizeof(uint64_t));
  uint64_t frag_offset = 0, result_offset = 0;

  if (num_target_rows_each_stream == 0) {
    std::cerr << "join_refer_write_dup_nomask_pipeline_write: insufficient GPU "
                 "memory to pipeline results."
              << std::endl;
    return;
  }

  uint64_t *h_row_offsets = static_cast<uint64_t *>(host_mgr.allocateBytes(
      sizeof(uint64_t) * (jc.num_frag_rows + 1), sizeof(uint64_t)));
  if (h_row_offsets == nullptr) {
    std::cerr
        << "Host memory allocation failed for refer join pipeline offsets."
        << std::endl;
    exit(EXIT_FAILURE);
  }

  cuchk(cudaMemcpyAsync(h_row_offsets, d_row_offsets,
                        sizeof(uint64_t) * (jc.num_frag_rows + 1),
                        cudaMemcpyDeviceToHost, jc.join_stream));
  cuchk(cudaStreamSynchronize(jc.join_stream));

  uint32_t *d_data_buffer = nullptr;
  uint64_t *d_parents_buffer = nullptr;
  if (!allocate_pipeline_write_buffers(num_target_rows_each_stream,
                                       d_data_buffer, d_parents_buffer)) {
    std::cerr << "join_refer_write_dup_nomask_pipeline_write: failed to "
                 "allocate GPU buffers for pipelining."
              << std::endl;
    exit(EXIT_FAILURE);
  }

  uint32_t *d_data_ptrs[JoinContext::nStreams] = {
      d_data_buffer, d_data_buffer + num_target_rows_each_stream,
      d_data_buffer + 2 * num_target_rows_each_stream};
  uint64_t *d_parents_ptrs[JoinContext::nStreams] = {
      d_parents_buffer, d_parents_buffer + num_target_rows_each_stream,
      d_parents_buffer + 2 * num_target_rows_each_stream};

  while (result_offset < jc.h_num_res[0]) {
    for (int i = 0;
         i < JoinContext::nStreams && result_offset < jc.h_num_res[0]; ++i) {
      uint64_t frag_left = frag_offset;
      uint64_t frag_right =
          find_frag_range(h_row_offsets, frag_left, jc.num_frag_rows,
                          num_target_rows_each_stream);
      uint64_t num_frag_rows_this_iter = frag_right - frag_left;

      join_refer_write_dup_nomask_pipeline<<<GRID_DIM, BLOCK_DIM, 0,
                                             jc.pipeline_streams[i]>>>(
          frag_left, num_frag_rows_this_iter, jc.d_data_arrays_,
          jc.d_parent_arrays_, jc.root_trie->num_cols, jc.col_key,
          jc.d_edge_values_, jc.d_edge_offsets_,
          ht.h_num_buckets_[jc.directed_edge_idx],
          ht.h_hash_table_offs_[jc.directed_edge_idx], jc.directed_edge_idx,
          d_row_offsets, d_data_ptrs[i], d_parents_ptrs[i], result_offset,
          jc.dup_check_type, jc.num_dup_cols, jc.d_dup_cols, jc.up_to_col);

      uint64_t num_target_rows_this_iter =
          h_row_offsets[frag_right] - h_row_offsets[frag_left];
      if (num_target_rows_this_iter == 0) {
        frag_offset += num_frag_rows_this_iter;
        continue;
      }

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

void join_refer_write_dup_pipeline(uint64_t *d_row_offsets) {
  std::cout << "Entering refer join with duplication pipeline write."
            << std::endl;
  auto &mem_mgr = MemoryManager::getInstance();
  auto &host_mgr = HostMemoryManager::getInstance();
  auto &jc = JoinContext::getInstance();

  mem_mgr.p_type = ProcessType::PIPELINE;

  uint32_t *h_data_ptr = static_cast<uint32_t *>(host_mgr.allocateManagedBytes(
      sizeof(uint32_t) * jc.h_num_res[0], sizeof(uint32_t)));
  if (h_data_ptr == nullptr) {
    std::cerr << "Host memory allocation failed for refer join pipeline write."
              << std::endl;
    exit(EXIT_FAILURE);
  }
  uint64_t *h_parents_ptr =
      static_cast<uint64_t *>(host_mgr.allocateManagedBytes(
          sizeof(uint64_t) * jc.h_num_res[0], sizeof(uint64_t)));
  if (h_parents_ptr == nullptr) {
    std::cerr << "Host memory allocation failed for refer join pipeline write."
              << std::endl;
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
  uint64_t num_target_rows_each_stream =
      free_mem_each_stream / (sizeof(uint32_t) + sizeof(uint64_t));
  uint64_t frag_offset = 0, result_offset = 0;

  if (num_target_rows_each_stream == 0) {
    std::cerr << "join_refer_write_dup_pipeline: insufficient GPU memory to "
                 "pipeline results."
              << std::endl;
    return;
  }

  uint64_t *h_row_offsets = static_cast<uint64_t *>(host_mgr.allocateBytes(
      sizeof(uint64_t) * (jc.num_frag_rows + 1), sizeof(uint64_t)));
  if (h_row_offsets == nullptr) {
    std::cerr
        << "Host memory allocation failed for refer join pipeline offsets."
        << std::endl;
    exit(EXIT_FAILURE);
  }

  cuchk(cudaMemcpyAsync(h_row_offsets, d_row_offsets,
                        sizeof(uint64_t) * (jc.num_frag_rows + 1),
                        cudaMemcpyDeviceToHost, jc.join_stream));
  cuchk(cudaStreamSynchronize(jc.join_stream));

  // allocate buffer
  uint32_t *d_data_buffer = nullptr;
  uint64_t *d_parents_buffer = nullptr;
  if (!allocate_pipeline_write_buffers(num_target_rows_each_stream,
                                       d_data_buffer, d_parents_buffer)) {
    std::cerr << "join_refer_write_dup_pipeline: failed to allocate GPU "
                 "buffers for pipelining."
              << std::endl;
    exit(EXIT_FAILURE);
  }
  uint32_t *d_data_ptrs[JoinContext::nStreams] = {
      d_data_buffer, d_data_buffer + num_target_rows_each_stream,
      d_data_buffer + 2 * num_target_rows_each_stream};
  uint64_t *d_parents_ptrs[JoinContext::nStreams] = {
      d_parents_buffer, d_parents_buffer + num_target_rows_each_stream,
      d_parents_buffer + 2 * num_target_rows_each_stream};

  while (result_offset < jc.h_num_res[0]) {
    for (int i = 0;
         i < JoinContext::nStreams && result_offset < jc.h_num_res[0]; ++i) {
      // perform binary search on h_row_offsets to find how many fragment rows
      // can be processed
      uint64_t frag_left = frag_offset;
      uint64_t frag_right =
          find_frag_range(h_row_offsets, frag_left, jc.num_frag_rows,
                          num_target_rows_each_stream);
      uint64_t num_frag_rows_this_iter = frag_right - frag_left;

      join_write_kernel_dup_pipeline<<<GRID_DIM, BLOCK_DIM, 0,
                                       jc.pipeline_streams[i]>>>(
          // trie_inter
          frag_left, num_frag_rows_this_iter,
          // trie_edge
          jc.d_edge_values_, jc.d_mask, jc.mask_length, d_row_offsets,
          jc.d_st_array,
          // output buffers
          d_data_ptrs[i], d_parents_ptrs[i], result_offset);

      uint64_t num_target_rows_this_iter =
          h_row_offsets[frag_right] - h_row_offsets[frag_left];
      if (num_target_rows_this_iter == 0) {
        frag_offset += num_frag_rows_this_iter;
        continue;
      }

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

  for (int i = 0; i < JoinContext::nStreams; ++i)
    cuchk(cudaStreamSynchronize(jc.pipeline_streams[i]));

  mem_mgr.deallocate_temporal(d_data_buffer);
  mem_mgr.deallocate_temporal(d_parents_buffer);
  cuchk(cudaStreamSynchronize(jc.join_stream));
}

void join_refer_write_nodup_pipeline(uint64_t *d_selected_row_ids,
                                     uint64_t num_selected_rows,
                                     uint32_t *d_selected_count,
                                     uint64_t *d_row_offsets) {
  std::cout << "Entering refer join without duplication pipeline write."
            << std::endl;
  auto &mem_mgr = MemoryManager::getInstance();
  auto &host_mgr = HostMemoryManager::getInstance();
  auto &jc = JoinContext::getInstance();

  mem_mgr.p_type = ProcessType::PIPELINE;

  uint32_t *h_data_ptr = static_cast<uint32_t *>(host_mgr.allocateManagedBytes(
      sizeof(uint32_t) * jc.h_num_res[0], sizeof(uint32_t)));
  if (h_data_ptr == nullptr) {
    std::cerr << "Host memory allocation failed for refer join pipeline write."
              << std::endl;
    exit(EXIT_FAILURE);
  }
  uint64_t *h_parents_ptr =
      static_cast<uint64_t *>(host_mgr.allocateManagedBytes(
          sizeof(uint64_t) * jc.h_num_res[0], sizeof(uint64_t)));
  if (h_parents_ptr == nullptr) {
    std::cerr << "Host memory allocation failed for refer join pipeline write."
              << std::endl;
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
  uint64_t num_target_rows_each_stream =
      free_mem_each_stream / (sizeof(uint32_t) + sizeof(uint64_t));
  uint64_t frag_offset = 0, result_offset = 0;

  uint64_t *h_row_offsets = static_cast<uint64_t *>(host_mgr.allocateBytes(
      sizeof(uint64_t) * (num_selected_rows + 1), sizeof(uint64_t)));
  if (h_row_offsets == nullptr) {
    std::cerr
        << "Host memory allocation failed for refer join pipeline offsets."
        << std::endl;
    exit(EXIT_FAILURE);
  }
  cuchk(cudaMemcpyAsync(h_row_offsets, d_row_offsets,
                        sizeof(uint64_t) * (num_selected_rows + 1),
                        cudaMemcpyDeviceToHost, jc.join_stream));
  cuchk(cudaStreamSynchronize(jc.join_stream));

  if (num_target_rows_each_stream == 0) {
    std::cerr << "join_refer_write_nodup_pipeline: insufficient GPU memory to "
                 "pipeline results."
              << std::endl;
    return;
  }

  // allocate buffer
  uint32_t *d_data_buffer = nullptr;
  uint64_t *d_parents_buffer = nullptr;
  if (!allocate_pipeline_write_buffers(num_target_rows_each_stream,
                                       d_data_buffer, d_parents_buffer)) {
    std::cerr << "join_refer_write_nodup_pipeline: failed to allocate GPU "
                 "buffers for pipelining."
              << std::endl;
    exit(EXIT_FAILURE);
  }
  uint32_t *d_data_ptrs[JoinContext::nStreams] = {
      d_data_buffer, d_data_buffer + num_target_rows_each_stream,
      d_data_buffer + 2 * num_target_rows_each_stream};
  uint64_t *d_parents_ptrs[JoinContext::nStreams] = {
      d_parents_buffer, d_parents_buffer + num_target_rows_each_stream,
      d_parents_buffer + 2 * num_target_rows_each_stream};

  while (result_offset < jc.h_num_res[0]) {
    for (int i = 0;
         i < JoinContext::nStreams && result_offset < jc.h_num_res[0]; ++i) {
      uint64_t frag_left = frag_offset;
      uint64_t frag_right =
          find_frag_range(h_row_offsets, frag_left, num_selected_rows,
                          num_target_rows_each_stream);
      uint64_t num_frag_rows_this_iter = frag_right - frag_left;
      uint64_t num_target_rows_this_iter =
          h_row_offsets[frag_right] - h_row_offsets[frag_left];
      if (num_target_rows_this_iter == 0) {
        frag_offset += num_frag_rows_this_iter;
        continue;
      }

      join_write_refer_kernel_nodup_pipeline<<<GRID_DIM, BLOCK_DIM, 0,
                                               jc.pipeline_streams[i]>>>(
          frag_left, num_frag_rows_this_iter, d_selected_row_ids,
          jc.d_edge_values_, d_selected_count, jc.d_st_array, d_row_offsets,
          d_data_ptrs[i], d_parents_ptrs[i], result_offset);

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
