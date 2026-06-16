#include "host_select.cuh"
#include "host_select_kernel.cuh"
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
} // namespace

#define exit(code) fssmExit((code), __FILE__, __LINE__)

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

  void compute_pipeline_launch(uint64_t num_rows, dim3 &grid_dim, dim3 &block_dim)
  {
    uint64_t num_tiles = (num_rows + WARP_SIZE - 1) / WARP_SIZE;
    num_tiles = std::max<uint64_t>(num_tiles, 1);

    const uint32_t max_warps_per_block = BLOCK_DIM / WARP_SIZE;
    uint32_t warps_per_block = static_cast<uint32_t>(std::min<uint64_t>(num_tiles, max_warps_per_block));
    warps_per_block = std::max<uint32_t>(warps_per_block, 1);

    block_dim = dim3(warps_per_block * WARP_SIZE, 1, 1);

    uint64_t grid_x64 = (num_tiles + warps_per_block - 1) / warps_per_block;
    uint32_t grid_x = static_cast<uint32_t>(std::min<uint64_t>(grid_x64, GRID_DIM));
    grid_x = std::max<uint32_t>(grid_x, 1);
    grid_dim = dim3(grid_x, 1, 1);
  }
} // namespace

void direct_select_pipeline()
{
  std::cout << "Entering direct select pipeline." << std::endl;

  auto &mem_mgr = MemoryManager::getInstance();
  auto &ht = HashTable::getInstance();
  auto &jc = JoinContext::getInstance();
  auto &host_mgr = HostMemoryManager::getInstance();

  jc.direct_select_pipeline_context();

  const int nStreams = JoinContext::nStreams;
  const bool output_on_host = (mem_mgr.p_type == ProcessType::PIPELINE);
  uint32_t *host_mask = nullptr;
  if (output_on_host)
  {
    const int target_mask_block_id = jc.target_trie->id * 2;
    host_mask = static_cast<uint32_t *>(mem_mgr.memory_blocks[target_mask_block_id].host_ptr);
    if (host_mask == nullptr)
    {
      std::cerr << "direct_select_pipeline: missing host mask for target trie " << jc.target_trie->id << std::endl;
      exit(EXIT_FAILURE);
    }
  }

  const uint32_t num_cols = jc.frag_trie->num_cols;
  const uint32_t col_upper_bound = std::min(jc.col_key, jc.col_value);

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

  const uint64_t num_tiles_per_stream = (rows_per_stream + 31) >> 5;
  if (num_tiles_per_stream == 0)
  {
    std::cerr << "direct_select_pipeline: insufficient GPU memory for pipelining." << std::endl;
    exit(EXIT_FAILURE);
  }

  uint64_t *d_result_stream = nullptr;
  uint64_t *d_result_agg = nullptr;
  mem_mgr.allocate_temporal(d_result_stream, sizeof(uint64_t) * nStreams);
  mem_mgr.allocate_temporal(d_result_agg, sizeof(uint64_t));
  cuchk(cudaMemsetAsync(d_result_stream, 0, sizeof(uint64_t) * nStreams, jc.join_stream));
  cuchk(cudaMemsetAsync(d_result_agg, 0, sizeof(uint64_t), jc.join_stream));

  void *d_temp_storage_base = nullptr;
  size_t temp_storage_bytes = 0;
  using PopcIterator = cub::TransformInputIterator<uint32_t, PopcToCount, uint32_t *>;
  cub::DeviceReduce::Sum(d_temp_storage_base, temp_storage_bytes, PopcIterator(nullptr, PopcToCount{}),
                         d_result_stream, num_tiles_per_stream, jc.join_stream);
  temp_storage_bytes = (temp_storage_bytes + 255) & ~255;
  mem_mgr.allocate_temporal(d_temp_storage_base, temp_storage_bytes * nStreams);

  uint32_t *d_mask_tile_base = nullptr;
  if (output_on_host)
    mem_mgr.allocate_temporal(d_mask_tile_base, sizeof(uint32_t) * num_tiles_per_stream * nStreams);

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

  const size_t parent_info_count = jc.h_parent_info_arrays_.size();
  const bool need_parent_updates = (num_partial_parents > 0);
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
        std::cerr << "direct_select_pipeline: host allocation failed for parent JoinInfo buffers." << std::endl;
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
      uint64_t num_tiles_this_iter = (num_rows_this_iter + 31) >> 5;
      cudaStream_t stream = jc.pipeline_streams[stream_id];

      compute_column_ranges(jc.frag_trie, col_upper_bound, cur_offset, cur_offset + num_rows_this_iter, ranges, jc.join_stream);

      JoinInfo<uint64_t> *d_parent_info_ptr = jc.d_parent_info_arrays_;
      if (need_parent_updates)
      {
        JoinInfo<uint64_t> *h_parent_infos = h_parent_info_arrays[stream_id];
        for (size_t col = 0; col < parent_info_count; ++col)
          h_parent_infos[col] = jc.h_parent_info_arrays_[col];

        // Stage partial parent segments for this tile
        for (size_t i = 0; i < num_partial_parents; ++i)
        {
          uint32_t col = jc.partial_parent_cols[i];
          uint32_t col_trie_id = jc.frag_trie->column_ids[col];
          const int block_idx = static_cast<int>(col_trie_id) * 2 + 1;
          auto &block = mem_mgr.memory_blocks[block_idx];
          if (block.host_ptr == nullptr)
          {
            std::cerr << "direct_select_pipeline: missing host parents for trie " << col_trie_id << std::endl;
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

      // Prepare key/value infos (stage partial data segments if needed)
      JoinInfo<uint32_t> key_info = jc.h_data_info_arrays_[jc.col_key];
      JoinInfo<uint32_t> val_info = jc.h_data_info_arrays_[jc.col_value];
      for (size_t i = 0; i < num_partial_data; ++i)
      {
        uint32_t col = jc.partial_data_cols[i];
        uint32_t col_trie_id = jc.frag_trie->column_ids[col];
        const int block_idx = static_cast<int>(col_trie_id) * 2;
        auto &block = mem_mgr.memory_blocks[block_idx];
        if (block.host_ptr == nullptr)
        {
          std::cerr << "direct_select_pipeline: missing host data for trie " << col_trie_id << std::endl;
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
          if (col == jc.col_key)
          {
            key_info.array_ptr = d_data_buf;
            key_info.offset = r.start;
            key_info.is_partial = 1;
          }
          if (col == jc.col_value)
          {
            val_info.array_ptr = d_data_buf;
            val_info.offset = r.start;
            val_info.is_partial = 1;
          }
        }
      }

      uint32_t *d_out_mask = nullptr;
      if (output_on_host)
        d_out_mask = d_mask_tile_base + num_tiles_per_stream * static_cast<size_t>(stream_id);
      else
        d_out_mask = jc.result_mask_ + (cur_offset >> 5);

      dim3 grid_dim;
      dim3 block_dim;
      compute_pipeline_launch(num_rows_this_iter, grid_dim, block_dim);
      select_trie_kernel_pipeline_info<<<grid_dim, block_dim, 0, stream>>>(
          key_info, val_info,
          d_parent_info_ptr, num_cols, jc.col_key, jc.col_value,
          cur_offset, num_rows_this_iter,
          jc.d_edge_keys_, jc.d_edge_values_, jc.d_edge_offsets_,
          ht.h_num_buckets_[jc.directed_edge_idx], ht.h_hash_table_offs_[jc.directed_edge_idx], jc.directed_edge_idx,
          d_out_mask);

      PopcIterator popc_iter(d_out_mask, PopcToCount{});
      void *d_stream_storage = static_cast<char *>(d_temp_storage_base) + temp_storage_bytes * stream_id;
      size_t stream_storage_bytes = temp_storage_bytes;
      cub::DeviceReduce::Sum(d_stream_storage, stream_storage_bytes, popc_iter,
                             d_result_stream + stream_id, num_tiles_this_iter, stream);
      host_atomicAdd<<<1, 1, 0, stream>>>(d_result_agg, d_result_stream + stream_id);

      if (output_on_host)
      {
        cuchk(cudaMemcpyAsync(host_mask + (cur_offset >> 5), d_out_mask,
                              sizeof(uint32_t) * num_tiles_this_iter,
                              cudaMemcpyDeviceToHost, stream));
      }

      cuchk(cudaEventRecord(done_events[stream_id], stream));
      cur_offset += num_rows_this_iter;
    }
  }

  for (int i = 0; i < nStreams; ++i)
  {
    cuchk(cudaEventSynchronize(done_events[i]));
    cuchk(cudaEventDestroy(done_events[i]));
  }

  cuchk(cudaMemcpyAsync(jc.h_num_res, d_result_agg, sizeof(uint64_t),
                        cudaMemcpyDeviceToHost, jc.join_stream));

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
  if (d_partial_parent_base != nullptr)
    mem_mgr.deallocate_temporal(d_partial_parent_base);
  if (d_partial_data_base != nullptr)
    mem_mgr.deallocate_temporal(d_partial_data_base);
  if (d_mask_tile_base != nullptr)
    mem_mgr.deallocate_temporal(d_mask_tile_base);
  if (d_temp_storage_base != nullptr)
    mem_mgr.deallocate_temporal(d_temp_storage_base);
  if (d_result_stream != nullptr)
    mem_mgr.deallocate_temporal(d_result_stream);
  if (d_result_agg != nullptr)
    mem_mgr.deallocate_temporal(d_result_agg);
  mem_mgr.unpin_climb_data();
  mem_mgr.unpin_climb_parents();

  cuchk(cudaStreamSynchronize(jc.join_stream));
}

void direct_select_write_pipeline()
{
  std::cout << "Entering direct select write pipeline." << std::endl;
  auto &utm = UnifiedTrieManager::getInstance();
  auto &mem_mgr = MemoryManager::getInstance();
  auto &host_mgr = HostMemoryManager::getInstance();
  auto &ht = HashTable::getInstance();
  auto &jc = JoinContext::getInstance();

  if (!jc.direct_select_context())
  {
    direct_select_pipeline();
    return;
  }

  uint64_t *d_result_; // device result for each stream
  uint64_t *d_result_agg;
  mem_mgr.allocate_temporal(d_result_, sizeof(uint64_t) * JoinContext::nStreams);
  mem_mgr.allocate_temporal(d_result_agg, sizeof(uint64_t));
  cuchk(cudaMemsetAsync(d_result_, 0, sizeof(uint64_t) * JoinContext::nStreams, jc.join_stream));
  cuchk(cudaMemsetAsync(d_result_agg, 0, sizeof(uint64_t), jc.join_stream));

  size_t free_mem_total = std::max(
      mem_mgr.stats.available_gpu_memory / 100 * 95,       // 95% of available memory
      mem_mgr.stats.gpu_memory_limit - 100 * 1024 * 1024); // leave 100MB margin
  size_t free_mem_each_stream = free_mem_total / JoinContext::nStreams;
  uint64_t num_tiles_per_stream = free_mem_each_stream / sizeof(uint32_t);
  uint64_t rows_per_stream = num_tiles_per_stream * 32;
  uint64_t cur_offset = 0;

  void *d_temp_storage_base = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceReduce::Sum(d_temp_storage_base, temp_storage_bytes, (uint32_t *)nullptr,
                         (uint32_t *)nullptr, num_tiles_per_stream, jc.join_stream);
  temp_storage_bytes = (temp_storage_bytes + 255) & ~255; // align to 256 bytes
  mem_mgr.allocate_temporal(d_temp_storage_base, temp_storage_bytes * JoinContext::nStreams);

  uint32_t *d_partial_mask_;
  mem_mgr.allocate_temporal(d_partial_mask_, sizeof(uint32_t) * num_tiles_per_stream * JoinContext::nStreams);

  using PopcIterator = cub::TransformInputIterator<uint32_t, PopcToCount, uint32_t *>;

  int target_mask_block_id = jc.target_trie->id * 2;
  uint32_t *host_mask = (uint32_t *)mem_mgr.memory_blocks[target_mask_block_id].host_ptr;

  while (cur_offset < jc.num_frag_rows)
  {
    for (int i = 0; i < JoinContext::nStreams && cur_offset < jc.num_frag_rows; ++i)
    {
      uint64_t num_rows_this_iter = std::min(rows_per_stream, jc.num_frag_rows - cur_offset);
      uint64_t num_tiles_this_iter = (num_rows_this_iter + 31) >> 5;

      dim3 grid_dim;
      dim3 block_dim;
      compute_pipeline_launch(num_rows_this_iter, grid_dim, block_dim);
      select_trie_kernel_pipeline<<<grid_dim, block_dim, 0, jc.pipeline_streams[i]>>>(
          jc.d_frag_key_array_, jc.d_frag_value_array_,
          jc.d_parent_arrays_, jc.frag_trie->num_cols, jc.col_key, jc.col_value,
          cur_offset, num_rows_this_iter,
          // edge
          jc.d_edge_keys_, jc.d_edge_values_, jc.d_edge_offsets_,
          // hash
          ht.h_num_buckets_[jc.directed_edge_idx], ht.h_hash_table_offs_[jc.directed_edge_idx], jc.directed_edge_idx,
          // result
          d_partial_mask_ + num_tiles_per_stream * i);

      // gonna do cub reduction here...

      void *d_stream_storage = (char *)d_temp_storage_base + temp_storage_bytes * i;
      size_t stream_storage_bytes = temp_storage_bytes;
      PopcIterator popc_iter(d_partial_mask_ + num_tiles_per_stream * i, PopcToCount{});

      cub::DeviceReduce::Sum(d_stream_storage, stream_storage_bytes, popc_iter,
                             d_result_ + i, num_tiles_this_iter, jc.pipeline_streams[i]);
      host_atomicAdd<<<1, 1, 0, jc.pipeline_streams[i]>>>(d_result_agg, d_result_ + i);
      cuchk(cudaMemcpyAsync(
          host_mask + (cur_offset >> 5), d_partial_mask_ + num_tiles_per_stream * i,
          sizeof(uint32_t) * ((num_rows_this_iter + 31) >> 5),
          cudaMemcpyDeviceToHost, jc.pipeline_streams[i]));
      cur_offset += num_rows_this_iter;
    }
  }
  for (int i = 0; i < JoinContext::nStreams; ++i)
  {
    cuchk(cudaStreamSynchronize(jc.pipeline_streams[i]));
  }
  // Copy result back
  cuchk(cudaMemcpyAsync(jc.h_num_res, d_result_agg, sizeof(uint64_t),
                        cudaMemcpyDeviceToHost, jc.join_stream));

  mem_mgr.deallocate_temporal(d_temp_storage_base);
  mem_mgr.deallocate_temporal(d_partial_mask_);
  mem_mgr.deallocate_temporal(d_result_);
  mem_mgr.deallocate_temporal(d_result_agg);
  mem_mgr.unpin_climb_parents();

  cuchk(cudaStreamSynchronize(jc.join_stream));
}

void direct_select()
{
  auto &utm = UnifiedTrieManager::getInstance();
  auto &mem_mgr = MemoryManager::getInstance();
  auto &ht = HashTable::getInstance();
  auto &jc = JoinContext::getInstance();

  if (!jc.direct_select_context())
  {
    direct_select_pipeline();
    return;
  }

  std::cout << "Performing Direct Select" << std::endl;
#ifndef NDEBUG
  std::cout << "Performing Direct Select" << std::endl;
#endif

#ifndef NDEBUG
  std::cout << "preparing select, num_rows: " << jc.num_frag_rows << std::endl;
#endif

  select_trie_kernel<<<GRID_DIM, BLOCK_DIM, 0, jc.join_stream>>>(
      // frag
      jc.d_frag_key_array_, jc.d_frag_value_array_,
      jc.d_parent_arrays_, jc.frag_trie->num_cols, jc.col_key, jc.col_value,
      jc.num_frag_rows,
      // edge
      jc.d_edge_keys_, jc.d_edge_values_, jc.d_edge_offsets_,
      // hash
      ht.h_num_buckets_[jc.directed_edge_idx], ht.h_hash_table_offs_[jc.directed_edge_idx], jc.directed_edge_idx,
      // result
      jc.result_mask_);
  mem_mgr.unpin_climb_parents();
  cuchk(cudaStreamSynchronize(jc.join_stream));
}

void direct_select_debug()
{
  direct_select();
}

void refer_select_pipeline()
{
  std::cout << "Entering refer select pipeline." << std::endl;

  auto &mem_mgr = MemoryManager::getInstance();
  auto &ht = HashTable::getInstance();
  auto &jc = JoinContext::getInstance();
  auto &host_mgr = HostMemoryManager::getInstance();

  jc.refer_select_pipeline_context();

  const int nStreams = JoinContext::nStreams;
  const bool output_on_host = (mem_mgr.p_type == ProcessType::PIPELINE);
  uint32_t *host_mask = nullptr;
  if (output_on_host)
  {
    const int target_mask_block_id = jc.target_trie->id * 2;
    host_mask = static_cast<uint32_t *>(mem_mgr.memory_blocks[target_mask_block_id].host_ptr);
    if (host_mask == nullptr)
    {
      std::cerr << "refer_select_pipeline: missing host mask for target trie " << jc.target_trie->id << std::endl;
      exit(EXIT_FAILURE);
    }
  }

  const uint32_t num_cols = jc.root_trie->num_cols;
  const uint32_t col_upper_bound = std::min(jc.col_key, jc.col_value);

  const size_t num_partial_parents = jc.partial_parent_cols.size();
  const size_t num_partial_data = jc.partial_data_cols.size();

  const bool select_mask_on_device = (jc.select_bool_mask_ != nullptr);
  uint32_t *host_select_mask = nullptr;
  if (!select_mask_on_device)
  {
    const int mask_block_id = jc.frag_trie->id * 2;
    host_select_mask = static_cast<uint32_t *>(mem_mgr.memory_blocks[mask_block_id].host_ptr);
    if (host_select_mask == nullptr)
    {
      std::cerr << "refer_select_pipeline: missing host select mask for trie " << jc.frag_trie->id << std::endl;
      exit(EXIT_FAILURE);
    }
  }

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

  const uint64_t num_tiles_per_stream = (rows_per_stream + 31) >> 5;
  if (num_tiles_per_stream == 0)
  {
    std::cerr << "refer_select_pipeline: insufficient GPU memory for pipelining." << std::endl;
    exit(EXIT_FAILURE);
  }

  uint64_t *d_result_stream = nullptr;
  uint64_t *d_result_agg = nullptr;
  mem_mgr.allocate_temporal(d_result_stream, sizeof(uint64_t) * nStreams);
  mem_mgr.allocate_temporal(d_result_agg, sizeof(uint64_t));
  cuchk(cudaMemsetAsync(d_result_stream, 0, sizeof(uint64_t) * nStreams, jc.join_stream));
  cuchk(cudaMemsetAsync(d_result_agg, 0, sizeof(uint64_t), jc.join_stream));

  void *d_temp_storage_base = nullptr;
  size_t temp_storage_bytes = 0;
  using PopcIterator = cub::TransformInputIterator<uint32_t, PopcToCount, uint32_t *>;
  cub::DeviceReduce::Sum(d_temp_storage_base, temp_storage_bytes, PopcIterator(nullptr, PopcToCount{}),
                         d_result_stream, num_tiles_per_stream, jc.join_stream);
  temp_storage_bytes = (temp_storage_bytes + 255) & ~255;
  mem_mgr.allocate_temporal(d_temp_storage_base, temp_storage_bytes * nStreams);

  uint32_t *d_mask_tile_base = nullptr;
  if (output_on_host)
    mem_mgr.allocate_temporal(d_mask_tile_base, sizeof(uint32_t) * num_tiles_per_stream * nStreams);

  uint32_t *d_select_mask_tile_base = nullptr;
  if (!select_mask_on_device)
    mem_mgr.allocate_temporal(d_select_mask_tile_base, sizeof(uint32_t) * num_tiles_per_stream * nStreams);

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

  const size_t parent_info_count = jc.h_parent_info_arrays_.size();
  const bool need_parent_updates = (num_partial_parents > 0);
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
        std::cerr << "refer_select_pipeline: host allocation failed for parent JoinInfo buffers." << std::endl;
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
      uint64_t num_tiles_this_iter = (num_rows_this_iter + 31) >> 5;
      cudaStream_t stream = jc.pipeline_streams[stream_id];

      compute_column_ranges(jc.root_trie, col_upper_bound, cur_offset, cur_offset + num_rows_this_iter, ranges, jc.join_stream);

      uint32_t *d_select_mask_ptr = nullptr;
      if (select_mask_on_device)
        d_select_mask_ptr = jc.select_bool_mask_ + (cur_offset >> 5);
      else
      {
        d_select_mask_ptr = d_select_mask_tile_base + num_tiles_per_stream * static_cast<size_t>(stream_id);
        cuchk(cudaMemcpyAsync(d_select_mask_ptr, host_select_mask + (cur_offset >> 5),
                              sizeof(uint32_t) * num_tiles_this_iter,
                              cudaMemcpyHostToDevice, stream));
      }

      JoinInfo<uint64_t> *d_parent_info_ptr = jc.d_parent_info_arrays_;
      if (need_parent_updates)
      {
        JoinInfo<uint64_t> *h_parent_infos = h_parent_info_arrays[stream_id];
        for (size_t col = 0; col < parent_info_count; ++col)
          h_parent_infos[col] = jc.h_parent_info_arrays_[col];

        for (size_t i = 0; i < num_partial_parents; ++i)
        {
          uint32_t col = jc.partial_parent_cols[i];
          uint32_t col_trie_id = jc.root_trie->column_ids[col];
          const int block_idx = static_cast<int>(col_trie_id) * 2 + 1;
          auto &block = mem_mgr.memory_blocks[block_idx];
          if (block.host_ptr == nullptr)
          {
            std::cerr << "refer_select_pipeline: missing host parents for trie " << col_trie_id << std::endl;
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

      JoinInfo<uint32_t> key_info = jc.h_data_info_arrays_[jc.col_key];
      JoinInfo<uint32_t> val_info = jc.h_data_info_arrays_[jc.col_value];
      for (size_t i = 0; i < num_partial_data; ++i)
      {
        uint32_t col = jc.partial_data_cols[i];
        uint32_t col_trie_id = jc.root_trie->column_ids[col];
        const int block_idx = static_cast<int>(col_trie_id) * 2;
        auto &block = mem_mgr.memory_blocks[block_idx];
        if (block.host_ptr == nullptr)
        {
          std::cerr << "refer_select_pipeline: missing host data for trie " << col_trie_id << std::endl;
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
          if (col == jc.col_key)
          {
            key_info.array_ptr = d_data_buf;
            key_info.offset = r.start;
            key_info.is_partial = 1;
          }
          if (col == jc.col_value)
          {
            val_info.array_ptr = d_data_buf;
            val_info.offset = r.start;
            val_info.is_partial = 1;
          }
        }
      }

      uint32_t *d_out_mask = nullptr;
      if (output_on_host)
        d_out_mask = d_mask_tile_base + num_tiles_per_stream * static_cast<size_t>(stream_id);
      else
        d_out_mask = jc.result_mask_ + (cur_offset >> 5);

      dim3 grid_dim;
      dim3 block_dim;
      compute_pipeline_launch(num_rows_this_iter, grid_dim, block_dim);
      select_trie_refer_kernel_pipeline_info<<<grid_dim, block_dim, 0, stream>>>(
          key_info, val_info,
          d_parent_info_ptr, num_cols, jc.col_key, jc.col_value,
          d_select_mask_ptr,
          cur_offset, num_rows_this_iter,
          jc.d_edge_keys_, jc.d_edge_values_, jc.d_edge_offsets_,
          ht.h_num_buckets_[jc.directed_edge_idx], ht.h_hash_table_offs_[jc.directed_edge_idx], jc.directed_edge_idx,
          d_out_mask);

      PopcIterator popc_iter(d_out_mask, PopcToCount{});
      void *d_stream_storage = static_cast<char *>(d_temp_storage_base) + temp_storage_bytes * stream_id;
      size_t stream_storage_bytes = temp_storage_bytes;
      cub::DeviceReduce::Sum(d_stream_storage, stream_storage_bytes, popc_iter,
                             d_result_stream + stream_id, num_tiles_this_iter, stream);
      host_atomicAdd<<<1, 1, 0, stream>>>(d_result_agg, d_result_stream + stream_id);

      if (output_on_host)
      {
        cuchk(cudaMemcpyAsync(host_mask + (cur_offset >> 5), d_out_mask,
                              sizeof(uint32_t) * num_tiles_this_iter,
                              cudaMemcpyDeviceToHost, stream));
      }

      cuchk(cudaEventRecord(done_events[stream_id], stream));
      cur_offset += num_rows_this_iter;
    }
  }

  for (int i = 0; i < nStreams; ++i)
  {
    cuchk(cudaEventSynchronize(done_events[i]));
    cuchk(cudaEventDestroy(done_events[i]));
  }

  cuchk(cudaMemcpyAsync(jc.h_num_res, d_result_agg, sizeof(uint64_t),
                        cudaMemcpyDeviceToHost, jc.join_stream));

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
  if (d_partial_parent_base != nullptr)
    mem_mgr.deallocate_temporal(d_partial_parent_base);
  if (d_partial_data_base != nullptr)
    mem_mgr.deallocate_temporal(d_partial_data_base);
  if (d_select_mask_tile_base != nullptr)
    mem_mgr.deallocate_temporal(d_select_mask_tile_base);
  if (d_mask_tile_base != nullptr)
    mem_mgr.deallocate_temporal(d_mask_tile_base);
  if (d_temp_storage_base != nullptr)
    mem_mgr.deallocate_temporal(d_temp_storage_base);
  if (d_result_stream != nullptr)
    mem_mgr.deallocate_temporal(d_result_stream);
  if (d_result_agg != nullptr)
    mem_mgr.deallocate_temporal(d_result_agg);
  mem_mgr.unpin_climb_data();
  mem_mgr.unpin_climb_parents();

  cuchk(cudaStreamSynchronize(jc.join_stream));
}

void refer_select_write_pipeline()
{
  std::cout << "Entering refer select write pipeline." << std::endl;

  auto &utm = UnifiedTrieManager::getInstance();
  auto &mem_mgr = MemoryManager::getInstance();
  auto &host_mgr = HostMemoryManager::getInstance();
  auto &ht = HashTable::getInstance();
  auto &jc = JoinContext::getInstance();

  if (!jc.refer_select_context())
  {
    refer_select_pipeline();
    return;
  }

  uint64_t *d_result_; // device result for each stream
  uint64_t *d_result_agg;
  mem_mgr.allocate_temporal(d_result_, sizeof(uint64_t) * JoinContext::nStreams);
  mem_mgr.allocate_temporal(d_result_agg, sizeof(uint64_t));
  cuchk(cudaMemsetAsync(d_result_, 0, sizeof(uint64_t) * JoinContext::nStreams, jc.join_stream));
  cuchk(cudaMemsetAsync(d_result_agg, 0, sizeof(uint64_t), jc.join_stream));

  size_t free_mem_total = std::max(
      mem_mgr.stats.available_gpu_memory / 100 * 95,       // 95% of available memory
      mem_mgr.stats.gpu_memory_limit - 100 * 1024 * 1024); // leave 100MB margin
  size_t free_mem_each_stream = free_mem_total / JoinContext::nStreams;
  uint64_t num_tiles_per_stream = free_mem_each_stream / sizeof(uint32_t);
  uint64_t rows_per_stream = num_tiles_per_stream * 32;
  uint64_t cur_offset = 0;

  void *d_temp_storage_base = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceReduce::Sum(d_temp_storage_base, temp_storage_bytes, (uint32_t *)nullptr,
                         (uint32_t *)nullptr, num_tiles_per_stream, jc.join_stream);
  temp_storage_bytes = (temp_storage_bytes + 255) & ~255; // align to 256 bytes
  mem_mgr.allocate_temporal(d_temp_storage_base, temp_storage_bytes * JoinContext::nStreams);

  uint32_t *d_partial_mask_;
  mem_mgr.allocate_temporal(d_partial_mask_, sizeof(uint32_t) * num_tiles_per_stream * JoinContext::nStreams);

  using PopcIterator = cub::TransformInputIterator<uint32_t, PopcToCount, uint32_t *>;

  int target_mask_block_id = jc.target_trie->id * 2;
  uint32_t *host_mask = (uint32_t *)mem_mgr.memory_blocks[target_mask_block_id].host_ptr;

  while (cur_offset < jc.num_frag_rows)
  {
    for (int i = 0; i < JoinContext::nStreams && cur_offset < jc.num_frag_rows; ++i)
    {
      uint64_t num_rows_this_iter = std::min(rows_per_stream, jc.num_frag_rows - cur_offset);
      uint64_t num_tiles_this_iter = (num_rows_this_iter + 31) >> 5;

      dim3 grid_dim;
      dim3 block_dim;
      compute_pipeline_launch(num_rows_this_iter, grid_dim, block_dim);
      select_trie_refer_kernel_pipeline<<<grid_dim, block_dim, 0, jc.pipeline_streams[i]>>>(
          // original data to refer to
          jc.d_frag_key_array_, jc.d_frag_value_array_,
          jc.d_parent_arrays_, jc.root_trie->num_cols, jc.col_key, jc.col_value,
          // selection mask
          jc.select_bool_mask_,
          cur_offset, num_rows_this_iter,
          // trie_edge
          jc.d_edge_keys_, jc.d_edge_values_, jc.d_edge_offsets_,
          // hash
          ht.h_num_buckets_[jc.directed_edge_idx], ht.h_hash_table_offs_[jc.directed_edge_idx], jc.directed_edge_idx,
          // result
          d_partial_mask_ + num_tiles_per_stream * i);

      // gonna do cub reduction here...

      void *d_stream_storage = (char *)d_temp_storage_base + temp_storage_bytes * i;
      size_t stream_storage_bytes = temp_storage_bytes;
      PopcIterator popc_iter(d_partial_mask_ + num_tiles_per_stream * i, PopcToCount{});

      cub::DeviceReduce::Sum(d_stream_storage, stream_storage_bytes, popc_iter,
                             d_result_ + i, num_tiles_this_iter, jc.pipeline_streams[i]);
      host_atomicAdd<<<1, 1, 0, jc.pipeline_streams[i]>>>(d_result_agg, d_result_ + i);
      cuchk(cudaMemcpyAsync(
          host_mask + (cur_offset >> 5), d_partial_mask_ + num_tiles_per_stream * i,
          sizeof(uint32_t) * ((num_rows_this_iter + 31) >> 5),
          cudaMemcpyDeviceToHost, jc.pipeline_streams[i]));
      cur_offset += num_rows_this_iter;
    }
  }
  for (int i = 0; i < JoinContext::nStreams; ++i)
  {
    cuchk(cudaStreamSynchronize(jc.pipeline_streams[i]));
  }
  // Copy result back
  cuchk(cudaMemcpyAsync(jc.h_num_res, d_result_agg, sizeof(uint64_t),
                        cudaMemcpyDeviceToHost, jc.join_stream));

  mem_mgr.deallocate_temporal(d_temp_storage_base);
  mem_mgr.deallocate_temporal(d_partial_mask_);
  mem_mgr.deallocate_temporal(d_result_);
  mem_mgr.deallocate_temporal(d_result_agg);
  mem_mgr.unpin_climb_parents();

  cuchk(cudaStreamSynchronize(jc.join_stream));
}

void refer_select()
{
  auto &utm = UnifiedTrieManager::getInstance();
  auto &mem_mgr = MemoryManager::getInstance();
  auto &ht = HashTable::getInstance();
  auto &jc = JoinContext::getInstance();

  std::cout << "Performing Refer Select" << std::endl;
#ifndef NDEBUG
  std::cout << "Performing Refer Select" << std::endl;
#endif

  if (!jc.refer_select_context())
  {
    refer_select_pipeline();
    return;
  }

  select_trie_refer_kernel<<<GRID_DIM, BLOCK_DIM, 0, jc.join_stream>>>(
      // original data to refer to
      jc.d_frag_key_array_, jc.d_frag_value_array_,
      jc.d_parent_arrays_, jc.root_trie->num_cols, jc.col_key, jc.col_value,
      // selection mask
      jc.select_bool_mask_, jc.num_frag_rows,
      // trie_edge
      jc.d_edge_keys_, jc.d_edge_values_, jc.d_edge_offsets_,
      // hash
      ht.h_num_buckets_[jc.directed_edge_idx], ht.h_hash_table_offs_[jc.directed_edge_idx], jc.directed_edge_idx,
      // result
      jc.result_mask_);
  mem_mgr.unpin_climb_parents();

  cuchk(cudaStreamSynchronize(jc.join_stream));
}

void refer_select_debug()
{
  refer_select();
}

void host_select()
{
  auto lat = &Lattice::getInstance();
  auto utm = &UnifiedTrieManager::getInstance();
  auto mem_mgr = &MemoryManager::getInstance();
  auto ht = &HashTable::getInstance();
  auto &jc = JoinContext::getInstance();

  jc.host_select_general_context();

  // Check if fragment has a mask parent (reference chain)
  if (jc.frag_trie->mask_parent_id == UINT32_MAX)
  {
#ifndef NDEBUG
    direct_select_debug();
#else
    if (mem_mgr->p_type == ProcessType::DIRECT)
      direct_select();
    else if (mem_mgr->p_type == ProcessType::PIPELINE)
      direct_select_write_pipeline();
    else
      std::cout << "FALLBACK to cpu algorithm" << std::endl;
#endif
  }
  else
  {
#ifndef NDEBUG
    refer_select_debug();
#else
    if (mem_mgr->p_type == ProcessType::DIRECT)
      refer_select();
    else if (mem_mgr->p_type == ProcessType::PIPELINE)
      refer_select_write_pipeline();
    else
      std::cout << "FALLBACK to cpu algorithm" << std::endl;
#endif
  }

  if (mem_mgr->p_type == ProcessType::DIRECT)
  {
    uint64_t num_compressed_blocks = jc.target_trie->data->num_compressed_blocks;
    // Count the number of set bits using CUB

    using PopcIterator = cub::TransformInputIterator<uint32_t, PopcToCount, uint32_t *>;
    PopcIterator popc_iter(jc.result_mask_, PopcToCount{});

    size_t temp_storage_bytes = 0;
    void *d_temp_storage = nullptr;
    uint64_t *d_result;
    mem_mgr->allocate_temporal(d_result, sizeof(uint64_t));
    cuchk(cudaMemsetAsync(d_result, 0, sizeof(uint64_t), jc.join_stream));

    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, popc_iter,
                           d_result, num_compressed_blocks, jc.join_stream);
    mem_mgr->allocate_temporal(d_temp_storage, temp_storage_bytes);
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, popc_iter,
                           d_result, num_compressed_blocks, jc.join_stream);

    // Copy result back
    cuchk(cudaMemcpyAsync(jc.h_num_res, d_result, sizeof(uint64_t),
                          cudaMemcpyDeviceToHost, jc.join_stream));
    // Cleanup
    mem_mgr->deallocate_temporal(d_temp_storage);
    mem_mgr->deallocate_temporal(d_result);
    cuchk(cudaStreamSynchronize(jc.join_stream));
  }

  // Update target properties
  jc.target_trie->updateNumResults(*jc.h_num_res);
  // target_trie->data->num_rows = *jc.h_num_res;

  if (jc.skip_final_materialization)
  {
    mem_mgr->deallocate_trie(jc.target_trie->id);
  }
  cuchk(cudaStreamSynchronize(jc.join_stream));
  // reset process type to DIRECT
  mem_mgr->p_type = ProcessType::DIRECT;

#ifndef NDEBUG
  std::cout << "select: " << jc.h_num_res[0] << " results selected" << std::endl;
#endif
}
