#include "join_context.cuh"
#include "host_memory_manager.cuh"

#include <algorithm>
#include <iostream>
#include <string>
#include <stdexcept>

namespace
{
  template <typename Fn>
  void swallow(Fn &&fn) noexcept
  {
    try
    {
      fn();
    }
    catch (...)
    {
    }
  }

  void syncStream(cudaStream_t stream) noexcept
  {
    if (stream != nullptr)
      cudaStreamSynchronize(stream);
  }

  void destroyStream(cudaStream_t &stream) noexcept
  {
    if (stream != nullptr)
    {
      cudaStreamDestroy(stream);
      stream = nullptr;
    }
  }

  template <typename T>
  void deleteArray(T *&ptr) noexcept
  {
    if (ptr != nullptr)
    {
      delete[] ptr;
      ptr = nullptr;
    }
  }

  template <typename T>
  void deleteObject(T *&ptr) noexcept
  {
    if (ptr != nullptr)
    {
      delete ptr;
      ptr = nullptr;
    }
  }

  template <typename T>
  void freeAsyncIf(cudaStream_t stream, T *&ptr) noexcept
  {
    if (ptr != nullptr && stream != nullptr)
    {
      cudaFreeAsync(ptr, stream);
      ptr = nullptr;
    }
  }
} // namespace

void JoinContext::init_tries(UnifiedTrie *frag, UnifiedTrie *probe, UnifiedTrie *target)
{
  frag_trie = frag;
  probe_trie = probe;
  target_trie = target;
}

void JoinContext::clear()
{
  frag_trie = nullptr;
  probe_trie = nullptr;
  target_trie = nullptr;
  root_trie = nullptr;

  col_key = UINT32_MAX;
  col_value = UINT32_MAX;

  d_frag_key_array_ = nullptr;
  d_frag_value_array_ = nullptr;
  d_edge_keys_ = nullptr;
  d_edge_values_ = nullptr;
  d_edge_offsets_ = nullptr;
  edge_forward = true;
  directed_edge_idx = UINT32_MAX;

  d_data_arrays_ = nullptr;
  d_parent_arrays_ = nullptr;
  d_data_info_arrays_ = nullptr;
  d_parent_info_arrays_ = nullptr;

  num_frag_rows = 0;
  num_frag_cols = 0;
  num_root_cols = 0;
  select_bool_mask_ = nullptr;
  result_mask_ = nullptr;

  d_dup_cols = nullptr;
  num_dup_cols = 0;
  h_dup_cols = nullptr;
  skip_final_materialization = false;

  // d_key_array_ = nullptr;
  // d_value_array_ = nullptr;
  hash_table_ = nullptr;

  found[0] = found[1] = false;
  col[0] = col[1] = UINT32_MAX;
  is_subset = false;

  h_data_info_arrays_.clear();
  h_parent_info_arrays_.clear();
  partial_data_cols.clear();
  partial_parent_cols.clear();
  pin_cols.clear();

  d_lookup_ = nullptr;
  d_num_res_for_each = nullptr;
  d_st_array = nullptr;
  h_num_res = nullptr;
  d_mask = nullptr;
  mask_length = 0;

  join_stream = nullptr;
  for (int i = 0; i < nStreams; ++i)
  {
    pipeline_streams[i] = nullptr;
  }
}

void JoinContext::join_context()
{
  clear();
  auto &mem_mgr = MemoryManager::getInstance();
  auto &ht = HashTable::getInstance();

  cuchk(cudaStreamCreate(&join_stream));
  mem_mgr.set_stream(&join_stream);
  for (int i = 0; i < nStreams; ++i)
    cuchk(cudaStreamCreate(&pipeline_streams[i]));

  // mem_mgr doesn't track this.
  cuchk(cudaMallocAsync((void **)&d_lookup_, sizeof(HashLookupTables), join_stream));

  // Create host-side structure and populate with device pointers from HashTable
  HashLookupTables h_lookup_stack;
  h_lookup_stack.hash_constants = ht.d_hash_constants_;
  for (int i = 0; i < NUM_TABLES; ++i)
  {
    h_lookup_stack.keys[i] = ht.d_keys_[i];
    h_lookup_stack.pos[i] = ht.d_pos_in_org[i];
  }

  // Copy the structure (containing device pointers) from host to device
  cuchk(cudaMemcpyAsync(d_lookup_, &h_lookup_stack, sizeof(HashLookupTables), cudaMemcpyHostToDevice, join_stream));

  // Set the __constant__ pointer to point to the device structure
  cuchk(cudaMemcpyToSymbolAsync(lookup, &d_lookup_, sizeof(HashLookupTables *), 0, cudaMemcpyHostToDevice, join_stream));

  mem_mgr.allocate_permanent(d_dup_cols, sizeof(uint32_t) * NUM_VQ);
  h_dup_cols = new uint32_t[NUM_VQ];

  cuchk(cudaMallocAsync((void **)&d_data_arrays_, sizeof(vtype *) * MAX_VQ, join_stream));
  cuchk(cudaMallocAsync((void **)&d_parent_arrays_, sizeof(uint64_t *) * MAX_VQ, join_stream));
  cuchk(cudaMallocAsync((void **)&d_data_info_arrays_, sizeof(JoinInfo<uint32_t>) * MAX_VQ, join_stream));
  cuchk(cudaMallocAsync((void **)&d_parent_info_arrays_, sizeof(JoinInfo<uint64_t>) * MAX_VQ, join_stream));
  h_data_info_arrays_.resize(MAX_VQ);
  h_parent_info_arrays_.resize(MAX_VQ);

  h_num_res = new uint64_t;
  *h_num_res = 0;
}

void JoinContext::cleanup() noexcept
{
  auto &mem_mgr = MemoryManager::getInstance();

  for (int i = 0; i < nStreams; ++i)
  {
    swallow([&]() { syncStream(pipeline_streams[i]); });
  }
  swallow([&]() { syncStream(join_stream); });

  swallow([&]() { mem_mgr.deallocate_temporal(d_mask); });
  swallow([&]() { mem_mgr.deallocate_temporal(d_num_res_for_each); });
  swallow([&]() { mem_mgr.deallocate_temporal(d_st_array); });

  swallow([&]() { mem_mgr.deallocate_permanent(d_dup_cols); });
  swallow([&]() { deleteArray(h_dup_cols); });
  swallow([&]() { deleteObject(h_num_res); });

  swallow([&]() { freeAsyncIf(join_stream, d_data_arrays_); });
  swallow([&]() { freeAsyncIf(join_stream, d_parent_arrays_); });
  swallow([&]() { freeAsyncIf(join_stream, d_data_info_arrays_); });
  swallow([&]() { freeAsyncIf(join_stream, d_parent_info_arrays_); });
  swallow([&]() { freeAsyncIf(join_stream, d_lookup_); });

  swallow([&]() { syncStream(join_stream); });

  for (int i = 0; i < nStreams; ++i)
  {
    swallow([&]() { destroyStream(pipeline_streams[i]); });
  }

  swallow([&]() { mem_mgr.reset_stream(); });
  swallow([&]() { destroyStream(join_stream); });
}

void JoinContext::process_context()
{
  found[0] = found[1] = false;
  col[0] = col[1] = UINT32_MAX;
  is_subset = false;

  for (int i = 0; i < frag_trie->num_cols; ++i)
  {
    for (int j = 0; j < probe_trie->num_cols; ++j)
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
}

void JoinContext::process_is_subset_context()
{
  col_key = std::min(col[0], col[1]);
  col_value = std::max(col[0], col[1]);
}

void JoinContext::process_refer_join_context()
{
  // Subset selection does not add a vertex; keep column count equal to the fragment.
  target_trie->num_cols++;
  col_key = found[0] ? col[0] : col[1];
  edge_forward = found[0];
}

void JoinContext::process_direct_join_context()
{
  // Subset selection does not add a vertex; keep column count equal to the fragment.
  target_trie->num_cols++;
  col_key = found[0] ? col[0] : col[1];
  edge_forward = found[0];
}

void JoinContext::host_select_general_context()
{
  auto &mem_mgr = MemoryManager::getInstance();
  auto &utm = UnifiedTrieManager::getInstance();
  auto &ht = HashTable::getInstance();
  auto &lat = Lattice::getInstance();

  edge_forward = frag_trie->mapped_query_us_[col_key] == probe_trie->edge_vertices[0];

  etype edge_idx = probe_trie->et._Find_first();
  directed_edge_idx = edge_idx * 2 + (edge_forward ? 0 : 1);

  vtype key_u = probe_trie->edge_vertices[0];
  vtype value_u = probe_trie->edge_vertices[1];
  if (!edge_forward)
    std::swap(key_u, value_u);

  // Get device pointers for fragment data
  if (!mem_mgr.is_data_loaded(frag_trie->column_ids[col_key]))
    mem_mgr.load_data(frag_trie->column_ids[col_key]);
  if (!mem_mgr.is_data_loaded(frag_trie->column_ids[col_value]))
    mem_mgr.load_data(frag_trie->column_ids[col_value]);
  d_frag_key_array_ = utm.getTrie(frag_trie->column_ids[col_key])->data->d_data;
  d_frag_value_array_ = utm.getTrie(frag_trie->column_ids[col_value])->data->d_data;
  // Get edge trie offsets and data
  d_edge_offsets_ = ht.d_offsets_ + ht.edge_offset_starts_[directed_edge_idx];

  d_edge_values_ = nullptr;
  d_edge_keys_ = utm.getTrie(lat.get_vertex_node_id(key_u))->data->d_data;
  if (edge_forward)
    d_edge_values_ = probe_trie->data->d_data;
  else
    d_edge_values_ = probe_trie->r_data->d_data;

  utm.linkMask(target_trie->id, frag_trie->id);
  if (mem_mgr.p_type == ProcessType::DIRECT)
  {
    result_mask_ = target_trie->data->d_data;

    uint64_t num_compressed_blocks = target_trie->data->num_compressed_blocks;
    cuchk(cudaMemsetAsync(result_mask_, 0, sizeof(uint32_t) * num_compressed_blocks, join_stream));
  }
  else if (mem_mgr.p_type == ProcessType::PIPELINE)
  {
    auto &host_mgr = HostMemoryManager::getInstance();
    void *h_result_mask = host_mgr.allocateManagedBytes(
        sizeof(uint32_t) * target_trie->data->num_compressed_blocks,
        sizeof(uint32_t));
    if (h_result_mask == nullptr)
    {
      throw std::runtime_error("JoinContext: Host memory allocation failed for result mask of trie " +
                               std::to_string(target_trie->id));
    }
    int block_idx = target_trie->id * 2;
    auto &block = mem_mgr.memory_blocks[block_idx];
    block.host_ptr = h_result_mask;
    block.device_ptr = nullptr;
    block.size_in_bytes = sizeof(uint32_t) * target_trie->data->num_compressed_blocks;
  }
  else
  {
    throw std::runtime_error("Fall Back to CPU Processing due to insufficient GPU memory.");
  }
}

void JoinContext::direct_select_pipeline_context()
{
  up_to_col = std::min(col_key, col_value);
  std::vector<uint32_t> pinCols = {col_key, col_value};
  std::sort(pinCols.begin(), pinCols.end());
  pinCols.erase(std::unique(pinCols.begin(), pinCols.end()), pinCols.end());

  get_data_arrays_pipeline(frag_trie->id, pinCols, h_data_info_arrays_);
  get_parent_arrays_pipeline(frag_trie->id, up_to_col, h_parent_info_arrays_);

  cuchk(cudaMemcpyAsync(
      d_data_info_arrays_, h_data_info_arrays_.data(),
      sizeof(JoinInfo<uint32_t>) * h_data_info_arrays_.size(),
      cudaMemcpyHostToDevice, join_stream));
  cuchk(cudaMemcpyAsync(
      d_parent_info_arrays_, h_parent_info_arrays_.data(),
      sizeof(JoinInfo<uint64_t>) * h_parent_info_arrays_.size(),
      cudaMemcpyHostToDevice, join_stream));

  num_frag_rows = frag_trie->num_results;
}

bool JoinContext::direct_select_context()
{
  auto &utm = UnifiedTrieManager::getInstance();
  auto &mem_mgr = MemoryManager::getInstance();

  if (d_frag_key_array_ == nullptr || d_frag_value_array_ == nullptr)
    return false;

  up_to_col = std::min(col_key, col_value);
  auto d_parent_pointers = utm.get_parent_arrays(frag_trie->id, up_to_col);
  // auto d_parent_pointers = utm.get_parent_arrays(frag_trie->id);

  for (uint32_t col_id = up_to_col + 1; col_id < frag_trie->num_cols; ++col_id)
  {
    if (d_parent_pointers[col_id] == nullptr)
    {
      mem_mgr.unpin_climb_parents();
      return false;
    }
  }

  cuchk(cudaMemcpyAsync(d_parent_arrays_, d_parent_pointers.data(), sizeof(uint64_t *) * d_parent_pointers.size(), cudaMemcpyHostToDevice, join_stream));
  num_frag_rows = frag_trie->num_results;
  return true;
}

bool JoinContext::refer_select_context()
{
  auto &utm = UnifiedTrieManager::getInstance();
  auto &mem_mgr = MemoryManager::getInstance();

  root_trie = frag_trie;
  do
  {
    root_trie = utm.getTrie(root_trie->mask_parent_id);
  } while (root_trie->mask_parent_id != UINT32_MAX);

  // Get original data arrays from root_data
  uint32_t root_key_trie_id = root_trie->column_ids[col_key];
  uint32_t root_val_trie_id = root_trie->column_ids[col_value];
  if (!mem_mgr.is_data_loaded(root_key_trie_id) && !mem_mgr.load_data(root_key_trie_id))
    return false;
  if (!mem_mgr.is_data_loaded(root_val_trie_id) && !mem_mgr.load_data(root_val_trie_id))
    return false;
  d_frag_key_array_ = utm.getTrie(root_key_trie_id)->data->d_data;
  d_frag_value_array_ = utm.getTrie(root_val_trie_id)->data->d_data;
  if (d_frag_key_array_ == nullptr || d_frag_value_array_ == nullptr)
    return false;

  up_to_col = std::min(col_key, col_value);
  auto d_parent_pointers = utm.get_parent_arrays(root_trie->id, up_to_col);
  // auto d_parent_pointers = utm.get_parent_arrays(root_trie->id);

  for (uint32_t col_id = up_to_col + 1; col_id < root_trie->num_cols; ++col_id)
  {
    if (d_parent_pointers[col_id] == nullptr)
    {
      mem_mgr.unpin_climb_parents();
      return false;
    }
  }

  cuchk(cudaMemcpyAsync(d_parent_arrays_, d_parent_pointers.data(), sizeof(uint64_t *) * d_parent_pointers.size(), cudaMemcpyHostToDevice, join_stream));
  // uint32_t **d_original_parent_pointers = utm->parent_addr_list[root_data->id].data();

  // Get selection mask from fragment
  if (!mem_mgr.is_data_loaded(frag_trie->id) && !mem_mgr.load_data(frag_trie->id))
  {
    mem_mgr.unpin_climb_parents();
    return false;
  }
  select_bool_mask_ = frag_trie->data->d_data;
  if (select_bool_mask_ == nullptr)
  {
    mem_mgr.unpin_climb_parents();
    return false;
  }
  num_frag_rows = root_trie->num_results;
  return true;
}

void JoinContext::refer_select_pipeline_context()
{
  auto &utm = UnifiedTrieManager::getInstance();

  root_trie = frag_trie;
  do
  {
    root_trie = utm.getTrie(root_trie->mask_parent_id);
  } while (root_trie->mask_parent_id != UINT32_MAX);

  up_to_col = std::min(col_key, col_value);
  std::vector<uint32_t> pinCols = {col_key, col_value};
  std::sort(pinCols.begin(), pinCols.end());
  pinCols.erase(std::unique(pinCols.begin(), pinCols.end()), pinCols.end());

  get_data_arrays_pipeline(root_trie->id, pinCols, h_data_info_arrays_);
  get_parent_arrays_pipeline(root_trie->id, up_to_col, h_parent_info_arrays_);

  cuchk(cudaMemcpyAsync(
      d_data_info_arrays_, h_data_info_arrays_.data(),
      sizeof(JoinInfo<uint32_t>) * h_data_info_arrays_.size(),
      cudaMemcpyHostToDevice, join_stream));
  cuchk(cudaMemcpyAsync(
      d_parent_info_arrays_, h_parent_info_arrays_.data(),
      sizeof(JoinInfo<uint64_t>) * h_parent_info_arrays_.size(),
      cudaMemcpyHostToDevice, join_stream));

  select_bool_mask_ = frag_trie->data ? frag_trie->data->d_data : nullptr;
  num_frag_rows = root_trie->num_results;
}

bool JoinContext::direct_join_general_context()
{
  auto &utm = UnifiedTrieManager::getInstance();
  auto &mem_mgr = MemoryManager::getInstance();
  auto &ht = HashTable::getInstance();
  auto &lat = Lattice::getInstance();

  DataBlock *edge_data_block; // for edge value layer
  if (edge_forward)
  {
    d_edge_keys_ = utm.getTrie(lat.get_vertex_node_id(probe_trie->edge_vertices[0]))->data->d_data;
    edge_data_block = probe_trie->data;
  }
  else
  {
    d_edge_keys_ = utm.getTrie(lat.get_vertex_node_id(probe_trie->edge_vertices[1]))->data->d_data;
    edge_data_block = probe_trie->r_data;
  }
  d_edge_values_ = edge_data_block->d_data;

  vtype new_u = edge_forward ? probe_trie->edge_vertices[1] : probe_trie->edge_vertices[0];
  utm.linkExpansion(target_trie->id, frag_trie->id, 0, new_u);

  // Get fragment data pointers
  num_frag_rows = frag_trie->num_results;
#ifndef NDEBUG
  std::cout << "Fragment trie num rows: " << num_frag_rows << std::endl;
#endif

  // Get edge trie information
  etype edge_idx = probe_trie->et._Find_first();
  directed_edge_idx = edge_idx * 2 + (edge_forward ? 0 : 1);
  d_edge_offsets_ = ht.d_offsets_ + ht.edge_offset_starts_[directed_edge_idx];

  // Allocate counting arrays
  // *h_num_res = 0;

  dup_check_type = 0;
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

  pin_cols.clear();
  if (dup_check_type > 0)
    pin_cols.assign(h_dup_cols, h_dup_cols + num_dup_cols);
  bool join_col_pinned = false;
  for (auto c : pin_cols)
    if (c == col_key)
      join_col_pinned = true;
  if (!join_col_pinned)
    pin_cols.push_back(col_key);
  std::sort(pin_cols.begin(), pin_cols.end());

  up_to_col = UINT32_MAX;
  for (auto c : pin_cols)
    up_to_col = std::min(up_to_col, c);

  auto data_arrays = utm.get_data_arrays(frag_trie->id, pin_cols);
  auto parent_arrays = utm.get_parent_arrays(frag_trie->id, up_to_col);
  // auto parent_arrays = utm.get_parent_arrays(frag_trie->id);

  bool ready = true;
  for (auto col_id : pin_cols)
  {
    if (col_id >= data_arrays.size() || data_arrays[col_id] == nullptr)
    {
      ready = false;
      break;
    }
  }

  if (ready)
  {
    for (uint32_t col_id = frag_trie->num_cols - 1; col_id > up_to_col; --col_id)
    {
      if (col_id >= parent_arrays.size() || parent_arrays[col_id] == nullptr)
      {
        ready = false;
        break;
      }
    }
  }

  if (!ready)
  {
    mem_mgr.unpin_climb_data();
    mem_mgr.unpin_climb_parents();
    return false;
  }

  cuchk(cudaMemcpyAsync(d_data_arrays_, data_arrays.data(), sizeof(vtype *) * data_arrays.size(),
                        cudaMemcpyHostToDevice, join_stream));
  cuchk(cudaMemcpyAsync(d_parent_arrays_, parent_arrays.data(), sizeof(uint64_t *) * parent_arrays.size(),
                        cudaMemcpyHostToDevice, join_stream));

  if (!mem_mgr.allocate_temporal(d_num_res_for_each, sizeof(uint32_t) * (num_frag_rows + 1)))
  {
    mem_mgr.unpin_climb_data();
    mem_mgr.unpin_climb_parents();
    return false;
  }
  cuchk(cudaMemsetAsync(d_num_res_for_each, 0, sizeof(uint32_t) * (num_frag_rows + 1), join_stream));
  if (!mem_mgr.allocate_temporal(d_st_array, sizeof(uint32_t) * num_frag_rows))
  {
    mem_mgr.deallocate_temporal(d_num_res_for_each);
    mem_mgr.unpin_climb_data();
    mem_mgr.unpin_climb_parents();
    return false;
  }
  cuchk(cudaMemsetAsync(d_st_array, 0xFFFFFFFF, sizeof(uint32_t) * num_frag_rows, join_stream));

  return true;
}

void JoinContext::direct_join_without_duplication_context()
{
  auto &mem_mgr = MemoryManager::getInstance();
  auto &ht = HashTable::getInstance();

  if (num_frag_rows >= UINT32_MAX)
    task_per_warp = 32;
  else
  {
    uint32_t avg = num_frag_rows / (GRID_DIM * BLOCK_DIM / 32);
    task_per_warp = findClosestPowerOf2_Bitwise(avg);
    task_per_warp = std::min(task_per_warp, 32u);
  }

  assert(__builtin_popcount(task_per_warp) == 1);
}

void JoinContext::direct_join_with_duplication_context()
{
  auto &ht = HashTable::getInstance();
  auto &mem_mgr = MemoryManager::getInstance();

  // mask for avoiding 2nd binary search on edge values
  mask_length = ht.max_values_per_edge_[directed_edge_idx];
  // Convert to number of 32-bit words (round up) and align to 32 words (128 bytes) for coalescing
  mask_length = ((mask_length + 31) / 32 + 31) / 32 * 32;
  if (!mem_mgr.allocate_temporal(d_mask, sizeof(uint32_t) * num_frag_rows * mask_length))
  {
    d_mask = nullptr;
    return;
  }
  cuchk(cudaMemsetAsync(d_mask, 0, sizeof(uint32_t) * num_frag_rows * mask_length, join_stream));
}

bool JoinContext::refer_join_general_context()
{
  auto &utm = UnifiedTrieManager::getInstance();
  auto &mem_mgr = MemoryManager::getInstance();
  auto &ht = HashTable::getInstance();
  auto &lat = Lattice::getInstance();

  DataBlock *edge_data_block; // for edge value layer
  if (edge_forward)
  {
    d_edge_keys_ = utm.getTrie(lat.get_vertex_node_id(probe_trie->edge_vertices[0]))->data->d_data;
    edge_data_block = probe_trie->data;
  }
  else
  {
    d_edge_keys_ = utm.getTrie(lat.get_vertex_node_id(probe_trie->edge_vertices[1]))->data->d_data;
    edge_data_block = probe_trie->r_data;
  }
  d_edge_values_ = edge_data_block->d_data;
  // Determine join direction and new vertex

  vtype new_u = edge_forward ? probe_trie->edge_vertices[1] : probe_trie->edge_vertices[0];
  etype edge_idx = probe_trie->et._Find_first();
  directed_edge_idx = edge_idx * 2 + (edge_forward ? 0 : 1);
  utm.linkExpansion(target_trie->id, frag_trie->id, 0, new_u);

  // Find the root data trie in the reference chain
  root_trie = frag_trie;
  do
  {
    root_trie = utm.getTrie(root_trie->mask_parent_id);
  } while (root_trie->mask_parent_id != UINT32_MAX);

  num_frag_rows = root_trie->num_results;

  // Get selection mask from fragment's first layer (may be absent on device)
  select_bool_mask_ = frag_trie->data ? frag_trie->data->d_data : nullptr;
  if (select_bool_mask_ == nullptr && frag_trie->data && !mem_mgr.is_data_loaded(frag_trie->id))
  {
    (void)mem_mgr.load_data(frag_trie->id);
    select_bool_mask_ = frag_trie->data->d_data;
  }
  // Get edge trie information
  d_edge_offsets_ = ht.d_offsets_ + ht.edge_offset_starts_[directed_edge_idx];

  // Duplicate check setup
  num_dup_cols = 0;
  dup_check_type = 0;

  vltype new_u_label = hq->vLabels_[new_u];
  for (int i = 0; i < root_trie->num_cols; ++i)
    if (new_u_label == hq->vLabels_[root_trie->mapped_query_us_[i]])
      h_dup_cols[num_dup_cols++] = i;
  if (num_dup_cols == root_trie->num_cols)
    dup_check_type = 2;
  else if (num_dup_cols > 0)
    dup_check_type = 1;
  if (dup_check_type > 0)
    cuchk(cudaMemcpyAsync(d_dup_cols, h_dup_cols, sizeof(uint32_t) * num_dup_cols, cudaMemcpyHostToDevice, join_stream));
  // Count phase
  // uint32_t *d_num_res_total;
  // cuchk(cudaMalloc((void **)&d_num_res_total, sizeof(uint32_t)));
  // cuchk(cudaMemset(d_num_res_total, 0, sizeof(uint32_t)));

  pin_cols.clear();
  if (dup_check_type > 0)
    pin_cols.assign(h_dup_cols, h_dup_cols + num_dup_cols);
  bool join_col_pinned = false;
  for (auto c : pin_cols)
    if (c == col_key)
      join_col_pinned = true;
  if (!join_col_pinned)
    pin_cols.push_back(col_key);
  std::sort(pin_cols.begin(), pin_cols.end());

  up_to_col = UINT32_MAX;
  for (auto c : pin_cols)
    up_to_col = std::min(up_to_col, c);

  auto data_arrays = utm.get_data_arrays(root_trie->id, pin_cols);
  auto parent_arrays = utm.get_parent_arrays(root_trie->id, up_to_col);

  bool ready = (select_bool_mask_ != nullptr);
  if (ready)
  {
    for (auto col_id : pin_cols)
    {
      if (col_id >= data_arrays.size() || data_arrays[col_id] == nullptr)
      {
        ready = false;
        break;
      }
    }
  }

  if (ready)
  {
    for (uint32_t col_id = root_trie->num_cols - 1; col_id > up_to_col; --col_id)
    {
      if (col_id >= parent_arrays.size() || parent_arrays[col_id] == nullptr)
      {
        ready = false;
        break;
      }
    }
  }

  if (!ready)
  {
    mem_mgr.unpin_climb_data();
    mem_mgr.unpin_climb_parents();
    return false;
  }

  cuchk(cudaMemcpyAsync(d_data_arrays_, data_arrays.data(), sizeof(vtype *) * data_arrays.size(),
                        cudaMemcpyHostToDevice, join_stream));
  cuchk(cudaMemcpyAsync(d_parent_arrays_, parent_arrays.data(), sizeof(uint64_t *) * parent_arrays.size(),
                        cudaMemcpyHostToDevice, join_stream));

  if (!mem_mgr.allocate_temporal(d_num_res_for_each, sizeof(uint32_t) * (num_frag_rows + 1)))
  {
    mem_mgr.unpin_climb_data();
    mem_mgr.unpin_climb_parents();
    return false;
  }
  cuchk(cudaMemsetAsync(d_num_res_for_each, 0, sizeof(uint32_t) * (num_frag_rows + 1), join_stream));
  if (!mem_mgr.allocate_temporal(d_st_array, sizeof(uint32_t) * num_frag_rows))
  {
    mem_mgr.deallocate_temporal(d_num_res_for_each);
    mem_mgr.unpin_climb_data();
    mem_mgr.unpin_climb_parents();
    return false;
  }
  cuchk(cudaMemsetAsync(d_st_array, 0xFFFFFFFF, sizeof(uint32_t) * num_frag_rows, join_stream));

  return true;
}

void JoinContext::refer_join_without_duplication_context()
{
  if (num_frag_rows >= UINT32_MAX)
    task_per_warp = 32;
  else
  {
    uint32_t avg = num_frag_rows / (GRID_DIM * BLOCK_DIM / 32);
    task_per_warp = findClosestPowerOf2_Bitwise(avg);
    task_per_warp = std::min(task_per_warp, 32u);
  }

  assert(__builtin_popcount(task_per_warp) == 1);
}

void JoinContext::refer_join_with_duplication_context()
{
  auto &ht = HashTable::getInstance();
  auto &mem_mgr = MemoryManager::getInstance();

  // std::cout << "Join with duplication check." << std::endl;
  mask_length = ht.max_values_per_edge_[directed_edge_idx];
  // Convert to number of 32-bit words (round up) and align to 32 words (128 bytes) for coalescing
  mask_length = ((mask_length + 31) / 32 + 31) / 32 * 32;
  if (!mem_mgr.allocate_temporal(d_mask, sizeof(uint32_t) * num_frag_rows * mask_length))
  {
    d_mask = nullptr;
    return;
  }
  cuchk(cudaMemsetAsync(d_mask, 0, sizeof(uint32_t) * num_frag_rows * mask_length, join_stream));
}

void get_parent_arrays_pipeline(
    uint32_t trie_id,
    uint32_t up_to_col,
    std::vector<JoinInfo<uint64_t>> &parent_info_arrays)
{
  auto &utm = UnifiedTrieManager::getInstance();
  auto &mem_mgr = MemoryManager::getInstance();
  auto &jc = JoinContext::getInstance();

  UnifiedTrie *trie = utm.tries[trie_id];
  mem_mgr.climb_parents.clear();
  parent_info_arrays.clear();
  parent_info_arrays.reserve(MAX_VQ);
  jc.partial_parent_cols.clear();

  for (int i = 0; i < trie->column_ids.size(); ++i)
  {
    uint32_t col_trie_id = trie->column_ids[i];
    JoinInfo<uint64_t> info;
    if (i <= up_to_col)
    {
      info.array_ptr = nullptr; // offset = 0, is_partial = false by default
      parent_info_arrays.push_back(info);
      continue;
    }

    UnifiedTrie *col_trie = utm.tries[col_trie_id];
    assert(col_trie != nullptr);

    if (col_trie->data)
    {
      int block_idx = col_trie_id * 2 + 1;
      if (mem_mgr.memory_blocks[block_idx].onDevice())
      {
        info.array_ptr = col_trie->data->d_parents;
        parent_info_arrays.push_back(info);
      }
      else // on Host
      {
        bool load_success = mem_mgr.load_parent(col_trie_id);
        if (load_success)
        {
          info.array_ptr = col_trie->data->d_parents;
          parent_info_arrays.push_back(info);
        }
        else
        {
          info.array_ptr = nullptr;
          info.offset = 0;
          info.is_partial = 1;
          parent_info_arrays.push_back(info);
          jc.partial_parent_cols.push_back(i);
        }
      }
    }
    else if (col_trie->r_data)
    {
      info.array_ptr = col_trie->r_data->d_parents;
      parent_info_arrays.push_back(info);
    }
    else
    {
      info.array_ptr = nullptr;
      parent_info_arrays.push_back(info);
    }
    if (info.is_partial == false)
    {
      mem_mgr.pin_parent(col_trie_id);
      mem_mgr.climb_parents.push_back(col_trie_id);
    }
  }
}

void get_data_arrays_pipeline(
    uint32_t trie_id,
    std::vector<uint32_t> &pin_cols,
    std::vector<JoinInfo<uint32_t>> &data_info_arrays)
{
  auto &utm = UnifiedTrieManager::getInstance();
  auto &mem_mgr = MemoryManager::getInstance();
  auto &jc = JoinContext::getInstance();

  UnifiedTrie *trie = utm.tries[trie_id];
  mem_mgr.climb_data.clear();
  data_info_arrays.clear();
  data_info_arrays.reserve(MAX_VQ);
  jc.partial_data_cols.clear();
  JoinInfo<uint32_t> info;
  int pin_idx = 0;

  for (int i = 0; i < trie->num_cols; ++i)
  {
    auto col_trie_id = trie->column_ids[i];

    info.array_ptr = nullptr;
    info.is_partial = 0;
    info.offset = 0;

    UnifiedTrie *col_trie = utm.tries[col_trie_id];
    if (col_trie->data)
    {
      int block_idx = col_trie_id * 2;
      if (mem_mgr.memory_blocks[block_idx].onDevice())
      {
        info.array_ptr = col_trie->data->d_data;
        data_info_arrays.push_back(info);
      }
      else
      {
        bool should_pin = (pin_idx < static_cast<int>(pin_cols.size())) && (pin_cols[pin_idx] == i);
        if (should_pin)
        {
          bool load_success = mem_mgr.load_data(col_trie_id);
          if (load_success)
          {
            info.array_ptr = col_trie->data->d_data;
            data_info_arrays.push_back(info);
          }
          else
          {
            info.array_ptr = nullptr;
            info.offset = 0;
            info.is_partial = 1;
            data_info_arrays.push_back(info);
            jc.partial_data_cols.push_back(i);
          }
        }
        else
        {
          info.array_ptr = nullptr;
          data_info_arrays.push_back(info);
        }
      }
    }
    else if (col_trie->r_data)
    {
      info.array_ptr = col_trie->r_data->d_data;
      data_info_arrays.push_back(info);
    }
    else
    {
      info.array_ptr = nullptr;
      data_info_arrays.push_back(info);
    }
    if (pin_idx < static_cast<int>(pin_cols.size()) && pin_cols[pin_idx] == i)
    {
      pin_idx++;
      if (info.is_partial == false)
      {
        mem_mgr.pin_data(col_trie_id);
        mem_mgr.climb_data.push_back(col_trie_id);
      }
    }
  }
}
