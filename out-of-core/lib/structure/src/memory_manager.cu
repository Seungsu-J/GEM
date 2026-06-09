#include "memory_manager.cuh"
#include "unifiedTrie.cuh"
#include "lattice.h"
#include "policy.h"
#include "cuda_helpers.cuh"
#include "host_memory_manager.cuh"

#include <cuda_runtime.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <iostream>

// Sentinel value for "no victim available"
constexpr uint32_t NO_VICTIM = std::numeric_limits<uint32_t>::max();

// ============================================================================
// MemoryManager Implementation
// ============================================================================

static MemoryManager *g_instance = nullptr;

MemoryManager &MemoryManager::getInstance()
{
  if (!g_instance)
  {
    std::cerr << "ERROR: MemoryManager not initialized. Call initialize() first.\n";
    exit(1);
  }
  return *g_instance;
}

void MemoryManager::initialize(size_t gpu_memory_limit)
{
  if (g_instance)
  {
    std::cerr << "WARNING: MemoryManager already initialized.\n";
    return;
  }

  g_instance = new MemoryManager();

  g_instance->stats.gpu_memory_limit = gpu_memory_limit;
  g_instance->stats.used_gpu_memory = 0;
  g_instance->stats.available_gpu_memory = gpu_memory_limit;
  g_instance->stats.memory_threshold = static_cast<size_t>(gpu_memory_limit * 0.95); // 95% threshold

  g_instance->cur_stream = nullptr;
  g_instance->eviction_policy = nullptr;

  g_instance->backup_stream = new cudaStream_t;
  cuchk(cudaStreamCreateWithFlags(g_instance->backup_stream, cudaStreamNonBlocking));
}

void MemoryManager::shutdown()
{
  if (g_instance)
  {
    if (g_instance->eviction_policy)
    {
      delete g_instance->eviction_policy;
      g_instance->eviction_policy = nullptr;
    }
    delete g_instance;
    g_instance = nullptr;
  }
}

void MemoryManager::set_stream(cudaStream_t *stream)
{
  cur_stream = stream;
}

void MemoryManager::reset_stream()
{
  cur_stream = nullptr;
}

void MemoryManager::init_size(int max_tries)
{
  ptr_to_trie.reserve(max_tries * 2);
  ptr_to_size.reserve(max_tries * 2);
  trie_data_ptrs.resize(max_tries, nullptr);
  trie_parent_ptrs.resize(max_tries, nullptr);
  memory_blocks.resize(max_tries * 2, MemoryBlock());
  reverse_memory_blocks.resize(max_tries, MemoryBlock());

  block_pinned.resize(max_tries * 2, false);
  block_pinned_fix.resize(max_tries * 2, false);
  trie_data_backedup.resize(max_tries, false);
  trie_parent_backedup.resize(max_tries, false);
}

// ============================================================================
// Memory Operations
// ============================================================================

bool MemoryManager::allocate_temporal(void *&ptr, size_t size)
{
  if (!ensure_memory(size))
  {
    alert_insufficient_memory(size);
    return false;
  }

  cuchk(cudaMallocAsync(&ptr, size, *cur_stream));

  ptr_to_size[ptr] = size;
  // Update stats
  stats.updateAllocation(size);
  stats.temporal_memory += size;
  return true;
}

bool MemoryManager::allocate_temporal(uint64_t *&ptr, size_t size)
{
  if (!ensure_memory(size))
  {
    alert_insufficient_memory(size);
    return false;
  }

  cuchk(cudaMallocAsync(&ptr, size, *cur_stream));

  ptr_to_size[ptr] = size;
  // Update stats
  stats.updateAllocation(size);
  stats.temporal_memory += size;
  return true;
}

bool MemoryManager::allocate_temporal(uint32_t *&ptr, size_t size)
{
  if (!ensure_memory(size))
  {
    alert_insufficient_memory(size);
    return false;
  }

  cuchk(cudaMallocAsync(&ptr, size, *cur_stream));

  ptr_to_size[ptr] = size;
  // Update stats
  stats.updateAllocation(size);
  stats.temporal_memory += size;
  return true;
}

bool MemoryManager::allocate_permanent(void *&ptr, size_t size)
{
  if (!ensure_memory(size))
  {
    alert_insufficient_memory(size);
    return false;
  }

  cuchk(cudaMallocAsync(&ptr, size, *cur_stream));

  ptr_to_size[ptr] = size;
  // Update stats
  stats.updateAllocation(size);
  return true;
}

bool MemoryManager::allocate_permanent(uint64_t *&ptr, size_t size)
{
  if (!ensure_memory(size))
  {
    alert_insufficient_memory(size);
    return false;
  }

  cuchk(cudaMallocAsync(&ptr, size, *cur_stream));

  ptr_to_size[ptr] = size;
  stats.updateAllocation(size);
  return true;
}

bool MemoryManager::allocate_permanent(uint32_t *&ptr, size_t size)
{
  if (!ensure_memory(size))
  {
    alert_insufficient_memory(size);
    return false;
  }

  cuchk(cudaMallocAsync(&ptr, size, *cur_stream));

  ptr_to_size[ptr] = size;
  // Update stats
  stats.updateAllocation(size);
  return true;
}

bool MemoryManager::allocate_trie(int trie_id, bool forward_flag)
{
  auto &utm = UnifiedTrieManager::getInstance();
  auto trie = utm.getTrie(trie_id);
  auto data_block = forward_flag ? trie->data : trie->r_data;
  size_t sz = 0;

  int data_block_id = trie_id * 2;
  int parent_block_id = data_block_id + 1;
  auto &data_mem_block = memory_blocks[data_block_id];
  auto &parent_mem_block = memory_blocks[parent_block_id];

  if (data_block->num_compressed_blocks == 0) // expansion
  {
    sz += data_block->num_rows * sizeof(uint32_t);
    if (!ensure_memory(sz))
    {
      alert_insufficient_memory(sz);
      return false;
    }
    cuchk(cudaMallocAsync(&data_block->d_data, data_block->num_rows * sizeof(uint32_t), *cur_stream));
    ptr_to_size[data_block->d_data] = sz;
    data_mem_block.device_ptr = data_block->d_data;
    data_mem_block.size_in_bytes = sz;
    stats.updateAllocation(sz);
    eviction_policy->onAllocate(data_block_id, sz);
    if (trie->num_cols > 1) // one-vertex trie has no parents.
    {
      size_t this_size = data_block->num_rows * sizeof(uint64_t);
      if (!ensure_memory(this_size))
      {
        alert_insufficient_memory(this_size);
        return false;
      }
      cuchk(cudaMallocAsync(&data_block->d_parents, data_block->num_rows * sizeof(uint64_t), *cur_stream));
      // sz += this_size;
      ptr_to_size[data_block->d_parents] = this_size;
      parent_mem_block.device_ptr = data_block->d_parents;
      parent_mem_block.size_in_bytes = this_size;
      eviction_policy->onAllocate(parent_block_id, this_size);
      stats.updateAllocation(this_size);
    }
  }
  else // mask
  {
    sz = sizeof(uint32_t) * data_block->num_compressed_blocks;
    if (!ensure_memory(sz))
    {
      alert_insufficient_memory(sz);
      return false;
    }
    cuchk(cudaMallocAsync(&data_block->d_data, sizeof(uint32_t) * data_block->num_compressed_blocks, *cur_stream));
    ptr_to_size[data_block->d_data] = sz;
    data_mem_block.device_ptr = data_block->d_data;
    data_mem_block.size_in_bytes = sz;
    eviction_policy->onAllocate(data_block_id, sz);
    stats.updateAllocation(sz);
  }
  // Update stats

  return true;
}

bool MemoryManager::deallocate_temporal(void *&ptr)
{
  cuchk(cudaFreeAsync(ptr, *cur_stream));
  auto sz = ptr_to_size[ptr];

  ptr_to_size.erase(ptr);
  ptr = nullptr;
  stats.updateDeallocation(sz);
  stats.temporal_memory -= sz;
  return true;
}

bool MemoryManager::deallocate_temporal(uint64_t *&ptr)
{
  cuchk(cudaFreeAsync(ptr, *cur_stream));
  auto sz = ptr_to_size[ptr];

  ptr_to_size.erase(ptr);
  ptr = nullptr;
  stats.updateDeallocation(sz);
  stats.temporal_memory -= sz;
  return true;
}

bool MemoryManager::deallocate_temporal(uint32_t *&ptr)
{
  cuchk(cudaFreeAsync(ptr, *cur_stream));
  auto sz = ptr_to_size[ptr];

  ptr_to_size.erase(ptr);
  ptr = nullptr;
  stats.updateDeallocation(sz);
  stats.temporal_memory -= sz;
  return true;
}

bool MemoryManager::deallocate_permanent(void *&ptr)
{
  cuchk(cudaFreeAsync(ptr, *cur_stream));
  auto sz = ptr_to_size[ptr];

  ptr_to_size.erase(ptr);
  ptr = nullptr;
  stats.updateDeallocation(sz);
  return true;
}

bool MemoryManager::deallocate_permanent(uint64_t *&ptr)
{
  cuchk(cudaFreeAsync(ptr, *cur_stream));
  auto sz = ptr_to_size[ptr];

  ptr_to_size.erase(ptr);
  ptr = nullptr;
  stats.updateDeallocation(sz);
  return true;
}

bool MemoryManager::deallocate_permanent(uint32_t *&ptr)
{
  cuchk(cudaFreeAsync(ptr, *cur_stream));
  auto sz = ptr_to_size[ptr];

  ptr_to_size.erase(ptr);
  ptr = nullptr;
  stats.updateDeallocation(sz);
  return true;
}

bool MemoryManager::deallocate_trie(int trie_id)
{
  auto &utm = UnifiedTrieManager::getInstance();
  auto trie = utm.getTrie(trie_id);
  size_t sz = 0;

  int data_mem_block_id = trie_id * 2;
  int parent_mem_block_id = data_mem_block_id + 1;
  auto &data_mem_block = memory_blocks[data_mem_block_id];
  auto &parent_mem_block = memory_blocks[parent_mem_block_id];

  if (trie->data->d_data)
  {
    cuchk(cudaFreeAsync(trie->data->d_data, *cur_stream));
    sz += ptr_to_size[trie->data->d_data];
    ptr_to_size.erase(trie->data->d_data);
    trie->data->d_data = nullptr;

    data_mem_block.device_ptr = nullptr;
    data_mem_block.size_in_bytes = 0;
  }
  if (trie->data->d_parents)
  {
    cuchk(cudaFreeAsync(trie->data->d_parents, *cur_stream));
    sz += ptr_to_size[trie->data->d_parents];
    ptr_to_size.erase(trie->data->d_parents);
    trie->data->d_parents = nullptr;

    parent_mem_block.device_ptr = nullptr;
    parent_mem_block.size_in_bytes = 0;
  }
  stats.updateDeallocation(sz);
  return true;
}

// TODO: maybe failed for insufficient host memory. Need try-catch, correctly handling eviction failure.
bool MemoryManager::evict_data(int trie_id)
{
  std::cout << "Evicting data for trie ID " << trie_id << std::endl;
  int block_id = trie_id * 2;
  if (block_pinned[block_id])
    return false; // cannot evict pinned tries.
  auto &utm = UnifiedTrieManager::getInstance();
  auto trie = utm.getTrie(trie_id);

  if (trie_data_backedup[trie_id] == true)
  {
    // evicted before. No need to copy again.
    MemoryBlock &block = memory_blocks[block_id];
    size_t sz = block.size_in_bytes;
    cuchk(cudaFreeAsync(trie->data->d_data, *cur_stream));
    ptr_to_size.erase(trie->data->d_data);
    stats.updateDeallocation(sz);
    trie->data->d_data = nullptr;
    eviction_policy->onFree(block_id);
    return true;
  }

  // first eviction for this block, need to backup on the host.
  MemoryBlock &block = memory_blocks[block_id];
  trie_data_backedup[trie_id] = true;

  void *temp_device_ptr = trie->data->d_data;
  size_t sz = ptr_to_size[temp_device_ptr];
  // cuchk(cudaStreamSynchronize(*cur_stream));
  // get pinned memory from host memory pool managed by HostMemoryManager
  auto &host_mem_mgr = HostMemoryManager::getInstance();
  block.host_ptr = host_mem_mgr.get_next_ptr(sizeof(uint32_t));
  if (!host_mem_mgr.allocate(sz))
  {
    std::cerr << "ERROR: Host memory pool exhausted during eviction of trie data for trie ID " << trie_id << ".\n";
    return false;
  }
  // cuchk(cudaMallocHost(&block.host_ptr, sz)); // pinned host memory
  // block.host_ptr = new uint32_t[sz / sizeof(uint32_t)]; // TODO: get from host memory pool.
  cuchk(cudaMemcpyAsync(
      block.host_ptr, trie->data->d_data, sz,
      cudaMemcpyDeviceToHost, *backup_stream));
  cuchk(cudaFreeAsync(trie->data->d_data, *backup_stream));
  ptr_to_size.erase(temp_device_ptr);
  block.size_in_bytes = sz;
  trie->data->d_data = nullptr;
  block.device_ptr = nullptr;
  eviction_policy->onFree(block_id);
  stats.updateDeallocation(sz);
  return true;
}

// TODO: maybe failed for insufficient host memory. Need try-catch, correctly handling eviction failure.
bool MemoryManager::evict_parent(int trie_id)
{
  std::cout << "Evicting parents for trie ID " << trie_id << std::endl;
  int block_id = trie_id * 2 + 1;
  if (block_pinned[block_id])
    return false; // cannot evict pinned tries.
  auto &utm = UnifiedTrieManager::getInstance();
  auto trie = utm.getTrie(trie_id);

  if (!trie->data->d_parents)
    return true; // No parents to evict

  if (trie_parent_backedup[trie_id] == true)
  {
    // evicted before. No need to copy again.
    MemoryBlock &block = memory_blocks[block_id];
    size_t sz = block.size_in_bytes;
    cuchk(cudaFreeAsync(trie->data->d_parents, *cur_stream));
    ptr_to_size.erase(trie->data->d_parents);
    stats.updateDeallocation(sz);
    trie->data->d_parents = nullptr;
    eviction_policy->onFree(block_id);
    return true;
  }

  // first eviction for this block, need to backup on the host.

  MemoryBlock &block = memory_blocks[block_id];
  trie_parent_backedup[trie_id] = true;

  void *temp_device_ptr = trie->data->d_parents;
  size_t sz = ptr_to_size[temp_device_ptr];
  // cuchk(cudaStreamSynchronize(*cur_stream));
  // get pinned memory from host memory pool managed by HostMemoryManager
  auto &host_mem_mgr = HostMemoryManager::getInstance();
  block.host_ptr = host_mem_mgr.get_next_ptr(sizeof(uint64_t));
  if (!host_mem_mgr.allocate(sz))
  {
    std::cerr << "ERROR: Host memory pool exhausted during eviction of trie data for trie ID " << trie_id << ".\n";
    return false;
  }
  // cuchk(cudaMallocHost(&block.host_ptr, sz)); // pinned host memory
  // block.host_ptr = new uint64_t[trie->data->num_rows]; // TODO: get from host memory pool.
  cuchk(cudaMemcpyAsync(
      block.host_ptr, trie->data->d_parents, sz,
      cudaMemcpyDeviceToHost, *backup_stream));
  cuchk(cudaFreeAsync(trie->data->d_parents, *backup_stream));
  ptr_to_size.erase(temp_device_ptr);
  block.size_in_bytes = sz;
  trie->data->d_parents = nullptr;
  block.device_ptr = nullptr;
  stats.updateDeallocation(sz);
  eviction_policy->onFree(block_id);
  return true;
}

// TODO: maybe failed for insufficient host memory. Need try-catch, correctly handling eviction failure.
bool MemoryManager::evict_trie(int trie_id)
{
  int block_id_data = trie_id * 2;
  int block_id_parent = trie_id * 2 + 1;
  if (block_pinned[block_id_data] || block_pinned[block_id_parent])
    return false; // cannot evict pinned tries.

  bool success = evict_data(trie_id);
  success = success && evict_parent(trie_id);
  return success;
}

bool MemoryManager::load_data(int trie_id)
{
  int block_id = trie_id * 2;
  if (block_pinned_fix[block_id])
    return true; // already pinned, no need to load.

  auto &block = memory_blocks[block_id];
  if (!ensure_memory(block.size_in_bytes))
  {
    alert_insufficient_memory(block.size_in_bytes);
    std::cerr << "ERROR: Failed to evict enough memory for loading data of trie " << trie_id << ".\n";
    return false;
  }

  auto &utm = UnifiedTrieManager::getInstance();
  auto trie = utm.getTrie(trie_id);
  size_t sz = block.size_in_bytes;
  cuchk(cudaMallocAsync(&trie->data->d_data, sz, *cur_stream));
  cuchk(cudaMemcpyAsync(
      trie->data->d_data, block.host_ptr, sz,
      cudaMemcpyHostToDevice, *cur_stream));
  block.device_ptr = trie->data->d_data;
  ptr_to_size[trie->data->d_data] = sz;
  stats.updateAllocation(sz);

  return true;
}

bool MemoryManager::load_parent(int trie_id)
{
  int block_id = trie_id * 2 + 1;
  if (block_pinned_fix[block_id])
    return true; // already pinned, no need to load.
  auto &block = memory_blocks[block_id];
  if (!ensure_memory(block.size_in_bytes))
  {
    alert_insufficient_memory(block.size_in_bytes);
    std::cerr << "ERROR: Failed to evict enough memory for loading parent of trie " << trie_id << ".\n";
    return false;
  }

  auto &utm = UnifiedTrieManager::getInstance();
  auto trie = utm.getTrie(trie_id);
  size_t sz = block.size_in_bytes;
  cuchk(cudaMallocAsync(&trie->data->d_parents, sz, *cur_stream));
  cuchk(cudaMemcpyAsync(
      trie->data->d_parents, block.host_ptr, sz,
      cudaMemcpyHostToDevice, *cur_stream));
  block.device_ptr = trie->data->d_parents;
  ptr_to_size[trie->data->d_parents] = sz;
  stats.updateAllocation(sz);

  return true;
}

bool MemoryManager::load_trie(int trie_id)
{
  int block_id_data = trie_id * 2;
  int block_id_parent = trie_id * 2 + 1;
  if (block_pinned_fix[block_id_data] || block_pinned_fix[block_id_parent])
    return true; // already pinned, no need to load.
  bool success = load_data(trie_id);
  success = success && load_parent(trie_id);
  return success;
}

bool MemoryManager::is_trie_loaded(int trie_id, bool forward_flag)
{
  int block_id_data = trie_id * 2;
  int block_id_parent = trie_id * 2 + 1;
  if (block_pinned_fix[block_id_data] || block_pinned_fix[block_id_parent])
    return true; // pinned tries are considered loaded.

  auto &utm = UnifiedTrieManager::getInstance();
  auto trie = utm.getTrie(trie_id);
  auto data_block = forward_flag ? trie->data : trie->r_data;
  bool success = (data_block->d_data != nullptr);
  if (data_block->num_compressed_blocks == 0) // expansion
  {
    success = success && (data_block->d_parents != nullptr || trie->num_cols == 1);
  }
  return success;
}

bool MemoryManager::is_data_loaded(int trie_id, bool forward_flag)
{
  int block_id = trie_id * 2;
  if (block_pinned_fix[block_id])
    return true; // pinned tries are considered loaded.

  auto &utm = UnifiedTrieManager::getInstance();
  auto trie = utm.getTrie(trie_id);
  auto block = forward_flag ? trie->data : trie->r_data;
  return block->d_data != nullptr;
}

bool MemoryManager::is_parent_loaded(int trie_id, bool forward_flag)
{
  int block_id = trie_id * 2 + 1;
  if (block_pinned_fix[block_id])
    return true; // pinned tries are considered loaded.
  auto &utm = UnifiedTrieManager::getInstance();
  auto trie = utm.getTrie(trie_id);
  auto block = forward_flag ? trie->data : trie->r_data;
  if (block->num_compressed_blocks > 0) // mask, no parents
    return true;
  return block->d_parents != nullptr;
}

bool MemoryManager::ensure_trie(int trie_id, bool forward_flag)
{
  int block_id_data = trie_id * 2;
  int block_id_parent = trie_id * 2 + 1;
  if (block_pinned_fix[block_id_data] || block_pinned_fix[block_id_parent])
    return true; // fixed pinned, always in memory.

  auto &utm = UnifiedTrieManager::getInstance();
  auto trie = utm.getTrie(trie_id);
  if (trie->data == nullptr)
    forward_flag = false;

  bool ret;
  bool if_cache_hit = true;
  if (!is_data_loaded(trie_id, forward_flag))
  {
    ret = load_data(trie_id);
    if (!ret)
    {
      std::cerr << "ERROR: Failed to load data of trie " << trie_id << " to GPU memory.\n";
      return false;
    }
    if_cache_hit = false;
  }

  auto data_block = forward_flag ? trie->data : trie->r_data;
  if (trie->num_cols > 1 && data_block->num_compressed_blocks == 0) // has parents
  {
    if (!is_parent_loaded(trie_id, forward_flag))
    {
      ret = ret && load_parent(trie_id);
      if (!ret)
      {
        std::cerr << "ERROR: Failed to load parents of trie " << trie_id << " to GPU memory.\n";
        return false;
      }
      if_cache_hit = false;
    }
  }
  if (if_cache_hit)
    ++stats.cache_hits;
  else
    ++stats.cache_misses;
  return ret;
}

bool MemoryManager::pin_trie(int trie_id, bool update_flag, bool fixed_flag)
{
  bool success = pin_data(trie_id, update_flag, fixed_flag);
  success = success && pin_parent(trie_id, update_flag, fixed_flag);
  return success;
}

bool MemoryManager::pin_data(int trie_id, bool update_flag, bool fixed_flag)
{
  int block_id = trie_id * 2;
  if (fixed_flag)
    block_pinned_fix[block_id] = true;
  if (!block_pinned[block_id])
  {
    block_pinned[block_id] = true;
    if (update_flag)
      stats.pinned_memory += memory_blocks[block_id].size_in_bytes;
  }
  return true;
}

bool MemoryManager::pin_parent(int trie_id, bool update_flag, bool fixed_flag)
{
  int block_id = trie_id * 2 + 1;
  if (fixed_flag)
    block_pinned_fix[block_id] = true;
  if (!block_pinned[block_id])
  {
    block_pinned[block_id] = true;
    if (update_flag)
      stats.pinned_memory += memory_blocks[block_id].size_in_bytes;
  }
  return true;
}

// TODO: may fail. If failed, no partial unpin.
bool MemoryManager::unpin_trie(int trie_id, bool update_flag)
{
  bool success = unpin_data(trie_id, update_flag);
  success = success && unpin_parent(trie_id, update_flag);
  return success;
}

bool MemoryManager::unpin_data(int trie_id, bool update_flag)
{
  int block_id = trie_id * 2;
  if (block_pinned_fix[block_id])
    return false; // cannot unpin fixed pinned blocks.
  if (block_pinned[block_id])
  {
    block_pinned[block_id] = false;
    if (update_flag)
      stats.pinned_memory -= memory_blocks[block_id].size_in_bytes;
  }
  return true;
}

bool MemoryManager::unpin_parent(int trie_id, bool update_flag)
{
  int block_id = trie_id * 2 + 1;
  if (block_pinned_fix[block_id])
    return false; // cannot unpin fixed pinned blocks.
  if (block_pinned[block_id])
  {
    block_pinned[block_id] = false;
    if (update_flag)
      stats.pinned_memory -= memory_blocks[block_id].size_in_bytes;
  }
  return true;
}

bool MemoryManager::unpin_climb_data()
{
  for (int trie_id : climb_data)
  {
    unpin_data(trie_id);
  }
  climb_data.clear();
  return true;
}

bool MemoryManager::unpin_climb_parents()
{
  for (int trie_id : climb_parents)
  {
    unpin_parent(trie_id);
  }
  climb_parents.clear();
  return true;
}

void MemoryManager::record_trie_access(int trie_id)
{
  eviction_policy->onAccess(trie_id * 2);
  eviction_policy->onAccess(trie_id * 2 + 1);
}

void MemoryManager::record_data_access(int trie_id)
{
  eviction_policy->onAccess(trie_id * 2);
}

void MemoryManager::record_parent_access(int trie_id)
{
  eviction_policy->onAccess(trie_id * 2 + 1);
}

bool MemoryManager::evict(size_t needBytes)
{
  if (!eviction_policy)
  {
    std::cerr << "ERROR: No eviction policy set.\n";
    return false;
  }

  size_t freed_memory = 0;
  int max_iterations = 1000; // Prevent infinite loop
  int iteration = 0;

  while (freed_memory < needBytes && iteration < max_iterations)
  {
    // Pass block_pinned directly to the eviction policy
    uint32_t victim_block_id = eviction_policy->pickVictim(needBytes - freed_memory, block_pinned);

    if (victim_block_id == NO_VICTIM)
    {
      // std::cerr << "WARNING: No victim available for eviction. Freed " << freed_memory << " bytes out of " << needBytes << " needed.\n";
      return freed_memory >= needBytes;
    }

    // Find trie_id corresponding to this block_id
    int trie_id = victim_block_id / 2;
    bool is_parent_block = (victim_block_id & 1);

    // Check if pinned (double-check, though policy should have filtered this)
    if (block_pinned[victim_block_id])
    {
      // std::cerr << "WARNING: Attempting to evict pinned block " << victim_block_id << ". Skipping.\n";
      iteration++;
      continue;
    }

    bool success;
    if (is_parent_block)
      success = evict_parent(trie_id);
    else
      success = evict_data(trie_id);
    // Evict the trie
    if (!success)
    {
      std::cerr << "WARNING: Failed to evict trie " << trie_id << ".\n";
      iteration++;
      continue;
    }

    // Track freed memory (need to check both data and parent)
    // This is approximate; we already updated used_gpu_memory in deallocate
    // For accurate tracking, we should query the size before eviction
    // For now, assume eviction succeeded and freed some memory

    stats.total_evictions++;
    iteration++;

    // Check if we've freed enough
    if (stats.available_gpu_memory >= needBytes)
    {
      freed_memory = needBytes;
    }
  }

  return freed_memory >= needBytes;
}

bool MemoryManager::ensure_memory(size_t size)
{
  if (stats.available_gpu_memory >= size)
  {
    return true;
  }

  return evict(size);
}

void MemoryManager::alert_insufficient_memory(size_t needBytes)
{
  double bytes_in_GB = 1024 * 1024 * 1024;
  std::cout << "ERROR: Insufficient GPU memory. Need additional " << needBytes / bytes_in_GB << " GB.\n";
  std::cout << "Current used memory: " << stats.used_gpu_memory / bytes_in_GB << " GB, available memory: " << stats.available_gpu_memory / bytes_in_GB << " GB.\n";
  std::cout << "Pinned Memory: " << stats.pinned_memory / bytes_in_GB << " GB.\n";
  std::cout << "Temporal Memory: " << stats.temporal_memory / bytes_in_GB << " GB.\n";
}

void MemoryManager::set_eviction_policy(VirtualEvictPolicy *policy)
{
  if (eviction_policy)
  {
    delete eviction_policy;
  }
  eviction_policy = policy;
}

void MemoryManager::reclaim_all_except(void *keep_ptr)
{
  // Save keep_ptr size before iterating
  size_t keep_size = 0;
  if (keep_ptr != nullptr)
  {
    auto it = ptr_to_size.find(keep_ptr);
    if (it != ptr_to_size.end())
      keep_size = it->second;
  }

  // Free all tracked allocations except keep_ptr
  for (auto &kv : ptr_to_size)
  {
    void *ptr = kv.first;
    if (ptr == keep_ptr)
      continue;
    cudaFreeAsync(ptr, *cur_stream);
  }
  ptr_to_size.clear();

  // Re-track keep_ptr
  if (keep_ptr != nullptr && keep_size > 0)
    ptr_to_size[keep_ptr] = keep_size;

  // Clear per-query tracking state
  ptr_to_trie.clear();
  trie_data_ptrs.clear();
  trie_parent_ptrs.clear();
  memory_blocks.clear();
  reverse_memory_blocks.clear();
  block_pinned.clear();
  block_pinned_fix.clear();
  trie_data_backedup.clear();
  trie_parent_backedup.clear();
  climb_parents.clear();
  climb_data.clear();

  // Reset memory accounting: only keep_size remains allocated
  stats.used_gpu_memory = keep_size;
  stats.available_gpu_memory = stats.gpu_memory_limit - keep_size;
  stats.pinned_memory = 0;
  stats.temporal_memory = 0;
}
