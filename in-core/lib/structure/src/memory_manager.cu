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
#include <stdexcept>

// Sentinel value for "no victim available"
constexpr uint32_t NO_VICTIM = std::numeric_limits<uint32_t>::max();

inline size_t real_size(size_t sz) { return sz * 1.1 + (256 << 20); } // 10% overhead + 256MB buffer
double bytes_in_GB = 1024 * 1024 * 1024;

// ============================================================================
// MemoryManager Implementation
// ============================================================================

static MemoryManager *g_instance = nullptr;

MemoryManager &MemoryManager::getInstance()
{
  if (!g_instance)
  {
    throw std::runtime_error("MemoryManager not initialized. Call initialize() first.");
  }
  return *g_instance;
}

bool MemoryManager::isInitialized()
{
  return g_instance != nullptr;
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

  g_instance->p_type = ProcessType::DIRECT;
}

void MemoryManager::shutdown()
{
  if (g_instance)
  {
    // Best-effort: release any tracked device allocations before tearing down.
    try
    {
      cudaDeviceSynchronize();
      cudaStream_t cleanup_stream = nullptr;
      if (cudaStreamCreateWithFlags(&cleanup_stream, cudaStreamNonBlocking) == cudaSuccess)
      {
        for (const auto &kv : g_instance->ptr_to_size)
        {
          void *ptr = kv.first;
          if (ptr == nullptr)
            continue;
          cudaError_t free_err = cudaFreeAsync(ptr, cleanup_stream);
          if (free_err == cudaErrorInvalidDevicePointer)
            cudaGetLastError();
        }
        cudaStreamSynchronize(cleanup_stream);
        cudaStreamDestroy(cleanup_stream);
      }
      g_instance->ptr_to_size.clear();
      g_instance->ptr_to_trie.clear();
      g_instance->stats.used_gpu_memory = 0;
      g_instance->stats.available_gpu_memory = g_instance->stats.gpu_memory_limit;
      g_instance->stats.temporal_memory = 0;
      g_instance->stats.pinned_memory = 0;
    }
    catch (...)
    {
    }

    if (g_instance->backup_stream != nullptr)
    {
      cudaStreamSynchronize(*g_instance->backup_stream);
      cudaStreamDestroy(*g_instance->backup_stream);
      delete g_instance->backup_stream;
      g_instance->backup_stream = nullptr;
    }
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
  ptr_to_trie.clear();
  ptr_to_trie.reserve(max_tries * 2);
  ptr_to_size.reserve(max_tries * 2);
  trie_data_ptrs.assign(max_tries, nullptr);
  trie_parent_ptrs.assign(max_tries, nullptr);
  memory_blocks.assign(max_tries * 2, MemoryBlock());
  reverse_memory_blocks.assign(max_tries, MemoryBlock());

  block_pinned.assign(max_tries * 2, false);
  block_pinned_fix.assign(max_tries * 2, false);
  trie_data_backedup.assign(max_tries, false);
  trie_parent_backedup.assign(max_tries, false);
  trie_trashed.assign(max_tries, false);
  trash_queue.clear();
  climb_parents.clear();
  climb_data.clear();
}

void MemoryManager::mark_trie_trash(int trie_id)
{
  if (trie_id < 0)
    return;
  if (static_cast<size_t>(trie_id) >= trie_trashed.size())
    return;
  if (trie_trashed[trie_id])
    return;

  trie_trashed[trie_id] = true;
  trash_queue.push_back(static_cast<uint32_t>(trie_id));

  // Best-effort: make it evictable if it's only dynamically pinned.
  unpin_trie(trie_id, false);
}

// ============================================================================
// Memory Operations
// ============================================================================

bool MemoryManager::safe_allocate(void *&ptr, size_t size)
{
  ptr = nullptr;
  cudaError_t err = cudaMallocAsync(&ptr, size, *cur_stream);
  if (err == cudaSuccess)
    return true;
  else
  {
    // clear error
    cuchk(cudaStreamSynchronize(*cur_stream));
    cuchk(cudaStreamSynchronize(*backup_stream));
    cudaGetLastError();

    // send memory
    cudaMemPool_t dafault_pool;
    cudaDeviceGetDefaultMemPool(&dafault_pool, GPU_NUM);
    cudaMemPoolTrimTo(dafault_pool, 0);
    std::cout << "call trim" << std::endl;

    ptr = nullptr;
    cudaError_t err_retry = cudaMallocAsync(&ptr, size, *cur_stream);
    if (err_retry != cudaSuccess)
    {
      alert_insufficient_memory(size);
      ptr = nullptr;
      cudaGetLastError();
      return false;
    }
    return true;
  }
}

bool MemoryManager::safe_allocate(uint32_t *&ptr, size_t size)
{
  ptr = nullptr;
  cudaError_t err = cudaMallocAsync(&ptr, size, *cur_stream);
  if (err == cudaSuccess)
    return true;
  else
  {
    // clear error
    cuchk(cudaStreamSynchronize(*cur_stream));
    cuchk(cudaStreamSynchronize(*backup_stream));
    cudaGetLastError();

    // send memory
    cudaMemPool_t dafault_pool;
    cudaDeviceGetDefaultMemPool(&dafault_pool, GPU_NUM);
    cudaMemPoolTrimTo(dafault_pool, 0);
    std::cout << "call trim" << std::endl;

    ptr = nullptr;
    cudaError_t err_retry = cudaMallocAsync(&ptr, size, *cur_stream);
    if (err_retry != cudaSuccess)
    {
      alert_insufficient_memory(size);
      ptr = nullptr;
      cudaGetLastError();
      return false;
    }
    return true;
  }
}

bool MemoryManager::safe_allocate(uint64_t *&ptr, size_t size)
{
  ptr = nullptr;
  cudaError_t err = cudaMallocAsync(&ptr, size, *cur_stream);
  if (err == cudaSuccess)
    return true;
  else
  {
    // clear error
    cuchk(cudaStreamSynchronize(*cur_stream));
    cuchk(cudaStreamSynchronize(*backup_stream));
    cudaGetLastError();

    // send memory
    cudaMemPool_t dafault_pool;
    cudaDeviceGetDefaultMemPool(&dafault_pool, GPU_NUM);
    cudaMemPoolTrimTo(dafault_pool, 0);
    std::cout << "call trim" << std::endl;

    ptr = nullptr;
    cudaError_t err_retry = cudaMallocAsync(&ptr, size, *cur_stream);
    if (err_retry != cudaSuccess)
    {
      alert_insufficient_memory(size);
      ptr = nullptr;
      cudaGetLastError();
      return false;
    }
    return true;
  }
}

bool MemoryManager::allocate_temporal(void *&ptr, size_t size)
{
  ptr = nullptr;
  if (!ensure_memory(size))
  {
    alert_insufficient_memory(size);
    return false;
  }

  if (safe_allocate(ptr, size))
  {
    ptr_to_size[ptr] = size;
    // Update stats
    stats.updateAllocation(size);
    stats.temporal_memory += size;
    return true;
  }
  return false;
}

bool MemoryManager::allocate_temporal(uint64_t *&ptr, size_t size)
{
  ptr = nullptr;
  if (!ensure_memory(size))
  {
    alert_insufficient_memory(size);
    return false;
  }

  if (safe_allocate(ptr, size))
  {
    ptr_to_size[ptr] = size;
    // Update stats
    stats.updateAllocation(size);
    stats.temporal_memory += size;
    return true;
  }
  return false;
}

bool MemoryManager::allocate_temporal(uint32_t *&ptr, size_t size)
{
  ptr = nullptr;
  if (!ensure_memory(size))
  {
    alert_insufficient_memory(size);
    return false;
  }

  if (safe_allocate(ptr, size))
  {
    ptr_to_size[ptr] = size;
    // Update stats
    stats.updateAllocation(size);
    stats.temporal_memory += size;
    return true;
  }
  return false;
}

bool MemoryManager::allocate_permanent(void *&ptr, size_t size)
{
  ptr = nullptr;
  if (!ensure_memory(size))
  {
    alert_insufficient_memory(size);
    return false;
  }

  if (safe_allocate(ptr, size))
  {
    ptr_to_size[ptr] = size;
    // Update stats
    stats.updateAllocation(size);
    return true;
  }
  return false;
}

bool MemoryManager::allocate_permanent(uint32_t *&ptr, size_t size)
{
  ptr = nullptr;
  if (!ensure_memory(size))
  {
    alert_insufficient_memory(size);
    return false;
  }
  if (safe_allocate(ptr, size))
  {
    ptr_to_size[ptr] = size;
    // Update stats
    stats.updateAllocation(size);
    return true;
  }
  return false;
}

bool MemoryManager::allocate_trie(int trie_id, bool forward_flag)
{
  auto &utm = UnifiedTrieManager::getInstance();
  auto trie = utm.getTrie(trie_id);
  auto data_block = forward_flag ? trie->data : trie->r_data;
  size_t cur_sz = 0, tot_size = 0;

  int data_block_id = trie_id * 2;
  int parent_block_id = data_block_id + 1;
  auto &data_mem_block = memory_blocks[data_block_id];
  auto &parent_mem_block = memory_blocks[parent_block_id];

  if (data_block->num_compressed_blocks == 0) // expansion
  {
    cur_sz = data_block->num_rows * sizeof(uint32_t);
    tot_size += cur_sz;
    tot_size += (trie->num_cols > 1) ? data_block->num_rows * sizeof(uint64_t) : 0;

    if (!ensure_memory(tot_size))
    {
      alert_insufficient_memory(tot_size);
      return false;
    }

    if (safe_allocate(data_block->d_data, cur_sz))
    {
      ptr_to_size[data_block->d_data] = cur_sz;
      data_mem_block.device_ptr = data_block->d_data;
      data_mem_block.size_in_bytes = cur_sz;
      stats.updateAllocation(cur_sz);
      eviction_policy->onAllocate(data_block_id, cur_sz);
    }
    else
      return false;
    if (trie->num_cols > 1) // one-vertex trie has no parents.
    {
      cur_sz = data_block->num_rows * sizeof(uint64_t);
      if (safe_allocate(data_block->d_parents, cur_sz))
      {
        // sz += this_size;
        ptr_to_size[data_block->d_parents] = cur_sz;
        parent_mem_block.device_ptr = data_block->d_parents;
        parent_mem_block.size_in_bytes = cur_sz;
        eviction_policy->onAllocate(parent_block_id, cur_sz);
        stats.updateAllocation(cur_sz);
      }
      else
      {
        // nested rollback
        cuchk(cudaFreeAsync(data_block->d_data, *cur_stream));
        ptr_to_size.erase(data_block->d_data);
        data_block->d_data = nullptr;
        data_mem_block.device_ptr = nullptr;
        data_mem_block.size_in_bytes = 0;
        stats.updateDeallocation(cur_sz >> 1);
        eviction_policy->onFree(data_block_id);

        return false;
      }
    }
  }
  else // mask
  {
    cur_sz = sizeof(uint32_t) * data_block->num_compressed_blocks;
    if (!ensure_memory(cur_sz))
    {
      alert_insufficient_memory(cur_sz);
      return false;
    }
    if (safe_allocate(data_block->d_data, sizeof(uint32_t) * data_block->num_compressed_blocks))
    {
      ptr_to_size[data_block->d_data] = cur_sz;
      data_mem_block.device_ptr = data_block->d_data;
      data_mem_block.size_in_bytes = cur_sz;
      eviction_policy->onAllocate(data_block_id, cur_sz);
      stats.updateAllocation(cur_sz);
    }
    else
      return false;
  }
  // Update stats

  return true;
}

bool MemoryManager::deallocate_temporal(void *&ptr)
{
  if (ptr == nullptr)
    return true;
  if (cur_stream == nullptr)
    throw std::runtime_error("MemoryManager: deallocate_temporal called without an active stream.");

  cuchk(cudaFreeAsync(ptr, *cur_stream));

  size_t sz = 0;
  auto it = ptr_to_size.find(ptr);
  if (it != ptr_to_size.end())
  {
    sz = it->second;
    ptr_to_size.erase(it);
  }
  ptr = nullptr;

  if (sz != 0)
  {
    stats.updateDeallocation(sz);
    if (stats.temporal_memory >= sz)
      stats.temporal_memory -= sz;
    else
      stats.temporal_memory = 0;
  }

  return true;
}

bool MemoryManager::deallocate_temporal(uint64_t *&ptr)
{
  if (ptr == nullptr)
    return true;
  if (cur_stream == nullptr)
    throw std::runtime_error("MemoryManager: deallocate_temporal called without an active stream.");

  cuchk(cudaFreeAsync(ptr, *cur_stream));

  size_t sz = 0;
  auto it = ptr_to_size.find(ptr);
  if (it != ptr_to_size.end())
  {
    sz = it->second;
    ptr_to_size.erase(it);
  }
  ptr = nullptr;

  if (sz != 0)
  {
    stats.updateDeallocation(sz);
    if (stats.temporal_memory >= sz)
      stats.temporal_memory -= sz;
    else
      stats.temporal_memory = 0;
  }

  return true;
}

bool MemoryManager::deallocate_temporal(uint32_t *&ptr)
{
  if (ptr == nullptr)
    return true;
  if (cur_stream == nullptr)
    throw std::runtime_error("MemoryManager: deallocate_temporal called without an active stream.");

  cuchk(cudaFreeAsync(ptr, *cur_stream));

  size_t sz = 0;
  auto it = ptr_to_size.find(ptr);
  if (it != ptr_to_size.end())
  {
    sz = it->second;
    ptr_to_size.erase(it);
  }
  ptr = nullptr;

  if (sz != 0)
  {
    stats.updateDeallocation(sz);
    if (stats.temporal_memory >= sz)
      stats.temporal_memory -= sz;
    else
      stats.temporal_memory = 0;
  }

  return true;
}

bool MemoryManager::deallocate_permanent(void *&ptr)
{
  if (ptr == nullptr)
    return true;
  if (cur_stream == nullptr)
    throw std::runtime_error("MemoryManager: deallocate_permanent called without an active stream.");

  cuchk(cudaFreeAsync(ptr, *cur_stream));

  size_t sz = 0;
  auto it = ptr_to_size.find(ptr);
  if (it != ptr_to_size.end())
  {
    sz = it->second;
    ptr_to_size.erase(it);
  }
  ptr = nullptr;
  if (sz != 0)
    stats.updateDeallocation(sz);
  return true;
}

bool MemoryManager::deallocate_permanent(uint32_t *&ptr)
{
  if (ptr == nullptr)
    return true;
  if (cur_stream == nullptr)
    throw std::runtime_error("MemoryManager: deallocate_permanent called without an active stream.");

  cuchk(cudaFreeAsync(ptr, *cur_stream));

  size_t sz = 0;
  auto it = ptr_to_size.find(ptr);
  if (it != ptr_to_size.end())
  {
    sz = it->second;
    ptr_to_size.erase(it);
  }
  ptr = nullptr;
  if (sz != 0)
    stats.updateDeallocation(sz);
  return true;
}

bool MemoryManager::deallocate_permanent(uint64_t *&ptr)
{
  if (ptr == nullptr)
    return true;
  if (cur_stream == nullptr)
    throw std::runtime_error("MemoryManager: deallocate_permanent called without an active stream.");

  cuchk(cudaFreeAsync(ptr, *cur_stream));

  size_t sz = 0;
  auto it = ptr_to_size.find(ptr);
  if (it != ptr_to_size.end())
  {
    sz = it->second;
    ptr_to_size.erase(it);
  }
  ptr = nullptr;
  if (sz != 0)
    stats.updateDeallocation(sz);
  return true;
}

bool MemoryManager::deallocate_trie(int trie_id, UnifiedTrie *trie)
{
  if (trie_id < 0)
    return true;
  if (static_cast<size_t>(trie_id) >= memory_blocks.size() / 2)
    return true;

  if (trie == nullptr)
  {
    auto &utm = UnifiedTrieManager::getInstance();
    trie = utm.getTrie(trie_id);
  }

  const int data_block_id = trie_id * 2;
  const int parent_block_id = data_block_id + 1;
  auto &data_block = memory_blocks[data_block_id];
  auto &parent_block = memory_blocks[parent_block_id];

  void *data_device_ptr = data_block.device_ptr;
  void *parent_device_ptr = parent_block.device_ptr;
  const size_t data_sz = data_block.size_in_bytes;
  const size_t parent_sz = parent_block.size_in_bytes;

  size_t freed_bytes = 0;

  if (data_device_ptr != nullptr)
  {
    cuchk(cudaFreeAsync(data_device_ptr, *cur_stream));
    auto it = ptr_to_size.find(data_device_ptr);
    if (it != ptr_to_size.end())
    {
      freed_bytes += it->second;
      ptr_to_size.erase(it);
    }
    data_block.device_ptr = nullptr;
  }

  if (parent_device_ptr != nullptr)
  {
    cuchk(cudaFreeAsync(parent_device_ptr, *cur_stream));
    auto it = ptr_to_size.find(parent_device_ptr);
    if (it != ptr_to_size.end())
    {
      freed_bytes += it->second;
      ptr_to_size.erase(it);
    }
    parent_block.device_ptr = nullptr;
  }

  if (eviction_policy != nullptr)
  {
    eviction_policy->onFree(data_block_id);
    eviction_policy->onFree(parent_block_id);
  }

  if (trie != nullptr)
  {
    if (trie->data != nullptr)
    {
      if (reinterpret_cast<void *>(trie->data->d_data) == data_device_ptr)
        trie->data->d_data = nullptr;
      if (reinterpret_cast<void *>(trie->data->d_parents) == parent_device_ptr)
        trie->data->d_parents = nullptr;
    }
    if (trie->r_data != nullptr)
    {
      if (reinterpret_cast<void *>(trie->r_data->d_data) == data_device_ptr)
        trie->r_data->d_data = nullptr;
      if (reinterpret_cast<void *>(trie->r_data->d_parents) == parent_device_ptr)
        trie->r_data->d_parents = nullptr;
    }
  }

  auto &host_mgr = HostMemoryManager::getInstance();
  if (data_block.host_ptr != nullptr)
  {
    host_mgr.deallocateManaged(data_block.host_ptr);
    data_block.host_ptr = nullptr;
  }
  if (parent_block.host_ptr != nullptr)
  {
    host_mgr.deallocateManaged(parent_block.host_ptr);
    parent_block.host_ptr = nullptr;
  }

  data_block.size_in_bytes = 0;
  parent_block.size_in_bytes = 0;

  if (static_cast<size_t>(trie_id) < trie_data_backedup.size())
    trie_data_backedup[trie_id] = false;
  if (static_cast<size_t>(trie_id) < trie_parent_backedup.size())
    trie_parent_backedup[trie_id] = false;

  if (block_pinned[data_block_id])
  {
    if (stats.pinned_memory >= data_sz)
      stats.pinned_memory -= data_sz;
    else
      stats.pinned_memory = 0;
    block_pinned[data_block_id] = false;
  }
  if (block_pinned[parent_block_id])
  {
    if (stats.pinned_memory >= parent_sz)
      stats.pinned_memory -= parent_sz;
    else
      stats.pinned_memory = 0;
    block_pinned[parent_block_id] = false;
  }
  block_pinned_fix[data_block_id] = false;
  block_pinned_fix[parent_block_id] = false;

  if (freed_bytes > 0)
    stats.updateDeallocation(freed_bytes);

  return true;
}

// TODO: maybe failed for insufficient host memory. Need try-catch, correctly handling eviction failure.
bool MemoryManager::evict_data(int trie_id)
{
  std::cout << "Evicting data for trie ID " << trie_id << std::endl;
  int block_id = trie_id * 2;
  if (block_pinned[block_id])
    return false; // cannot evict pinned tries.
  if (trie_id < 0 || static_cast<size_t>(trie_id) >= trie_trashed.size())
  {
    eviction_policy->onFree(block_id);
    return true;
  }
  auto &utm = UnifiedTrieManager::getInstance();
  auto trie = utm.getTrie(trie_id);
  if (trie == nullptr || trie->data == nullptr || trie->data->d_data == nullptr || memory_blocks[block_id].device_ptr == nullptr)
  {
    eviction_policy->onFree(block_id);
    memory_blocks[block_id].device_ptr = nullptr;
    return true;
  }

  if (trie_data_backedup[trie_id] == true)
  {
    // evicted before. No need to copy again.
    MemoryBlock &block = memory_blocks[block_id];
    size_t sz = block.size_in_bytes;
    cuchk(cudaFreeAsync(trie->data->d_data, *cur_stream));
    ptr_to_size.erase(trie->data->d_data);
    stats.updateDeallocation(sz);
    trie->data->d_data = nullptr;
    block.device_ptr = nullptr;
    eviction_policy->onFree(block_id);
    return true;
  }

  // first eviction for this block, need to backup on the host.
  MemoryBlock &block = memory_blocks[block_id];
  trie_data_backedup[trie_id] = true;

  void *temp_device_ptr = trie->data->d_data;
  size_t sz = block.size_in_bytes;
  if (sz == 0)
  {
    auto it_sz = ptr_to_size.find(temp_device_ptr);
    if (it_sz != ptr_to_size.end())
      sz = it_sz->second;
  }
  if (sz == 0)
  {
    std::cerr << "ERROR: Missing size metadata for trie data eviction (trie ID " << trie_id << ").\n";
    return false;
  }
  // cuchk(cudaStreamSynchronize(*cur_stream));
  // get pinned memory from HostMemoryManager (managed allocations can be freed later)
  auto &host_mem_mgr = HostMemoryManager::getInstance();
  block.host_ptr = host_mem_mgr.allocateManagedBytes(sz, sizeof(uint32_t));
  if (block.host_ptr == nullptr)
  {
    std::cerr << "ERROR: Host memory allocation failed during eviction of trie data for trie ID " << trie_id << ".\n";
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
  if (trie_id < 0 || static_cast<size_t>(trie_id) >= trie_trashed.size())
  {
    eviction_policy->onFree(block_id);
    return true;
  }
  auto &utm = UnifiedTrieManager::getInstance();
  auto trie = utm.getTrie(trie_id);
  if (trie == nullptr || trie->data == nullptr || memory_blocks[block_id].device_ptr == nullptr)
  {
    eviction_policy->onFree(block_id);
    memory_blocks[block_id].device_ptr = nullptr;
    return true;
  }

  if (!trie->data->d_parents)
  {
    eviction_policy->onFree(block_id);
    memory_blocks[block_id].device_ptr = nullptr;
    return true; // No parents to evict
  }

  if (trie_parent_backedup[trie_id] == true)
  {
    // evicted before. No need to copy again.
    MemoryBlock &block = memory_blocks[block_id];
    size_t sz = block.size_in_bytes;
    cuchk(cudaFreeAsync(trie->data->d_parents, *cur_stream));
    ptr_to_size.erase(trie->data->d_parents);
    stats.updateDeallocation(sz);
    trie->data->d_parents = nullptr;
    block.device_ptr = nullptr;
    eviction_policy->onFree(block_id);
    return true;
  }

  // first eviction for this block, need to backup on the host.

  MemoryBlock &block = memory_blocks[block_id];
  trie_parent_backedup[trie_id] = true;

  void *temp_device_ptr = trie->data->d_parents;
  size_t sz = block.size_in_bytes;
  if (sz == 0)
  {
    auto it_sz = ptr_to_size.find(temp_device_ptr);
    if (it_sz != ptr_to_size.end())
      sz = it_sz->second;
  }
  if (sz == 0)
  {
    std::cerr << "ERROR: Missing size metadata for trie parent eviction (trie ID " << trie_id << ").\n";
    return false;
  }
  // cuchk(cudaStreamSynchronize(*cur_stream));
  // get pinned memory from HostMemoryManager (managed allocations can be freed later)
  auto &host_mem_mgr = HostMemoryManager::getInstance();
  block.host_ptr = host_mem_mgr.allocateManagedBytes(sz, sizeof(uint64_t));
  if (block.host_ptr == nullptr)
  {
    std::cerr << "ERROR: Host memory allocation failed during eviction of trie data for trie ID " << trie_id << ".\n";
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
    {
      const size_t sz = memory_blocks[block_id].size_in_bytes;
      if (stats.pinned_memory >= sz)
        stats.pinned_memory -= sz;
      else
        stats.pinned_memory = 0;
    }
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
    {
      const size_t sz = memory_blocks[block_id].size_in_bytes;
      if (stats.pinned_memory >= sz)
        stats.pinned_memory -= sz;
      else
        stats.pinned_memory = 0;
    }
  }
  return true;
}

void MemoryManager::reclaim_all_except(void *keep_ptr)
{
  if (cur_stream == nullptr)
    return;

  std::vector<void *> to_free;
  to_free.reserve(ptr_to_size.size());
  for (const auto &kv : ptr_to_size)
  {
    void *ptr = kv.first;
    if (ptr == nullptr || ptr == keep_ptr)
      continue;
    to_free.push_back(ptr);
  }

  for (void *ptr : to_free)
  {
    cudaError_t free_err = cudaFreeAsync(ptr, *cur_stream);
    if (free_err == cudaErrorInvalidDevicePointer)
      cudaGetLastError();
    ptr_to_size.erase(ptr);
  }

  cudaStreamSynchronize(*cur_stream);

  size_t used_bytes = 0;
  for (const auto &kv : ptr_to_size)
    used_bytes += kv.second;

  stats.used_gpu_memory = used_bytes;
  stats.available_gpu_memory = (stats.gpu_memory_limit > used_bytes) ? (stats.gpu_memory_limit - used_bytes) : 0;
  stats.temporal_memory = 0;
  stats.pinned_memory = 0;
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

    // std::cout << "Free Memory: " << stats.available_gpu_memory / bytes_in_GB << " GB, need Bytes: " << needBytes / bytes_in_GB << " GB" << std::endl;
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

bool MemoryManager::purge_trash(size_t needBytes)
{
  if (trash_queue.empty())
    return false;

  const size_t initial_queue_size = trash_queue.size();
  for (size_t i = 0; i < initial_queue_size && stats.available_gpu_memory < needBytes; ++i)
  {
    int trie_id = static_cast<int>(trash_queue.front());
    trash_queue.pop_front();

    if (trie_id < 0 || static_cast<size_t>(trie_id) >= trie_trashed.size() || !trie_trashed[trie_id])
      continue;

    if (!purge_trash_trie(trie_id))
    {
      // Try again later (likely temporarily pinned).
      trash_queue.push_back(static_cast<uint32_t>(trie_id));
    }
  }

  return stats.available_gpu_memory >= needBytes;
}

bool MemoryManager::purge_trash_trie(int trie_id)
{
  if (trie_id < 0)
    return true;
  if (static_cast<size_t>(trie_id) >= memory_blocks.size() / 2)
    return true;

  const int data_block_id = trie_id * 2;
  const int parent_block_id = data_block_id + 1;

  // Don't purge actively-pinned blocks (dynamic pin). Fixed-pinned blocks can still be purged if trashed.
  if (block_pinned[data_block_id] && !block_pinned_fix[data_block_id])
    return false;
  if (block_pinned[parent_block_id] && !block_pinned_fix[parent_block_id])
    return false;

  auto &utm = UnifiedTrieManager::getInstance();
  UnifiedTrie *trie = utm.getTrie(trie_id);

  auto &data_block = memory_blocks[data_block_id];
  auto &parent_block = memory_blocks[parent_block_id];

  void *data_device_ptr = data_block.device_ptr;
  void *parent_device_ptr = parent_block.device_ptr;
  const size_t data_sz = data_block.size_in_bytes;
  const size_t parent_sz = parent_block.size_in_bytes;

  if (data_device_ptr != nullptr)
  {
    deallocate_permanent(data_block.device_ptr);
  }
  if (parent_device_ptr != nullptr)
  {
    deallocate_permanent(parent_block.device_ptr);
  }

  if (eviction_policy != nullptr)
  {
    eviction_policy->onFree(data_block_id);
    eviction_policy->onFree(parent_block_id);
  }

  if (trie != nullptr)
  {
    if (trie->data != nullptr)
    {
      if (reinterpret_cast<void *>(trie->data->d_data) == data_device_ptr)
        trie->data->d_data = nullptr;
      if (reinterpret_cast<void *>(trie->data->d_parents) == parent_device_ptr)
        trie->data->d_parents = nullptr;
    }
    if (trie->r_data != nullptr)
    {
      if (reinterpret_cast<void *>(trie->r_data->d_data) == data_device_ptr)
        trie->r_data->d_data = nullptr;
      if (reinterpret_cast<void *>(trie->r_data->d_parents) == parent_device_ptr)
        trie->r_data->d_parents = nullptr;
    }
  }

  auto &host_mgr = HostMemoryManager::getInstance();
  if (data_block.host_ptr != nullptr)
  {
    host_mgr.deallocateManaged(data_block.host_ptr);
    data_block.host_ptr = nullptr;
  }
  if (parent_block.host_ptr != nullptr)
  {
    host_mgr.deallocateManaged(parent_block.host_ptr);
    parent_block.host_ptr = nullptr;
  }

  data_block.size_in_bytes = 0;
  parent_block.size_in_bytes = 0;

  if (static_cast<size_t>(trie_id) < trie_data_backedup.size())
    trie_data_backedup[trie_id] = false;
  if (static_cast<size_t>(trie_id) < trie_parent_backedup.size())
    trie_parent_backedup[trie_id] = false;

  if (block_pinned[data_block_id])
  {
    if (stats.pinned_memory >= data_sz)
      stats.pinned_memory -= data_sz;
    else
      stats.pinned_memory = 0;
    block_pinned[data_block_id] = false;
  }
  if (block_pinned[parent_block_id])
  {
    if (stats.pinned_memory >= parent_sz)
      stats.pinned_memory -= parent_sz;
    else
      stats.pinned_memory = 0;
    block_pinned[parent_block_id] = false;
  }

  block_pinned_fix[data_block_id] = false;
  block_pinned_fix[parent_block_id] = false;

  return true;
}

bool MemoryManager::ensure_memory(size_t size)
{
  size = real_size(size); // consider overhead
  if (stats.available_gpu_memory >= size)
  {
    return true;
  }

  if (evict(size))
    return true;

  purge_trash(size);
  if (stats.available_gpu_memory >= size)
    return true;

  return evict(size);
}

void MemoryManager::alert_insufficient_memory(size_t needBytes)
{
  needBytes = real_size(needBytes); // consider overhead

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
