#include "host_memory_manager.cuh"
#include "cuda_helpers.cuh"

HostMemoryManager &HostMemoryManager::getInstance()
{
  static HostMemoryManager instance;
  return instance;
}

void HostMemoryManager::initialize(size_t total_size_bytes)
{
  HostMemoryManager &manager = getInstance();
  manager.total_size_bytes_ = total_size_bytes;
  manager.current_index_tail = 0;
  cuchk(cudaMallocHost(&manager.h_data, total_size_bytes));

  manager.stats.mem_pool_size = total_size_bytes;
  manager.stats.mem_pool_used = 0;
  manager.stats.mem_pool_free = total_size_bytes;
  manager.stats.num_allocations = 0;
}

void HostMemoryManager::shutdown()
{
  HostMemoryManager &manager = getInstance();
  cudaFreeHost(manager.h_data);
  manager.h_data = nullptr;
}

void *HostMemoryManager::get_next_ptr(size_t elem_size)
{
  if (current_index_tail & 1) // odd
    current_index_tail++;

  return reinterpret_cast<void *>(h_data + current_index_tail);
}

bool HostMemoryManager::allocate(size_t size_bytes)
{
  if (stats.mem_pool_free >= size_bytes)
  {
    current_index_tail += size_bytes / sizeof(uint32_t);
  }
  else
  {
    return false;
  }

  stats.mem_pool_used += size_bytes;
  stats.mem_pool_free -= size_bytes;
  stats.num_allocations++;

  return true;
}