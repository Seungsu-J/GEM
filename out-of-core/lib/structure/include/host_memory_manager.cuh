#ifndef STRUCTURE_HOSTMEMORYMANAGER_CUH
#define STRUCTURE_HOSTMEMORYMANAGER_CUH

#include <cinttypes>

struct HostStats
{
  size_t mem_pool_size;
  size_t mem_pool_used;
  size_t mem_pool_free;
  size_t num_allocations;
};

class HostMemoryManager
{
public:
  static HostMemoryManager &getInstance();
  static void initialize(size_t total_size_bytes);
  static void shutdown();

  uint32_t *h_data;
  size_t total_size_bytes_;
  uint32_t current_index_tail;

  HostStats stats;

  void *get_next_ptr(size_t elem_size);
  bool allocate(size_t size_bytes);
  void reset() { current_index_tail = 0; }
};

#endif