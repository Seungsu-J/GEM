#ifndef STRUCTURE_MEMORYMANAGER_CUH
#define STRUCTURE_MEMORYMANAGER_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cstddef>
#include <chrono>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#include "policy.h"

class MemoryBlock
{
public:
  void *device_ptr;
  void *host_ptr;
  size_t size_in_bytes;

  MemoryBlock()
      : device_ptr(nullptr), host_ptr(nullptr), size_in_bytes(0) {}
  inline bool onDevice() const
  {
    return device_ptr != nullptr;
  }
  inline bool onHost() const
  {
    return host_ptr != nullptr && device_ptr == nullptr;
  }
  inline bool isBackedUp() const
  {
    return host_ptr != nullptr;
  }
};

class MemoryStats
{
public:
  size_t gpu_memory_limit;     // in bytes
  size_t used_gpu_memory;      // in bytes
  size_t available_gpu_memory; // in bytes
  size_t pinned_memory;        // in bytes
  size_t temporal_memory;      // in bytes
  size_t memory_threshold;     // in bytes, threshold to trigger eviction.
  int active_memory_blocks;
  int total_allocations;
  int total_evictions;
  int cache_hits;
  int cache_misses;
  double hit_ratio;
  size_t peak_memory_usage;
  std::chrono::high_resolution_clock::time_point last_update;

  MemoryStats()
      : gpu_memory_limit(0), used_gpu_memory(0), available_gpu_memory(0),
        pinned_memory(0), temporal_memory(0), memory_threshold(0),
        active_memory_blocks(0), total_allocations(0), total_evictions(0),
        cache_hits(0), cache_misses(0), hit_ratio(0.0), peak_memory_usage(0)
  {
    last_update = std::chrono::high_resolution_clock::now();
  }

  void updateAllocation(size_t size)
  {
    used_gpu_memory += size;
    available_gpu_memory -= size;
    total_allocations++;
    if (used_gpu_memory > peak_memory_usage)
    {
      peak_memory_usage = used_gpu_memory;
    }
    last_update = std::chrono::high_resolution_clock::now();
  }
  void updateDeallocation(size_t size)
  {
    used_gpu_memory -= size;
    available_gpu_memory += size;
    last_update = std::chrono::high_resolution_clock::now();
  }
};

class MemoryManager
{
public:
  MemoryStats stats;
  cudaStream_t *cur_stream;
  cudaStream_t *backup_stream;
  VirtualEvictPolicy *eviction_policy;

  // Mapping between device pointers and trie IDs
  std::unordered_map<void *, int> ptr_to_trie;
  std::vector<void *> trie_data_ptrs;
  std::vector<void *> trie_parent_ptrs;

  // Track allocated pointer sizes
  std::unordered_map<void *, size_t> ptr_to_size;

  // uint32_t next_block_id = 0;
  std::vector<MemoryBlock> memory_blocks;
  std::vector<MemoryBlock> reverse_memory_blocks;
  // std::unordered_map<uint32_t, MemoryBlock *> blockid_to_block;

  // Pinned tries (cannot be evicted)
  std::vector<bool> block_pinned;
  std::vector<bool> block_pinned_fix;
  std::vector<bool> trie_data_backedup;
  std::vector<bool> trie_parent_backedup;
  std::vector<uint32_t> climb_parents;
  std::vector<uint32_t> climb_data;

  static MemoryManager &getInstance();
  static void initialize(size_t gpu_memory_limit);
  static void shutdown();
  void init_size(int max_tries);

  // stream config
  void set_stream(cudaStream_t *stream);
  void reset_stream();

  // memory operations
  bool allocate_temporal(void *&ptr, size_t size);
  bool allocate_temporal(uint64_t *&ptr, size_t size);
  bool allocate_temporal(uint32_t *&ptr, size_t size);
  bool allocate_permanent(void *&ptr, size_t size);
  bool allocate_permanent(uint64_t *&ptr, size_t size);
  bool allocate_permanent(uint32_t *&ptr, size_t size);
  bool allocate_trie(int trie_id, bool forward_flag = true);

  bool deallocate_temporal(void *&ptr);
  bool deallocate_temporal(uint64_t *&ptr);
  bool deallocate_temporal(uint32_t *&ptr);
  bool deallocate_permanent(void *&ptr);
  bool deallocate_permanent(uint64_t *&ptr);
  bool deallocate_permanent(uint32_t *&ptr);
  bool deallocate_trie(int trie_id);

  // transfer operations
  bool evict_data(int trie_id);
  bool evict_parent(int trie_id);
  bool evict_trie(int trie_id);

  bool load_data(int trie_id);
  bool load_parent(int trie_id);
  bool load_trie(int trie_id);

  bool is_trie_loaded(int trie_id, bool forward_flag = true);
  bool is_data_loaded(int trie_id, bool forward_flag = true);
  bool is_parent_loaded(int trie_id, bool forward_flag = true);

  bool ensure_trie(int trie_id, bool forward_flag = true);

  bool pin_trie(int trie_id, bool update_flag = true, bool fixed_flag = false);
  bool pin_data(int trie_id, bool update_flag = true, bool fixed_flag = false);
  bool pin_parent(int trie_id, bool update_flag = true, bool fixed_flag = false);
  bool unpin_trie(int trie_id, bool update_flag = true);
  bool unpin_data(int trie_id, bool update_flag = true);
  bool unpin_parent(int trie_id, bool update_flag = true);
  bool unpin_climb_parents();
  bool unpin_climb_data();

  void record_trie_access(int trie_id);
  void record_data_access(int trie_id);
  void record_parent_access(int trie_id);

  bool evict(size_t needBytes);

  // eviction policy
  void set_eviction_policy(VirtualEvictPolicy *policy);

  void reclaim_all_except(void *keep_ptr);

private:
  void alert_insufficient_memory(size_t needBytes);
  bool ensure_memory(size_t size);
};

#endif
