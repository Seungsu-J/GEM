#ifndef STRUCTURE_HOSTMEMORYMANAGER_CUH
#define STRUCTURE_HOSTMEMORYMANAGER_CUH

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

struct HostStats
{
  size_t mem_pool_size;
  size_t mem_pool_used;
  size_t mem_pool_free;
  size_t num_allocations;
  size_t num_expansions;
  size_t peak_pool_size;
  size_t managed_bytes_used;
  size_t managed_peak_bytes;
  size_t managed_allocations;
  size_t managed_frees;
};

class HostMemoryManager
{
public:
  static HostMemoryManager &getInstance();
  static void initialize(size_t total_size_bytes);
  static void shutdown();

  HostStats stats;

  void *allocateBytes(size_t size_bytes, size_t alignment_bytes);
  void *allocateManagedBytes(size_t size_bytes, size_t alignment_bytes);
  bool deallocateManaged(void *&ptr);
  void reset();

  template <typename T>
  T *allocate(size_t count)
  {
    return static_cast<T *>(allocateBytes(sizeof(T) * count, alignof(T)));
  }

private:
  struct Segment
  {
    uint8_t *data = nullptr;
    size_t size_bytes = 0;
    size_t used_bytes = 0;
  };

  std::vector<Segment> segments_;
  std::unordered_map<void *, size_t> managed_allocations_;
  size_t total_size_bytes_ = 0;
  size_t current_segment_idx_ = 0;
  size_t max_total_pinned_bytes_ = 0;

  bool expand_pool(size_t required_size_bytes, size_t alignment_bytes);
  static constexpr double GROWTH_FACTOR = 1.5; // Expand total size by 50%
};

#endif
