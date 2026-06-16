#include "host_memory_manager.cuh"
#include "cuda_helpers.cuh"
#include "globals.h"
#include <algorithm>
#include <iostream>
#include <chrono>
#include <stdexcept>

namespace
{
  size_t align_up(size_t value, size_t alignment)
  {
    if (alignment == 0)
      return value;
    const size_t rem = value % alignment;
    return rem == 0 ? value : (value + (alignment - rem));
  }
} // namespace

HostMemoryManager &HostMemoryManager::getInstance()
{
  static HostMemoryManager instance;
  return instance;
}

void HostMemoryManager::initialize(size_t total_size_bytes)
{
  HostMemoryManager &manager = getInstance();
  if (!manager.segments_.empty())
  {
    shutdown();
  }

  manager.segments_.clear();
  manager.managed_allocations_.clear();
  manager.total_size_bytes_ = 0;
  manager.current_segment_idx_ = 0;
  manager.max_total_pinned_bytes_ = getSafePinnedMemorySize();
  manager.stats = {};

  if (manager.max_total_pinned_bytes_ == 0)
  {
    throw std::runtime_error("HostMemoryManager: safe pinned memory budget is 0 bytes; "
                             "refusing to allocate pinned memory pool.");
  }

  if (total_size_bytes > manager.max_total_pinned_bytes_)
  {
    std::cerr << "WARNING: Requested host pinned memory pool size "
              << (total_size_bytes / 1024.0 / 1024.0 / 1024.0) << " GB exceeds safe budget "
              << (manager.max_total_pinned_bytes_ / 1024.0 / 1024.0 / 1024.0) << " GB; clamping."
              << std::endl;
    total_size_bytes = manager.max_total_pinned_bytes_;
  }
  if (total_size_bytes == 0)
  {
    throw std::runtime_error("HostMemoryManager: requested pinned memory pool size is 0 bytes.");
  }

  Segment seg;
  seg.size_bytes = total_size_bytes;
  seg.used_bytes = 0;
  cuchk(cudaMallocHost(&seg.data, total_size_bytes));
  manager.segments_.push_back(seg);
  manager.total_size_bytes_ = total_size_bytes;

  manager.stats.mem_pool_size = total_size_bytes;
  manager.stats.mem_pool_used = 0;
  manager.stats.mem_pool_free = total_size_bytes;
  manager.stats.num_allocations = 0;
  manager.stats.num_expansions = 0;
  manager.stats.peak_pool_size = total_size_bytes;
  manager.stats.managed_bytes_used = 0;
  manager.stats.managed_peak_bytes = 0;
  manager.stats.managed_allocations = 0;
  manager.stats.managed_frees = 0;
}

void HostMemoryManager::shutdown()
{
  HostMemoryManager &manager = getInstance();
  for (auto &kv : manager.managed_allocations_)
  {
    if (kv.first != nullptr)
    {
      cudaFreeHost(kv.first);
    }
  }
  manager.managed_allocations_.clear();
  for (auto &seg : manager.segments_)
  {
    if (seg.data != nullptr)
    {
      cudaFreeHost(seg.data);
      seg.data = nullptr;
    }
  }
  manager.segments_.clear();
  manager.total_size_bytes_ = 0;
  manager.current_segment_idx_ = 0;
  manager.max_total_pinned_bytes_ = 0;
  manager.stats = {};
}

void HostMemoryManager::reset()
{
  HostMemoryManager &manager = getInstance();
  if (manager.segments_.empty())
    return;

  for (auto &kv : manager.managed_allocations_)
  {
    if (kv.first != nullptr)
    {
      cudaFreeHost(kv.first);
    }
  }
  manager.managed_allocations_.clear();

  for (auto &seg : manager.segments_)
  {
    seg.used_bytes = 0;
  }
  manager.current_segment_idx_ = 0;

  manager.stats.mem_pool_used = 0;
  manager.stats.mem_pool_free = manager.stats.mem_pool_size;
  manager.stats.num_allocations = 0;

  manager.stats.managed_bytes_used = 0;
  manager.stats.managed_allocations = 0;
  manager.stats.managed_frees = 0;
}

void *HostMemoryManager::allocateBytes(size_t size_bytes, size_t alignment_bytes)
{
  HostMemoryManager &manager = getInstance();
  if (manager.segments_.empty())
  {
    std::cerr << "ERROR: HostMemoryManager not initialized. Call initialize() first.\n";
    return nullptr;
  }

  Segment *seg = &manager.segments_[manager.current_segment_idx_];
  size_t old_used = seg->used_bytes;
  size_t aligned_used = align_up(old_used, alignment_bytes);
  size_t new_used = aligned_used + size_bytes;

  if (new_used > seg->size_bytes)
  {
    bool found_existing_segment = false;
    for (size_t idx = manager.current_segment_idx_ + 1; idx < manager.segments_.size(); ++idx)
    {
      Segment *candidate = &manager.segments_[idx];
      size_t candidate_old_used = candidate->used_bytes;
      size_t candidate_aligned_used = align_up(candidate_old_used, alignment_bytes);
      size_t candidate_new_used = candidate_aligned_used + size_bytes;
      if (candidate_new_used <= candidate->size_bytes)
      {
        manager.current_segment_idx_ = idx;
        seg = candidate;
        old_used = candidate_old_used;
        aligned_used = candidate_aligned_used;
        new_used = candidate_new_used;
        found_existing_segment = true;
        break;
      }
    }

    if (found_existing_segment)
    {
      // Proceed with allocation on the selected existing segment
    }
    else
    {
      if (!expand_pool(size_bytes, alignment_bytes))
        return nullptr;

      seg = &manager.segments_[manager.current_segment_idx_];
      old_used = seg->used_bytes;
      aligned_used = align_up(old_used, alignment_bytes);
      new_used = aligned_used + size_bytes;
    }

    if (new_used > seg->size_bytes)
    {
      std::cerr << "ERROR: Host memory pool expansion logic error. Requested: "
                << (size_bytes / 1024.0 / 1024.0) << " MB, "
                << "Available in new segment: "
                << ((seg->size_bytes - aligned_used) / 1024.0 / 1024.0) << " MB"
                << std::endl;
      return nullptr;
    }
  }

  void *ptr = seg->data + aligned_used;
  seg->used_bytes = new_used;

  const size_t actual_bytes_consumed = new_used - old_used;
  manager.stats.mem_pool_used += actual_bytes_consumed;
  manager.stats.mem_pool_free = manager.stats.mem_pool_size - manager.stats.mem_pool_used;
  manager.stats.num_allocations++;

  return ptr;
}

void *HostMemoryManager::allocateManagedBytes(size_t size_bytes, size_t alignment_bytes)
{
  auto time_st = std::chrono::high_resolution_clock::now();

  HostMemoryManager &manager = getInstance();
  if (size_bytes == 0)
    return nullptr;

  if (manager.max_total_pinned_bytes_ != 0)
  {
    const size_t pinned_in_use = manager.total_size_bytes_ + manager.stats.managed_bytes_used;
    if (size_bytes > manager.max_total_pinned_bytes_ || pinned_in_use > manager.max_total_pinned_bytes_ - size_bytes)
    {
      std::cerr << "ERROR: Host pinned memory budget exceeded. Requested managed allocation "
                << (size_bytes / 1024.0 / 1024.0 / 1024.0) << " GB; current pinned usage "
                << (pinned_in_use / 1024.0 / 1024.0 / 1024.0) << " GB; budget "
                << (manager.max_total_pinned_bytes_ / 1024.0 / 1024.0 / 1024.0) << " GB."
                << std::endl;
      return nullptr;
    }
  }

  void *ptr = nullptr;
  cudaError_t err = cudaMallocHost(&ptr, size_bytes);
  auto time_ed = std::chrono::high_resolution_clock::now();
  auto time_duration = std::chrono::duration_cast<std::chrono::milliseconds>(time_ed - time_st);
  HOST_MEM_EXPANSION_TIME += time_duration.count();
  if (err != cudaSuccess)
  {
    std::cerr << "ERROR: Failed to allocate managed host memory. cudaMallocHost failed: "
              << cudaGetErrorString(err) << std::endl;
    return nullptr;
  }

  manager.managed_allocations_.emplace(ptr, size_bytes);
  manager.stats.managed_bytes_used += size_bytes;
  manager.stats.managed_peak_bytes = std::max(manager.stats.managed_peak_bytes, manager.stats.managed_bytes_used);
  manager.stats.managed_allocations++;

  return ptr;
}

bool HostMemoryManager::deallocateManaged(void *&ptr)
{
  auto time_st = std::chrono::high_resolution_clock::now();
  HostMemoryManager &manager = getInstance();
  if (ptr == nullptr)
    return true;

  auto it = manager.managed_allocations_.find(ptr);
  if (it == manager.managed_allocations_.end())
    return false;

  size_t size_bytes = it->second;
  cuchk(cudaFreeHost(ptr));
  manager.managed_allocations_.erase(it);
  auto time_ed = std::chrono::high_resolution_clock::now();
  auto time_duration = std::chrono::duration_cast<std::chrono::milliseconds>(time_ed - time_st);
  HOST_MEM_EXPANSION_TIME += time_duration.count();

  if (manager.stats.managed_bytes_used >= size_bytes)
    manager.stats.managed_bytes_used -= size_bytes;
  else
    manager.stats.managed_bytes_used = 0;

  manager.stats.managed_frees++;
  ptr = nullptr;
  return true;
}

bool HostMemoryManager::expand_pool(size_t required_size_bytes, size_t alignment_bytes)
{
  auto expansion_start = std::chrono::high_resolution_clock::now();

  // Grow total capacity by 50% OR ensure enough headroom for (used + this request).
  // Also ensure the new segment itself can satisfy the request (single contiguous allocation).
  size_t min_new_total = stats.mem_pool_used + required_size_bytes;
  size_t headroom_new_total = min_new_total + (min_new_total / 5); // 20% headroom
  size_t growth_new_total = static_cast<size_t>(total_size_bytes_ * GROWTH_FACTOR);

  size_t desired_total = std::max({growth_new_total, headroom_new_total, min_new_total});
  size_t additional_bytes = desired_total > total_size_bytes_ ? (desired_total - total_size_bytes_) : 0;
  size_t required_segment_bytes = required_size_bytes + (required_size_bytes / 5); // 20% headroom for this request
  additional_bytes = std::max(additional_bytes, required_segment_bytes);

  // Add a little padding so we can always align within the new segment.
  additional_bytes = align_up(additional_bytes, std::max<size_t>(alignment_bytes, 1));
  size_t new_total = total_size_bytes_ + additional_bytes;

  if (max_total_pinned_bytes_ != 0)
  {
    const size_t pinned_in_use = total_size_bytes_ + stats.managed_bytes_used;
    if (pinned_in_use >= max_total_pinned_bytes_)
    {
      std::cerr << "ERROR: Host pinned memory budget exceeded before expansion. "
                << "Pinned in use: " << (pinned_in_use / 1024.0 / 1024.0 / 1024.0) << " GB, budget: "
                << (max_total_pinned_bytes_ / 1024.0 / 1024.0 / 1024.0) << " GB." << std::endl;
      return false;
    }

    const size_t max_additional = max_total_pinned_bytes_ - pinned_in_use;
    if (additional_bytes > max_additional)
    {
      additional_bytes = align_up(max_additional, std::max<size_t>(alignment_bytes, 1));
      new_total = total_size_bytes_ + additional_bytes;
    }

    if (additional_bytes < required_size_bytes)
    {
      std::cerr << "ERROR: Host pinned memory budget prevents allocating required segment. "
                << "Required: " << (required_size_bytes / 1024.0 / 1024.0) << " MB, "
                << "available for expansion: " << (max_additional / 1024.0 / 1024.0) << " MB."
                << std::endl;
      return false;
    }
  }

  std::cout << "Expanding host memory pool from "
            << (total_size_bytes_ / 1024.0 / 1024.0) << " MB to "
            << (new_total / 1024.0 / 1024.0) << " MB (expansion #"
            << (stats.num_expansions + 1) << ")" << std::endl;

  uint8_t *new_data = nullptr;
  cudaError_t err = cudaMallocHost(&new_data, additional_bytes);

  if (err != cudaSuccess)
  {
    std::cerr << "ERROR: Failed to expand host memory pool. cudaMallocHost failed: "
              << cudaGetErrorString(err) << std::endl;
    std::cerr << "       Attempted additional allocation size: " << (additional_bytes / 1024.0 / 1024.0) << " MB"
              << std::endl;

    auto expansion_end = std::chrono::high_resolution_clock::now();
    auto expansion_duration = std::chrono::duration_cast<std::chrono::milliseconds>(expansion_end - expansion_start);
    HOST_MEM_EXPANSION_TIME += expansion_duration.count();

    return false;
  }

  Segment seg;
  seg.data = new_data;
  seg.size_bytes = additional_bytes;
  seg.used_bytes = 0;
  segments_.push_back(seg);
  current_segment_idx_ = segments_.size() - 1;
  total_size_bytes_ = new_total;

  // Update statistics
  stats.mem_pool_size = new_total;
  stats.mem_pool_free = new_total - stats.mem_pool_used;
  stats.num_expansions++;
  stats.peak_pool_size = std::max(stats.peak_pool_size, new_total);

  auto expansion_end = std::chrono::high_resolution_clock::now();
  auto expansion_duration = std::chrono::duration_cast<std::chrono::milliseconds>(expansion_end - expansion_start);
  double expansion_time_ms = expansion_duration.count();
  HOST_MEM_EXPANSION_TIME += expansion_time_ms;

  std::cout << "Host memory pool expansion successful. New free space: "
            << (stats.mem_pool_free / 1024.0 / 1024.0) << " MB"
            << " (took " << expansion_time_ms << " ms)" << std::endl;

  return true;
}
