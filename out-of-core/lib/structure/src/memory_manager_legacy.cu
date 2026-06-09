// #include "lattice.h"
// #include "memory_manager.h"
// #include "policy.h"  // NEW: Include policy classes
// #include "unifiedTrie.cuh"
// #include <algorithm>
// #include <cmath>
// #include <cuda_runtime.h>
// #include <iomanip>
// #include <iostream>
// #include <mutex>
// #include <limits>  // NEW: For NO_VICTIM sentinel

// #include "cuda_helpers.cuh"

// // AccessPattern implementation
// void AccessPattern::record_access()
// {
//   auto now = std::chrono::high_resolution_clock::now();
//   access_times.push_back(now);
//   access_count++;

//   // Keep only recent access times (last 10 accesses)
//   if (access_times.size() > 10)
//   {
//     access_times.erase(access_times.begin());
//   }

//   update_prediction();
// }

// void AccessPattern::update_prediction()
// {
//   if (access_times.size() < 2)
//     return;

//   // Calculate average interval between accesses
//   double total_interval = 0.0;
//   for (size_t i = 1; i < access_times.size(); i++)
//   {
//     auto interval = std::chrono::duration_cast<std::chrono::milliseconds>(
//                         access_times[i] - access_times[i - 1])
//                         .count();
//     total_interval += interval;
//   }

//   average_interval = total_interval / (access_times.size() - 1);

//   // Check if access pattern is periodic
//   if (access_times.size() >= 3)
//   {
//     double variance = 0.0;
//     for (size_t i = 1; i < access_times.size(); i++)
//     {
//       auto interval = std::chrono::duration_cast<std::chrono::milliseconds>(
//                           access_times[i] - access_times[i - 1])
//                           .count();
//       variance += std::pow(interval - average_interval, 2);
//     }
//     variance /= (access_times.size() - 1);

//     // Consider periodic if variance is low
//     is_periodic = (variance < average_interval * 0.3);
//   }

//   // Predict next access time
//   if (is_periodic && !access_times.empty())
//   {
//     predicted_next_access =
//         access_times.back() +
//         std::chrono::milliseconds(static_cast<int>(average_interval));
//   }
// }

// bool AccessPattern::should_prefetch() const
// {
//   if (!is_periodic)
//     return false;

//   auto now = std::chrono::high_resolution_clock::now();
//   auto time_to_predicted =
//       std::chrono::duration_cast<std::chrono::milliseconds>(
//           predicted_next_access - now)
//           .count();

//   // Prefetch if predicted access is within the next 1000ms
//   return (time_to_predicted > 0 && time_to_predicted < 1000);
// }

// // MemoryManager implementation
// MemoryManager::MemoryManager()
//     : gpu_memory_limit(0), current_gpu_memory(0), memory_threshold(0),
//       current_policy(EvictionPolicy::LRU),
//       transfer_strategy(TransferStrategy::IMMEDIATE),
//       current_policy_impl(nullptr), next_block_id(0),  // NEW
//       batch_size_threshold(1024 * 1024), // 1MB default batch size
//       enable_prediction(true), enable_stats(true), hybrid_lru_weight(0.6),
//       hybrid_contribution_weight(0.4), max_prefetch_items(5),
//       prefetch_timeout(std::chrono::milliseconds(100))
// {

//   // Get GPU memory information
//   size_t free_memory, total_memory;
//   cudaMemGetInfo(&free_memory, &total_memory);

//   if (gpu_memory_limit == 0)
//   {
//     // Use 90% of available GPU memory by default
//     gpu_memory_limit = static_cast<size_t>(total_memory * 0.9);
//   }

//   memory_threshold = static_cast<size_t>(gpu_memory_limit * 0.9);
//   stats.total_gpu_memory = total_memory;

// #ifndef NDEBUG
//   std::cout << "MemoryManager initialized with "
//             << (gpu_memory_limit / (1024 * 1024)) << " MB GPU memory limit"
//             << std::endl;
// #endif
// }

// MemoryManager::~MemoryManager()
// {
//   // NEW: Clean up policy object
//   if (current_policy_impl)
//   {
//     delete current_policy_impl;
//     current_policy_impl = nullptr;
//   }

//   cleanup_memory_pool();
//   if (enable_stats)
//   {
//     print_detailed_stats();
//   }
// }

// bool MemoryManager::allocate_gpu_memory(int intermediate_id, size_t size,
//                                         void **ptr)
// {
//   if (!ensure_available_memory(size))
//   {
//     std::cerr << "Failed to ensure " << size << " bytes for intermediate "
//               << intermediate_id << std::endl;
//     return false;
//   }

//   // Try to allocate from memory pool first
//   *ptr = allocate_from_pool(size, intermediate_id);
//   if (*ptr == nullptr)
//   {
//     // Fall back to direct CUDA allocation
//     cudaError_t err = cudaMalloc(ptr, size);
//     if (err != cudaSuccess)
//     {
//       std::cerr << "CUDA allocation failed: " << cudaGetErrorString(err)
//                 << std::endl;
//       return false;
//     }

//     // For CUDA fallback allocations, we need to track the intermediate_id
//     // separately since they're not in the memory pool
//     direct_allocations[*ptr] = intermediate_id;
//   }

//   // Update tracking (works for both pool and direct CUDA allocations)
//   allocation_map[*ptr] = size;
//   current_gpu_memory += size;
//   stats.used_gpu_memory = current_gpu_memory;
//   stats.total_allocations++;

//   if (current_gpu_memory > stats.peak_memory_usage)
//   {
//     stats.peak_memory_usage = current_gpu_memory;
//   }

//   // NEW: Notify policy about allocation
//   if (current_policy_impl)
//   {
//     uint32_t block_id = get_block_id(intermediate_id);
//     current_policy_impl->onAllocate(block_id, size);
//   }

//   record_access(intermediate_id);

//   if (enable_stats)
//   {
//     update_statistics();
//   }

//   return true;
// }

// bool MemoryManager::deallocate_gpu_memory(int intermediate_id)
// {
//   // NEW: Notify policy about deallocation BEFORE freeing
//   if (current_policy_impl)
//   {
//     uint32_t block_id = get_block_id(intermediate_id);
//     current_policy_impl->onFree(block_id);
//   }

//   // Cleanup mapping
//   auto it_map = intermediate_to_blockid.find(intermediate_id);
//   if (it_map != intermediate_to_blockid.end())
//   {
//     uint32_t block_id = it_map->second;
//     intermediate_to_blockid.erase(it_map);
//     blockid_to_intermediate.erase(block_id);
//   }

//   // First, try to find the allocation in the memory pool
//   for (auto &block : memory_pool)
//   {
//     if (block.intermediate_id == intermediate_id && !block.is_free)
//     {
//       block.is_free = true;
//       block.intermediate_id = -1;
//       current_gpu_memory -= block.size;
//       stats.used_gpu_memory = current_gpu_memory;

//       if (enable_stats)
//       {
//         update_statistics();
//       }
//       return true;
//     }
//   }

//   // If not found in pool, check direct CUDA allocations
//   for (auto it = direct_allocations.begin(); it != direct_allocations.end();
//        ++it)
//   {
//     if (it->second == intermediate_id)
//     {
//       void *ptr = it->first;

//       // Find the size in allocation_map
//       auto size_it = allocation_map.find(ptr);
//       if (size_it != allocation_map.end())
//       {
//         size_t size = size_it->second;

//         // Free the CUDA memory
//         cudaFree(ptr);

//         // Update tracking
//         current_gpu_memory -= size;
//         stats.used_gpu_memory = current_gpu_memory;
//         allocation_map.erase(size_it);
//         direct_allocations.erase(it);

//         if (enable_stats)
//         {
//           update_statistics();
//         }
//         return true;
//       }
//     }
//   }

//   std::cerr << "Warning: Could not find allocation for intermediate "
//             << intermediate_id << std::endl;
//   return false;
// }

// bool MemoryManager::ensure_available_memory(size_t required_size)
// {
//   if (current_gpu_memory + required_size <= memory_threshold)
//   {
//     return true;
//   }

// #ifndef NDEBUG
//   std::cout << "Need to free memory: required=" << required_size
//             << ", current=" << current_gpu_memory
//             << ", threshold=" << memory_threshold << std::endl;
// #endif

//   // Calculate how much memory we need to free
//   size_t memory_to_free =
//       (current_gpu_memory + required_size) - memory_threshold;

//   // Get eviction candidates based on current policy
//   std::vector<int> candidates;

//   // NEW: Use policy object if available
//   if (current_policy_impl)
//   {
//     constexpr uint32_t NO_VICTIM = std::numeric_limits<uint32_t>::max();
//     size_t freed_memory = 0;

//     while (freed_memory < memory_to_free)
//     {
//       uint32_t victim_block_id = current_policy_impl->pickVictim(memory_to_free - freed_memory);

//       if (victim_block_id == NO_VICTIM)
//       {
//         // No more victims available
//         break;
//       }

//       int victim_intermediate_id = get_intermediate_id(victim_block_id);
//       if (victim_intermediate_id < 0)
//       {
//         std::cerr << "Warning: Invalid block_id " << victim_block_id << " from policy" << std::endl;
//         break;
//       }

//       candidates.push_back(victim_intermediate_id);
//       freed_memory += get_intermediate_memory_size(victim_intermediate_id);
//     }
//   }
//   else
//   {
//     // Fallback: use old switch-based logic
//     switch (current_policy)
//     {
//     case EvictionPolicy::LRU:
//       candidates = get_lru_eviction_candidates(memory_to_free);
//       break;
//     case EvictionPolicy::LFU:
//       candidates = get_lfu_eviction_candidates(memory_to_free);
//       break;
//     case EvictionPolicy::CONTRIBUTION:
//       candidates = get_contribution_eviction_candidates(memory_to_free);
//       break;
//     case EvictionPolicy::HYBRID:
//       candidates = get_hybrid_eviction_candidates(memory_to_free);
//       break;
//     case EvictionPolicy::SIZE_BASED:
//       candidates = get_size_based_eviction_candidates(memory_to_free);
//       break;
//     default:
//       candidates = get_lru_eviction_candidates(memory_to_free);
//       break;
//     }
//   }

//   if (candidates.empty())
//   {
//     std::cerr << "No eviction candidates found!" << std::endl;
//     return false;
//   }

//   // Execute eviction
//   execute_eviction(candidates);

//   return (current_gpu_memory + required_size <= memory_threshold);
// }

// void MemoryManager::set_eviction_policy(std::string policy)
// {
//   if (policy == "lru")
//   {
//     current_policy = EvictionPolicy::LRU;
//     set_policy_object(new LRUEvictPolicy());
//   }
//   else if (policy == "lfu")
//   {
//     current_policy = EvictionPolicy::LFU;
//     set_policy_object(new LFUEvictPolicy());
//   }
//   else if (policy == "contribution")
//   {
//     current_policy = EvictionPolicy::CONTRIBUTION;
//     set_policy_object(new ContributionEvictPolicy());
//   }
//   else if (policy == "hybrid")
//   {
//     current_policy = EvictionPolicy::HYBRID;
//     auto* hybrid = new HybridEvictPolicy();
//     hybrid->setWeights(hybrid_lru_weight, hybrid_contribution_weight);
//     set_policy_object(hybrid);
//   }
//   else if (policy == "size_based")
//   {
//     current_policy = EvictionPolicy::SIZE_BASED;
//     set_policy_object(new SizeBasedEvictPolicy());
//   }
//   else if (policy == "fifo")
//   {
//     current_policy = EvictionPolicy::FIFO;
//     set_policy_object(new FIFOEvictPolicy());
//   }
//   else
//   {
//     std::cerr << "Unknown eviction policy: " << policy << std::endl;
//   }
// }

// void MemoryManager::set_eviction_policy(EvictionPolicy policy)
// {
//   current_policy = policy;
// #ifndef NDEBUG
//   std::cout << "Eviction policy changed to: ";
//   switch (policy)
//   {
//   case EvictionPolicy::LRU:
//     std::cout << "LRU";
//     break;
//   case EvictionPolicy::LFU:
//     std::cout << "LFU";
//     break;
//   case EvictionPolicy::CONTRIBUTION:
//     std::cout << "CONTRIBUTION";
//     break;
//   case EvictionPolicy::HYBRID:
//     std::cout << "HYBRID";
//     break;
//   case EvictionPolicy::SIZE_BASED:
//     std::cout << "SIZE_BASED";
//     break;
//   case EvictionPolicy::FIFO:
//     std::cout << "FIFO";
//     break;
//   }
//   std::cout << std::endl;
// #endif
// }

// std::vector<int>
// MemoryManager::get_lru_eviction_candidates(size_t required_memory)
// {
//   std::vector<std::pair<std::chrono::high_resolution_clock::time_point, int>>
//       candidates;

//   for (const auto &[id, access_time] : last_access_times)
//   {
//     if (pinned_intermediates.find(id) ==
//         pinned_intermediates.end()) // Not pinned
//     {
//       candidates.emplace_back(access_time, id);
//     }
//   }

//   // Sort by access time (oldest first)
//   std::sort(candidates.begin(), candidates.end());

//   std::vector<int> result;
//   size_t freed_memory = 0;

//   for (const auto &[time, id] : candidates)
//   {
//     result.push_back(id);
//     freed_memory += get_intermediate_memory_size(id);
//     if (freed_memory >= required_memory)
//       break;
//   }

//   return result;
// }

// std::vector<int>
// MemoryManager::get_contribution_eviction_candidates(size_t required_memory)
// {
//   std::vector<std::pair<int, int>> candidates; // (score, id)

//   for (const auto &[id, score] : contribution_scores)
//   {
//     if (pinned_intermediates.find(id) == pinned_intermediates.end())
//     {
//       candidates.emplace_back(score, id);
//     }
//   }

//   // Sort by contribution score (lowest first)
//   std::sort(candidates.begin(), candidates.end());

//   std::vector<int> result;
//   size_t freed_memory = 0;

//   for (const auto &[score, id] : candidates)
//   {
//     result.push_back(id);
//     freed_memory += get_intermediate_memory_size(id);
//     if (freed_memory >= required_memory)
//       break;
//   }

//   return result;
// }

// std::vector<int>
// MemoryManager::get_hybrid_eviction_candidates(size_t required_memory)
// {
//   std::vector<std::pair<double, int>> candidates; // (hybrid_score, id)

//   for (const auto &[id, access_time] : last_access_times)
//   {
//     if (pinned_intermediates.find(id) == pinned_intermediates.end())
//     {
//       double hybrid_score = calculate_hybrid_score(id);
//       candidates.emplace_back(hybrid_score, id);
//     }
//   }

//   // Sort by hybrid score (lowest first)
//   std::sort(candidates.begin(), candidates.end());

//   std::vector<int> result;
//   size_t freed_memory = 0;

//   for (const auto &[score, id] : candidates)
//   {
//     result.push_back(id);
//     freed_memory += get_intermediate_memory_size(id);
//     if (freed_memory >= required_memory)
//       break;
//   }

//   return result;
// }

// std::vector<int>
// MemoryManager::get_size_based_eviction_candidates(size_t required_memory)
// {
//   std::vector<std::pair<size_t, int>> candidates; // (size, id)

//   for (const auto &block : memory_pool)
//   {
//     if (!block.is_free && block.intermediate_id != -1 &&
//         pinned_intermediates.find(block.intermediate_id) ==
//             pinned_intermediates.end())
//     {
//       candidates.emplace_back(block.size, block.intermediate_id);
//     }
//   }

//   // Sort by size (largest first)
//   std::sort(candidates.rbegin(), candidates.rend());

//   std::vector<int> result;
//   size_t freed_memory = 0;

//   for (const auto &[size, id] : candidates)
//   {
//     result.push_back(id);
//     freed_memory += size;
//     if (freed_memory >= required_memory)
//       break;
//   }

//   return result;
// }

// std::vector<int>
// MemoryManager::get_lfu_eviction_candidates(size_t required_memory)
// {
//   std::vector<std::pair<int, int>> candidates; // (frequency, id)

//   for (const auto &[id, frequency] : access_frequencies)
//   {
//     if (pinned_intermediates.find(id) == pinned_intermediates.end())
//     {
//       candidates.emplace_back(frequency, id);
//     }
//   }

//   // Sort by frequency (lowest first)
//   std::sort(candidates.begin(), candidates.end());

//   std::vector<int> result;
//   size_t freed_memory = 0;

//   for (const auto &[freq, id] : candidates)
//   {
//     result.push_back(id);
//     freed_memory += get_intermediate_memory_size(id);
//     if (freed_memory >= required_memory)
//       break;
//   }

//   return result;
// }

// void MemoryManager::record_access(int intermediate_id)
// {
//   auto now = std::chrono::high_resolution_clock::now();
//   last_access_times[intermediate_id] = now;
//   access_frequencies[intermediate_id]++;

//   // NEW: Notify policy about access
//   if (current_policy_impl)
//   {
//     uint32_t block_id = get_block_id(intermediate_id);
//     current_policy_impl->onAccess(block_id);
//   }

//   if (enable_prediction)
//   {
//     access_patterns[intermediate_id].record_access();
//   }

//   stats.cache_hits++;
//   if (enable_stats)
//   {
//     update_statistics();
//   }
// }

// void MemoryManager::pin_intermediate(int intermediate_id)
// {
//   pinned_intermediates.insert(intermediate_id);
// #ifndef NDEBUG
//   std::cout << "Pinned intermediate " << intermediate_id << " to GPU"
//             << std::endl;
// #endif
// }

// void MemoryManager::unpin_intermediate(int intermediate_id)
// {
//   pinned_intermediates.erase(intermediate_id);
// #ifndef NDEBUG
//   std::cout << "Unpinned intermediate " << intermediate_id << " from GPU"
//             << std::endl;
// #endif
// }

// void MemoryManager::execute_eviction(const std::vector<int> &candidates)
// {
//   for (int id : candidates)
//   {
//     if (deallocate_gpu_memory(id))
//     {
//       stats.total_evictions++;
// #ifndef NDEBUG
//       std::cout << "Evicted intermediate " << id << " from GPU" << std::endl;
// #endif
//     }
//   }

//   // Merge free blocks after eviction
//   merge_free_blocks();
// }

// void MemoryManager::update_statistics()
// {
//   stats.used_gpu_memory = current_gpu_memory;
//   stats.available_gpu_memory = gpu_memory_limit - current_gpu_memory;
//   stats.active_intermediates = allocation_map.size();

//   if (stats.cache_hits + stats.cache_misses > 0)
//   {
//     stats.hit_ratio = static_cast<double>(stats.cache_hits) /
//                       (stats.cache_hits + stats.cache_misses);
//   }

//   stats.last_update = std::chrono::high_resolution_clock::now();
// }

// MemoryStats MemoryManager::get_statistics() const { return stats; }

// void MemoryManager::print_memory_status() const
// {
// #ifndef NDEBUG
//   std::cout << "\n=== Memory Manager Status ===" << std::endl;
//   std::cout << "GPU Memory: " << (current_gpu_memory / (1024 * 1024)) << "MB / "
//             << (gpu_memory_limit / (1024 * 1024)) << "MB (" << std::fixed
//             << std::setprecision(1)
//             << (100.0 * current_gpu_memory / gpu_memory_limit) << "%)"
//             << std::endl;
//   std::cout << "Active intermediates: " << allocation_map.size() << std::endl;
//   std::cout << "Eviction policy: ";
//   switch (current_policy)
//   {
//   case EvictionPolicy::LRU:
//     std::cout << "LRU";
//     break;
//   case EvictionPolicy::CONTRIBUTION:
//     std::cout << "CONTRIBUTION";
//     break;
//   case EvictionPolicy::HYBRID:
//     std::cout << "HYBRID";
//     break;
//   default:
//     std::cout << "OTHER";
//     break;
//   }
//   std::cout << std::endl;
// #endif
// }

// void MemoryManager::print_detailed_stats() const
// {
// #ifndef NDEBUG
//   std::cout << "\n=== Detailed Memory Statistics ===" << std::endl;
//   std::cout << "Total allocations: " << stats.total_allocations << std::endl;
//   std::cout << "Total evictions: " << stats.total_evictions << std::endl;
//   std::cout << "Cache hit ratio: " << std::fixed << std::setprecision(2)
//             << (stats.hit_ratio * 100) << "%" << std::endl;
//   std::cout << "Peak memory usage: "
//             << (stats.peak_memory_usage / (1024 * 1024)) << "MB" << std::endl;
// #endif
// }

// double MemoryManager::calculate_hybrid_score(int intermediate_id) const
// {
//   // Combine LRU and contribution scores
//   double lru_score = 0.0;
//   double contribution_score = 0.0;

//   auto access_it = last_access_times.find(intermediate_id);
//   if (access_it != last_access_times.end())
//   {
//     auto now = std::chrono::high_resolution_clock::now();
//     auto time_since_access =
//         std::chrono::duration_cast<std::chrono::milliseconds>(now -
//                                                               access_it->second)
//             .count();
//     lru_score = static_cast<double>(time_since_access);
//   }

//   auto contrib_it = contribution_scores.find(intermediate_id);
//   if (contrib_it != contribution_scores.end())
//   {
//     contribution_score = static_cast<double>(contrib_it->second);
//   }

//   // Normalize and combine scores
//   return (hybrid_lru_weight * lru_score) +
//          (hybrid_contribution_weight * (1000.0 - contribution_score));
// }

// size_t MemoryManager::get_intermediate_memory_size(int intermediate_id) const
// {
//   for (const auto &block : memory_pool)
//   {
//     if (block.intermediate_id == intermediate_id && !block.is_free)
//     {
//       return block.size;
//     }
//   }
//   return 0;
// }

// void MemoryManager::initialize_memory_pool(size_t pool_size)
// {
//   void *pool_ptr;
//   cudaError_t err = cudaMalloc(&pool_ptr, pool_size);
//   if (err != cudaSuccess)
//   {
//     std::cerr << "Failed to initialize memory pool: " << cudaGetErrorString(err)
//               << std::endl;
//     return;
//   }

//   memory_pool.emplace_back(pool_ptr, pool_size);
// #ifndef NDEBUG
//   std::cout << "Initialized memory pool with " << (pool_size / (1024 * 1024))
//             << "MB" << std::endl;
// #endif
// }

// void *MemoryManager::allocate_from_pool(size_t size, int intermediate_id)
// {
//   // Find a suitable free block
//   for (auto &block : memory_pool)
//   {
//     if (block.is_free && block.size >= size)
//     {
//       // Split the block if it's much larger than needed
//       if (block.size > size * 2)
//       {
//         void *new_ptr = static_cast<char *>(block.ptr) + size;
//         size_t remaining_size = block.size - size;
//         memory_pool.emplace_back(new_ptr, remaining_size);
//         block.size = size;
//       }

//       block.is_free = false;
//       block.intermediate_id = intermediate_id; // Associate with intermediate
//       return block.ptr;
//     }
//   }

//   return nullptr; // No suitable block found
// }

// void MemoryManager::merge_free_blocks()
// {
//   // Sort blocks by address for efficient merging
//   std::sort(
//       memory_pool.begin(), memory_pool.end(),
//       [](const MemoryBlock &a, const MemoryBlock &b)
//       { return a.ptr < b.ptr; });

//   // Merge adjacent free blocks
//   for (size_t i = 0; i < memory_pool.size() - 1;)
//   {
//     if (memory_pool[i].is_free && memory_pool[i + 1].is_free)
//     {
//       char *end_of_first =
//           static_cast<char *>(memory_pool[i].ptr) + memory_pool[i].size;
//       if (end_of_first == memory_pool[i + 1].ptr)
//       {
//         // Merge the blocks
//         memory_pool[i].size += memory_pool[i + 1].size;
//         memory_pool.erase(memory_pool.begin() + i + 1);
//         continue;
//       }
//     }
//     i++;
//   }
// }

// void MemoryManager::cleanup_memory_pool()
// {
//   for (const auto &block : memory_pool)
//   {
//     if (block.ptr != nullptr)
//       cudaFree(block.ptr);
//   }
//   memory_pool.clear();
//   allocation_map.clear();
// }

// // Additional missing methods
// void MemoryManager::set_gpu_memory_limit(size_t limit)
// {
//   gpu_memory_limit = limit;
//   memory_threshold = static_cast<size_t>(limit * 0.8);
//   stats.total_gpu_memory = limit;
// #ifndef NDEBUG
//   std::cout << "GPU memory limit set to " << (limit / (1024 * 1024)) << " MB"
//             << std::endl;
// #endif
// }

// // void MemoryManager::register_intermediate(LayeredTrie *intermediate)
// // {
// //   if (intermediate)
// //   {
// //     access_patterns[intermediate->inter_id] = AccessPattern(intermediate->inter_id);
// // #ifndef NDEBUG
// //     std::cout << "Registered intermediate " << intermediate->inter_id
// //               << " with memory manager" << std::endl;
// // #endif
// //   }
// // }

// // void MemoryManager::register_intermediate_trie(LayeredTrie *trie)
// // {
// //   if (trie)
// //   {
// //     access_patterns[trie->inter_id] = AccessPattern(trie->inter_id);
// // #ifndef NDEBUG
// //     std::cout << "Registered LayeredTrie " << trie->inter_id
// //               << " with memory manager" << std::endl;
// // #endif
// //   }
// // }

// void MemoryManager::unregister_intermediate(int intermediate_id)
// {
//   access_patterns.erase(intermediate_id);
//   last_access_times.erase(intermediate_id);
//   access_frequencies.erase(intermediate_id);
//   contribution_scores.erase(intermediate_id);
//   pinned_intermediates.erase(intermediate_id);

//   // Clean up any memory allocations
//   deallocate_gpu_memory(intermediate_id);

// #ifndef NDEBUG
//   std::cout << "Unregistered intermediate " << intermediate_id
//             << " from memory manager" << std::endl;
// #endif
// }

// // bool MemoryManager::load_intermediate_to_gpu(LayeredTrie *intermediate)
// // {
// //   if (!intermediate)
// //   {
// //     std::cerr << "Cannot load null intermediate to GPU" << std::endl;
// //     return false;
// //   }

// //   // Calculate memory requirement for this intermediate using the specific
// //   // object
// //   size_t memory_needed = calculate_memory_requirement(intermediate);

// //   // Ensure enough memory is available
// //   if (!ensure_available_memory(memory_needed))
// //   {
// //     std::cerr << "Failed to ensure memory for intermediate " << intermediate->inter_id
// //               << std::endl;
// //     return false;
// //   }

// //   // Note: LayeredTrie uses a different memory management approach with layers
// //   // and device arrays. The actual GPU memory allocation is handled by the
// //   // LayeredTrie's own device array management system.

// //   bool success = true;

// //   // For LayeredTrie, we just need to ensure the device arrays are built
// //   if (!intermediate->arrays_allocated && !intermediate->layer_ids.empty())
// //   {
// //     // This would be handled by LayeredTrieManager through build_device_arrays
// //     // For now, mark as successful since the memory has been allocated
// //     success = true;
// //   }

// //   if (success)
// //   {
// // #ifndef NDEBUG
// //     std::cout << "Successfully registered intermediate " << intermediate->inter_id
// //               << " with memory manager (" << (memory_needed / (1024 * 1024)) << " MB)"
// //               << std::endl;
// // #endif
// //     record_access(intermediate->inter_id);
// //   }
// //   else
// //   {
// // #ifndef NDEBUG
// //     std::cerr << "Failed to register intermediate " << intermediate->inter_id
// //               << std::endl;
// // #endif
// //   }

// //   return success;
// // }

// bool MemoryManager::load_intermediate_to_gpu(int intermediate_id, Lattice *lat,
//                                              UnifiedTrieManager *utm)
// {
//   if (!lat || !utm)
//   {
//     std::cerr << "Cannot load intermediate: null manager pointers" << std::endl;
//     return false;
//   }

//   // Get the trie from UnifiedTrieManager
//   UnifiedTrie *trie = utm->getTrie(intermediate_id);
//   if (!trie)
//   {
//     std::cerr << "UnifiedTrie intermediate " << intermediate_id << " not found"
//               << std::endl;
//     return false;
//   }

//   // Determine if this is an edge (single bit) or multi-edge trie
//   ettype et = trie->et;
//   bool is_edge = (et.count() == 1);

//   if (is_edge)
//   {
//     // Edge candidates are always resident on GPU after construction
//     // They don't support eviction/loading, so just record the access
//     record_access(intermediate_id);

// #ifndef NDEBUG
//     std::cout << "Edge candidate " << intermediate_id
//               << " is always GPU-resident (no load needed)" << std::endl;
// #endif
//     return true;
//   }
//   else
//   {
//     // This is a multi-edge intermediate result
//     // Check if data is already on GPU
//     if (trie->data && trie->data->d_data)
//     {
//       // Already on GPU, just record access
//       record_access(intermediate_id);
// #ifndef NDEBUG
//       std::cout << "Intermediate " << intermediate_id
//                 << " already on GPU" << std::endl;
// #endif
//       return true;
//     }

//     // Need to load to GPU
//     if (!trie->data)
//     {
//       std::cerr << "Intermediate " << intermediate_id
//                 << " has no data block" << std::endl;
//       return false;
//     }

//     // Calculate memory requirement
//     size_t memory_needed = trie->data->size_in_byte;

//     // Ensure enough memory is available
//     if (!ensure_available_memory(memory_needed))
//     {
//       std::cerr << "Failed to ensure memory for intermediate "
//                 << intermediate_id << std::endl;
//       return false;
//     }

//     // Allocate GPU memory
//     if (trie->data->num_compressed_blocks == 0)
//     {
//       cuchk(cudaMalloc(&(trie->data->d_data), sizeof(uint32_t) * trie->data->num_rows));
//       cuchk(cudaMalloc((&trie->data->d_parents),
//                        sizeof(uint32_t) * trie->data->num_rows));
//     }
//     else
//     {
//       cuchk(cudaMalloc(&(trie->data->d_data),
//                        sizeof(uint32_t) * trie->data->num_compressed_blocks));
//       trie->data->d_parents = nullptr; // mask layer doesn't have parent pointers
//     }

//     // void *d_data_ptr;
//     // if (!allocate_gpu_memory(intermediate_id, memory_needed, &d_data_ptr))
//     // {
//     //   std::cerr << "Failed to allocate GPU memory for intermediate "
//     //             << intermediate_id << std::endl;
//     //   return false;
//     // }

//     // // Set the device pointer
//     // trie->data->d_space = d_data_ptr;
//     // trie->data->d_data = reinterpret_cast<uint32_t *>(trie->data->d_space);
//     // if (trie->data->num_compressed_blocks == 0)
//     // {
//     //   trie->data->d_parents = reinterpret_cast<uint32_t *>(
//     //       reinterpret_cast<char *>(trie->data->d_space) + trie->data->num_rows * sizeof(uint32_t));
//     // }
//     // else
//     // {
//     //   trie->data->d_parents = nullptr; // mask layer doesn't have parent pointers
//     // }

//     // TODO: If there's host data that needs to be transferred, do it here
//     // cudaMemcpy(trie->data->d_space, host_data, memory_needed, cudaMemcpyHostToDevice);

//     record_access(intermediate_id);

// #ifndef NDEBUG
//     std::cout << "Successfully loaded intermediate " << intermediate_id
//               << " to GPU (" << (memory_needed / (1024 * 1024)) << " MB)"
//               << std::endl;
// #endif
//     return true;
//   }
// }

// // bool MemoryManager::load_intermediate_to_gpu(int intermediate_id, Lattice *lat,
// //                                              LayeredTrieManager *ltm,
// //                                              EdgeCandidateManager *ecm)
// // {
// //   if (!lat || !ltm || !ecm)
// //   {
// //     std::cerr << "Cannot load intermediate: null manager pointers" << std::endl;
// //     return false;
// //   }

// //   // Determine if this is an edge (single bit set) or a layered trie (multiple bits)
// //   ettype et = lat->id2et[intermediate_id];
// //   bool is_edge = (et.count() == 1);

// //   if (is_edge)
// //   {
// //     // This is an EdgeCandidateTrie - edges are always on GPU after construction
// //     // EdgeCandidateTrie doesn't support eviction/loading, so just return true
// //     EdgeCandidateTrie *edge = ecm->all_edge_tries[et._Find_first()];
// //     if (!edge)
// //     {
// //       std::cerr << "Edge candidate " << intermediate_id << " not found" << std::endl;
// //       return false;
// //     }

// //     // Edge candidates are always resident on GPU
// //     // Just record the access
// //     record_access(intermediate_id);

// // #ifndef NDEBUG
// //     std::cout << "Edge candidate " << intermediate_id
// //               << " is always GPU-resident (no load needed)" << std::endl;
// // #endif
// //     return true;
// //   }
// //   else
// //   {
// //     // This is a LayeredTrie - use the existing function
// //     LayeredTrie *trie = ltm->all_tries[intermediate_id];
// //     if (!trie)
// //     {
// //       std::cerr << "LayeredTrie intermediate " << intermediate_id << " not found"
// //                 << std::endl;
// //       return false;
// //     }

// //     return load_intermediate_to_gpu(trie);
// //   }
// // }

// // bool MemoryManager::evict_intermediate_from_gpu(LayeredTrie *intermediate)
// // {
// //   if (!intermediate)
// //   {
// //     std::cerr << "Cannot evict null intermediate from GPU" << std::endl;
// //     return false;
// //   }

// //   bool success = true;

// //   // For LayeredTrie, eviction involves freeing the device arrays
// //   // This is handled by the LayeredTrie's free_device_arrays method
// //   if (intermediate->arrays_allocated)
// //   {
// //     intermediate->free_device_arrays();
// //   }

// //   // Update memory tracking
// //   if (success)
// //   {
// //     deallocate_gpu_memory(intermediate->inter_id);
// //     stats.total_evictions++;
// // #ifndef NDEBUG
// //     std::cout << "Successfully evicted intermediate " << intermediate->inter_id
// //               << " from GPU" << std::endl;
// // #endif
// //   }
// //   else
// //   {
// // #ifndef NDEBUG
// //     std::cerr << "Failed to completely evict intermediate " << intermediate->inter_id
// //               << " from GPU" << std::endl;
// // #endif
// //   }

// //   return success;
// // }

// // Additional helper methods
// size_t MemoryManager::calculate_memory_requirement(int intermediate_id) const
// {
//   // Look up the intermediate in the memory pool to get actual size
//   for (const auto &block : memory_pool)
//   {
//     if (block.intermediate_id == intermediate_id && !block.is_free)
//     {
//       return block.size;
//     }
//   }

//   // If not found in pool, we need to calculate based on intermediate structure
//   // This is a fallback calculation - in practice, we should have the
//   // intermediate object

//   // Default estimation based on typical intermediate table sizes
//   // This includes space for data array, index keys, and index values
//   size_t base_size = 1024 * 1024; // 1MB base

//   // Scale based on intermediate ID (higher IDs might be larger tables)
//   size_t scaled_size = base_size * (1 + intermediate_id / 10);

//   return std::min(scaled_size,
//                   static_cast<size_t>(512 * 1024 * 1024)); // Cap at 512MB
// }

// // size_t MemoryManager::calculate_memory_requirement(
// //     const LayeredTrie *intermediate) const
// // {
// //   if (!intermediate)
// //   {
// //     return 0;
// //   }

// //   size_t total_size = 0;

// //   // For LayeredTrie, the memory usage is tracked in total_memory_usage
// //   if (intermediate->total_memory_usage > 0)
// //   {
// //     total_size = intermediate->total_memory_usage;
// //   }
// //   else
// //   {
// //     // Fallback calculation based on dimensions
// //     if (intermediate->num_layers > 0 && intermediate->num_cols > 0)
// //     {
// //       // Estimate based on the layered structure
// //       size_t estimated_rows = 1000; // Default estimation
// //       size_t data_size = estimated_rows * static_cast<size_t>(intermediate->num_cols) * sizeof(vtype);
// //       total_size += data_size;
// //     }

// //     // Add memory for device arrays (pointers to each layer)
// //     if (!intermediate->layer_ids.empty())
// //     {
// //       size_t arrays_size = intermediate->layer_ids.size() *
// //                            (sizeof(vtype *) + sizeof(uint32_t *));
// //       total_size += arrays_size;
// //     }
// //   }

// //   // Add overhead for memory alignment and metadata (typically 5-10%)
// //   total_size = static_cast<size_t>(total_size * 1.1);

// //   // Ensure minimum allocation size
// //   total_size = std::max(total_size, static_cast<size_t>(4096)); // 4KB minimum

// //   return total_size;
// // }

// bool MemoryManager::is_intermediate_on_gpu(int intermediate_id) const
// {
//   // Check memory pool allocations
//   for (const auto &block : memory_pool)
//   {
//     if (block.intermediate_id == intermediate_id && !block.is_free)
//     {
//       return true;
//     }
//   }

//   // Check direct CUDA allocations (fallback allocations)
//   for (const auto &entry : direct_allocations)
//   {
//     if (entry.second == intermediate_id)
//     {
//       return true;
//     }
//   }

//   return false;
// }

// // bool MemoryManager::is_intermediate_on_gpu(
// //     const LayeredTrie *intermediate) const
// // {
// //   if (!intermediate)
// //     return false;

// //   // For LayeredTrie, check if device arrays are allocated
// //   bool has_device_arrays = intermediate->arrays_allocated;

// //   // Also verify that the MemoryManager is tracking this intermediate's
// //   // allocations
// //   bool tracked_in_pool = is_intermediate_on_gpu(intermediate->inter_id);

// //   // Both conditions must be true for the intermediate to be considered "on GPU"
// //   return has_device_arrays && tracked_in_pool;
// // }

// // bool MemoryManager::validate_intermediate_state(
// //     const LayeredTrie *intermediate) const
// // {
// //   if (!intermediate)
// //     return false;

// //   bool pool_tracked = is_intermediate_on_gpu(intermediate->inter_id);
// //   bool has_device_arrays = intermediate->arrays_allocated;

// //   // Check for inconsistent states
// //   if (pool_tracked && !has_device_arrays)
// //   {
// //     std::cerr << "Warning: Intermediate " << intermediate->inter_id
// //               << " is tracked in memory pool but has no device arrays allocated"
// //               << std::endl;
// //     return false;
// //   }

// //   if (!pool_tracked && has_device_arrays)
// //   {
// //     std::cerr << "Warning: Intermediate " << intermediate->inter_id
// //               << " has device arrays but is not tracked in memory pool"
// //               << std::endl;
// //     return false;
// //   }

// //   return true;
// // }

// std::vector<int> MemoryManager::get_gpu_resident_intermediates() const
// {
//   std::vector<int> result;
//   for (const auto &block : memory_pool)
//   {
//     if (!block.is_free && block.intermediate_id != -1)
//     {
//       result.push_back(block.intermediate_id);
//     }
//   }
//   return result;
// }

// void MemoryManager::validate_memory_consistency()
// {
//   size_t calculated_usage = 0;
//   for (const auto &block : memory_pool)
//   {
//     if (!block.is_free)
//     {
//       calculated_usage += block.size;
//     }
//   }

//   if (calculated_usage != current_gpu_memory)
//   {
//     std::cerr << "Memory consistency error: calculated=" << calculated_usage
//               << ", tracked=" << current_gpu_memory << std::endl;
//   }
// }

// double MemoryManager::get_memory_utilization() const
// {
//   if (gpu_memory_limit == 0)
//     return 0.0;
//   return static_cast<double>(current_gpu_memory) / gpu_memory_limit;
// }

// size_t MemoryManager::get_available_memory() const
// {
//   return (gpu_memory_limit > current_gpu_memory)
//              ? (gpu_memory_limit - current_gpu_memory)
//              : 0;
// }

// size_t MemoryManager::get_largest_free_block() const
// {
//   size_t largest = 0;
//   for (const auto &block : memory_pool)
//   {
//     if (block.is_free && block.size > largest)
//     {
//       largest = block.size;
//     }
//   }
//   return largest;
// }

// // Missing method implementations
// void MemoryManager::enable_statistics(bool enable)
// {
//   enable_stats = enable;
// #ifndef NDEBUG
//   if (enable)
//   {
//     std::cout << "Memory statistics enabled" << std::endl;
//   }
//   else
//   {
//     std::cout << "Memory statistics disabled" << std::endl;
//   }
// #endif
// }

// void MemoryManager::enable_access_prediction(bool enable)
// {
//   enable_prediction = enable;
// #ifndef NDEBUG
//   if (enable)
//   {
//     std::cout << "Access prediction enabled" << std::endl;
//   }
//   else
//   {
//     std::cout << "Access prediction disabled" << std::endl;
//   }
// #endif
// }

// void MemoryManager::set_memory_threshold(double threshold_ratio)
// {
//   memory_threshold = static_cast<size_t>(gpu_memory_limit * threshold_ratio);
// #ifndef NDEBUG
//   std::cout << "Memory threshold set to " << (memory_threshold / (1024 * 1024))
//             << " MB (" << (threshold_ratio * 100) << "%)" << std::endl;
// #endif
// }

// void MemoryManager::set_transfer_strategy(TransferStrategy strategy)
// {
//   transfer_strategy = strategy;
// #ifndef NDEBUG
//   std::cout << "Transfer strategy changed to: ";
//   switch (strategy)
//   {
//   case TransferStrategy::IMMEDIATE:
//     std::cout << "IMMEDIATE";
//     break;
//   case TransferStrategy::BATCHED:
//     std::cout << "BATCHED";
//     break;
//   case TransferStrategy::PREFETCH:
//     std::cout << "PREFETCH";
//     break;
//   case TransferStrategy::ADAPTIVE:
//     std::cout << "ADAPTIVE";
//     break;
//   }
//   std::cout << std::endl;
// #endif
// }

// void MemoryManager::configure_hybrid_weights(double lru_weight,
//                                              double contribution_weight)
// {
//   hybrid_lru_weight = lru_weight;
//   hybrid_contribution_weight = contribution_weight;
// #ifndef NDEBUG
//   std::cout << "Hybrid weights configured: LRU=" << lru_weight
//             << ", Contribution=" << contribution_weight << std::endl;
// #endif
// }

// void MemoryManager::set_batch_size_threshold(size_t threshold)
// {
//   batch_size_threshold = threshold;
// #ifndef NDEBUG
//   std::cout << "Batch size threshold set to " << (threshold / (1024 * 1024))
//             << " MB" << std::endl;
// #endif
// }

// void MemoryManager::set_max_prefetch_items(int max_items)
// {
//   max_prefetch_items = max_items;
// #ifndef NDEBUG
//   std::cout << "Maximum prefetch items set to " << max_items << std::endl;
// #endif
// }

// void MemoryManager::schedule_prefetch(int intermediate_id)
// {
//   if (prefetch_queue.size() < static_cast<size_t>(max_prefetch_items))
//   {
//     prefetch_queue.push(intermediate_id);
// #ifndef NDEBUG
//     std::cout << "Scheduled prefetch for intermediate " << intermediate_id
//               << std::endl;
// #endif
//   }
// }

// void MemoryManager::execute_prefetch_batch()
// {
// #ifndef NDEBUG
//   std::cout << "Executing prefetch batch with " << prefetch_queue.size()
//             << " items" << std::endl;
// #endif

//   std::vector<int> failed_prefetches;

//   while (!prefetch_queue.empty())
//   {
//     int id = prefetch_queue.front();
//     prefetch_queue.pop();

//     // Find the intermediate in our tracking structures
//     bool found = false;
//     for (const auto &[access_id, pattern] : access_patterns)
//     {
//       if (access_id == id)
//       {
//         found = true;
//         break;
//       }
//     }

//     if (found)
//     {
//       // Check if we have enough memory for prefetch
//       size_t required_memory = calculate_memory_requirement(id);

//       if (get_available_memory() >= required_memory)
//       {
//         // Simulate prefetch by allocating memory and recording access
//         void *prefetch_ptr;
//         if (allocate_gpu_memory(id, required_memory, &prefetch_ptr))
//         {
// #ifndef NDEBUG
//           std::cout << "Successfully prefetched intermediate " << id
//                     << std::endl;
// #endif
//           record_access(id);
//         }
//         else
//         {
// #ifndef NDEBUG
//           std::cout << "Failed to prefetch intermediate " << id
//                     << " (allocation failed)" << std::endl;
// #endif
//           failed_prefetches.push_back(id);
//         }
//       }
//       else
//       {
// #ifndef NDEBUG
//         std::cout << "Skipping prefetch for intermediate " << id
//                   << " (insufficient memory)" << std::endl;
// #endif
//         failed_prefetches.push_back(id);
//       }
//     }
//     else
//     {
// #ifndef NDEBUG
//       std::cout << "Skipping prefetch for unknown intermediate " << id
//                 << std::endl;
// #endif
//     }
//   }

//   // Re-queue failed prefetches for later attempt
//   for (int failed_id : failed_prefetches)
//   {
//     if (prefetch_queue.size() < static_cast<size_t>(max_prefetch_items))
//     {
//       prefetch_queue.push(failed_id);
//     }
//   }
// }

// void MemoryManager::add_to_transfer_batch(int intermediate_id)
// {
//   transfer_batch.push_back(intermediate_id);

//   // Execute batch if it reaches threshold
//   if (transfer_batch.size() * 1024 * 1024 >= batch_size_threshold)
//   {
//     execute_transfer_batch();
//   }
// }

// void MemoryManager::execute_transfer_batch()
// {
//   if (transfer_batch.empty())
//     return;

// #ifndef NDEBUG
//   std::cout << "Executing transfer batch with " << transfer_batch.size()
//             << " intermediates" << std::endl;
// #endif

//   // Sort batch for optimal transfer order (largest first to minimize
//   // fragmentation)
//   optimize_batch_order(transfer_batch);

//   std::vector<int> successful_transfers;
//   std::vector<int> failed_transfers;

//   for (int id : transfer_batch)
//   {
//     size_t required_memory = calculate_memory_requirement(id);

//     // Check if we can accommodate this transfer
//     if (ensure_available_memory(required_memory))
//     {
//       void *transfer_ptr;
//       if (allocate_gpu_memory(id, required_memory, &transfer_ptr))
//       {
// #ifndef NDEBUG
//         std::cout << "Successfully transferred intermediate " << id << " ("
//                   << (required_memory / (1024 * 1024)) << " MB)" << std::endl;
// #endif
//         successful_transfers.push_back(id);
//         record_access(id);
//       }
//       else
//       {
// #ifndef NDEBUG
//         std::cout << "Failed to allocate memory for intermediate " << id
//                   << std::endl;
// #endif
//         failed_transfers.push_back(id);
//       }
//     }
//     else
//     {
// #ifndef NDEBUG
//       std::cout << "Insufficient memory for intermediate " << id
//                 << " (required: " << (required_memory / (1024 * 1024)) << " MB)"
//                 << std::endl;
// #endif
//       failed_transfers.push_back(id);
//     }
//   }

//   // Update statistics
//   if (enable_stats)
//   {
//     update_statistics();
//   }
// #ifndef NDEBUG
//   std::cout << "Transfer batch completed: " << successful_transfers.size()
//             << " successful, " << failed_transfers.size() << " failed"
//             << std::endl;
// #endif
//   // Clear the batch
//   transfer_batch.clear();

// // Optionally retry failed transfers individually later
// #ifndef NDEBUG
//   if (!failed_transfers.empty())
//   {
//     std::cout << "Note: " << failed_transfers.size()
//               << " transfers failed and may be retried later" << std::endl;
//   }
// #endif
// }

// void MemoryManager::add_dependency(int from_id, int to_id)
// {
//   dependency_graph[from_id].push_back(to_id);
// #ifndef NDEBUG
//   std::cout << "Added dependency: " << from_id << " -> " << to_id << std::endl;
// #endif
// }

// void MemoryManager::remove_dependency(int from_id, int to_id)
// {
//   auto &deps = dependency_graph[from_id];
//   deps.erase(std::remove(deps.begin(), deps.end(), to_id), deps.end());
// #ifndef NDEBUG
//   std::cout << "Removed dependency: " << from_id << " -> " << to_id
//             << std::endl;
// #endif
// }

// void MemoryManager::update_contribution_scores()
// {
//   // Simple implementation - count number of dependents
//   for (const auto &[from_id, to_ids] : dependency_graph)
//   {
//     contribution_scores[from_id] = to_ids.size();
//   }
// #ifndef NDEBUG
//   std::cout << "Updated contribution scores for " << contribution_scores.size()
//             << " intermediates" << std::endl;
// #endif
// }

// void MemoryManager::set_contribution_score(int intermediate_id, int score)
// {
//   contribution_scores[intermediate_id] = score;
// }

// std::vector<int> MemoryManager::get_dependents(int intermediate_id)
// {
//   auto it = dependency_graph.find(intermediate_id);
//   if (it != dependency_graph.end())
//   {
//     return it->second;
//   }
//   return {};
// }

// void MemoryManager::reset_statistics()
// {
//   stats = MemoryStats();
// #ifndef NDEBUG
//   std::cout << "Reset memory statistics" << std::endl;
// #endif
// }

// // Additional helper methods for completeness
// void MemoryManager::analyze_access_patterns()
// {
//   cleanup_expired_patterns();

//   for (auto &[id, pattern] : access_patterns)
//   {
//     pattern.update_prediction();
//   }
// #ifndef NDEBUG
//   std::cout << "Analyzed access patterns for " << access_patterns.size()
//             << " intermediates" << std::endl;
// #endif
// }

// std::vector<int> MemoryManager::predict_next_accesses(int lookahead_steps)
// {
//   std::vector<int> predictions;

//   for (const auto &[id, pattern] : access_patterns)
//   {
//     if (pattern.should_prefetch())
//     {
//       predictions.push_back(id);
//     }
//   }

//   return predictions;
// }

// bool MemoryManager::should_prefetch_intermediate(int intermediate_id) const
// {
//   auto it = access_patterns.find(intermediate_id);
//   if (it != access_patterns.end())
//   {
//     return it->second.should_prefetch();
//   }
//   return false;
// }

// void MemoryManager::update_access_prediction_model()
// {
//   analyze_access_patterns();
// }

// void MemoryManager::cleanup_expired_patterns()
// {
//   auto now = std::chrono::high_resolution_clock::now();

//   for (auto it = access_patterns.begin(); it != access_patterns.end();)
//   {
//     if (it->second.access_times.empty())
//     {
//       it = access_patterns.erase(it);
//     }
//     else
//     {
//       auto last_access = it->second.access_times.back();
//       auto time_since_access =
//           std::chrono::duration_cast<std::chrono::hours>(now - last_access)
//               .count();

//       // Remove patterns older than 24 hours
//       if (time_since_access > 24)
//       {
//         it = access_patterns.erase(it);
//       }
//       else
//       {
//         ++it;
//       }
//     }
//   }
// }

// bool MemoryManager::deallocate_to_pool(void *ptr)
// {
//   if (!ptr)
//     return false;

//   for (auto &block : memory_pool)
//   {
//     if (block.ptr == ptr)
//     {
//       block.is_free = true;
//       block.intermediate_id = -1;

//       // Update allocation map
//       allocation_map.erase(ptr);

//       // Update memory tracking
//       current_gpu_memory -= block.size;
//       stats.used_gpu_memory = current_gpu_memory;

//       // Merge adjacent free blocks
//       merge_free_blocks();

//       if (enable_stats)
//       {
//         update_statistics();
//       }

//       return true;
//     }
//   }

//   return false;
// }

// void MemoryManager::defragment_memory_pool()
// {
//   size_t initial_fragmentation = 0;
//   size_t num_free_blocks = 0;

//   // Calculate initial fragmentation
//   for (const auto &block : memory_pool)
//   {
//     if (block.is_free)
//     {
//       num_free_blocks++;
//       initial_fragmentation += block.size;
//     }
//   }
// #ifndef NDEBUG
//   std::cout << "Starting defragmentation: " << num_free_blocks
//             << " free blocks totaling "
//             << (initial_fragmentation / (1024 * 1024)) << " MB" << std::endl;
// #endif
//   // First, merge adjacent free blocks
//   size_t blocks_before_merge = memory_pool.size();
//   merge_free_blocks();
//   size_t blocks_after_merge = memory_pool.size();

//   // Count free blocks after merge
//   size_t merged_free_blocks = 0;
//   size_t largest_free_block = 0;

//   for (const auto &block : memory_pool)
//   {
//     if (block.is_free)
//     {
//       merged_free_blocks++;
//       if (block.size > largest_free_block)
//       {
//         largest_free_block = block.size;
//       }
//     }
//   }

// #ifndef NDEBUG
//   std::cout << "Defragmentation completed:" << std::endl;
//   std::cout << "  Blocks reduced from " << blocks_before_merge << " to "
//             << blocks_after_merge << std::endl;
//   std::cout << "  Free blocks reduced from " << num_free_blocks << " to "
//             << merged_free_blocks << std::endl;
//   std::cout << "  Largest free block: " << (largest_free_block / (1024 * 1024))
//             << " MB" << std::endl;
// #endif

//   // Update statistics
//   if (enable_stats)
//   {
//     update_statistics();
//   }
// }

// void MemoryManager::sort_by_contribution_score(std::vector<int> &candidates)
// {
//   std::sort(candidates.begin(), candidates.end(), [this](int a, int b)
//             { return contribution_scores[a] > contribution_scores[b]; });
// }

// void MemoryManager::sort_by_access_time(std::vector<int> &candidates)
// {
//   std::sort(candidates.begin(), candidates.end(), [this](int a, int b)
//             {
//               return last_access_times[a] > last_access_times[b]; // More recent first
//             });
// }

// void MemoryManager::sort_by_memory_size(std::vector<int> &candidates)
// {
//   std::sort(candidates.begin(), candidates.end(), [this](int a, int b)
//             {
//               return get_intermediate_memory_size(a) >
//                      get_intermediate_memory_size(b); // Larger first
//             });
// }

// bool MemoryManager::can_allocate_contiguous(size_t size) const
// {
//   // Check if we can allocate a contiguous block of the given size
//   for (const auto &block : memory_pool)
//   {
//     if (block.is_free && block.size >= size)
//     {
//       return true;
//     }
//   }

//   // Check if CUDA can allocate directly
//   return check_cuda_memory_available(size);
// }

// bool MemoryManager::check_cuda_memory_available(size_t size) const
// {
//   size_t free_memory, total_memory;
//   cudaMemGetInfo(&free_memory, &total_memory);
//   return free_memory >= size;
// }

// void MemoryManager::sync_gpu_operations() const { cudaDeviceSynchronize(); }

// bool MemoryManager::batch_load_intermediates(
//     const std::vector<int> &intermediate_ids)
// {
//   std::vector<int> successful_loads;
//   std::vector<int> failed_loads;

// #ifndef NDEBUG
//   std::cout << "Starting batch load of " << intermediate_ids.size()
//             << " intermediates" << std::endl;
// #endif

//   for (int id : intermediate_ids)
//   {
//     size_t required_memory = calculate_memory_requirement(id);

//     if (ensure_available_memory(required_memory))
//     {
//       void *load_ptr;
//       if (allocate_gpu_memory(id, required_memory, &load_ptr))
//       {
//         successful_loads.push_back(id);
//         record_access(id);
// #ifndef NDEBUG
//         std::cout << "Successfully loaded intermediate " << id << std::endl;
// #endif
//       }
//       else
//       {
//         failed_loads.push_back(id);
//         std::cerr << "Failed to allocate memory for intermediate " << id
//                   << std::endl;
//       }
//     }
//     else
//     {
//       failed_loads.push_back(id);
//       std::cerr << "Insufficient memory for intermediate " << id << std::endl;
//     }
//   }

//   // Update statistics
//   if (enable_stats)
//   {
//     update_statistics();
//   }
// #ifndef NDEBUG
//   std::cout << "Batch load completed: " << successful_loads.size()
//             << " successful, " << failed_loads.size() << " failed" << std::endl;
// #endif
//   // Return true only if all loads were successful
//   return failed_loads.empty();
// }

// bool MemoryManager::batch_evict_intermediates(
//     const std::vector<int> &intermediate_ids)
// {
//   bool success = true;

//   // Evict each intermediate
//   for (int id : intermediate_ids)
//   {
//     // Find the intermediate in our tracking and evict it
//     bool evicted = false;
//     for (auto &block : memory_pool)
//     {
//       if (block.intermediate_id == id && !block.is_free)
//       {
//         block.is_free = true;
//         block.intermediate_id = -1;
//         current_gpu_memory -= block.size;
//         evicted = true;
//         break;
//       }
//     }

//     if (!evicted)
//     {
//       success = false;
//       std::cerr << "Failed to evict intermediate " << id << std::endl;
//     }
//   }

//   stats.used_gpu_memory = current_gpu_memory;

//   if (enable_stats)
//   {
//     update_statistics();
//   }

//   return success;
// }

// void MemoryManager::optimize_batch_order(std::vector<int> &intermediate_ids)
// {
//   // Simple optimization: sort by memory size (largest first) to improve memory
//   // utilization
//   std::sort(intermediate_ids.begin(), intermediate_ids.end(),
//             [this](int a, int b)
//             {
//               return get_intermediate_memory_size(a) >
//                      get_intermediate_memory_size(b);
//             });
// }

// void MemoryManager::emergency_cleanup()
// {
// #ifndef NDEBUG
//   std::cout << "Performing emergency memory cleanup..." << std::endl;
// #endif
//   // Force eviction of all non-pinned intermediates
//   std::vector<int> candidates;
//   for (const auto &block : memory_pool)
//   {
//     if (!block.is_free && block.intermediate_id != -1 &&
//         pinned_intermediates.find(block.intermediate_id) ==
//             pinned_intermediates.end())
//     {
//       candidates.push_back(block.intermediate_id);
//     }
//   }

//   execute_eviction(candidates);

//   // Try to free additional memory by defragmenting
//   defragment_memory_pool();

// #ifndef NDEBUG
//   std::cout << "Emergency cleanup completed. Available memory: "
//             << get_available_memory() << " bytes" << std::endl;
// #endif
// }

// bool MemoryManager::force_evict_oldest()
// {
//   // Find the oldest non-pinned intermediate
//   std::chrono::high_resolution_clock::time_point oldest_time =
//       std::chrono::high_resolution_clock::now();
//   int oldest_id = -1;

//   for (const auto &entry : last_access_times)
//   {
//     int id = entry.first;
//     auto access_time = entry.second;

//     // Skip pinned intermediates
//     if (pinned_intermediates.find(id) != pinned_intermediates.end())
//       continue;

//     if (access_time < oldest_time)
//     {
//       oldest_time = access_time;
//       oldest_id = id;
//     }
//   }

//   if (oldest_id != -1)
//   {
//     std::vector<int> candidates = {oldest_id};
//     execute_eviction(candidates);
//     return true;
//   }

//   return false;
// }

// void MemoryManager::compact_memory() { defragment_memory_pool(); }

// // Additional missing method implementations
// void MemoryManager::auto_tune_parameters()
// {
//   // Analyze current performance and adjust parameters
//   double utilization = get_memory_utilization();

//   if (utilization > 0.9)
//   {
//     // Memory is highly utilized, increase aggressiveness
//     memory_threshold = static_cast<size_t>(gpu_memory_limit * 0.7);
//     max_prefetch_items = std::max(1, max_prefetch_items - 1);
//   }
//   else if (utilization < 0.5)
//   {
//     // Memory is underutilized, we can be more permissive
//     memory_threshold = static_cast<size_t>(gpu_memory_limit * 0.85);
//     max_prefetch_items = std::min(10, max_prefetch_items + 1);
//   }

// #ifndef NDEBUG
//   std::cout << "Auto-tuned parameters: threshold="
//             << (memory_threshold / (1024 * 1024))
//             << "MB, max_prefetch=" << max_prefetch_items << std::endl;
// #endif
// }

// void MemoryManager::optimize_for_workload_pattern()
// {
//   // Analyze access patterns and optimize accordingly
//   analyze_access_patterns();

//   // Count pattern types
//   int periodic_patterns = 0;
//   int random_patterns = 0;

//   for (const auto &[id, pattern] : access_patterns)
//   {
//     if (pattern.is_periodic)
//       periodic_patterns++;
//     else
//       random_patterns++;
//   }

//   // Adjust strategy based on workload
//   if (periodic_patterns > random_patterns)
//   {
//     // More predictable workload - enable prefetching
//     enable_access_prediction(true);
//     set_transfer_strategy(TransferStrategy::PREFETCH);
//   }
//   else
//   {
//     // More random workload - use immediate transfers
//     set_transfer_strategy(TransferStrategy::IMMEDIATE);
//   }

// #ifndef NDEBUG
//   std::cout << "Optimized for workload: periodic=" << periodic_patterns
//             << ", random=" << random_patterns << std::endl;
// #endif
// }

// void MemoryManager::adjust_eviction_policy_dynamically()
// {
//   // Analyze current performance and adjust eviction policy
//   double hit_ratio = stats.hit_ratio;
//   int total_accesses = stats.cache_hits + stats.cache_misses;

//   if (total_accesses < 100)
//     return; // Not enough data yet

//   if (hit_ratio < 0.6)
//   {
//     // Poor hit ratio, try different policy
//     if (current_policy == EvictionPolicy::LRU)
//     {
//       set_eviction_policy(EvictionPolicy::CONTRIBUTION);
//     }
//     else if (current_policy == EvictionPolicy::CONTRIBUTION)
//     {
//       set_eviction_policy(EvictionPolicy::HYBRID);
//     }
//     else
//     {
//       set_eviction_policy(EvictionPolicy::LRU);
//     }

// #ifndef NDEBUG
//     std::cout << "Adjusted eviction policy due to poor hit ratio: " << hit_ratio
//               << std::endl;
// #endif
//   }
// }

// std::unique_ptr<MemoryManager> MemoryManager::instance = nullptr;
// std::mutex MemoryManager::instance_mutex;

// MemoryManager &MemoryManager::getInstance()
// {
//   std::lock_guard<std::mutex> lock(instance_mutex);
//   if (!instance)
//   {
//     instance.reset(new MemoryManager());
//   }
//   return *instance;
// }

// void MemoryManager::initialize(size_t gpu_memory_limit)
// {
//   auto &manager = getInstance();
//   manager.set_gpu_memory_limit(gpu_memory_limit);
// }

// void MemoryManager::shutdown()
// {
//   std::lock_guard<std::mutex> lock(instance_mutex);
//   instance.reset();
// }

// // ============================================================================
// // NEW: Policy Object Management
// // ============================================================================

// void MemoryManager::set_policy_object(VirtualEvictPolicy* policy)
// {
//   if (current_policy_impl)
//   {
//     delete current_policy_impl;
//   }
//   current_policy_impl = policy;

//   // Set lattice for contribution-based policies
//   if (policy)
//   {
//     Lattice* lat = &Lattice::getInstance();
//     policy->setLattice(lat);
//   }
// }

// VirtualEvictPolicy* MemoryManager::get_policy_object() const
// {
//   return current_policy_impl;
// }

// // ============================================================================
// // NEW: Mapping Helpers
// // ============================================================================

// uint32_t MemoryManager::get_block_id(int intermediate_id)
// {
//   auto it = intermediate_to_blockid.find(intermediate_id);
//   if (it != intermediate_to_blockid.end())
//   {
//     return it->second;
//   }

//   // Assign new block_id
//   uint32_t block_id = next_block_id++;
//   intermediate_to_blockid[intermediate_id] = block_id;
//   blockid_to_intermediate[block_id] = intermediate_id;
//   return block_id;
// }

// int MemoryManager::get_intermediate_id(uint32_t block_id) const
// {
//   auto it = blockid_to_intermediate.find(block_id);
//   if (it != blockid_to_intermediate.end())
//   {
//     return it->second;
//   }
//   return -1;  // Invalid
// }
