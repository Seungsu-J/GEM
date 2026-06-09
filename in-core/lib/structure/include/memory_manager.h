#ifndef STRUCTURE_MEMORY_MANAGER_H
#define STRUCTURE_MEMORY_MANAGER_H

#include "defs.h"
#include "memory_types.h"

#include <algorithm>
#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// Forward declaration
// class Intermediate;

// Memory pool for efficient allocation/deallocation

class UnifiedTrieManager;
class UnifiedTrie;
class Lattice;

struct MemoryBlock
{
  void *ptr;
  size_t size;
  bool is_free;
  int intermediate_id; // -1 if not assigned
  std::chrono::high_resolution_clock::time_point allocation_time;

  MemoryBlock(void *p, size_t s)
      : ptr(p), size(s), is_free(true), intermediate_id(-1)
  {
    allocation_time = std::chrono::high_resolution_clock::now();
  }
};

// Statistics for monitoring memory usage
struct MemoryStats
{
  size_t total_gpu_memory;
  size_t used_gpu_memory;
  size_t available_gpu_memory;
  int active_intermediates;
  int total_allocations;
  int total_evictions;
  int cache_hits;
  int cache_misses;
  double hit_ratio;
  size_t peak_memory_usage;
  std::chrono::high_resolution_clock::time_point last_update;

  MemoryStats()
      : total_gpu_memory(0), used_gpu_memory(0), available_gpu_memory(0),
        active_intermediates(0), total_allocations(0), total_evictions(0),
        cache_hits(0), cache_misses(0), hit_ratio(0.0), peak_memory_usage(0)
  {
    last_update = std::chrono::high_resolution_clock::now();
  }
};

// Access pattern tracking for predictive management
struct AccessPattern
{
  int intermediate_id;
  std::vector<std::chrono::high_resolution_clock::time_point> access_times;
  int access_count;
  double average_interval;
  bool is_periodic;
  std::chrono::high_resolution_clock::time_point predicted_next_access;

  AccessPattern()
      : intermediate_id(-1), access_count(0), average_interval(0.0),
        is_periodic(false) {}
  AccessPattern(int id)
      : intermediate_id(id), access_count(0), average_interval(0.0),
        is_periodic(false) {}

  void record_access();
  void update_prediction();
  bool should_prefetch() const;
};

class MemoryManager
{
private:
  // Core memory management
  size_t gpu_memory_limit;
  size_t current_gpu_memory;
  size_t memory_threshold; // Trigger eviction when exceeded

  // Memory pools
  std::vector<MemoryBlock> memory_pool;
  std::unordered_map<void *, size_t> allocation_map;
  std::unordered_map<void *, int>
      direct_allocations; // Track intermediate_id for direct CUDA allocations

  // Eviction policies
  EvictionPolicy current_policy;
  TransferStrategy transfer_strategy;

  // LRU tracking
  std::unordered_map<int, std::chrono::high_resolution_clock::time_point>
      last_access_times;
  std::unordered_map<int, int> access_frequencies;

  // Priority queues for different eviction strategies
  std::priority_queue<
      std::pair<std::chrono::high_resolution_clock::time_point, int>,
      std::vector<
          std::pair<std::chrono::high_resolution_clock::time_point, int>>,
      std::greater<>>
      lru_queue;

  // Contribution tracking
  std::unordered_map<int, int> contribution_scores;
  std::unordered_map<int, std::vector<int>> dependency_graph;

  // Transfer optimization
  std::queue<int> prefetch_queue;
  std::unordered_set<int> pinned_intermediates;
  std::vector<int> transfer_batch;
  size_t batch_size_threshold;

  // Access pattern analysis
  std::unordered_map<int, AccessPattern> access_patterns;
  bool enable_prediction;

  // Statistics and monitoring
  MemoryStats stats;
  bool enable_stats;

  // Configuration parameters
  double hybrid_lru_weight;
  double hybrid_contribution_weight;
  int max_prefetch_items;
  std::chrono::milliseconds prefetch_timeout;

public:
  static MemoryManager &getInstance();
  static void initialize(size_t gpu_memory_limit);
  static void shutdown();

  // Core memory management interface
  ~MemoryManager();

  // Core memory management interface
  bool allocate_gpu_memory(int intermediate_id, size_t size, void **ptr);
  bool deallocate_gpu_memory(int intermediate_id);
  bool ensure_available_memory(size_t required_size);

  // Configuration methods
  void set_gpu_memory_limit(size_t limit);
  void set_memory_threshold(double threshold_ratio = 0.8);
  void set_eviction_policy(EvictionPolicy policy);
  void set_eviction_policy(std::string policy);
  void set_transfer_strategy(TransferStrategy strategy);
  void enable_statistics(bool enable = true);
  void enable_access_prediction(bool enable = true);

  // Policy configuration
  void configure_hybrid_weights(double lru_weight, double contribution_weight);
  void set_batch_size_threshold(size_t threshold);
  void set_max_prefetch_items(int max_items);

  // Eviction strategies
  std::vector<int> get_lru_eviction_candidates(size_t required_memory);
  std::vector<int> get_lfu_eviction_candidates(size_t required_memory);
  std::vector<int> get_contribution_eviction_candidates(size_t required_memory);
  std::vector<int> get_size_based_eviction_candidates(size_t required_memory);
  std::vector<int> get_hybrid_eviction_candidates(size_t required_memory);

  // Transfer optimization
  void record_access(int intermediate_id);
  void pin_intermediate(int intermediate_id);
  void unpin_intermediate(int intermediate_id);
  void schedule_prefetch(int intermediate_id);
  void execute_prefetch_batch();
  void add_to_transfer_batch(int intermediate_id);
  void execute_transfer_batch();

  // Dependency management
  void add_dependency(int from_id, int to_id);
  void remove_dependency(int from_id, int to_id);
  void update_contribution_scores();
  std::vector<int> get_dependents(int intermediate_id);
  // Manually seed or override a contribution score for an intermediate
  void set_contribution_score(int intermediate_id, int score);

  // Memory pool management
  void initialize_memory_pool(size_t pool_size);
  void *allocate_from_pool(size_t size, int intermediate_id);
  bool deallocate_to_pool(void *ptr);
  void defragment_memory_pool();
  void cleanup_memory_pool();

  // Statistics and monitoring
  MemoryStats get_statistics() const;
  void print_memory_status() const;
  void print_detailed_stats() const;
  void reset_statistics();
  double get_memory_utilization() const;
  size_t get_available_memory() const;
  size_t get_largest_free_block() const;

  // Access pattern analysis
  void analyze_access_patterns();
  std::vector<int> predict_next_accesses(int lookahead_steps = 3);
  bool should_prefetch_intermediate(int intermediate_id) const;
  void update_access_prediction_model();

  // Adaptive management
  void auto_tune_parameters();
  void optimize_for_workload_pattern();
  void adjust_eviction_policy_dynamically();

  // Integration with Intermediate class
  // bool load_intermediate_to_gpu(LayeredTrie *intermediate);
  // bool load_intermediate_to_gpu(int intermediate_id, class Lattice *lat,
  //                               class LayeredTrieManager *ltm,
  //                               class EdgeCandidateManager *ecm);
  bool load_intermediate_to_gpu(int intermediate_id, Lattice *lat,
                                UnifiedTrieManager *utm);
  // bool evict_intermediate_from_gpu(LayeredTrie *intermediate);
  // void register_intermediate(LayeredTrie *intermediate);
  // void register_intermediate_trie(class LayeredTrie *trie); // LayeredTrie registration
  void unregister_intermediate(int intermediate_id);

  // Batch operations
  bool batch_load_intermediates(const std::vector<int> &intermediate_ids);
  bool batch_evict_intermediates(const std::vector<int> &intermediate_ids);
  void optimize_batch_order(std::vector<int> &intermediate_ids);

  // Emergency management
  void emergency_cleanup();
  bool force_evict_oldest();
  void compact_memory();

  // Utility methods
  size_t calculate_memory_requirement(int intermediate_id) const;
  // size_t calculate_memory_requirement(const LayeredTrie *intermediate) const;
  bool is_intermediate_on_gpu(int intermediate_id) const;
  // bool is_intermediate_on_gpu(const LayeredTrie *intermediate) const;
  // bool validate_intermediate_state(const LayeredTrie *intermediate) const;
  std::vector<int> get_gpu_resident_intermediates() const;
  void validate_memory_consistency();

private:
  MemoryManager();
  MemoryManager(const MemoryManager &) = delete;
  MemoryManager(MemoryManager &&) = delete;
  MemoryManager &operator=(const MemoryManager &) = delete;
  MemoryManager &operator=(MemoryManager &&) = delete;

  // Internal helper methods
  void update_statistics();
  void execute_eviction(const std::vector<int> &candidates);
  size_t get_intermediate_memory_size(int intermediate_id) const;
  void cleanup_expired_patterns();
  double calculate_hybrid_score(int intermediate_id) const;
  void sort_by_contribution_score(std::vector<int> &candidates);
  void sort_by_access_time(std::vector<int> &candidates);
  void sort_by_memory_size(std::vector<int> &candidates);
  bool can_allocate_contiguous(size_t size) const;
  void merge_free_blocks();

  // CUDA helper methods
  bool check_cuda_memory_available(size_t size) const;
  void sync_gpu_operations() const;

  static std::unique_ptr<MemoryManager> instance;
  static std::mutex instance_mutex;
};

#endif // STRUCTURE_MEMORY_MANAGER_H
