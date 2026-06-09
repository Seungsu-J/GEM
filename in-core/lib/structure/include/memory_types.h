#ifndef STRUCTURE_MEMORY_TYPES_H
#define STRUCTURE_MEMORY_TYPES_H

// Eviction policy enumeration for easy switching between strategies
enum class EvictionPolicy {
  LRU,          // Least Recently Used
  LFU,          // Least Frequently Used
  CONTRIBUTION, // Based on contribution to target subgraphs
  HYBRID,       // Combination of LRU and contribution
  SIZE_BASED,   // Evict largest items first
  FIFO          // First In, First Out
};

// Transfer optimization strategy
enum class TransferStrategy {
  IMMEDIATE, // Transfer immediately when needed
  BATCHED,   // Batch transfers for efficiency
  PREFETCH,  // Predictive prefetching
  ADAPTIVE   // Adaptive based on access patterns
};

#endif // STRUCTURE_MEMORY_TYPES_H
