#ifndef STRUCTURE_POLICY_H
#define STRUCTURE_POLICY_H

#include <cinttypes>
#include <cstddef>
#include <list>
#include <queue>
#include <unordered_map>
#include <map>
#include <set>
#include <chrono>

// Forward declaration
class Lattice;

class VirtualEvictPolicy
{
public:
  virtual void onAllocate(uint32_t block_id, size_t sz) = 0;
  virtual void onFree(uint32_t block_id) = 0;

  // pickVictim now takes a reference to pinned blocks to avoid evicting pinned data
  // If a block_id is >= pinned.size() or pinned[block_id] == false, it can be evicted
  virtual uint32_t pickVictim(size_t needBytes, const std::vector<bool>& pinned) = 0;

  // Optional methods with default implementations
  virtual void onAccess(uint32_t block_id) {}
  virtual void setLattice(Lattice* lat) {}
  virtual void setWeights(double lru_w, double contrib_w) {}

  virtual ~VirtualEvictPolicy() {}
};

class LRUEvictPolicy : public VirtualEvictPolicy
{
private:
  // Array-based doubly-linked list simulation using vectors
  static constexpr uint32_t NULL_NODE = std::numeric_limits<uint32_t>::max();
  static constexpr size_t INITIAL_CAPACITY = 1024;

  std::vector<uint32_t> prev;   // prev[block_id] = previous node in LRU list
  std::vector<uint32_t> next;   // next[block_id] = next node in LRU list
  std::vector<size_t> sizes;    // sizes[block_id] = block size
  std::vector<bool> allocated;  // allocated[block_id] = is this block active?

  uint32_t head;  // Most recently used
  uint32_t tail;  // Least recently used

  // Helper methods for list manipulation
  void ensureCapacity(uint32_t block_id);
  void moveToFront(uint32_t block_id);
  void removeNode(uint32_t block_id);
  void addToFront(uint32_t block_id);

public:
  LRUEvictPolicy();

  void onAllocate(uint32_t block_id, size_t sz) override;
  void onFree(uint32_t block_id) override;
  uint32_t pickVictim(size_t needBytes, const std::vector<bool>& pinned) override;
  void onAccess(uint32_t block_id) override;
};

class FIFOEvictPolicy : public VirtualEvictPolicy
{
private:
  std::list<uint32_t> fifo_list;  // Front = oldest, back = newest
  std::unordered_map<uint32_t, size_t> block_sizes;
  std::unordered_map<uint32_t, std::list<uint32_t>::iterator> block_map;

public:
  void onAllocate(uint32_t block_id, size_t sz) override;
  void onFree(uint32_t block_id) override;
  uint32_t pickVictim(size_t needBytes, const std::vector<bool>& pinned) override;
};

class LFUEvictPolicy : public VirtualEvictPolicy
{
private:
  std::unordered_map<uint32_t, int> frequency;
  std::unordered_map<uint32_t, size_t> block_sizes;

public:
  void onAllocate(uint32_t block_id, size_t sz) override;
  void onFree(uint32_t block_id) override;
  uint32_t pickVictim(size_t needBytes, const std::vector<bool>& pinned) override;
  void onAccess(uint32_t block_id) override;
};

class SizeBasedEvictPolicy : public VirtualEvictPolicy
{
private:
  std::unordered_map<uint32_t, size_t> block_sizes;
  std::multimap<size_t, uint32_t> size_to_blocks;  // size -> block_id (sorted by size)

public:
  void onAllocate(uint32_t block_id, size_t sz) override;
  void onFree(uint32_t block_id) override;
  uint32_t pickVictim(size_t needBytes, const std::vector<bool>& pinned) override;
};

class GreedyEvictPolicy : public VirtualEvictPolicy
{
private:
  std::unordered_map<uint32_t, size_t> block_sizes;
  std::multimap<size_t, uint32_t> size_to_blocks;  // size -> block_id (sorted by size)

public:
  void onAllocate(uint32_t block_id, size_t sz) override;
  void onFree(uint32_t block_id) override;
  uint32_t pickVictim(size_t needBytes, const std::vector<bool>& pinned) override;
};

class ContributionEvictPolicy : public VirtualEvictPolicy
{
private:
  Lattice* lattice;
  std::unordered_map<uint32_t, size_t> block_sizes;

public:
  ContributionEvictPolicy() : lattice(nullptr) {}

  void onAllocate(uint32_t block_id, size_t sz) override;
  void onFree(uint32_t block_id) override;
  uint32_t pickVictim(size_t needBytes, const std::vector<bool>& pinned) override;
  void setLattice(Lattice* lat) override;
};

class HybridEvictPolicy : public VirtualEvictPolicy
{
private:
  Lattice* lattice;
  std::unordered_map<uint32_t, size_t> block_sizes;
  std::unordered_map<uint32_t, std::chrono::high_resolution_clock::time_point> access_times;
  double lru_weight;
  double contribution_weight;

public:
  HybridEvictPolicy() : lattice(nullptr), lru_weight(0.6), contribution_weight(0.4) {}

  void onAllocate(uint32_t block_id, size_t sz) override;
  void onFree(uint32_t block_id) override;
  uint32_t pickVictim(size_t needBytes, const std::vector<bool>& pinned) override;
  void onAccess(uint32_t block_id) override;
  void setLattice(Lattice* lat) override;
  void setWeights(double lru_w, double contrib_w) override;
};

#endif