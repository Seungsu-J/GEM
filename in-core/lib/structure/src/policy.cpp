#include "policy.h"
#include "lattice.h"
#include <algorithm>
#include <limits>

// Sentinel value for "no victim available"
constexpr uint32_t NO_VICTIM = std::numeric_limits<uint32_t>::max();

// ============================================================================
// LRUEvictPolicy Implementation
// ============================================================================

LRUEvictPolicy::LRUEvictPolicy()
  : head(NULL_NODE), tail(NULL_NODE)
{
  // Pre-allocate with initial capacity
  prev.resize(INITIAL_CAPACITY, NULL_NODE);
  next.resize(INITIAL_CAPACITY, NULL_NODE);
  sizes.resize(INITIAL_CAPACITY, 0);
  allocated.resize(INITIAL_CAPACITY, false);
}

void LRUEvictPolicy::ensureCapacity(uint32_t block_id)
{
  if (block_id >= prev.size())
  {
    size_t new_size = std::max(prev.size() * 2, static_cast<size_t>(block_id + 1));
    prev.resize(new_size, NULL_NODE);
    next.resize(new_size, NULL_NODE);
    sizes.resize(new_size, 0);
    allocated.resize(new_size, false);
  }
}

void LRUEvictPolicy::removeNode(uint32_t block_id)
{
  if (block_id >= allocated.size() || !allocated[block_id])
  {
    return;
  }

  uint32_t prev_id = prev[block_id];
  uint32_t next_id = next[block_id];

  // Update previous node's next pointer
  if (prev_id != NULL_NODE)
  {
    next[prev_id] = next_id;
  }
  else
  {
    // This was the head
    head = next_id;
  }

  // Update next node's prev pointer
  if (next_id != NULL_NODE)
  {
    prev[next_id] = prev_id;
  }
  else
  {
    // This was the tail
    tail = prev_id;
  }
}

void LRUEvictPolicy::addToFront(uint32_t block_id)
{
  // If this block is already at the head, nothing to do.
  if (head == block_id)
  {
    return;
  }

  if (head == NULL_NODE)
  {
    // List is empty
    prev[block_id] = NULL_NODE;
    next[block_id] = NULL_NODE;
    head = block_id;
    tail = block_id;
  }
  else
  {
    // Insert at front
    prev[block_id] = NULL_NODE;
    next[block_id] = head;
    prev[head] = block_id;
    head = block_id;
  }
}

void LRUEvictPolicy::moveToFront(uint32_t block_id)
{
  if (head == block_id)
  {
    // Already at front
    return;
  }

  removeNode(block_id);
  addToFront(block_id);
}

void LRUEvictPolicy::onAllocate(uint32_t block_id, size_t sz)
{
  ensureCapacity(block_id);

  // If this block is already allocated and present in the list,
  // just update its size and move it to the front instead of
  // inserting a duplicate and corrupting the links.
  if (block_id < allocated.size() && allocated[block_id])
  {
    sizes[block_id] = sz;
    moveToFront(block_id);
    return;
  }

  // Initialize node data for a new block
  prev[block_id] = NULL_NODE;
  next[block_id] = NULL_NODE;
  sizes[block_id] = sz;
  allocated[block_id] = true;

  // Add to front (most recent)
  addToFront(block_id);
}

void LRUEvictPolicy::onFree(uint32_t block_id)
{
  if (block_id >= allocated.size() || !allocated[block_id])
  {
    return;
  }

  removeNode(block_id);
  allocated[block_id] = false;
}

uint32_t LRUEvictPolicy::pickVictim(size_t needBytes, const std::vector<bool>& pinned)
{
  if (tail == NULL_NODE)
  {
    return NO_VICTIM;
  }

  // Traverse from tail (LRU) looking for unpinned block
  uint32_t current = tail;
  while (current != NULL_NODE)
  {
    // Check if block can be evicted (not pinned)
    if (current >= pinned.size() || !pinned[current])
    {
      return current;
    }
    current = prev[current];  // Move to next less-recently-used block
  }

  return NO_VICTIM;  // All blocks are pinned
}

void LRUEvictPolicy::onAccess(uint32_t block_id)
{
  if (block_id >= allocated.size() || !allocated[block_id])
  {
    return;
  }

  // Move to front (most recent)
  moveToFront(block_id);
}

// ============================================================================
// FIFOEvictPolicy Implementation
// ============================================================================

void FIFOEvictPolicy::onAllocate(uint32_t block_id, size_t sz)
{
  // Add to back of FIFO list (newest)
  fifo_list.push_back(block_id);
  block_map[block_id] = --fifo_list.end();
  block_sizes[block_id] = sz;
}

void FIFOEvictPolicy::onFree(uint32_t block_id)
{
  auto it = block_map.find(block_id);
  if (it != block_map.end())
  {
    fifo_list.erase(it->second);
    block_map.erase(it);
  }
  block_sizes.erase(block_id);
}

uint32_t FIFOEvictPolicy::pickVictim(size_t needBytes, const std::vector<bool>& pinned)
{
  if (fifo_list.empty())
  {
    return NO_VICTIM;
  }

  // Find first unpinned block from oldest to newest
  for (auto it = fifo_list.begin(); it != fifo_list.end(); ++it)
  {
    uint32_t block_id = *it;
    if (block_id >= pinned.size() || !pinned[block_id])
    {
      return block_id;
    }
  }

  return NO_VICTIM;  // All blocks are pinned
}

// ============================================================================
// LFUEvictPolicy Implementation
// ============================================================================

void LFUEvictPolicy::onAllocate(uint32_t block_id, size_t sz)
{
  frequency[block_id] = 0;  // Start with 0 accesses
  block_sizes[block_id] = sz;
}

void LFUEvictPolicy::onFree(uint32_t block_id)
{
  frequency.erase(block_id);
  block_sizes.erase(block_id);
}

uint32_t LFUEvictPolicy::pickVictim(size_t needBytes, const std::vector<bool>& pinned)
{
  if (frequency.empty())
  {
    return NO_VICTIM;
  }

  // Find unpinned block with minimum frequency
  uint32_t victim = NO_VICTIM;
  int min_freq = std::numeric_limits<int>::max();

  for (const auto& [block_id, freq] : frequency)
  {
    // Skip pinned blocks
    if (block_id < pinned.size() && pinned[block_id])
    {
      continue;
    }

    if (freq < min_freq)
    {
      min_freq = freq;
      victim = block_id;
    }
  }

  return victim;
}

void LFUEvictPolicy::onAccess(uint32_t block_id)
{
  auto it = frequency.find(block_id);
  if (it != frequency.end())
  {
    it->second++;
  }
}

// ============================================================================
// SizeBasedEvictPolicy Implementation
// ============================================================================

void SizeBasedEvictPolicy::onAllocate(uint32_t block_id, size_t sz)
{
  block_sizes[block_id] = sz;
  size_to_blocks.insert({sz, block_id});
}

void SizeBasedEvictPolicy::onFree(uint32_t block_id)
{
  auto it = block_sizes.find(block_id);
  if (it != block_sizes.end())
  {
    size_t sz = it->second;

    // Find and remove from multimap
    auto range = size_to_blocks.equal_range(sz);
    for (auto iter = range.first; iter != range.second; ++iter)
    {
      if (iter->second == block_id)
      {
        size_to_blocks.erase(iter);
        break;
      }
    }

    block_sizes.erase(it);
  }
}

uint32_t SizeBasedEvictPolicy::pickVictim(size_t needBytes, const std::vector<bool>& pinned)
{
  if (size_to_blocks.empty())
  {
    return NO_VICTIM;
  }

  // Evict largest unpinned block (iterate from end)
  for (auto it = size_to_blocks.rbegin(); it != size_to_blocks.rend(); ++it)
  {
    uint32_t block_id = it->second;
    if (block_id >= pinned.size() || !pinned[block_id])
    {
      return block_id;
    }
  }

  return NO_VICTIM;
}

// ============================================================================
// GreedyEvictPolicy Implementation
// ============================================================================

void GreedyEvictPolicy::onAllocate(uint32_t block_id, size_t sz)
{
  block_sizes[block_id] = sz;
  size_to_blocks.insert({sz, block_id});
}

void GreedyEvictPolicy::onFree(uint32_t block_id)
{
  auto it = block_sizes.find(block_id);
  if (it != block_sizes.end())
  {
    size_t sz = it->second;

    // Find and remove from multimap
    auto range = size_to_blocks.equal_range(sz);
    for (auto iter = range.first; iter != range.second; ++iter)
    {
      if (iter->second == block_id)
      {
        size_to_blocks.erase(iter);
        break;
      }
    }

    block_sizes.erase(it);
  }
}

uint32_t GreedyEvictPolicy::pickVictim(size_t needBytes, const std::vector<bool>& pinned)
{
  if (size_to_blocks.empty())
  {
    return NO_VICTIM;
  }

  // Greedy: pick smallest unpinned block that satisfies needBytes
  for (auto it = size_to_blocks.lower_bound(needBytes);
       it != size_to_blocks.end(); ++it)
  {
    uint32_t block_id = it->second;
    if (it->first >= needBytes &&
        (block_id >= pinned.size() || !pinned[block_id]))
    {
      return block_id;
    }
  }

  // If no single block satisfies, return largest unpinned block
  for (auto it = size_to_blocks.rbegin(); it != size_to_blocks.rend(); ++it)
  {
    uint32_t block_id = it->second;
    if (block_id >= pinned.size() || !pinned[block_id])
    {
      return block_id;
    }
  }

  return NO_VICTIM;
}

// ============================================================================
// ContributionEvictPolicy Implementation
// ============================================================================

void ContributionEvictPolicy::onAllocate(uint32_t block_id, size_t sz)
{
  block_sizes[block_id] = sz;
}

void ContributionEvictPolicy::onFree(uint32_t block_id)
{
  block_sizes.erase(block_id);
}

uint32_t ContributionEvictPolicy::pickVictim(size_t needBytes, const std::vector<bool>& pinned)
{
  if (block_sizes.empty())
  {
    return NO_VICTIM;
  }

  if (!lattice)
  {
    // If no lattice set, find first unpinned block
    for (const auto& [block_id, sz] : block_sizes)
    {
      if (block_id >= pinned.size() || !pinned[block_id])
      {
        return block_id;
      }
    }
    return NO_VICTIM;
  }

  // Find unpinned block with minimum contribution value
  uint32_t victim = NO_VICTIM;
  double min_contribution = std::numeric_limits<double>::max();

  for (const auto& [block_id, sz] : block_sizes)
  {
    // Skip pinned blocks
    if (block_id < pinned.size() && pinned[block_id])
    {
      continue;
    }

    // Get contribution value from lattice
    double contrib = 0.0;
    if (lattice->id2et.find(block_id) != lattice->id2et.end())
    {
      auto et = lattice->id2et.at(block_id);
      contrib = lattice->get_contribution_value(et);
    }

    if (contrib < min_contribution)
    {
      min_contribution = contrib;
      victim = block_id;
    }
  }

  return victim;
}

void ContributionEvictPolicy::setLattice(Lattice* lat)
{
  lattice = lat;
}

// ============================================================================
// HybridEvictPolicy Implementation
// ============================================================================

void HybridEvictPolicy::onAllocate(uint32_t block_id, size_t sz)
{
  block_sizes[block_id] = sz;
  access_times[block_id] = std::chrono::high_resolution_clock::now();
}

void HybridEvictPolicy::onFree(uint32_t block_id)
{
  block_sizes.erase(block_id);
  access_times.erase(block_id);
}

uint32_t HybridEvictPolicy::pickVictim(size_t needBytes, const std::vector<bool>& pinned)
{
  if (block_sizes.empty())
  {
    return NO_VICTIM;
  }

  auto now = std::chrono::high_resolution_clock::now();

  uint32_t victim = NO_VICTIM;
  double max_score = -std::numeric_limits<double>::max();

  for (const auto& [block_id, sz] : block_sizes)
  {
    // Skip pinned blocks
    if (block_id < pinned.size() && pinned[block_id])
    {
      continue;
    }

    // Compute LRU component (time since last access, normalized)
    double lru_score = 0.0;
    auto it = access_times.find(block_id);
    if (it != access_times.end())
    {
      auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - it->second).count();
      lru_score = static_cast<double>(elapsed);
    }

    // Compute contribution component
    double contribution_score = 1000.0;  // Default high value
    if (lattice)
    {
      if (lattice->id2et.find(block_id) != lattice->id2et.end())
      {
        auto et = lattice->id2et.at(block_id);
        contribution_score = lattice->get_contribution_value(et);
      }
    }

    // Hybrid score: higher score = better eviction candidate
    // (older access time + lower contribution = higher score)
    double hybrid_score = lru_weight * lru_score +
                         contribution_weight * (1000.0 - contribution_score);

    if (hybrid_score > max_score)
    {
      max_score = hybrid_score;
      victim = block_id;
    }
  }

  return victim;
}

void HybridEvictPolicy::onAccess(uint32_t block_id)
{
  access_times[block_id] = std::chrono::high_resolution_clock::now();
}

void HybridEvictPolicy::setLattice(Lattice* lat)
{
  lattice = lat;
}

void HybridEvictPolicy::setWeights(double lru_w, double contrib_w)
{
  lru_weight = lru_w;
  contribution_weight = contrib_w;
}
