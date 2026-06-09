#include "cpuGraph.h"
#include "globals.h"

#include <iostream>
#include <queue>
#include <set>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <cassert>
#include <memory>

// Static member definitions
std::unordered_map<vltype, vltype> cpuGraph::vlmapo2n;
std::unordered_map<vltype, vltype> cpuGraph::vlmapn2o;
ettype cpuGraph::bridge_edge_mask;

// Performance optimization: Cache frequently accessed values
thread_local std::unordered_map<vtype, uint32_t> vertex_offset_cache;

cpuGraph::cpuGraph()
    : num_v(0), num_e(0), num_v_labels(0), maxDegree(0), maxLabelFreq(0), keep(nullptr)
{
  // Reserve space for common graph sizes to avoid frequent reallocations
  in_degree_.reserve(32);
  out_degree_.reserve(32);
  vLabels_.reserve(32);
  vertexIDs_.reserve(64);
  offsets_.reserve(33);
  neighbors_.reserve(128);
  edgeIDs_.reserve(128);
}

cpuGraph::~cpuGraph()
{
  // Use smart pointer principles - no need for manual array deletion
  delete[] keep;
  keep = nullptr;

  // Clear cache for this graph instance
  vertex_offset_cache.clear();
}

offtype cpuGraph::get_u_off(vtype u)
{
  // Performance optimization: Use caching for frequently accessed vertices
  auto cache_it = vertex_offset_cache.find(u);
  if (cache_it != vertex_offset_cache.end())
  {
    return cache_it->second;
  }

  // Optimized search with early termination and better memory access patterns
  const auto *vertex_data = vertexIDs_.data();
  const auto vertex_count = static_cast<uint32_t>(num_v);

  // Use pointer arithmetic for better performance
  for (uint32_t i = 0; i < vertex_count; ++i)
  {
    if (__builtin_expect(vertex_data[i] == u, 0))
    {                             // Branch prediction hint
      vertex_offset_cache[u] = i; // Cache the result
      return i;
    }
  }

  // Error handling with better diagnostics
  std::cerr << "ERROR! in get_u_off(): vertex " << u << " not found in graph with "
            << vertex_count << " vertices" << std::endl;
  assert(false && "Vertex not found in graph");
  std::exit(EXIT_FAILURE);
}

bool cpuGraph::check_connectivity(ettype tag)
{
  // Optimized vertex set collection using unordered_set for O(1) operations
  static thread_local std::unordered_set<vtype> vertex_set;
  vertex_set.clear();
  vertex_set.reserve(NUM_VQ); // Pre-allocate for performance

  // Cache evv data pointer for better memory access
  const auto &edge_map = evv;

  for (int i = 0; i < NUM_EQ; ++i)
  {
    if (__builtin_expect(tag[i], 1))
    { // Branch prediction hint for common case
      const etype eid = i << 1;
      auto evv_it = edge_map.find(eid);
      if (__builtin_expect(evv_it != edge_map.end(), 1))
      {
        vertex_set.insert(evv_it->second.first);
        vertex_set.insert(evv_it->second.second);
      }
    }
  }

  const bool is_complete = (vertex_set.size() == NUM_VQ);

  if (!is_complete)
  {
    // Optimized BFS with stack allocation and better memory access patterns
    static thread_local std::vector<bool> visited; // More cache-friendly than int array
    visited.assign(NUM_VQ, false);                 // Reset and resize for each call
    std::queue<vtype> bfs_queue;

    vtype root = *(vertex_set.begin());

    bfs_queue.push(root); // ✅ Use the actual root from vertex_set
    visited[root] = true; // ✅ Mark the actual root as visited

    // Cache frequently accessed data
    const auto *offsets_data = offsets_.data();
    const auto *neighbors_data = neighbors_.data();
    const auto *edge_ids_data = edgeIDs_.data();

    while (!bfs_queue.empty())
    {
      const vtype current = bfs_queue.front();
      bfs_queue.pop();

      const auto start_offset = offsets_data[current];
      const auto end_offset = offsets_data[current + 1];

      for (auto u_off = start_offset; u_off < end_offset; ++u_off)
      {
        const vtype neighbor = neighbors_data[u_off];
        const etype eid = edge_ids_data[u_off];

        if (tag[eid >> 1] && !visited[neighbor])
        {
          bfs_queue.push(neighbor);
          visited[neighbor] = true;
        }
      }
    }

    // Check if all vertices in the set are reachable
    auto &visited_ref = visited; // Create a reference to avoid static capture warning
    return std::all_of(vertex_set.begin(), vertex_set.end(),
                       [&visited_ref](vtype v)
                       { return visited_ref[v]; });
  }
  else
  {
    // Complete case with bridge mask optimization
    if ((tag & bridge_edge_mask) != bridge_edge_mask)
    {
      return false;
    }

    // Optimized BFS for complete case
    std::vector<bool> visited(NUM_VQ, false);
    std::queue<vtype> bfs_queue;

    bfs_queue.push(0);
    visited[0] = true;

    // Cache data pointers
    const auto *offsets_data = offsets_.data();
    const auto *neighbors_data = neighbors_.data();
    const auto *edge_ids_data = edgeIDs_.data();

    while (!bfs_queue.empty())
    {
      const vtype current = bfs_queue.front();
      bfs_queue.pop();

      const auto start_offset = offsets_data[current];
      const auto end_offset = offsets_data[current + 1];

      for (auto u_off = start_offset; u_off < end_offset; ++u_off)
      {
        const vtype neighbor = neighbors_data[u_off];
        const etype eid = edge_ids_data[u_off];

        if (tag[eid >> 1] && !visited[neighbor])
        {
          bfs_queue.push(neighbor);
          visited[neighbor] = true;
        }
      }
    }

    // Check if all vertices in vertex_set are visited (same logic as incomplete case)
    // For complete case, vertex_set should contain all vertices 0 to NUM_VQ-1
    for (vtype v = 0; v < NUM_VQ; ++v)
    {
      if (!visited[v])
        return false;
    }
    return true;
  }
}

void cpuGraph::dfs_tarjan(
    vtype u, vtype parent,
    int &time,
    std::vector<vtype> &disc, std::vector<vtype> &low)
{
  disc[u] = low[u] = ++time;

  // Cache data pointers for better performance
  const auto *offsets_data = offsets_.data();
  const auto *neighbors_data = neighbors_.data();
  const auto *edge_ids_data = edgeIDs_.data();

  const auto start_offset = offsets_data[u];
  const auto end_offset = offsets_data[u + 1];

  for (auto u_off = start_offset; u_off < end_offset; ++u_off)
  {
    const vtype u_nbr = neighbors_data[u_off];

    if (__builtin_expect(u_nbr == parent, 0))
    {           // Branch prediction for rare case
      continue; // Skip the edge to parent
    }

    if (disc[u_nbr] == static_cast<vtype>(-1))
    { // If u_nbr is not visited
      // Recursive call with tail call optimization potential
      dfs_tarjan(u_nbr, u, time, disc, low);

      // Update low value efficiently
      low[u] = std::min(low[u], low[u_nbr]);

      // Bridge detection with optimized bit manipulation
      if (__builtin_expect(low[u_nbr] > disc[u], 0))
      { // Branch prediction for rare bridges
        const etype edge_id = edge_ids_data[u_off];
        bridge_edge_mask[edge_id >> 1] = 1; // Mark the edge as a bridge
      }
    }
    else
    {
      // Update low value for back edge
      low[u] = std::min(low[u], disc[u_nbr]);
    }
  }
}

void cpuGraph::set_bridge_edge_mask()
{
  // Pre-allocate vectors with better initialization
  std::vector<vtype> disc(NUM_VQ);
  std::vector<vtype> low(NUM_VQ);

  // Initialize efficiently
  std::fill(disc.begin(), disc.end(), static_cast<vtype>(-1));
  std::fill(low.begin(), low.end(), static_cast<vtype>(-1));

  int time = 0;

  // Initialize the bridge_edge_mask efficiently
  bridge_edge_mask.reset();

  // Process all components with optimized loop
  for (vtype u = 0; u < NUM_VQ; ++u)
  {
    if (__builtin_expect(disc[u] == static_cast<vtype>(-1), 0))
    { // Branch prediction
      dfs_tarjan(u, static_cast<vtype>(-1), time, disc, low);
    }
  }
}

void cpuGraph::Print()
{
  // Use more efficient I/O operations
  std::ios_base::sync_with_stdio(false);

  std::cout << "============================\n"
            << "num_v: " << num_v << '\n'
            << "num_e: " << num_e << '\n'
            << "maxDegree: " << maxDegree << '\n';

  // Optimized array printing with single loop and better formatting
  std::cout << "out_degree_: ";
  for (int i = 0; i < NUM_VQ; ++i)
  {
    std::cout << out_degree_[i] << (i == NUM_VQ - 1 ? '\n' : ' ');
  }

  std::cout << "vLabels: ";
  for (int i = 0; i < NUM_VQ; ++i)
  {
    std::cout << vLabels_[i] << (i == NUM_VQ - 1 ? '\n' : ' ');
  }

  std::cout << "maxLabelFreq: " << maxLabelFreq << '\n';

  std::cout << "vertexIDs_: ";
  for (int i = 0; i < num_v; ++i)
  {
    std::cout << vertexIDs_[i] << (i == num_v - 1 ? '\n' : ' ');
  }

  std::cout << "offsets_: ";
  for (int i = 0; i <= NUM_VQ; ++i)
  {
    std::cout << offsets_[i] << (i == NUM_VQ ? '\n' : ' ');
  }

  const auto edge_count = num_e * 2;
  std::cout << "neighbors_: ";
  for (int i = 0; i < edge_count; ++i)
  {
    std::cout << neighbors_[i] << (i == edge_count - 1 ? '\n' : ' ');
  }

  std::cout << "edgeIDs_: ";
  for (int i = 0; i < edge_count; ++i)
  {
    std::cout << edgeIDs_[i] << (i == edge_count - 1 ? '\n' : ' ');
  }

  std::cout << "============================" << std::endl;
}
