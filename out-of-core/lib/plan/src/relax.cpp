#include "relax.h"
#include "globals.h"

#include <unordered_map>
#include <unordered_set>
#include <set>
#include <queue>
#include <vector>

// Union-Find data structure for efficient connectivity checking
class UnionFind {
private:
  std::vector<vtype> parent;
  std::vector<int> rank;
  int num_components;

public:
  UnionFind(int n) : parent(n), rank(n, 0), num_components(n) {
    for (int i = 0; i < n; ++i) {
      parent[i] = i;
    }
  }

  vtype find(vtype x) {
    if (parent[x] != x) {
      parent[x] = find(parent[x]); // Path compression
    }
    return parent[x];
  }

  bool unite(vtype x, vtype y) {
    vtype root_x = find(x);
    vtype root_y = find(y);

    if (root_x == root_y)
      return false; // Already in same component

    // Union by rank
    if (rank[root_x] < rank[root_y]) {
      parent[root_x] = root_y;
    } else if (rank[root_x] > rank[root_y]) {
      parent[root_y] = root_x;
    } else {
      parent[root_y] = root_x;
      rank[root_x]++;
    }

    num_components--;
    return true;
  }

  int get_num_components() const {
    return num_components;
  }
};

// Optimized connectivity check using Union-Find
bool checkConnectivity(const ettype &cur_ET, cpuGraph *graph)
{
  if (graph->num_v <= 1)
    return true;

  UnionFind uf(graph->num_v);

  // Unite vertices for each present edge
  for (etype i = 0; i < graph->num_e; ++i) 
  {
    if (cur_ET[i])
     {
      etype eid = i << 1; // Convert to edge ID
      auto it = graph->evv.find(eid);
      if (it != graph->evv.end()) 
      {
        auto [u, v] = it->second;
        uf.unite(u, v);
      }
    }
  }

  // Graph is connected if there's exactly 1 component
  return uf.get_num_components() == 1;
}

void relax(
    cpuGraph *graph,
    std::unordered_set<ettype> &edgeTagSet)
{
  // Early exit for trivial cases
  if (graph->num_e <= 1)
    return;

  // Use a queue that tracks both the edge configuration and distance from original
  std::queue<std::pair<ettype, uint32_t>> q;
  q.push({graph->edgeTag, 0}); // Original graph has distance 0
  edgeTagSet.insert(graph->edgeTag);

  const uint32_t original_edge_count = graph->edgeTag.count();

  while (!q.empty())
  {
    auto [cur_ET, distance] = q.front();
    q.pop();

    // Skip if we've reached the threshold distance
    if (distance >= THRESHOLD)
      continue;

    // Skip if already a spanning tree
    if (cur_ET.count() <= graph->num_v - 1)
      continue;

    // Try removing each edge
    for (etype i = 0; i < graph->num_e; ++i)
    {
      if (!cur_ET[i])
        continue; // Edge not present

      ettype next_ET = cur_ET;
      next_ET.flip(i);

      // Calculate distance from original graph (number of edges removed)
      uint32_t next_distance = distance + 1;

      // Optimized: Try to insert first, check if it was already there
      auto [iter, inserted] = edgeTagSet.insert(next_ET);
      if (!inserted)
        continue; // Already visited this configuration

      // Check connectivity with the new edge configuration
      if (checkConnectivity(next_ET, graph))
      {
        // Already inserted above, just add to queue
        q.push({next_ET, next_distance});
      }
      else
      {
        // Not connected, remove from set
        edgeTagSet.erase(iter);
      }
    }
  }
}

void constructRelaxedQueryGraph(
    ettype ET, cpuGraph *graph, cpuGraph *relaxedGraph)
{
  // Basic properties
  relaxedGraph->num_v = graph->num_v;
  relaxedGraph->num_e = ET.count();
  relaxedGraph->num_v_labels = graph->num_v_labels;
  relaxedGraph->edgeTag = ET;
  relaxedGraph->isQuery = graph->isQuery;

  // Copy label information
  relaxedGraph->vLabelFreq = graph->vLabelFreq;
  relaxedGraph->maxLabelFreq = graph->maxLabelFreq;
  relaxedGraph->vLabels_ = graph->vLabels_;
  relaxedGraph->vertexIDs_ = graph->vertexIDs_;

  // Initialize degree arrays and other data structures
  relaxedGraph->in_degree_.assign(graph->num_v, 0);
  relaxedGraph->out_degree_.assign(graph->num_v, 0);
  relaxedGraph->offsets_.assign(graph->num_v + 1, 0);
  relaxedGraph->neighbors_.resize(relaxedGraph->num_e << 1);
  relaxedGraph->edgeIDs_.resize(relaxedGraph->num_e << 1);

  // Copy other structures
  // relaxedGraph->keep = graph->keep;
  relaxedGraph->vve = graph->vve;
  relaxedGraph->evv = graph->evv;
  relaxedGraph->v2global = graph->v2global;

  // Build the relaxed graph structure efficiently
  offtype off_global = 0;
  relaxedGraph->maxDegree = 0;

  // First pass: count degrees and build offset structure
  for (vtype u = 0; u < graph->num_v; ++u)
  {
    for (offtype off = graph->offsets_[u]; off < graph->offsets_[u + 1]; ++off)
    {
      etype edgeID = graph->edgeIDs_[off];
      if (ET[edgeID >> 1])
      {
        relaxedGraph->offsets_[u + 1]++;
        relaxedGraph->out_degree_[u]++;
        relaxedGraph->in_degree_[u]++;
      }
    }
    relaxedGraph->maxDegree = std::max(relaxedGraph->maxDegree, relaxedGraph->out_degree_[u]);
  }

  // Convert counts to offsets
  for (vtype i = 0; i < graph->num_v; ++i)
  {
    relaxedGraph->offsets_[i + 1] += relaxedGraph->offsets_[i];
  }

  // Second pass: fill neighbors and edge IDs
  std::vector<offtype> current_offset = relaxedGraph->offsets_; // Copy for tracking
  for (vtype u = 0; u < graph->num_v; ++u)
  {
    for (offtype off = graph->offsets_[u]; off < graph->offsets_[u + 1]; ++off)
    {
      vtype neighbor = graph->neighbors_[off];
      etype edgeID = graph->edgeIDs_[off];
      if (ET[edgeID >> 1])
      {
        relaxedGraph->neighbors_[current_offset[u]] = neighbor;
        relaxedGraph->edgeIDs_[current_offset[u]] = edgeID;
        current_offset[u]++;
      }
    }
  }
}