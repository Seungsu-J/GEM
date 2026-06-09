#ifndef CPUGRAPH_H
#define CPUGRAPH_H

#include <iostream>
#include <vector>
#include <map>
#include <bitset>
#include <unordered_map>

#include "defs.h"

class cpuGraph
{
public:
  numtype num_v; // number of vertices
  uint64_t num_e; // number of edges (count undirected edge once)
  numtype num_v_labels;
  degtype maxDegree;

  std::map<vltype, numtype> vLabelFreq; // only for query graph.
  numtype maxLabelFreq;

  std::vector<degtype> in_degree_;
  std::vector<degtype> out_degree_;
  std::vector<vltype> vLabels_;
  // std::vector<etype> eLabels;

  // CSR
  std::vector<vtype> vertexIDs_; // size = num_v
  std::vector<offtype> offsets_; // size = num_v + 1
  std::vector<vtype> neighbors_; // size = num_e * 2
  std::vector<etype> edgeIDs_;   // size = num_e * 2

  /**
   * edgeTag is a bitset, each bit represents an edge. (undirected edge)
   * edgeTag[1] represents `eid` 2 and 3.
   * in opposite, eid >> 1 represents the corresponding edgeTag bit.
   */
  ettype edgeTag;

  bool isQuery;
  bool *keep;

  std::map<std::pair<vtype, vtype>, etype> vve;
  std::map<etype, std::pair<vtype, vtype>> evv; // e is `eid`, for an undirected edge, eids are different.

  std::map<vtype, vtype> v2global;
  static std::unordered_map<vltype, vltype> vlmapo2n; // v_label_map original to new
  static std::unordered_map<vltype, vltype> vlmapn2o; // v_label_map new to original
  static ettype bridge_edge_mask;

public:
  cpuGraph();
  ~cpuGraph();

  void Print();
  void set_bridge_edge_mask();
  bool check_connectivity(ettype tag);

  offtype get_u_off(vtype u);

private:
  void dfs_tarjan(
      vtype u, vtype parent,
      int &time,
      std::vector<vtype> &disc, std::vector<vtype> &low);
};

#endif
