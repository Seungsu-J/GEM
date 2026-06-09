#ifndef GPUGRAPH_H
#define GPUGRAPH_H

#include <cinttypes>

#include "cpuGraph.h"
#include "globals.h"

class gpuGraph
{
public:
  void *allocation_base_; // single packed allocation owned by this graph
  uint32_t *degree_; // arr
  vltype *vLabels_;  // arr
  // eltype *eLabels_; // arr

  // CSR
  offtype *offsets_;        // arr
  vtype *neighbors_;        // arr size = 2*|E|
  etype *edgeIDs_;          // arr

  gpuGraph();
  ~gpuGraph();
};

#endif
