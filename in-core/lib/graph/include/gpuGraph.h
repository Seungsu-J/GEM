#ifndef GPUGRAPH_H
#define GPUGRAPH_H

#include <cinttypes>

#include "cpuGraph.h"
#include "globals.h"

class gpuGraph
{
public:
  uint32_t *degree_; // arr
  vltype *vLabels_;  // arr
  // eltype *eLabels_; // arr

  // CSR
  uint32_t *offsets_;       // arr
  vtype *neighbors_;        // arr size = 2*|E|
  etype *edgeIDs_;          // arr

  uint32_t *data_; // used for actual data.

  gpuGraph();
  ~gpuGraph();
};

#endif