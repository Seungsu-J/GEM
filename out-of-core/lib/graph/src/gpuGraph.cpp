#include "gpuGraph.h"
#include "cuda_helpers.cuh"

gpuGraph::gpuGraph()
{
  // elCount = 0;
  allocation_base_ = nullptr;
  degree_ = nullptr;
  vLabels_ = nullptr;
  // eLabels_ = nullptr;
  offsets_ = nullptr;
  neighbors_ = nullptr;
  edgeIDs_ = nullptr;
}

gpuGraph::~gpuGraph()
{
  if (allocation_base_ != nullptr)
  {
    cudaFree(allocation_base_);
    allocation_base_ = nullptr;
    degree_ = nullptr;
    vLabels_ = nullptr;
    offsets_ = nullptr;
    neighbors_ = nullptr;
    edgeIDs_ = nullptr;
    return;
  }

  if (degree_ != nullptr)
  {
    cudaFree(degree_);
    degree_ = nullptr;
  }
  if (vLabels_ != nullptr)
  {
    cudaFree(vLabels_);
    vLabels_ = nullptr;
  }
  if (offsets_ != nullptr)
  {
    cudaFree(offsets_);
    offsets_ = nullptr;
  }
  if (neighbors_ != nullptr)
  {
    cudaFree(neighbors_);
    neighbors_ = nullptr;
  }
  if (edgeIDs_ != nullptr)
  {
    cudaFree(edgeIDs_);
    edgeIDs_ = nullptr;
  }
}
