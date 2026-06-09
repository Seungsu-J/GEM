#include "cuda_helpers.cuh"
#include "filter.h"
#include "filter_kernel.cuh"
#include "join_trie.cuh"
#include "memory_manager.h"

#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cub/cub.cuh>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <set>
#include <stack>
#include <unordered_set>
#include <vector>

#define row_size ((NUM_VD - 1) / 32 + 1)

// Custom transformation functor to compute popcount for CUB scan
struct PopcountTransform
{
  __device__ __forceinline__ uint32_t operator()(const uint32_t &bitmap) const
  {
    return __popc(bitmap);
  }
};

void filterSSM(cpuGraph *hq, cpuGraph *hg, gpuGraph *dq, gpuGraph *dg,
               vtype *&h_u_candidate_vs_, numtype *&h_num_u_candidate_vs_,
               vtype *&d_u_candidate_vs_, numtype *&d_num_u_candidate_vs_,
               uint32_t *&d_bitmap, uint32_t& bitmap_pitch)
{
  cudaStream_t filter_stream;
  // cuchk(cudaStreamCreateWithFlags(&filter_stream, cudaStreamNonBlocking));
  cuchk(cudaStreamCreate(&filter_stream));

  size_t candidate_count_size = sizeof(numtype) * NUM_VQ;
  size_t candidate_data_size = sizeof(vtype) * NUM_VQ * MAX_L_FREQ;
  size_t query_table_size = sizeof(uint32_t) * NUM_VQ * NUM_VLQ;
  // size_t total_gpu_memory = candidate_count_size + candidate_data_size + query_table_size;

  uint32_t *d_query_nlc_table_;

  cuchk(cudaMallocAsync((void **)&d_query_nlc_table_, query_table_size, filter_stream));
  cuchk(cudaMemsetAsync(d_query_nlc_table_, 0, query_table_size, filter_stream));

  dim3 grid_dim(GRID_DIM);
  dim3 block_dim(BLOCK_DIM);

  // Build Query NLC table
  buildQueryNLCSSM_kernel<<<grid_dim, block_dim, 0, filter_stream>>>(
      dq->degree_, dq->vLabels_, dq->offsets_, dq->neighbors_,
      d_query_nlc_table_);

  bitmap_pitch = ((NUM_VD + 31) / 32 + 3) & ~3; // align to 4
  cuchk(cudaMallocAsync((void **)&d_bitmap, sizeof(uint32_t) * bitmap_pitch * NUM_VQ, filter_stream));
  cuchk(cudaMemsetAsync(d_bitmap, 0, sizeof(uint32_t) * bitmap_pitch * NUM_VQ, filter_stream));

  LDfilterSSM_kernel<<<grid_dim, block_dim, 0, filter_stream>>>(
      dq->degree_, dq->vLabels_, dg->offsets_, dg->neighbors_, dg->vLabels_,
      dg->degree_, d_query_nlc_table_, d_bitmap, bitmap_pitch);

  // Allocate offset array for exclusive scan results
  uint32_t *d_offset_for_each;
  cuchk(cudaMallocAsync((void **)&d_offset_for_each, sizeof(uint32_t) * NUM_VQ * bitmap_pitch, filter_stream));

  // Perform exclusive scan with popcount transformation for each query vertex
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  PopcountTransform popcount_transform;

  // Type alias for the transform iterator
  typedef cub::TransformInputIterator<uint32_t, PopcountTransform, uint32_t *> PopcountIterator;

  for (uint32_t u = 0; u < NUM_VQ; ++u)
  {
    uint32_t *bitmap_u = d_bitmap + u * bitmap_pitch;
    uint32_t *offset_u = d_offset_for_each + u * bitmap_pitch;

    // Create a transformation iterator that applies popcount
    PopcountIterator popcount_iter(bitmap_u, popcount_transform);

    // Determine temporary storage size (only once)
    if (u == 0)
    {
      cuchk(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
                                          popcount_iter, offset_u, bitmap_pitch, filter_stream));
      cuchk(cudaMallocAsync(&d_temp_storage, temp_storage_bytes, filter_stream));
    }

    // Run exclusive scan with transformation
    cuchk(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
                                        popcount_iter, offset_u, bitmap_pitch, filter_stream));
  }

  add_kernel<<<GRID_DIM, BLOCK_DIM, 0, filter_stream>>>(
      d_bitmap, d_offset_for_each, bitmap_pitch, d_num_u_candidate_vs_);

  // Free temporary storage
  cuchk(cudaFreeAsync(d_temp_storage, filter_stream));
  // Call write_results kernel to materialize candidates
  write_results<<<grid_dim, block_dim, 0, filter_stream>>>(
      d_bitmap, bitmap_pitch, d_offset_for_each, d_u_candidate_vs_, d_num_u_candidate_vs_);

  // Cleanup
  cuchk(cudaFreeAsync(d_offset_for_each, filter_stream));
  // cuchk(cudaFreeAsync(d_bitmap, filter_stream));

  cuchk(cudaMemcpyAsync(h_num_u_candidate_vs_, d_num_u_candidate_vs_,
                        candidate_count_size, cudaMemcpyDeviceToHost, filter_stream));

// Report filtering statistics
#ifndef NDEBUG
  cuchk(cudaStreamSynchronize(filter_stream));
  numtype total_candidates = 0;
  for (uint32_t i = 0; i < NUM_VQ; ++i)
  {
    total_candidates += h_num_u_candidate_vs_[i];
  }

  std::cout << "Filtering completed successfully:" << std::endl;
  std::cout << "  - Query vertices processed: " << NUM_VQ << std::endl;
  std::cout << "  - Total candidates found: " << total_candidates << std::endl;
  std::cout << "  - Average candidates per vertex: "
            << (total_candidates / (float)NUM_VQ) << std::endl;

  /* === vertex filter passed === */
  // h_u_candidate_vs_ = new vtype[NUM_VQ * MAX_L_FREQ];
  // cuchk(cudaMemcpy(h_u_candidate_vs_, d_u_candidate_vs_, candidate_data_size,
  //                  cudaMemcpyDeviceToHost));
  // std::ofstream candidates_file("filtering_candidates.txt");
  // if (candidates_file.is_open()) {
  //   vtype candidate_offset = 0;
  //   for (uint32_t u = 0; u < NUM_VQ; ++u) {
  //     candidates_file << "Query vertex " << u << " ("
  //                     << h_num_u_candidate_vs_[u] << " candidates): ";
  //     for (numtype i = 0; i < h_num_u_candidate_vs_[u]; ++i) {
  //       if (i > 0)
  //         candidates_file << " ";
  //       candidates_file << h_u_candidate_vs_[candidate_offset + i];
  //     }
  //     candidates_file << std::endl;
  //     candidate_offset += MAX_L_FREQ;
  //   }
  //   candidates_file.close();
  //   std::cout << "  - Candidates written to filtering_candidates.txt"
  //             << std::endl;
  // } else {
  //   std::cerr << "Warning: Could not open filtering_candidates.txt for
  //   writing"
  //             << std::endl;
  // }
  // delete[] h_u_candidate_vs_;
#endif

  // Cleanup
  // cuchk(cudaFree(device_space));
  cuchk(cudaFreeAsync(d_query_nlc_table_, filter_stream));
  cuchk(cudaStreamSynchronize(filter_stream));
  cuchk(cudaStreamDestroy(filter_stream));
}
