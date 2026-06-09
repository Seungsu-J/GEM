#ifndef FILTER_CUH
#define FILTER_CUH

#include "defs.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void
buildQueryNLCSSM_kernel(
    degtype *query_out_degrees_, vltype *query_vLabels_,
    offtype *query_offsets_, vtype *query_nbrs_,
    uint32_t *d_query_nlc_table_);

__global__ void
LDfilterSSM_kernel(
    degtype *query_out_degrees_,
    vltype *query_vLabels_,
    offtype *d_offsets_,
    vtype *d_nbrs_,
    vltype *d_v_labels_,
    degtype *d_v_degrees_,
    uint32_t *d_query_nlc_table_,
    uint32_t *d_bitmap, uint32_t bitmap_pitch);

__global__ void
write_results(
    uint32_t *d_bitmap, uint32_t bitmap_pitch,
    uint32_t *d_offset_for_each,
    vtype *d_u_candidate_vs_, numtype *d_num_u_candidate_vs_);

__global__ void
add_kernel(
    uint32_t *d_bitmap,
    uint32_t *d_offset_for_each,
    uint32_t bitmap_pitch,
    uint32_t *d_num_u_candidate_vs_);

#endif //! FILTER_CUH