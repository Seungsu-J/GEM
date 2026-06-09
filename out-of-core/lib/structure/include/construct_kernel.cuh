#ifndef STRUCTURE_CONSTRUCT_KERNEL_CUH
#define STRUCTURE_CONSTRUCT_KERNEL_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "defs.h"
#include "globals.h"

__global__ void
exclusive_scan_on_mask(
    uint32_t *bitmap_mask_, uint32_t mask_length,
    uint32_t num_rows,
    uint32_t *scan_result);

__global__ void
first_join_count_new(
    // data graph
    offtype *offsets_, vtype *nbrs_, degtype *degree_,
    uint32_t *bitmap_mask_, uint32_t mask_length,
    vtype u, vtype u_nbr,
    numtype num_can_u, numtype num_can_u_nbr,
    vtype *d_u_candidate_vs_);

__global__ void
first_join_count_global_workpool_32_32(
    // data graph
    offtype *offsets_, vtype *nbrs_, degtype *degree_,
    uint32_t *bitmap_mask_, uint32_t mask_length,
    vtype u, vtype u_nbr,
    numtype num_can_u, numtype num_can_u_nbr,
    vtype *d_u_candidate_vs_, uint32_t *workpool);

__global__ void first_join_count_global_32_32(
    // data graph
    offtype *offsets_, vtype *nbrs_, degtype *degree_,
    uint32_t *bitmap_mask_, uint32_t mask_length,
    vtype u, vtype u_nbr,
    numtype num_can_u, numtype num_can_u_nbr,
    vtype *d_u_candidate_vs_);

__global__ void
first_join_count_global_workpool_1_32(
    // data graph
    offtype *offsets_, vtype *nbrs_, degtype *degree_,
    uint32_t *bitmap_mask_, uint32_t mask_length,
    vtype u, vtype u_nbr,
    numtype num_can_u, numtype num_can_u_nbr,
    vtype *d_u_candidate_vs_, uint32_t *workpool);

__global__ void first_join_count_global_1_32(
    // data graph
    offtype *offsets_, vtype *nbrs_, degtype *degree_,
    uint32_t *bitmap_mask_, uint32_t mask_length,
    vtype u, vtype u_nbr,
    numtype num_can_u, numtype num_can_u_nbr,
    vtype *d_u_candidate_vs_);

__global__ void
first_join_count_global_1_1(
    // data graph
    offtype *offsets_, vtype *nbrs_, degtype *degree_,
    uint32_t *bitmap_mask_, uint32_t mask_length,
    vtype u, vtype u_nbr,
    numtype num_can_u, numtype num_can_u_nbr,
    vtype *d_u_candidate_vs_);

__global__ void
first_join_count_global_para_serial(
    // data graph
    offtype *offsets_, vtype *nbrs_, degtype *degree_,
    uint32_t *bitmap_mask_, uint32_t mask_length,
    vtype u, vtype u_nbr,
    numtype num_can_u, numtype num_can_u_nbr,
    vtype *d_u_candidate_vs_);

__global__ void
first_join_count_global_merge_path(
    // data graph
    offtype *offsets_, vtype *nbrs_, degtype *degree_,
    uint32_t *bitmap_mask_, uint32_t mask_length,
    vtype u, vtype u_nbr,
    numtype num_can_u, numtype num_can_u_nbr,
    vtype *d_u_candidate_vs_);

__global__ void first_join_write_new(
    offtype *offsets_, vtype *nbrs_, degtype *degree_,
    uint32_t *bitmap_mask_, uint32_t mask_length,
    vtype u, vtype u_nbr,
    numtype num_can_u, numtype num_can_u_nbr,
    vtype *d_u_candidate_vs_,
    vtype *d_res, offtype *d_offsets_each_row, uint32_t *d_offsets_each_tile, numtype scan_res_length,
    uint64_t *d_parent_res);

__global__ void
first_join_write_global_32_32(
    offtype *offsets_, vtype *nbrs_, degtype *degree_,
    uint32_t *bitmap_mask_, uint32_t mask_length,
    vtype u, vtype u_nbr,
    numtype num_can_u, numtype num_can_u_nbr,
    vtype *d_u_candidate_vs_,
    vtype *d_res, offtype *d_offsets_each_row, uint32_t *d_offsets_each_tile, numtype scan_res_length,
    uint64_t *d_parent_res);

__global__ void
first_join_write_global_1_32(
    offtype *offsets_, vtype *nbrs_, degtype *degree_,
    uint32_t *bitmap_mask_, uint32_t mask_length,
    vtype u, vtype u_nbr,
    numtype num_can_u, numtype num_can_u_nbr,
    vtype *d_u_candidate_vs_,
    vtype *d_res, offtype *d_offsets_each_row, uint32_t *d_offsets_each_tile, numtype scan_res_length,
    uint64_t *d_parent_res);

__global__ void
first_join_count_bitmap_task1_shared(
    // data graph
    offtype *offsets_, vtype *nbrs_, degtype *degree_,
    uint32_t *d_bitmap_u_nbr, // bitmap from filter.
    vtype u, numtype num_can_u, vtype *d_u_candidate_vs_,
    uint32_t *neighbor_mask_, uint32_t *neighbor_mask_offsets_, uint32_t *workpool);

__global__ void first_join_count_bitmap_task1(
    // data graph
    offtype *offsets_, vtype *nbrs_, degtype *degree_,
    uint32_t *d_bitmap_u_nbr, // bitmap from filter.
    vtype u, numtype num_can_u, vtype *d_u_candidate_vs_,
    uint32_t *neighbor_mask_, uint32_t *neighbor_mask_offsets_, uint32_t *workpool);

__global__ void
first_join_write_bitmap_task1_shared(
    offtype *offsets_, vtype *nbrs_, degtype *degree_,
    uint32_t *neighbor_mask_, uint32_t *neighbor_mask_offsets_, uint32_t *scan_result,
    vtype u, numtype num_can_u, vtype *d_u_candidate_vs_,
    vtype *d_res, offtype *d_offsets_each_row, uint64_t *d_parent_res,
    uint32_t *workpool);

__global__ void
first_join_write_bitmap_task1(
    offtype *offsets_, vtype *nbrs_, degtype *degree_,
    uint32_t *neighbor_mask_, uint32_t *neighbor_mask_offsets_, uint32_t *scan_result,
    vtype u, numtype num_can_u, vtype *d_u_candidate_vs_,
    vtype *d_res, offtype *d_offsets_each_row, uint64_t *d_parent_res,
    uint32_t *workpool);

__global__ void
getMax(
    vtype *d_u_candidate_vs_, uint32_t num_can_u,
    uint32_t *scan_output, uint32_t *neighbor_mask_offsets_,
    uint32_t *dst, offtype *d_edge_offsets);

// launch by <<<1,1>>>
__global__ void
add_last_for_exclusive_sum_div32ceil(
    uint32_t *org_data_array, uint32_t *scan_output_array, uint32_t org_length, uint32_t *dst);

// launch by <<<1,1>>>
__global__ void
add_last_for_exclusive_sum_popc(
    uint32_t *org_data_array, uint32_t *scan_output_array, uint32_t org_length, uint32_t *dst);
#endif
