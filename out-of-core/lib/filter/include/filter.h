#ifndef FILTER_FILTER_H
#define FILTER_FILTER_H

#include "defs.h"
#include "cpuGraph.h"
#include "gpuGraph.h"
// #include "intermediate.h"
#include "memory_manager.cuh"

void filterSSM(
		cpuGraph *hq, cpuGraph *hg, gpuGraph *dq, gpuGraph *dg,
		vtype *&h_u_candidate_vs_, numtype *&h_num_u_candidate_vs_,
		vtype *&d_u_candidate_vs_, numtype *&d_num_u_candidate_vs_,
		uint32_t *&d_bitmap, uint32_t &bitmap_pitch);

// void construct_edge_candidates(
// 		cpuGraph *hq, cpuGraph *hg,
// 		gpuGraph *dq, gpuGraph *dg,
// 		vtype *&h_u_candidate_vs_, numtype *&h_num_u_candidate_vs_,
// 		vtype *&d_u_candidate_vs_, numtype *&d_num_u_candidate_vs_,
// 		IntermediateManager *&im,
// 		MemoryManager *mem_mgr);

// // Unit test functions
// void verify_sorting(Intermediate *edge_can);
// void verify_index(Intermediate *edge_can, int col_id);

#endif