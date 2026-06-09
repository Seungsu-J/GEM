#ifndef JOIN_JOIN_H
#define JOIN_JOIN_H

#include "cpuGraph.h"
#include "gpuGraph.h"
#include "join_trie_kernel.cuh"
#include "lattice.h"
#include "memory_manager.h"
#include "unifiedTrie.cuh"

#include "cuda_helpers.cuh"

extern __device__ __constant__ HashLookupTables *lookup;

void host_select(
    cpuGraph *hq, UnifiedTrie *frag_trie, UnifiedTrie *probe_trie, UnifiedTrie *target_trie,
    uint32_t col_key, uint32_t col_value);

void direct_join(
    cpuGraph *hq, UnifiedTrie *frag_trie, UnifiedTrie *probe_trie, UnifiedTrie *target_trie,
    uint32_t col_key, bool is_forward);

void refer_join(
    cpuGraph *hq, UnifiedTrie *frag_trie, UnifiedTrie *probe_trie, UnifiedTrie *target_trie,
    uint32_t col_key, bool is_forward);

void process(
    cpuGraph *hq, cpuGraph *hg, gpuGraph *dg,
    UnifiedTrie *frag_trie, UnifiedTrie *probe_trie, UnifiedTrie *target_trie);

void join(
    cpuGraph *hq, cpuGraph *hg, gpuGraph *dg);

void join_count_nodup_dispatcher(
    // trie_inter 
    vtype **vertex_arrays_, uint64_t **parent_pointers_,
    uint64_t num_rows, uint32_t num_cols, uint32_t join_col,
    // trie_edge
    vtype *edge_key_array_, vtype *edge_value_array_, uint32_t *edge_offsets_,
    // hash
    uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx,
    // mask
    uint32_t *d_num_res_for_each,
    uint32_t *d_st_array, uint32_t task_per_warp);

void join_write_nodup_dispatcher(
    uint64_t num_rows,
    // trie_edge
    vtype *edge_value_array_,
    uint32_t *d_num_res_for_each, uint32_t *d_st_array,
    uint32_t *d_res, uint64_t *d_offsets_of_, uint32_t num_tasks_per_warp, uint64_t *d_parent_res);

void join_count_refer_no_dup_dispatcher(
    vtype **vertex_arrays, uint64_t **parent_pointers,
    uint64_t num_rows, uint32_t num_cols, uint32_t join_col,
    uint32_t *select_bool_vec,
    vtype *edge_key_array, vtype *edge_value_array, uint32_t *edge_offsets,
    uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx,
    uint32_t *d_num_res_for_each, uint32_t *d_st_array, uint32_t task_per_warp);

void join_write_refer_no_dup_dispatcher(
    uint64_t *valid_row_ids, uint64_t num_valid_rows,
    vtype *edge_value_array_,
    uint32_t *d_selected_count, uint32_t *d_st_array,
    uint32_t *d_res, uint64_t *d_row_offsets, uint64_t *d_parent_res, uint32_t task_per_warp);

#endif //! JOIN_JOIN_H
