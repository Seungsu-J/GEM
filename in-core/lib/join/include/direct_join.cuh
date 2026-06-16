#ifndef JOIN_DIRECT_JOIN_CUH
#define JOIN_DIRECT_JOIN_CUH

#include "join_helpers.h"

void direct_join();

void direct_join_with_duplication_pipeline_write(uint64_t *d_row_offsets);
void direct_join_with_duplication_nomask_pipeline_write(uint64_t *d_row_offsets);
void direct_join_without_duplication_pipeline_write(uint64_t *d_row_offsets);

void join_count_nodup_dispatcher(
    // trie_inter
    vtype **vertex_arrays_, uint64_t **parent_pointers_,
    uint64_t num_rows, uint32_t num_cols, uint32_t join_col,
    // trie_edge
    vtype *edge_key_array_, vtype *edge_value_array_, uint32_t *edge_offsets_,
    // hash
    uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx,
    // mask
    uint32_t *d_num_res_for_each, uint32_t *d_st_array, uint32_t task_per_warp);

void join_write_nodup_dispatcher(
    uint64_t num_rows,
    // trie_edge
    vtype *edge_value_array_,
    uint32_t *d_num_res_for_each, uint32_t *d_st_array,
    uint32_t *d_res, uint64_t *d_offsets_of_, uint32_t num_tasks_per_warp, uint64_t *d_parent_res);

#endif
