#ifndef JOIN_REFER_JOIN_CUH
#define JOIN_REFER_JOIN_CUH

#include "join_helpers.h"

void refer_join();
void join_refer_write_dup_pipeline(uint64_t *d_row_offsets);
void join_refer_write_nodup_pipeline(
    uint64_t *d_selected_row_ids, uint64_t num_selected_rows,
    uint32_t *d_selected_count, uint64_t *d_row_offsets);

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
#endif
