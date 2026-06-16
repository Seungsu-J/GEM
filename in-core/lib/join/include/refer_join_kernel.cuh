#ifndef JOIN_REFER_JOIN_KERNEL_CUH
#define JOIN_REFER_JOIN_KERNEL_CUH

#include "defs.h"

template <typename T>
class JoinInfo;

template <int ChunkSize>
__global__ void
join_count_refer_kernel_nodup(
    // data trie
    vtype **vertex_arrays_, uint64_t **parent_pointers_,
    uint64_t num_rows, uint32_t num_cols, uint32_t join_col,
    // mask
    uint32_t *select_bool_vec_,
    // trie_edge
    vtype *edge_key_array_, vtype *edge_value_array_, uint32_t *edge_offsets_,
    // hash
    uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx,
    // mask
    uint32_t *d_num_res_for_each, uint32_t *d_st_array);

template <int ChunkSize>
__global__ void
join_count_refer_kernel_nodup_pipeline_info(
    // data trie (JoinInfo supports partial staging)
    JoinInfo<uint32_t> *data_info_arrays_, JoinInfo<uint64_t> *parent_info_arrays_,
    // selection mask (already windowed to this pipeline tile)
    uint32_t *select_bool_vec_,
    uint64_t pipeline_off_st, uint64_t pipeline_num_rows,
    uint32_t num_cols, uint32_t join_col,
    // trie_edge
    vtype *edge_key_array_, vtype *edge_value_array_, uint32_t *edge_offsets_,
    // hash
    uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx,
    // result
    uint32_t *d_num_res_for_each, uint32_t *d_st_array);

__global__ void
join_refer_count_dup_kernel(
    // data trie
    vtype **vertex_arrays_, uint64_t **parent_pointers_, uint32_t num_layers,
    // mask
    uint32_t *select_bool_vec_, uint64_t num_rows,
    uint32_t num_cols, uint32_t join_col,
    // trie_edge
    vtype *edge_key_array_, vtype *edge_value_array_, uint32_t *edge_offsets_,
    // hash
    uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx,
    // mask
    uint32_t *value_mask, uint32_t mask_length, uint32_t *d_st_array,
    // options
    int dup_check_type, uint32_t num_dup_cols, uint32_t *dup_cols);

__global__ void
join_refer_count_dup_kernel_2(
    // data trie
    vtype **vertex_arrays_, uint64_t **parent_pointers_,
    // mask
    uint32_t *select_bool_vec_, uint64_t num_rows,
    uint32_t num_cols, uint32_t join_col,
    // trie_edge
    vtype *edge_value_array_, uint32_t *edge_offsets_,
    // hash
    uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx,
    // mask
    uint32_t *value_mask, uint32_t mask_length, uint32_t *d_st_array,
    // options
    int dup_check_type, uint32_t num_dup_cols, uint32_t *dup_cols, uint32_t up_to_col);

__global__ void
join_refer_count_dup_kernel_2_pipeline_info(
    // data trie (JoinInfo supports partial staging)
    JoinInfo<uint32_t> *data_info_arrays_, JoinInfo<uint64_t> *parent_info_arrays_,
    // selection mask (already windowed to this pipeline tile)
    uint32_t *select_bool_vec_,
    uint64_t pipeline_off_st, uint64_t pipeline_num_rows,
    uint32_t num_cols, uint32_t join_col,
    // trie_edge
    vtype *edge_value_array_, uint32_t *edge_offsets_,
    // hash
    uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx,
    // mask and result
    uint32_t *value_mask, uint32_t mask_length, uint32_t *d_st_array,
    // options
    int dup_check_type, uint32_t num_dup_cols, uint32_t *dup_cols, uint32_t up_to_col);

__global__ void
join_refer_count_dup_nomask_kernel_2(
    // data trie
    vtype **vertex_arrays_, uint64_t **parent_pointers_,
    // mask
    uint32_t *select_bool_vec_, uint64_t num_rows,
    uint32_t num_cols, uint32_t join_col,
    // trie_edge
    vtype *edge_value_array_, uint32_t *edge_offsets_,
    // hash
    uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx,
    // outputs
    uint32_t *d_num_res_for_each, uint32_t *d_st_array,
    // options
    int dup_check_type, uint32_t num_dup_cols, uint32_t *dup_cols, uint32_t up_to_col);

__global__ void
join_refer_count_dup_nomask_kernel_2_pipeline_info(
    // data trie (JoinInfo supports partial staging)
    JoinInfo<uint32_t> *data_info_arrays_, JoinInfo<uint64_t> *parent_info_arrays_,
    // selection mask (already windowed to this pipeline tile)
    uint32_t *select_bool_vec_,
    uint64_t pipeline_off_st, uint64_t pipeline_num_rows,
    uint32_t num_cols, uint32_t join_col,
    // trie_edge
    vtype *edge_value_array_, uint32_t *edge_offsets_,
    // hash
    uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx,
    // outputs
    uint32_t *d_num_res_for_each, uint32_t *d_st_array,
    // options
    int dup_check_type, uint32_t num_dup_cols, uint32_t *dup_cols, uint32_t up_to_col);

__global__ void
join_refer_write_dup_nomask_pipeline(
    // row range
    uint64_t frag_offset, uint64_t num_frag_rows_pipeline,
    // data trie
    vtype **vertex_arrays_, uint64_t **parent_pointers_,
    uint32_t num_cols, uint32_t join_col,
    // trie_edge
    vtype *edge_value_array_, uint32_t *edge_offsets_,
    // hash
    uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx,
    // offsets and output
    uint64_t *d_row_offsets,
    uint32_t *d_res, uint64_t *d_parent_res, uint64_t result_offset,
    // options
    int dup_check_type, uint32_t num_dup_cols, uint32_t *dup_cols, uint32_t up_to_col);

__global__ void join_refer_write_dup_kernel(
    // data trie
    vtype **vertex_arrays_, uint64_t **parent_pointers_, uint32_t num_layers,
    uint64_t num_rows, uint32_t num_cols, uint32_t join_col,
    // trie_edge
    vtype *edge_key_array_, vtype *edge_value_array_, uint32_t *edge_offsets_,
    // hash
    uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx,
    // mask and result
    uint32_t *value_mask, uint32_t mask_length, uint32_t *d_st_array,
    uint32_t *d_res,
    uint64_t *d_row_offsets,
    uint64_t *d_parent_res,
    // options
    int dup_check_type, uint32_t num_dup_cols, uint32_t *dup_cols);

template <int ChunkSize>
__global__ void
join_write_refer_kernel_nodup(
    // trie_inter
    uint64_t *valid_row_ids, uint64_t num_valid_rows,
    // trie_edge
    vtype *edge_value_array_,
    uint32_t *d_selected_count, uint32_t *d_st_array,
    uint32_t *d_res, uint64_t *d_offsets_of_, uint64_t *d_parent_res);

__global__ void
join_write_refer_kernel_nodup_pipeline(
    // trie_inter
    uint64_t frag_offset, uint64_t num_valid_rows_pipeline,
    uint64_t *valid_row_ids,
    // trie_edge
    vtype *edge_value_array_,
    uint32_t *d_selected_count, uint32_t *d_st_array,
    // offsets and output
    uint64_t *d_offsets_of_,
    uint32_t *d_res, uint64_t *d_parent_res,
    // options
    uint64_t result_offset);
#endif
