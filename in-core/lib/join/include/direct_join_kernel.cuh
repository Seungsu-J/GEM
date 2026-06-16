#ifndef JOIN_DIRECT_JOIN_KERNEL_CUH
#define JOIN_DIRECT_JOIN_KERNEL_CUH

#include "defs.h"

template <typename T>
class JoinInfo;

template <int ChunkSize>
__global__ void join_count_kernel_nodup(
    // trie_inter
    vtype **vertex_arrays_, uint64_t **parent_pointers_,
    uint64_t num_rows, uint32_t num_cols, uint32_t join_col,
    // trie_edge
    vtype *edge_key_array_, vtype *edge_value_array_, uint32_t *edge_offsets_,
    // hash
    uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx,
    // mask
    uint32_t *d_num_res_for_each, uint32_t *d_st_array);

template <int ChunkSize>
__global__ void join_count_kernel_nodup_pipeline_info(
    // trie_inter (JoinInfo supports partial staging)
    JoinInfo<uint32_t> *data_info_arrays_, JoinInfo<uint64_t> *parent_info_arrays_,
    uint64_t pipeline_off_st, uint64_t pipeline_num_rows,
    uint32_t num_cols, uint32_t join_col,
    // trie_edge
    vtype *edge_key_array_, vtype *edge_value_array_, uint32_t *edge_offsets_,
    // hash
    uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx,
    // mask
    uint32_t *d_num_res_for_each, uint32_t *d_st_array);

template <int ChunkSize>
__global__ void join_count_kernel_nodup_pipeline_local(
    // trie_inter (JoinInfo supports partial staging)
    JoinInfo<uint32_t> *data_info_arrays_, JoinInfo<uint64_t> *parent_info_arrays_,
    uint64_t pipeline_off_st, uint64_t pipeline_num_rows,
    uint32_t num_cols, uint32_t join_col,
    // trie_edge
    vtype *edge_key_array_, vtype *edge_value_array_, uint32_t *edge_offsets_,
    // hash
    uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx,
    // outputs (chunk-local)
    uint32_t *d_num_res_for_each, uint32_t *d_st_array);

__global__ void
join_count_kernel_dup(
    // trie_inter
    vtype **vertex_arrays_, uint64_t **parent_pointers_,
    uint64_t num_rows, uint32_t num_cols, uint32_t join_col,
    // trie_edge
    vtype *edge_key_array_, vtype *edge_value_array_, uint32_t *edge_offsets_,
    // hash
    uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx,
    // mask
    uint32_t *value_mask, uint32_t mask_length, uint32_t *d_st_array,
    // options
    int dup_check_type, uint32_t num_dup_cols, uint32_t *dup_cols);

__global__ void
join_count_kernel_dup_2(
    // trie_inter
    vtype **vertex_arrays_, uint64_t **parent_pointers_,
    uint64_t num_rows, uint32_t num_cols, uint32_t join_col,
    // trie_edge
    vtype *edge_key_array_, vtype *edge_value_array_, uint32_t *edge_offsets_,
    // hash
    uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx,
    // mask
    uint32_t *value_mask, uint32_t mask_length, uint32_t *d_st_array,
    // options
    int dup_check_type, uint32_t num_dup_cols, uint32_t *dup_cols, uint32_t up_to_col);

__global__ void
join_count_kernel_dup_pipeline_info(
    // trie_inter (JoinInfo supports partial staging)
    JoinInfo<uint32_t> *data_info_arrays_, JoinInfo<uint64_t> *parent_info_arrays_,
    uint64_t pipeline_off_st, uint64_t pipeline_num_rows,
    uint32_t num_cols, uint32_t join_col,
    // trie_edge
    vtype *edge_key_array_, vtype *edge_value_array_, uint32_t *edge_offsets_,
    // hash
    uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx,
    // mask
    uint32_t *value_mask, uint32_t mask_length, uint32_t *d_st_array,
    // options
    int dup_check_type, uint32_t num_dup_cols, uint32_t *dup_cols, uint32_t up_to_col);

template <int ChunkSize>
__global__ void join_write_kernel_nodup(
    // trie_inter
    uint64_t num_rows,
    // trie_edge
    vtype *edge_value_array_,
    uint32_t *d_num_res_for_each, uint32_t *d_st_array,
    uint32_t *d_res, uint64_t *d_offsets_of_, uint64_t *d_parent_res);

__global__ void
join_write_kernel_dup(
    // trie_inter
    uint64_t num_rows,
    // trie_edge
    vtype *edge_value_array_,
    uint32_t *value_mask, uint32_t mask_length,
    uint64_t *d_offsets_of_, uint32_t *d_st_array,
    uint32_t *d_res, uint64_t *d_parent_res);

__global__ void
join_write_kernel_dup_pipeline(
    // trie_inter
    uint64_t frag_offset, uint64_t num_frag_rows_pipeline,
    // trie_edge
    vtype *edge_value_array_,
    uint32_t *value_mask, uint32_t mask_length,
    uint64_t *d_offsets_of_, uint32_t *d_st_array, // these two resident in GPU.
    uint32_t *d_res, uint64_t *d_parent_res, uint64_t result_offset);

__global__ void
join_write_kernel_nodup_pipeline(
    // trie_inter
    uint64_t frag_offset, uint64_t num_frag_rows_pipeline,
    // trie_edge
    vtype *edge_value_array_,
    uint32_t *d_st_array, uint64_t *d_offsets_of_,
    uint32_t *d_res, uint64_t *d_parent_res, uint64_t result_offset);

__global__ void
join_write_kernel_nodup_pipeline_local(
    // row range (global row ids)
    uint64_t frag_offset, uint64_t num_frag_rows_pipeline,
    // trie_edge
    vtype *edge_value_array_,
    // chunk-local st and offsets for this fragment range
    uint32_t *d_st_array, uint64_t *d_row_offsets, uint64_t base_result_offset,
    // output buffers (0-based for this fragment range)
    uint32_t *d_res, uint64_t *d_parent_res);

__global__ void
join_count_kernel_dup_nomask_2(
    // trie_inter
    vtype **vertex_arrays_, uint64_t **parent_pointers_,
    uint64_t num_rows, uint32_t num_cols, uint32_t join_col,
    // trie_edge
    vtype *edge_key_array_, vtype *edge_value_array_, uint32_t *edge_offsets_,
    // hash
    uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx,
    // outputs
    uint32_t *d_num_res_for_each, uint32_t *d_st_array,
    // options
    int dup_check_type, uint32_t num_dup_cols, uint32_t *dup_cols, uint32_t up_to_col);

__global__ void
join_write_kernel_dup_nomask_pipeline(
    // row range
    uint64_t frag_offset, uint64_t num_frag_rows_pipeline,
    // trie_inter
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

__global__ void
join_count_kernel_dup_nomask_pipeline_local_info(
    // trie_inter (JoinInfo supports partial staging)
    JoinInfo<uint32_t> *data_info_arrays_, JoinInfo<uint64_t> *parent_info_arrays_,
    uint64_t pipeline_off_st, uint64_t pipeline_num_rows,
    uint32_t num_cols, uint32_t join_col,
    // trie_edge
    vtype *edge_value_array_, uint32_t *edge_offsets_,
    // hash
    uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx,
    // outputs (chunk-local)
    uint32_t *d_num_res_for_each,
    // options
    int dup_check_type, uint32_t num_dup_cols, uint32_t *dup_cols, uint32_t up_to_col);

__global__ void
join_write_kernel_dup_nomask_pipeline_local_info(
    // row range (global row ids)
    uint64_t frag_offset, uint64_t num_frag_rows_pipeline,
    // trie_inter (JoinInfo supports partial staging)
    JoinInfo<uint32_t> *data_info_arrays_, JoinInfo<uint64_t> *parent_info_arrays_,
    uint32_t num_cols, uint32_t join_col,
    // trie_edge
    vtype *edge_value_array_, uint32_t *edge_offsets_,
    // hash
    uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx,
    // chunk-local offsets and output (0-based)
    uint64_t *d_row_offsets, uint64_t base_result_offset,
    uint32_t *d_res, uint64_t *d_parent_res,
    // options
    int dup_check_type, uint32_t num_dup_cols, uint32_t *dup_cols, uint32_t up_to_col);

__global__ void join_write_kernel_dup_debug(
    // trie_inter
    uint64_t num_rows,
    // trie_edge
    vtype *edge_value_array_,
    uint32_t *value_mask, uint32_t mask_length,
    uint64_t *d_offsets_of_, uint32_t *d_st_array,
    uint32_t *d_res, uint64_t *d_parent_res,
    // for debug
    uint64_t *num_processed, uint64_t num_total_res);

#endif
