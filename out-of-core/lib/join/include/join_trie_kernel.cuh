#ifndef JOIN_JOIN_TRIE_KERNEL_CUH
#define JOIN_JOIN_TRIE_KERNEL_CUH

#include "defs.h"

#include "unifiedTrie.cuh"

extern __device__ __constant__ HashLookupTables *lookup;



__global__ void select_trie_kernel_hybrid(
    // frag
    vtype *frag_key_data, vtype *frag_value_data_,
    uint64_t **parent_pointers_, uint32_t num_cols, uint32_t key_col, uint32_t val_col,
    uint64_t frag_num_rows,
    // edge
    vtype *edge_key_array_, vtype *edge_value_array_, offtype *edge_offsets_,
    // hash
    uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx,
    // result
    uint32_t *bool_vec);

__global__ void select_trie_kernel(
    // frag
    vtype *frag_key_data_, vtype *frag_value_data_,
    uint64_t **parent_pointers_, uint32_t num_cols,
    uint32_t key_col, uint32_t val_col, // make sure val_col > key_col
    uint64_t frag_num_rows,
    // edge
    vtype *edge_key_array_, vtype *edge_value_array_, offtype *edge_offsets_,
    // hash
    uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx,
    // result
    uint32_t *bool_vec);

__global__ void select_trie_refer_kernel(
    // original data to refer to
    vtype *d_original_frag_key_data_, vtype *d_original_frag_value_data_,
    uint64_t **d_original_parent_pointers_, uint32_t num_cols,
    uint32_t key_col, uint32_t val_col,
    // selection mask
    uint32_t *select_bool_vec_, uint64_t frag_num_rows,
    // trie_edge
    vtype *edge_key_array_, vtype *edge_value_array_, offtype *edge_offsets_,
    // hash
    uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx,
    // result
    uint32_t *bool_vec);

template <int ChunkSize>
__global__ void join_count_kernel_nodup(
    // trie_inter
    vtype **vertex_arrays_, uint64_t **parent_pointers_,
    uint64_t num_rows, uint32_t num_cols, uint32_t join_col,
    // trie_edge
    vtype *edge_key_array_, vtype *edge_value_array_, offtype *edge_offsets_,
    // hash
    uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx,
    // mask
    uint32_t *d_num_res_for_each, uint32_t *d_st_array);

__global__ void
join_count_kernel_dup(
    // trie_inter
    vtype **vertex_arrays_, uint64_t **parent_pointers_,
    uint64_t num_rows, uint32_t num_cols, uint32_t join_col,
    // trie_edge
    vtype *edge_key_array_, vtype *edge_value_array_, offtype *edge_offsets_,
    // hash
    uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx,
    // mask
    uint32_t *value_mask, uint32_t mask_length, uint32_t *d_st_array,
    // options
    int dup_check_type, uint32_t num_dup_cols, uint32_t *dup_cols);

template <int ChunkSize>
__global__ void
join_write_kernel_nodup(
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

template <int ChunkSize>
__global__ void
join_count_refer_kernel_nodup(
    // data trie
    vtype **vertex_arrays_, uint64_t **parent_pointers_,
    uint64_t num_rows, uint32_t num_cols, uint32_t join_col,
    // mask
    uint32_t *select_bool_vec_,
    // trie_edge
    vtype *edge_key_array_, vtype *edge_value_array_, offtype *edge_offsets_,
    // hash
    uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx,
    // mask
    uint32_t *d_num_res_for_each, uint32_t *d_st_array);

__global__ void
join_refer_count_dup_kernel(
    // data trie
    vtype **vertex_arrays_, uint64_t **parent_pointers_, uint32_t num_layers,
    // mask
    uint32_t *select_bool_vec_, uint64_t num_rows,
    uint32_t num_cols, uint32_t join_col,
    // trie_edge
    vtype *edge_key_array_, vtype *edge_value_array_, offtype *edge_offsets_,
    // hash
    uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx,
    // mask
    uint32_t *value_mask, uint32_t mask_length, uint32_t *d_st_array,
    // options
    int dup_check_type, uint32_t num_dup_cols, uint32_t *dup_cols);

__global__ void
join_refer_write_dup_kernel(
    // data trie
    vtype **vertex_arrays_, uint64_t **parent_pointers_, uint32_t num_layers,
    uint64_t num_rows, uint32_t num_cols, uint32_t join_col,
    // trie_edge
    vtype *edge_key_array_, vtype *edge_value_array_, offtype *edge_offsets_,
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

// Debug kernels
__global__ void
select_trie_kernel_debug(
    // frag
    vtype *frag_key_data_, vtype *frag_value_data_,
    uint64_t **parent_pointers_, uint32_t num_cols,
    uint32_t key_col, uint32_t val_col,
    uint64_t frag_num_rows,
    // edge
    vtype *edge_key_array_, vtype *edge_value_array_, offtype *edge_offsets_,
    // hash
    uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx,
    // result
    uint32_t *bool_vec, uint64_t *num_result, uint32_t *d_res_debug);

__global__ void
join_write_kernel_dup_debug(
    // trie_inter
    uint64_t num_rows,
    // trie_edge
    vtype *edge_value_array_,
    uint32_t *value_mask, uint32_t mask_length,
    uint64_t *d_offsets_of_, uint32_t *d_st_array,
    uint32_t *d_res, uint64_t *d_parent_res,
    // for debug
    uint64_t *num_processed, uint64_t num_total_res);

__global__ void print_trie_kernel(
    // trie_inter
    vtype **vertex_arrays_, uint64_t **parent_pointers_,
    uint64_t num_rows, uint32_t num_cols,
    // for debug
    uint32_t *output_buffer, uint64_t *output_count);

__global__ void
naiive_count(uint32_t *d_bool_vec, uint32_t num_compressed_blocks, uint32_t *debug_mask_bits);

__global__ void
retrive_valid_parents_kernel(
    uint64_t *parent_array, uint64_t num_rows, uint32_t *data_array, uint32_t *res);

__global__ void
check_same_kernel(
    uint32_t *data_array_1, uint32_t *data_array_2, uint32_t num_rows, uint32_t *res);

__global__ void
extract_edge_kernel(
    uint32_t *d_key_array, uint32_t *d_value_array,
    uint32_t num_rows, uint32_t num_cols,
    uint32_t key_col, uint32_t value_col,
    uint64_t **d_parent_arrays,
    uint32_t *d_res);

__global__ void
convert_ull_to_uint32(
    uint32_t *d_count_for_each, uint32_t *d_count_for_each_32, uint32_t num_items);

__global__ void
gather_counts_kernel(
    uint32_t *d_count_src, uint64_t *d_indices, uint64_t num_indices, uint32_t *d_count_dst);

__global__ void
collect_res_for_each(
    uint32_t *d_scan_res, uint32_t num_rows, uint32_t scan_length, uint32_t *d_res_for_each);

#endif //! JOIN_JOIN_TRIE_KERNEL_CUH
