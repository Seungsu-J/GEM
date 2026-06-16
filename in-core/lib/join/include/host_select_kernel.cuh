#ifndef JOIN_SELECT_KERNEL_CUH
#define JOIN_SELECT_KERNEL_CUH

#include "defs.h"
#include "join_context.cuh"

// __global__ void select_trie_kernel_hybrid(
//     // frag
//     vtype *frag_key_data, vtype *frag_value_data_,
//     uint64_t **parent_pointers_, uint32_t num_cols, uint32_t key_col, uint32_t val_col,
//     uint64_t frag_num_rows,
//     // edge
//     vtype *edge_key_array_, vtype *edge_value_array_, uint32_t *edge_offsets_,
//     // hash
//     uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx,
//     // result
//     uint32_t *bool_vec);

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

__global__ void host_atomicAdd(uint64_t *addr, uint64_t *val);

__global__ void select_trie_kernel_pipeline(
    // frag
    vtype *frag_key_data_, vtype *frag_value_data_,
    uint64_t **parent_pointers_, uint32_t num_cols,
    uint32_t key_col, uint32_t val_col,
    uint64_t pipeline_off_st, uint64_t pipeline_num_rows,
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
    vtype *edge_key_array_, vtype *edge_value_array_, uint32_t *edge_offsets_,
    // hash
    uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx,
    // result
    uint32_t *bool_vec);

__global__ void select_trie_refer_kernel_pipeline(
    // original data to refer to
    vtype *frag_key_data_, vtype *frag_value_data_,
    uint64_t **parent_pointers_, uint32_t num_cols,
    uint32_t key_col, uint32_t val_col,
    // selection mask
    uint32_t *select_bool_vec_,
    uint64_t pipeline_off_st, uint64_t pipeline_num_rows,
    // trie_edge
    vtype *edge_key_array_, vtype *edge_value_array_, uint32_t *edge_offsets_,
    // hash
    uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx,
    // result
    uint32_t *bool_vec);

// True input-pipeline kernels: use JoinInfo offset/pointers for segmented columns.
__global__ void select_trie_kernel_pipeline_info(
    // frag (key/value column data)
    JoinInfo<uint32_t> frag_key_info, JoinInfo<uint32_t> frag_value_info,
    // parents per column
    JoinInfo<uint64_t> *parent_info_arrays, uint32_t num_cols,
    uint32_t key_col, uint32_t val_col,
    uint64_t pipeline_off_st, uint64_t pipeline_num_rows,
    // trie_edge
    vtype *edge_key_array_, vtype *edge_value_array_, offtype *edge_offsets_,
    // hash
    uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx,
    // result
    uint32_t *bool_vec);

__global__ void select_trie_refer_kernel_pipeline_info(
    // original data to refer to (key/value column data)
    JoinInfo<uint32_t> frag_key_info, JoinInfo<uint32_t> frag_value_info,
    // parents per column
    JoinInfo<uint64_t> *parent_info_arrays, uint32_t num_cols,
    uint32_t key_col, uint32_t val_col,
    // selection mask (already windowed to this pipeline tile)
    uint32_t *select_bool_vec_,
    uint64_t pipeline_off_st, uint64_t pipeline_num_rows,
    // trie_edge
    vtype *edge_key_array_, vtype *edge_value_array_, uint32_t *edge_offsets_,
    // hash
    uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx,
    // result
    uint32_t *bool_vec);

// Debug kernels
__global__ void select_trie_kernel_debug(
    // frag
    vtype *frag_key_data_, vtype *frag_value_data_,
    uint64_t **parent_pointers_, uint32_t num_cols,
    uint32_t key_col, uint32_t val_col,
    uint64_t frag_num_rows,
    // edge
    vtype *edge_key_array_, vtype *edge_value_array_, uint32_t *edge_offsets_,
    // hash
    uint32_t num_buckets, uint32_t edge_offset, uint32_t directed_edge_idx,
    // result
    uint32_t *bool_vec, uint64_t *num_result, uint32_t *d_res_debug);

#endif
