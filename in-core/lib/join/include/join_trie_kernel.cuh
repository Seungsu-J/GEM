#ifndef JOIN_JOIN_TRIE_KERNEL_CUH
#define JOIN_JOIN_TRIE_KERNEL_CUH

#include "defs.h"

#include "unifiedTrie.cuh"

extern __device__ __constant__ HashLookupTables *lookup;








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
    uint32_t *data_array_1, uint32_t *data_array_2, uint64_t num_rows, uint32_t *res);

__global__ void
extract_edge_kernel(
    uint32_t *d_key_array, uint32_t *d_value_array,
    uint64_t num_rows, uint32_t num_cols,
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
    uint32_t *d_scan_res, uint64_t num_rows, uint32_t scan_length, uint32_t *d_res_for_each);

#endif //! JOIN_JOIN_TRIE_KERNEL_CUH
