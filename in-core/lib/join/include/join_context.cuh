#ifndef JOIN_JOIN_CONTEXT_CUH
#define JOIN_JOIN_CONTEXT_CUH

#include "join_helpers.h"

template <typename T>
class JoinInfo
{
  static_assert(std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>,
                "JoinInfo only supports uint32_t and uint64_t types.");

public:
  std::uint64_t offset = 0;
  T *array_ptr = nullptr;
  uint32_t is_partial = 0;

private:
  uint32_t pad; // for alignment (pad + is_partial = 8 bytes)
};

class JoinContext
{
public:
  // for join
  cpuGraph *hq;
  UnifiedTrie *frag_trie, *probe_trie, *target_trie, *root_trie;
  uint32_t col_key, col_value;
  uint32_t *d_frag_key_array_, *d_frag_value_array_;
  uint32_t *d_edge_keys_, *d_edge_values_;
  offtype *d_edge_offsets_;
  bool edge_forward;
  etype directed_edge_idx;

  // non pipeline
  vtype **d_data_arrays_;
  uint64_t **d_parent_arrays_;
  // pipeline
  std::vector<JoinInfo<uint32_t>> h_data_info_arrays_;
  std::vector<JoinInfo<uint64_t>> h_parent_info_arrays_;
  JoinInfo<uint32_t> *d_data_info_arrays_;
  JoinInfo<uint64_t> *d_parent_info_arrays_;
  std::vector<uint32_t> partial_data_cols;
  std::vector<uint32_t> partial_parent_cols;

  uint64_t num_frag_rows;
  uint32_t num_frag_cols, num_root_cols;
  uint32_t *select_bool_mask_; // used as refer_join&refer_select mask.
  uint32_t *result_mask_;      // used for selection_results for direct_select&refer_select.

  uint32_t *d_dup_cols, num_dup_cols, *h_dup_cols, dup_check_type;
  bool skip_final_materialization;
  uint32_t task_per_warp;

  cudaStream_t join_stream;
  const static int nStreams = 3;
  cudaStream_t pipeline_streams[nStreams];
  HashLookupTables *d_lookup_;

  // uint32_t *d_key_array_;
  // uint32_t *d_value_array_;
  HashTable *hash_table_;

  // for process
  bool found[2];
  uint32_t col[2];
  bool is_subset;

  std::vector<uint32_t> pin_cols;
  uint32_t up_to_col;

  // temporal variables
  uint32_t *d_num_res_for_each;
  uint32_t *d_st_array;
  uint64_t *h_num_res;
  uint32_t *d_mask;
  uint32_t mask_length;

public:
  static JoinContext &getInstance()
  {
    static JoinContext instance;
    return instance;
  }
  void init_tries(UnifiedTrie *frag, UnifiedTrie *probe, UnifiedTrie *target);
  void clear();

  void join_context();
  void cleanup() noexcept;

  void process_context();
  void process_is_subset_context();
  void process_direct_join_context();
  void process_refer_join_context();

  void host_select_general_context();
  bool direct_select_context();
  void direct_select_pipeline_context();
  bool refer_select_context();
  void refer_select_pipeline_context();

  bool direct_join_general_context();
  void direct_join_without_duplication_context();
  void direct_join_with_duplication_context();

  bool refer_join_general_context();
  void refer_join_without_duplication_context();
  void refer_join_with_duplication_context();
};

void get_parent_arrays_pipeline(
    uint32_t trie_id,
    uint32_t up_to_col,
    std::vector<JoinInfo<uint64_t>> &parent_info_arrays);

void get_data_arrays_pipeline(
    uint32_t trie_id,
    std::vector<uint32_t> &pin_cols,
    std::vector<JoinInfo<uint32_t>> &data_info_arrays);

#endif
