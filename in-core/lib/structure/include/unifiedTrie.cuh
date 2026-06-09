#ifndef STRUCTURE_UNIFIEDTRIE_H
#define STRUCTURE_UNIFIEDTRIE_H

#include "defs.h"
#include "globals.h"
#include "cpuGraph.h"
#include "gpuGraph.h"

#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <memory>
#include <mutex>

class Lattice;
class MemoryManager;

class HashTable
{
public:
  static HashTable &getInstance();
  static void initialize();
  static void shutdown();

  // hash table data
  uint32_t *d_keys_[NUM_TABLES];
  uint32_t *d_values_[NUM_TABLES]; // not used.
  uint32_t hash_constants_[(MAX_EQ * 2) * NUM_TABLES * 2];
  uint32_t *d_hash_constants_; // the same size as hash_constants_
  uint32_t *d_pos_in_org[NUM_TABLES];

  uint32_t *d_buffer_;
  uint32_t *d_hash_table_offs_;
  uint32_t *d_num_buckets_;

  // for temporal mask length
  std::vector<uint32_t> max_values_per_edge_;

  // for the 0-th layer
  offtype *d_offsets_;                      // device memory: flat array of all edge offsets
  std::vector<offtype> edge_offset_starts_; // host: starting position of each edge's offsets

  // host copy
  uint32_t *h_buffer;
  // uint32_t *h_num_candidates_;
  uint32_t *h_num_buckets_;
  uint32_t *h_hash_table_offs_;

public:
  HashTable(const HashTable &) = delete;
  HashTable(HashTable &&) = delete;
  HashTable &operator=(const HashTable &) = delete;
  HashTable &operator=(HashTable &&) = delete;
  ~HashTable();

  void init(uint32_t num_edges);
  void clear();
  void build(cpuGraph *hq,
             numtype *h_num_u_candidate_vs_,
             vtype *d_u_candidate_vs_,
             numtype *d_num_u_candidate_vs_);

private:
  HashTable();
  static std::unique_ptr<HashTable> instance;
  static std::mutex instance_mutex;
};

struct HashLookupTables
{
  uint32_t *hash_constants;
  uint32_t *keys[NUM_TABLES];
  uint32_t *pos[NUM_TABLES];
};

extern __device__ __constant__ HashLookupTables *lookup;
__device__ __host__ __forceinline__ uint32_t edge_hash_function(uint32_t key,
                                                                uint32_t C0, uint32_t C1,
                                                                uint32_t num_buckets)
{
  return (C0 ^ key + C1) % num_buckets;
}

__device__ __forceinline__ uint32_t hash_search_in_kvtrie(uint32_t *hash_constants, // c + offsets
                                                          uint32_t num_buckets, uint32_t edge_offset,
                                                          uint32_t *d_hash_keys_[2], // d_keys + offsets
                                                          uint32_t *d_pos_in_org[2], // org + offsets
                                                          vtype search_key)
{
  for (int table_id = 0; table_id < 2; table_id++)
  {
    uint32_t C0 = hash_constants[table_id * 2 + 0];
    uint32_t C1 = hash_constants[table_id * 2 + 1];
    uint32_t bucket_idx = edge_hash_function(search_key, C0, C1, num_buckets);

    uint32_t *bucket = d_hash_keys_[table_id] + bucket_idx * BUCKET_DIM;
    for (uint32_t i = 0; i < BUCKET_DIM; ++i)
    {
      if (bucket[i] == search_key)
      {
        return d_pos_in_org[table_id][bucket_idx * BUCKET_DIM + i]; // index in CSR key array
      }
    }
  }
  return UINT32_MAX; // Not Found.
}

__device__ __forceinline__ uint32_t hash_lookup(
    uint32_t num_buckets, uint32_t edge_offset,
    uint32_t directed_edge_idx, // for constant
    vtype search_key)
{
  uint32_t *var_hash_constants = lookup->hash_constants + directed_edge_idx * NUM_TABLES * 2;
#ifndef NDEBUG
  uint32_t off = directed_edge_idx * NUM_TABLES * 2;
  if (off >= MAX_EQ * 2 * NUM_TABLES * 2)
  {
    printf("Error: hash_lookup out of bounds access: %u >= %u\n", off, MAX_EQ * 2 * NUM_TABLES * 2);
  }
#endif
  uint32_t *var_keys[2] = {lookup->keys[0] + edge_offset, lookup->keys[1] + edge_offset};
  uint32_t *var_pos[2] = {lookup->pos[0] + edge_offset, lookup->pos[1] + edge_offset};
  return hash_search_in_kvtrie(var_hash_constants, num_buckets,
                               edge_offset, var_keys, var_pos, search_key);
}

class DataBlock
{
public:
  size_t size_in_byte; // size of the block

  uint32_t *d_data;    // column data or mask.
  uint64_t *d_parents; // parent indices

  uint64_t num_rows;              // real number(for both)
  uint64_t num_compressed_blocks; // only for mask
  // see if this block is a mask by checking num_compressed_blocks > 0

public:
  DataBlock();
  ~DataBlock();

  void init(uint64_t num_rows, bool is_mask = false);
  void allocate();
  void deallocate();
};

class UnifiedTrie
{
public:
  // my info
  uint32_t id;                               // inter_id
  ettype et;                                 // edge tag
  uint32_t num_layers;                       // #edges + 1 = #expansion + #mask <==> layers
  uint32_t num_cols;                         // #vertices <==> data columns
  std::shared_ptr<vtype[]> mapped_query_us_; // query vertices.
  uint64_t num_results;

  // chain relations
  uint32_t mask_parent_id;
  uint32_t expansion_parent_id;
  std::vector<uint32_t> layer_ids;  // All layers including masks
  std::vector<uint32_t> column_ids; // Only expansion layers (actual data columns)

  // data
  DataBlock *data;   // one column of data.
  DataBlock *r_data; // reversed data, only for edge.

  // for one-edge tries only. (Edge Attributes)
  vtype edge_vertices[2]; // u, v. ascending order.

public:
  UnifiedTrie();
  ~UnifiedTrie();

  // Initialization
  void init(uint32_t trie_id, ettype edge_tag, uint32_t levels, uint32_t cols);
  void setQueryVertices(vtype *query_vertices);
  void setQueryVerticesExpand(const UnifiedTrie *other, uint32_t new_u);
  void setQueryVerticesMask(const UnifiedTrie *other);
  void setLayers(const std::vector<uint32_t> &layers);
  void setLayersFrom(const UnifiedTrie *other, uint32_t new_layer);
  void setColumnsFrom(const UnifiedTrie *other, uint32_t new_column, bool is_expansion);

  // Parent relationships
  void setMaskParent(uint32_t parent_id);
  void setExpansionParent(uint32_t parent_id);
  void addLayer(uint32_t layer_id);

  // Data management
  void allocateData(uint64_t num_rows, bool is_mask = false);
  void allocateDataMemMgr(uint32_t id, uint64_t num_rows, bool is_mask = false);
  void deallocateData();
  void updateNumResults(uint64_t count);

  // Query methods
  bool hasMaskParent() const;
  bool hasExpansionParent() const;
  uint32_t getNumLayers() const;
  uint32_t getNumChildren() const;
  bool isEmpty() const;
};

class UnifiedTrieManager
{
public:
  static UnifiedTrieManager &getInstance();
  static void initialize();
  static void shutdown();

  std::vector<UnifiedTrie *> tries; // one trie - one lattice node
  uint32_t num_tries;

  // relation info
  std::vector<std::vector<vtype *>> parent_addr_list; // device_pointers
  std::vector<std::vector<vtype *>> data_addr_list;   // device_pointers

public:
  UnifiedTrieManager(const UnifiedTrieManager &) = delete;
  UnifiedTrieManager(UnifiedTrieManager &&) = delete;
  UnifiedTrieManager &operator=(const UnifiedTrieManager &) = delete;
  UnifiedTrieManager &operator=(UnifiedTrieManager &&) = delete;
  ~UnifiedTrieManager();

  // Initialization
  void init(uint32_t var_num_tries);
  void clear();

  // Trie management
  void initVertexTries(vtype *d_u_candidate_vs_, numtype *h_num_u_candidate_vs_);
  UnifiedTrie *createTrie(uint32_t trie_id, ettype edge_tag, uint32_t levels, uint32_t cols);
  UnifiedTrie *getTrie(uint32_t trie_id);
  void removeTrie(uint32_t trie_id);

  // relation management
  void linkMask(uint32_t target_id, uint32_t parent_id);
  void linkExpansion(uint32_t target_id, uint32_t parent_id, uint32_t num_new_rows, vtype new_u);
  std::vector<uint64_t *> &get_parent_arrays(uint32_t trie_id);
  std::vector<vtype *> &get_data_arrays(uint32_t trie_id);

  uint32_t getNumTries() const;

  // compute
  void construct_edge_candidates(
      cpuGraph *hq, cpuGraph *hg, gpuGraph *dg,
      vtype *d_u_candidate_vs_, numtype *d_num_u_candidate_vs_, numtype *h_num_u_candidate_vs_,
      uint32_t *d_bitmap, uint32_t bitmap_pitch);
  uint32_t find_best_edge_candidate(
      const ettype &cur_et = ettype());

  // Memory management
  void allocateAllData();
  void deallocateAllData();
  size_t getTotalMemoryUsage() const;

  // Debug and validation
  void printStats() const;
  void validate() const;

private:
  UnifiedTrieManager();
  static std::unique_ptr<UnifiedTrieManager> instance;
  static std::mutex instance_mutex;
};

#endif
