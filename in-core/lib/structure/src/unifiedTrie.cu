#include "unifiedTrie.cuh"
#include "cuda_helpers.cuh"
#include "cpuGraph.h"
#include "gpuGraph.h"
#include "cuckooHash.cuh"
#include "construct_kernel.cuh"
#include "lattice.h"
#include "memory_manager.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <random>
#include <cub/cub.cuh>

#include <chrono>
#include <fstream>

/* =========================================================================================================== */
/* =========================================================================================================== */
/* ================================================ HashTable ================================================ */
/* =========================================================================================================== */
/* =========================================================================================================== */

__device__ __constant__ HashLookupTables *lookup;
// __device__ __constant__ HashLookuptables lookup;

std::unique_ptr<HashTable> HashTable::instance = nullptr;
std::mutex HashTable::instance_mutex;

HashTable &HashTable::getInstance()
{
  std::lock_guard<std::mutex> lock(instance_mutex);
  if (!instance)
  {
    instance.reset(new HashTable());
  }
  return *instance;
}

void HashTable::initialize()
{
  std::lock_guard<std::mutex> lock(instance_mutex);
  instance.reset(new HashTable());
}

void HashTable::shutdown()
{
  std::lock_guard<std::mutex> lock(instance_mutex);
  instance.reset();
}

__global__ void populate_pos_in_org_kernel(
    const vtype *candidates,
    uint32_t num_candidates,
    uint32_t num_buckets,
    uint32_t c0, uint32_t c1,
    uint32_t c2, uint32_t c3,
    const uint32_t *keys_table0,
    const uint32_t *keys_table1,
    uint32_t *pos_table0,
    uint32_t *pos_table1)
{
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_candidates || num_buckets == 0)
    return;

  vtype key = candidates[idx];
  uint32_t total_slots = num_buckets * BUCKET_DIM;

  uint32_t bucket = edge_hash_function(key, c0, c1, num_buckets);
  uint32_t bucket_start = bucket * BUCKET_DIM;

  for (uint32_t i = 0; i < BUCKET_DIM; ++i)
  {
    uint32_t pos = bucket_start + i;
    if (pos >= total_slots)
      break;
    if (keys_table0[pos] == key)
    {
      pos_table0[pos] = idx;
      return;
    }
  }

  bucket = edge_hash_function(key, c2, c3, num_buckets);
  bucket_start = bucket * BUCKET_DIM;

  for (uint32_t i = 0; i < BUCKET_DIM; ++i)
  {
    uint32_t pos = bucket_start + i;
    if (pos >= total_slots)
      break;
    if (keys_table1[pos] == key)
    {
      pos_table1[pos] = idx;
      return;
    }
  }
}

HashTable::HashTable()
{
  for (int i = 0; i < NUM_TABLES; ++i)
  {
    d_keys_[i] = nullptr;
    d_values_[i] = nullptr;
    d_pos_in_org[i] = nullptr;
  }
  d_num_buckets_ = nullptr;
  d_hash_table_offs_ = nullptr;
  d_offsets_ = nullptr;
  max_values_per_edge_.resize(NUM_EQ * 2);
}

HashTable::~HashTable() { clear(); }

void HashTable::init(uint32_t num_edges)
{
}

void HashTable::clear()
{
  for (int i = 0; i < NUM_TABLES; ++i)
  {
    if (d_keys_[i])
    {
      cudaFree(d_keys_[i]);
      d_keys_[i] = nullptr;
    }
    if (d_values_[i])
    {
      cudaFree(d_values_[i]);
      d_values_[i] = nullptr;
    }
    if (d_pos_in_org[i])
    {
      cudaFree(d_pos_in_org[i]);
      d_pos_in_org[i] = nullptr;
    }
  }

  if (d_buffer_)
  {
    cudaFree(d_buffer_);
    d_buffer_ = nullptr;
    d_num_buckets_ = nullptr;
    d_hash_table_offs_ = nullptr;
  }

  if (d_hash_constants_)
  {
    cudaFree(d_hash_constants_);
    d_hash_constants_ = nullptr;
  }

  if (d_offsets_)
  {
    cudaFree(d_offsets_);
    d_offsets_ = nullptr;
  }

  if (h_buffer)
  {
    delete[] h_buffer;
    h_buffer = nullptr;
    h_num_buckets_ = nullptr;
    h_hash_table_offs_ = nullptr;
  }

  edge_offset_starts_.clear();
}

void HashTable::build(cpuGraph *hq,
                      numtype *h_num_u_candidate_vs_,
                      vtype *d_u_candidate_vs_,
                      numtype *d_num_u_candidate_vs_)
{
  cudaStream_t build_stream;
  cuchk(cudaStreamCreate(&build_stream));

  const uint32_t NUM_TRIES = NUM_EQ * 2 + 1;
  cuchk(cudaMallocAsync(&d_buffer_, sizeof(uint32_t) * NUM_TRIES * 2, build_stream));
  d_num_buckets_ = d_buffer_;
  d_hash_table_offs_ = d_num_buckets_ + NUM_TRIES;

  h_buffer = new uint32_t[(NUM_EQ * 2 + 1) * 2];
  h_num_buckets_ = h_buffer;
  h_hash_table_offs_ = h_num_buckets_ + NUM_TRIES;

  h_hash_table_offs_[0] = 0;
  for (int i = 0; i < NUM_EQ * 2; ++i)
  {
    vtype u = hq->evv[i].first;
    h_num_buckets_[i] = (h_num_u_candidate_vs_[u] * CUCKOO_SCALE + BUCKET_DIM - 1) / BUCKET_DIM;
    h_hash_table_offs_[i + 1] = h_hash_table_offs_[i] + h_num_buckets_[i] * BUCKET_DIM;
  }

  uint32_t total_size_per_table = h_hash_table_offs_[NUM_EQ * 2];

  cuchk(cudaMemcpyAsync(d_num_buckets_, h_num_buckets_, sizeof(uint32_t) * NUM_TRIES, cudaMemcpyHostToDevice, build_stream));
  cuchk(cudaMemcpyAsync(d_hash_table_offs_, h_hash_table_offs_, sizeof(uint32_t) * NUM_TRIES, cudaMemcpyHostToDevice, build_stream));

  for (int i = 0; i < NUM_TABLES; ++i)
  {
    cuchk(cudaMallocAsync(&d_keys_[i], sizeof(uint32_t) * total_size_per_table, build_stream));
    cuchk(cudaMallocAsync((void **)&d_pos_in_org[i], sizeof(uint32_t) * total_size_per_table, build_stream)); // new line
    // cuchk(cudaMallocAsync(&d_values_[i], sizeof(uint32_t) * (total_size_per_table * 2 + 1), build_stream));
    // cuchk(cudaMemsetAsync(d_values_[i], 0, sizeof(uint32_t) * (total_size_per_table * 2 + 1), build_stream));
  }

  std::unordered_map<uint32_t, uint32_t> finished_built_u;
  std::mt19937 gen(0);
  std::uniform_int_distribution<uint32_t> dis(0, 0xFFFFFFFF);
  uint32_t success = 0;

  uint32_t *progress;
  cuchk(cudaMallocAsync(&progress, sizeof(uint32_t) * 3, build_stream));
  for (etype e = 0; e < NUM_EQ * 2; ++e)
  {
    vtype u = hq->evv[e].first;
    if (hq->out_degree_[u] == 1)
      continue;

    if (finished_built_u.find(u) == finished_built_u.end()) // `u` did not processed.
    {
      do
      {
        cuchk(cudaMemsetAsync(progress, 0, sizeof(uint32_t) * 3, build_stream));
        for (int table_id = 0; table_id < NUM_TABLES; ++table_id)
        {
          cuchk(cudaMemsetAsync(d_keys_[table_id] + h_hash_table_offs_[e], 0xFF, sizeof(uint32_t) * h_num_buckets_[e] * BUCKET_DIM, build_stream));
          hash_constants_[e * NUM_TABLES * 2 + table_id * 2 + 0] = std::max(1u, dis(gen) % HASH_PRIME);
          hash_constants_[e * NUM_TABLES * 2 + table_id * 2 + 1] = dis(gen) % HASH_PRIME;
        }

        buildHashKeys<<<GRID_DIM, BLOCK_DIM, 0, build_stream>>>(
            d_u_candidate_vs_ + u * MAX_L_FREQ,
            h_num_u_candidate_vs_[u],
            d_keys_[0] + h_hash_table_offs_[e],
            d_keys_[1] + h_hash_table_offs_[e],
            h_num_buckets_[e],
            hash_constants_[e * NUM_TABLES * 2 + 0 * 2 + 0],
            hash_constants_[e * NUM_TABLES * 2 + 0 * 2 + 1],
            hash_constants_[e * NUM_TABLES * 2 + 1 * 2 + 0],
            hash_constants_[e * NUM_TABLES * 2 + 1 * 2 + 1],
            progress, progress + 2);
        cuchk(cudaMemcpyAsync(&success, progress + 2, sizeof(uint32_t), cudaMemcpyDeviceToHost, build_stream));
        cuchk(cudaStreamSynchronize(build_stream));
      } while (success != 0);
      finished_built_u[u] = e;
    }
    else
    {
      uint32_t reference_eidx = finished_built_u[u];
      for (int table_id = 0; table_id < NUM_TABLES; ++table_id)
      {
        hash_constants_[e * NUM_TABLES * 2 + table_id * 2 + 0] =
            hash_constants_[reference_eidx * NUM_TABLES * 2 + table_id * 2 + 0];
        hash_constants_[e * NUM_TABLES * 2 + table_id * 2 + 1] =
            hash_constants_[reference_eidx * NUM_TABLES * 2 + table_id * 2 + 1];
        cuchk(cudaMemcpyAsync(d_keys_[table_id] + h_hash_table_offs_[e],
                              d_keys_[table_id] + h_hash_table_offs_[reference_eidx],
                              sizeof(uint32_t) * h_num_buckets_[reference_eidx] * BUCKET_DIM,
                              cudaMemcpyDeviceToDevice, build_stream));
        cuchk(cudaStreamSynchronize(build_stream));
      }
    }
  }

  for (int table_id = 0; table_id < NUM_TABLES; ++table_id)
  {
    cuchk(cudaMemsetAsync(d_pos_in_org[table_id], 0xFF,
                          sizeof(uint32_t) * total_size_per_table, build_stream));
  }
  finished_built_u.clear();
  constexpr uint32_t POS_BLOCK_DIM = 256u;
  for (etype e = 0; e < NUM_EQ * 2; ++e)
  {
    vtype u = hq->evv[e].first;
    if (hq->out_degree_[u] == 1)
      continue;

    if (finished_built_u.find(u) == finished_built_u.end())
    {
#ifndef NDEBUG
      std::cout << "Populating pos in org for vertex u=" << u << "\n";
#endif
      uint32_t num_candidates = h_num_u_candidate_vs_[u];
      uint32_t num_buckets = h_num_buckets_[e];
      if (num_candidates == 0 || num_buckets == 0)
        continue;

      uint32_t offset = h_hash_table_offs_[e];
      const vtype *candidates = d_u_candidate_vs_ + u * MAX_L_FREQ;

      uint32_t c0 = hash_constants_[e * NUM_TABLES * 2 + 0];
      uint32_t c1 = hash_constants_[e * NUM_TABLES * 2 + 1];
      uint32_t c2 = hash_constants_[e * NUM_TABLES * 2 + 2];
      uint32_t c3 = hash_constants_[e * NUM_TABLES * 2 + 3];

      uint32_t grid_dim = (num_candidates + POS_BLOCK_DIM - 1) / POS_BLOCK_DIM;

      populate_pos_in_org_kernel<<<grid_dim, POS_BLOCK_DIM, 0, build_stream>>>(
          candidates,
          num_candidates,
          num_buckets,
          c0, c1, c2, c3,
          d_keys_[0] + offset,
          d_keys_[1] + offset,
          d_pos_in_org[0] + offset,
          d_pos_in_org[1] + offset);
      finished_built_u[u] = e;
    }
    else
    {
      uint32_t reference_eidx = finished_built_u[u];
      for (int table_id = 0; table_id < NUM_TABLES; ++table_id)
      {
        cuchk(cudaMemcpyAsync(d_pos_in_org[table_id] + h_hash_table_offs_[e],
                              d_pos_in_org[table_id] + h_hash_table_offs_[reference_eidx],
                              sizeof(uint32_t) * h_num_buckets_[reference_eidx] * BUCKET_DIM,
                              cudaMemcpyDeviceToDevice, build_stream));
      }
    }
  }

  cuchk(cudaMallocAsync(&d_hash_constants_, sizeof(uint32_t) * MAX_EQ * 2 * NUM_TABLES * 2, build_stream));
  cuchk(cudaMemcpyAsync(d_hash_constants_, hash_constants_, sizeof(uint32_t) * MAX_EQ * 2 * NUM_TABLES * 2, cudaMemcpyHostToDevice, build_stream));
  cuchk(cudaFreeAsync(progress, build_stream));

#ifndef NDEBUG
  std::ofstream outfile("runtime_data/HashTables.txt");
  vtype *h_keys_[2] = {new vtype[total_size_per_table], new vtype[total_size_per_table]};
  cuchk(cudaMemcpyAsync(h_keys_[0], d_keys_[0], sizeof(vtype) * total_size_per_table, cudaMemcpyDeviceToHost, build_stream));
  cuchk(cudaMemcpyAsync(h_keys_[1], d_keys_[1], sizeof(vtype) * total_size_per_table, cudaMemcpyDeviceToHost, build_stream));
  uint32_t flag[NUM_VQ] = {0, 0, 0, 0};
  cuchk(cudaStreamSynchronize(build_stream));
  for (etype e = 0; e < NUM_EQ * 2; ++e)
  {
    vtype u = hq->evv[e].first;
    if (flag[u])
      continue;
    flag[u] = 1;

    uint32_t offset = h_hash_table_offs_[e];
    uint32_t num_buckets = h_num_buckets_[e];
    outfile << "Hash Table for vertex u=" << u << " \n";
    for (int table_idx = 0; table_idx < NUM_TABLES; ++table_idx)
    {
      outfile << "table " << table_idx << ":\n";
      for (int i = 0; i < num_buckets * BUCKET_DIM; ++i)
      {
        vtype key = h_keys_[table_idx][offset + i];
        outfile << key << " ";
      }
      outfile << "\n";
    }
    outfile << '\n';
  }
  delete[] h_keys_[0];
  delete[] h_keys_[1];
  outfile.close();
  outfile.open("runtime_data/HashPos.txt");
  uint32_t *h_pos_[2] = {new uint32_t[total_size_per_table], new uint32_t[total_size_per_table]};
  cuchk(cudaMemcpyAsync(h_pos_[0], d_pos_in_org[0], sizeof(uint32_t) * total_size_per_table, cudaMemcpyDeviceToHost, build_stream));
  cuchk(cudaMemcpyAsync(h_pos_[1], d_pos_in_org[1], sizeof(uint32_t) * total_size_per_table, cudaMemcpyDeviceToHost, build_stream));
  cuchk(cudaStreamSynchronize(build_stream));
  for (etype e = 0; e < NUM_EQ * 2; ++e)
  {
    vtype u = hq->evv[e].first;
    if (flag[u] == 2)
      continue;
    flag[u] = 2;

    uint32_t offset = h_hash_table_offs_[e];
    uint32_t num_buckets = h_num_buckets_[e];
    outfile << "Hash Pos Table for vertex u=" << u << " \n";
    for (int table_idx = 0; table_idx < NUM_TABLES; ++table_idx)
    {
      outfile << "table " << table_idx << ":\n";
      for (int i = 0; i < num_buckets * BUCKET_DIM; ++i)
      {
        uint32_t pos = h_pos_[table_idx][offset + i];
        outfile << pos << " ";
      }
      outfile << '\n';
    }
    outfile << '\n';
  }
  delete[] h_pos_[0];
  delete[] h_pos_[1];
  outfile.close();
  outfile.open("runtime_data/HashConstants.txt");
  for (etype e = 0; e < NUM_EQ * 2; ++e)
  {
    vtype u = hq->evv[e].first;
    if (flag[u] == 3)
      continue;
    flag[u] = 3;

    outfile << "Hash Constants for vertex u=" << u << " \n";
    for (int table_idx = 0; table_idx < NUM_TABLES; ++table_idx)
    {
      uint32_t c0 = hash_constants_[e * NUM_TABLES * 2 + table_idx * 2 + 0];
      uint32_t c1 = hash_constants_[e * NUM_TABLES * 2 + table_idx * 2 + 1];
      outfile << "table " << table_idx << ": " << c0 << ", " << c1 << "\n";
    }
    outfile << '\n';
  }
  outfile.close();

  outfile.open("runtime_data/HashTableAddrs.txt");
  for (etype e = 0; e < NUM_EQ * 2; ++e)
  {
    vtype u = hq->evv[e].first;
    if (flag[u] == 4)
      continue;
    flag[u] = 4;

    outfile << "Hash Table Addrs for vertex u=" << u << " \n";
    for (int table_idx = 0; table_idx < NUM_TABLES; ++table_idx)
    {
      outfile << "table " << table_idx << ": " << d_keys_[table_idx] + h_hash_table_offs_[e] << "\n";
    }
    outfile << '\n';
  }
  outfile.close();
#endif
  cuchk(cudaStreamSynchronize(build_stream));
  cuchk(cudaStreamDestroy(build_stream));
}

/* =========================================================================================================== */
/* =========================================================================================================== */
/* ================================================ DataBlock ================================================ */
/* =========================================================================================================== */
/* =========================================================================================================== */
DataBlock::DataBlock() : size_in_byte(0),
                         d_data(nullptr),
                         d_parents(nullptr),
                         num_rows(0),
                         num_compressed_blocks(0) {}

DataBlock::~DataBlock()
{
  // deallocate();
}

void DataBlock::init(uint64_t var_num_rows, bool is_mask)
{
  num_rows = var_num_rows;
  if (is_mask)
  {
    num_compressed_blocks = (num_rows + 31) >> 5;
    size_in_byte = num_compressed_blocks * sizeof(uint32_t); // mask layer does not need parent pointer.
  }
  else
  {
    num_compressed_blocks = 0;
    size_in_byte = num_rows * (sizeof(uint32_t) + sizeof(uint64_t)); // data + parents
  }
}

void DataBlock::allocate()
{
  cudaStream_t alloc_stream;
  cuchk(cudaStreamCreate(&alloc_stream));
  if (num_compressed_blocks == 0) // expand
  {
    cuchk(cudaMallocAsync(&d_data, sizeof(vtype) * num_rows, alloc_stream));
    cuchk(cudaMallocAsync(&d_parents, sizeof(uint64_t) * num_rows, alloc_stream));
  }
  else // mask
  {
    cuchk(cudaMallocAsync(&d_data, sizeof(uint32_t) * num_compressed_blocks, alloc_stream));
    d_parents = nullptr;
  }
  cuchk(cudaStreamSynchronize(alloc_stream));
  cuchk(cudaStreamDestroy(alloc_stream));
}

void DataBlock::deallocate()
{
  cudaStream_t dealloc_stream;
  cuchk(cudaStreamCreate(&dealloc_stream));
  if (d_data)
    cudaFreeAsync(d_data, dealloc_stream);
  if (d_parents)
    cudaFreeAsync(d_parents, dealloc_stream);
  cuchk(cudaStreamSynchronize(dealloc_stream));
  cuchk(cudaStreamDestroy(dealloc_stream));
  d_data = nullptr;
  d_parents = nullptr;
  size_in_byte = 0;
  num_rows = 0;
  num_compressed_blocks = 0;
}
/* =========================================================================================================== */
/* =========================================================================================================== */
/* =============================================== UnifiedTrie =============================================== */
/* =========================================================================================================== */
/* =========================================================================================================== */

UnifiedTrie::UnifiedTrie() : id(UINT32_MAX), et(ettype()), num_layers(0u), num_cols(0u),
                             mapped_query_us_(nullptr), num_results(0),
                             mask_parent_id(UINT32_MAX), expansion_parent_id(UINT32_MAX),
                             layer_ids(), column_ids(),
                             data(nullptr), r_data(nullptr) {}

UnifiedTrie::~UnifiedTrie()
{
  // deallocateData();
}

void UnifiedTrie::init(uint32_t trie_id, ettype edge_tag, uint32_t levels, uint32_t cols)
{
  id = trie_id;
  et = edge_tag;
  num_layers = levels;
  num_cols = cols;
}

void UnifiedTrie::setQueryVertices(vtype *query_vertices) { /*mapped_query_us_ = query_vertices; */ }

void UnifiedTrie::setQueryVerticesExpand(const UnifiedTrie *other, uint32_t new_u)
{
  mapped_query_us_.reset(new vtype[num_cols]);
  memcpy(mapped_query_us_.get(), other->mapped_query_us_.get(), (num_cols - 1) * sizeof(vtype));
  mapped_query_us_[num_cols - 1] = new_u;
}

void UnifiedTrie::setQueryVerticesMask(const UnifiedTrie *other) { mapped_query_us_ = other->mapped_query_us_; }

void UnifiedTrie::setMaskParent(uint32_t parent_id)
{
  mask_parent_id = parent_id;
  expansion_parent_id = UINT32_MAX;
}

void UnifiedTrie::setExpansionParent(uint32_t parent_id)
{
  expansion_parent_id = parent_id;
  mask_parent_id = UINT32_MAX;
}

void UnifiedTrie::addLayer(uint32_t layer_id) { layer_ids.push_back(layer_id); }

void UnifiedTrie::setLayers(const std::vector<uint32_t> &layers)
{
  layer_ids = layers;
}

void UnifiedTrie::setLayersFrom(const UnifiedTrie *other, uint32_t new_layer)
{
  layer_ids = other->layer_ids;
  layer_ids.push_back(new_layer);
}

void UnifiedTrie::setColumnsFrom(const UnifiedTrie *other, uint32_t new_column, bool is_expansion)
{
  column_ids = other->column_ids;
  if (is_expansion)
  {
    column_ids.push_back(new_column);
  }
  // If it's a mask operation, we don't add new_column (just inherit parent's columns)
}

void UnifiedTrie::allocateDataMemMgr(uint32_t id, uint64_t num_rows, bool is_mask)
{
  auto &mem_mgr = MemoryManager::getInstance();
  if (data)
    deallocateData();
  data = new DataBlock();
  data->init(num_rows, is_mask);
  // mem_mgr.allocate_gpu_memory(id, data->size_in_byte, &data->d_space);
}

void UnifiedTrie::allocateData(uint64_t num_rows, bool is_mask)
{
  if (data)
    deallocateData();
  data = new DataBlock();
  data->init(num_rows, is_mask);
  data->allocate();
}

void UnifiedTrie::deallocateData()
{
  if (data)
  {
    data->deallocate();
    delete data;
    data = nullptr;
  }
  if (r_data)
  {
    r_data->deallocate();
    delete r_data;
    r_data = nullptr;
  }
}

void UnifiedTrie::updateNumResults(uint64_t count) { num_results = count; }

bool UnifiedTrie::hasMaskParent() const { return mask_parent_id != static_cast<uint32_t>(-1); }
bool UnifiedTrie::hasExpansionParent() const { return expansion_parent_id != static_cast<uint32_t>(-1); }
uint32_t UnifiedTrie::getNumLayers() const { return layer_ids.size(); }
bool UnifiedTrie::isEmpty() const { return num_results == 0; }

/* =========================================================================================================== */
/* =========================================================================================================== */
/* ============================================ UnifiedTrieManager =========================================== */
/* =========================================================================================================== */
/* =========================================================================================================== */

std::unique_ptr<UnifiedTrieManager> UnifiedTrieManager::instance = nullptr;
std::mutex UnifiedTrieManager::instance_mutex;

UnifiedTrieManager &UnifiedTrieManager::getInstance()
{
  std::lock_guard<std::mutex> lock(instance_mutex);
  if (!instance)
  {
    instance.reset(new UnifiedTrieManager());
  }
  return *instance;
}

void UnifiedTrieManager::initialize()
{
  std::lock_guard<std::mutex> lock(instance_mutex);
  instance.reset(new UnifiedTrieManager());
}

void UnifiedTrieManager::shutdown()
{
  std::lock_guard<std::mutex> lock(instance_mutex);
  instance.reset();
}

UnifiedTrieManager::UnifiedTrieManager() : tries(), num_tries(0),
                                           // complete_rq_ids(),
                                           // computed_rq_ids(),
                                           // uncomputed_rq_ids(),
                                           // computed_subgraphs_ids(),
                                           parent_addr_list(), data_addr_list()
{
}

UnifiedTrieManager::~UnifiedTrieManager() { clear(); }

void UnifiedTrieManager::init(uint32_t var_num_tries)
{
  clear();
  tries.resize(var_num_tries, nullptr);
  num_tries = var_num_tries;
  parent_addr_list.resize(var_num_tries);
  data_addr_list.resize(var_num_tries);
}

void UnifiedTrieManager::clear()
{
  for (auto trie : tries)
    if (trie)
      delete trie;
  tries.clear();
  num_tries = 0;
  // complete_rq_ids.clear();
  // computed_rq_ids.clear();
  // uncomputed_rq_ids.clear();
  // computed_subgraphs_ids.clear();
  parent_addr_list.clear();
  data_addr_list.clear();
}

UnifiedTrie *UnifiedTrieManager::createTrie(uint32_t trie_id, ettype edge_tag, uint32_t levels, uint32_t cols)
{
  if (trie_id >= num_tries)
    throw std::runtime_error("Trie ID out of range in createTrie.");
  if (tries[trie_id])
    throw std::runtime_error("Trie ID already exists in createTrie.");
  UnifiedTrie *trie = new UnifiedTrie();
  trie->init(trie_id, edge_tag, levels, cols);
  tries[trie_id] = trie;
  return trie;
}

UnifiedTrie *UnifiedTrieManager::getTrie(uint32_t trie_id)
{
  if (trie_id >= num_tries)
    throw std::runtime_error("Trie ID invalid in getTrie." + std::to_string(trie_id) + " >= " + std::to_string(num_tries));
  return tries[trie_id];
}

void UnifiedTrieManager::removeTrie(uint32_t trie_id)
{
  if (trie_id >= num_tries)
    throw std::runtime_error("Trie ID invalid in removeTrie.");
  delete tries[trie_id];
  tries[trie_id] = nullptr;
}

void UnifiedTrieManager::linkMask(uint32_t target_id, uint32_t parent_id)
{
  auto &mem_mgr = MemoryManager::getInstance();
  if (target_id >= num_tries || !tries[target_id])
    throw std::runtime_error("Target Trie ID invalid in linkMask.");
  if (parent_id >= num_tries || !tries[parent_id])
    throw std::runtime_error("Parent Trie ID invalid in linkMask.");

  UnifiedTrie *&target_trie = tries[target_id];
  UnifiedTrie *&parent_trie = tries[parent_id];

  if (parent_trie->data->num_compressed_blocks == 0) // parent is Expansion
  {
    target_trie->setQueryVerticesMask(parent_trie);
    target_trie->setMaskParent(parent_id);
    target_trie->setLayersFrom(parent_trie, target_id);
    target_trie->setColumnsFrom(parent_trie, target_id, false); // is_expansion = false (mask)
    // target_trie->allocateDataMemMgr(mem_mgr, target_id, parent_trie->num_results, true);
    target_trie->allocateData(parent_trie->num_results, true);
  }
  else // parent is Mask
  {
    target_trie->setQueryVerticesMask(parent_trie);
    target_trie->setMaskParent(parent_id);
    target_trie->setLayersFrom(parent_trie, target_id);
    target_trie->setColumnsFrom(parent_trie, target_id, false); // is_expansion = false (mask)
    // target_trie->allocateDataMemMgr(mem_mgr, target_id, parent_trie->data->num_rows, true);
    target_trie->allocateData(parent_trie->data->num_rows, true);
  }
  // target_trie->allocateData(parent_trie->num_results, true);

  // For mask operation: inherit parent's layer pointers (don't add mask layer itself)
  // parent_addr_list[target_id] = parent_addr_list[parent_id];
  // data_addr_list[target_id] = data_addr_list[parent_id];
  // Note: Mask layers are NOT added to the lists - they're just filters
}

void UnifiedTrieManager::linkExpansion(uint32_t target_id, uint32_t parent_id, uint32_t num_new_rows, vtype new_u)
{
  auto &mem_mgr = MemoryManager::getInstance();

  if (target_id >= num_tries || !tries[target_id])
    throw std::runtime_error("Target Trie ID invalid in linkExpansion.");
  if (parent_id >= num_tries || !tries[parent_id])
    throw std::runtime_error("Parent Trie ID invalid in linkExpansion.");

  UnifiedTrie *&target_trie = tries[target_id];
  UnifiedTrie *&parent_trie = tries[parent_id];

  target_trie->setQueryVerticesExpand(parent_trie, new_u);
  target_trie->setExpansionParent(parent_id);
  target_trie->setLayersFrom(parent_trie, target_id);
  target_trie->setColumnsFrom(parent_trie, target_id, true); // is_expansion = true
  // if (mem_mgr)
  // will not be triggerred in current implementation
  // target_trie->allocateDataMemMgr(mem_mgr, target_id, num_new_rows, false);

  // if (mem_mgr)
  // {
  // target_trie->allocateDataMemMgr(mem_mgr, target_id, num_new_rows, false);
  // target_trie->allocateData(parent_trie->num_results, false);

  // // Cache parent pointers
  // parent_addr_list[target_id] = parent_addr_list[parent_id];
  // parent_addr_list[target_id].push_back(parent_trie->data->d_data);

  // // Cache data arrays
  // data_addr_list[target_id] = data_addr_list[parent_id];
  // data_addr_list[target_id].push_back(parent_trie->data->d_data);
  // }
}

uint32_t UnifiedTrieManager::getNumTries() const { return num_tries; }

std::vector<uint64_t *> &UnifiedTrieManager::get_parent_arrays(uint32_t trie_id)
{
  if (trie_id >= num_tries)
    throw std::runtime_error("Trie ID invalid in get_parent_arrays.");

  UnifiedTrie *trie = tries[trie_id];
  if (!trie)
    throw std::runtime_error("Trie is null in get_parent_arrays.");

  // Use a static thread-local cache to avoid allocating on each call
  static thread_local std::vector<uint64_t *> result;
  result.clear();
  assert(result.size() == 0);
  result.reserve(trie->column_ids.size());

  // Use column_ids to directly fetch parent pointers
  // column_ids contains the trie IDs for each column in order (root to current)
  for (int i = 0; i < trie->column_ids.size(); ++i)
  {
    uint32_t col_trie_id = trie->column_ids[i];
    UnifiedTrie *col_trie = tries[col_trie_id];
    assert(col_trie != nullptr);

    if (i == 0)
    {
      // First column (vertex node) has no parent
      result.push_back(nullptr);
    }
    else
    {
      // Subsequent columns store parent pointers
      if (col_trie->data)
      {
        result.push_back(col_trie->data->d_parents);
#ifndef NDEBUG
        printf("Parent array for trie ID %u: %p\n", col_trie_id, col_trie->data->d_parents);
#endif
      }
      else if (col_trie->r_data)
      {
        result.push_back(col_trie->r_data->d_parents);
#ifndef NDEBUG
        printf("Parent array for trie ID %u (r_data): %p\n", col_trie_id, col_trie->r_data->d_parents);
#endif
      }
      else
      {
        // std::cerr << "Warning: Parent array for trie ID " << col_trie_id << " is null." << std::endl;
        result.push_back(nullptr);
#ifndef NDEBUG
        printf("Parent array for trie ID %u (null): %p\n", col_trie_id, nullptr);
#endif
      }
    }
  }

  return result;
}

std::vector<vtype *> &UnifiedTrieManager::get_data_arrays(uint32_t trie_id)
{
  if (trie_id >= num_tries)
    throw std::runtime_error("Trie ID invalid in get_data_arrays.");

  UnifiedTrie *trie = tries[trie_id];
  if (!trie)
    throw std::runtime_error("Trie is null in get_data_arrays.");

  // Use a static thread-local cache to avoid allocating on each call
  static thread_local std::vector<vtype *> result;
  result.clear();
  assert(result.size() == 0);
  result.reserve(trie->column_ids.size());
  // bool r_flag = false;
#ifndef NDEBUG
  printf("See column_ids: ");
  for (uint32_t col_trie_id : trie->column_ids)
    printf("%u, ", col_trie_id);
  printf("\n");
  printf("Expected: 6, 4\n");
#endif

  // Use column_ids to directly fetch data pointers
  // column_ids contains the trie IDs for each column in order (root to current)
  for (uint32_t col_trie_id : trie->column_ids)
  {
    UnifiedTrie *col_trie = tries[col_trie_id];
    assert(col_trie != nullptr);
    if (col_trie->data)
    {
      result.push_back(col_trie->data->d_data);
#ifndef NDEBUG
      printf("Data array for trie ID %u: %p\n", col_trie_id, col_trie->data->d_data);
#endif
    }
    else if (col_trie->r_data)
    {
      result.push_back(col_trie->r_data->d_data);
      // if (col_trie->column_ids.size() == 2)
      //   r_flag = true;
#ifndef NDEBUG
      printf("Data array for trie ID %u (r_data): %p\n", col_trie_id, col_trie->r_data->d_data);
#endif
    }
    else
    {
#ifndef NDEBUG
      std::cerr << "Warning: Data array for trie ID " << col_trie_id << " is null." << std::endl;
#endif
      result.push_back(nullptr);
    }
  }

  return result;
}

void UnifiedTrieManager::initVertexTries(
    vtype *d_u_candidate_vs_, numtype *h_num_u_candidate_vs_)
{
  cudaStream_t build_stream;
  cuchk(cudaStreamCreate(&build_stream));
  auto &mem_mgr = MemoryManager::getInstance();
  auto &lat = Lattice::getInstance();
  uint32_t vertex_node_start = lat.num_lattice_nodes - NUM_VQ;
  for (uint32_t i = 0; i < NUM_VQ; ++i)
  {
    uint32_t vertex_trie_id = vertex_node_start + i;
    UnifiedTrie *vertex_trie = this->createTrie(vertex_trie_id, ettype().set(), 1, 1);
    vertex_trie->mapped_query_us_.reset(new vtype[1]);
    vertex_trie->mapped_query_us_[0] = i;
    vertex_trie->num_results = h_num_u_candidate_vs_[i];
    vertex_trie->mask_parent_id = vertex_trie->expansion_parent_id = UINT32_MAX;
    vertex_trie->column_ids.push_back(vertex_trie_id);
    vertex_trie->layer_ids.push_back(vertex_trie_id);
    vertex_trie->data = new DataBlock();
    vertex_trie->data->init(h_num_u_candidate_vs_[i], false);
    vertex_trie->data->size_in_byte >>= 1; // no parent pointers.
    // mem_mgr->allocate_gpu_memory(vertex_trie_id, h_num_u_candidate_vs_[i] * sizeof(vtype), &vertex_trie->data->d_space);
    cuchk(cudaMallocAsync((void **)&vertex_trie->data->d_data, sizeof(vtype) * h_num_u_candidate_vs_[i], build_stream));
    cuchk(cudaMemcpyAsync(vertex_trie->data->d_data, d_u_candidate_vs_ + i * MAX_L_FREQ,
                          sizeof(vtype) * h_num_u_candidate_vs_[i], cudaMemcpyDeviceToDevice, build_stream));
    // data_addr_list[vertex_trie_id] = {vertex_trie->data->d_data};
    // parent_addr_list[vertex_trie_id] = {nullptr};
  }
  cuchk(cudaStreamSynchronize(build_stream));
  cuchk(cudaStreamDestroy(build_stream));

#ifndef NDEBUG
  std::ofstream outfile("runtime_data/vertex_trie_data_addr.txt");
  for (uint32_t i = 0; i < NUM_VQ; ++i)
  {
    uint32_t vertex_trie_id = vertex_node_start + i;
    UnifiedTrie *vertex_trie = this->getTrie(vertex_trie_id);
    outfile << "Vertex Trie ID: " << vertex_trie_id << ", d_data address: " << vertex_trie->data->d_data << "\n";
  }
  outfile.close();
#endif
}

uint32_t UnifiedTrieManager::find_best_edge_candidate(
    const ettype &cur_et)
{
  Lattice *lat = &Lattice::getInstance();
  uint32_t best_id = UINT32_MAX;
  double best_score = -1.0;

  if (cur_et.none())
  {
    const double alpha = 0.6; // size weight;
    const double beta = 0.3;  // contribution weight
    const double gamma = 0.1; // connectivity weight
    for (int i = 0; i < NUM_EQ; ++i)
    {
      ettype cur_edge = ettype().set(i);
      int cur_id = lat->et2id[cur_edge];
      if (lat->uncomputed_reachable_rq[cur_id].empty())
        continue;
      UnifiedTrie *trie = getTrie(cur_id);

      double size_factor = 1.0 / (1.0 + trie->num_results + 1);
      double contribution_factor = lat->get_contribution_value(cur_edge);
      double connectivity_factor =
          lat->reversed_linked_list[cur_id].size() * 0.1;

      double score = alpha * size_factor + beta * contribution_factor +
                     gamma * connectivity_factor;

      if (score > best_score)
      {
        best_score = score;
        best_id = cur_id;
      }
    }
  }
  else if (cur_et.count() == NUM_EQ)
  {
    // do nothing
  }
  else
  {
    const double alpha = 0.4; // size weight
    const double beta = 0.4;  // contribution weight
    const double gamma = 0.2; // selectivity weight

    int cur_id = lat->et2id[cur_et];

    for (auto next_id : lat->reversed_linked_list[cur_id])
    {
      if (lat->uncomputed_reachable_rq[next_id].empty() == false)
        ; // keep it
      else if (lat->is_uncomputed(next_id))
        ;         // not computed rq, keep it.
      else        // no contribution, not an rq.
        continue; // useless

      ettype next_et = lat->id2et[next_id];
      ettype diff = next_et ^ cur_et;
      if (diff.count() != 1)
        continue;

      int diff_id = lat->et2id[diff];
      UnifiedTrie *trie = getTrie(diff_id);

      double size_factor = 1.0 / (1.0 + trie->num_results + 1);
      double contribution_factor = lat->get_contribution_value(diff);
      // double estimated_join_size_val = estimate_join_size(lat, ltm, cur_id, diff_id);
      // double selectivity_factor =
      //     1.0 /
      //     (1.0 + estimated_join_size_val / trie->get_total_forward_pairs());
      double score = alpha * size_factor + beta * contribution_factor;
      if (score > best_score)
      {
        best_score = score;
        best_id = diff_id;
      }
    }
  }
  return best_id;
}

void UnifiedTrieManager::construct_edge_candidates(
    cpuGraph *hq, cpuGraph *hg, gpuGraph *dg,
    vtype *d_u_candidate_vs_, numtype *d_num_u_candidate_vs_, numtype *h_num_u_candidate_vs_,
    uint32_t *d_bitmap, uint32_t bitmap_pitch)
{
  cudaStream_t construct_stream;
  cuchk(cudaStreamCreate(&construct_stream));

  auto &ht = HashTable::getInstance();
  auto &mem_mgr = MemoryManager::getInstance();
  auto &lat = Lattice::getInstance();

  uint32_t h_num_res = 0;
  // uint32_t *d_count_for_each;
  // offtype *d_offsets_each;
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;

  // First pass: compute total size needed for all edge offsets
  uint32_t total_offset_size = 0;
  ht.edge_offset_starts_.resize(NUM_EQ * 2 + 1, 0);

  for (etype eid = 0; eid < NUM_EQ * 2; ++eid)
  {
    vtype u = hq->evv[eid].first;
    if (hq->out_degree_[u] == 1)
      continue;
    ht.edge_offset_starts_[eid] = total_offset_size;
    total_offset_size += h_num_u_candidate_vs_[u] + 1;
  }
  ht.edge_offset_starts_[NUM_EQ * 2] = total_offset_size;

  // Allocate the flat array for all edge offsets
  cuchk(cudaMallocAsync(&ht.d_offsets_, sizeof(offtype) * total_offset_size, construct_stream));

  // Second pass: compute and store offsets for each edge
  bool forward_flag = true;

  uint32_t h_max_value = 0;
  uint32_t *d_max_value;
  cuchk(cudaMallocAsync(&d_max_value, sizeof(uint32_t), construct_stream));

  uint32_t *neighbor_mask;
  uint32_t *neighbor_mask_scan_output;
  uint32_t *neighbor_mask_offsets;
  uint32_t h_neighbor_mask_tiles = 0;
  cuchk(cudaMallocAsync(&neighbor_mask_offsets, sizeof(uint32_t) * (NUM_VD + 1), construct_stream));

  // compute mask_offsets.
  void *d_temp_storage_for_offsets = nullptr;
  size_t temp_storage_bytes_for_offsets = 0;
  cub::TransformInputIterator<uint32_t, div32CeilOp, uint32_t *> neighbor_mask_iter(dg->degree_, div32CeilOp());

  cuchk(cub::DeviceScan::ExclusiveSum(
      d_temp_storage_for_offsets, temp_storage_bytes_for_offsets,
      neighbor_mask_iter, neighbor_mask_offsets, NUM_VD, construct_stream));
  cuchk(cudaMallocAsync((void **)&d_temp_storage_for_offsets, temp_storage_bytes_for_offsets, construct_stream));
  cuchk(cub::DeviceScan::ExclusiveSum(
      d_temp_storage_for_offsets, temp_storage_bytes_for_offsets,
      neighbor_mask_iter, neighbor_mask_offsets, NUM_VD, construct_stream));
  add_last_for_exclusive_sum_div32ceil<<<1, 1, 0, construct_stream>>>(dg->degree_, neighbor_mask_offsets, NUM_VD, neighbor_mask_offsets + NUM_VD);

  cuchk(cudaMemcpyAsync(&h_neighbor_mask_tiles, neighbor_mask_offsets + NUM_VD, sizeof(uint32_t), cudaMemcpyDeviceToHost, construct_stream));
  cuchk(cudaFreeAsync(d_temp_storage_for_offsets, construct_stream));
  cuchk(cudaStreamSynchronize(construct_stream));
  cuchk(cudaMallocAsync((void **)&neighbor_mask, sizeof(uint32_t) * (h_neighbor_mask_tiles + 1), construct_stream));
  cuchk(cudaMallocAsync((void **)&neighbor_mask_scan_output, sizeof(uint32_t) * (h_neighbor_mask_tiles + 1), construct_stream));
  cuchk(cudaMemsetAsync(neighbor_mask, 0, sizeof(uint32_t) * (h_neighbor_mask_tiles + 1), construct_stream));

  uint32_t *workpool;
  uint32_t num_warps = (BLOCK_DIM / 32) * GRID_DIM;
  cuchk(cudaMallocAsync(&workpool, sizeof(uint32_t), construct_stream));

  for (vtype u = 0; u < hq->num_v; ++u)
  {
    if (hq->out_degree_[u] == 1)
      continue;
    for (offtype u_off = hq->offsets_[u]; u_off < hq->offsets_[u + 1]; ++u_off)
    {
      // TODO: maybe can be removed. Test to remove it after the program runs correctly.
      // cuchk(cudaMemsetAsync(neighbor_mask, 0, sizeof(uint32_t) * (h_neighbor_mask_tiles + 1), construct_stream));
      // move GPU operations ahead.
      // cuchk(cudaMemsetAsync(d_count_for_each, 0, sizeof(uint32_t) * max_num, construct_stream));

      etype e = hq->edgeIDs_[u_off];
      vtype u_nbr = hq->neighbors_[u_off];
      forward_flag = u < u_nbr;
      uint32_t lattice_node_id = lat.et2id[ettype().set(e >> 1)];

      uint32_t dynamic_shared_memory_in_bytes = sizeof(vtype) * h_num_u_candidate_vs_[u];
      dynamic_shared_memory_in_bytes = UINT32_MAX; // disable shared memory version for now.

      cuchk(cudaMemcpyAsync(workpool, &num_warps, sizeof(uint32_t), cudaMemcpyHostToDevice, construct_stream));

#ifndef NDEBUG
      std::cout << "shared: " << dynamic_shared_memory_in_bytes << std::endl;
#endif

      if (dynamic_shared_memory_in_bytes <= SHARED_MEMORY_LIMIT)
        ;
      // first_join_count_new<<<GRID_DIM, BLOCK_DIM, dynamic_shared_memory_in_bytes, construct_stream>>>(
      //     dg->offsets_, dg->neighbors_, dg->degree_,
      //     d_bitmap_mask, mask_length, u, u_nbr,
      //     h_num_u_candidate_vs_[u], h_num_u_candidate_vs_[u_nbr],
      //     d_u_candidate_vs_);
      else
        first_join_count_bitmap_task1<<<GRID_DIM, BLOCK_DIM, 0, construct_stream>>>(
            dg->offsets_, dg->neighbors_, dg->degree_,
            d_bitmap + u_nbr * bitmap_pitch,
            u, h_num_u_candidate_vs_[u], d_u_candidate_vs_ + u * MAX_L_FREQ,
            neighbor_mask, neighbor_mask_offsets, workpool);
      cuchk(cudaMemcpyAsync(workpool, &num_warps, sizeof(uint32_t), cudaMemcpyHostToDevice, construct_stream));

      /* >>>>>>>>>>>>>>>>>>>>>>>>>>> prefix sum on neighbor_mask <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< */

      // TODO: optimize it. Can be reused. Move it outside the loop. -- remove freeAsync too.
      d_temp_storage = nullptr;
      temp_storage_bytes = 0;
      cub::TransformInputIterator<uint32_t, PopCountOp, uint32_t *> popcount_iter(neighbor_mask, PopCountOp());
      cuchk(cub::DeviceScan::ExclusiveSum(
          d_temp_storage, temp_storage_bytes,
          popcount_iter, neighbor_mask_scan_output, (h_neighbor_mask_tiles + 1), construct_stream));
      cuchk(cudaMallocAsync(&d_temp_storage, temp_storage_bytes, construct_stream));
      cuchk(cub::DeviceScan::ExclusiveSum(
          d_temp_storage, temp_storage_bytes,
          popcount_iter, neighbor_mask_scan_output, (h_neighbor_mask_tiles + 1), construct_stream));
      cuchk(cudaFreeAsync(d_temp_storage, construct_stream));
      d_temp_storage = nullptr;
      cuchk(cudaMemcpyAsync(&h_num_res, neighbor_mask_scan_output + h_neighbor_mask_tiles, sizeof(uint32_t), cudaMemcpyDeviceToHost, construct_stream));

      /* >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> max value <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< */

      cuchk(cudaMemsetAsync(d_max_value, 0, sizeof(uint32_t), construct_stream));
      offtype *d_edge_offsets = ht.d_offsets_ + ht.edge_offset_starts_[e];
      // fuse max computation with filling d_edge_offsets.
      getMax<<<GRID_DIM, BLOCK_DIM, 0, construct_stream>>>(
          d_u_candidate_vs_ + u * MAX_L_FREQ, h_num_u_candidate_vs_[u],
          neighbor_mask_scan_output, neighbor_mask_offsets, d_max_value, d_edge_offsets);
      cuchk(cudaMemcpyAsync(&h_max_value, d_max_value, sizeof(uint32_t), cudaMemcpyDeviceToHost, construct_stream));
      cuchk(cudaMemcpyAsync(d_edge_offsets + h_num_u_candidate_vs_[u], &h_num_res, sizeof(offtype), cudaMemcpyHostToDevice, construct_stream));
      cuchk(cudaStreamSynchronize(construct_stream));
      ht.max_values_per_edge_[e] = h_max_value;

#ifndef NDEBUG
      std::cout << "Max join count for edge " << e << " (u=" << u << ", u_nbr=" << u_nbr << ") is " << h_max_value << std::endl;
#endif

      /* >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> prefix sum <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< */
      // Compute offsets directly into HashTable's memory

      cuchk(cudaStreamSynchronize(construct_stream));
      UnifiedTrie *&cur_trie = tries[lattice_node_id];
      // if (hq->out_degree_[parent_u] == 1)
      // {
      //   parent_u = std::max(u, u_nbr);
      //   parent_node_id = lat.get_vertex_node_id(parent_u);
      // }
      if (cur_trie == nullptr)
      {
        uint32_t parent_node_id = lat.get_vertex_node_id(u);
        cur_trie = this->createTrie(lattice_node_id, ettype().set(e >> 1), 2, 2);
        this->linkExpansion(lattice_node_id, parent_node_id, h_num_res, u_nbr);
        cur_trie->edge_vertices[0] = std::min(u, u_nbr);
        cur_trie->edge_vertices[1] = std::max(u, u_nbr);
        // if (!forward_flag)
        // std::swap(cur_trie->mapped_query_us_[0], cur_trie->mapped_query_us_[1]);
      }

      DataBlock *&cur_block = forward_flag ? cur_trie->data : cur_trie->r_data;
      cur_block = new DataBlock();

      cur_trie->updateNumResults(h_num_res);
      cur_block->init(h_num_res, false);

#ifndef NDEBUG
      std::cout << "Allocating edge candidate block for trie " << lattice_node_id
                << " with " << h_num_res << " rows (" << cur_block->size_in_byte << " bytes)." << std::endl;
#endif
      cuchk(cudaMallocAsync(&cur_block->d_data, sizeof(uint32_t) * cur_block->num_rows, construct_stream));
      cuchk(cudaMallocAsync(&cur_block->d_parents, sizeof(uint64_t) * cur_block->num_rows, construct_stream));
      // cuchk(cudaMalloc(&cur_block->d_space, cur_block->size_in_byte));
      // mem_mgr->allocate_gpu_memory(lattice_node_id, cur_block->size_in_byte, &cur_block->d_space);

      // cur_block->d_data = reinterpret_cast<uint32_t *>(cur_block->d_space);
      // cur_block->d_parents = reinterpret_cast<uint32_t *>(reinterpret_cast<char *>(cur_block->d_space) + cur_block->num_rows * sizeof(uint32_t));

      // first_join_write_new<<<GRID_DIM, BLOCK_DIM, dynamic_shared_memory_in_bytes>>>(
      //     dg->offsets_, dg->neighbors_, dg->degree_,
      //     d_bitmap_mask, mask_length, u, u_nbr,
      //     h_num_u_candidate_vs_[u], h_num_u_candidate_vs_[u_nbr],
      //     d_u_candidate_vs_,
      //     cur_block->d_data, d_edge_offsets, cur_block->d_parents, d_progress_each);

      if (dynamic_shared_memory_in_bytes <= SHARED_MEMORY_LIMIT)
        ;
      // first_join_write_new<<<GRID_DIM, BLOCK_DIM, dynamic_shared_memory_in_bytes, construct_stream>>>(
      //     dg->offsets_, dg->neighbors_, dg->degree_,
      //     d_bitmap_mask, mask_length, u, u_nbr,
      //     h_num_u_candidate_vs_[u], h_num_u_candidate_vs_[u_nbr],
      //     d_u_candidate_vs_,
      //     cur_block->d_data, d_edge_offsets, d_scan_res, mask_length + 1, cur_block->d_parents);
      else
        first_join_write_bitmap_task1<<<GRID_DIM, BLOCK_DIM, 0, construct_stream>>>(
            dg->offsets_, dg->neighbors_, dg->degree_,
            neighbor_mask, neighbor_mask_offsets, neighbor_mask_scan_output,
            u, h_num_u_candidate_vs_[u], d_u_candidate_vs_ + u * MAX_L_FREQ,
            cur_block->d_data, d_edge_offsets, cur_block->d_parents, workpool);

      cuchk(cudaStreamSynchronize(construct_stream));

      // if forward is valid, it is. Or it is reversed.
      // if (data_addr_list[lattice_node_id].empty())
      // {
      //   data_addr_list[lattice_node_id] = data_addr_list[parent_node_id];
      //   data_addr_list[lattice_node_id].push_back(cur_block->d_data);

      //   parent_addr_list[lattice_node_id] = parent_addr_list[parent_node_id];
      //   parent_addr_list[lattice_node_id].push_back(cur_block->d_parents);
      // }
    }
  }

  // Cleanup temporary buffers (d_offsets_ in HashTable is kept)
  cuchk(cudaFreeAsync(d_max_value, construct_stream));
  cuchk(cudaFreeAsync(neighbor_mask, construct_stream));
  cuchk(cudaFreeAsync(neighbor_mask_scan_output, construct_stream));
  cuchk(cudaFreeAsync(neighbor_mask_offsets, construct_stream));
  cuchk(cudaFreeAsync(workpool, construct_stream));
  // cuchk(cudaFreeAsync(d_count_for_each, construct_stream));
  // cuchk(cudaFreeAsync(d_offsets_each, construct_stream));
  // cuchk(cudaFreeAsync(d_bitmap_mask, construct_stream));
  cuchk(cudaStreamSynchronize(construct_stream));

#ifndef NDEBUG
  std::ofstream outfile("runtime_data/edge_candidate_data_addresses.txt");
  for (uint32_t i = 0; i < NUM_EQ * 2; ++i)
  {
    auto [u, v] = hq->evv[i];
    uint32_t trie_id = lat.et2id[ettype().set(i >> 1)];
    if (hq->out_degree_[u] == 1)
      continue;
    UnifiedTrie *trie = this->getTrie(trie_id);
    if (trie && trie->data)
    {
      outfile << "Trie ID: " << trie_id << ", d_data address: " << trie->data->d_data << "\n";
    }
    if (trie && trie->r_data)
    {
      outfile << "Trie ID: " << trie_id << " (r_data), d_data address: " << trie->r_data->d_data << "\n";
    }
  }
  outfile.close();

  outfile.open("runtime_data/edge_candidate_parent_addresses.txt");
  for (uint32_t i = 0; i < NUM_EQ * 2; ++i)
  {
    auto [u, v] = hq->evv[i];
    uint32_t trie_id = lat.et2id[ettype().set(i >> 1)];
    if (hq->out_degree_[u] == 1)
      continue;
    UnifiedTrie *trie = this->getTrie(trie_id);
    if (trie && trie->data)
    {
      outfile << "Trie ID: " << trie_id << ", d_parents address: " << trie->data->d_parents << "\n";
    }
    if (trie && trie->r_data)
    {
      outfile << "Trie ID: " << trie_id << " (r_data), d_parents address: " << trie->r_data->d_parents << "\n";
    }
  }
  outfile.close();

  outfile.open("runtime_data/edge_candidate_value_layer_data.txt");
  for (uint32_t i = 0; i < NUM_EQ * 2; ++i)
  {
    auto [u, v] = hq->evv[i];
    if (hq->out_degree_[u] == 1)
      continue;
    uint32_t trie_id = lat.et2id[ettype().set(i >> 1)];
    UnifiedTrie *trie = this->getTrie(trie_id);
    vtype *h_data = new vtype[trie->num_results];
    if (u < v)
    {
      cuchk(cudaMemcpyAsync(h_data, trie->data->d_data, sizeof(vtype) * trie->data->num_rows, cudaMemcpyDeviceToHost, construct_stream));
      cuchk(cudaStreamSynchronize(construct_stream));
    }
    else
    {
      cuchk(cudaMemcpyAsync(h_data, trie->r_data->d_data, sizeof(vtype) * trie->r_data->num_rows, cudaMemcpyDeviceToHost, construct_stream));
      cuchk(cudaStreamSynchronize(construct_stream));
    }
    outfile << "Trie ID: " << trie_id << ", Data: ";
    for (uint32_t j = 0; j < trie->num_results; ++j)
    {
      outfile << h_data[j] << " ";
    }
    outfile << "\n";
    delete[] h_data;
  }
  outfile.close();

  outfile.open("runtime_data/edge_candidate_value_counts.txt");
  for (uint32_t i = 0; i < NUM_EQ * 2; ++i)
  {
    auto [u, v] = hq->evv[i];
    if (hq->out_degree_[u] == 1)
      continue;
    uint32_t trie_id = lat.et2id[ettype().set(i >> 1)];
    UnifiedTrie *trie = this->getTrie(trie_id);

    // Get the number of keys (u candidates)
    uint32_t num_keys = h_num_u_candidate_vs_[u];

    // Get the offset array for this edge from HashTable
    uint32_t edge_offset_start = ht.edge_offset_starts_[i];
    offtype *d_edge_offsets = ht.d_offsets_ + edge_offset_start;

    // Allocate host memory for offsets (num_keys + 1 elements)
    offtype *h_offsets_ = new offtype[num_keys + 1];

    // Copy offsets from device to host
    cuchk(cudaMemcpyAsync(h_offsets_, d_edge_offsets, sizeof(offtype) * (num_keys + 1), cudaMemcpyDeviceToHost, construct_stream));
    cuchk(cudaStreamSynchronize(construct_stream));

    // Output header for this edge
    outfile << "Edge " << (i >> 1) << " (directed edge " << i << "): u=" << u << ", v=" << v
            << ", num_keys=" << num_keys << "\n";

    // Output value count for each key
    for (uint32_t key_idx = 0; key_idx < num_keys; ++key_idx)
      outfile << h_offsets_[key_idx + 1] - h_offsets_[key_idx] << " ";
    outfile << "\n";

    // Output summary statistics
    offtype total_pairs = h_offsets_[num_keys];
    outfile << "  Total pairs: " << total_pairs << "\n";

    if (num_keys > 0)
    {
      // Find min and max value counts
      uint32_t min_count = UINT32_MAX;
      uint32_t max_count = 0;
      double avg_count = 0.0;
      uint32_t zero_counter = 0;

      for (uint32_t key_idx = 0; key_idx < num_keys; ++key_idx)
      {
        uint32_t count = h_offsets_[key_idx + 1] - h_offsets_[key_idx];
        min_count = std::min(min_count, count);
        max_count = std::max(max_count, count);
        avg_count += count;
        if (count == 0)
          zero_counter++;
      }
      avg_count /= num_keys;

      outfile << "  Zero value keys: " << zero_counter << "\n";
      outfile << "  Min values per key: " << min_count << "\n";
      outfile << "  Max values per key: " << max_count << "\n";
      outfile << "  Avg values per key: " << avg_count << "\n";
      outfile << "  Max from HashTable: " << ht.max_values_per_edge_[i] << "\n";
    }

    outfile << "\n";
    delete[] h_offsets_;
  }
  outfile.close();
#endif
  cuchk(cudaStreamSynchronize(construct_stream));
  // cuchk(cudaFreeAsync(d_scan_res, construct_stream));
  cuchk(cudaStreamDestroy(construct_stream));
}
