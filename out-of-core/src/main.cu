#include <algorithm>
#include <charconv>
#include <chrono>
#include <cerrno>
#include <cstring>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/wait.h>
#include <unistd.h>

#include "CLI11.hpp"
#include "cpuGraph.h"
#include "cuda_helpers.cuh"
#include "filter.h"
#include "globals.h"
#include "host_memory_manager.cuh"
#include "io.h"
#include "join_trie.cuh"
#include "lattice.h"
#include "memory_manager.cuh"
#include "relax.h"
#include "unifiedTrie.cuh"

#include <unordered_map>
#include <unordered_set>
#include <vector>

using std::cout;
using std::endl;

namespace fs = std::filesystem;

struct QueryJob
{
  uint32_t id;
  std::string path;
};

struct BatchFailure
{
  QueryJob job;
  std::string detail;
};

static bool tryParseQueryId(const fs::path &query_path, uint32_t &out_id)
{
  const std::string stem = query_path.stem().string();
  if (stem.empty())
    return false;

  size_t pos = 0;
  if (stem[pos] == 'Q' || stem[pos] == 'q')
    pos++;
  if (pos < stem.size() && (stem[pos] == '_' || stem[pos] == '-'))
    pos++;
  if (pos >= stem.size())
    return false;

  std::string_view digits(stem.data() + pos, stem.size() - pos);
  uint32_t value = 0;
  auto res = std::from_chars(digits.data(), digits.data() + digits.size(), value);
  if (res.ec != std::errc() || res.ptr != digits.data() + digits.size())
    return false;

  out_id = value;
  return true;
}

static std::vector<QueryJob> collectQueryJobs(const std::string &query_input, bool &is_directory)
{
  std::vector<QueryJob> jobs;
  std::error_code ec;
  fs::path input_path(query_input);

  if (fs::is_regular_file(input_path, ec))
  {
    is_directory = false;
    uint32_t query_id = 0;
    if (!tryParseQueryId(input_path, query_id))
      query_id = 0u;
    jobs.push_back({query_id, input_path.string()});
    return jobs;
  }

  if (!fs::is_directory(input_path, ec))
  {
    throw std::runtime_error("Query path is not a file or directory: " + query_input);
  }

  is_directory = true;

  std::unordered_set<uint32_t> seen_ids;
  for (const auto &entry : fs::directory_iterator(input_path))
  {
    if (!entry.is_regular_file())
      continue;

    const fs::path p = entry.path();
    if (p.extension() != ".in")
      continue;

    uint32_t query_id = 0;
    if (!tryParseQueryId(p, query_id))
    {
      throw std::runtime_error("Query file does not match expected pattern Q_<id>.in: " + p.string());
    }

    if (!seen_ids.insert(query_id).second)
    {
      throw std::runtime_error("Duplicate query id " + std::to_string(query_id) + " in directory: " + query_input);
    }

    jobs.push_back({query_id, p.string()});
  }

  std::sort(jobs.begin(), jobs.end(), [](const QueryJob &a, const QueryJob &b)
            { return a.id < b.id; });
  return jobs;
}

static int runSingleQuery(const QueryJob &job, cpuGraph &hg,
                          const std::string &data_path, uint32_t gpu_num,
                          const std::string &eviction_policy)
{
  (void)eviction_policy;

  try
  {
    int device = static_cast<int>(gpu_num);
    cuchk(cudaSetDevice(device));
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    SHARED_MEMORY_LIMIT = prop.sharedMemPerBlock;

#ifndef NDEBUG
    cout << "Device " << device << ": " << prop.name << endl;
#endif

    GPU_NUM = gpu_num;

    cudaStream_t main_stream;
    cuchk(cudaStreamCreate(&main_stream));
    size_t optimal_limit = aggressive_warmup(device, main_stream, 500);

    MemoryManager::initialize(optimal_limit);
    std::cout << "Memory Manager initialized with GPU memory limit: "
              << (optimal_limit / 1024.0 / 1024.0 / 1024.0) << " GB" << std::endl;
    MemoryManager *mem_mgr = &MemoryManager::getInstance();
    VirtualEvictPolicy *eviction_policy_ptr = new LRUEvictPolicy();
    mem_mgr->set_eviction_policy(eviction_policy_ptr);

    size_t host_mem_pool_size = getSafePinnedMemorySize();
    HostMemoryManager::initialize(host_mem_pool_size);

    gpuGraph dg;

    HostMemoryManager::getInstance().reset();

    vtype *d_u_candidate_vs_ = nullptr;
    numtype *d_num_u_candidates_ = nullptr;
    uint32_t *d_bitmap = nullptr;
    uint32_t bitmap_pitch = 0;
    numtype *h_num_u_candidates_ = nullptr;

    cpuGraph hq;
    hq.isQuery = true;
    cpuGraph::vlmapo2n.clear();
    cpuGraph::vlmapn2o.clear();
    readGraphToCPUStandard(&hq, job.path.c_str());

    copyMeta(&hq, &hg);

    std::unordered_set<ettype> et_set;
    hq.set_bridge_edge_mask();
    relax(&hq, et_set);

#ifndef NDEBUG
    std::cout << "=== FSSM Subgraph Similarity Matching ===" << std::endl;
    std::cout << "Query graph: " << job.path << " (V=" << hq.num_v
              << ", E=" << hq.num_e << ")" << std::endl;
    std::cout << "Data graph: " << data_path << " (V=" << hg.num_v
              << ", E=" << hg.num_e << ")" << std::endl;
    std::cout << "Generated " << et_set.size()
              << " intermediate edge combinations" << std::endl;
#endif

    auto start_time = std::chrono::high_resolution_clock::now();
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    Lattice::initialize(hq.num_e);
    Lattice *lat = &Lattice::getInstance();
    lat->construct_linked_list_extend(&hq);
    lat->compute_reachable_nodes(et_set);
    mem_mgr->init_size(lat->num_lattice_nodes);

    UnifiedTrieManager::initialize();
    UnifiedTrieManager *utm = &UnifiedTrieManager::getInstance();
    utm->init(lat->num_lattice_nodes);

    HashTable::initialize();
    HashTable *hash_table = &HashTable::getInstance();

    vtype *h_u_candidate_vs_ = new vtype[NUM_VQ * MAX_L_FREQ];
    h_num_u_candidates_ = new numtype[NUM_VQ];
    mem_mgr->set_stream(&main_stream);
    mem_mgr->allocate_permanent(d_u_candidate_vs_, sizeof(vtype) * NUM_VQ * MAX_L_FREQ);
    mem_mgr->allocate_permanent(d_num_u_candidates_, sizeof(numtype) * NUM_VQ);
    cuchk(cudaMemsetAsync(d_num_u_candidates_, 0, sizeof(numtype) * NUM_VQ, main_stream));
    cuchk(cudaStreamSynchronize(main_stream));

    start_time = std::chrono::high_resolution_clock::now();
    filterSSM(&hq, &hg, nullptr, nullptr, h_u_candidate_vs_, h_num_u_candidates_,
              d_u_candidate_vs_, d_num_u_candidates_, d_bitmap, bitmap_pitch);
#ifndef NDEBUG
    std::ofstream outfile("runtime_data/filter_output.txt");
    for (uint32_t u = 0; u < NUM_VQ; u++)
    {
      outfile << "Vertex " << u << " has " << h_num_u_candidates_[u] << " candidates: ";
      for (uint32_t i = 0; i < h_num_u_candidates_[u]; i++)
      {
        outfile << h_u_candidate_vs_[u * MAX_L_FREQ + i] << " ";
      }
      outfile << std::endl;
    }
    outfile.close();
#endif
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    std::cout << "Filter time: " << duration.count() << " us" << std::endl;

    start_time = std::chrono::high_resolution_clock::now();
    hash_table->build(&hq, h_num_u_candidates_, d_u_candidate_vs_, d_num_u_candidates_);
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    std::cout << "Build Hash Table time: " << duration.count() << " us" << std::endl;

    start_time = std::chrono::high_resolution_clock::now();
    utm->initVertexTries(d_u_candidate_vs_, h_num_u_candidates_);
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    std::cout << "initVertexTries time: " << duration.count() << " us" << std::endl;

    start_time = std::chrono::high_resolution_clock::now();
    utm->construct_edge_candidates_cpu(&hq, &hg, h_u_candidate_vs_, h_num_u_candidates_);
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    std::cout << "construct time: " << duration.count() << " us" << std::endl;

    size_t tries_bytes = utm->getTotalMemoryUsage();
    size_t graph_bytes = 0;
    graph_bytes += sizeof(degtype) * hg.num_v;
    graph_bytes += sizeof(vltype) * hg.num_v;
    graph_bytes += sizeof(offtype) * (hg.num_v + 1);
    graph_bytes += sizeof(vtype) * (hg.num_e * 2);
    std::cout << "Trie total size: " << tries_bytes << " bytes ("
              << (tries_bytes / (1024.0 * 1024.0)) << " MB)" << std::endl;
    std::cout << "Data graph size: " << graph_bytes << " bytes ("
              << (graph_bytes / (1024.0 * 1024.0)) << " MB)" << std::endl;
    std::cout << "Trie size larger than graph: "
              << (tries_bytes > graph_bytes ? "yes" : "no") << std::endl;

    mem_mgr->set_stream(&main_stream);
    mem_mgr->deallocate_permanent(d_u_candidate_vs_);
    d_u_candidate_vs_ = nullptr;
    mem_mgr->deallocate_permanent(d_num_u_candidates_);
    d_num_u_candidates_ = nullptr;
    cuchk(cudaStreamSynchronize(main_stream));

    delete[] h_u_candidate_vs_;
    h_u_candidate_vs_ = nullptr;
    delete[] h_num_u_candidates_;
    h_num_u_candidates_ = nullptr;

#ifndef NDEBUG
    std::cout << "\n=== Starting Join Operations with Transfer Optimization ===" << std::endl;
#endif

    start_time = std::chrono::high_resolution_clock::now();
    join(&hq, &hg, &dg);
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "Join Time: " << duration.count() << " us" << std::endl;
    std::cout << "Join Time: " << duration_ms.count() << " ms" << std::endl;

    std::cout << "\n=== Final Results ===" << std::endl;
    for (uint32_t rq_id : lat->computed_rq_ids)
    {
      ettype rq_et = lat->id2et[rq_id];
      uint32_t match_count = utm->getTrie(rq_id)->num_results;
      std::cout << "Relaxed Query ET " << rq_et << ": " << match_count
                << " matches" << std::endl;
    }
    std::cout << std::endl;

#ifndef NDEBUG
    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "Total computation time: " << duration.count() << " ms" << std::endl;
    std::cout << "Generated " << et_set.size() << " edge type combinations" << std::endl;
    std::cout << "GPU memory efficiently managed with transfer optimization." << std::endl;
    std::cout << "\n=== Cleanup ===" << std::endl;
    std::cout << "Cleanup completed successfully." << std::endl;
    std::cout << "\n=== FSSM Execution Completed ===" << std::endl;
#endif

    mem_mgr->set_stream(&main_stream);
    HashTable::shutdown();
    UnifiedTrieManager::shutdown();
    Lattice::shutdown();
    cuchk(cudaStreamSynchronize(main_stream));

    HostMemoryManager::getInstance().reset();
    HostMemoryManager::shutdown();
    MemoryManager::shutdown();
    cuchk(cudaStreamDestroy(main_stream));

    return 0;
  }
  catch (const std::exception &e)
  {
    std::cerr << "ERROR: Query " << job.path << " failed: " << e.what() << std::endl;
    return 1;
  }
}

static int runDirectoryBatch(const std::vector<QueryJob> &query_jobs, cpuGraph &hg,
                             const std::string &data_path, uint32_t gpu_num,
                             const std::string &eviction_policy)
{
  int success_count = 0;
  std::vector<BatchFailure> failures;

  for (const auto &job : query_jobs)
  {
    std::cout << job.id << std::endl;
    std::cout.flush();
    std::cerr.flush();
    std::fflush(nullptr);

    pid_t child_pid = fork();
    if (child_pid < 0)
    {
      std::cerr << "ERROR: fork() failed for query " << job.id << ": "
                << std::strerror(errno) << std::endl;
      return 1;
    }

    if (child_pid == 0)
    {
      const int exit_code = runSingleQuery(job, hg, data_path, gpu_num, eviction_policy);
      std::cout.flush();
      std::cerr.flush();
      std::fflush(nullptr);
      _exit(exit_code);
    }

    int status = 0;
    pid_t wait_result = -1;
    do
    {
      wait_result = waitpid(child_pid, &status, 0);
    } while (wait_result < 0 && errno == EINTR);

    if (wait_result < 0)
    {
      std::cerr << "ERROR: waitpid() failed for query " << job.id << ": "
                << std::strerror(errno) << std::endl;
      return 1;
    }

    if (WIFEXITED(status) && WEXITSTATUS(status) == 0)
    {
      success_count++;
      continue;
    }

    std::string detail;
    if (WIFEXITED(status))
      detail = "exit=" + std::to_string(WEXITSTATUS(status));
    else if (WIFSIGNALED(status))
      detail = "signal=" + std::to_string(WTERMSIG(status));
    else
      detail = "status=" + std::to_string(status);

    failures.push_back({job, detail});
    std::cout << "[batch-fail] query " << job.id
              << " file=" << fs::path(job.path).filename().string()
              << " " << detail << " continuing" << std::endl;
  }

  std::cout << "=== Batch Summary ===" << std::endl;
  std::cout << "success=" << success_count
            << " failed=" << failures.size() << std::endl;
  for (const auto &failure : failures)
  {
    std::cout << "[failed] query " << failure.job.id
              << " file=" << fs::path(failure.job.path).filename().string()
              << " " << failure.detail << std::endl;
  }

  return 0;
}

int main(int argc, char **argv)
{
  CLI::App app{
      "FSSM - Flexible Subgraph Similarity Matching with GPU Optimization"};

  std::string query_input, data_path;
  uint32_t gpu_num = 0u;
  std::string eviction_policy = "lru";
  THRESHOLD = 2;

  app.add_option("-q,--query", query_input, "Query graph file or directory")->required();
  app.add_option("-d,--data", data_path, "Data graph path")->required();
  app.add_option("-g, --gpu", gpu_num, "GPU device number (default: 0)")
      ->default_val(0);
  app.add_option("-t,--threshold", THRESHOLD,
                 "Threshold for intermediate edge combinations (default: 2)")
      ->check(CLI::Range(0, 10))
      ->default_val(2);
  app.add_option("--eviction", eviction_policy,
                 "Eviction policy: lru, contribution, hybrid (default: lru)");

  CLI11_PARSE(app, argc, argv);

  bool query_is_directory = false;
  std::vector<QueryJob> query_jobs;
  try
  {
    query_jobs = collectQueryJobs(query_input, query_is_directory);
  }
  catch (const std::exception &e)
  {
    std::cerr << "ERROR: " << e.what() << std::endl;
    return 1;
  }
  if (query_jobs.empty())
  {
    std::cerr << "ERROR: No query graphs found in: " << query_input << std::endl;
    return 1;
  }

  cpuGraph hg;
  hg.isQuery = false;
  readGraphToCPUStandard(&hg, data_path.c_str());

  if (!query_is_directory)
    return runSingleQuery(query_jobs.front(), hg, data_path, gpu_num, eviction_policy);

  return runDirectoryBatch(query_jobs, hg, data_path, gpu_num, eviction_policy);
}
