#include <algorithm>
#include <charconv>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <thread>

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

struct QueryJob {
  uint32_t id;
  std::string path;
};

static bool isFatalCudaError(cudaError_t code) {
  switch (code) {
  case cudaErrorIllegalAddress:
  case cudaErrorAssert:
  case cudaErrorLaunchFailure:
  case cudaErrorLaunchTimeout:
  case cudaErrorECCUncorrectable:
  case cudaErrorContextIsDestroyed:
  case cudaErrorUnknown:
  case cudaErrorCudartUnloading:
    return true;
  default:
    return false;
  }
}

static void clearGpuGraph(gpuGraph &graph) {
  graph.degree_ = nullptr;
  graph.vLabels_ = nullptr;
  graph.offsets_ = nullptr;
  graph.neighbors_ = nullptr;
  graph.edgeIDs_ = nullptr;
}

static bool tryParseQueryId(const fs::path &query_path, uint32_t &out_id) {
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
  auto res =
      std::from_chars(digits.data(), digits.data() + digits.size(), value);
  if (res.ec != std::errc() || res.ptr != digits.data() + digits.size())
    return false;

  out_id = value;
  return true;
}

static std::vector<QueryJob> collectQueryJobs(const std::string &query_input,
                                              bool &is_directory) {
  std::vector<QueryJob> jobs;
  std::error_code ec;
  fs::path input_path(query_input);

  if (fs::is_regular_file(input_path, ec)) {
    is_directory = false;
    uint32_t query_id = 0;
    if (!tryParseQueryId(input_path, query_id))
      query_id = 0u;
    jobs.push_back({query_id, input_path.string()});
    return jobs;
  }

  if (!fs::is_directory(input_path, ec)) {
    throw std::runtime_error("Query path is not a file or directory: " +
                             query_input);
  }

  is_directory = true;

  std::unordered_set<uint32_t> seen_ids;
  for (const auto &entry : fs::directory_iterator(input_path)) {
    if (!entry.is_regular_file())
      continue;

    const fs::path p = entry.path();
    if (p.extension() != ".in")
      continue;

    uint32_t query_id = 0;
    if (!tryParseQueryId(p, query_id)) {
      throw std::runtime_error(
          "Query file does not match expected pattern Q_<id>.in: " +
          p.string());
    }

    if (!seen_ids.insert(query_id).second) {
      throw std::runtime_error("Duplicate query id " +
                               std::to_string(query_id) +
                               " in directory: " + query_input);
    }

    jobs.push_back({query_id, p.string()});
  }

  std::sort(jobs.begin(), jobs.end(),
            [](const QueryJob &a, const QueryJob &b) { return a.id < b.id; });
  return jobs;
}

int main(int argc, char **argv) {
  CLI::App app{"GEM - Revisiting GPU-Accelerated Edge-Centric Subgraph "
               "Matching on Large Graphs"};

  std::string query_input, data_path;
  uint32_t gpu_num = 0u;
  std::string eviction_policy = "lru";
  THRESHOLD = 0;

  app.add_option("-q,--query", query_input, "Query graph file or directory")
      ->required();
  app.add_option("-d,--data", data_path, "Data graph path")->required();
  app.add_option("-g, --gpu", gpu_num, "GPU device number (default: 0)")
      ->default_val(0);
  // app.add_option("-t,--threshold", THRESHOLD,
  //                "Threshold for intermediate edge combinations (default: 0)")
  //     ->check(CLI::Range(0, 10))
  //     ->default_val(0);
  app.add_option("--eviction", eviction_policy,
                 "Eviction policy: lru, contribution, hybrid (default: lru)");

  CLI11_PARSE(app, argc, argv);

  bool query_is_directory = false;
  std::vector<QueryJob> query_jobs;
  query_jobs = collectQueryJobs(query_input, query_is_directory);
  if (query_jobs.empty()) {
    std::cerr << "ERROR: No query graphs found in: " << query_input
              << std::endl;
    return 1;
  }

  const int device = static_cast<int>(gpu_num);
  cudaDeviceProp prop;
  cuchk(cudaGetDeviceProperties(&prop, device));
  SHARED_MEMORY_LIMIT = prop.sharedMemPerBlock;
  cuchk(cudaSetDevice(device));

#ifndef NDEBUG
  cout << "Device " << device << ": " << prop.name << endl;
#endif
  GPU_NUM = gpu_num;

  cpuGraph hg;
  hg.isQuery = false;
  readGraphToCPUStandard(&hg, data_path.c_str());

  /* ==============================> HOST_MEM_MGR
   * <============================== */
  const size_t safe_host_pinned_budget = getSafePinnedMemorySize();
  if (safe_host_pinned_budget == 0) {
    std::cerr << "ERROR: Safe pinned host memory budget is 0; refusing to run."
              << std::endl;
    return 1;
  }

  size_t host_mem_pool_size =
      1ULL * 1024 * 1024 * 1024; // 1 GB default pinned pool
  if (host_mem_pool_size > safe_host_pinned_budget)
    host_mem_pool_size = safe_host_pinned_budget;

  cudaStream_t main_stream = nullptr;
  size_t optimal_limit = 0;
  MemoryManager *mem_mgr = nullptr;
  VirtualEvictPolicy *eviction_policy_ptr = nullptr;

  gpuGraph dg;
  clearGpuGraph(dg);

  auto initRuntime = [&]() {
    if (main_stream != nullptr) {
      cudaStreamDestroy(main_stream);
      main_stream = nullptr;
    }

    // init cuda context and memory pool.
    cuchk(cudaStreamCreate(&main_stream));
    optimal_limit = aggressive_warmup(device, main_stream, 500);

    /* ==============================> MEM_MGR <==============================*/
    // Create MemoryManager for advanced memory management
    MemoryManager::initialize(optimal_limit);
    std::cout << "Memory Manager initialized with GPU memory limit: "
              << (optimal_limit / 1024.0 / 1024.0 / 1024.0) << " GB"
              << std::endl;
    mem_mgr = &MemoryManager::getInstance();
    eviction_policy_ptr = new LRUEvictPolicy();
    mem_mgr->set_eviction_policy(eviction_policy_ptr);

    /* ==============================> HOST_MEM_MGR
     * <============================== */
    std::cout << "host mem pool size: "
              << (host_mem_pool_size / 1024.0 / 1024.0 / 1024.0) << " GB"
              << std::endl;
    HostMemoryManager::initialize(host_mem_pool_size);
    std::cout
        << "Host Memory Manager initialized with pinned memory pool size: "
        << (host_mem_pool_size / 1024.0 / 1024.0 / 1024.0) << " GB"
        << std::endl;

    // Initialize host memory expansion time tracking
    HOST_MEM_EXPANSION_TIME = 0.0;

    clearGpuGraph(dg);
    allocateMemGPU(&dg, &hg);
    copyGraphToGPU(&dg, &hg);
  };

  auto resetDeviceAndReload = [&]() {
    std::cerr << "Recovering from CUDA failure: resetting device " << device
              << std::endl;

    // Best-effort cleanup; this may fail if CUDA is in an error state.
    try {
      HostMemoryManager::shutdown();
    } catch (...) {
    }
    try {
      MemoryManager::shutdown();
    } catch (...) {
    }
    eviction_policy_ptr = nullptr;
    mem_mgr = nullptr;

    if (main_stream != nullptr) {
      cudaStreamDestroy(main_stream);
      main_stream = nullptr;
    }

    cudaGetLastError();
    cudaDeviceReset();
    cudaError_t set_err = cudaErrorUnknown;
    for (int attempt = 0; attempt < 50; ++attempt) {
      set_err = cudaSetDevice(device);
      if (set_err == cudaSuccess)
        break;

      std::cerr << "WARNING: cudaSetDevice(" << device
                << ") failed: " << cudaGetErrorString(set_err) << " (attempt "
                << (attempt + 1) << "/50)" << std::endl;
      cudaGetLastError();
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    if (set_err != cudaSuccess) {
      throw CudaException(set_err, __FILE__, __LINE__);
    }

    initRuntime();
  };

  try {
    initRuntime();
  } catch (const std::exception &e) {
    std::cerr << "ERROR: Failed to initialize runtime: " << e.what()
              << std::endl;
    try {
      HostMemoryManager::shutdown();
    } catch (...) {
    }
    try {
      MemoryManager::shutdown();
    } catch (...) {
    }
    if (main_stream != nullptr) {
      cudaStreamDestroy(main_stream);
      main_stream = nullptr;
    }
    return 1;
  }

  bool had_any_failures = false;
  int exit_code = 0;

  for (const auto &job : query_jobs) {
    if (query_is_directory)
      std::cout << job.id << std::endl;

    // Clear pools and per-query time tracking
    try {
      HostMemoryManager::getInstance().reset();
    } catch (...) {
    }
    HOST_MEM_EXPANSION_TIME = 0.0;
    cudaGetLastError();

    vtype *d_u_candidate_vs_ = nullptr;
    numtype *d_num_u_candidates_ = nullptr;
    uint32_t *d_bitmap = nullptr;
    uint32_t bitmap_pitch = 0;
    numtype *h_num_u_candidates_ = nullptr;

    gpuGraph dq;
    clearGpuGraph(dq);

    auto cleanupQuery = [&]() noexcept {
      try {
        if (h_num_u_candidates_ != nullptr) {
          delete[] h_num_u_candidates_;
          h_num_u_candidates_ = nullptr;
        }
      } catch (...) {
      }

      try {
        if (mem_mgr != nullptr && main_stream != nullptr) {
          mem_mgr->set_stream(&main_stream);
          if (d_u_candidate_vs_ != nullptr)
            mem_mgr->deallocate_permanent(d_u_candidate_vs_);
          if (d_num_u_candidates_ != nullptr)
            mem_mgr->deallocate_permanent(d_num_u_candidates_);
          if (dq.degree_ != nullptr)
            mem_mgr->deallocate_permanent(dq.degree_);
          if (d_bitmap != nullptr)
            mem_mgr->deallocate_temporal(d_bitmap);
          cudaStreamSynchronize(main_stream);
        }
      } catch (...) {
      }

      clearGpuGraph(dq);
      d_u_candidate_vs_ = nullptr;
      d_num_u_candidates_ = nullptr;
      d_bitmap = nullptr;

      try {
        HashTable::shutdown();
      } catch (...) {
      }
      try {
        UnifiedTrieManager::shutdown();
      } catch (...) {
      }
      try {
        Lattice::shutdown();
      } catch (...) {
      }

      // Best-effort: reclaim any remaining tracked allocations from the
      // MemoryManager, but keep the persistent data graph buffer (dg.degree_ is
      // the base pointer).
      try {
        if (mem_mgr != nullptr && main_stream != nullptr) {
          mem_mgr->set_stream(&main_stream);
          mem_mgr->reclaim_all_except(dg.degree_);
        }
      } catch (...) {
      }
    };

    bool case_ok = true;
    bool reset_required = false;

    try {
      cpuGraph hq;
      hq.isQuery = true;
      cpuGraph::vlmapo2n.clear();
      cpuGraph::vlmapn2o.clear();
      readGraphToCPUStandard(&hq, job.path.c_str());

      // recognize(&hq);

      copyMeta(&hq, &hg);

      allocateMemGPU(&dq, &hq);
      copyGraphToGPU(&dq, &hq);

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
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
          end_time - start_time);
      auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
          end_time - start_time);

      /* ==============================> Lattice <==============================
       */
      Lattice::initialize(hq.num_e);
      Lattice *lat = &Lattice::getInstance();
      lat->construct_linked_list_extend(&hq); // include vertex-lattice nodes.
      // lat->construct_linked_list(&hq);        // only edge-lattice nodes.
      lat->compute_reachable_nodes(et_set);
      mem_mgr->init_size(lat->num_lattice_nodes);

      /* ==============================> UTM <============================== */
      UnifiedTrieManager::initialize();
      UnifiedTrieManager *utm = &UnifiedTrieManager::getInstance();
      utm->init(lat->num_lattice_nodes);

      /* ==============================> HT <============================== */
      HashTable::initialize();
      HashTable *hash_table = &HashTable::getInstance();

      /* >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> FILTER
       * <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< */
      vtype *h_u_candidate_vs_ = nullptr; // useless.
      h_num_u_candidates_ = new numtype[NUM_VQ];

      mem_mgr->set_stream(&main_stream);
      mem_mgr->allocate_permanent(d_u_candidate_vs_,
                                  sizeof(vtype) * NUM_VQ * MAX_L_FREQ);
      mem_mgr->allocate_permanent(d_num_u_candidates_,
                                  sizeof(numtype) * NUM_VQ);
      cuchk(cudaMemsetAsync(d_num_u_candidates_, 0, sizeof(numtype) * NUM_VQ,
                            main_stream));
      cuchk(cudaStreamSynchronize(main_stream));

      start_time = std::chrono::high_resolution_clock::now();
      filterSSM(&hq, &hg, &dq, &dg, h_u_candidate_vs_, h_num_u_candidates_,
                d_u_candidate_vs_, d_num_u_candidates_, d_bitmap, bitmap_pitch);
#ifndef NDEBUG
      std::ofstream outfile("runtime_data/filter_output.txt");
      h_u_candidate_vs_ = new vtype[NUM_VQ * MAX_L_FREQ];
      cuchk(cudaMemcpyAsync(h_u_candidate_vs_, d_u_candidate_vs_,
                            sizeof(vtype) * NUM_VQ * MAX_L_FREQ,
                            cudaMemcpyDeviceToHost, main_stream));
      cuchk(cudaStreamSynchronize(main_stream));
      for (uint32_t u = 0; u < NUM_VQ; u++) {
        outfile << "Vertex " << u << " has " << h_num_u_candidates_[u]
                << " candidates: ";
        for (uint32_t i = 0; i < h_num_u_candidates_[u]; i++) {
          outfile << h_u_candidate_vs_[u * MAX_L_FREQ + i] << " ";
        }
        outfile << std::endl;
      }
      outfile.close();
      delete[] h_u_candidate_vs_;
#endif
      end_time = std::chrono::high_resolution_clock::now();
      duration = std::chrono::duration_cast<std::chrono::microseconds>(
          end_time - start_time);
      std::cout << "Filter time: " << duration.count() << " us" << std::endl;

      /* >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> BUILD HASH TABLE
       * <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< */
      start_time = std::chrono::high_resolution_clock::now();
      hash_table->build(&hq, h_num_u_candidates_, d_u_candidate_vs_,
                        d_num_u_candidates_);
      end_time = std::chrono::high_resolution_clock::now();
      duration = std::chrono::duration_cast<std::chrono::microseconds>(
          end_time - start_time);
      std::cout << "Build Hash Table time: " << duration.count() << " us"
                << std::endl;

      start_time = std::chrono::high_resolution_clock::now();
      utm->initVertexTries(d_u_candidate_vs_, h_num_u_candidates_);
      end_time = std::chrono::high_resolution_clock::now();
      duration = std::chrono::duration_cast<std::chrono::microseconds>(
          end_time - start_time);
      std::cout << "initVertexTries time: " << duration.count() << " us"
                << std::endl;

      /* >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> FIRST JOIN
       * <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< */
      start_time = std::chrono::high_resolution_clock::now();
      utm->construct_edge_candidates(&hq, &hg, &dg, d_u_candidate_vs_,
                                     d_num_u_candidates_, h_num_u_candidates_,
                                     d_bitmap, bitmap_pitch);
      end_time = std::chrono::high_resolution_clock::now();
      duration = std::chrono::duration_cast<std::chrono::microseconds>(
          end_time - start_time);
      std::cout << "construct time: " << duration.count() << " us" << std::endl;

      mem_mgr->set_stream(&main_stream);
      mem_mgr->deallocate_permanent(d_u_candidate_vs_);
      d_u_candidate_vs_ = nullptr;
      mem_mgr->deallocate_permanent(d_num_u_candidates_);
      d_num_u_candidates_ = nullptr;
      mem_mgr->deallocate_permanent(dq.degree_);
      clearGpuGraph(dq);
      mem_mgr->deallocate_temporal(d_bitmap);
      d_bitmap = nullptr;
      cuchk(cudaStreamSynchronize(main_stream));

#ifndef NDEBUG
      std::cout
          << "\n=== Starting Join Operations with Transfer Optimization ==="
          << std::endl;
#endif

      start_time = std::chrono::high_resolution_clock::now();
      join(&hq, &hg, &dg);

      end_time = std::chrono::high_resolution_clock::now();
      duration = std::chrono::duration_cast<std::chrono::microseconds>(
          end_time - start_time);
      duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
          end_time - start_time);

      // Calculate net join time (excluding host memory expansion overhead)
      double net_join_time_us =
          duration.count() - (HOST_MEM_EXPANSION_TIME * 1000.0);
      double net_join_time_ms = duration_ms.count() - HOST_MEM_EXPANSION_TIME;

      std::cout << "Join Time: " << net_join_time_us << " us" << std::endl;
      std::cout << "Join Time: " << net_join_time_ms << " ms" << std::endl;

      std::cout << "\n=== Final Results ===" << std::endl;
      for (uint32_t rq_id : lat->computed_rq_ids) {
        ettype rq_et = lat->id2et[rq_id];
        uint32_t match_count = utm->getTrie(rq_id)->num_results;
        std::cout << match_count << " matches" << std::endl;
        // std::cout << "Relaxed Query ET " << rq_et << ": " << match_count
        // << " matches" << std::endl;
      }
      std::cout << std::endl;

#ifndef NDEBUG
      std::cout << "\n=== Summary ===" << std::endl;
      std::cout << "Total computation time: " << duration.count() << " ms"
                << std::endl;
      std::cout << "Generated " << et_set.size() << " edge type combinations"
                << std::endl;
      std::cout << "GPU memory efficiently managed with transfer optimization."
                << std::endl;

      // Cleanup
      std::cout << "\n=== Cleanup ===" << std::endl;
      std::cout << "Cleanup completed successfully." << std::endl;
      std::cout << "\n=== FSSM Execution Completed ===" << std::endl;
#endif
    } catch (const CudaException &e) {
      case_ok = false;
      reset_required = isFatalCudaError(e.code());
      std::cerr << "ERROR: Query " << job.id << " failed: " << e.what()
                << std::endl;
    } catch (const std::exception &e) {
      case_ok = false;
      cudaError_t last_err = cudaGetLastError();
      if (last_err != cudaSuccess && isFatalCudaError(last_err))
        reset_required = true;

      cudaError_t sync_err = cudaDeviceSynchronize();
      if (sync_err != cudaSuccess && isFatalCudaError(sync_err))
        reset_required = true;
      cudaGetLastError();

      std::cerr << "ERROR: Query " << job.id << " failed: " << e.what()
                << std::endl;
    }

    if (!case_ok)
      had_any_failures = true;

    cleanupQuery();

    if (!case_ok) {
      if (!query_is_directory) {
        exit_code = 1;
        break;
      }

      if (reset_required) {
        try {
          resetDeviceAndReload();
        } catch (const std::exception &e) {
          std::cerr << "ERROR: Failed to recover CUDA context: " << e.what()
                    << std::endl;
          exit_code = 1;
          break;
        }
      }
      continue;
    }
  }

  try {
    if (mem_mgr != nullptr && main_stream != nullptr) {
      mem_mgr->set_stream(&main_stream);
      if (dg.degree_ != nullptr)
        mem_mgr->deallocate_permanent(dg.degree_);
      clearGpuGraph(dg);
      cuchk(cudaStreamSynchronize(main_stream));
    }
  } catch (const std::exception &e) {
    std::cerr << "WARNING: Failed to cleanup data graph GPU buffers: "
              << e.what() << std::endl;
  }

  try {
    HostMemoryManager::shutdown();
  } catch (const std::exception &e) {
    std::cerr << "WARNING: HostMemoryManager shutdown failed: " << e.what()
              << std::endl;
  }

  try {
    MemoryManager::shutdown();
  } catch (const std::exception &e) {
    std::cerr << "WARNING: MemoryManager shutdown failed: " << e.what()
              << std::endl;
  }

  if (main_stream != nullptr) {
    try {
      cuchk(cudaStreamDestroy(main_stream));
    } catch (...) {
    }
    main_stream = nullptr;
  }

  if (exit_code == 0 && query_is_directory && had_any_failures)
    exit_code = 1;

  return exit_code;
}
