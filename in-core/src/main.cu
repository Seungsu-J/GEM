#include <chrono>
#include <iostream>

#include "CLI11.hpp"
#include "cpuGraph.h"
#include "cuda_helpers.cuh"
#include "filter.h"
#include "globals.h"
#include "io.h"
#include "join_trie.cuh"
#include "lattice.h"
#include "memory_manager.h"
#include "relax.h"
#include "unifiedTrie.cuh"

#include <unordered_map>
#include <unordered_set>
#include <vector>

using std::cout;
using std::endl;

int main(int argc, char **argv)
{
  CLI::App app{
      "FSSM - Flexible Subgraph Similarity Matching with GPU Optimization"};

  std::string query_path, data_path;
  uint32_t gpu_num = 0u;
  std::string eviction_policy = "lru";
  THRESHOLD = 2;

  app.add_option("-q,--query", query_path, "Query graph path")->required();
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

  int device = gpu_num;
  cuchk(cudaSetDevice(device));
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);
  SHARED_MEMORY_LIMIT = prop.sharedMemPerBlock;

#ifndef NDEBUG
  cout << "Device " << device << ": " << prop.name << endl;
#endif
  // init cuda context and memory pool.
  cudaStream_t main_stream;
  cuchk(cudaStreamCreate(&main_stream));
  aggressive_warmup(device, main_stream, 500);

  GPU_NUM = gpu_num;

  cpuGraph hq, hg;
  gpuGraph dq, dg;
  hq.isQuery = true;
  hg.isQuery = false;
  readGraphToCPUStandard(&hq, query_path.c_str());
  readGraphToCPUStandard(&hg, data_path.c_str());

  // recognize(&hq);

  copyMeta(&hq, &hg);

  allocateMemGPU(&dg, &hg);
  allocateMemGPU(&dq, &hq);
  copyGraphToGPU(&dg, &hg);
  copyGraphToGPU(&dq, &hq);

  /// some test on graph
  // uint32_t total_tiles = 0;
  // for (int i = 0; i < NUM_VD; ++i)
  // {
  //   total_tiles += (hg.out_degree_[i] + 31) / 32;
  // }
  // std::cout << "Data graph total tiles (32 deg each): " << total_tiles << std::endl;

  // return 0;

  std::unordered_set<ettype> et_set;

  hq.set_bridge_edge_mask();

  relax(&hq, et_set);

#ifndef NDEBUG
  std::cout << "=== FSSM Subgraph Similarity Matching ===" << std::endl;
  std::cout << "Query graph: " << query_path << " (V=" << hq.num_v
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

  /* ==============================> Lattice <============================== */
  Lattice::initialize(hq.num_e);
  Lattice *lat = &Lattice::getInstance();
  lat->construct_linked_list_extend(&hq); // include vertex-lattice nodes.
  // lat->construct_linked_list(&hq);        // only edge-lattice nodes.
  lat->compute_reachable_nodes(et_set);

  /* ==============================> UTM <============================== */
  UnifiedTrieManager::initialize();
  UnifiedTrieManager *utm = &UnifiedTrieManager::getInstance();
  utm->init(lat->num_lattice_nodes);

  /* ==============================> HT <============================== */
  // EdgeCandidateManager *ecm = new EdgeCandidateManager();
  HashTable::initialize();
  HashTable *hash_table = &HashTable::getInstance();

  /* ==============================> MEM_MGR <============================== */
  // Create MemoryManager for advanced memory management
  MemoryManager *mem_mgr = &MemoryManager::getInstance();
  mem_mgr->set_eviction_policy(eviction_policy);
  size_t free_memory, total_memory;
  cuchk(cudaMemGetInfo(&free_memory, &total_memory));
  size_t optimal_limit = std::min(
      free_memory * 0.85, total_memory * 0.7); // Conservative but efficient
  mem_mgr->set_gpu_memory_limit(optimal_limit);

  /* >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> FILTER <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< */
  vtype *h_u_candidate_vs_ = nullptr; // useless.
  numtype *h_num_u_candidates_ = new numtype[NUM_VQ];
  vtype *d_u_candidate_vs_;
  numtype *d_num_u_candidates_;
  cuchk(cudaMallocAsync(&d_u_candidate_vs_, sizeof(vtype) * NUM_VQ * MAX_L_FREQ, main_stream));
  cuchk(cudaMallocAsync(&d_num_u_candidates_, sizeof(numtype) * NUM_VQ, main_stream));
  cuchk(cudaMemsetAsync(d_num_u_candidates_, 0, sizeof(numtype) * NUM_VQ, main_stream));
  cuchk(cudaStreamSynchronize(main_stream));
  uint32_t *d_bitmap, bitmap_pitch;

  start_time = std::chrono::high_resolution_clock::now();
  filterSSM(&hq, &hg, &dq, &dg, h_u_candidate_vs_, h_num_u_candidates_,
            d_u_candidate_vs_, d_num_u_candidates_, d_bitmap, bitmap_pitch);
#ifndef NDEBUG
  std::ofstream outfile("runtime_data/filter_output.txt");
  h_u_candidate_vs_ = new vtype[NUM_VQ * MAX_L_FREQ];
  cuchk(cudaMemcpyAsync(h_u_candidate_vs_, d_u_candidate_vs_,
                        sizeof(vtype) * NUM_VQ * MAX_L_FREQ, cudaMemcpyDeviceToHost, main_stream));
  cuchk(cudaStreamSynchronize(main_stream));
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
  delete[] h_u_candidate_vs_;
#endif
  end_time = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
  std::cout << "Filter time: " << duration.count() << " us" << std::endl;

  /* >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> BUILD HASH TABLE <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< */
  start_time = std::chrono::high_resolution_clock::now();
  hash_table->build(&hq, h_num_u_candidates_, d_u_candidate_vs_,
                    d_num_u_candidates_);
  end_time = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
  std::cout << "Build Hash Table time: " << duration.count() << " us" << std::endl;

  start_time = std::chrono::high_resolution_clock::now();
  utm->initVertexTries(d_u_candidate_vs_, h_num_u_candidates_);
  end_time = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
  std::cout << "initVertexTries time: " << duration.count() << " us" << std::endl;

  /* >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> FIRST JOIN <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< */
  start_time = std::chrono::high_resolution_clock::now();
  utm->construct_edge_candidates(
      &hq, &hg, &dg,
      d_u_candidate_vs_, d_num_u_candidates_, h_num_u_candidates_,
      d_bitmap, bitmap_pitch);
  end_time = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
  std::cout << "construct time: " << duration.count() << " us" << std::endl;

  // return 0;

  cuchk(cudaFreeAsync(d_num_u_candidates_, main_stream));
  cuchk(cudaFreeAsync(d_u_candidate_vs_, main_stream));
  cuchk(cudaFreeAsync(dg.degree_, main_stream));
  cuchk(cudaFreeAsync(d_bitmap, main_stream));
  cuchk(cudaStreamSynchronize(main_stream));
  // cuchk(cudaMemcpyAsync(h_u_candidate_vs_, d_u_candidate_vs_, sizeof(vtype) *
  // NUM_VQ * MAX_L_FREQ, cudaMemcpyDeviceToHost, cpy_stream));
  // cuchk(cudaFreeAsync(ecm->d_u_candidates_v_, cpy_stream));
  // cuchk(cudaFreeAsync(ecm->d_num_u_candidates_, cpy_stream));

#ifndef NDEBUG
  std::cout << "\n=== Starting Join Operations with Transfer Optimization ==="
            << std::endl;
#endif
  // im->initialize_total_rq_count(); // TODO: Add equivalent for
  // LayeredTrieManager if needed

  // free gpu graph memory.

  start_time = std::chrono::high_resolution_clock::now();
  // Call the enhanced join function with transfer minimization

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
  std::cout << "Total computation time: " << duration.count() << " ms"
            << std::endl;
  std::cout << "Generated " << et_set.size() << " edge type combinations"
            << std::endl;
  // std::cout << "Edge candidate tries created: " << ecm->get_num_tries()
  // << std::endl;
  std::cout << "GPU memory efficiently managed with transfer optimization."
            << std::endl;

  // GPU memory statistics
  cudaMemGetInfo(&free_memory, &total_memory);
  size_t used_memory = total_memory - free_memory;
  std::cout << "Final GPU memory usage: " << (used_memory / 1024.0 / 1024.0)
            << " MB / " << (total_memory / 1024.0 / 1024.0) << " MB"
            << std::endl;

  // Cleanup
  std::cout << "\n=== Cleanup ===" << std::endl;

  // Clean up intermediate manager and lattice
  // if (im->lat)
  //   delete im->lat;
  // delete im;

  // Free candidate arrays
  // if (ecm->h_num_u_candidates_)
  //   delete[] ecm->h_num_u_candidates_;

  std::cout << "Cleanup completed successfully." << std::endl;
  std::cout << "\n=== FSSM Execution Completed ===" << std::endl;

#endif

  return 0;
}
