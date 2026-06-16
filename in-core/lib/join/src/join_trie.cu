#include "join_trie.cuh"
#include "join_trie_kernel.cuh"
#include "join_helpers.h"

#include "globals.h"
#include "lattice.h"
#include "memory_manager.cuh"
#include "unifiedTrie.cuh"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <set>
#include <stack>
#include <unordered_set>
#include <vector>
#include <numeric>
#include <cstdio>

void process(
    cpuGraph *hq, cpuGraph *hg, gpuGraph *dg,
    UnifiedTrie *frag_trie, UnifiedTrie *probe_trie, UnifiedTrie *target_trie)
{
  auto &jc = JoinContext::getInstance();
  jc.init_tries(frag_trie, probe_trie, target_trie);
  jc.process_context();

  if (jc.is_subset)
  {
    // Subset selection: both probe vertices are in fragment
    // non-tree edge must be bidirectionally valid.
#ifndef NDEBUG
    std::cout << "Performing subset selection join." << std::endl;
#endif
    jc.process_is_subset_context();
    host_select();
  }
  else if (frag_trie->mask_parent_id != UINT32_MAX)
  {
#ifndef NDEBUG
    std::cout << "Performing reference join." << std::endl;
#endif
    // Reference join: fragment has a mask parent
    jc.process_refer_join_context();
    refer_join();
  }
  else
  {
#ifndef NDEBUG
    std::cout << "Performing direct join." << std::endl;
#endif
    // Direct join: fragment has no mask parent
    jc.process_direct_join_context();
    direct_join();
  }
  // std::cout << jc.h_num_res[0] << " results found." << std::endl;
}

void join(
    cpuGraph *hq, cpuGraph *hg, gpuGraph *dg)
{
  auto &lat = Lattice::getInstance();
  auto &utm = UnifiedTrieManager::getInstance();
  auto &mem_mgr = MemoryManager::getInstance();
  auto &ht = HashTable::getInstance();
  auto &jc = JoinContext::getInstance();

  try
  {
    jc.join_context();
    jc.hq = hq;

    std::stack<int> computation_stack;
    uint32_t computation_count = 0;
    uint32_t cache_hits = 0;

    while (!lat.uncomputed_rq_ids.empty())
    {
      int cur_trie_id = -1;

      if (computation_stack.empty())
      {
        cur_trie_id = utm.find_best_edge_candidate2();
        if (cur_trie_id == UINT32_MAX)
          break;
        computation_stack.push(cur_trie_id);
      }

      while (!computation_stack.empty())
      {
        cur_trie_id = computation_stack.top();
        computation_stack.pop();

        jc.skip_final_materialization = false;

        if (computation_stack.empty())
        {
          if (lat.uncomputed_reachable_nodes[cur_trie_id].size() < lat.uncomputed_rq_ids.size())
            cur_trie_id = utm.find_best_edge_candidate2();
        }

        UnifiedTrie *cur_trie = utm.getTrie(cur_trie_id);

        // mem_mgr->record_access(cur_trie_id);

        if (lat.uncomputed_reachable_rq[cur_trie_id].empty())
        {
          mem_mgr.mark_trie_trash(cur_trie_id);
          continue;
        }

        uint32_t probe_trie_id = utm.find_best_edge_candidate2(cur_trie->et);
        if (probe_trie_id == UINT32_MAX)
          continue;

        UnifiedTrie *probe_trie = utm.getTrie(probe_trie_id);
        if (probe_trie == nullptr)
          continue;

        ettype target_trie_et = cur_trie->et | probe_trie->et;
        uint32_t target_trie_id = lat.et2id[target_trie_et];
        UnifiedTrie *target_trie = utm.getTrie(target_trie_id);
        if (target_trie == nullptr)
        {
          target_trie = utm.createTrie(
              target_trie_id, target_trie_et,
              cur_trie->num_layers + 1, cur_trie->num_cols);
        }
        else
        {
          computation_stack.push(cur_trie_id);
          computation_stack.push(target_trie_id);
        }

#ifndef NDEBUG
        std::cout << "processing " << cur_trie_id << " and " << probe_trie_id
                  << " to produce " << target_trie_id << std::endl;
#endif
        mem_mgr.ensure_trie(cur_trie_id);
        mem_mgr.pin_trie(cur_trie_id);
        mem_mgr.pin_trie(target_trie_id, false);
        mem_mgr.record_trie_access(cur_trie_id);
        mem_mgr.record_trie_access(probe_trie_id);
        mem_mgr.record_trie_access(target_trie_id);

        jc.skip_final_materialization = (lat.uncomputed_reachable_rq[target_trie_id].size() == 0);

        process(hq, hg, dg, cur_trie, probe_trie, target_trie);

        lat.mark_computed(target_trie_id);
        computation_count++;

        lat.update_contribution_on_computation(target_trie_et);

        mem_mgr.unpin_trie(cur_trie_id);
        // mem_mgr.unpin_trie(probe_trie_id);
        // if (lat.uncomputed_reachable_rq[target_trie_id].empty())
        mem_mgr.unpin_trie(target_trie_id, false);

        computation_stack.push(cur_trie_id);
        computation_stack.push(target_trie_id);
      }
    }
  }
  catch (...)
  {
    jc.cleanup();
    throw;
  }

  jc.cleanup();
}
