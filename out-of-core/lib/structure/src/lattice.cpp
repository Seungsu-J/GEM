#include "lattice.h"
#include "defs.h"
#include "globals.h"

#include <algorithm>
#include <assert.h>
#include <iostream>
#include <queue>
#include <unordered_map>
#include <unordered_set>

std::unique_ptr<Lattice> Lattice::instance = nullptr;
std::mutex Lattice::instance_mutex;

Lattice &Lattice::getInstance()
{
  std::lock_guard<std::mutex> lock(instance_mutex);
  assert(instance && "Lattice singleton not initialized");
  return *instance;
}

void Lattice::initialize(int num_edges)
{
  std::lock_guard<std::mutex> lock(instance_mutex);
  instance = std::unique_ptr<Lattice>(new Lattice(num_edges));
}

void Lattice::shutdown()
{
  std::lock_guard<std::mutex> lock(instance_mutex);
  instance.reset();
}

Lattice::Lattice(int num_edges)
    : num_edges(num_edges), max_bit(ettype().set() >> (ET_SIZE - num_edges))
{
  num_lattice_nodes = 0;
  src = max_bit;
}

void Lattice::construct_linked_list_extend(cpuGraph *graph)
{
  if (graph == nullptr)
    return;

  // Enhanced reservation strategy based on empirical analysis
  // Extended lattice includes one-vertex nodes, so more nodes than regular lattice
  int estimated_lattice_size;
  if (num_edges <= 10)
    estimated_lattice_size = std::min(100, 1 << num_edges);
  else if (num_edges <= 20)
    estimated_lattice_size = std::min(2000, num_edges * num_edges * 5);
  else
    estimated_lattice_size = std::min(20000, num_edges * num_edges * 10);

  linked_list.reserve(estimated_lattice_size);

  std::vector<std::unordered_set<ettype>> visited_tag_vec;
  visited_tag_vec.reserve(num_edges + 1); // Reserve enough space for all levels
  visited_tag_vec.resize(2);              // Start with two levels

  ettype FULL_ET = ettype().set();
  std::queue<ettype> rq_queue;
  rq_queue.push(src);
  rq_queue.push(FULL_ET);
  visited_tag_vec[0].insert(src);
  et2id[src] = 0; // src is the first tag, index 0
  id2et[0] = src;

  int level = 0; // # missing edges

  while (!rq_queue.empty())
  {
    ettype cur = rq_queue.front();
    rq_queue.pop();

    if (cur == FULL_ET)
    {
      if (rq_queue.empty())
        break;
      level++;
      visited_tag_vec.push_back(std::unordered_set<ettype>());
      rq_queue.push(FULL_ET);
      continue;
    }

    // Only increment for actual lattice nodes
    ++num_lattice_nodes;

#ifndef NDEBUG
    // std::cout << "cur = " << cur << ", lattice_node_id = " << num_lattice_nodes-1 << std::endl;
#endif

    if (level + 1 == num_edges) // final
    {
      continue;
    }

    for (int i = 0; i < num_edges; ++i)
    {
      if (cur[i] == 0)
        continue;

      // create new tag
      ettype new_tag = cur;
      new_tag.flip(i); // remove edge i

      if (visited_tag_vec[level + 1].count(new_tag))
      {
        linked_list[et2id[cur]].push_back(et2id[new_tag]);
      }
      else if (graph->check_connectivity(new_tag) == true)
      {
        int temp = et2id.size();
        et2id[new_tag] = temp;
        id2et[temp] = new_tag;
        visited_tag_vec[level + 1].insert(new_tag);
        rq_queue.push(new_tag);

        if (linked_list.size() <= et2id[new_tag])
        {
          linked_list.resize(et2id[new_tag] + 1);
        }
        linked_list[et2id[cur]].push_back(et2id[new_tag]);
      }
    }
  }
  if (linked_list.size() < num_lattice_nodes + NUM_VQ)
    linked_list.resize(num_lattice_nodes + NUM_VQ);
  for (int i = 0; i < NUM_VQ; ++i)
    id2et[num_lattice_nodes + i] = ettype().set(); // one-vertex nodes
  for (uint32_t e = 0; e < NUM_EQ; ++e)
  {
    uint32_t node_id = et2id[ettype().set(e)];
    auto [u, v] = graph->evv[e << 1];
    linked_list[node_id].push_back(u + num_lattice_nodes);
    linked_list[node_id].push_back(v + num_lattice_nodes);
  }

  num_lattice_nodes += NUM_VQ; // add one-vertex nodes

  for (int i = 0; i < num_lattice_nodes; ++i)
  {
    if (linked_list[i].size())
    {
      std::sort(linked_list[i].begin(), linked_list[i].end());
    }
  }

// Post-construction analysis for memory optimization
#ifndef NDEBUG
  std::cout << "Lattice construction completed:" << std::endl;
  std::cout << "  - Final lattice nodes: " << num_lattice_nodes << std::endl;
  std::cout << "  - Reserved capacity: " << linked_list.capacity() << std::endl;
  std::cout << "  - Memory efficiency: "
            << (100.0 * num_lattice_nodes / linked_list.capacity()) << "%"
            << std::endl;

  if (num_lattice_nodes > linked_list.capacity() * 0.8)
  {
    std::cout << "  - Warning: Consider increasing reservation for num_edges="
              << num_edges << std::endl;
  }

  std::cout << "num_lattice_nodes: " << num_lattice_nodes << std::endl;
  std::cout << "size: " << et2id.size() << std::endl;
  ASSERT_EQ(num_lattice_nodes, et2id.size() + NUM_VQ); // Alternative: simpler syntax
  std::cout << "num_lattice_nodes assert passed" << std::endl;
#endif

  // construct reversed linked list
  reversed_linked_list.resize(num_lattice_nodes);
  for (int i = 0; i < num_lattice_nodes; ++i)
  {
    if (linked_list[i].size())
      for (auto j : linked_list[i])
      {
        reversed_linked_list[j].push_back(i);
      }
  }
  for (int i = 0; i < num_lattice_nodes; ++i)
    if (reversed_linked_list[i].size())
      std::sort(reversed_linked_list[i].begin(), reversed_linked_list[i].end());
}

void Lattice::construct_linked_list(cpuGraph *graph)
{
  if (graph == nullptr)
    return;

  // Enhanced reservation strategy based on empirical analysis
  // Lattice size is typically much smaller than 2^num_edges due to connectivity
  // constraints
  int estimated_lattice_size;
  if (num_edges <= 10)
    estimated_lattice_size = std::min(50, 1 << num_edges);
  else if (num_edges <= 20)
    estimated_lattice_size = std::min(1000, num_edges * num_edges * 3);
  else
    estimated_lattice_size = std::min(10000, num_edges * num_edges * 5);

  linked_list.reserve(estimated_lattice_size);

  std::vector<std::unordered_set<ettype>> visited_tag_vec;
  visited_tag_vec.reserve(num_edges + 1); // Reserve enough space for all levels
  visited_tag_vec.resize(2);              // Start with two levels

  ettype FULL_ET = ettype().set();
  std::queue<ettype> rq_queue;
  rq_queue.push(src);
  rq_queue.push(FULL_ET);
  visited_tag_vec[0].insert(src);
  et2id[src] = 0; // src is the first tag, index 0
  id2et[0] = src;

  int level = 0; // # missing edges

  while (!rq_queue.empty())
  {
    ettype cur = rq_queue.front();
    rq_queue.pop();

    if (cur == FULL_ET)
    {
      if (rq_queue.empty())
        break;
      level++;
      visited_tag_vec.push_back(std::unordered_set<ettype>());
      rq_queue.push(FULL_ET);
      continue;
    }

    // Only increment for actual lattice nodes
    ++num_lattice_nodes;

#ifndef NDEBUG
    std::cout << "cur = " << cur
              << ", num_lattice_nodes = " << num_lattice_nodes << std::endl;
#endif

    if (level + 1 == num_edges) // final
    {
      continue;
    }

    for (int i = 0; i < num_edges; ++i)
    {
      if (cur[i] == 0)
        continue;

      // create new tag
      ettype new_tag = cur;
      new_tag.flip(i); // remove edge i

      if (visited_tag_vec[level + 1].count(new_tag))
      {
        linked_list[et2id[cur]].push_back(et2id[new_tag]);
      }
      else if (graph->check_connectivity(new_tag) == true)
      {
        int temp = et2id.size();
        et2id[new_tag] = temp;
        id2et[temp] = new_tag;
        visited_tag_vec[level + 1].insert(new_tag);
        rq_queue.push(new_tag);

        if (linked_list.size() <= et2id[new_tag])
        {
          linked_list.resize(et2id[new_tag] + 1);
        }
        linked_list[et2id[cur]].push_back(et2id[new_tag]);
      }
    }
  }
  visited_tag_vec.back().insert(ettype(0));

  for (int i = 0; i < num_lattice_nodes; ++i)
  {
    if (linked_list[i].size())
    {
      std::sort(linked_list[i].begin(), linked_list[i].end());
    }
  }

// Post-construction analysis for memory optimization
#ifndef NDEBUG
  std::cout << "Lattice construction completed:" << std::endl;
  std::cout << "  - Final lattice nodes: " << num_lattice_nodes << std::endl;
  std::cout << "  - Reserved capacity: " << linked_list.capacity() << std::endl;
  std::cout << "  - Memory efficiency: "
            << (100.0 * num_lattice_nodes / linked_list.capacity()) << "%"
            << std::endl;

  if (num_lattice_nodes > linked_list.capacity() * 0.8)
  {
    std::cout << "  - Warning: Consider increasing reservation for num_edges="
              << num_edges << std::endl;
  }

  std::cout << "num_lattice_nodes: " << num_lattice_nodes << std::endl;
  std::cout << "size: " << et2id.size() << std::endl;
  ASSERT_EQ(num_lattice_nodes, et2id.size()); // Alternative: simpler syntax
  std::cout << "num_lattice_nodes assert passed" << std::endl;
#endif

  // construct reversed linked list
  reversed_linked_list.resize(num_lattice_nodes);
  for (int i = 0; i < num_lattice_nodes; ++i)
  {
    if (linked_list[i].size())
      for (auto j : linked_list[i])
      {
        reversed_linked_list[j].push_back(i);
      }
  }
  for (int i = 0; i < num_lattice_nodes; ++i)
    if (reversed_linked_list[i].size())
      std::sort(reversed_linked_list[i].begin(), reversed_linked_list[i].end());
}

void Lattice::compute_reachable_nodes(
    const std::unordered_set<ettype> &rq_et_set)
{
#ifndef NDEBUG
  std::cout << "num_lattice_nodes: " << num_lattice_nodes << std::endl;
#endif

  for (auto &et : rq_et_set)
    rq_id_set.insert(et2id[et]);

  reachable_nodes.resize(num_lattice_nodes);
  computed_reachable_nodes.resize(num_lattice_nodes);
  uncomputed_reachable_nodes.resize(num_lattice_nodes);
  num_reachable_nodes.resize(num_lattice_nodes, 0);

  reachable_rq.resize(num_lattice_nodes);
  computed_reachable_rq.resize(num_lattice_nodes);
  uncomputed_reachable_rq.resize(num_lattice_nodes);
  num_reachable_rq.resize(num_lattice_nodes, 0);

  // For each node in the lattice, perform DFS to find reachable nodes
  int src_id = et2id[src];
  ASSERT_EQ(src_id, 0); // Simple and clean
  std::queue<int> q;
  bool vis[num_lattice_nodes] = {false};
  q.push(src_id);
  vis[src_id] = true;

  while (!q.empty())
  {
    auto cur_id = q.front();
    q.pop();

    num_reachable_nodes[cur_id] = reachable_nodes[cur_id].size();
    num_reachable_rq[cur_id] = reachable_rq[cur_id].size();
    uncomputed_reachable_nodes[cur_id] = reachable_nodes[cur_id];
    uncomputed_reachable_rq[cur_id] = reachable_rq[cur_id];
    uncomputed_node_ids.insert(cur_id);
    if (is_rq(cur_id))
      uncomputed_rq_ids.insert(cur_id);

    if (linked_list[cur_id].empty()) // skip leaf nodes
      continue;

    for (auto next_id : linked_list[cur_id])
    {
#ifndef NDEBUG
      // std::cout << "cur_id = " << cur_id << ", next_id = " << next_id << std::endl;
#endif
      reachable_nodes[next_id].insert(reachable_nodes[cur_id].begin(), reachable_nodes[cur_id].end());
      reachable_nodes[next_id].insert(cur_id);

      reachable_rq[next_id].insert(reachable_rq[cur_id].begin(), reachable_rq[cur_id].end());
      if (is_rq(cur_id))
        reachable_rq[next_id].insert(cur_id);

      if (!vis[next_id])
      {
        q.push(next_id);
        vis[next_id] = true;
      }
    }
  }

#ifndef NDEBUG
  std::cout << "compute reachable nodes done." << std::endl;
#endif
}

void Lattice::update_contribution_on_computation(ettype computed_et)
{
  // erase et from all nodes' uncomputed_set
  int computed_id = et2id[computed_et];
  bool is_rq_flag = rq_id_set.count(computed_id) > 0;

  std::queue<int> q;
  bool vis[num_lattice_nodes] = {false};
  q.push(computed_id);
  vis[computed_id] = true;

  while (!q.empty())
  {
    int cur_id = q.front();
    q.pop();

    if (linked_list[cur_id].empty()) // skip leaf nodes
      continue;

    for (auto next_id : linked_list[cur_id])
    {
      if (is_rq_flag)
      {
        computed_reachable_rq[next_id].insert(computed_id);
        uncomputed_reachable_rq[next_id].erase(computed_id);

        // Invalidate cache for next_id since its contribution value changed
        // When a reachable RQ is computed, all nodes that can reach it
        // have their contribution values decreased
        contribution_value_cache.erase(next_id);
      }
      // computed_reachable_nodes[next_id].insert(computed_id);
      // uncomputed_reachable_nodes[next_id].erase(computed_id);
      if (!vis[next_id])
      {
        q.push(next_id);
        vis[next_id] = true;
      }
    }
  }

  // Also invalidate the computed node itself
  contribution_value_cache.erase(computed_id);
}

int Lattice::get_current_contribution_count(ettype et)
{
  int et_id = et2id[et];
  return uncomputed_reachable_rq[et_id].size();
}

double Lattice::get_contribution_value(ettype et)
{

  // Check cache first
  int current_id = et2id[et];
  auto cache_it = contribution_value_cache.find(current_id);
  if (cache_it != contribution_value_cache.end())
  {
    return cache_it->second;
  }

  // Compute contribution value with decay factor for distant targets
  double total_value = 0.0;

  for (int target_id : uncomputed_reachable_rq[current_id])
  {
    ettype target_et = id2et[target_id];
    int distance = (target_et ^ et).count();
    double decay_factor = 1.0 / (1.0 + 0.1 * distance);
    total_value += decay_factor;
  }

  contribution_value_cache[current_id] = total_value;
  return total_value;
}

void Lattice::invalidate_contribution_cache(ettype et) { contribution_value_cache.erase(et2id[et]); }

void Lattice::mark_computed(uint32_t lat_node_id)
{
  computed_node_ids.insert(lat_node_id);
  uncomputed_node_ids.erase(lat_node_id);
  if (is_rq(lat_node_id))
  {
    computed_rq_ids.insert(lat_node_id);
    uncomputed_rq_ids.erase(lat_node_id);
  }
}

void Lattice::mark_uncomputed(uint32_t lat_node_id)
{
  uncomputed_node_ids.insert(lat_node_id);
  computed_node_ids.erase(lat_node_id);
  if (is_rq(lat_node_id))
  {
    uncomputed_rq_ids.insert(lat_node_id);
    computed_rq_ids.erase(lat_node_id);
  }
}

bool Lattice::is_rq(uint32_t lat_node_id) { return rq_id_set.count(lat_node_id) > 0; }

bool Lattice::is_computed(uint32_t lat_node_id) { return computed_node_ids.count(lat_node_id) > 0; }

bool Lattice::is_uncomputed(uint32_t lat_node_id) { return uncomputed_node_ids.count(lat_node_id) > 0; }

uint32_t Lattice::get_global_computed_rq_count() { return computed_rq_ids.size(); }

uint32_t Lattice::get_global_uncomputed_rq_count() { return uncomputed_rq_ids.size(); }

uint32_t Lattice::get_my_computed_rq_count(uint32_t lat_node_id)
{
  return computed_reachable_rq[lat_node_id].size();
}

uint32_t Lattice::get_my_uncomputed_rq_count(uint32_t lat_node_id)
{
  return uncomputed_reachable_rq[lat_node_id].size();
}

uint32_t Lattice::get_my_computed_node_count(uint32_t lat_node_id)
{
  return computed_reachable_nodes[lat_node_id].size();
}

uint32_t Lattice::get_my_uncomputed_node_count(uint32_t lat_node_id)
{
  return uncomputed_reachable_nodes[lat_node_id].size();
}
