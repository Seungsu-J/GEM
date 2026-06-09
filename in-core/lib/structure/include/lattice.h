#ifndef STRUCTURE_LATTICE_H
#define STRUCTURE_LATTICE_H

#include <bitset>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "cpuGraph.h"
#include "globals.h"
#include "defs.h"

class Lattice
{
public:
	static Lattice &getInstance();
	static void initialize(int num_edges);
	static void shutdown();

	int num_edges;
	ettype max_bit;
	int num_lattice_nodes;

	ettype src;
	std::vector<std::vector<int>> linked_list;					// store one's children nodes. (subset)
	std::vector<std::vector<int>> reversed_linked_list; // store one's parents (superset)
	std::unordered_map<ettype, int> et2id;
	std::unordered_map<int, ettype> id2et;
	std::unordered_set<uint32_t> rq_id_set;

	/* node-level tables for regular nodes */
	std::vector<std::unordered_set<uint32_t>> reachable_nodes; // what nodes can I contribute to?
	std::vector<std::unordered_set<uint32_t>> computed_reachable_nodes;
	std::vector<std::unordered_set<uint32_t>> uncomputed_reachable_nodes;
	std::vector<uint32_t> num_reachable_nodes; // how many nodes can I contribute to?

	/* node-level tables for relaxed queries */
	std::vector<std::unordered_set<uint32_t>> reachable_rq;
	std::vector<std::unordered_set<uint32_t>> computed_reachable_rq;
	std::vector<std::unordered_set<uint32_t>> uncomputed_reachable_rq;
	std::vector<uint32_t> num_reachable_rq; // how many relaxed queries can I contribute to?

	/* Globally computed/uncomputed nodes/rqs */
	std::unordered_set<uint32_t> computed_rq_ids;
	std::unordered_set<uint32_t> uncomputed_rq_ids;
	std::unordered_set<uint32_t> computed_node_ids;
	std::unordered_set<uint32_t> uncomputed_node_ids;

	std::unordered_map<uint32_t, double> contribution_value_cache;

	// Enhanced contribution management

	void construct_linked_list_extend(cpuGraph *graph);
	void construct_linked_list(cpuGraph *graph);
	void compute_reachable_nodes(const std::unordered_set<ettype> &rq_et_set);
	inline uint32_t get_vertex_node_id(vtype u) { return num_lattice_nodes - NUM_VQ + u; }

	// Contribution management methods
	void update_contribution_on_computation(ettype computed_et);
	int get_current_contribution_count(ettype et);
	double get_contribution_value(ettype et);
	void invalidate_contribution_cache(ettype et);

	// update status methods
	void mark_computed(uint32_t lat_node_id);
	void mark_uncomputed(uint32_t lat_node_id);

	// query methods
	bool is_rq(uint32_t lat_node_id);
	bool is_computed(uint32_t lat_node_id);
	bool is_uncomputed(uint32_t lat_node_id);
	uint32_t get_global_computed_rq_count();
	uint32_t get_global_uncomputed_rq_count();
	uint32_t get_my_computed_rq_count(uint32_t lat_node_id);
	uint32_t get_my_uncomputed_rq_count(uint32_t lat_node_id);
	uint32_t get_my_computed_node_count(uint32_t lat_node_id);
	uint32_t get_my_uncomputed_node_count(uint32_t lat_node_id);

private:
	explicit Lattice(int);

	static std::unique_ptr<Lattice> instance;
	static std::mutex instance_mutex;
};

#endif //! STRUCTURE_LATTICE_H
