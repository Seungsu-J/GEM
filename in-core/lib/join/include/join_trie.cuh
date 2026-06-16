#ifndef JOIN_JOIN_H
#define JOIN_JOIN_H

#include "cpuGraph.h"
#include "gpuGraph.h"
#include "join_trie_kernel.cuh"
#include "lattice.h"
#include "memory_manager.cuh"
#include "unifiedTrie.cuh"

#include "host_select.cuh"
#include "direct_join.cuh"
#include "refer_join.cuh"
#include "join_context.cuh"

#include "cuda_helpers.cuh"

extern __device__ __constant__ HashLookupTables *lookup;

void process(
    cpuGraph *hq, cpuGraph *hg, gpuGraph *dg,
    UnifiedTrie *frag_trie, UnifiedTrie *probe_trie, UnifiedTrie *target_trie);

void join(
    cpuGraph *hq, cpuGraph *hg, gpuGraph *dg);

#endif //! JOIN_JOIN_H
