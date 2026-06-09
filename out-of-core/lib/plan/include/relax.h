#ifndef PLAN_RELAX_H
#define PLAN_RELAX_H

#include "cpuGraph.h"

#include <unordered_map>
#include <unordered_set>
#include <vector>

bool checkConnectivity(
    const ettype &cur_ET, cpuGraph *graph);

void constructRelaxedQueryGraph(
    ettype ET, cpuGraph *graph, cpuGraph *relaxedGraph);

void relax(
    cpuGraph *graph,
    std::unordered_set<ettype> &edgeTagSet);

#endif // PREPROCESS_RELAX_H