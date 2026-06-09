#include "io.h"
#include "globals.h"
#include "cuda_helpers.cuh"
#include "FastFileReader.hpp"

#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <stdlib.h>
#include <cstdio>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <memory>
#include <cassert>
#include <cstdint> // For uintptr_t

#include <algorithm>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <map>
#include <set>

using std::cerr;
using std::cout;
using std::endl;

// Helper to align offsets
auto align_up = [](size_t size, size_t alignment)
{
	return (size + alignment - 1) & ~(alignment - 1);
};

/**
 * Optimized graph reader with fast I/O techniques
 * - Buffered file reading for better performance
 * - Memory pre-allocation to avoid reallocations
 * - Optimized data structures for faster lookups
 * - Bulk operations for better cache utilization
 */
void readGraphToCPUStandard(
		cpuGraph *graph,
		const char *filename)
{
	try
	{
		FastFileReader reader(filename);

		// Read header with fast I/O
		char type = reader.read_char();
		numtype num_v = reader.read_integer<numtype>();
		numtype num_e = reader.read_integer<numtype>();

		if (type != 't')
		{
			cerr << "Error: Expected 't' at the beginning, found " << type << endl;
			exit(1);
		}

		graph->num_v = num_v;
		graph->num_e = num_e;

		// Pre-allocate all vectors with exact sizes to avoid reallocations
		graph->vertexIDs_.resize(num_v);
		graph->offsets_.resize(num_v + 1, 0);

		const size_t edge_count = num_e << 1;
		graph->neighbors_.resize(edge_count);
		graph->edgeIDs_.resize(edge_count);
		graph->in_degree_.resize(num_v, 0);
		graph->out_degree_.resize(num_v, 0);
		graph->vLabels_.resize(num_v);

		std::vector<offtype> offs(num_v, 0);
		graph->offsets_[0] = 0;
		etype eid_global = 0;

		int vlcount = cpuGraph::vlmapo2n.size();		// now 0
		int tot_labels = cpuGraph::vlmapn2o.size(); // now 0

		// Batch read vertices for better performance
		std::vector<std::tuple<vtype, vltype, degtype>> vertices;
		std::vector<std::pair<vtype, vtype>> edges;

		vertices.reserve(num_v);
		edges.reserve(num_e);

		// Read all data in one pass with proper EOF handling
		char current_type;
		while (true)
		{
			current_type = reader.read_char();
			if (current_type == EOF)
			{
				break; // End of file reached normally
			}

			// Skip whitespace characters
			if (current_type == ' ' || current_type == '\t' || current_type == '\n' || current_type == '\r')
			{
				continue;
			}

			if (current_type == 'v')
			{
				vtype vid = reader.read_integer<vtype>();
				vltype vLabel = reader.read_integer<vltype>();
				degtype deg = reader.read_integer<degtype>();
				vertices.emplace_back(vid, vLabel, deg);
			}
			else if (current_type == 'e')
			{
				vtype src = reader.read_integer<vtype>();
				vtype dst = reader.read_integer<vtype>();
				edges.emplace_back(src, dst);
			}
			else
			{
				cerr << "Error: Invalid type '" << (int)current_type << "' (char: '" << current_type << "')" << endl;
				cerr << "Expected 'v' for vertex or 'e' for edge" << endl;
				exit(1);
			}
		}

		// Process vertices in batch
		for (const auto &vertex_data : vertices)
		{
			auto [vid, vLabel, deg] = vertex_data;
			bool useful = true;

			if (graph->isQuery)
			{
				auto it = cpuGraph::vlmapo2n.find(vLabel);
				if (it == cpuGraph::vlmapo2n.end())
				{
					cpuGraph::vlmapo2n[vLabel] = vlcount;
					cpuGraph::vlmapn2o[vlcount] = vLabel;
					vLabel = vlcount++;
				}
				else
				{
					vLabel = it->second;
				}
			}
			else
			{
				auto it = cpuGraph::vlmapo2n.find(vLabel);
				if (it == cpuGraph::vlmapo2n.end())
				{
					vLabel = tot_labels;
					useful = false;
				}
				else
				{
					vLabel = it->second;
				}
			}

			graph->vertexIDs_[vid] = vid;
			graph->vLabels_[vid] = vLabel;
			graph->in_degree_[vid] = deg;
			graph->out_degree_[vid] = deg;
			graph->maxDegree = std::max(graph->maxDegree, deg);

			if (useful)
				graph->vLabelFreq[vLabel]++;

			graph->offsets_[vid + 1] = graph->offsets_[vid] + deg;
		}

		// Process edges in batch
		for (const auto &edge_data : edges)
		{
			auto [src, dst] = edge_data;

			if (graph->isQuery)
			{
				graph->vve.emplace(std::make_pair(src, dst), eid_global);
				graph->evv.emplace(eid_global, std::make_pair(src, dst));
				graph->vve.emplace(std::make_pair(dst, src), eid_global + 1);
				graph->evv.emplace(eid_global + 1, std::make_pair(dst, src));
			}

			offtype off = graph->offsets_[src] + offs[src];
			graph->neighbors_[off] = dst;
			graph->edgeIDs_[off] = eid_global;
			offs[src]++;

			off = graph->offsets_[dst] + offs[dst];
			graph->neighbors_[off] = src;
			graph->edgeIDs_[off] = eid_global + 1;
			offs[dst]++;

			eid_global += 2;
		}

		graph->num_v_labels = graph->vLabelFreq.size();
		if (!graph->isQuery)
		{
			graph->num_v_labels++;
		}

		// graph->maxLabelFreq = 0;
		for (const auto &l_c : graph->vLabelFreq)
			graph->maxLabelFreq = std::max(graph->maxLabelFreq, l_c.second);

		if (graph->isQuery)
			for (int i = 0; i < graph->num_e; ++i)
				graph->edgeTag.set(i);
	}
	catch (const std::exception &e)
	{
		cerr << "Error reading graph file: " << e.what() << endl;
		exit(1);
	}
}

void readGraphToCPU_CStyle(
		cpuGraph *graph,
		const char *filename)
{
	try
	{
		FastFileReader reader(filename);

		// Fast header reading
		char type = reader.read_char();
		numtype num_v = reader.read_integer<numtype>();
		numtype num_e = reader.read_integer<numtype>();

		if (type != 't')
		{
			cerr << "Error: Expected 't' at the beginning, found " << type << endl;
			exit(1);
		}

		graph->num_v = num_v;
		graph->num_e = num_e;

		// Pre-allocate with exact sizes
		graph->vertexIDs_.resize(num_v);

		graph->offsets_.resize(num_v + 1, 0);

		const size_t edge_count = num_e << 1;
		graph->neighbors_.resize(edge_count);

		graph->edgeIDs_.resize(edge_count);

		graph->in_degree_.resize(num_v, 0);

		graph->out_degree_.resize(num_v, 0);

		graph->vLabels_.resize(num_v);

		std::vector<offtype> offs(num_v, 0);
		graph->offsets_[0] = 0;
		etype eid_global = 0;

		// Batch processing for better performance with proper EOF handling
		char current_type;
		while (true)
		{
			current_type = reader.read_char();
			if (current_type == EOF)
			{
				break; // End of file reached normally
			}

			// Skip whitespace characters
			if (current_type == ' ' || current_type == '\t' || current_type == '\n' || current_type == '\r')
			{
				continue;
			}

			if (current_type == 'v')
			{
				vtype vid = reader.read_integer<vtype>();
				vltype vLabel = reader.read_integer<vltype>();
				degtype deg = reader.read_integer<degtype>();

				graph->vertexIDs_[vid] = vid;
				graph->vLabels_[vid] = vLabel;
				graph->in_degree_[vid] = deg;
				graph->out_degree_[vid] = deg;
				graph->maxDegree = std::max(graph->maxDegree, deg);
				graph->vLabelFreq[vLabel]++;
				graph->offsets_[vid + 1] = graph->offsets_[vid] + deg;
			}
			else if (current_type == 'e')
			{
				vtype src = reader.read_integer<vtype>();
				vtype dst = reader.read_integer<vtype>();

				if (graph->isQuery)
				{
					graph->vve.emplace(std::make_pair(src, dst), eid_global);
					graph->evv.emplace(eid_global, std::make_pair(src, dst));
					graph->vve.emplace(std::make_pair(dst, src), eid_global + 1);
					graph->evv.emplace(eid_global + 1, std::make_pair(dst, src));
				}

				offtype off = graph->offsets_[src] + offs[src];
				graph->neighbors_[off] = dst;
				graph->edgeIDs_[off] = eid_global;
				offs[src]++;

				off = graph->offsets_[dst] + offs[dst];
				graph->neighbors_[off] = src;
				graph->edgeIDs_[off] = eid_global + 1;
				offs[dst]++;

				eid_global += 2;
			}
			else
			{
				cerr << "Error: Invalid type '" << (int)current_type << "' (char: '" << current_type << "')" << endl;
				cerr << "Expected 'v' for vertex or 'e' for edge" << endl;
				exit(1);
			}
		}

		for (const auto &l_c : graph->vLabelFreq)
		{
			graph->maxLabelFreq = std::max(graph->maxLabelFreq, l_c.second);
		}

		if (graph->isQuery)
		{
			for (int i = 0; i < graph->num_e; ++i)
			{
				graph->edgeTag.set(i);
			}
		}
	}
	catch (const std::exception &e)
	{
		cerr << "Error reading graph file: " << e.what() << endl;
		exit(1);
	}
}

// Optimized GPU memory allocation with error checking and performance improvements
void allocateMemGPU(
		gpuGraph *gpuGraph,
		const cpuGraph *cpuGraph)
{
	const uint32_t num = cpuGraph->isQuery ? NUM_VQ : cpuGraph->num_v;
	const size_t edge_count = cpuGraph->num_e << 1;

	// Allocate memory with proper alignment and error checking
	try
	{
		// Define the alignment you want. 128 is a common safe choice for GPU memory.
		const size_t alignment = 128;

		// Calculate aligned offsets one by one
		size_t degree_offset = 0;
		size_t degree_size = sizeof(degtype) * num;

		size_t vlabel_offset = align_up(degree_offset + degree_size, alignment);
		size_t vlabel_size = sizeof(vltype) * num;

		size_t offset_list_offset = align_up(vlabel_offset + vlabel_size, alignment);
		size_t offset_list_size = sizeof(offtype) * (num + 1);

		size_t neighbor_offset = align_up(offset_list_offset + offset_list_size, alignment);
		size_t neighbor_size = sizeof(vtype) * edge_count;

		// The total memory to allocate is the start of the last array plus its size
		size_t mem_to_allocate = neighbor_offset + neighbor_size;

		// Use char* for byte-level pointer arithmetic
		char *dev_space = nullptr;
		cudaStream_t stream;
		cuchk(cudaStreamCreate(&stream));
		cuchk(cudaMallocAsync(reinterpret_cast<void **>(&dev_space), mem_to_allocate, stream));
		cuchk(cudaStreamSynchronize(stream));
		cuchk(cudaStreamDestroy(stream));

		// Assign pointers using the calculated aligned offsets
		gpuGraph->degree_ = reinterpret_cast<degtype *>(dev_space + degree_offset);
		gpuGraph->vLabels_ = reinterpret_cast<vltype *>(dev_space + vlabel_offset);
		gpuGraph->offsets_ = reinterpret_cast<offtype *>(dev_space + offset_list_offset);
		gpuGraph->neighbors_ = reinterpret_cast<vtype *>(dev_space + neighbor_offset);

		// Initialize allocated memory to zero for better debugging
		//! no need to initialize memory to zero here, as it will be filled later by `copyGraphToGPU()`

		// cuchk(cudaMemset(gpuGraph->degree_, 0, degree_size));
		// cuchk(cudaMemset(gpuGraph->vLabels_, 0, vlabel_size));
		// cuchk(cudaMemset(gpuGraph->offsets_, 0, offset_size));
		// cuchk(cudaMemset(gpuGraph->neighbors_, 0, neighbor_size));
	}
	catch (const std::exception &e)
	{
		cerr << "Error allocating GPU memory: " << e.what() << endl;
		exit(1);
	}
}

// Optimized GPU memory copy operations with async transfers and proper error handling
void copyGraphToGPU(
		gpuGraph *gpuGraph,
		const cpuGraph *cpuGraph)
{
	const uint32_t num = cpuGraph->isQuery ? NUM_VQ : cpuGraph->num_v;
	const size_t edge_count = cpuGraph->num_e * 2;

	// Create CUDA stream for asynchronous operations
	cudaStream_t stream;
	cuchk(cudaStreamCreate(&stream));

	try
	{
		// Use async memory copies for better performance
		cuchk(cudaMemcpyAsync(gpuGraph->degree_,
													cpuGraph->out_degree_.data(),
													sizeof(degtype) * num,
													cudaMemcpyHostToDevice, stream));

		cuchk(cudaMemcpyAsync(gpuGraph->vLabels_,
													cpuGraph->vLabels_.data(),
													sizeof(vltype) * num,
													cudaMemcpyHostToDevice, stream));

		cuchk(cudaMemcpyAsync(gpuGraph->offsets_,
													cpuGraph->offsets_.data(),
													sizeof(offtype) * (num + 1),
													cudaMemcpyHostToDevice, stream));

		cuchk(cudaMemcpyAsync(gpuGraph->neighbors_,
													cpuGraph->neighbors_.data(),
													sizeof(vtype) * edge_count,
													cudaMemcpyHostToDevice, stream));

		// Synchronize to ensure all transfers complete
		cuchk(cudaStreamSynchronize(stream));
	}
	catch (const std::exception &e)
	{
		cerr << "Error copying graph to GPU: " << e.what() << endl;
		exit(1);
	}

	// Clean up stream
	cuchk(cudaStreamDestroy(stream));
}

// Optimized GPU to CPU copy with async operations
void copyGraphToCPU(
		gpuGraph *gpuGraph,
		cpuGraph *cpuGraph)
{
	const uint32_t num = NUM_VQ;
	const size_t edge_count = cpuGraph->num_e * 2;

	// Create CUDA stream for asynchronous operations
	cudaStream_t stream;
	cuchk(cudaStreamCreate(&stream));

	try
	{
		// Use async memory copies for better performance
		cuchk(cudaMemcpyAsync(cpuGraph->out_degree_.data(),
													gpuGraph->degree_,
													sizeof(degtype) * num,
													cudaMemcpyDeviceToHost, stream));

		cuchk(cudaMemcpyAsync(cpuGraph->vLabels_.data(),
													gpuGraph->vLabels_,
													sizeof(vltype) * num,
													cudaMemcpyDeviceToHost, stream));

		cuchk(cudaMemcpyAsync(cpuGraph->offsets_.data(),
													gpuGraph->offsets_,
													sizeof(offtype) * (num + 1),
													cudaMemcpyDeviceToHost, stream));

		cuchk(cudaMemcpyAsync(cpuGraph->neighbors_.data(),
													gpuGraph->neighbors_,
													sizeof(vtype) * edge_count,
													cudaMemcpyDeviceToHost, stream));

		// Synchronize to ensure all transfers complete
		cuchk(cudaStreamSynchronize(stream));
	}
	catch (const std::exception &e)
	{
		cerr << "Error copying graph from GPU: " << e.what() << endl;
		exit(1);
	}

	// Clean up stream
	cuchk(cudaStreamDestroy(stream));
}

// Optimized metadata copy with better error handling and performance
void copyMeta(cpuGraph *query, cpuGraph *data)
{
	// Cache frequently used values
	NUM_VQ = query->num_v;
	NUM_EQ = query->num_e;
	NUM_VLQ = query->num_v_labels;

	MAX_L_FREQ = data->maxLabelFreq;
	MAX_DATA_DEGREE = data->maxDegree;

	NUM_VD = data->num_v;
	NUM_ED = data->num_e;
	COL_LEN = (NUM_VD - 1) / 32 + 1;

	// Batch copy all constants to GPU device memory for better performance
	try
	{
		// Create array of values for batch copy
		const uint32_t constants[] = {
				NUM_VQ, NUM_EQ, NUM_VLQ, MAX_L_FREQ, MAX_DATA_DEGREE,
				NUM_VD, NUM_ED, COL_LEN, QUERY_TYPE, THRESHOLD};

		// Use async copies for better performance
		cudaStream_t stream;
		cuchk(cudaStreamCreate(&stream));

		cuchk(cudaMemcpyToSymbolAsync(C_NUM_VQ, &constants[0], sizeof(uint32_t), 0,
																	cudaMemcpyHostToDevice, stream));
		cuchk(cudaMemcpyToSymbolAsync(C_NUM_EQ, &constants[1], sizeof(uint32_t), 0,
																	cudaMemcpyHostToDevice, stream));
		cuchk(cudaMemcpyToSymbolAsync(C_NUM_VLQ, &constants[2], sizeof(uint32_t), 0,
																	cudaMemcpyHostToDevice, stream));
		cuchk(cudaMemcpyToSymbolAsync(C_MAX_L_FREQ, &constants[3], sizeof(uint32_t), 0,
																	cudaMemcpyHostToDevice, stream));
		cuchk(cudaMemcpyToSymbolAsync(C_MAX_DEGREE, &constants[4], sizeof(uint32_t), 0,
																	cudaMemcpyHostToDevice, stream));
		cuchk(cudaMemcpyToSymbolAsync(C_NUM_VD, &constants[5], sizeof(uint32_t), 0,
																	cudaMemcpyHostToDevice, stream));
		cuchk(cudaMemcpyToSymbolAsync(C_NUM_ED, &constants[6], sizeof(uint32_t), 0,
																	cudaMemcpyHostToDevice, stream));
		cuchk(cudaMemcpyToSymbolAsync(C_COL_LEN, &constants[7], sizeof(uint32_t), 0,
																	cudaMemcpyHostToDevice, stream));
		cuchk(cudaMemcpyToSymbolAsync(C_QUERY_TYPE, &constants[8], sizeof(uint32_t), 0,
																	cudaMemcpyHostToDevice, stream));
		cuchk(cudaMemcpyToSymbolAsync(C_THRESHOLD, &constants[9], sizeof(uint32_t), 0,
																	cudaMemcpyHostToDevice, stream));

		// Synchronize to ensure all copies complete
		cuchk(cudaStreamSynchronize(stream));
		cuchk(cudaStreamDestroy(stream));
	}
	catch (const std::exception &e)
	{
		cerr << "Error copying metadata to GPU: " << e.what() << endl;
		exit(1);
	}
}