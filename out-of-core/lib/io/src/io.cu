#include "io.h"
#include "globals.h"
#include "cuda_helpers.cuh"
#include "FastFileReader.hpp"
#include "memory_manager.cuh"

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
#include <limits>

#include <algorithm>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <map>
#include <set>

using std::cerr;
using std::cout;
using std::endl;

namespace
{
constexpr uint64_t kVertexProgressInterval = 50'000'000ull;
constexpr uint64_t kEdgeProgressInterval = 250'000'000ull;

size_t estimateResidentBytes(vtype num_v, uint64_t num_e, bool is_query)
{
	const size_t vertex_count = static_cast<size_t>(num_v);
	const size_t edge_count = static_cast<size_t>(num_e * 2ull);

	size_t bytes = 0;
	bytes += sizeof(offtype) * (vertex_count + 1);
	bytes += sizeof(vtype) * edge_count;
	bytes += sizeof(degtype) * vertex_count; // out_degree_
	bytes += sizeof(vltype) * vertex_count;

	if (is_query)
	{
		bytes += sizeof(vtype) * vertex_count; // vertexIDs_
		bytes += sizeof(degtype) * vertex_count; // in_degree_
		bytes += sizeof(etype) * edge_count; // edgeIDs_
	}

	return bytes;
}
} // namespace

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
	readGraphToCPU_CStyle(graph, filename);
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
		uint64_t raw_num_v = reader.read_integer<uint64_t>();
		uint64_t raw_num_e = reader.read_integer<uint64_t>();

		if (type != 't')
		{
			cerr << "Error: Expected 't' at the beginning, found " << type << endl;
			exit(1);
		}

		if (raw_num_v > std::numeric_limits<vtype>::max())
		{
			throw std::runtime_error("Vertex count exceeds 32-bit vertex id limit: " + std::to_string(raw_num_v));
		}
		if (graph->isQuery && raw_num_e > ET_SIZE)
		{
			throw std::runtime_error("Query edge count exceeds ET_SIZE: " + std::to_string(raw_num_e));
		}
		if (raw_num_e > (std::numeric_limits<size_t>::max() / 2ull))
		{
			throw std::runtime_error("Edge count is too large for this address space: " + std::to_string(raw_num_e));
		}

		const vtype num_v = static_cast<vtype>(raw_num_v);
		const uint64_t num_e = raw_num_e;
		const size_t edge_count = static_cast<size_t>(num_e * 2ull);
		const bool store_query_metadata = graph->isQuery;

		graph->num_v = num_v;
		graph->num_e = num_e;
		graph->num_v_labels = 0;
		graph->maxDegree = 0;
		graph->maxLabelFreq = 0;
		graph->vLabelFreq.clear();
		graph->vve.clear();
		graph->evv.clear();

		if (!graph->isQuery)
		{
			const double resident_gib =
					static_cast<double>(estimateResidentBytes(num_v, num_e, false)) /
					(1024.0 * 1024.0 * 1024.0);
			cout << "Loading data graph: " << filename << " (V=" << raw_num_v
					 << ", E=" << raw_num_e << ", estimated resident CSR=" << resident_gib
					 << " GiB)" << endl;
		}

		// Pre-allocate with exact sizes
		if (store_query_metadata)
		{
			graph->vertexIDs_.assign(num_v, 0);
			graph->edgeIDs_.resize(edge_count);
			graph->in_degree_.assign(num_v, 0);
		}
		else
		{
			std::vector<vtype>().swap(graph->vertexIDs_);
			std::vector<etype>().swap(graph->edgeIDs_);
			std::vector<degtype>().swap(graph->in_degree_);
		}

		graph->offsets_.assign(static_cast<size_t>(num_v) + 1, 0);
		graph->neighbors_.resize(edge_count);
		graph->out_degree_.assign(num_v, 0);
		graph->vLabels_.assign(num_v, 0);

		std::vector<degtype> offs(static_cast<size_t>(num_v), 0);
		graph->offsets_[0] = 0;
		etype eid_global = 0;
		uint64_t vertices_read = 0;
		uint64_t edges_read = 0;
		uint64_t next_vertex_progress = kVertexProgressInterval;
		uint64_t next_edge_progress = kEdgeProgressInterval;
		int vlcount = static_cast<int>(cpuGraph::vlmapo2n.size());

		// Stream directly into CSR buffers to avoid duplicating large graphs in memory.
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

				if (vid >= num_v)
				{
					throw std::runtime_error("Vertex id out of bounds: " + std::to_string(vid));
				}

				if (graph->isQuery)
				{
					auto it = cpuGraph::vlmapo2n.find(vLabel);
					if (it == cpuGraph::vlmapo2n.end())
					{
						cpuGraph::vlmapo2n[vLabel] = vlcount;
						cpuGraph::vlmapn2o[vlcount] = vLabel;
						vLabel = static_cast<vltype>(vlcount++);
					}
					else
					{
						vLabel = it->second;
					}
					graph->vertexIDs_[vid] = vid;
					graph->in_degree_[vid] = deg;
				}

				graph->vLabels_[vid] = vLabel;
				graph->out_degree_[vid] = deg;
				graph->maxDegree = std::max(graph->maxDegree, deg);
				graph->vLabelFreq[vLabel]++;
				graph->offsets_[vid + 1] = graph->offsets_[vid] + deg;
				++vertices_read;

				if (!graph->isQuery && vertices_read >= next_vertex_progress)
				{
					cout << "Data graph load progress: vertices " << vertices_read << " / "
							 << raw_num_v << ", edges " << edges_read << " / " << raw_num_e
							 << endl;
					next_vertex_progress += kVertexProgressInterval;
				}
			}
			else if (current_type == 'e')
			{
				vtype src = reader.read_integer<vtype>();
				vtype dst = reader.read_integer<vtype>();

				if (src >= num_v || dst >= num_v)
				{
					throw std::runtime_error("Edge endpoint out of bounds: (" +
																 std::to_string(src) + ", " +
																 std::to_string(dst) + ")");
				}

				if (store_query_metadata)
				{
					graph->vve.emplace(std::make_pair(src, dst), eid_global);
					graph->evv.emplace(eid_global, std::make_pair(src, dst));
					graph->vve.emplace(std::make_pair(dst, src), eid_global + 1);
					graph->evv.emplace(eid_global + 1, std::make_pair(dst, src));
				}

				offtype off = graph->offsets_[src] + offs[src];
				graph->neighbors_[static_cast<size_t>(off)] = dst;
				if (store_query_metadata)
					graph->edgeIDs_[static_cast<size_t>(off)] = eid_global;
				offs[src]++;

				off = graph->offsets_[dst] + offs[dst];
				graph->neighbors_[static_cast<size_t>(off)] = src;
				if (store_query_metadata)
					graph->edgeIDs_[static_cast<size_t>(off)] = eid_global + 1;
				offs[dst]++;

				if (store_query_metadata)
					eid_global += 2;
				++edges_read;

				if (!graph->isQuery && edges_read >= next_edge_progress)
				{
					cout << "Data graph load progress: vertices " << vertices_read << " / "
							 << raw_num_v << ", edges " << edges_read << " / " << raw_num_e
							 << endl;
					next_edge_progress += kEdgeProgressInterval;
				}
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
		graph->num_v_labels = static_cast<numtype>(graph->vLabelFreq.size());
		if (!graph->isQuery)
		{
			graph->num_v_labels++;
		}

		if (graph->isQuery)
		{
			for (uint64_t i = 0; i < graph->num_e; ++i)
			{
				graph->edgeTag.set(static_cast<size_t>(i));
			}
		}
		else
		{
			cout << "Data graph loaded: vertices=" << vertices_read
					 << ", edges=" << edges_read << endl;
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

	auto &mem_mgr = MemoryManager::getInstance();

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

		// Use char* for byte-level pointer arithmetic+

		void *dev_space = nullptr;
		cudaStream_t stream;
		cuchk(cudaStreamCreate(&stream));
		mem_mgr.set_stream(&stream);
		mem_mgr.allocate_permanent(dev_space, mem_to_allocate);
		gpuGraph->allocation_base_ = dev_space;

		// Assign pointers using the calculated aligned offsets
		gpuGraph->degree_ = reinterpret_cast<degtype *>(dev_space + degree_offset);
		gpuGraph->vLabels_ = reinterpret_cast<vltype *>(dev_space + vlabel_offset);
		gpuGraph->offsets_ = reinterpret_cast<offtype *>(dev_space + offset_list_offset);
		gpuGraph->neighbors_ = reinterpret_cast<vtype *>(dev_space + neighbor_offset);

		cuchk(cudaStreamSynchronize(stream));
		cuchk(cudaStreamDestroy(stream));
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

	MAX_L_FREQ = 0;
	for (const auto &kv : query->vLabelFreq)
	{
		const vltype query_label_id = kv.first;
		auto it_orig = cpuGraph::vlmapn2o.find(query_label_id);
		if (it_orig == cpuGraph::vlmapn2o.end())
			continue;
		const vltype orig_label = it_orig->second;
		auto it_freq = data->vLabelFreq.find(orig_label);
		if (it_freq != data->vLabelFreq.end())
			MAX_L_FREQ = std::max(MAX_L_FREQ, it_freq->second);
	}
	MAX_DATA_DEGREE = data->maxDegree;

	NUM_VD = data->num_v;
	if (data->num_e > std::numeric_limits<uint32_t>::max())
	{
		NUM_ED = std::numeric_limits<uint32_t>::max();
		cout << "Warning: data graph edge count exceeds 32-bit NUM_ED metadata; "
				 << "clamping NUM_ED for legacy constants while keeping 64-bit host offsets."
				 << endl;
	}
	else
	{
		NUM_ED = static_cast<uint32_t>(data->num_e);
	}
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
