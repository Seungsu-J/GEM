#include "cuda_helpers.cuh"
#include "filter.h"
#include "cpuGraph.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace
{
  struct LabelRequirement
  {
    vltype label = 0;
    uint32_t count = 0;
  };

  struct QueryProfile
  {
    vltype vertex_label = std::numeric_limits<vltype>::max();
    degtype min_data_degree = 0;
    uint32_t num_requirements = 0;
    std::array<LabelRequirement, MAX_VLQ> requirements{};
  };

  struct LabelGroupProfile
  {
    uint32_t num_queries = 0;
    uint32_t num_relevant_labels = 0;
    std::array<vtype, MAX_VQ> queries{};
    std::array<uint8_t, MAX_VLQ> relevant_neighbor_labels{};
  };

  struct ChunkCandidates
  {
    std::array<std::vector<vtype>, MAX_VQ> per_query;
  };

  uint32_t getFilterWorkerCount()
  {
#ifdef _OPENMP
    const uint32_t runtime_threads =
        static_cast<uint32_t>(std::max(1, omp_get_max_threads()));
    if (std::getenv("OMP_NUM_THREADS") != nullptr)
      return runtime_threads;
    return std::min<uint32_t>(16u, runtime_threads);
#else
    return 1u;
#endif
  }
} // namespace

void filterSSM(cpuGraph *hq, cpuGraph *hg, gpuGraph *dq, gpuGraph *dg,
               vtype *&h_u_candidate_vs_, numtype *&h_num_u_candidate_vs_,
               vtype *&d_u_candidate_vs_, numtype *&d_num_u_candidate_vs_,
               uint32_t *&d_bitmap, uint32_t &bitmap_pitch)
{
  (void)dq;
  (void)dg;

  ASSERT_WITH_LOG(NUM_VQ <= MAX_VQ, "NUM_VQ exceeds MAX_VQ in CPU filter.");
  ASSERT_WITH_LOG(NUM_VLQ <= MAX_VLQ, "NUM_VLQ exceeds MAX_VLQ in CPU filter.");

  // Build label remap: original data label -> query-local label id.
  // Data graph keeps original labels; query labels are remapped sequentially.
  vltype max_orig_label = 0;
  for (const auto &kv : cpuGraph::vlmapo2n)
    if (kv.first > max_orig_label)
      max_orig_label = kv.first;
  // Also scan data graph labels for the upper bound
  for (vtype v = 0; v < NUM_VD; ++v)
    if (hg->vLabels_[v] > max_orig_label)
      max_orig_label = hg->vLabels_[v];
  const vltype INVALID_LABEL = static_cast<vltype>(NUM_VLQ);
  std::vector<vltype> label_remap(max_orig_label + 1, INVALID_LABEL);
  for (const auto &kv : cpuGraph::vlmapo2n)
    label_remap[kv.first] = kv.second;

  std::fill(h_num_u_candidate_vs_, h_num_u_candidate_vs_ + NUM_VQ, 0);

  std::array<QueryProfile, MAX_VQ> query_profiles{};
  std::array<LabelGroupProfile, MAX_VLQ> label_groups{};
  std::array<uint32_t, MAX_VLQ> query_label_counts{};

  for (vtype u = 0; u < NUM_VQ; ++u)
  {
    QueryProfile &profile = query_profiles[u];
    profile.vertex_label = hq->vLabels_[u];
    profile.min_data_degree =
        (hq->out_degree_[u] > THRESHOLD) ? (hq->out_degree_[u] - THRESHOLD) : 0;

    query_label_counts.fill(0);
    for (offtype off = hq->offsets_[u]; off < hq->offsets_[u + 1]; ++off)
    {
      const vtype u_nbr = hq->neighbors_[off];
      const vltype nbr_label = hq->vLabels_[u_nbr];
      if (nbr_label < NUM_VLQ)
      {
        ++query_label_counts[nbr_label];
      }
    }

    for (vltype label = 0; label < NUM_VLQ; ++label)
    {
      const uint32_t count = query_label_counts[label];
      if (count == 0)
        continue;

      profile.requirements[profile.num_requirements++] = {label, count};
    }

    std::sort(
        profile.requirements.begin(),
        profile.requirements.begin() + profile.num_requirements,
        [](const LabelRequirement &lhs, const LabelRequirement &rhs)
        {
          if (lhs.count != rhs.count)
            return lhs.count > rhs.count;
          return lhs.label < rhs.label;
        });

    if (profile.vertex_label >= NUM_VLQ)
      continue;

    LabelGroupProfile &group = label_groups[profile.vertex_label];
    ASSERT_WITH_LOG(group.num_queries < MAX_VQ,
                    "Query-by-label bucket overflow in CPU filter.");
    group.queries[group.num_queries++] = u;

    for (uint32_t i = 0; i < profile.num_requirements; ++i)
    {
      const vltype required_label = profile.requirements[i].label;
      if (group.relevant_neighbor_labels[required_label])
        continue;

      group.relevant_neighbor_labels[required_label] = 1;
      ++group.num_relevant_labels;
    }
  }

  const uint32_t worker_count = getFilterWorkerCount();
  const uint32_t desired_chunks = (worker_count > 1) ? (worker_count * 4u) : 1u;
  const uint32_t chunk_count =
      std::max(1u, std::min((NUM_VD == 0) ? 1u : NUM_VD, desired_chunks));
  const vtype chunk_size = (NUM_VD + chunk_count - 1) / chunk_count;
  const size_t reserve_per_chunk =
      std::max<size_t>(4, (static_cast<size_t>(MAX_L_FREQ) + chunk_count - 1) /
                              chunk_count);

  std::vector<ChunkCandidates> chunk_candidates(chunk_count);

#pragma omp parallel for schedule(dynamic, 1) if (chunk_count > 1) \
    num_threads(worker_count)
  for (int chunk_idx = 0; chunk_idx < static_cast<int>(chunk_count); ++chunk_idx)
  {
    ChunkCandidates &local_candidates = chunk_candidates[chunk_idx];
    for (vtype u = 0; u < NUM_VQ; ++u)
    {
      if (query_profiles[u].vertex_label < NUM_VLQ)
      {
        local_candidates.per_query[u].reserve(reserve_per_chunk);
      }
    }

    std::array<uint32_t, MAX_VLQ> data_label_counts{};
    std::array<vltype, MAX_VLQ> touched_labels{};

    const vtype v_begin = static_cast<vtype>(chunk_idx) * chunk_size;
    const vtype v_end = std::min(NUM_VD, v_begin + chunk_size);
    for (vtype v = v_begin; v < v_end; ++v)
    {
      const vltype orig_label = hg->vLabels_[v];
      const vltype data_label = (orig_label < label_remap.size()) ? label_remap[orig_label] : INVALID_LABEL;
      if (data_label >= NUM_VLQ)
        continue;

      const LabelGroupProfile &group = label_groups[data_label];
      if (group.num_queries == 0)
        continue;

      uint32_t num_touched_labels = 0;
      if (group.num_relevant_labels != 0)
      {
        for (offtype off = hg->offsets_[v]; off < hg->offsets_[v + 1]; ++off)
        {
          const vtype v_nbr = hg->neighbors_[off];
          const vltype orig_nbr_label = hg->vLabels_[v_nbr];
          const vltype nbr_label = (orig_nbr_label < label_remap.size()) ? label_remap[orig_nbr_label] : INVALID_LABEL;
          if (nbr_label >= NUM_VLQ || !group.relevant_neighbor_labels[nbr_label])
            continue;

          if (data_label_counts[nbr_label] == 0)
          {
            touched_labels[num_touched_labels++] = nbr_label;
          }
          ++data_label_counts[nbr_label];
        }
      }

      const degtype data_degree = hg->out_degree_[v];
      for (uint32_t q_idx = 0; q_idx < group.num_queries; ++q_idx)
      {
        const vtype u = group.queries[q_idx];
        const QueryProfile &profile = query_profiles[u];
        if (data_degree < profile.min_data_degree)
          continue;

        uint32_t total_diff = 0;
        for (uint32_t req_idx = 0; req_idx < profile.num_requirements; ++req_idx)
        {
          const LabelRequirement &req = profile.requirements[req_idx];
          const uint32_t data_count = data_label_counts[req.label];
          if (req.count > data_count)
          {
            total_diff += (req.count - data_count);
            if (total_diff > THRESHOLD)
              break;
          }
        }

        if (total_diff <= THRESHOLD)
        {
          local_candidates.per_query[u].push_back(v);
        }
      }

      for (uint32_t i = 0; i < num_touched_labels; ++i)
      {
        data_label_counts[touched_labels[i]] = 0;
      }
    }
  }

  for (vtype u = 0; u < NUM_VQ; ++u)
  {
    numtype total_count = 0;
    for (const ChunkCandidates &chunk : chunk_candidates)
    {
      total_count += static_cast<numtype>(chunk.per_query[u].size());
    }

    ASSERT_WITH_LOG(total_count <= MAX_L_FREQ,
                    "Candidate list overflow in CPU filter.");

    vtype *dst = h_u_candidate_vs_ + u * MAX_L_FREQ;
    numtype offset = 0;
    for (const ChunkCandidates &chunk : chunk_candidates)
    {
      const std::vector<vtype> &src = chunk.per_query[u];
      if (!src.empty())
      {
        std::copy(src.begin(), src.end(), dst + offset);
        offset += static_cast<numtype>(src.size());
      }
    }
    h_num_u_candidate_vs_[u] = total_count;
  }

  if (d_u_candidate_vs_)
  {
    cuchk(cudaMemcpy(d_u_candidate_vs_, h_u_candidate_vs_,
                     sizeof(vtype) * NUM_VQ * MAX_L_FREQ,
                     cudaMemcpyHostToDevice));
  }
  if (d_num_u_candidate_vs_)
  {
    cuchk(cudaMemcpy(d_num_u_candidate_vs_, h_num_u_candidate_vs_,
                     sizeof(numtype) * NUM_VQ, cudaMemcpyHostToDevice));
  }

  d_bitmap = nullptr;
  bitmap_pitch = 0;

#ifndef NDEBUG
  numtype total_candidates = 0;
  for (uint32_t i = 0; i < NUM_VQ; ++i)
  {
    total_candidates += h_num_u_candidate_vs_[i];
  }

  std::cout << "Filtering completed successfully:" << std::endl;
  std::cout << "  - Query vertices processed: " << NUM_VQ << std::endl;
  std::cout << "  - Total candidates found: " << total_candidates << std::endl;
  std::cout << "  - Average candidates per vertex: "
            << (total_candidates / static_cast<float>(NUM_VQ)) << std::endl;
#endif
}
