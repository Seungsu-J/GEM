#ifndef STRUCTURE_CUCKOO_HASH_H
#define STRUCTURE_CUCKOO_HASH_H

#include "defs.h"
#include "globals.h"

__global__ void buildHashKeys(const uint32_t *in, const uint32_t in_size,
                              uint32_t *keys0, uint32_t *keys1,
                              const uint32_t num_bucket, const uint32_t C0,
                              const uint32_t C1, const uint32_t C2,
                              const uint32_t C3,
                              uint32_t *progress,
                              uint32_t *success);

__global__ void buildHashValuesCount(const offtype *d_offsets_,
                                     const vtype *d_neighbors_,
                                     const uint32_t *flags_second_level,
                                     const uint32_t num_flags,
                                     uint32_t *progress, uint32_t *hash_keys,
                                     uint32_t *hash_values,
                                     const uint32_t num_bucket);

__global__ void
buildHashValuesWrite(const offtype *d_offsets_, const vtype *d_neighbors_,
                     const uint32_t *flags_second_level,
                     const uint32_t num_flags, uint32_t *progress,
                     uint32_t *hash_keys, uint32_t *hash_values,
                     const uint32_t num_bucket, uint32_t *neighbors);

__global__ void create_bitmap_from_candidates(const vtype *candidates,
                                              const uint32_t num_candidates,
                                              const uint32_t max_vertex_id,
                                              uint32_t *bitmap,
                                              const uint32_t num_flags);

#endif