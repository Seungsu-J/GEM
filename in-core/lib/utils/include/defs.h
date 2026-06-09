#ifndef GLOBAL_DEFS_H
#define GLOBAL_DEFS_H

#include <cstdint>
#include <cinttypes>
#include <bitset>
#include <iostream>
#include <assert.h>

#define GRID_DIM 504u
// #define GRID_DIM 252u
// #define GRID_DIM 168u // for A6000 252u theoretically
// #define GRID_DIM 1024u
// #define BLOCK_DIM 512u
#define BLOCK_DIM 256u
#define WARP_SIZE 32u
#define WARP_PER_BLOCK (BLOCK_DIM / WARP_SIZE)
#define NWARPS_TOTAL (GRID_DIM * BLOCK_DIM / WARP_SIZE)

#define MAX_VQ 16u
#define MAX_EQ 16u
#define MAX_VLQ 32u
#define ET_SIZE 32u
#define MASK_HIGH31 (0xfffffffeu) // 111...1110

#define HASH_PRIME 4294967291u
#define BUCKET_DIM 8u
#define CUCKOO_SCALE 2u
#define NUM_TABLES 2u
#define MAX_CUCKOO_LOOP 64u

using vtype = uint32_t;
using etype = uint32_t;
using vltype = uint32_t;
using numtype = uint32_t;
using offtype = uint32_t;
using degtype = uint32_t;
// using eltype = uint32_t;
using ettype = std::bitset<ET_SIZE>;

inline unsigned int calc_grid_dim(int N, int block_size)
{
  if (N == 0)
    ++N;
  return (N - 1) / block_size + 1;
}

// #define TID (threadIdx.x)
// #define BID (blockIdx.x)
// #define IDX (TID + BID * blockDim.x)
// #define LID (TID & 31)
// #define WID (TID >> 5)
// #define WID_G (IDX >> 5)

// Custom assert macro with logging
#ifdef NDEBUG
#define ASSERT_WITH_LOG(condition, message) ((void)0)
#else
#define ASSERT_WITH_LOG(condition, message)                                     \
  do                                                                            \
  {                                                                             \
    if (!(condition))                                                           \
    {                                                                           \
      std::cerr << "ASSERTION FAILED: " << #condition << std::endl;             \
      std::cerr << "File: " << __FILE__ << ", Line: " << __LINE__ << std::endl; \
      std::cerr << "Function: " << __FUNCTION__ << std::endl;                   \
      std::cerr << "Message: " << message << std::endl;                         \
      std::cerr << "Additional debug info:" << std::endl;                       \
      assert(condition);                                                        \
    }                                                                           \
  } while (0)
#endif

// Alternative: Simple logging assert function
template <typename T1, typename T2>
void assert_with_values(bool condition, const std::string &expr, T1 actual, T2 expected,
                        const std::string &file, int line, const std::string &func)
{
  if (!condition)
  {
    std::cerr << "ASSERTION FAILED: " << expr << std::endl;
    std::cerr << "Expected: " << expected << ", Actual: " << actual << std::endl;
    std::cerr << "File: " << file << ", Line: " << line << ", Function: " << func << std::endl;
    assert(condition);
  }
}

#define ASSERT_EQ(actual, expected) \
  assert_with_values((actual) == (expected), #actual " == " #expected, actual, expected, __FILE__, __LINE__, __FUNCTION__)

#endif // !GLOBAL_DEFS_H