#include "join_helpers.h"
#include "defs.h"
#include "globals.h"

#include <cinttypes>

/**
 * Global Variables:
 * 1. duplication informations
 * 2. data_ptr and parent_ptr
 * 3. flag_ if skip final materialization
 */

// // dup info
// uint32_t *d_dup_cols; // allocate once for reuse
// uint32_t num_dup_cols;
// uint32_t *h_dup_cols;

// // data_ptr, parent_ptr
// vtype **d_data_arrays;
// uint64_t **d_parent_arrays;

// // skip flag
// bool skip_final_materialization = false;

// cudaStream_t join_stream;

uint32_t ceilPowerOf2(uint32_t x)
{
  if (x <= 1)
    return 1;

  // 1. 为了处理 x 本身就是 2 的 n 次幂的情况
  // (例如 x=16)，我们先减 1。
  // x=16 -> 15 (0...01111) -> 16
  // x=17 -> 16 (0...10000) -> 32
  x--;

  // 2. 将最高位的 1 向右"涂抹"，填满所有低位
  // 假设 x = 100 (0110 0100)
  // x-- -> 99 (0110 0011)
  x |= x >> 1;  // 0111 0011
  x |= x >> 2;  // 0111 1111
  x |= x >> 4;  // 0111 1111
  x |= x >> 8;  // 0111 1111
  x |= x >> 16; // 0111 1111  (这是 127)
  // x |= x >> 32; // 对于 64 位整数，这一步是必要的

  // 3. 最后加 1，得到 128 (1000 0000)
  x++;

  return x;
}

uint32_t findClosestPowerOf2_Bitwise(uint32_t x)
{
  if (x <= 1)
    return 1; // 2^0 = 1

  // 1. 找到 "天花板"
  uint32_t high_pow = ceilPowerOf2(x); // 例如 x=100, high_pow=128

  // 2. 找到 "地板"
  // (注意：如果 high_pow 是 1, low_pow 会是 0，但这被 x<=1 捕获了)
  uint32_t low_pow = high_pow >> 1; // 128 >> 1 = 64
  // 3. 比较距离 (与方法一相同)
  if ((x - low_pow) < (high_pow - x))
    return low_pow;
  else
    return high_pow;
}
