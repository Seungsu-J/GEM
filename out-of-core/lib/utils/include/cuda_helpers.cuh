#ifndef CUDA_HELPERS_H
#define CUDA_HELPERS_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cstddef>
#include <iostream>

struct timespec timespec_diff(struct timespec start, struct timespec end);
size_t aggressive_warmup(int var_device_id, cudaStream_t stream, size_t safety_buffer_mb);
size_t getSafePinnedMemorySize();

#define FULL_MASK 0xffffffff

#define DIV_CEIL(a, b) ((a) / (b) + ((a) % (b) != 0))

#define TO_GB(x) (x / 1024.0 / 1024.0 / 1024.0)

#define cuchk(ans)                        \
  {                                       \
    gpuAssert((ans), __FILE__, __LINE__); \
  }

    inline void gpuAssert(cudaError_t code, const char *file, int line,
                          bool abort = true)
{
  if (code != cudaSuccess)
  {
    std::cout << "GPU assert: " << cudaGetErrorName(code) << " "
              << cudaGetErrorString(code) << " " << file << " " << line << " "
              << std::endl;
    if (abort)
      exit(code);
  }
}

#define cuchk_kernel(call)                            \
  {                                                   \
    cudaError_t cucheck_err = (call);                 \
    if (cucheck_err != cudaSuccess)                   \
    {                                                 \
      std::cout << __FILE__ << " " << __LINE__ << " " \
                << cudaGetErrorString(cucheck_err);   \
      assert(0);                                      \
    }                                                 \
  }

// Recording Time

#define TIME_INIT()                                           \
  cudaEvent_t gpu_start, gpu_end;                             \
  float kernel_time, total_kernel = 0.0, total_host = 0.0;    \
  auto cpu_start = std::chrono::high_resolution_clock::now(); \
  auto cpu_end = std::chrono::high_resolution_clock::now();   \
  std::chrono::duration<double> diff = cpu_end - cpu_start;

#define TIME_START()                                     \
  cpu_start = std::chrono::high_resolution_clock::now(); \
  cudaEventCreate(&gpu_start);                           \
  cudaEventCreate(&gpu_end);                             \
  cudaEventRecord(gpu_start)

#define TIME_END()                                        \
  cpu_end = std::chrono::high_resolution_clock::now();    \
  cudaEventRecord(gpu_end);                               \
  cudaEventSynchronize(gpu_start);                        \
  cudaEventSynchronize(gpu_end);                          \
  cudaEventElapsedTime(&kernel_time, gpu_start, gpu_end); \
  total_kernel += kernel_time;                            \
  diff = cpu_end - cpu_start;                             \
  total_host += diff.count();

#define PRINT_LOCAL_TIME(name)                                               \
  std::cout << name << ", time (ms): "                                       \
            << static_cast<unsigned long>(diff.count() * 1000) << "(host), " \
            << static_cast<unsigned long>(kernel_time) << "(kernel)\n"

#define PRINT_TOTAL_TIME(name)                                                 \
  std::cout << name                                                            \
            << " time (ms): " << static_cast<unsigned long>(total_host * 1000) \
            << "(host) " << static_cast<unsigned long>(total_kernel)           \
            << "(kernel)\n";

#define micro_init()       \
  struct timespec time_st; \
  struct timespec time_ed; \
  struct timespec diff_micro;

#define micro_start() clock_gettime(CLOCK_MONOTONIC, &time_st);

#define micro_end() clock_gettime(CLOCK_MONOTONIC, &time_ed);

#define micro_print_local(name)                                      \
  diff_micro = timespec_diff(time_st, time_ed);                      \
  std::cout << name << " time (us): " << diff_micro.tv_nsec / 1000.0 \
            << std::endl;

#define MEM_INIT() size_t mf, ma;

#define PRINT_MEM_INFO(name)                                                \
  cudaMemGetInfo(&mf, &ma);                                                 \
  std::cout << name << ", Free " << TO_GB(mf) << "GB, Total: " << TO_GB(ma) \
            << "GB\n";

struct nonZeroOp
{
  __host__ __device__ bool operator()(const uint32_t &x) { return x != 0; }
};

struct nonUintMax
{
  __host__ __device__ bool operator()(const uint32_t &x)
  {
    return x != UINT32_MAX;
  }
};

struct div32CeilOp
{
  __host__ __device__ uint32_t operator()(const uint32_t &x) const
  {
    return (x + 31) >> 5;
  }
};

struct CountPositive
{
  const uint32_t *counts;
  __device__ bool operator()(int i) const
  {
    return counts[i] > 0;
  }
};

struct PopCountOp
{
  __host__ __device__ uint32_t operator()(const uint32_t &x) const
  {
#ifdef __CUDA_ARCH__
    return __popc(x);
#else
    return __builtin_popcount(x);
#endif
  }
};

template <typename T>
__device__ __forceinline__ T fast_load(const T *ptr)
{
  return __ldg(ptr);
}

template <typename T>
__forceinline__ __device__ uint32_t lower_bound(T *array, uint32_t size,
                                                const T &v)
{
  if (array == nullptr || size == 0)
    return UINT32_MAX;

  if (array[size - 1] < v)
    return UINT32_MAX;

  uint32_t low = 0u, high = size - 1, mid = (low + high) / 2;
  while (low < high)
  {
    if (array[mid] < v)
    {
      low = mid + 1;
    }
    else
    {
      high = mid;
    }
    mid = (low + high) / 2;
  }
  return mid;
}

// Overloaded version that returns UINT32_MAX if exact match is not found
// Optimized: NO extra check - integrates exact match logic into the search loop
template <typename T>
__forceinline__ __device__ uint32_t lower_bound_exact(T *array, uint32_t size,
                                                      const T &v)
{
  if (array == nullptr || size == 0)
    return UINT32_MAX;

  if (array[size - 1] < v)
    return UINT32_MAX;

  // Early check: if the last element equals v, we can return immediately
  if (array[size - 1] == v)
  {
    // Find the first occurrence of v
    uint32_t low = 0u, high = size - 1;
    while (low < high)
    {
      uint32_t mid = (low + high) / 2;
      if (array[mid] < v)
      {
        low = mid + 1;
      }
      else
      {
        high = mid;
      }
    }
    return low;
  }

  // Standard binary search with integrated exact match check
  uint32_t low = 0u, high = size - 1;
  while (low <= high)
  {
    uint32_t mid = low + (high - low) / 2;

    if (array[mid] < v)
    {
      low = mid + 1;
    }
    else if (array[mid] > v)
    {
      if (mid == 0)
        break; // Prevent underflow
      high = mid - 1;
    }
    else
    {
      // Found exact match! Now find the FIRST occurrence
      while (mid > 0 && array[mid - 1] == v)
      {
        mid--;
      }
      return mid;
    }
  }

  return UINT32_MAX; // Not found
}

template <typename T>
__forceinline__ __device__ bool binary_search(T *array, uint32_t size,
                                              const T &v)
{
  if (array == nullptr || size == 0)
    return false;

  if (fast_load(array + size - 1) < v)
    return false;

  uint32_t low = 0u, high = size - 1, mid;
  while (low <= high)
  {
    mid = low + (high - low) / 2;
    auto mid_v = fast_load(array + mid);

    if (mid_v < v)
    {
      low = mid + 1;
    }
    else if (mid_v > v)
    {
      // Check for underflow before subtracting
      if (mid == 0)
        break;
      high = mid - 1;
    }
    else
    {
      return true; // Found
    }
  }
  return false; // Not found
}

// Cooperative binary search - all threads participate
template <typename T>
__forceinline__ __device__ int cooperative_search(T *array, uint32_t size,
                                                  const T &target, int lid)
{
  int low = 0, high = size - 1;

  while (low <= high)
  {
    int mid = low + (high - low) / 2;

    // All threads load the same value (coalesced access)
    T mid_val = array[mid];

    // All threads participate in comparison
    if (mid_val < target)
    {
      low = mid + 1;
    }
    else if (mid_val > target)
    {
      high = mid - 1;
    }
    else
    {
      return mid; // Found
    }
  }
  return -1; // Not found
}

// Performance optimization macros
#ifndef VECTORIZED_LOAD_SIZE
#define VECTORIZED_LOAD_SIZE 4
#endif

// // Memory coalescing helpers
// template <typename T>
// __device__ __forceinline__ T fast_load(const T *ptr)
// {
// #ifdef __CUDA_ARCH__
//   return __ldg(ptr); // Use read-only cache for better performance
// #else
//   return *ptr; // Host fallback
// #endif
// }

// Fast load for uint2 - loads two consecutive uint32_t elements at once
__device__ __forceinline__ uint2 fast_load_uint2(const uint32_t *ptr)
{
#ifdef __CUDA_ARCH__
  if ((uintptr_t)(ptr) % 8 == 0)
  {
    // 8-byte aligned - use vectorized load
    return __ldg(reinterpret_cast<const uint2 *>(ptr));
  }
  else
  {
    // Unaligned - fall back to two separate loads
    uint2 result;
    result.x = __ldg(ptr);
    result.y = __ldg(ptr + 1);
    return result;
  }
#else
  // Host fallback
  uint2 result;
  result.x = ptr[0];
  result.y = ptr[1];
  return result;
#endif
}

// Fast load for uint4 - loads four consecutive uint32_t elements at once (128-bit load)
__device__ __forceinline__ uint4 fast_load_uint4(const uint32_t *ptr)
{
#ifdef __CUDA_ARCH__
  if ((uintptr_t)(ptr) % 16 == 0)
  {
    // 16-byte aligned - use vectorized 128-bit load
    return __ldg(reinterpret_cast<const uint4 *>(ptr));
  }
  else
  {
    // Unaligned - fall back to four separate loads
    uint4 result;
    result.x = __ldg(ptr);
    result.y = __ldg(ptr + 1);
    result.z = __ldg(ptr + 2);
    result.w = __ldg(ptr + 3);
    return result;
  }
#else
  // Host fallback
  uint4 result;
  result.x = ptr[0];
  result.y = ptr[1];
  result.z = ptr[2];
  result.w = ptr[3];
  return result;
#endif
}

// Convenience function for loading pairs of values
template <typename PtrT, typename OutT>
__device__ __forceinline__ void
fast_load_pair(PtrT *ptr, OutT &first, OutT &second)
{
  first = static_cast<OutT>(fast_load(ptr));
  second = static_cast<OutT>(fast_load(ptr + 1));
}

__device__ __forceinline__ void
fast_load_pair(uint32_t *ptr, uint32_t &first, uint32_t &second)
{
  uint2 data = fast_load_uint2(ptr);
  first = data.x;
  second = data.y;
}

__device__ __forceinline__ void safe_pair_read(uint32_t *ptr, uint32_t &low,
                                               uint32_t &high)
{
#ifdef __CUDA_ARCH__
  if ((uintptr_t)(ptr) % 8 == 0)
  {
    uint64_t pair = __ldg((uint64_t *)(ptr));
    low = static_cast<uint32_t>(pair & 0xFFFFFFFF);
    high = static_cast<uint32_t>(pair >> 32);
  }
  else
  {
    low = __ldg(ptr);
    high = __ldg(ptr + 1);
  }
#else
  // Host fallback - simple individual reads
  low = ptr[0];
  high = ptr[1];
#endif
}

__device__ __forceinline__ void safe_quad_read(uint32_t *ptr, uint32_t &a,
                                               uint32_t &b, uint32_t &c,
                                               uint32_t &d, int num = 4)
{
#ifdef __CUDA_ARCH__
  if ((uintptr_t)(ptr) % 16 == 0)
  {
    // Single 128-bit read - much more efficient than four 32-bit reads
    uint4 quad = __ldg((uint4 *)(ptr));
    if (num > 0)
      a = static_cast<uint32_t>(quad.x);
    if (num > 1)
      b = static_cast<uint32_t>(quad.y);
    if (num > 2)
      c = static_cast<uint32_t>(quad.z);
    if (num > 3)
      d = static_cast<uint32_t>(quad.w);
  }
  else
  {
    // Fall back to individual reads for unaligned access
    if (num > 0)
      a = __ldg(ptr);
    if (num > 1)
      b = __ldg(ptr + 1);
    if (num > 2)
      c = __ldg(ptr + 2);
    if (num > 3)
      d = __ldg(ptr + 3);
  }
#else
  // Host fallback - simple individual reads
  if (num > 0)
    a = ptr[0];
  if (num > 1)
    b = ptr[1];
  if (num > 2)
    c = ptr[2];
  if (num > 3)
    d = ptr[3];
#endif
}

__device__ __forceinline__ int warp_reduce_add_sync(int val)
{
#ifdef __CUDA_ARCH__
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2)
  {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return __shfl_sync(0xffffffff, val, 0);
#else
  return val; // Host fallback
#endif
}

// Vectorized memcpy for GPU device functions
template <typename T>
__device__ __forceinline__ void vectorized_memcpy(T *__restrict__ dst,
                                                  const T *__restrict__ src,
                                                  uint32_t count)
{
  if (!count)
    return;
#ifdef __CUDA_ARCH__
  static_assert(sizeof(T) == 4,
                "vectorized_memcpy currently supports only 4-byte types");

  if (count == 0)
    return;

  const uintptr_t src_addr = reinterpret_cast<uintptr_t>(src);
  const uintptr_t dst_addr = reinterpret_cast<uintptr_t>(dst);

  // Check for 16-byte alignment for both src and dst
  if ((src_addr % 16 == 0) && (dst_addr % 16 == 0) && (count >= 4))
  {
    // Process 4-element vectors (128-bit)
    uint32_t vec4_count = count & ~3u; // Round down to multiple of 4
    const uint4 *src_vec4 = reinterpret_cast<const uint4 *>(src);
    uint4 *dst_vec4 = reinterpret_cast<uint4 *>(dst);

    for (uint32_t i = 0; i < vec4_count; i += 4)
    {
      uint4 data = __ldg(src_vec4 + (i >> 2));
      dst_vec4[i >> 2] = data;
    }

    // Handle remaining elements
    for (uint32_t i = vec4_count; i < count; ++i)
    {
      dst[i] = __ldg(src + i);
    }
  }
  // Check for 8-byte alignment for both src and dst
  else if ((src_addr % 8 == 0) && (dst_addr % 8 == 0) && (count >= 2))
  {
    // Process 2-element vectors (64-bit)
    uint32_t vec2_count = count & ~1u; // Round down to multiple of 2
    const uint64_t *src_vec2 = reinterpret_cast<const uint64_t *>(src);
    uint64_t *dst_vec2 = reinterpret_cast<uint64_t *>(dst);

    for (uint32_t i = 0; i < vec2_count; i += 2)
    {
      uint64_t data = __ldg(src_vec2 + (i >> 1));
      dst_vec2[i >> 1] = data;
    }

    // Handle remaining elements
    for (uint32_t i = vec2_count; i < count; ++i)
    {
      dst[i] = __ldg(src + i);
    }
  }
  else
  {
    // Fall back to scalar copy for unaligned or small transfers
    for (uint32_t i = 0; i < count; ++i)
    {
      dst[i] = __ldg(src + i);
    }
  }
#endif
}

// Specialized version for common case of uint32_t
__device__ __forceinline__ void
vectorized_memcpy_u32(uint32_t *__restrict__ dst,
                      const uint32_t *__restrict__ src, uint32_t count)
{
  if (!count)
    return;
  vectorized_memcpy<uint32_t>(dst, src, count);
}

// High-performance bulk copy for large transfers
template <typename T>
__device__ __forceinline__ void bulk_vectorized_copy(T *__restrict__ dst,
                                                     const T *__restrict__ src,
                                                     uint32_t count)
{
#ifdef __CUDA_ARCH__
  static_assert(sizeof(T) == 4,
                "bulk_vectorized_copy currently supports only 4-byte types");

  if (count == 0)
    return;

  const uintptr_t src_addr = reinterpret_cast<uintptr_t>(src);
  const uintptr_t dst_addr = reinterpret_cast<uintptr_t>(dst);

  // For large transfers, prioritize maximum vectorization
  if ((src_addr % 16 == 0) && (dst_addr % 16 == 0) && (count >= 8))
  {
    // Use 128-bit transfers with loop unrolling
    uint32_t vec4_count = count & ~3u;
    const uint4 *src_vec4 = reinterpret_cast<const uint4 *>(src);
    uint4 *dst_vec4 = reinterpret_cast<uint4 *>(dst);

    uint32_t unroll_count =
        vec4_count & ~7u; // Process in groups of 8 elements (2 vec4s)

    // Unrolled loop for better instruction-level parallelism
    for (uint32_t i = 0; i < unroll_count; i += 8)
    {
      uint32_t vec_idx = i >> 2;
      uint4 data0 = __ldg(src_vec4 + vec_idx);
      uint4 data1 = __ldg(src_vec4 + vec_idx + 1);
      dst_vec4[vec_idx] = data0;
      dst_vec4[vec_idx + 1] = data1;
    }

    // Handle remaining vec4s
    for (uint32_t i = unroll_count; i < vec4_count; i += 4)
    {
      uint4 data = __ldg(src_vec4 + (i >> 2));
      dst_vec4[i >> 2] = data;
    }

    // Handle remaining scalar elements
    for (uint32_t i = vec4_count; i < count; ++i)
    {
      dst[i] = __ldg(src + i);
    }
  }
  else
  {
    // Fall back to standard vectorized copy
    vectorized_memcpy<T>(dst, src, count);
  }
#endif
}

template <typename T, int VecWidth>
struct WarpVecTraits;

template <>
struct WarpVecTraits<uint32_t, 2>
{
  using Type = uint2;
};
template <>
struct WarpVecTraits<uint32_t, 4>
{
  using Type = uint4;
};

template <typename T, int VecWidth>
__device__ inline void warp_copy_segment(const T *__restrict__ src_ptr,
                                         T *__restrict__ dst_ptr,
                                         int count,
                                         const uint32_t lid,
                                         unsigned mask = 0xffffffffu)
{
  static_assert(VecWidth == 2 || VecWidth == 4,
                "warp_copy_segment only supports VecWidth 2 or 4");
  static_assert(sizeof(T) == sizeof(uint32_t),
                "warp_copy_segment expects 4-byte element types");
  using VecT = typename WarpVecTraits<T, VecWidth>::Type;
  static_assert(sizeof(VecT) == VecWidth * sizeof(T),
                "VecWidth does not match intrinsic vector size");

  constexpr int kWarpSize = 32;
  unsigned warp_mask;
#ifdef __CUDA_ARCH__
  warp_mask = mask ? mask : __activemask();
#else
  warp_mask = 0xffffffffu;
#endif

  uintptr_t src_addr = reinterpret_cast<uintptr_t>(src_ptr);
  uintptr_t dst_addr = reinterpret_cast<uintptr_t>(dst_ptr);
  constexpr int kAlignBytes = sizeof(VecT);
  constexpr int kAlignMask = kAlignBytes - 1;
  constexpr int kTileSpan = kWarpSize * VecWidth;

  bool aligned_pair = ((src_addr ^ dst_addr) & kAlignMask) == 0;

  int scalar_prefix = 0;
  if (aligned_pair && count >= VecWidth)
  {
    int misalignment = static_cast<int>(src_addr & kAlignMask);
    if (misalignment != 0)
    {
      int align_elems = (kAlignBytes - misalignment) / sizeof(T);
      scalar_prefix = (align_elems < count) ? align_elems : count;
    }
  }

  for (int idx = lid; idx < scalar_prefix; idx += kWarpSize)
  {
    dst_ptr[idx] = src_ptr[idx];
  }
#ifdef __CUDA_ARCH__
  __syncwarp(warp_mask);
#endif

  int total_vec = 0;
  if (aligned_pair)
  {
    total_vec = (count - scalar_prefix) / VecWidth;
    if (total_vec > 0)
    {
      const VecT *src_vec =
          reinterpret_cast<const VecT *>(src_ptr + scalar_prefix);
      VecT *dst_vec =
          reinterpret_cast<VecT *>(dst_ptr + scalar_prefix);
#pragma unroll
      for (int vec_idx = lid; vec_idx < total_vec; vec_idx += kWarpSize)
      {
        dst_vec[vec_idx] = src_vec[vec_idx];
      }
    }
  }
#ifdef __CUDA_ARCH__
  __syncwarp(warp_mask);
#endif

  const int consumed = scalar_prefix + total_vec * VecWidth;
  for (int idx = consumed + lid; idx < count; idx += kWarpSize)
  {
    dst_ptr[idx] = src_ptr[idx];
  }
#ifdef __CUDA_ARCH__
  __syncwarp(warp_mask);
#endif
}

#include <thrust/system/cuda/execution_policy.h>
#include <thrust/device_vector.h>
#include <cuda_runtime.h>

struct PoolAllocator
{
  cudaStream_t stream;

  PoolAllocator(cudaStream_t s) : stream(s) {}

  // 分配 memory pool 内存
  __host__ __device__ void *allocate(size_t n)
  {
    void *ptr = nullptr;
    cudaMallocAsync(&ptr, n, stream);
    return ptr;
  }

  __host__ __device__ void deallocate(void *ptr, size_t n)
  {
    cudaFreeAsync(ptr, stream);
  }
};

#endif // CUDA_HELPERS_H
