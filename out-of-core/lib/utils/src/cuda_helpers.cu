#include "cuda_helpers.cuh"

#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <sys/resource.h> // 用于获取 ulimit 限制
#include <sys/sysinfo.h>  // 用于获取系统内存信息

struct timespec timespec_diff(struct timespec start, struct timespec end)
{
  struct timespec temp;
  if ((end.tv_nsec - start.tv_nsec) < 0)
  {
    temp.tv_sec = end.tv_sec - start.tv_sec - 1;
    temp.tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
  }
  else
  {
    temp.tv_sec = end.tv_sec - start.tv_sec;
    temp.tv_nsec = end.tv_nsec - start.tv_nsec;
  }
  return temp;
}

__global__ void warp_up_kernel()
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < 32)
  {
    // 每个线程访问一些内存，确保内核不会被优化掉
    volatile int sink = 0;
    for (int i = 0; i < 1000; ++i)
    {
      sink += i * tid;
    }
  }
}

/**
 * @brief 激进预热函数：初始化 Context，锁定内存池，并按最大安全水位撑大池子
 * @param stream 用于异步操作的流
 * @param safety_buffer_mb 预留给系统和驱动的缓冲大小 (默认 500MB)
 */
size_t aggressive_warmup(int var_device_id, cudaStream_t stream, size_t safety_buffer_mb)
{
  // std::cout << "[Init] Initializing CUDA Context..." << std::endl;

  // 1. 强制初始化 Context (耗时约 200~500ms)
  // 这一步能确保后续 cudaMemGetInfo 拿到的剩余显存是扣除了 Context 开销后的真实值
  cuchk(cudaFree(0));

  int device_id = var_device_id;
  cuchk(cudaGetDevice(&device_id));

  // 2. 配置内存池策略：只进不出
  // 将 ReleaseThreshold 设为极大值，告诉驱动：只要我不显式 Trim，别把内存还给 OS
  cudaMemPool_t mem_pool;
  cuchk(cudaDeviceGetDefaultMemPool(&mem_pool, device_id));
  uint64_t threshold = UINT64_MAX;
  cuchk(cudaMemPoolSetAttribute(mem_pool, cudaMemPoolAttrReleaseThreshold, &threshold));

  // 3. 获取当前真实可用显存
  size_t free_byte, total_byte;
  cuchk(cudaMemGetInfo(&free_byte, &total_byte));

  // 计算安全水位：剩余显存 - 缓冲保护区
  size_t buffer_byte = safety_buffer_mb * 1024 * 1024;

  // 如果剩余空间不够缓冲，就不强行预热了，防止崩坏
  if (free_byte < buffer_byte)
  {
    std::cerr << "[Warn] Low memory! Skipping aggressive warmup." << std::endl;
    return 0;
  }

  size_t alloc_size = free_byte - buffer_byte;

  // std::cout << "[WarmUp] GPU Total: " << total_byte / (1024.0 * 1024.0 * 1024.0) << " GB\n"
  //           << "[WarmUp] Current Free: " << free_byte / (1024.0 * 1024.0 * 1024.0) << " GB\n"
  //           << "[WarmUp] Allocating: " << alloc_size / (1024.0 * 1024.0 * 1024.0) << " GB to fill the pool..."
  //           << std::endl;

  // 4. 异步申请并立即释放
  // 这会向 OS 申请物理页，Free 后物理页留在 Pool 中供你后续无损复用
  void *d_ptr = nullptr;
  cuchk(cudaMallocAsync(&d_ptr, alloc_size, stream));
  cuchk(cudaFreeAsync(d_ptr, stream));

  warp_up_kernel<<<1, 32, 0, stream>>>();

  // 5. 必须同步！
  // 确保上述操作全部在 GPU 端落地，Pool 状态完全就绪
  cuchk(cudaStreamSynchronize(stream));
  cuchk(cudaDeviceSynchronize());

  return alloc_size;

  // std::cout << "[Init] Ready. Memory pool is fully populated." << std::endl;
}

size_t getSafePinnedMemorySize()
{
  // --- 配置：保留给 OS 的安全水位 (例如 2GB) ---
  const size_t OS_HEADROOM = 2ULL * 1024 * 1024 * 1024;

  // 1. 获取物理可用内存 (MemAvailable)
  size_t mem_available = 0;
  std::ifstream meminfo("/proc/meminfo");
  if (meminfo.is_open())
  {
    std::string line;
    while (std::getline(meminfo, line))
    {
      if (line.find("MemAvailable:") != std::string::npos)
      {
        std::string val_str;
        for (char c : line)
        {
          if (isdigit(c))
            val_str += c;
        }
        // MemAvailable 单位是 kB，转换为 Bytes
        mem_available = std::stoull(val_str) * 1024;
        break;
      }
    }
    meminfo.close();
  }

  // 如果读取失败，回退到 sysinfo (不推荐，但作为 fallback)
  if (mem_available == 0)
  {
    struct sysinfo info;
    sysinfo(&info);
    mem_available = info.freeram; // 注意：freeram 比 MemAvailable 小得多，不包含 Cache
  }

  // 计算物理上的安全值
  size_t safe_physical = 0;
  if (mem_available > OS_HEADROOM)
  {
    safe_physical = mem_available - OS_HEADROOM;
  }
  else
  {
    safe_physical = 0; // 内存极其紧张，不建议分配
  }

  // 2. 获取用户锁页限制 (RLIMIT_MEMLOCK)
  struct rlimit limit;
  size_t safe_limit = 0;
  if (getrlimit(RLIMIT_MEMLOCK, &limit) == 0)
  {
    if (limit.rlim_cur == RLIM_INFINITY)
    {
      safe_limit = (size_t)-1; // 无限制
    }
    else
    {
      safe_limit = limit.rlim_cur;
    }
  }

  // 3. 决策：取两者的较小值
  size_t final_size = std::min(safe_physical, safe_limit);

  // --- 调试信息 (可选，实际使用可注释掉) ---
  /*
  std::cout << "[Memory Check]" << std::endl;
  std::cout << "  系统可用 (MemAvailable): " << mem_available / 1024 / 1024 << " MB" << std::endl;
  std::cout << "  用户限制 (ulimit -l)   : ";
  if (safe_limit == (size_t)-1) std::cout << "Unlimited";
  else std::cout << safe_limit / 1024 / 1024 << " MB";
  std::cout << std::endl;
  std::cout << "  扣除预留后建议值       : " << final_size / 1024 / 1024 << " MB" << std::endl;
  */

  return final_size;
}