#include "cuda_helpers.cuh"

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

/**
 * @brief 激进预热函数：初始化 Context，锁定内存池，并按最大安全水位撑大池子
 * @param stream 用于异步操作的流
 * @param safety_buffer_mb 预留给系统和驱动的缓冲大小 (默认 500MB)
 */
void aggressive_warmup(int var_device_id, cudaStream_t stream, size_t safety_buffer_mb)
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
    return;
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

  // 5. 必须同步！
  // 确保上述操作全部在 GPU 端落地，Pool 状态完全就绪
  cuchk(cudaStreamSynchronize(stream));

  // std::cout << "[Init] Ready. Memory pool is fully populated." << std::endl;
}