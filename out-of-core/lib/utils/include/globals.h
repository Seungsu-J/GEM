#ifndef GLOBALS_H
#define GLOBALS_H

#include <cinttypes>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "defs.h"
// #include "cpuGraph.h"

extern int GPU_NUM;
extern uint32_t NUM_VQ;
extern uint32_t NUM_EQ;
extern uint32_t NUM_VLQ;
// extern uint32_t NUM_ELQ;
extern uint32_t NUM_VD;
extern uint32_t NUM_ED;
// extern uint32_t NUM_VLD;
// extern uint32_t NUM_ELD;
extern uint32_t COL_LEN;
extern uint32_t NUM_BLOCKS;
extern uint32_t THRESHOLD;

extern uint32_t NUM_CAN_UB;
extern __device__ __constant__ uint32_t C_NUM_CAN_UB;

extern uint32_t STOP_LEVEL;

extern uint32_t MAX_DATA_DEGREE;
extern uint32_t MAX_L_FREQ;
extern uint32_t TABLE_SIZE;

extern size_t SHARED_MEMORY_LIMIT;

extern __device__ __constant__ uint32_t C_NUM_VQ;
extern __device__ __constant__ uint32_t C_NUM_EQ;
extern __device__ __constant__ uint32_t C_NUM_VLQ;
// extern __device__ __constant__ uint32_t C_NUM_ELQ;
extern __device__ __constant__ uint32_t C_NUM_VD;
extern __device__ __constant__ uint32_t C_NUM_ED;
// extern __device__ __constant__ uint32_t C_NUM_VLD;
// extern __device__ __constant__ uint32_t C_NUM_ELD;
extern __device__ __constant__ uint32_t C_NUM_BLOCKS;
extern __device__ __constant__ uint32_t C_COL_LEN;
extern __device__ __constant__ uint32_t C_STOP_LEVEL;

extern __device__ __constant__ uint32_t C_MAX_DEGREE;
extern __device__ __constant__ uint32_t C_MAX_L_FREQ;
extern __device__ __constant__ uint32_t C_TABLE_SIZE;

extern __device__ __constant__ uint32_t C_THRESHOLD;

extern uint32_t QUERY_TYPE;

extern __device__ __constant__ uint32_t C_QUERY_TYPE;

#endif