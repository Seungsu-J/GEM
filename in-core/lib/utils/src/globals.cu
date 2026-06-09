#include "globals.h"
// #include "cpuGraph.h"

int GPU_NUM;
uint32_t NUM_VQ;
uint32_t NUM_EQ;
uint32_t NUM_VLQ;
// uint32_t NUM_ELQ;
uint32_t NUM_VD;
uint32_t NUM_ED;
// uint32_t NUM_VLD;
// uint32_t NUM_ELD;
uint32_t COL_LEN;
uint32_t NUM_BLOCKS;
uint32_t THRESHOLD;

uint32_t NUM_CAN_UB;
__device__ __constant__ uint32_t C_NUM_CAN_UB;

uint32_t STOP_LEVEL;

uint32_t MAX_DATA_DEGREE;
uint32_t MAX_L_FREQ;
uint32_t TABLE_SIZE;

uint32_t QUERY_TYPE;

size_t SHARED_MEMORY_LIMIT;

__device__ __constant__ uint32_t C_NUM_VQ;
__device__ __constant__ uint32_t C_NUM_EQ;
__device__ __constant__ uint32_t C_NUM_VLQ;
// __constant__ uint32_t C_NUM_ELQ;
__device__ __constant__ uint32_t C_NUM_VD;
__device__ __constant__ uint32_t C_NUM_ED;
// __constant__ uint32_t C_NUM_VLD;
// __constant__ uint32_t C_NUM_ELD;
__device__ __constant__ uint32_t C_NUM_BLOCKS;
__device__ __constant__ uint32_t C_COL_LEN;
__device__ __constant__ uint32_t C_STOP_LEVEL;

__device__ __constant__ uint32_t C_MAX_DEGREE;
__device__ __constant__ uint32_t C_MAX_L_FREQ;
__device__ __constant__ uint32_t C_TABLE_SIZE;

__device__ __constant__ uint32_t C_THRESHOLD;
__device__ __constant__ uint32_t C_QUERY_TYPE;