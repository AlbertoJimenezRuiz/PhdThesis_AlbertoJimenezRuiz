#include "main.h"

__device__ int get_idx_first_summand(int level, int tid) {
  return level + (2 * level * (tid / level)) - 1;
}

__device__ int get_idx_second_summand(int level, int tid) {
  return level + (tid % level) + (2 * level * (tid / level));
}
__global__ void cumsum_CUDA(int *array, int number_values, int level) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  int idx_first = get_idx_first_summand(level, tid);
  int idx_second = get_idx_second_summand(level, tid);

  if (idx_second >= number_values)
    return;

  array[idx_second] += array[idx_first];
}

void call_cumsum_CUDA(int *array, int number_values) {
  const int cumsum_blocksize = 1024;
  int current_level_cumsum = 1;
  while (current_level_cumsum < 2 * number_values) {
    cumsum_CUDA<<<DIVISION_UP(number_values, cumsum_blocksize),
                  cumsum_blocksize>>>(array, number_values,
                                      current_level_cumsum);
    current_level_cumsum *= 2;
  }
}
