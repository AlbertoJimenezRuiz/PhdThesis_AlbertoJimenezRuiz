
cuComplexType value_temp = CREATE_COMPLEX_CUDA(0, 0);

if (tid_stage2b < number_elements_max_stage2b) {
  shared_memory[threadIdx.x] = MULTIPLY_COMPLEX_CUDA(
      input_current_multipliers_stage2b, temporary_vector_1[tid_stage2b + 1]);
}

__syncthreads();

// Must use cooperative groups here. If not, it hangs.
for (unsigned int s = ELECTRICAL_SYSTEM_BLOCKSIZE / 2; s > 0; s >>= 1) {
  if (tid_stage2b < number_elements_max_stage2b) {
    if (threadIdx.x < s &&
        (tid_stage2b + s < number_elements_max_stage2b)) // Do only depending on
                                                         // current vector size
    {
      shared_memory[threadIdx.x] = ADD_COMPLEX_CUDA(
          shared_memory[threadIdx.x], shared_memory[threadIdx.x + s]);
    }
  }
  __syncthreads();
}

if (tid_stage2b < number_elements_max_stage2b) {
  if (threadIdx.x == 0) {
    cuComplexType temporary_val_stage3 = shared_memory[threadIdx.x];
    atomicAdd(&temporary_vector_3[0].x, temporary_val_stage3.x);
    atomicAdd(&temporary_vector_3[0].y, temporary_val_stage3.y);
  }
}

__syncthreads();
