
if (tid_stage1_stage2a_stage3a < number_elements_max_stage2a) {
  if (position_current_stage2a != -1) {
    shared_memory[threadIdx.x] =
        MULTIPLY_COMPLEX_CUDA(input_current_multipliers_stage2a,
                              temporary_vector_1[position_current_stage2a]);
  } else {
    shared_memory[threadIdx.x] = CREATE_COMPLEX_CUDA(0, 0);
  }
}
__syncthreads();

for (unsigned int s = ELECTRICAL_SYSTEM_BLOCKSIZE / 2; s >= warpSize; s >>= 1) {
  if (tid_stage1_stage2a_stage3a < number_elements_max_stage2a) {
    if (number_element_current_stage2a <
        s) // If number_element_current_stage2a equal to -1, this if will ensure
           // it does not further execute
    {
      if (s < current_element_size_stage2a) {
        shared_memory[threadIdx.x] = ADD_COMPLEX_CUDA(
            shared_memory[threadIdx.x], shared_memory[threadIdx.x + s]);
      }
    }
  }
  __syncthreads();
}

if (tid_stage1_stage2a_stage3a < number_elements_max_stage2a) {
  cuComplexType temporary_val_stage2 = shared_memory[threadIdx.x];
  for (int s = warpSize / 2; s > 0; s >>= 1) {
    auto t2x = __shfl_xor_sync(0xFFFFFFFF, temporary_val_stage2.x, s);
    auto t2y = __shfl_xor_sync(0xFFFFFFFF, temporary_val_stage2.y, s);
    if (current_element_size_stage2a > s) {
      temporary_val_stage2.x += t2x;
      temporary_val_stage2.y += t2y;
    }
  }

  if (corresponding_element_stage2a != -1) {
    temporary_vector_2[corresponding_element_stage2a] = temporary_val_stage2;
  }
}
