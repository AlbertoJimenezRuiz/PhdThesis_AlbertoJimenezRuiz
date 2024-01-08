
if (tid_stage1_stage2a_stage3a < number_elements_max_stage1) {
  if (position_current_stage1 != -1) {
    shared_memory[threadIdx.x] =
        MULTIPLY_COMPLEX_CUDA(input_current_multipliers_stage1,
                              unified_input[position_current_stage1]);
  } else {
    shared_memory[threadIdx.x] = CREATE_COMPLEX_CUDA(0, 0);
  }
}

__syncthreads();
for (unsigned int s = ELECTRICAL_SYSTEM_BLOCKSIZE / 2; s >= warpSize; s >>= 1) {
  if (tid_stage1_stage2a_stage3a < number_elements_max_stage1) {
    if (number_element_current_stage1 <
        s) // If number_element_current_stage1 equal to -1, this if will ensure
           // it does not further execute
    {
      if (s < current_element_size_stage1) {
        shared_memory[threadIdx.x] = ADD_COMPLEX_CUDA(
            shared_memory[threadIdx.x], shared_memory[threadIdx.x + s]);
      }
    }
  }
  __syncthreads();
}

if (tid_stage1_stage2a_stage3a < number_elements_max_stage1) {

  cuComplexType temporary_val_stage1 = shared_memory[threadIdx.x];

  for (int s = warpSize / 2; s > 0; s >>= 1) {
    // If inserted directly, cooperative groups hang
    auto t1x = __shfl_xor_sync(0xFFFFFFFF, temporary_val_stage1.x, s);
    auto t1y = __shfl_xor_sync(0xFFFFFFFF, temporary_val_stage1.y, s);

    if (current_element_size_stage1 > s) {
      temporary_val_stage1.x += t1x;
      temporary_val_stage1.y += t1y;
    }
  }

  if (corresponding_element_stage1 != -1) {
    atomicAdd(&(temporary_vector_1[corresponding_element_stage1].x),
              temporary_val_stage1.x);
    atomicAdd(&(temporary_vector_1[corresponding_element_stage1].y),
              temporary_val_stage1.y);
  }
}
