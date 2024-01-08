
// How much memory should be allocated per row?
// At least the same amount of memory
// Max limit: current size + certain number of elements OR maximum non-zero
// values per row Max number of zeros reported can be larger than the real
// number. That's why a limit is set.

__global__ void
calculate_new_temporary_memory_size(int matrix_size, int *row_temporary_memory,
                                    int *new_row_temporary_memory,
                                    int *written_elements_temporary_memory) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid >= matrix_size)
    return;

  const int new_max_allocation_memory_iteration = 128;

  int elements_per_row =
      row_temporary_memory[tid + 1] - row_temporary_memory[tid];

  // The first value is because we make a cudaMemset
  new_row_temporary_memory[tid + 1] =
      MAX2(elements_per_row,
           MIN2(elements_per_row + new_max_allocation_memory_iteration,
                written_elements_temporary_memory[tid]));
}
