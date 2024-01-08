__global__ void calculate_new_matrix_sizes_only_row_length_CUDA(
    int matrix_size, const int *old_matrix_row_start,
    const int *new_written_elements, int *new_matrix_row_start) {

  new_matrix_row_start[0] = 0;

  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < matrix_size) {
    new_matrix_row_start[tid + 1] =
        old_matrix_row_start[tid + 1] - old_matrix_row_start[tid];
    new_matrix_row_start[tid + 1] += new_written_elements[tid];
  }
}
