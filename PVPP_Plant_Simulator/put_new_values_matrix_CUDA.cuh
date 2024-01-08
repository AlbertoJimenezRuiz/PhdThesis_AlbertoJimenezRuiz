// Merge arr1[0..n1-1] and arr2[0..n2-1] into
// arr3[0..n1+n2-1]
void __device__ mergeArrays(const int *arr1, const int *arr2, const int n1,
                            const int n2, int *arr3) {

  if (arr2 == NULL) {
    for (int i = 0; i < n1; i++) {
      arr3[i] = arr1[i];
    }
  } else {

    int i = 0, j = 0, k = 0;

    // Traverse both array
    while (i < n1 && j < n2) {
      // Check if current element of first
      // array is smaller than current element
      // of second array. If yes, store first
      // array element and increment first array
      // index. Otherwise do same with second array
      if (arr1[i] < arr2[j])
        arr3[k++] = arr1[i++];
      else
        arr3[k++] = arr2[j++];
    }

    // Store remaining elements of first array
    while (i < n1)
      arr3[k++] = arr1[i++];

    // Store remaining elements of second array
    while (j < n2)
      arr3[k++] = arr2[j++];
  }
}

__global__ void put_new_values_matrix_CUDA(
    int matrix_size, const int *old_matrix_row_start,
    const int *old_raw_column_positions, const int *new_matrix_row_start,
    int *new_raw_column_positions, const int *row_temporary_memory_unified_temp,
    const int *d_data_mem_temporary_unified,
    const int *row_elements_temporary_unified_temp) {
  int current_row = blockIdx.x * blockDim.x + threadIdx.x;

  if (current_row >= matrix_size)
    return;

  mergeArrays(old_raw_column_positions + old_matrix_row_start[current_row],
              d_data_mem_temporary_unified +
                  row_temporary_memory_unified_temp[current_row],

              old_matrix_row_start[current_row + 1] -
                  old_matrix_row_start[current_row],
              row_elements_temporary_unified_temp[current_row],

              new_raw_column_positions + new_matrix_row_start[current_row]);
}
