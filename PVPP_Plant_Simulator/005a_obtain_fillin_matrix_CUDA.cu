#include "electrical_system.h"
#include "gpu_util.h"
#include "main.h"
#include "matrix_functions.h"
#include "simulator_functions.h"
#include "sparse_matrix.h"

#include <math.h>
#include <unistd.h>

#include "check_possible_new_nonzeros_CUDA.h"
#include "kernel_calculate_new_sizes.cuh"
#include "put_new_values_matrix_CUDA.cuh"
void call_cumsum_CUDA(int *array, int number_values);
#include "calculate_new_temporary_memory_size_CUDA.cuh"
#include "get_rows_and_columns_to_be_processed_CUDA.cuh"

vector<vector<int>>
obtain_fillin_matrix_CUDA(Sparse_Matrix<FLOATING_TYPE> &original_matrix) {

  assert(original_matrix.m == original_matrix.n);
  const int matrix_size = original_matrix.m;

  GPU_array_int *d_Start_Matrix_Rows;
  GPU_array_int *d_Positions_Column_Raw;
  GPU_array_int *d_new_matrix_row_start;
  GPU_array_int *d_new_raw_column_positions;

  auto original_matrix_vector_place =
      original_matrix.convert_to_boolean_equivalent_rowmajor();
  create_csr_boolean_CUDA(original_matrix_vector_place, d_Start_Matrix_Rows,
                          d_Positions_Column_Raw);

  d_new_matrix_row_start = new GPU_array_int(matrix_size + 1);
  d_new_raw_column_positions = NULL;

  GPU_array_int d_written_elements_memory(matrix_size);

  bool *d_added_zero_somewhere;
  bool *d_must_allocate_memory;

  gpuErrchk(cudaMalloc(&d_added_zero_somewhere, sizeof(bool)));
  gpuErrchk(cudaMalloc(&d_must_allocate_memory, sizeof(bool)));

  GPU_array_uint8 d_rows_process(matrix_size);
  GPU_array_uint8 d_columns_process(matrix_size);

  // First process all columns
  d_rows_process.memset(1);
  d_columns_process.memset(1);

  GPU_array_uint8 d_added_hole_rows(matrix_size);
  GPU_array_uint8 d_added_hole_columns(matrix_size);

  GPU_array_int *d_row_temporary_memory = new GPU_array_int(matrix_size + 1);
  GPU_array_int *d_new_row_temporary_memory =
      new GPU_array_int(matrix_size + 1);

  d_row_temporary_memory->memset(0);
  d_new_row_temporary_memory->memset(
      0); // Important! that first element is zero. Will always be zero, won't
          // change in the whole function

  GPU_array_int *d_data_temporary =
      new GPU_array_int(d_row_temporary_memory->get_last_DtoH());
  ;

  // How it works:
  // Check no more fill-in
  // A y B are non-zero, but in position C, which in this example is zero, would
  // generate a new non-zero element due to the way the dq0 decomposition works
  //   |#------------------
  //   | #>B
  //   | ^#v
  //   | ^ #
  //   | ^ v#
  //   | ^ v #
  //   | ^ v  #
  //   | A<C   #
  //   |
  //   |----------------------
  //  Iterate until no more zeros are generate

  for (;;) {

    for (;;) { // Loop to know how much memory should be allocated in CUDA

      d_written_elements_memory.memset(0); // How many new non-zeros per row?
      d_added_hole_rows.memset(
          0); // Which rows should be processed to find new non-zeros?
      d_added_hole_columns.memset(
          0); // Which columns should be processed to find new non-zeros?

      gpuErrchk(cudaMemset(d_added_zero_somewhere, 0, sizeof(bool)));
      gpuErrchk(cudaMemset(d_must_allocate_memory, 0, sizeof(bool)));

      int *pointer_data_temporary = NULL;
      if (d_data_temporary)
        pointer_data_temporary = d_data_temporary->m_data;

      const int threads_search_zeros_function = 1024;

      check_possible_new_nonzeros_CUDA<<<
          DIVISION_UP(matrix_size, threads_search_zeros_function),
          threads_search_zeros_function>>>(
          *d_Start_Matrix_Rows, *d_Positions_Column_Raw, matrix_size,
          pointer_data_temporary, *d_row_temporary_memory,
          d_written_elements_memory,

          d_rows_process, d_columns_process,

          d_added_hole_rows, d_added_hole_columns,

          d_added_zero_somewhere, d_must_allocate_memory);

      gpuErrchk(cudaDeviceSynchronize());

      bool temp_must_allocate_memory;
      gpuErrchk(cudaMemcpy(&temp_must_allocate_memory, d_must_allocate_memory,
                           sizeof(bool), cudaMemcpyDeviceToHost));

      if (!temp_must_allocate_memory) // No more memory allocation
        break;

      // Check how much new memory per row to calculate
      const int blocksize_calculate_new_temporary_memory = 1024;
      calculate_new_temporary_memory_size<<<
          DIVISION_UP(matrix_size, blocksize_calculate_new_temporary_memory),
          blocksize_calculate_new_temporary_memory>>>(
          matrix_size, *d_row_temporary_memory, *d_new_row_temporary_memory,
          d_written_elements_memory);

      // We have all elements per row. After concatenating them,
      // where does the row inside the vector start? First row starts in
      // position 0,
      // next where the former starts and so on. The operation cumsum solves
      // this issue.
      call_cumsum_CUDA(*d_new_row_temporary_memory, matrix_size + 1);

      // Exchange pointers. They have the same size.
      std::swap(d_row_temporary_memory, d_new_row_temporary_memory);
      delete d_data_temporary;
      d_data_temporary =
          new GPU_array_int(d_row_temporary_memory->get_last_DtoH());

      gpuErrchk(cudaDeviceSynchronize());
    }

    // How many column will the new matrix have given the number of non-zeros
    // obtained?
    const int blocksize_calculate_new_sizes = 1024;
    calculate_new_matrix_sizes_only_row_length_CUDA<<<
        DIVISION_UP(matrix_size, blocksize_calculate_new_sizes),
        blocksize_calculate_new_sizes>>>(matrix_size, *d_Start_Matrix_Rows,
                                         d_written_elements_memory,
                                         *d_new_matrix_row_start);

    // Do the cumsum. Justification similar as above
    call_cumsum_CUDA(*d_new_matrix_row_start, matrix_size + 1);

    // Allocate memory so that the new matrix fits.
    d_new_raw_column_positions =
        new GPU_array_int(d_new_matrix_row_start->get_last_DtoH());

    // Take for each row the non zeros of the old and new matrix
    // and unify in a single vector. Both are ordered, and so should the result
    // Time to create the new vector to merge both vectors
    const int blocksize_put_new_values_matrix_CUDA = 32 * 4;
    assert(d_data_temporary);
    put_new_values_matrix_CUDA<<<
        DIVISION_UP(matrix_size, blocksize_put_new_values_matrix_CUDA),
        blocksize_put_new_values_matrix_CUDA>>>(
        matrix_size, *d_Start_Matrix_Rows, *d_Positions_Column_Raw,
        *d_new_matrix_row_start, *d_new_raw_column_positions,
        *d_row_temporary_memory, *d_data_temporary, d_written_elements_memory);

    // This new non-zeros might mean new non-zeros. In which rows could they be
    // the new non-zeros be ?
    //  Do this to not iterate over all columns and rows. In this way the
    //  program is faster
    const int blocksize_kernel_get_rows_and_columns_to_be_processed = 1024;
    get_rows_and_columns_to_be_processed<<<
        DIVISION_UP(matrix_size,
                    blocksize_kernel_get_rows_and_columns_to_be_processed),
        blocksize_kernel_get_rows_and_columns_to_be_processed>>>(
        matrix_size, d_rows_process, d_columns_process, d_added_hole_rows,
        d_added_hole_columns, *d_Positions_Column_Raw, *d_Start_Matrix_Rows);

    std::swap(d_Start_Matrix_Rows, d_new_matrix_row_start);

    delete d_Positions_Column_Raw;
    d_Positions_Column_Raw = d_new_raw_column_positions;
    d_new_raw_column_positions = NULL;

    // If not a single element, was added, there is no more fill-in

    bool temp_added_zero_somewhere;
    gpuErrchk(cudaMemcpy(&temp_added_zero_somewhere, d_added_zero_somewhere,
                         sizeof(bool), cudaMemcpyDeviceToHost));

    if (!temp_added_zero_somewhere)
      break;
  }

  // Get Final matrix. Generate a vector where the vector of each row that
  // contains all columns with a non-zero
  vector<vector<int>> final_result;
  {
    vector<int> positions_matrix_new_represent(matrix_size + 1);
    d_Start_Matrix_Rows->Copy_DtoH(&positions_matrix_new_represent[0]);
    vector<int> data_matrix_new_represent(
        positions_matrix_new_represent[matrix_size]);
    d_Positions_Column_Raw->Copy_DtoH(&data_matrix_new_represent[0]);
    for (int i = 0; i < matrix_size; i++) {
      vector<int> new_row;

      for (int idx_j = positions_matrix_new_represent[i];
           idx_j < positions_matrix_new_represent[i + 1]; idx_j++) {
        new_row.push_back(data_matrix_new_represent[idx_j]);
      }
      final_result.push_back(new_row);
    }
  }

  gpuErrchk(cudaFree(d_added_zero_somewhere));
  gpuErrchk(cudaFree(d_must_allocate_memory));

  delete d_Start_Matrix_Rows;
  delete d_Positions_Column_Raw;
  delete d_new_matrix_row_start;
  delete d_row_temporary_memory;
  delete d_new_row_temporary_memory;
  delete d_data_temporary;

  return final_result;
}
