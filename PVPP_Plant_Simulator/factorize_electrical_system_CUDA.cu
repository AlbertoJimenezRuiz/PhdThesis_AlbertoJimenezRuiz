#include "factorize_electrical_system.h"
#include "factorize_matrix_GLU_CUDA.cuh"
#include "main.h"
#include "matrix_functions.h"
#include "sparse_matrix.h"

void factorize_electrical_system_CPU_sequential(
    struct_CPU_CUDA_arrays &CPU_and_CUDA_arrays, Time_Measurer &measurer) {
  measurer.measure("Before factorization");

  GLU_CSC_CPU_sequential(
      CPU_and_CUDA_arrays.n,

      &CPU_and_CUDA_arrays.mat_A_CSR_start_rows[0],
      &CPU_and_CUDA_arrays.mat_A_CSR_position_columns[0],
      &CPU_and_CUDA_arrays.mat_A_CSR_corresponding_value_in_CSC[0],

      &CPU_and_CUDA_arrays.mat_A_CSC_start_columns[0],
      &CPU_and_CUDA_arrays.mat_A_CSC_position_rows[0],
      &CPU_and_CUDA_arrays.mat_A_CSC_values[0],

      &CPU_and_CUDA_arrays.mat_A_CSR_start_from_diagonal[0],
      &CPU_and_CUDA_arrays.mat_A_CSC_start_from_diagonal[0]

  );

  measurer.measure("After factorization");
}

void factorize_electrical_system_CPU_levels(
    struct_CPU_CUDA_arrays &CPU_and_CUDA_arrays, Time_Measurer &measurer) {
  measurer.measure("Before factorization");

  for (int current_level = 0;
       current_level <
       (int)CPU_and_CUDA_arrays.start_levels_unified_LU.size() - 1;
       current_level++) {

    factorize_all_chains_of_columns_in_level_GLU_CSR_CPU(
        &CPU_and_CUDA_arrays.start_levels_unified_LU[0], current_level,
        &CPU_and_CUDA_arrays.start_chains_columns_unified_LU[0],
        &CPU_and_CUDA_arrays.chains_columns_unified_LU[0],

        &CPU_and_CUDA_arrays.mat_A_CSR_start_rows[0],
        &CPU_and_CUDA_arrays.mat_A_CSR_position_columns[0],
        &CPU_and_CUDA_arrays.mat_A_CSR_values[0],

        &CPU_and_CUDA_arrays.mat_A_CSC_start_columns[0],
        &CPU_and_CUDA_arrays.mat_A_CSC_position_rows[0],
        &CPU_and_CUDA_arrays.mat_A_CSC_corresponding_value_in_CSR[0],

        &CPU_and_CUDA_arrays.mat_A_CSR_start_from_diagonal[0],
        &CPU_and_CUDA_arrays.mat_A_CSC_start_from_diagonal[0]);

    measurer.measure("Level CPU Completed");
  }

  measurer.measure("After factorization");
}

void factorize_electrical_system_CUDA(
    struct_CPU_CUDA_arrays &CPU_and_CUDA_arrays, Time_Measurer &measurer) {
  bool is_data_in_gpu = false;

  measurer.measure("Before factorization");

  for (int current_level = 0;
       current_level <
       (int)CPU_and_CUDA_arrays.start_levels_unified_LU.size() - 1;
       current_level++) {

    int threads_per_column = 1;
    int strings_per_block = 256;

    if (CPU_and_CUDA_arrays.max_elements_per_column[current_level] > 64) {
      threads_per_column = 32;
      strings_per_block = 32;
    }

    int number_strings_of_columns =
        CPU_and_CUDA_arrays.levels_with_strings_LU[current_level].size();

    int number_blocks =
        DIVISION_UP(number_strings_of_columns, strings_per_block);
    if (number_blocks > 2048)
      number_blocks = 2048;

    if (current_level <
        (int)CPU_and_CUDA_arrays.start_levels_unified_LU.size() - 3) {

      if (!is_data_in_gpu) {
        CPU_and_CUDA_arrays.CUDA_mat_A_CSC_values->Copy_HtoD(
            CPU_and_CUDA_arrays.mat_A_CSC_values);
        is_data_in_gpu = true;
      }
      factorize_all_chains_of_columns_in_level_GLU_CSC_CUDA<<<
          number_blocks, dim3(threads_per_column, strings_per_block)>>>(
          *CPU_and_CUDA_arrays.CUDA_start_levels_unified_LU, current_level,
          *CPU_and_CUDA_arrays.CUDA_start_chains_columns_unified_LU,
          *CPU_and_CUDA_arrays.CUDA_chains_columns_unified_LU,

          *CPU_and_CUDA_arrays.CUDA_mat_A_CSR_start_rows,
          *CPU_and_CUDA_arrays.CUDA_mat_A_CSR_position_columns,
          *CPU_and_CUDA_arrays.CUDA_mat_A_CSR_corresponding_value_in_CSC,

          *CPU_and_CUDA_arrays.CUDA_mat_A_CSC_start_columns,
          *CPU_and_CUDA_arrays.CUDA_mat_A_CSC_position_rows,
          *CPU_and_CUDA_arrays.CUDA_mat_A_CSC_values,

          *CPU_and_CUDA_arrays.CUDA_mat_A_CSR_start_from_diagonal,
          *CPU_and_CUDA_arrays.CUDA_mat_A_CSC_start_from_diagonal

      );

      gpuErrchk(cudaDeviceSynchronize());

      measurer.measure("Level CUDA Complete");
    } else {
      if (is_data_in_gpu) {
        CPU_and_CUDA_arrays.CUDA_mat_A_CSC_values->Copy_DtoH(
            CPU_and_CUDA_arrays.mat_A_CSC_values);

        is_data_in_gpu = false;
      }

      factorize_all_chains_of_columns_in_level_GLU_CSC_CPU(
          &CPU_and_CUDA_arrays.start_levels_unified_LU[0], current_level,
          &CPU_and_CUDA_arrays.start_chains_columns_unified_LU[0],
          &CPU_and_CUDA_arrays.chains_columns_unified_LU[0],

          &CPU_and_CUDA_arrays.mat_A_CSR_start_rows[0],
          &CPU_and_CUDA_arrays.mat_A_CSR_position_columns[0],
          &CPU_and_CUDA_arrays.mat_A_CSR_corresponding_value_in_CSC[0],

          &CPU_and_CUDA_arrays.mat_A_CSC_start_columns[0],
          &CPU_and_CUDA_arrays.mat_A_CSC_position_rows[0],
          &CPU_and_CUDA_arrays.mat_A_CSC_values[0],

          &CPU_and_CUDA_arrays.mat_A_CSR_start_from_diagonal[0],
          &CPU_and_CUDA_arrays.mat_A_CSC_start_from_diagonal[0]);

      measurer.measure("Level CPU Completed");
    }
  }
  if (is_data_in_gpu) {
    CPU_and_CUDA_arrays.CUDA_mat_A_CSC_values->Copy_DtoH(
        CPU_and_CUDA_arrays.mat_A_CSC_values);
    is_data_in_gpu = false;
  }

  measurer.measure("After factorization");
}
