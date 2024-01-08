#include "main.h"
#include "sparse_matrix.h"

#include "factorize_electrical_system.h"
#include "matrix_functions.h"

template <bool is_L>
void Substitution_LU_All_Chains_Columns_in_level_GLU_CSC_CPU(
    const int *start_levels_unified, const int current_level,
    const int *start_chains_columns_unified, const int *chains_columns_unified,
    const int *CSC_start_columns, const int *CSC_position_rows,
    const FLOATING_TYPE *CSC_values, const int *CSC_start_from_diagonal,

    FLOATING_TYPE *solution) {

  int index_position_chain_columns_level = start_levels_unified[current_level];
  int number_chains_columns_per_level =
      start_levels_unified[current_level + 1] -
      start_levels_unified[current_level];

  for (int current_chain = 0; current_chain < number_chains_columns_per_level;
       current_chain++) {

    const int index_start_chain_column =
        start_chains_columns_unified[index_position_chain_columns_level +
                                     current_chain];
    const int index_end_chain_column =
        start_chains_columns_unified[index_position_chain_columns_level +
                                     current_chain + 1];

    for (int idx_current_column = index_start_chain_column;
         idx_current_column < index_end_chain_column; idx_current_column++) {

      const auto j = chains_columns_unified[idx_current_column];

      if (is_L) {
        Process_Column_L_CSC_CPU(j, CSC_start_columns, CSC_position_rows,
                                 CSC_values, CSC_start_from_diagonal, solution);
      } else {
        Process_Column_U_CSC_CPU(j, CSC_start_columns, CSC_position_rows,
                                 CSC_values, CSC_start_from_diagonal, solution);
      }
    }
  }
}

void triangular_substitution_LU_CPU_sequential(
    struct_CPU_CUDA_arrays &CPU_and_CUDA_arrays, Time_Measurer &measurer,
    FLOATING_TYPE *solution) {
  measurer.measure("Before CPU Triangular Substitution");

  Substitute_LU_Vector_b_CSC_CPU(
      CPU_and_CUDA_arrays.n, &CPU_and_CUDA_arrays.mat_A_CSC_start_columns[0],
      &CPU_and_CUDA_arrays.mat_A_CSC_position_rows[0],
      &CPU_and_CUDA_arrays.mat_A_CSC_values[0],
      &CPU_and_CUDA_arrays.mat_A_CSC_start_from_diagonal[0], solution);

  measurer.measure("After CPU Triangular Substitution");
}

void __device__ Process_Column_L_CSC_CUDA(const int j,
                                          const int *CSC_start_columns,
                                          const int *CSC_position_rows,
                                          const FLOATING_TYPE *CSC_values,
                                          const int *CSC_start_from_diagonal,
                                          FLOATING_TYPE *b) {

  const auto val_b = b[j];
  for (int row_index = CSC_start_from_diagonal[j] + threadIdx.x;
       row_index < CSC_start_columns[j + 1]; row_index += blockDim.x) {
    custom_atomicAdd(&b[CSC_position_rows[row_index]],
                     -CSC_values[row_index] * val_b);
  }
}

void __device__ Process_Column_U_CSC_CUDA(const int j,
                                          const int *CSC_start_columns,
                                          const int *CSC_position_rows,
                                          const FLOATING_TYPE *CSC_values,
                                          const int *CSC_start_from_diagonal,
                                          FLOATING_TYPE *b) {
  auto index_diagonal = CSC_start_from_diagonal[j] - 1;
  const auto val_b = b[j] / CSC_values[index_diagonal];

  if (threadIdx.x == 0) {
    b[j] = val_b;
  }

  for (int row_index = CSC_start_columns[j] + threadIdx.x;
       row_index < index_diagonal; row_index += blockDim.x) {
    custom_atomicAdd(&b[CSC_position_rows[row_index]],
                     -CSC_values[row_index] * val_b);
  }
}

template <bool is_L>
__global__ void Substitution_LU_All_Chains_Columns_in_level_GLU_CSC_CUDA(
    const int *__restrict__ start_levels_unified, const int current_level,
    const int *__restrict__ start_chains_columns_unified,
    const int *__restrict__ chains_columns_unified,

    const int *__restrict__ CSC_start_columns,
    const int *__restrict__ CSC_position_rows,
    const FLOATING_TYPE *__restrict__ CSC_values,
    const int *__restrict__ CSC_start_from_diagonal,

    FLOATING_TYPE *__restrict__ solution) {

  int index_position_chain_columns_level = start_levels_unified[current_level];
  int number_chains_columns_per_level =
      start_levels_unified[current_level + 1] -
      start_levels_unified[current_level];

  for (int current_chain = blockIdx.x * blockDim.y + threadIdx.y;
       current_chain < number_chains_columns_per_level;
       current_chain += gridDim.x * blockDim.y) {

    const int index_start_chain_column =
        start_chains_columns_unified[index_position_chain_columns_level +
                                     current_chain];
    const int index_end_chain_column =
        start_chains_columns_unified[index_position_chain_columns_level +
                                     current_chain + 1];

    for (int idx_current_column = index_start_chain_column;
         idx_current_column < index_end_chain_column; idx_current_column++) {

      const auto j = chains_columns_unified[idx_current_column];

      if (is_L) {
        Process_Column_L_CSC_CUDA(j, CSC_start_columns, CSC_position_rows,
                                  CSC_values, CSC_start_from_diagonal,
                                  solution);
      } else {
        Process_Column_U_CSC_CUDA(j, CSC_start_columns, CSC_position_rows,
                                  CSC_values, CSC_start_from_diagonal,
                                  solution);
      }

      if (blockDim.x != 1) {
        __syncthreads();
      }
    }
  }
}

template <bool is_L>
void Substitution_LU_CPU_Arreglo(struct_CPU_CUDA_arrays &CPU_and_CUDA_arrays,
                                 Time_Measurer &measurer, int current_level,
                                 GPU_array_f *solution_CUDA,
                                 FLOATING_TYPE *solution,
                                 bool &is_data_in_gpu) {
  int threads_per_column = 1;
  int strings_per_block = 256;

  if (CPU_and_CUDA_arrays.max_elements_per_column[current_level] > 64) {
    threads_per_column = 32;
    strings_per_block = 32;
  }

  int number_strings_of_columns =
      CPU_and_CUDA_arrays.levels_with_strings_LU[current_level].size();

  int number_blocks = DIVISION_UP(number_strings_of_columns, strings_per_block);
  if (number_blocks > 2048)
    number_blocks = 2048;

  if (current_level <
      (int)CPU_and_CUDA_arrays.start_levels_unified_LU.size() - 3) {

    if (!is_data_in_gpu) {
      solution_CUDA->Copy_HtoD(solution);
      is_data_in_gpu = true;
    }

    if (is_L) {
      Substitution_LU_All_Chains_Columns_in_level_GLU_CSC_CUDA<true>
          <<<number_blocks, dim3(threads_per_column, strings_per_block)>>>(
              *CPU_and_CUDA_arrays.CUDA_start_levels_unified_L, current_level,
              *CPU_and_CUDA_arrays.CUDA_start_chains_columns_unified_L,
              *CPU_and_CUDA_arrays.CUDA_chains_columns_unified_L,
              *CPU_and_CUDA_arrays.CUDA_mat_A_CSC_start_columns,
              *CPU_and_CUDA_arrays.CUDA_mat_A_CSC_position_rows,
              *CPU_and_CUDA_arrays.CUDA_mat_A_CSC_values,
              *CPU_and_CUDA_arrays.CUDA_mat_A_CSC_start_from_diagonal,

              *solution_CUDA);
    } else {
      Substitution_LU_All_Chains_Columns_in_level_GLU_CSC_CUDA<false>
          <<<number_blocks, dim3(threads_per_column, strings_per_block)>>>(
              *CPU_and_CUDA_arrays.CUDA_start_levels_unified_U, current_level,
              *CPU_and_CUDA_arrays.CUDA_start_chains_columns_unified_U,
              *CPU_and_CUDA_arrays.CUDA_chains_columns_unified_U,
              *CPU_and_CUDA_arrays.CUDA_mat_A_CSC_start_columns,
              *CPU_and_CUDA_arrays.CUDA_mat_A_CSC_position_rows,
              *CPU_and_CUDA_arrays.CUDA_mat_A_CSC_values,
              *CPU_and_CUDA_arrays.CUDA_mat_A_CSC_start_from_diagonal,
              *solution_CUDA);
    }
    gpuErrchk(cudaDeviceSynchronize());
    measurer.measure("Level CUDA Complete");

  } else {
    if (is_data_in_gpu) {
      solution_CUDA->Copy_DtoH(solution);
      is_data_in_gpu = false;
    }

    if (is_L) {
      Substitution_LU_All_Chains_Columns_in_level_GLU_CSC_CPU<true>(
          &CPU_and_CUDA_arrays.start_levels_unified_L[0], current_level,
          &CPU_and_CUDA_arrays.start_chains_columns_unified_L[0],
          &CPU_and_CUDA_arrays.chains_columns_unified_L[0],
          &CPU_and_CUDA_arrays.mat_A_CSC_start_columns[0],
          &CPU_and_CUDA_arrays.mat_A_CSC_position_rows[0],
          &CPU_and_CUDA_arrays.mat_A_CSC_values[0],
          &CPU_and_CUDA_arrays.mat_A_CSC_start_from_diagonal[0], solution);
    } else {
      Substitution_LU_All_Chains_Columns_in_level_GLU_CSC_CPU<false>(
          &CPU_and_CUDA_arrays.start_levels_unified_U[0], current_level,
          &CPU_and_CUDA_arrays.start_chains_columns_unified_U[0],
          &CPU_and_CUDA_arrays.chains_columns_unified_U[0],
          &CPU_and_CUDA_arrays.mat_A_CSC_start_columns[0],
          &CPU_and_CUDA_arrays.mat_A_CSC_position_rows[0],
          &CPU_and_CUDA_arrays.mat_A_CSC_values[0],
          &CPU_and_CUDA_arrays.mat_A_CSC_start_from_diagonal[0], solution);
    }
    measurer.measure("Level CPU Complete");
  }
}

void triangular_substitution_LU_CUDA(
    struct_CPU_CUDA_arrays &CPU_and_CUDA_arrays, Time_Measurer &measurer,
    GPU_array_f *solution_CUDA, FLOATING_TYPE *solution) {

  measurer.measure("Before LU Substitution");

  bool is_data_in_gpu = false;

  for (int current_level = 0;
       current_level <
       (int)CPU_and_CUDA_arrays.start_levels_unified_L.size() - 1;
       current_level++) {

    Substitution_LU_CPU_Arreglo<true>(CPU_and_CUDA_arrays, measurer,
                                      current_level, solution_CUDA, solution,
                                      is_data_in_gpu);

    if (is_data_in_gpu) {
      solution_CUDA->Copy_DtoH(solution);
      is_data_in_gpu = false;
    }
  }

  for (int current_level = 0;
       current_level <
       (int)CPU_and_CUDA_arrays.start_levels_unified_U.size() - 1;
       current_level++) {

    Substitution_LU_CPU_Arreglo<false>(CPU_and_CUDA_arrays, measurer,
                                       current_level, solution_CUDA, solution,
                                       is_data_in_gpu);
  }

  if (is_data_in_gpu) {
    solution_CUDA->Copy_DtoH(solution);
    is_data_in_gpu = false;
  }

  measurer.measure("After LU Substitution");
}
