#include "main.h"

__device__ inline void factorize_one_column_GLU_CSR_CUDA(
    const int j,

    const int *__restrict__ CSR_start_rows,
    const int *__restrict__ CSR_position_columns,
    FLOATING_TYPE *__restrict__ CSR_values,

    const int *__restrict__ CSC_start_columns,
    const int *__restrict__ CSC_position_rows,

    const int *__restrict__ CSC_corresponding_place_in_CSR,

    const int *__restrict__ CSR_start_from_diagonal,
    const int *__restrict__ CSC_start_from_diagonal) {
  assert(CSR_start_rows[j] != CSR_start_from_diagonal[j]);
  FLOATING_TYPE large_number = CSR_values[CSR_start_from_diagonal[j] - 1];

  if (std::abs(large_number) < 1e-8) {
    assert(0);
  }

  for (int index_col_2 = CSC_start_from_diagonal[j] + threadIdx.x;
       index_col_2 < CSC_start_columns[j + 1]; index_col_2 += blockDim.x) {
    int rows_substitute = CSC_position_rows[index_col_2];

    FLOATING_TYPE &dominant_value =
        CSR_values[CSC_corresponding_place_in_CSR[index_col_2]];
    dominant_value /= large_number;

    for (int index_row_3 = CSR_start_from_diagonal[j];
         index_row_3 < CSR_start_rows[j + 1]; index_row_3++) {
      int current_column = CSR_position_columns[index_row_3];
      FLOATING_TYPE elem = CSR_values[index_row_3];

      bool none_found = true;
      for (int index_row_1 = CSR_start_rows[rows_substitute];
           index_row_1 < CSR_start_rows[rows_substitute + 1]; index_row_1++) {
        if (CSR_position_columns[index_row_1] == current_column) {
          custom_atomicAdd(&CSR_values[index_row_1], -elem * dominant_value);
          none_found = false;
          break;
        }
      }
      assert(!none_found);
    }
  }
}

__device__ inline void factorize_one_column_GLU_CSC_CUDA(
    const int j,

    const int *__restrict__ CSR_start_rows,
    const int *__restrict__ CSR_position_columns,
    const int *__restrict__ CSR_corresponding_place_in_CSC,

    const int *__restrict__ CSC_start_columns,
    const int *__restrict__ CSC_position_rows,
    FLOATING_TYPE *__restrict__ CSC_values,

    const int *__restrict__ CSR_start_from_diagonal,
    const int *__restrict__ CSC_start_from_diagonal) {
  assert(CSC_start_columns[j] != CSC_start_from_diagonal[j]);
  FLOATING_TYPE large_number = CSC_values[CSC_start_from_diagonal[j] - 1];

  if (std::abs(large_number) < 1e-8) {
    assert(0);
  }

  for (int index_col_1 = CSC_start_from_diagonal[j] + threadIdx.x;
       index_col_1 < CSC_start_columns[j + 1]; index_col_1 += blockDim.x) {
    CSC_values[index_col_1] /= large_number;
  }

  __syncthreads();

  for (int index_row_2 = CSR_start_from_diagonal[j] + threadIdx.x;
       index_row_2 < CSR_start_rows[j + 1]; index_row_2 += blockDim.x) {

    int columns_substitution = CSR_position_columns[index_row_2];

    FLOATING_TYPE &dominant_value =
        CSC_values[CSR_corresponding_place_in_CSC[index_row_2]];

    for (int index_col_3 = CSC_start_from_diagonal[j];
         index_col_3 < CSC_start_columns[j + 1]; index_col_3++) {
      int current_row = CSC_position_rows[index_col_3];
      FLOATING_TYPE elem = CSC_values[index_col_3];

      for (int index_col_1 = CSC_start_columns[columns_substitution];
           index_col_1 < CSC_start_columns[columns_substitution + 1];
           index_col_1++) {
        int possible_row_1 = CSC_position_rows[index_col_1];

        if (possible_row_1 != current_row)
          continue;

        custom_atomicAdd(&CSC_values[index_col_1], -elem * dominant_value);

        break;
      }
    }
  }
}

__global__ void factorize_all_chains_of_columns_in_level_GLU_CSR_CUDA(
    const int *__restrict__ start_levels_unified, const int current_level,
    const int *__restrict__ start_chains_columns_unified,
    const int *__restrict__ chains_columns_unified,

    const int *__restrict__ CSR_start_rows,
    const int *__restrict__ CSR_position_columns,
    FLOATING_TYPE *__restrict__ CSR_values,

    const int *__restrict__ CSC_start_columns,
    const int *__restrict__ CSC_position_rows,
    const int *__restrict__ CSC_corresponding_place_in_CSR,

    const int *__restrict__ CSR_start_from_diagonal,
    const int *__restrict__ CSC_start_from_diagonal) {

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
      const int j = chains_columns_unified[idx_current_column];

      factorize_one_column_GLU_CSR_CUDA(
          j,

          CSR_start_rows, CSR_position_columns, CSR_values,

          CSC_start_columns, CSC_position_rows,

          CSC_corresponding_place_in_CSR,

          CSR_start_from_diagonal, CSC_start_from_diagonal);

      if (blockDim.x != 1) {
        __syncthreads();
      }
    }
  }
}

__global__ void factorize_all_chains_of_columns_in_level_GLU_CSC_CUDA(
    const int *__restrict__ start_levels_unified, const int current_level,
    const int *__restrict__ start_chains_columns_unified,
    const int *__restrict__ chains_columns_unified,

    const int *__restrict__ CSR_start_rows,
    const int *__restrict__ CSR_position_columns,
    const int *__restrict__ CSR_corresponding_place_in_CSC,

    const int *__restrict__ CSC_start_columns,
    const int *__restrict__ CSC_position_rows,
    FLOATING_TYPE *__restrict__ CSC_values,

    const int *__restrict__ CSR_start_from_diagonal,
    const int *__restrict__ CSC_start_from_diagonal) {

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
      const int j = chains_columns_unified[idx_current_column];

      factorize_one_column_GLU_CSC_CUDA(
          j,

          CSR_start_rows, CSR_position_columns, CSR_corresponding_place_in_CSC,

          CSC_start_columns, CSC_position_rows, CSC_values,

          CSR_start_from_diagonal, CSC_start_from_diagonal);

      if (blockDim.x != 1) {
        __syncthreads();
      }
    }
  }
}
