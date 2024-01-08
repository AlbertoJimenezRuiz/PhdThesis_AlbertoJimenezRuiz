#include "main.h"

void factorize_one_column_GLU_CSR_CPU(const int j,

                                      const int *CSR_start_rows,
                                      const int *CSR_position_columns,
                                      FLOATING_TYPE *CSR_values,

                                      const int *CSC_start_columns,
                                      const int *CSC_position_rows,
                                      const int *CSC_corresponding_value_in_CSR,

                                      const int *CSR_start_from_diagonal,
                                      const int *CSC_start_from_diagonal) {

  if (CSR_start_rows[j] == CSR_start_from_diagonal[j]) {
    std::cerr << "ERROR CSR_start_rows[j]==CSR_start_from_diagonal[j]. Column: "
              << j << std::endl;
    assert(0);
  }

  FLOATING_TYPE large_number =
      CSR_values[CSR_start_from_diagonal[j] -
                 1]; // The previous value is the one in the diagonal

  if (std::abs(large_number) < 1e-8) {
    std::cerr << "ERROR std::abs(large_number) < 1e-8. Column: " << j
              << std::endl;
    assert(0);
  }

  for (int index_col_2 = CSC_start_from_diagonal[j];
       index_col_2 < CSC_start_columns[j + 1]; index_col_2++) {
    int rows_substitute = CSC_position_rows[index_col_2];

    FLOATING_TYPE &dominant_value =
        CSR_values[CSC_corresponding_value_in_CSR[index_col_2]];
    dominant_value /= large_number;

    for (int index_row_3 = CSR_start_from_diagonal[j];
         index_row_3 < CSR_start_rows[j + 1]; index_row_3++) {
      int current_column = CSR_position_columns[index_row_3];
      FLOATING_TYPE elem = CSR_values[index_row_3];

      for (int index_row_1 = CSR_start_rows[rows_substitute];
           index_row_1 < CSR_start_rows[rows_substitute + 1]; index_row_1++) {
        int possible_col_1 = CSR_position_columns[index_row_1];

        if (possible_col_1 != current_column)
          continue;

        CSR_values[index_row_1] -= elem * dominant_value;

        break;
      }
    }
  }
}

void factorize_all_chains_of_columns_in_level_GLU_CSR_CPU(
    const int *start_levels_unified, const int current_level,
    const int *start_chains_columns_unified, const int *chains_columns_unified,

    const int *CSR_start_rows, const int *CSR_position_columns,
    FLOATING_TYPE *CSR_values,

    const int *CSC_start_columns, const int *CSC_position_rows,
    const int *CSC_corresponding_value_in_CSR,

    const int *CSR_start_from_diagonal, const int *CSC_start_from_diagonal) {

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
      factorize_one_column_GLU_CSR_CPU(
          chains_columns_unified[idx_current_column],

          CSR_start_rows, CSR_position_columns, CSR_values,

          CSC_start_columns, CSC_position_rows, CSC_corresponding_value_in_CSR,

          CSR_start_from_diagonal, CSC_start_from_diagonal);
    }
  }
}

void GLU_CSR_CPU_sequential(const int n,

                            const int *CSR_start_rows,
                            const int *CSR_position_columns,
                            FLOATING_TYPE *CSR_values,

                            const int *CSC_start_columns,
                            const int *CSC_position_rows,
                            const int *CSC_corresponding_value_in_CSR,

                            const int *CSR_start_from_diagonal,
                            const int *CSC_start_from_diagonal) {
  for (int j = 0; j < n; j++) {

    factorize_one_column_GLU_CSR_CPU(j,

                                     CSR_start_rows, CSR_position_columns,

                                     CSR_values,

                                     CSC_start_columns, CSC_position_rows,

                                     CSC_corresponding_value_in_CSR,

                                     CSR_start_from_diagonal,
                                     CSC_start_from_diagonal);
  }
}
