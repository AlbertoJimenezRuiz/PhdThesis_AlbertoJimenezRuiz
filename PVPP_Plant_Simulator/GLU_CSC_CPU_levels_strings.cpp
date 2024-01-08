#include "main.h"

void factorize_one_column_GLU_CSC_CPU(const int j,

                                      const int *CSR_start_rows,
                                      const int *CSR_position_columns,
                                      const int *CSR_corresponding_value_in_CSC,

                                      const int *CSC_start_columns,
                                      const int *CSC_position_rows,
                                      FLOATING_TYPE *CSC_values,

                                      const int *CSR_start_from_diagonal,
                                      const int *CSC_start_from_diagonal) {
  if (CSC_start_columns[j] == CSC_start_from_diagonal[j]) {
    std::cerr
        << "ERROR CSC_start_columns[j]==CSC_start_from_diagonal[j]. Column: "
        << j << std::endl;
    assert(0);
  }

  FLOATING_TYPE large_number =
      CSC_values[CSC_start_from_diagonal[j] -
                 1]; // Previous value is the diagonal one

  if (std::abs(large_number) < 1e-8) {
    std::cerr << "ERROR std::abs(large_number) < 1e-8. Column: " << j
              << std::endl;
    assert(0);
  }

  for (int index_col_1 = CSC_start_from_diagonal[j];
       index_col_1 < CSC_start_columns[j + 1]; index_col_1++) {
    CSC_values[index_col_1] /= large_number;
  }

  for (int index_row_2 = CSR_start_from_diagonal[j];
       index_row_2 < CSR_start_rows[j + 1]; index_row_2++) {
    int columns_substitution = CSR_position_columns[index_row_2];

    FLOATING_TYPE &dominant_value =
        CSC_values[CSR_corresponding_value_in_CSC[index_row_2]];

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

        CSC_values[index_col_1] -= elem * dominant_value;

        break;
      }
    }
  }
}

void factorize_all_chains_of_columns_in_level_GLU_CSC_CPU(
    const int *start_levels_unified, const int current_level,
    const int *start_chains_columns_unified, const int *chains_columns_unified,

    const int *CSR_start_rows, const int *CSR_position_columns,
    const int *CSR_corresponding_value_in_CSC,

    const int *CSC_start_columns, const int *CSC_position_rows,

    FLOATING_TYPE *CSC_values,

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

      factorize_one_column_GLU_CSC_CPU(
          chains_columns_unified[idx_current_column],

          CSR_start_rows, CSR_position_columns, CSR_corresponding_value_in_CSC,

          CSC_start_columns, CSC_position_rows, CSC_values,

          CSR_start_from_diagonal, CSC_start_from_diagonal);
    }
  }
}

void GLU_CSC_CPU_sequential(const int n,

                            const int *CSR_start_rows,
                            const int *CSR_position_columns,
                            const int *CSR_corresponding_value_in_CSC,

                            const int *CSC_start_columns,
                            const int *CSC_position_rows,
                            FLOATING_TYPE *CSC_values,

                            const int *CSR_start_from_diagonal,
                            const int *CSC_start_from_diagonal) {

  for (int j = 0; j < n; j++) {

    factorize_one_column_GLU_CSC_CPU(
        j,

        CSR_start_rows, CSR_position_columns, CSR_corresponding_value_in_CSC,

        CSC_start_columns, CSC_position_rows, CSC_values,

        CSR_start_from_diagonal, CSC_start_from_diagonal);
  }
}
