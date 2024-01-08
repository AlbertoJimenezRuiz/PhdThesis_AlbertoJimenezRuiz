#include "main.h"

void Process_Column_L_CSC_CPU(const int j, const int *CSC_start_columns,
                              const int *CSC_position_rows,
                              const FLOATING_TYPE *CSC_values,
                              const int *CSC_start_from_diagonal,
                              FLOATING_TYPE *b) {
  for (int row_index = CSC_start_from_diagonal[j];
       row_index < CSC_start_columns[j + 1]; row_index++) {
    b[CSC_position_rows[row_index]] -= CSC_values[row_index] * b[j];
  }
}

void Process_Column_U_CSC_CPU(const int j, const int *CSC_start_columns,
                              const int *CSC_position_rows,
                              const FLOATING_TYPE *CSC_values,
                              const int *CSC_start_from_diagonal,
                              FLOATING_TYPE *b) {
  auto index_diagonal = CSC_start_from_diagonal[j] - 1;
  b[j] /= CSC_values[index_diagonal];

  for (int row_index = CSC_start_columns[j]; row_index < index_diagonal;
       row_index++) {
    b[CSC_position_rows[row_index]] -= CSC_values[row_index] * b[j];
  }
}
void Substitute_LU_Vector_b_CSC_CPU(const unsigned int n,
                                    const int *CSC_start_columns,
                                    const int *CSC_position_rows,
                                    const FLOATING_TYPE *CSC_values,
                                    const int *CSC_start_from_diagonal,
                                    FLOATING_TYPE *b) {

  for (int j = 0; j < (int)n; j++) {
    Process_Column_L_CSC_CPU(j, CSC_start_columns, CSC_position_rows,
                             CSC_values, CSC_start_from_diagonal, b);
  }

  for (int j = n - 1; j >= 0; j--) {
    Process_Column_U_CSC_CPU(j, CSC_start_columns, CSC_position_rows,
                             CSC_values, CSC_start_from_diagonal, b);
  }
}
