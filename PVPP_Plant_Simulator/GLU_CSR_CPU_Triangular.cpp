#include "main.h"

void Substitute_LU_Vector_b_CSR_CPU(const unsigned int m,
                                    const int *CSR_start_rows,
                                    const int *CSR_position_columns,
                                    const FLOATING_TYPE *CSR_values,
                                    const int *CSR_start_from_diagonal,
                                    FLOATING_TYPE *b) {

  for (int i = 0; i < (int)m; i++) {
    for (int indice_col = CSR_start_rows[i];
         indice_col < CSR_start_from_diagonal[i] - 1; indice_col++) {
      b[i] -= CSR_values[indice_col] * b[CSR_position_columns[indice_col]];
    }
  }

  for (int i = m - 1; i >= 0; i--) {
    for (int indice_col = CSR_start_from_diagonal[i];
         indice_col < CSR_start_rows[i + 1]; indice_col++) {
      b[i] -= CSR_values[indice_col] * b[CSR_position_columns[indice_col]];
    }

    b[i] /= CSR_values[CSR_start_from_diagonal[i] -
                       1]; // Previous one is the diagonal one
  }
}
