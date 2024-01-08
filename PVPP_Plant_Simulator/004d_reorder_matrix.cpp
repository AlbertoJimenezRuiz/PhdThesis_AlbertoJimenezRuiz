#include "main.h"
#include "matrix_functions.h"

void algorithm_amd_cpp(Sparse_Matrix<FLOATING_TYPE> &matrix,
                       std::vector<int> &interchanges_position_row_columns);
void remove_zeros_diagonal(Sparse_Matrix<FLOATING_TYPE> &input_output,
                           std::vector<int> &interchanges_position_row_columns,
                           bool move_only_rows);

Sparse_Matrix<FLOATING_TYPE>
reorder_matrix(const Sparse_Matrix<FLOATING_TYPE> &mat,
               std::vector<int> &interchanges_position_row_columns) {
  auto res = mat;
  assert(mat.m == mat.n);
  interchanges_position_row_columns.clear();
  interchanges_position_row_columns.resize(mat.m);
  for (unsigned int i = 0; i < mat.m; i++) {
    interchanges_position_row_columns[i] = i;
  }

  algorithm_amd_cpp(res, interchanges_position_row_columns);
  remove_zeros_diagonal(res, interchanges_position_row_columns, false);

  return res;
}

Sparse_Matrix<FLOATING_TYPE>
reorder_matrix(const Sparse_Matrix<FLOATING_TYPE> &mat,
               std::vector<int> &interchanges_position_row_columns,
               std::vector<int> &interchanges_position_row_columns_only_rows) {
  auto res = mat;
  assert(mat.m == mat.n);
  interchanges_position_row_columns.clear();
  interchanges_position_row_columns.resize(mat.m);
  interchanges_position_row_columns_only_rows.clear();
  interchanges_position_row_columns_only_rows.resize(mat.m);

  for (unsigned int i = 0; i < mat.m; i++) {
    interchanges_position_row_columns[i] = i;
    interchanges_position_row_columns_only_rows[i] = i;
  }

  algorithm_amd_cpp(res, interchanges_position_row_columns);
  remove_zeros_diagonal(res, interchanges_position_row_columns_only_rows, true);

  return res;
}
