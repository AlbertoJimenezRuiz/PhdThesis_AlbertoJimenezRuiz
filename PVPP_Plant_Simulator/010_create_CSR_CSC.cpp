#include "main.h"
#include "matrix_functions.h"
#include "sparse_matrix.h"

template <typename T> vector<T> flatten(const vector<vector<T>> &vec) {
  vector<T> res;

  for (const auto &x : vec)
    for (const auto &y : x)
      res.push_back(y);

  return res;
}

void create_CSR_CSC_matrices(Sparse_Matrix<FLOATING_TYPE> &mat,

                             vector<int> &CSR_start_rows,
                             vector<int> &CSR_position_columns,
                             vector<FLOATING_TYPE> &CSR_values,

                             vector<int> &CSC_start_columns,
                             vector<int> &CSC_position_rows,
                             vector<FLOATING_TYPE> &CSC_values,

                             vector<int> &CSR_corresponding_value_in_CSC,
                             vector<int> &CSC_corresponding_value_in_CSR,

                             vector<int> &CSR_start_from_diagonal,
                             vector<int> &CSC_start_from_diagonal) {

  CSR_start_rows.clear();
  CSR_position_columns.clear();
  CSR_values.clear();

  CSC_start_columns.clear();
  CSC_position_rows.clear();
  CSC_values.clear();

  CSR_corresponding_value_in_CSC.clear();
  CSC_corresponding_value_in_CSR.clear();

  const auto m = mat.m;
  const auto n = mat.n;

  vector<vector<int>> CSC_corresponding_value_in_CSR_temporary;
  vector<vector<int>> CSR_corresponding_value_in_CSC_temporary;

  for (unsigned int i = 0; i < n; i++) {
    CSC_corresponding_value_in_CSR_temporary.push_back(vector<int>());
  }
  for (unsigned int i = 0; i < m; i++) {
    CSR_corresponding_value_in_CSC_temporary.push_back(vector<int>());
  }

  for (unsigned int i = 0; i < m; i++) {
    CSR_start_rows.push_back((unsigned int)CSR_values.size());

    for (const auto &elem : mat.data[i]) {
      auto current_column = std::get<0>(elem);

      CSC_corresponding_value_in_CSR_temporary[current_column].push_back(
          (int)CSR_values.size());
      CSR_position_columns.push_back(current_column);
      CSR_values.push_back(std::get<1>(elem));
    }
  }
  CSR_start_rows.push_back((unsigned int)CSR_values.size());

  const auto mat_transposed = mat.transpose();
  for (unsigned int j = 0; j < n; j++) {
    CSC_start_columns.push_back((unsigned int)CSC_values.size());

    for (const auto &elem : mat_transposed.data[j]) {
      auto current_row = std::get<0>(elem);
      CSR_corresponding_value_in_CSC_temporary[current_row].push_back(
          (int)CSC_values.size());
      CSC_position_rows.push_back(current_row);
      CSC_values.push_back(std::get<1>(elem));
    }
  }
  CSC_start_columns.push_back((unsigned int)CSC_position_rows.size());

  CSC_corresponding_value_in_CSR =
      flatten(CSC_corresponding_value_in_CSR_temporary);
  CSR_corresponding_value_in_CSC =
      flatten(CSR_corresponding_value_in_CSC_temporary);

  CSR_start_from_diagonal.clear();
  for (int i = 0; i < (int)m; i++) {
    bool found = false;
    for (int j_index = CSR_start_rows[i]; j_index < CSR_start_rows[i + 1];
         j_index++) {
      auto j = CSR_position_columns[j_index];

      if (j > i) {
        CSR_start_from_diagonal.push_back(j_index);
        found = true;

        break;
      }
    }

    if (!found) {
      CSR_start_from_diagonal.push_back(CSR_start_rows[i + 1]);
    }
  }

  CSC_start_from_diagonal.clear();
  for (int j = 0; j < (int)n; j++) // Square matrix
  {
    bool found = false;
    for (int i_index = CSC_start_columns[j]; i_index < CSC_start_columns[j + 1];
         i_index++) {
      auto i = CSC_position_rows[i_index];

      if (i > j) {
        CSC_start_from_diagonal.push_back(i_index);
        found = true;

        break;
      }
    }

    if (!found) {
      CSC_start_from_diagonal.push_back(CSC_start_columns[j + 1]);
    }
  }
}

void fill_values_CSR_only(const Sparse_Matrix<FLOATING_TYPE> &mat,
                          const vector<int> &start_rows,
                          const vector<int> &position_columns,
                          vector<FLOATING_TYPE> &values) {

  for (unsigned int i = 0; i < mat.m; i++) {
    for (int col_idx = start_rows[i]; col_idx < start_rows[i + 1]; col_idx++) {
      values[col_idx] = mat.D(i, position_columns[col_idx]);
    }
  }
}

void fill_values_CSC_only(const Sparse_Matrix<FLOATING_TYPE> &mat,
                          const vector<int> &start_columns,
                          const vector<int> &position_rows,
                          vector<FLOATING_TYPE> &values) {

  auto mat_transposed = mat.transpose();

  for (unsigned int i = 0; i < mat_transposed.m; i++) {
    for (int col_idx = start_columns[i]; col_idx < start_columns[i + 1];
         col_idx++) {
      values[col_idx] = mat_transposed.D(i, position_rows[col_idx]);
    }
  }
}
