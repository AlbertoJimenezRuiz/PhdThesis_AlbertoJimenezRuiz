#include "main.h"
#include "sparse_matrix.h"

#include <iterator>
#include <numeric>
#include <string_view>
#include <vector>

template <typename container>
container unique_merge(container c1, container c2) {
  std::sort(c1.begin(), c1.end());
  std::sort(c2.begin(), c2.end());
  container mergeTarget;
  std::merge(c1.begin(), c1.end(), c2.begin(), c2.end(),
             std::insert_iterator<container>(mergeTarget, mergeTarget.end()));
  auto last = std::unique(mergeTarget.begin(), mergeTarget.end());
  mergeTarget.erase(last, mergeTarget.end());

  return mergeTarget;
}

std::vector<int> get_row_elements_position(Sparse_Matrix<FLOATING_TYPE> &mat,
                                           int row) {
  std::vector<int> row_elements;
  row_elements.reserve(mat.data[row].size());
  for (const auto &keyValue : mat.data[row]) {
    row_elements.push_back(keyValue.first);
  }
  return row_elements;
}

template <typename T, typename U>
void exchange_map(std::map<T, U> &data, const T f1, const T f2) {
  std::map<T, U> new_data;

  for (const auto &elem : data) {
    T col_nueva;

    if (elem.first == f1) {
      col_nueva = f2;
    } else if (elem.first == f2) {
      col_nueva = f1;
    } else {
      col_nueva = elem.first;
    }

    new_data[col_nueva] = elem.second;
  }

  data = new_data;
}

void interchange_rows_columns_matrix_and_tranpose_fast(
    Sparse_Matrix<FLOATING_TYPE> &mat,
    Sparse_Matrix<FLOATING_TYPE> &mat_transposed, int f1, int f2) {
  // Exchange row f1 and row f2 and col1 with col2, for both mat and its
  // transpose

  std::vector<int> rows_with_elements_to_exchange =
      unique_merge(unique_merge(get_row_elements_position(mat, f1),
                                get_row_elements_position(mat, f2)),
                   unique_merge(get_row_elements_position(mat_transposed, f1),
                                get_row_elements_position(mat_transposed, f2)));

  for (const auto &current_row : rows_with_elements_to_exchange) {
    exchange_map(mat.data[current_row], f1, f2);
    exchange_map(mat_transposed.data[current_row], f1, f2);
  }

  std::swap(mat.data[f1], mat.data[f2]);
  std::swap(mat_transposed.data[f1], mat_transposed.data[f2]);
}

void interchange_only_rows_matrix_and_tranpose_fast(
    Sparse_Matrix<FLOATING_TYPE> &mat,
    Sparse_Matrix<FLOATING_TYPE> &mat_transposed, int f1, int f2) {

  std::vector<int> rows_with_elements_to_exchange =
      unique_merge(unique_merge(get_row_elements_position(mat, f1),
                                get_row_elements_position(mat, f2)),
                   unique_merge(get_row_elements_position(mat_transposed, f1),
                                get_row_elements_position(mat_transposed, f2)));

  for (const auto &current_row : rows_with_elements_to_exchange) {
    exchange_map(mat_transposed.data[current_row], f1, f2);
  }

  std::swap(mat.data[f1], mat.data[f2]);
}
