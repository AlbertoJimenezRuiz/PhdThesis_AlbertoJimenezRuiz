#include "main.h"
#include "matrix_functions.h"
#include "sparse_matrix.h"

void interchange_rows_columns_matrix_and_tranpose_fast(
    Sparse_Matrix<FLOATING_TYPE> &mat,
    Sparse_Matrix<FLOATING_TYPE> &mat_transposed, int f1, int f2);
void interchange_only_rows_matrix_and_tranpose_fast(
    Sparse_Matrix<FLOATING_TYPE> &mat,
    Sparse_Matrix<FLOATING_TYPE> &mat_transposed, int f1, int f2);

void remove_zeros_diagonal(Sparse_Matrix<FLOATING_TYPE> &input_output,
                           std::vector<int> &interchanges_position_row_columns,
                           bool move_only_rows) {

  Sparse_Matrix<FLOATING_TYPE> Mat_horizontal = input_output;
  Sparse_Matrix<FLOATING_TYPE> Mat_vertical = Mat_horizontal.transpose();

  assert(Mat_horizontal.m == Mat_horizontal.n);

  int n = Mat_horizontal.n;

  for (int current_row = 0; current_row < n; current_row++) {
    if (Mat_horizontal.exists(current_row, current_row))
      continue;

    /*
     * The item marked "O" is empty. Will it be filled in first?
     * It is necessary to see if when doing the decomposition this situation
     * occurs, that the column of element X on the left is equal to the row of
     * the upper element. It does not guarantee that the zero then cancels out,
     * but this matrix is already doing well.
     *
     * #····
     * ·#·X·
     * ··#··
     * ·X·O·
     * ····#
     */
    // Are there other number that might fill that hole?

    bool would_fill = false;

    for (const auto &possible_row_element : Mat_horizontal.data[current_row]) {
      if (would_fill)
        break;

      const auto row_position = possible_row_element.first;

      if (row_position >= current_row)
        break;

      for (const auto &possible_column_element :
           Mat_vertical.data[current_row]) {
        const auto column_position = possible_column_element.first;

        if (column_position == row_position) {
          would_fill = true;
        } else if (column_position > row_position) {
          // It won't be found in any further element
          break;
        }
      }
    }

    if (!would_fill) {

      int row_interchange = -1;

      for (const auto &possible_column_element :
           Mat_vertical.data[current_row]) {
        const auto column_position = possible_column_element.first;

        // First row after this one which has this element.
        // std::map is guaranteed to be ordered
        if (column_position > current_row) {
          row_interchange = column_position;
          break;
        }
      }

      assert(row_interchange != -1);

      // Time to interchange rows and columns

      if (move_only_rows) {
        interchange_only_rows_matrix_and_tranpose_fast(
            Mat_horizontal, Mat_vertical, current_row, row_interchange);
      } else {
        interchange_rows_columns_matrix_and_tranpose_fast(
            Mat_horizontal, Mat_vertical, current_row, row_interchange);
      }

      std::swap(interchanges_position_row_columns[current_row],
                interchanges_position_row_columns[row_interchange]);
    }
  }

  input_output = Mat_horizontal;
}
