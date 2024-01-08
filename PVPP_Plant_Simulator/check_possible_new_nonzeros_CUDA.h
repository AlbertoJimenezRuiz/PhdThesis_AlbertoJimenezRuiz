
template <typename T>
inline void __host__ __device__ move_memory_one_formward(T *src,
                                                         const int tam) {
  for (int idx = tam - 1; idx >= 0; idx--) {
    src[idx + 1] = src[idx];
  }
}

inline bool __host__ __device__ add_hole(const int row_new_element,
                                         int column_nuevo_elemento,
                                         int *data_temporary_memory,
                                         int *row_temporary_memory,
                                         int *written_elements_temporary,
                                         bool *must_allocate_memory) {

  int reserved_size = row_temporary_memory[row_new_element + 1] -
                      row_temporary_memory[row_new_element];
  int &current_written_elements = written_elements_temporary[row_new_element];
  int *current_temporary_pointer =
      data_temporary_memory + row_temporary_memory[row_new_element];

  int index = 0;
  for (; index < MIN2(current_written_elements, reserved_size); index++) {
    if (current_temporary_pointer[index] ==
        column_nuevo_elemento) // It was already put
    {
      return false;
    } else if (current_temporary_pointer[index] > column_nuevo_elemento) {
      break; // We have found the maximum
    }
  }

  // Need more memory?

  if (reserved_size <= current_written_elements) {
    // Can give false positives. Of course, the previous loop can't check this
    // new zero with the rest of zeros already in place due to lack of space
    current_written_elements++;
    *must_allocate_memory = true;
  } else {
    move_memory_one_formward(current_temporary_pointer + index,
                             current_written_elements - index);
    current_temporary_pointer[index] = column_nuevo_elemento;
    current_written_elements++;
  }

  return true;
}

void __host__ __device__ search_nonzero_row_matrix(
    int row_element_A, int *Start_Matrix_Rows, int *Positions_Column_Raw,
    int matrix_size, int *data_temporary_memory, int *row_temporary_memory,
    int *written_elements_temporary, uint8_t *rows_process,
    uint8_t *columns_process, uint8_t *added_hole_rows,
    uint8_t *added_hole_columns, bool *added_zero_somewhere,
    bool *must_allocate_memory) {
  if (row_element_A >= matrix_size)
    return;

  if (rows_process[row_element_A] == 0)
    return;

  // Check no new zeros
  //   |#------------------
  //   | #>B
  //   | ^#v
  //   | ^ #
  //   | ^ v#
  //   | ^ v #
  //   | ^ v  #
  //   | A<C   #
  //   |
  //   |----------------------
  //  if processing the column where value A is, and A,B are non-zero and C is a
  //  zero, then that is a fill-in value.

  const int *row_A_vect =
      Positions_Column_Raw + Start_Matrix_Rows[row_element_A];
  const int number_elements_row_A =
      Start_Matrix_Rows[row_element_A + 1] - Start_Matrix_Rows[row_element_A];

  for (int idx_col_row_A = 0; idx_col_row_A < number_elements_row_A;
       idx_col_row_A++) {
    const int column_elem_A = row_A_vect[idx_col_row_A];

    if (columns_process[column_elem_A] == 0)
      continue;

    // Only lower triangular
    if (row_element_A <= column_elem_A) {
      continue;
    }

    const int row_elem_B = column_elem_A;
    const auto &row_B_vect =
        Positions_Column_Raw + Start_Matrix_Rows[row_elem_B];
    const int number_elements_row_B =
        Start_Matrix_Rows[row_elem_B + 1] - Start_Matrix_Rows[row_elem_B];

    // Iterate over rows b and C

    //       |#------------------
    //  col b| #>B
    //       | ^#v
    //       | ^ #
    //       | ^ v#
    //       | ^ v #
    //       | ^ v  #
    //  col c| A<C   #
    //       |
    //       |----------------------

    // Is element C non-zero if B is on that column?
    // Advance B and C

    int idx_col_row_B = 0;
    int idx_col_row_C = idx_col_row_A + 1;

    // B cannot be on the left of A or on top. Advance

    for (;;) {
      const int column_elem_B = row_B_vect[idx_col_row_B];

      if (column_elem_B > column_elem_A) {
        break;
      }

      idx_col_row_B++;
    }

    while (idx_col_row_B <
           number_elements_row_B) // If end of B is reached, no more zeros
    {
      const int column_elem_B = row_B_vect[idx_col_row_B];

      int column_elem_C;

      if (idx_col_row_C == number_elements_row_A) {
        // A and C in the same column
        column_elem_C = matrix_size; // Just in case no more items in C
      } else {
        column_elem_C = row_A_vect[idx_col_row_C];
      }

      if (column_elem_B < column_elem_C) {
        // In this position there is hole in intersection row A and col B

        //       |#------------------
        //  col b| # B
        //       |  #
        //       |   #
        //       |    #
        //       |     #
        //       |      #
        //  col c| A     #   C
        //       |
        //       |----------------------

        bool hole_was_added =
            add_hole(row_element_A, column_elem_B, data_temporary_memory,
                     row_temporary_memory, written_elements_temporary,
                     must_allocate_memory);

        if (hole_was_added) {

          *added_zero_somewhere = true;
          added_hole_rows[row_element_A] = 1;
          added_hole_columns[column_elem_B] = 1;
        }

        // Advance B. Perhaps there might be more non-zero elements before C
        idx_col_row_B++;

      } else if (column_elem_B == column_elem_C) {
        // In this position advance B and C to check what happens next

        //       |#------------------
        //  col b| # B
        //       |  #V
        //       |   V
        //       |   V#
        //       |   V #
        //       |   V  #
        //  col c| A C   #
        //       |
        //       |----------------------

        idx_col_row_B++;
        idx_col_row_C++;

      } else {
        // Advance C to reach B

        //       |#------------------
        //  col b| #           B
        //       |  #
        //       |   #
        //       |    #
        //       |     #
        //       |      #
        //  col c| A   C #
        //       |
        //       |----------------------

        idx_col_row_C++;
      }
    }
  }
}

void __global__ check_possible_new_nonzeros_CUDA(
    int *Start_Matrix_Rows, int *Positions_Column_Raw, int matrix_size,
    int *data_temporary_memory, int *row_temporary_memory,
    int *written_elements_temporary, uint8_t *rows_process,
    uint8_t *columns_process, uint8_t *added_hole_rows,
    uint8_t *added_hole_columns, bool *added_zero_somewhere,
    bool *must_allocate_memory) {

  int row_element_A = blockIdx.x * blockDim.x + threadIdx.x;

  search_nonzero_row_matrix(
      row_element_A, Start_Matrix_Rows, Positions_Column_Raw, matrix_size,
      data_temporary_memory, row_temporary_memory, written_elements_temporary,
      rows_process, columns_process, added_hole_rows, added_hole_columns,
      added_zero_somewhere, must_allocate_memory);
}
