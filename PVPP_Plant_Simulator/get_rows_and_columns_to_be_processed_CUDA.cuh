

__global__ void get_rows_and_columns_to_be_processed(
    int matrix_size, uint8_t *rows_current_process,
    uint8_t *columns_current_process,

    const uint8_t *added_hole_rows, const uint8_t *added_hole_columns,

    const int *Positions_Column_Raw, const int *Start_Matrix_Rows) {

  // What rows must be processed? Consider the new hole D

  //  |#------------------
  //  | #
  //  |  #-----D
  //  |  |#
  //  |  | #
  //  |  |  #
  //  |  |   #
  //  |       #
  //  |
  //  |----------------------

  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid >= matrix_size)
    return;

  // Zero all elements of both rows and columns. Until now, no row and column
  // must be processed
  uint8_t tid_rows_process = 0;
  uint8_t tid_columns_process = 0;

  // For rows:
  if (added_hole_rows[tid] == 1) // 1) The row where the hole is
  {
    tid_rows_process = 1;
  } else // 2) All rows that have an element whose column is  the row of the new
         // hole
  {
    const int *current_row = Positions_Column_Raw + Start_Matrix_Rows[tid];
    const int num_j = Start_Matrix_Rows[tid + 1] - Start_Matrix_Rows[tid];

    for (int idx_j = 0; idx_j < num_j; idx_j++) {
      const int val_j = current_row[idx_j];

      if (added_hole_rows[val_j] == 1) {
        tid_rows_process = 1;
      }
    }
  }

  // For the columns, those of the first loop, to each column of block A
  // 1) The column where the hole is
  // 2) All columns matching the row of the hole

  if ((added_hole_columns[tid] == 1) || (added_hole_rows[tid] == 1)) {
    tid_columns_process = 1;
  }

  rows_current_process[tid] = tid_rows_process;
  columns_current_process[tid] = tid_columns_process;
}
