#include "gpu_util.h"
#include "main.h"
#include "matrix_functions.h"
#include <cuda.h>

void create_csr_boolean_CUDA(vector<vector<int>> matr,
                             GPU_array_int *&d_Start_Matrix_Rows,
                             GPU_array_int *&d_Positions_Column_Raw) {
  int *Start_Matrix_Rows;
  int *Positions_Column_Raw;

  const size_t matrix_size = matr.size();

  create_csr_boolean(matr, Start_Matrix_Rows, Positions_Column_Raw);

  d_Start_Matrix_Rows = new GPU_array_int(matrix_size + 1, Start_Matrix_Rows);
  d_Positions_Column_Raw =
      new GPU_array_int(Start_Matrix_Rows[matrix_size], Positions_Column_Raw);

  free(Start_Matrix_Rows);
  free(Positions_Column_Raw);
}
