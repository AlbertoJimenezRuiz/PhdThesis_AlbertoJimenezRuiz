#include "main.h"
#include "matrix_functions.h"
#include "sparse_matrix.h"

extern "C" {
#include "amd.h"
}

// Algorithm 837: "An Approximate Minimum Degree Ordering Algorithm"
void algorithm_amd_cpp(Sparse_Matrix<FLOATING_TYPE> &matrix,
                       std::vector<int> &interchanges_position_row_columns) {

  auto matrix_boolean_equivalent =
      matrix.convert_to_boolean_equivalent_rowmajor();

  int n = matrix.n;

  int *Ap;
  int *Ai;
  int *P;
  int *Pinv;

  create_csr_boolean(matrix_boolean_equivalent, Ap, Ai);

  P = (int *)malloc(sizeof(int) * n);
  Pinv = (int *)malloc(sizeof(int) * n);

  assert(matrix.m == matrix.n);

  int j, k, result;

  result = amd_order(n, Ap, Ai, P, NULL, NULL);
  assert(result == AMD_OK);

  for (k = 0; k < n; k++) {
    j = P[k];
    Pinv[j] = k;
  }

  for (int i = 0; i < n; i++) {
    interchanges_position_row_columns[i] = P[i];
  }

  matrix = matrix.interchange_row_columns(Pinv);

  free(Ap);
  free(Ai);
  free(P);
  free(Pinv);
}
