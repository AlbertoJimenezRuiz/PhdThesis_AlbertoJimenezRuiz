#include "main.h"
#include "sparse_matrix.h"

void fill_fillins(const vector<vector<int>> &fillin_dependencies,
                  Sparse_Matrix<FLOATING_TYPE> &A, const FLOATING_TYPE hole) {
  auto n = A.n;
  for (unsigned int i = 0; i < n; i++) {
    for (const auto &value : fillin_dependencies[i]) {
      if (!A.exists(i, value)) {
        A.write(i, value, hole);
      }
    }
  }
}
