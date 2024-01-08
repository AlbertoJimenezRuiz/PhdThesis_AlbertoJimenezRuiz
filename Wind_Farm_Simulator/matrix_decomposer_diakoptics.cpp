#include "main.h"
#include "obtain_electrical_system.h"

Matrix_Decomposer_Diakoptics::Matrix_Decomposer_Diakoptics(
    const Sparse_Matrix<std::complex<long double>> &A) {
  assert(A.m == A.n);

  number_unknowns = A.m;
  number_groups = 0;

  auto A_transpose = A.transpose();

  const int unknowns_common_row = 1;
  unsigned int max_row_position_until_now = 0;
  unsigned int max_column_position_until_now = 0;

  std::vector<unsigned int> limits_diagonal_subgroups;

  int lower_limit_subgroup = 0;

  PartF = A.submatrix(A.m - unknowns_common_row, A.m, A.n - unknowns_common_row,
                      A.n);

  for (unsigned int i = 0; i < A.m - unknowns_common_row; i++) {
    unsigned int max_row_position = 0;
    unsigned int min_row_position = A.m;

    unsigned int max_column_position = 0;
    unsigned int min_column_position = A.m;

    for (const auto &elem : A.data[i]) {
      unsigned int j = std::get<0>(elem);

      if (j >= A.m - unknowns_common_row)
        continue;

      min_row_position = MIN2(min_row_position, (unsigned int)j);
      max_row_position = MAX2(max_row_position, (unsigned int)j);
    }

    for (const auto &elem : A_transpose.data[i]) {
      unsigned int j = std::get<0>(elem);

      if (j >= A.m - unknowns_common_row)
        continue;

      min_column_position = MIN2(min_column_position, (unsigned int)j);
      max_column_position = MAX2(max_column_position, (unsigned int)j);
    }

    max_row_position_until_now =
        MAX2(max_row_position_until_now, max_row_position);
    max_column_position_until_now =
        MAX2(max_column_position_until_now, max_column_position);

    if (max_row_position_until_now <= i && max_column_position_until_now <= i) {
      limits_diagonal_subgroups.push_back(i);

      PartA.push_back(A.submatrix(lower_limit_subgroup, i + 1,
                                  lower_limit_subgroup, i + 1));
      PartD.push_back(A.submatrix(lower_limit_subgroup, i + 1,
                                  A.n - unknowns_common_row, A.n));
      PartE.push_back(A.submatrix(A.m - unknowns_common_row, A.m,
                                  lower_limit_subgroup, i + 1));

      limit_inferior.push_back(lower_limit_subgroup);
      limit_superior.push_back(i);

      lower_limit_subgroup = i + 1;

      number_groups++;
    }
  }
}
