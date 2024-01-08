#include "main.h"
#include "obtain_electrical_system.h"

template <typename T> Solver_Matrix_LDL<T>::~Solver_Matrix_LDL() {}

template <typename T>
Solver_Matrix_LDL<T>::Solver_Matrix_LDL(
    const Sparse_Matrix<std::complex<long double>> &A) {
  assert(A.m == A.n);
  number_unknowns = A.m;

  Sparse_Matrix<std::complex<long double>> L_temp;
  Sparse_Matrix<std::complex<long double>> D_temp;

  LDLt_no_permutations(A, L_temp, D_temp);

  L_matrix = L_temp.convert_to<T>();
  for (unsigned int i = 0; i < A.m; i++) {
    D_matrix.push_back((T)D_temp.D(i, i));
  }
}

template <typename T> void Solver_Matrix_LDL<T>::solve(T *input_output) {
  solve_substitution_LDL_decomposition(L_matrix, &(D_matrix[0]), input_output);
}

template class Solver_Matrix_LDL<C_LONG_DOUBLE>;
template class Solver_Matrix_LDL<std::complex<float>>;
template class Solver_Matrix_LDL<std::complex<double>>;
