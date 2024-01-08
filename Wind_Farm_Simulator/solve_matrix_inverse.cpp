#include "main.h"
#include "obtain_electrical_system.h"

template <typename T> Solver_Matrix_Inverse<T>::~Solver_Matrix_Inverse() {
  delete[] temporary_vector;
  delete[] A_inverse;
}

template <typename T>
Solver_Matrix_Inverse<T>::Solver_Matrix_Inverse(
    const Sparse_Matrix<C_LONG_DOUBLE> &A) {
  assert(A.m == A.n);
  number_unknowns = A.m;

  temporary_vector = new T[number_unknowns];

  Sparse_Matrix<C_LONG_DOUBLE> L_temp;
  Sparse_Matrix<C_LONG_DOUBLE> D_temp;

  LDLt_no_permutations(A, L_temp, D_temp);

  auto L_inv = invert_lower_triangular_matrix(L_temp);
  auto A_inv = (L_inv.transpose() * invert_diagonal_matrix(D_temp) * L_inv);

  A_inverse = A_inv.convert_to<T>().convert_dense_rowmajor();
}

template <typename T> void Solver_Matrix_Inverse<T>::solve(T *input_output) {
  for (int i = 0; i < this->number_unknowns; i++) {
    T temporary = T(0);
    T *row = A_inverse + this->number_unknowns * i;

    for (int j = 0; j < this->number_unknowns; j++) {
      temporary += row[j] * input_output[j];
    }
    temporary_vector[i] = temporary;
  }

  for (int i = 0; i < this->number_unknowns; i++) {
    input_output[i] = temporary_vector[i];
  }
}

template class Solver_Matrix_Inverse<std::complex<float>>;
template class Solver_Matrix_Inverse<std::complex<double>>;
template class Solver_Matrix_Inverse<std::complex<long double>>;
