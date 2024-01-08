#include "main.h"
#include "sparse_matrix.h"

template <typename T>
void LDLt_no_permutations(
    // Inputs:
    Sparse_Matrix<T> A, // Changes in function. Passed by value.
    // Outputs:
    // unit lower triangular L
    Sparse_Matrix<T> &L,
    // real block diagonal D
    Sparse_Matrix<T> &D) {
  // tanken for granted that matrix is invertible.

  const int n = A.n;

  D = Sparse_Matrix<T>::identity(n);
  L = Sparse_Matrix<T>::identity(n);

  if (n == 1) {
    D = A;
    return;
  }
  for (int k = 0; k < n - 1; k++) {
    T diagonal_value = A.D(k, k);

    D.write(k, k, diagonal_value);
    for (int cont = k + 1; cont < n; cont++) {
      auto val_temp = A.D(cont, k) / diagonal_value;
      if (abs(val_temp) < EPS)
        continue;

      A.write(cont, k, val_temp);
      L.write(cont, k, val_temp);
    }

    for (int cont1 = k + 1; cont1 < n; cont1++) {
      auto Temp_A_D = A.D(cont1, k);
      if (abs(Temp_A_D) < EPS)
        continue;

      for (const auto &elem2 : A.data[k]) {
        int cont2 = std::get<0>(elem2);
        T val_temp_2 = std::get<1>(elem2);
        if (cont2 < k + 1)
          continue;

        A.write_add(cont1, cont2, -Temp_A_D * val_temp_2);
      }
    }
  }

  D.write(n - 1, n - 1, A.D(n - 1, n - 1));
}

template void LDLt_no_permutations(Sparse_Matrix<std::complex<float>> A,
                                   Sparse_Matrix<std::complex<float>> &L,
                                   Sparse_Matrix<std::complex<float>> &D);
template void LDLt_no_permutations(Sparse_Matrix<std::complex<double>> A,
                                   Sparse_Matrix<std::complex<double>> &L,
                                   Sparse_Matrix<std::complex<double>> &D);
template void LDLt_no_permutations(Sparse_Matrix<std::complex<long double>> A,
                                   Sparse_Matrix<std::complex<long double>> &L,
                                   Sparse_Matrix<std::complex<long double>> &D);
template void LDLt_no_permutations(Sparse_Matrix<float> A,
                                   Sparse_Matrix<float> &L,
                                   Sparse_Matrix<float> &D);
template void LDLt_no_permutations(Sparse_Matrix<double> A,
                                   Sparse_Matrix<double> &L,
                                   Sparse_Matrix<double> &D);
template void LDLt_no_permutations(Sparse_Matrix<long double> A,
                                   Sparse_Matrix<long double> &L,
                                   Sparse_Matrix<long double> &D);

template <typename T>
Sparse_Matrix<T> invert_lower_triangular_matrix(const Sparse_Matrix<T> &A) {
  Sparse_Matrix<T> L = Sparse_Matrix<T>::identity(A.m);

  for (unsigned int j = 0; j < A.n; j++) {
    for (unsigned int i = j + 1; i < A.n; i++) {
      T res = -A.D(i, j);

      for (const auto &elem2 : A.data[i]) {
        unsigned int c = std::get<0>(elem2);
        T val_temp_2 = std::get<1>(elem2);

        if (c < j + 1)
          continue;
        if (c >= i)
          continue;

        res -= val_temp_2 * L.D(c, j);
      }
      L.write(i, j, res);
    }
  }

  return L;
}

template Sparse_Matrix<std::complex<float>>
invert_lower_triangular_matrix(const Sparse_Matrix<std::complex<float>> &A);
template Sparse_Matrix<std::complex<double>>
invert_lower_triangular_matrix(const Sparse_Matrix<std::complex<double>> &A);
template Sparse_Matrix<std::complex<long double>>
invert_lower_triangular_matrix(
    const Sparse_Matrix<std::complex<long double>> &A);
template Sparse_Matrix<long double>
invert_lower_triangular_matrix(const Sparse_Matrix<long double> &A);

template <typename T>
Sparse_Matrix<T> invert_diagonal_matrix(const Sparse_Matrix<T> &A) {
  Sparse_Matrix<T> res = Sparse_Matrix<T>::identity(A.m);

  for (unsigned int j = 0; j < A.n; j++) {
    res.write(j, j, T(1) / A.D(j, j));
  }

  return res;
}

template Sparse_Matrix<std::complex<float>>
invert_diagonal_matrix(const Sparse_Matrix<std::complex<float>> &A);
template Sparse_Matrix<std::complex<double>>
invert_diagonal_matrix(const Sparse_Matrix<std::complex<double>> &A);
template Sparse_Matrix<std::complex<long double>>
invert_diagonal_matrix(const Sparse_Matrix<std::complex<long double>> &A);
template Sparse_Matrix<long double>
invert_diagonal_matrix(const Sparse_Matrix<long double> &A);

template <typename T>
void solve_substitution_LDL_decomposition(const Sparse_Matrix<T> &L, const T *D,
                                          T *input_output) {
  assert(L.m == L.n);
  const int number_values = L.m;

  // L Substitution
  for (int i = 0; i < number_values; i++) {
    for (const auto &tuple_values : L.data[i]) {
      auto j = std::get<0>(tuple_values);
      auto val = std::get<1>(tuple_values);

      if (j < i) {
        input_output[i] -= val * input_output[j];
      }
    }
  }

  // D Substitution
  for (int j = 0; j < number_values; j++) {
    input_output[j] /= D[j];
  }

  // Transposed L Substitution
  for (int j = number_values - 1; j > 0; j--) {
    auto &temp = input_output[j];
    for (const auto &tuple_values : L.data[j]) {
      auto i = std::get<0>(tuple_values);
      auto val = std::get<1>(tuple_values);

      if (i < j) {
        input_output[i] -= val * temp;
      }
    }
  }
}
template void solve_substitution_LDL_decomposition<float>(
    const Sparse_Matrix<float> &L, const float *D, float *input_output);
template void solve_substitution_LDL_decomposition<double>(
    const Sparse_Matrix<double> &L, const double *D, double *input_output);
template void solve_substitution_LDL_decomposition<std::complex<float>>(
    const Sparse_Matrix<std::complex<float>> &L, const std::complex<float> *D,
    std::complex<float> *input_output);
template void solve_substitution_LDL_decomposition<std::complex<double>>(
    const Sparse_Matrix<std::complex<double>> &L, const std::complex<double> *D,
    std::complex<double> *input_output);
template void solve_substitution_LDL_decomposition<std::complex<long double>>(
    const Sparse_Matrix<std::complex<long double>> &L,
    const std::complex<long double> *D,
    std::complex<long double> *input_output);
