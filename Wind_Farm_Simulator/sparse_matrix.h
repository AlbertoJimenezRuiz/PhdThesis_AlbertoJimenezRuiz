#ifndef SPARSE_MATRIX_H
#define SPARSE_MATRIX_H

#include "main.h"

#include <assert.h>
#include <complex>
#include <iostream>
#include <list>
#include <map>
#include <string.h>
#include <tuple>
#include <vector>

#define EPS 1e-15

template <typename T> class Sparse_Matrix {
public:
  std::vector<std::list<std::tuple<int, T>>> data;

  void create_matrix_internal(unsigned int a_m, unsigned int a_n) {
    m = a_m;
    n = a_n;
    data.clear();
    data.reserve(m);
    for (unsigned int i = 0; i < m; i++) {
      data.push_back(std::list<std::tuple<int, T>>());
    }
  }
  void destroy_matrix_internal() { data.clear(); }

  void write_add(int n1, int n2, T val) {
    return write(n1, n2, D(n1, n2) + val);
  }

  const T D(const int dx, const int dy) const {

    if ((dx < 0) || (dx >= (int)m) || (dy < 0) || (dy >= (int)n)) {
      abort_msg("Error Reading %d,%d.  Dimensions %d,%d.\n", dx, dy, m, n);
    }

    const auto &elem_find = data[dx];
    for (const auto &elem : elem_find) {
      if ((std::get<0>(elem) == dy)) {
        return std::get<1>(elem);
      }
    }
    return 0;
  }

  bool exists(const int dx, const int dy) const {

    if ((dx < 0) || (dx >= (int)m) || (dy < 0) || (dy >= (int)n)) {
      abort_msg("Error Reading %d,%d.  Dimensions %d,%d.\n", dx, dy, m, n);
    }

    const auto &elem_find = data[dx];
    for (const auto &elem : elem_find) {
      if ((std::get<0>(elem) == dy)) {
        return true;
      }
    }

    return false;
  }

  int nnz() const {
    int number_nonzeros = 0;

    for (auto &fila : data) {
      number_nonzeros += (int)fila.size();
    }
    return number_nonzeros;
  }

  void write(const int dx, const int dy, const T val) {
    if ((dx < 0) || (dx >= (int)m) || (dy < 0) || (dy >= (int)n)) {
      abort_msg("Error Reading %d,%d.  Dimensions %d,%d.\n", dx, dy, m, n);
    }

    if (abs(val) < EPS) {
      auto &elem_find = data[dx];

      for (auto elem = elem_find.begin(); elem != elem_find.end(); ++elem) {
        if ((std::get<0>(*elem) == dy)) {
          elem_find.erase(elem);
          return;
        }
      }
    } else {
      auto &elem_find = data[dx];
      for (auto &elem : elem_find) {
        if ((std::get<0>(elem) == dy)) {
          std::get<1>(elem) = val;
          return;
        }
      }
      elem_find.push_back({dy, val});
    }
  }

  Sparse_Matrix(unsigned int a_m, unsigned int a_n) {
    create_matrix_internal(a_m, a_n);
  }

  Sparse_Matrix(unsigned int a_m) : Sparse_Matrix(a_m, 1) {}

  Sparse_Matrix() { create_matrix_internal(0, 0); }

  ~Sparse_Matrix() { destroy_matrix_internal(); }

  // Copy constructor
  Sparse_Matrix(const Sparse_Matrix<T> &src) : Sparse_Matrix(src.m, src.n) {
    copy_data(src);
  }

  Sparse_Matrix &operator=(const Sparse_Matrix &src) {

    destroy_matrix_internal();
    create_matrix_internal(src.m, src.n);
    copy_data(src);

    return *this;
  }

  Sparse_Matrix operator+(Sparse_Matrix const &obj) const {
    Sparse_Matrix res(m, n);
    if ((obj.m != m) || (obj.n != n)) {
      abort_msg("ERROR. Matrix dimensions mismatch.\n");
    }

    for (unsigned int i = 0; i < m; i++) {
      std::map<int, T> temp_map;

      for (const auto &elem : data[i]) {
        temp_map[std::get<0>(elem)] = std::get<1>(elem);
      }

      for (const auto &elem : obj.data[i]) {
        temp_map[std::get<0>(elem)] += std::get<1>(elem);
      }

      for (const auto &elem : temp_map) {
        res.data[i].push_back({elem.first, elem.second});
      }
    }

    return res;
  }

  Sparse_Matrix operator-(Sparse_Matrix const &obj) const {
    Sparse_Matrix res(m, n);
    if ((obj.m != m) || (obj.n != n)) {
      abort_msg("ERROR. Matrix dimensions mismatch.\n");
    }

    for (unsigned int i = 0; i < m; i++) {

      std::map<int, T> temp_map;

      for (const auto &elem : data[i]) {
        temp_map[std::get<0>(elem)] = std::get<1>(elem);
      }

      for (const auto &elem : obj.data[i]) {
        temp_map[std::get<0>(elem)] -= std::get<1>(elem);
      }

      for (const auto &elem : temp_map) {
        res.data[i].push_back({elem.first, elem.second});
      }
    }

    return res;
  }

  template <typename T2>
  Sparse_Matrix<std::complex<T2>> convert_to_complex() const {
    Sparse_Matrix<std::complex<T2>> temp(this->m, this->n);

    for (unsigned int i = 0; i < m; i++) {
      for (unsigned int j = 0; j < n; j++) {
        temp.write(i, j, std::complex<T2>((T2)D(i, j), 0));
      }
    }
    return temp;
  }

  template <typename T2> Sparse_Matrix<T2> convert_to() const {
    Sparse_Matrix<T2> temp(this->m, this->n);
    for (unsigned int i = 0; i < m; i++) {
      for (unsigned int j = 0; j < n; j++) {
        temp.write(i, j, T2(D(i, j)));
      }
    }
    return temp;
  }

  T *convert_dense_rowmajor() {
    T *res = new T[m * n];
    for (unsigned int i = 0; i < m * n; i++) {
      res[i] = T(0);
    }

    for (unsigned int i = 0; i < m; i++) {
      for (const auto &elem : data[i]) {
        int j = std::get<0>(elem);
        T val_D = std::get<1>(elem);

        res[i * n + j] = val_D;
      }
    }

    return res;
  }

  Sparse_Matrix<T> submatrix(int row_min, int row_max, int column_min,
                             int column_max) const {
    Sparse_Matrix<T> res(row_max - row_min, column_max - column_min);

    for (int i = row_min; i < row_max; i++) {
      for (const auto &elem : data[i]) {
        int j = std::get<0>(elem);
        auto temporary_value = std::get<1>(elem);

        if (j < column_min)
          continue;

        if (j >= column_max)
          continue;

        res.write(i - row_min, j - column_min, temporary_value);
      }
    }

    return res;
  }

  void exchange_rows_and_columns(int f1, int f2) {
    std::swap(data[f1], data[f2]);

    for (auto &row : data) {
      for (auto &elem : row) {
        auto xtemp = std::get<0>(elem);

        if (xtemp == f1)
          xtemp = f2;
        else if (xtemp == f2)
          xtemp = f1;

        std::get<0>(elem) = xtemp;
      }
    }
  }

  Sparse_Matrix operator*(const Sparse_Matrix &obj) const {
    if (n != obj.m) {
      abort_msg("ERROR. Matrix dimensions mismatch.\n");
    }

    Sparse_Matrix<T> res(m, obj.n);

    for (unsigned int i = 0; i < res.m; i++) {
      for (unsigned int j = 0; j < res.n; j++) {
        T r_temp = 0.0;

        for (const auto &elem : data[i]) {
          int z = std::get<0>(elem);
          T val_D = std::get<1>(elem);
          r_temp += val_D * obj.D(z, j);
        }

        if (std::abs(r_temp) > EPS) {
          res.data[i].push_back({j, r_temp});
        }
      }
    }
    return res;
  }

  Sparse_Matrix operator*(const T coeff) const {

    Sparse_Matrix res(m, n);

    for (unsigned int i = 0; i < m; i++) {
      const auto &row_matD = data[i];
      for (const auto &elem : row_matD) {
        int j = std::get<0>(elem);
        T val_D = std::get<1>(elem);

        if (abs(val_D) > EPS) {
          res.data[i].push_back({j, val_D * coeff});
        }
      }
    }
    return res;
  }

  static Sparse_Matrix<T> identity(int arg_m) {
    Sparse_Matrix<T> res(arg_m, arg_m);

    for (int i = 0; i < arg_m; i++) {
      res.write(i, i, 1.0);
    }

    return res;
  }

  Sparse_Matrix<T> transpose() const {
    Sparse_Matrix<T> res(n, m);

    for (unsigned int i = 0; i < m; i++) {
      for (const auto &elem : data[i]) {
        res.data[std::get<0>(elem)].push_back({i, std::get<1>(elem)});
      }
    }
    return res;
  }

  void copy_data(const Sparse_Matrix<T> &src) {
    for (unsigned int i = 0; i < m; i++) {
      const auto &row_matrix_src = src.data[i];
      for (const auto &elem : row_matrix_src) {
        write(i, std::get<0>(elem), std::get<1>(elem));
      }
    }
  }

  Sparse_Matrix<T> invert_gauss() const {
    assert(n == m);

    Sparse_Matrix<T> temp(n, n * 2);

    for (unsigned int i = 0; i < n; i++) {
      for (const auto &elem : data[i]) {
        temp.write(i, std::get<0>(elem), std::get<1>(elem));
      }
      temp.write(i, i + n, T(1));
    }

    for (unsigned int variable = 0; variable < n; variable++) {
      unsigned int row_largest_number = variable;
      auto largest_number = temp.D(row_largest_number, variable);

      for (unsigned int potential_row = variable; potential_row < n;
           potential_row++) {
        auto potential_larger_number = temp.D(potential_row, variable);
        if (std::abs(potential_larger_number) > std::abs(largest_number)) {
          largest_number = potential_larger_number;
          row_largest_number = potential_row;
        }
      }

      std::swap(temp.data[variable], temp.data[row_largest_number]);
      assert(std::abs(largest_number) > 1e-5);
      for (auto &elem : temp.data[variable]) {
        std::get<1>(elem) /= largest_number;
      }

      for (unsigned int rows_substitute = 0; rows_substitute < n;
           rows_substitute++) {
        if (rows_substitute == variable)
          continue;

        if (!temp.exists(rows_substitute, variable))
          continue;

        auto dominating_value = temp.D(rows_substitute, variable);

        for (const auto &elem : temp.data[variable]) {
          temp.write_add(rows_substitute, std::get<0>(elem),
                         -std::get<1>(elem) * dominating_value);
        }
      }
    }

    Sparse_Matrix<T> res(n, n);
    for (unsigned int i = 0; i < n; i++) {
      for (const auto &elem : temp.data[i]) {
        unsigned int j = std::get<0>(elem);
        if (j < n)
          continue;
        res.write(i, j - n, std::get<1>(elem));
      }
    }

    return res;
  }
  unsigned int m, n;
};

typedef Sparse_Matrix<C_FLOATING_TYPE> C_Sparse_Matrix;
typedef Sparse_Matrix<FLOATING_TYPE> R_Sparse_Matrix;

typedef Sparse_Matrix<std::complex<long double>> LONG_DOUBLE_C_Sparse_Matrix;
typedef Sparse_Matrix<long double> LONG_DOUBLE_R_Sparse_Matrix;

// LDLt solvers

template <typename T>
void LDLt_no_permutations(Sparse_Matrix<T> A, Sparse_Matrix<T> &L,
                          Sparse_Matrix<T> &D);

template <typename T>
Sparse_Matrix<T> invert_lower_triangular_matrix(const Sparse_Matrix<T> &A);

template <typename T>
Sparse_Matrix<T> invert_diagonal_matrix(const Sparse_Matrix<T> &A);

template <typename T>
void solve_substitution_LDL_decomposition(const Sparse_Matrix<T> &L, const T *D,
                                          T *input_output);

#endif
