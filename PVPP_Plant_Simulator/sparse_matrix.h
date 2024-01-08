#ifndef SPARSE_MATRIX_H
#define SPARSE_MATRIX_H

#include <map>

template <typename T> class Sparse_Matrix {
public:
  std::vector<std::map<int, T>> data;

  Sparse_Matrix(unsigned int a_m, unsigned int a_n) {

    create_matrix_internal(a_m, a_n);
  }

  Sparse_Matrix() : Sparse_Matrix(0, 0) {}

  void create_matrix_internal(unsigned int a_m, unsigned int a_n) {
    m = a_m;
    n = a_n;
    data.clear();
    data.resize(m);
  }

  void write(const int dx, const int dy, const T val) {
    if ((dx < 0) || (dx >= (int)m) || (dy < 0) || (dy >= (int)n)) {
      abortmsg("ERROR Reading %d,%d  Dimensions %d,%d    \n", dx, dy, m, n);
    }
    data[dx][dy] = val;
  }

  const T D(const int dx, const int dy) const {

    if ((dx < 0) || (dx >= (int)m) || (dy < 0) || (dy >= (int)n)) {
      abortmsg("ERROR Reading %d,%d  Dimensions %d,%d    \n", dx, dy, m, n);
    }

    const auto &data_search = data[dx];

    auto it = data_search.find(dy);
    if (it != data_search.end())
      return get<1>(*it);
    else
      return 0;
  }

  int nnz() const {
    int res = 0;

    for (const auto &x : data) {
      res += (int)x.size();
    }

    return res;
  }

  void write_add(int n1, int n2, T val) {
    return write(n1, n2, D(n1, n2) + val);
  }

  vector<T> convert_to_std_vector() const {
    vector<T> res(m * n);
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

  vector<vector<int>> convert_to_boolean_equivalent_rowmajor() const {
    vector<vector<int>> res(m);

    for (unsigned int i = 0; i < m; i++) {
      auto &current_row = res[i];
      for (const auto &elem : data[i]) {
        current_row.push_back(std::get<0>(elem));
      }
    }

    return res;
  }

  bool exists(const int dx, const int dy) const {
    if ((dx < 0) || (dx >= (int)m) || (dy < 0) || (dy >= (int)n)) {
      abortmsg("ERROR Reading %d,%d  Dimensions %d,%d    \n", dx, dy, m, n);
    }
    const auto &data_find = data[dx];

    auto it = data_find.find(dy);
    return (it != data_find.end());
  }

  Sparse_Matrix<T> transpose() const {
    Sparse_Matrix res(n, m);

    for (unsigned int i = 0; i < m; i++) {
      for (const auto &elem : data[i]) {
        int j = std::get<0>(elem);
        T val_D = std::get<1>(elem);

        res.data[j].insert({i, val_D});
      }
    }

    return res;
  }

  void exchange_rows(int f1, int f2) { std::swap(data[f1], data[f2]); }

  Sparse_Matrix<T> interchange_row_columns(int *new_order) const {
    Sparse_Matrix<FLOATING_TYPE> new_matrix = (*this);

    for (auto &v : new_matrix.data) {
      v.clear();
    }

    for (unsigned int i = 0; i < n; i++) {
      std::map<int, FLOATING_TYPE> new_row;

      for (const auto &elem : data[i]) {
        new_row[new_order[elem.first]] = elem.second;
      }

      new_matrix.data[new_order[i]] = new_row;
    }
    return new_matrix;
  }

  Sparse_Matrix<T> exchange_rows(int *new_order) const {
    Sparse_Matrix<FLOATING_TYPE> new_matrix = (*this);

    for (auto &v : new_matrix.data) {
      v.clear();
    }

    for (unsigned int i = 0; i < n; i++) {
      new_matrix.data[new_order[i]] = data[i];
    }
    return new_matrix;
  }

  unsigned int m, n;
};

#endif
