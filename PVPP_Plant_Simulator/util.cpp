
#include "main.h"
#include "sparse_matrix.h"
#include <iomanip>

#include <cuda.h>
#include <cuda_runtime.h>

int random_int(int upper) { return rand() % (upper); }

void __attribute__((format(printf, 1, 0))) abortmsg(const char *format, ...) {
  va_list args;
  va_start(args, format);
  vfprintf(stderr, format, args);
  va_end(args);
  abort();
}

Time_Measurer::Time_Measurer() {
  start = std::chrono::high_resolution_clock::now();
}

long int Time_Measurer::measure(const char *msg) {
  auto now = std::chrono::high_resolution_clock::now();
  auto diff = chrono::duration_cast<chrono::nanoseconds>(now - start).count();

  cout << setprecision(3) << fixed;
  if (msg) {
    cout << "Time:" << setw(50) << msg << " " << setw(15)
         << (double)diff / 1000000 << " ms" << endl;
  } else {
    cout << " Time: " << setw(50) << (double)diff / 1000000 << " ms" << endl;
  }
  start = std::chrono::high_resolution_clock::now();
  return (long int)(diff / 1000);
}

void create_csr_boolean(const vector<vector<int>> &data,
                        int *&Start_Matrix_Rows, int *&Positions_Column_Raw) {
  int matrix_size = (int)data.size();

  size_t nnz = 0;
  for (int i = 0; i < matrix_size; i++) {
    nnz += data[i].size();
  }

  Start_Matrix_Rows = (int *)malloc(sizeof(int) * (matrix_size + 1));
  Positions_Column_Raw = (int *)malloc(sizeof(int) * nnz);

  int number_elements_matrix_temp = 0;
  Start_Matrix_Rows[0] = 0;
  for (int i = 0; i < matrix_size; i++) {
    for (size_t j = 0; j < data[i].size(); j++) {
      Positions_Column_Raw[number_elements_matrix_temp] = data[i][j];
      number_elements_matrix_temp++;
    }
    Start_Matrix_Rows[i + 1] = number_elements_matrix_temp;
  }
}

void apply_permutation(vector<FLOATING_TYPE> &vec, const int *perm) {
  vector<FLOATING_TYPE> output = vec;

  for (unsigned int i = 0; i < vec.size(); i++) {
    output[perm[i]] = vec[i];
  }
  vec = output;
}

vector<int> invert_permutation(const vector<int> &perm) {
  vector<int> res;
  res.resize(perm.size());
  for (unsigned int idx = 0; idx < perm.size(); idx++) {
    res[perm[idx]] = idx;
  }
  return res;
}
