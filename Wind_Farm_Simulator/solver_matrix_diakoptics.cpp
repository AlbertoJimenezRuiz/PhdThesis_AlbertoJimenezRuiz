#include "main.h"
#include "obtain_electrical_system.h"

template <typename T> Solver_Matrix_Diakoptics<T>::~Solver_Matrix_Diakoptics() {
  for (const auto &punt : Matrices_D_dense) {
    delete[] punt;
  }

  for (const auto &punt : Matrices_E_dense) {
    delete[] punt;
  }

  for (const auto &punt : Matrices_A_inverse_dense) {
    delete[] punt;
  }
  delete[] AD_vector;
  delete[] temporary_ABT;
  delete[] temporary_vector;
}

template <typename T>
Solver_Matrix_Diakoptics<T>::Solver_Matrix_Diakoptics(
    const LONG_DOUBLE_C_Sparse_Matrix &A)
    : Matrix_Decomposer_Diakoptics(A) {
  assert(this->PartF.m == 1);
  assert(this->PartF.n == 1);

  std::vector<Sparse_Matrix<C_LONG_DOUBLE>>
      Decomposition_LU_Matrix_L_large_precision;
  std::vector<std::vector<C_LONG_DOUBLE>>
      Decomposition_LU_Diagonal_large_precision;

  std::vector<C_LONG_DOUBLE *> Matrices_D_dense_large_precision;
  std::vector<C_LONG_DOUBLE *> Matrices_E_dense_large_precision;

  for (int current_matrix = 0; current_matrix < this->number_groups;
       current_matrix++) {
    auto matA_temp = this->PartA[current_matrix];

    Matrices_A_inverse_dense.push_back(matA_temp.invert_gauss()
                                           .template convert_to<T>()
                                           .convert_dense_rowmajor());

    LONG_DOUBLE_C_Sparse_Matrix AL_temp;
    LONG_DOUBLE_C_Sparse_Matrix AD_temp;

    LDLt_no_permutations(matA_temp, AL_temp, AD_temp);

    Decomposition_LU_Matrix_L_large_precision.push_back(AL_temp);
    Decomposition_LU_Matrix_L.push_back(AL_temp.convert_to<T>());

    std::vector<T> AD_vector_vector;
    std::vector<C_LONG_DOUBLE> AD_vector_large_precision;

    for (unsigned int i = 0; i < AD_temp.m; i++) {
      AD_vector_vector.push_back(AD_temp.convert_to<T>().D(i, i));
      AD_vector_large_precision.push_back(AD_temp.D(i, i));
    }
    Decomposition_LU_Diagonal.push_back(AD_vector_vector);
    Decomposition_LU_Diagonal_large_precision.push_back(
        AD_vector_large_precision);

    Matrices_D_dense.push_back(this->PartD[current_matrix]
                                   .template convert_to<T>()
                                   .convert_dense_rowmajor());
    Matrices_E_dense.push_back(this->PartE[current_matrix]
                                   .template convert_to<T>()
                                   .convert_dense_rowmajor());

    Matrices_D_dense_large_precision.push_back(
        this->PartD[current_matrix].convert_dense_rowmajor());
    Matrices_E_dense_large_precision.push_back(
        this->PartE[current_matrix].convert_dense_rowmajor());

    group_sizes.push_back(AD_temp.m);
  }

  C_LONG_DOUBLE *AD_vector_large_precision =
      new C_LONG_DOUBLE[this->number_unknowns];

  int vector_index = 0;

  C_LONG_DOUBLE Temporary_value_1_constant_max_precision =
      -(this->PartF.D(0, 0));

  for (int current_matrix = 0; current_matrix < this->number_groups;
       current_matrix++) {
    C_LONG_DOUBLE *AD_current = &AD_vector_large_precision[vector_index];

    for (int i = 0; i < group_sizes[current_matrix]; i++) {
      AD_current[i] = Matrices_D_dense_large_precision[current_matrix][i];
    }
    solve_substitution_LDL_decomposition(
        Decomposition_LU_Matrix_L_large_precision[current_matrix],
        &(Decomposition_LU_Diagonal_large_precision[current_matrix][0]),
        AD_current);

    for (int i = 0; i < group_sizes[current_matrix]; i++) {
      Temporary_value_1_constant_max_precision +=
          Matrices_E_dense_large_precision[current_matrix][i] * AD_current[i];
    }

    pointers_start.push_back(vector_index);
    vector_index += group_sizes[current_matrix];
  }

  AD_vector = new T[this->number_unknowns];
  for (int i = 0; i < this->number_unknowns; i++) {
    AD_vector[i] = AD_vector_large_precision[i];
  }
  delete[] AD_vector_large_precision;

  temporary_ABT = new T[this->number_unknowns];
  temporary_vector = new T[this->number_unknowns];

  temporary_value_1_constant = Temporary_value_1_constant_max_precision;

  for (const auto &punt : Matrices_D_dense_large_precision) {
    delete[] punt;
  }

  for (const auto &punt : Matrices_E_dense_large_precision) {
    delete[] punt;
  }
}

template <typename T> void Solver_Matrix_Diakoptics<T>::solve(T *input_output) {
  for (int i = 0; i < this->number_unknowns; i++) {
    temporary_ABT[i] = input_output[i];
  }

  for (int current_matrix = 0; current_matrix < this->number_groups;
       current_matrix++) {
    T *vector_abt_temp = temporary_ABT + pointers_start[current_matrix];
    T *temporary_vector_2 = temporary_vector + pointers_start[current_matrix];

    for (int i = 0; i < group_sizes[current_matrix]; i++) {
      T temporary = T(0);
      T *current_row = Matrices_A_inverse_dense[current_matrix] +
                       group_sizes[current_matrix] * i;

      for (int j = 0; j < group_sizes[current_matrix]; j++) {
        temporary += current_row[j] * vector_abt_temp[j];
      }
      temporary_vector_2[i] = temporary;
    }

    for (int i = 0; i < group_sizes[current_matrix]; i++) {
      vector_abt_temp[i] = temporary_vector_2[i];
    }
  }

  T value_temporary_3 = -input_output[this->number_unknowns - 1];

  for (int current_matrix = 0; current_matrix < this->number_groups;
       current_matrix++) {
    T *temporary_ABT_vector = temporary_ABT + pointers_start[current_matrix];
    for (int i = 0; i < group_sizes[current_matrix]; i++) {
      value_temporary_3 +=
          Matrices_E_dense[current_matrix][i] * temporary_ABT_vector[i];
    }
  }

  auto value_union = value_temporary_3 / temporary_value_1_constant;

  for (int i = 0; i < this->number_unknowns - 1; i++) {
    input_output[i] = temporary_ABT[i] - AD_vector[i] * value_union;
  }

  input_output[this->number_unknowns - 1] = value_union;
}

template class Solver_Matrix_Diakoptics<std::complex<float>>;
template class Solver_Matrix_Diakoptics<std::complex<double>>;
template class Solver_Matrix_Diakoptics<std::complex<long double>>;
