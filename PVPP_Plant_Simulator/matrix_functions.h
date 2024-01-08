
#include "sparse_matrix.h"

void create_CSR_CSC_matrices(Sparse_Matrix<FLOATING_TYPE> &mat,

                             vector<int> &CSR_start_rows,
                             vector<int> &CSR_position_columns,
                             vector<FLOATING_TYPE> &CSR_values,

                             vector<int> &CSC_start_columns,
                             vector<int> &CSC_position_rows,
                             vector<FLOATING_TYPE> &CSC_values,

                             vector<int> &CSR_corresponding_value_in_CSC,
                             vector<int> &CSC_corresponding_value_in_CSR,

                             vector<int> &CSR_start_from_diagonal,
                             vector<int> &CSC_start_from_diagonal);

void fill_values_CSR_only(const Sparse_Matrix<FLOATING_TYPE> &mat,
                          const vector<int> &start_rows,
                          const vector<int> &position_columns,
                          vector<FLOATING_TYPE> &values);

void fill_values_CSC_only(const Sparse_Matrix<FLOATING_TYPE> &mat,
                          const vector<int> &start_columns,
                          const vector<int> &position_rows,
                          vector<FLOATING_TYPE> &values);

vector<vector<int>>
obtain_fillin_matrix_CUDA(Sparse_Matrix<FLOATING_TYPE> &original_matrix);

void fill_fillins(const vector<vector<int>> &fillin_dependencies,
                  Sparse_Matrix<FLOATING_TYPE> &A,
                  const FLOATING_TYPE hole = 0.0);
vector<set<int>>
get_dependencies_GLU_3_0(int n, const vector<vector<int>> &fillin_dependencies);

vector<vector<list<int>>> convert_dependence_list_in_chains_of_columns(
    const vector<set<int>> &list_column_dependences);

void convert_chains_columns_levels_into_vector(
    const vector<vector<list<int>>> &levels, vector<int> &start_levels_unified,
    vector<int> &start_chains_columns_unified,
    vector<int> &chains_columns_unified);

void create_csr_boolean(const vector<vector<int>> &data,
                        int *&Start_Matrix_Rows, int *&Positions_Column_Raw);

Sparse_Matrix<FLOATING_TYPE>
reorder_matrix(const Sparse_Matrix<FLOATING_TYPE> &mat,
               std::vector<int> &interchanges_position_row_columns);
Sparse_Matrix<FLOATING_TYPE>
reorder_matrix(const Sparse_Matrix<FLOATING_TYPE> &mat,
               std::vector<int> &interchanges_position_row_columns,
               std::vector<int> &interchanges_position_row_columns_only_rows);

void apply_permutation(vector<FLOATING_TYPE> &vec, const int *perm);
vector<int> invert_permutation(const vector<int> &perm);

vector<int>
get_max_numbers_per_column(const vector<vector<list<int>>> &levels_with_strings,
                           const vector<vector<int>> &nonzero_values);

void Substitute_LU_Vector_b_CSR_CPU(const unsigned int m,
                                    const int *CSR_start_rows,
                                    const int *CSR_position_columns,
                                    const FLOATING_TYPE *CSR_values,
                                    const int *CSR_start_from_diagonal,
                                    FLOATING_TYPE *b);

void Substitute_LU_Vector_b_CSC_CPU(const unsigned int n,
                                    const int *CSC_start_columns,
                                    const int *CSC_position_rows,
                                    const FLOATING_TYPE *CSC_values,
                                    const int *CSC_start_from_diagonal,
                                    FLOATING_TYPE *b);

vector<set<int>>
get_dependencies_matrix_L(int n,
                          const vector<vector<int>> &fillin_dependencies);

vector<set<int>>
get_dependencies_matrix_U(int n,
                          const vector<vector<int>> &fillin_dependencies);

void Process_Column_L_CSC_CPU(const int j, const int *CSC_start_columns,
                              const int *CSC_position_rows,
                              const FLOATING_TYPE *CSC_values,
                              const int *CSC_start_from_diagonal,
                              FLOATING_TYPE *b);

void Process_Column_U_CSC_CPU(const int j, const int *CSC_start_columns,
                              const int *CSC_position_rows,
                              const FLOATING_TYPE *CSC_values,
                              const int *CSC_start_from_diagonal,
                              FLOATING_TYPE *b);

void factorize_all_chains_of_columns_in_level_GLU_CSR_CPU(
    const int *start_levels_unified, const int current_level,
    const int *start_chains_columns_unified, const int *chains_columns_unified,

    const int *CSR_start_rows, const int *CSR_position_columns,
    FLOATING_TYPE *CSR_values,

    const int *CSC_start_columns, const int *CSC_position_rows,
    const int *CSC_corresponding_value_in_CSR,

    const int *CSR_start_from_diagonal, const int *CSC_start_from_diagonal);

void factorize_all_chains_of_columns_in_level_GLU_CSC_CPU(
    const int *start_levels_unified, const int current_level,
    const int *start_chains_columns_unified, const int *chains_columns_unified,

    const int *CSR_start_rows, const int *CSR_position_columns,
    const int *CSR_corresponding_value_in_CSC,

    const int *CSC_start_columns, const int *CSC_position_rows,

    FLOATING_TYPE *CSC_values,

    const int *CSR_start_from_diagonal, const int *CSC_start_from_diagonal);

void GLU_CSR_CPU_sequential(const int n,

                            const int *CSR_start_rows,
                            const int *CSR_position_columns,
                            FLOATING_TYPE *CSR_values,

                            const int *CSC_start_columns,
                            const int *CSC_position_rows,
                            const int *CSC_corresponding_value_in_CSR,

                            const int *CSR_start_from_diagonal,
                            const int *CSC_start_from_diagonal);

void GLU_CSC_CPU_sequential(const int n,

                            const int *CSR_start_rows,
                            const int *CSR_position_columns,
                            const int *CSR_corresponding_value_in_CSC,

                            const int *CSC_start_columns,
                            const int *CSC_position_rows,
                            FLOATING_TYPE *CSC_values,

                            const int *CSR_start_from_diagonal,
                            const int *CSC_start_from_diagonal);
