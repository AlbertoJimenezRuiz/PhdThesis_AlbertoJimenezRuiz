#include "gpu_util.h"

struct struct_CPU_CUDA_arrays {
  vector<int> mat_A_CSR_start_rows;
  vector<int> mat_A_CSR_position_columns;
  vector<FLOATING_TYPE> mat_A_CSR_values;

  vector<int> mat_A_CSC_start_columns;
  vector<int> mat_A_CSC_position_rows;
  vector<FLOATING_TYPE> mat_A_CSC_values;

  GPU_array_int *CUDA_mat_A_CSR_start_rows;
  GPU_array_int *CUDA_mat_A_CSR_position_columns;
  GPU_array_f *CUDA_mat_A_CSR_values;

  GPU_array_int *CUDA_mat_A_CSC_start_columns;
  GPU_array_int *CUDA_mat_A_CSC_position_rows;
  GPU_array_f *CUDA_mat_A_CSC_values;

  vector<vector<list<int>>> levels_with_strings_LU;
  vector<int> start_levels_unified_LU;
  vector<int> start_chains_columns_unified_LU;
  vector<int> chains_columns_unified_LU;

  vector<vector<list<int>>> levels_with_strings_L;
  vector<int> start_levels_unified_L;
  vector<int> start_chains_columns_unified_L;
  vector<int> chains_columns_unified_L;

  vector<vector<list<int>>> levels_with_strings_U;
  vector<int> start_levels_unified_U;
  vector<int> start_chains_columns_unified_U;
  vector<int> chains_columns_unified_U;

  GPU_array_int *CUDA_start_levels_unified_LU;
  GPU_array_int *CUDA_start_chains_columns_unified_LU;
  GPU_array_int *CUDA_chains_columns_unified_LU;

  GPU_array_int *CUDA_start_levels_unified_L;
  GPU_array_int *CUDA_start_chains_columns_unified_L;
  GPU_array_int *CUDA_chains_columns_unified_L;

  GPU_array_int *CUDA_start_levels_unified_U;
  GPU_array_int *CUDA_start_chains_columns_unified_U;
  GPU_array_int *CUDA_chains_columns_unified_U;

  vector<int> mat_A_CSR_corresponding_value_in_CSC;
  vector<int> mat_A_CSC_corresponding_value_in_CSR;

  GPU_array_int *CUDA_mat_A_CSR_corresponding_value_in_CSC;
  GPU_array_int *CUDA_mat_A_CSC_corresponding_value_in_CSR;

  vector<int> mat_A_CSR_start_from_diagonal;
  vector<int> mat_A_CSC_start_from_diagonal;

  GPU_array_int *CUDA_mat_A_CSR_start_from_diagonal;
  GPU_array_int *CUDA_mat_A_CSC_start_from_diagonal;

  int n;

  vector<int> max_elements_per_column;

  void create_CUDA() {
    CUDA_mat_A_CSR_start_rows = new GPU_array_int(mat_A_CSR_start_rows);
    CUDA_mat_A_CSR_position_columns =
        new GPU_array_int(mat_A_CSR_position_columns);
    CUDA_mat_A_CSR_values = new GPU_array_f(mat_A_CSR_values);

    CUDA_mat_A_CSC_start_columns = new GPU_array_int(mat_A_CSC_start_columns);
    CUDA_mat_A_CSC_position_rows = new GPU_array_int(mat_A_CSC_position_rows);
    CUDA_mat_A_CSC_values = new GPU_array_f(mat_A_CSC_values);

    CUDA_start_levels_unified_LU = new GPU_array_int(start_levels_unified_LU);
    CUDA_start_chains_columns_unified_LU =
        new GPU_array_int(start_chains_columns_unified_LU);
    CUDA_chains_columns_unified_LU =
        new GPU_array_int(chains_columns_unified_LU);

    CUDA_start_levels_unified_L = new GPU_array_int(start_levels_unified_L);
    CUDA_start_chains_columns_unified_L =
        new GPU_array_int(start_chains_columns_unified_L);
    CUDA_chains_columns_unified_L = new GPU_array_int(chains_columns_unified_L);

    CUDA_start_levels_unified_U = new GPU_array_int(start_levels_unified_U);
    CUDA_start_chains_columns_unified_U =
        new GPU_array_int(start_chains_columns_unified_U);
    CUDA_chains_columns_unified_U = new GPU_array_int(chains_columns_unified_U);

    CUDA_mat_A_CSR_corresponding_value_in_CSC =
        new GPU_array_int(mat_A_CSR_corresponding_value_in_CSC);
    CUDA_mat_A_CSC_corresponding_value_in_CSR =
        new GPU_array_int(mat_A_CSC_corresponding_value_in_CSR);

    CUDA_mat_A_CSR_start_from_diagonal =
        new GPU_array_int(mat_A_CSR_start_from_diagonal);
    CUDA_mat_A_CSC_start_from_diagonal =
        new GPU_array_int(mat_A_CSC_start_from_diagonal);
  }

  void destroy_everything() {
    delete CUDA_mat_A_CSR_start_rows;
    delete CUDA_mat_A_CSR_position_columns;
    delete CUDA_mat_A_CSR_values;

    delete CUDA_start_levels_unified_LU;
    delete CUDA_start_chains_columns_unified_LU;
    delete CUDA_chains_columns_unified_LU;

    delete CUDA_mat_A_CSC_start_columns;
    delete CUDA_mat_A_CSC_position_rows;
    delete CUDA_mat_A_CSC_values;

    delete CUDA_mat_A_CSR_corresponding_value_in_CSC;
    delete CUDA_mat_A_CSC_corresponding_value_in_CSR;

    delete CUDA_mat_A_CSR_start_from_diagonal;
    delete CUDA_mat_A_CSC_start_from_diagonal;

    delete CUDA_start_levels_unified_L;
    delete CUDA_start_chains_columns_unified_L;
    delete CUDA_chains_columns_unified_L;

    delete CUDA_start_levels_unified_U;
    delete CUDA_start_chains_columns_unified_U;
    delete CUDA_chains_columns_unified_U;
  }

  ~struct_CPU_CUDA_arrays() {}
};

void factorize_electrical_system_CPU_sequential(
    struct_CPU_CUDA_arrays &CPU_and_CUDA_arrays, Time_Measurer &measurer);

void factorize_electrical_system_CPU_levels(
    struct_CPU_CUDA_arrays &CPU_and_CUDA_arrays, Time_Measurer &measurer);
void factorize_electrical_system_CUDA(
    struct_CPU_CUDA_arrays &CPU_and_CUDA_arrays, Time_Measurer &measurer);

void triangular_substitution_LU_CPU_sequential(
    struct_CPU_CUDA_arrays &CPU_and_CUDA_arrays, Time_Measurer &measurer,
    FLOATING_TYPE *solution);

void triangular_substitution_LU_CUDA(
    struct_CPU_CUDA_arrays &CPU_and_CUDA_arrays, Time_Measurer &measurer,
    GPU_array_f *solution_CUDA, FLOATING_TYPE *solution);
