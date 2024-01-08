#include "electrical_system.h"
#include "factorize_electrical_system.h"
#include "gpu_util.h"
#include "main.h"
#include "matrix_functions.h"
#include "simulator_functions.h"
#include "sparse_matrix.h"
#include <math.h>

void prepare_matrices(
    int &mat_m, const basic_electric_circuit &Circuit,
    std::vector<int> &interchanges_position_row_columns,
    std::vector<int> &interchanges_position_row_columns_reverse,
    std::vector<int> &interchanges_position_row_columns_only_rows,
    std::vector<int> &interchanges_position_row_columns_only_rows_reverse,
    struct_CPU_CUDA_arrays &CPU_and_CUDA_arrays) {

  auto matrices_return_preprocess = Circuit.create_matrix_and_vector(1.0);
  auto &matrix_A_return_preprocess = std::get<0>(matrices_return_preprocess);
  assert(matrix_A_return_preprocess.m == matrix_A_return_preprocess.n);

  mat_m = matrix_A_return_preprocess.m;

  matrix_A_return_preprocess = reorder_matrix(
      matrix_A_return_preprocess, interchanges_position_row_columns,
      interchanges_position_row_columns_only_rows);

  interchanges_position_row_columns_reverse =
      invert_permutation(interchanges_position_row_columns);
  interchanges_position_row_columns_only_rows_reverse =
      invert_permutation(interchanges_position_row_columns_only_rows);

  auto nonzero_values = obtain_fillin_matrix_CUDA(matrix_A_return_preprocess);

  fill_fillins(nonzero_values, matrix_A_return_preprocess);

  auto dependencies_GLU_3_0 =
      get_dependencies_GLU_3_0(matrix_A_return_preprocess.n, nonzero_values);

  CPU_and_CUDA_arrays.levels_with_strings_LU =
      convert_dependence_list_in_chains_of_columns(dependencies_GLU_3_0);

  auto dependencias_L =
      get_dependencies_matrix_L(matrix_A_return_preprocess.n, nonzero_values);
  auto dependencias_U =
      get_dependencies_matrix_U(matrix_A_return_preprocess.n, nonzero_values);

  CPU_and_CUDA_arrays.levels_with_strings_L =
      convert_dependence_list_in_chains_of_columns(dependencias_L);
  CPU_and_CUDA_arrays.levels_with_strings_U =
      convert_dependence_list_in_chains_of_columns(dependencias_U);

  CPU_and_CUDA_arrays.max_elements_per_column = get_max_numbers_per_column(
      CPU_and_CUDA_arrays.levels_with_strings_LU, nonzero_values);

  convert_chains_columns_levels_into_vector(
      CPU_and_CUDA_arrays.levels_with_strings_LU,
      CPU_and_CUDA_arrays.start_levels_unified_LU,
      CPU_and_CUDA_arrays.start_chains_columns_unified_LU,
      CPU_and_CUDA_arrays.chains_columns_unified_LU);

  convert_chains_columns_levels_into_vector(
      CPU_and_CUDA_arrays.levels_with_strings_L,
      CPU_and_CUDA_arrays.start_levels_unified_L,
      CPU_and_CUDA_arrays.start_chains_columns_unified_L,
      CPU_and_CUDA_arrays.chains_columns_unified_L);

  convert_chains_columns_levels_into_vector(
      CPU_and_CUDA_arrays.levels_with_strings_U,
      CPU_and_CUDA_arrays.start_levels_unified_U,
      CPU_and_CUDA_arrays.start_chains_columns_unified_U,
      CPU_and_CUDA_arrays.chains_columns_unified_U);

  create_CSR_CSC_matrices(
      matrix_A_return_preprocess,

      CPU_and_CUDA_arrays.mat_A_CSR_start_rows,
      CPU_and_CUDA_arrays.mat_A_CSR_position_columns,
      CPU_and_CUDA_arrays.mat_A_CSR_values,

      CPU_and_CUDA_arrays.mat_A_CSC_start_columns,
      CPU_and_CUDA_arrays.mat_A_CSC_position_rows,
      CPU_and_CUDA_arrays.mat_A_CSC_values,

      CPU_and_CUDA_arrays.mat_A_CSR_corresponding_value_in_CSC,
      CPU_and_CUDA_arrays.mat_A_CSC_corresponding_value_in_CSR,

      CPU_and_CUDA_arrays.mat_A_CSR_start_from_diagonal,
      CPU_and_CUDA_arrays.mat_A_CSC_start_from_diagonal);

  CPU_and_CUDA_arrays.n = mat_m;
  CPU_and_CUDA_arrays.create_CUDA();
}

void PVPP_Simulator(Option_PVPP &options) {

  cout << "PVPP dimensions: " << options.number_blocks_total << " "
       << options.number_blocks_1D << " " << options.size_x_cloud_canvas << " "
       << options.size_y_cloud_canvas << endl;

  const vector<FLOATING_TYPE> conductances_convergence{
      100, 10, 1, 0.9, 0.8, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, -1};

  FLOATING_TYPE *canvas_clouds = new FLOATING_TYPE[options.size_x_cloud_canvas *
                                                   options.size_y_cloud_canvas];
  GPU_array_f canvas_clouds_CUDA(options.size_x_cloud_canvas *
                                 options.size_y_cloud_canvas);

  struct_CPU_CUDA_arrays CPU_and_CUDA_arrays;

  int mat_m = 0;

  std::vector<int> interchanges_position_row_columns;
  std::vector<int> interchanges_position_row_columns_reverse;

  std::vector<int> interchanges_position_row_columns_only_rows;
  std::vector<int> interchanges_position_row_columns_only_rows_reverse;

  int it = 0;
  for (int i = 0; i < options.number_cloud_snapshots; i++) {

    print_clouds_CUDA(canvas_clouds_CUDA, options.size_x_cloud_canvas,
                      options.size_y_cloud_canvas, options.cloud_list);

    cudaDeviceSynchronize();
    canvas_clouds_CUDA.Copy_DtoH(canvas_clouds);

    advance_clouds(options.cloud_list, options.size_x_cloud_canvas,
                   options.size_y_cloud_canvas);

    char path[100];
    sprintf(path, "bin/test_%04d.png", i);

    int_2 nodes_voltage_source;         // Where is the PCC
    int position_of_current_that_flows; // Find total current output in future

    vector<node_and_position_xy_panel> node_and_position_panels;
    basic_electric_circuit Circuit;

    create_full_electric_system(
        Circuit, options, canvas_clouds, nodes_voltage_source,
        position_of_current_that_flows, node_and_position_panels);

    if (mat_m == 0) {
      prepare_matrices(mat_m, Circuit, interchanges_position_row_columns,
                       interchanges_position_row_columns_reverse,
                       interchanges_position_row_columns_only_rows,
                       interchanges_position_row_columns_only_rows_reverse,
                       CPU_and_CUDA_arrays);
    }

    const int maximum_iterations = 5000;

    vector<FLOATING_TYPE> solution;

    if (options.early_exit) {
      if (it > 5)
        break;
    } else {
      it = 0;
    }

    for (const auto &conductance_parallel : conductances_convergence) {
      long double diff = 1e10;
      for (;;) {
        if (options.early_exit) {
          if (it > 5)
            break;
        }
        long double objective = 1e-6 * mat_m;
        if (diff < objective)
          break;

        printf("Iteration %d. Conductange %f  Difference:  %e (Objective %e)\n",
               it, (double)conductance_parallel, (double)diff,
               (double)objective);

        it++;
        if (it > maximum_iterations) {
          abortmsg("Did not converge");
        }

        Time_Measurer measurer;

        auto circuit_matrices =
            Circuit.create_matrix_and_vector(conductance_parallel);
        auto &matrix_A = std::get<0>(circuit_matrices);
        auto &Vector_b = std::get<1>(circuit_matrices);
        solution = Vector_b.convert_to_std_vector();

        GPU_array_f solution_CUDA(solution);

        matrix_A = matrix_A.interchange_row_columns(
            &interchanges_position_row_columns_reverse[0]);
        matrix_A = matrix_A.exchange_rows(
            &interchanges_position_row_columns_only_rows[0]);

        apply_permutation(solution,
                          &interchanges_position_row_columns_reverse[0]);
        apply_permutation(solution,
                          &interchanges_position_row_columns_only_rows[0]);

        fill_values_CSR_only(matrix_A, CPU_and_CUDA_arrays.mat_A_CSR_start_rows,
                             CPU_and_CUDA_arrays.mat_A_CSR_position_columns,
                             CPU_and_CUDA_arrays.mat_A_CSR_values);

        fill_values_CSC_only(matrix_A,
                             CPU_and_CUDA_arrays.mat_A_CSC_start_columns,
                             CPU_and_CUDA_arrays.mat_A_CSC_position_rows,
                             CPU_and_CUDA_arrays.mat_A_CSC_values);

        if (options.is_CUDA) {
          factorize_electrical_system_CUDA(CPU_and_CUDA_arrays, measurer);
        } else {
          factorize_electrical_system_CPU_sequential(CPU_and_CUDA_arrays,
                                                     measurer);
        }

        measurer.measure("After solving");

        if (options.is_CUDA) {
          CPU_and_CUDA_arrays.CUDA_mat_A_CSC_values->Copy_HtoD(
              CPU_and_CUDA_arrays.mat_A_CSC_values);

          triangular_substitution_LU_CUDA(CPU_and_CUDA_arrays, measurer,
                                          &solution_CUDA, &solution[0]);

        } else {
          triangular_substitution_LU_CPU_sequential(CPU_and_CUDA_arrays,
                                                    measurer, &solution[0]);
        }

        apply_permutation(solution, &interchanges_position_row_columns[0]);
        calculate_new_iteration(Circuit, solution, diff);
      }
    }

    if (!options.early_exit) {
      auto complex_voltage_in_last_node = Circuit.voltage_drop_ac(
          solution, get<0>(nodes_voltage_source), get<1>(nodes_voltage_source));

      auto generated_current_real = solution[position_of_current_that_flows];
      auto generated_current_imag =
          solution[position_of_current_that_flows + 1];

      auto power = complex_voltage_in_last_node *
                   std::conj(std::complex<FLOATING_TYPE>(
                       generated_current_real, generated_current_imag));

      std::cout << "Voltage:  " << std::abs(complex_voltage_in_last_node)
                << "  Generated power:   "
                << "   in MW  " << power / 1e6 << "   in  W  " << power
                << std::endl;
    }
  }

  delete[] canvas_clouds;
  CPU_and_CUDA_arrays.destroy_everything();
}
