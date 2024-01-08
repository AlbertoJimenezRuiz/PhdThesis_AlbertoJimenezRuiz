#include "dynamic_dq0_electric_system_CUDA.h"
#include "gpu_util.h"
#include "main.h"
#include "obtain_electrical_system.h"

void Dynamic_dq0_Electrical_System_CUDA::destroy_matrices() {
  cpu->destroy_matrices();

  delete cpu;

  delete multipliers_stage1_unified_CUDA;
  delete positions_stage1_unified_CUDA;
  delete element_size_stage1_unified_CUDA;
  delete number_elements_stage1_unified_CUDA;
  delete corresponding_element_stage1_unified_CUDA;

  delete multipliers_stage2_unified_CUDA;
  delete positions_stage2_unified_CUDA;
  delete element_size_stage2_unified_CUDA;
  delete number_elements_stage2_unified_CUDA;
  delete corresponding_element_stage2_unified_CUDA;

  delete multipliers_stage3_unified_CUDA;
  delete multipliers_stage4_unified_CUDA;

  delete large_vector_inputs_no_padding_CUDA;

  delete line_node_input_CUDA;
  delete line_node_output_CUDA;
  delete I0_multiplier_CUDA;
  delete V0_multiplier_CUDA;
  delete admittance_list_CUDA;

  delete first_column_CUDA;
  delete first_column_pos_CUDA;

  delete injected_current_CUDA;
  delete wt_connection_points_CUDA;
  delete voltages_wt_CUDA;

  delete temporary_vector_CUDA_1;
  delete temporary_vector_CUDA_2;
  delete temporary_vector_CUDA_3;
  delete temporary_vector_CUDA_4;
}

void Dynamic_dq0_Electrical_System_CUDA::create_matrices(
    const wt_parameters_struct &parameters, int arg_wt,
    const FLOATING_TYPE arg_delta_t, const C_FLOATING_TYPE fixed_voltage,
    const FLOATING_TYPE *const currents_injected_inicial,
    LIST_MATRIX_SOLVERS solver) {
  cpu = new Dynamic_dq0_Electrical_System();

  cpu->create_matrices(parameters, arg_wt, arg_delta_t, fixed_voltage,
                       currents_injected_inicial, solver);

  // The CUDA kernel reads of a single unified contiguous piece of memory with
  // all contributions. That value is multiplied by a value that depends on that
  // contribution. Contribution vector
  {
    std::vector<std::vector<C_FLOATING_TYPE>> multiplier_vectors_stage1;
    std::vector<std::vector<int>> position_contributions_stage1;

    // Create empty vectors
    for (unsigned int i = 0; i < cpu->A_reduced_LD.m + 2; i++) {
      multiplier_vectors_stage1.push_back(std::vector<C_FLOATING_TYPE>());
      position_contributions_stage1.push_back(std::vector<int>());
    }

    // Contribution 1: Slack voltage and the associated column values that are
    // substracted
    int start_next_section = 0;

    for (unsigned int i = 0; i < cpu->number_connections; i++) {
      if (cpu->connections.cables[i].cable_node_1 == 0) {
        multiplier_vectors_stage1[0].push_back(cpu->I0_multiplier[i]);
        multiplier_vectors_stage1[0].push_back(cpu->V0_multiplier[i]);

        position_contributions_stage1[0].push_back(1 + cpu->number_wt + i);
        position_contributions_stage1[0].push_back(1 + cpu->number_wt +
                                                   cpu->number_connections + i);
      }

      if (cpu->connections.cables[i].cable_node_2 == 0) {
        multiplier_vectors_stage1[0].push_back(-cpu->I0_multiplier[i]);
        multiplier_vectors_stage1[0].push_back(-cpu->V0_multiplier[i]);

        position_contributions_stage1[0].push_back(1 + cpu->number_wt + i);
        position_contributions_stage1[0].push_back(1 + cpu->number_wt +
                                                   cpu->number_connections + i);
      }
    }

    for (unsigned int i = 0; i < cpu->A_reduced_LD.m; i++) {
      multiplier_vectors_stage1[i + 1].push_back(-cpu->A_swing_column[i + 1]);
      position_contributions_stage1[i + 1].push_back(start_next_section +
                                                     0); // Fixed voltage
    }

    start_next_section += 1;

    // Contribution 2: Injected current by wind turbines
    for (int corresponding_wt = 0; corresponding_wt < cpu->number_wt;
         corresponding_wt++) {
      int i = cpu->connections.wt_connection_points[corresponding_wt];
      assert(i > 0); // Nothing is ever injected in slack bus

      multiplier_vectors_stage1[i].push_back(1.0);
      position_contributions_stage1[i].push_back(start_next_section +
                                                 corresponding_wt);
    }

    start_next_section += cpu->number_wt;

    // Contribution 3: Current and voltages of each line, as noted in the
    // discretization process
    for (unsigned int i = 0; i < cpu->number_connections; i++) {
      int bus_index = cpu->connections.cables[i].cable_node_1;

      // Contributions to the slack bus and to ground are discarded (0 and -1)

      if (bus_index > 0) {
        multiplier_vectors_stage1[bus_index].push_back(-cpu->I0_multiplier[i]);
        multiplier_vectors_stage1[bus_index].push_back(-cpu->V0_multiplier[i]);

        position_contributions_stage1[bus_index].push_back(start_next_section +
                                                           i);
        position_contributions_stage1[bus_index].push_back(
            start_next_section + cpu->number_connections + i);
      }

      bus_index = cpu->connections.cables[i].cable_node_2;

      if (bus_index > 0) {
        multiplier_vectors_stage1[bus_index].push_back(cpu->I0_multiplier[i]);
        multiplier_vectors_stage1[bus_index].push_back(cpu->V0_multiplier[i]);

        position_contributions_stage1[bus_index].push_back(start_next_section +
                                                           i);
        position_contributions_stage1[bus_index].push_back(
            start_next_section + cpu->number_connections + i);
      }
    }

    start_next_section += 2 * cpu->number_connections;

    std::vector<C_FLOATING_TYPE> multipliers_stage1_unified;
    std::vector<int> positions_stage1_unified;
    std::vector<int> element_size_stage1_unified;
    std::vector<int> number_elements_stage1_unified;
    std::vector<int> corresponding_element_stage1_unified;

    organize_data_vector_reduction_CUDA(
        true, multiplier_vectors_stage1, position_contributions_stage1,

        multipliers_stage1_unified, positions_stage1_unified,
        element_size_stage1_unified, number_elements_stage1_unified,
        corresponding_element_stage1_unified,

        ELECTRICAL_SYSTEM_BLOCKSIZE);

    multipliers_stage1_unified_CUDA =
        create_GPU_Array_complex(multipliers_stage1_unified);
    positions_stage1_unified_CUDA = new GPU_Array_int(positions_stage1_unified);
    element_size_stage1_unified_CUDA =
        new GPU_Array_int(element_size_stage1_unified);
    number_elements_stage1_unified_CUDA =
        new GPU_Array_int(number_elements_stage1_unified);
    corresponding_element_stage1_unified_CUDA =
        new GPU_Array_int(corresponding_element_stage1_unified);

    max_elements_stage1 = multipliers_stage1_unified.size();
    number_required_threads = max_elements_stage1;
  }

  {

    Solver_Matrix_Diakoptics<C_LONG_DOUBLE> local_solver(cpu->A_reduced_LD);

    std::vector<C_LONG_DOUBLE>
        multiplier_reduction_U_unified; // same size as E_times_Inv_A_unified
    std::vector<C_LONG_DOUBLE> E_times_Inv_A_unified;
    std::vector<std::vector<C_FLOATING_TYPE>> rows_matrices_A_inverse;
    std::vector<std::vector<int>> position_contributions_stage2;

    temporary_value_1_constant_high_precision = 0;

    int start_current_row = 0;

    for (int current_matrix = 0; current_matrix < local_solver.number_groups;
         current_matrix++) {

      auto E_times_Inv_A = local_solver.PartE[current_matrix] *
                           local_solver.PartA[current_matrix].invert_gauss();
      auto inv_A_times_D = local_solver.PartA[current_matrix].invert_gauss() *
                           local_solver.PartD[current_matrix];
      auto denominador_temporary =
          local_solver.PartE[current_matrix] * inv_A_times_D;

      assert(E_times_Inv_A.m == 1);
      assert(denominador_temporary.m == 1);
      assert(denominador_temporary.n == 1);
      assert(inv_A_times_D.n == 1);
      assert(E_times_Inv_A.n == inv_A_times_D.m);
      assert(local_solver.PartA[current_matrix].m ==
             local_solver.PartA[current_matrix].n);

      temporary_value_1_constant_high_precision +=
          denominador_temporary.D(0, 0);

      auto matrix_A_Inv = local_solver.PartA[current_matrix]
                              .invert_gauss()
                              .convert_to<C_FLOATING_TYPE>();

      for (unsigned int i = 0; i < matrix_A_Inv.m; i++) {
        std::vector<C_FLOATING_TYPE> rows_matrix;
        std::vector<int> positions_row_matrix;

        for (unsigned int j = 0; j < matrix_A_Inv.n; j++) {
          rows_matrix.push_back(matrix_A_Inv.D(i, j));
          positions_row_matrix.push_back(j + start_current_row + 1);
        }
        rows_matrices_A_inverse.push_back(rows_matrix);

        position_contributions_stage2.push_back(positions_row_matrix);
      }

      for (unsigned int i = 0; i < inv_A_times_D.m; i++) {
        multiplier_reduction_U_unified.push_back(inv_A_times_D.D(i, 0));
      }

      for (unsigned int i = 0; i < E_times_Inv_A.n; i++) {
        E_times_Inv_A_unified.push_back(E_times_Inv_A.D(0, i));
      }

      start_current_row += E_times_Inv_A.n;
    }

    rows_matrices_A_inverse.push_back({C_FLOATING_TYPE(1.0, 0.0)});
    position_contributions_stage2.push_back({(int)cpu->number_buses - 1});

    rows_matrices_A_inverse.push_back({C_FLOATING_TYPE(1.0, 0.0)});
    position_contributions_stage2.push_back({(int)0});

    temporary_value_1_constant_high_precision =
        C_LONG_DOUBLE(1.0, 0) / (temporary_value_1_constant_high_precision -
                                 (C_LONG_DOUBLE)local_solver.PartF.D(0, 0));
    temporary_value_1_constant_high_precision =
        -temporary_value_1_constant_high_precision;

    std::vector<C_FLOATING_TYPE> multipliers_stage2_unified;
    std::vector<int> positions_stage2_unified;
    std::vector<int> element_size_stage2_unified;
    std::vector<int> number_elements_stage2_unified;
    std::vector<int> corresponding_element_stage2_unified;

    organize_data_vector_reduction_CUDA(
        false, rows_matrices_A_inverse, position_contributions_stage2,

        multipliers_stage2_unified, positions_stage2_unified,
        element_size_stage2_unified, number_elements_stage2_unified,
        corresponding_element_stage2_unified,

        ELECTRICAL_SYSTEM_BLOCKSIZE);

    multipliers_stage2_unified_CUDA =
        create_GPU_Array_complex(multipliers_stage2_unified);
    positions_stage2_unified_CUDA = new GPU_Array_int(positions_stage2_unified);
    element_size_stage2_unified_CUDA =
        new GPU_Array_int(element_size_stage2_unified);
    number_elements_stage2_unified_CUDA =
        new GPU_Array_int(number_elements_stage2_unified);
    corresponding_element_stage2_unified_CUDA =
        new GPU_Array_int(corresponding_element_stage2_unified);
    max_elements_stage2 = multipliers_stage2_unified.size();

    number_required_threads =
        MAX2(number_required_threads, max_elements_stage2);

    multipliers_stage3_unified_CUDA =
        create_GPU_Array_complex(E_times_Inv_A_unified);
    max_elements_stage3 = E_times_Inv_A_unified.size();

    multipliers_stage4_unified_CUDA =
        create_GPU_Array_complex(multiplier_reduction_U_unified);
  }

  {
    std::vector<int> line_node_input_unified;
    std::vector<int> line_node_output_unified;
    for (unsigned int i = 0; i < cpu->number_connections; i++) {
      line_node_input_unified.push_back(
          cpu->connections.cables[i].cable_node_1);
      line_node_output_unified.push_back(
          cpu->connections.cables[i].cable_node_2);
    }

    line_node_input_CUDA = new GPU_Array_int(line_node_input_unified);
    line_node_output_CUDA = new GPU_Array_int(line_node_output_unified);
    I0_multiplier_CUDA = create_GPU_Array_complex(cpu->I0_multiplier);
    V0_multiplier_CUDA = create_GPU_Array_complex(cpu->V0_multiplier);
    admittance_list_CUDA = create_GPU_Array_complex(cpu->admittance_list);

    std::vector<C_FLOATING_TYPE> first_column;
    std::vector<int> first_column_pos;

    for (unsigned int i = 0; i < cpu->A_dynamic.m; i++) {
      if (cpu->A_dynamic.exists(i, 0)) {
        first_column.push_back(cpu->A_dynamic.D(i, 0));
        first_column_pos.push_back(i);
      }
    }

    first_column_CUDA = create_GPU_Array_complex(first_column);
    first_column_pos_CUDA = new GPU_Array_int(first_column_pos);
    first_column_size = first_column.size();

    std::vector<int> wt_connection_points_unified;

    for (int i = 0; i < cpu->number_wt; i++) {
      wt_connection_points_unified.push_back(
          cpu->connections.wt_connection_points[i]);
    }
    wt_connection_points_CUDA = new GPU_Array_int(wt_connection_points_unified);
  }

  std::vector<C_FLOATING_TYPE> large_vector_inputs_no_padding;

  large_vector_inputs_no_padding.push_back(0); // Dummy value for slack voltage

  for (int i = 0; i < cpu->number_wt; i++) {
    large_vector_inputs_no_padding.push_back(
        C_FLOATING_TYPE(0)); // Dummy values for injected current
  }

  for (unsigned int i = 0; i < cpu->number_connections; i++) {
    large_vector_inputs_no_padding.push_back(cpu->I0_previous[i]);
  }

  for (unsigned int i = 0; i < cpu->number_connections; i++) {
    large_vector_inputs_no_padding.push_back(cpu->V0_previous[i]);
  }

  large_vector_inputs_no_padding_CUDA = new GPU_Array_complex(
      1 + cpu->number_wt + cpu->number_connections + cpu->number_connections);
  large_vector_inputs_no_padding_CUDA->copy_HtoD(
      ((cuComplexType *)(&large_vector_inputs_no_padding[0])));

  injected_current_CUDA = new GPU_Array_float(2);
  voltages_wt_CUDA = new GPU_Array_float(cpu->number_wt * 2);

  temporary_vector_CUDA_1 = new GPU_Array_complex(number_required_threads * 2);
  temporary_vector_CUDA_1->memset(0);
  temporary_vector_CUDA_2 = new GPU_Array_complex(number_required_threads * 2);
  temporary_vector_CUDA_2->memset(0);
  temporary_vector_CUDA_3 = new GPU_Array_complex(number_required_threads * 2);
  temporary_vector_CUDA_3->memset(0);
  temporary_vector_CUDA_4 = new GPU_Array_complex(number_required_threads * 2);
  temporary_vector_CUDA_4->memset(0);
}
