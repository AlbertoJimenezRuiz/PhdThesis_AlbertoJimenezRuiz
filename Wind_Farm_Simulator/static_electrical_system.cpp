#include "main.h"
#include "obtain_electrical_system.h"
#include <memory>

C_FLOATING_TYPE create_static_admittance(const description_cable &connection,
                                         const FLOATING_TYPE base_impedance,
                                         const FLOATING_TYPE omega) {
  if (!std::equal_to<FLOATING_TYPE>()(connection.C, 0)) {
    auto impedance_C =
        C_FLOATING_TYPE(0, -1 / (omega * connection.C)) / base_impedance;
    return C_FLOATING_TYPE(1, 0) / impedance_C;
  } else {
    auto R = connection.R / base_impedance;
    auto L = (connection.L * omega) / base_impedance;

    return C_FLOATING_TYPE(1, 0) / C_FLOATING_TYPE(R, L);
  }
}

C_Sparse_Matrix
get_wt_static_grid_matrix(description_connections &connections) {
  const int rows = connections.nodes_connection;

  C_Sparse_Matrix matrix_admittances(rows, rows);

  for (auto &con : connections.cables) {

    auto Y = create_static_admittance(con, connections.base_impedance_mt,
                                      connections.grid_angular_speed);
    matrix_admittances.write_add(con.cable_node_1, con.cable_node_1, Y);

    if (con.cable_node_2 != -1) {
      matrix_admittances.write_add(con.cable_node_2, con.cable_node_2, Y);
      matrix_admittances.write_add(con.cable_node_1, con.cable_node_2, -Y);
      matrix_admittances.write_add(con.cable_node_2, con.cable_node_1, -Y);
    }
  }

  return matrix_admittances;
}

void Static_Electrical_System::create_matrices(
    const wt_parameters_struct &parameters, int arg_number_wt) {
  number_wt = arg_number_wt;

  int number_wt_strings, length_strings, remaining_wind_turbines;
  connections = get_connections_list(parameters, number_wt, number_wt_strings,
                                     length_strings, remaining_wind_turbines);
  auto A = get_wt_static_grid_matrix(connections);

  // Reduce admittance matrix without the slack bus column (first column)
  LONG_DOUBLE_C_Sparse_Matrix A_reduced(A.m - 1, A.n - 1);

  for (unsigned int i = 0; i < A_reduced.m; i++) {
    for (unsigned int j = 0; j < A_reduced.n; j++) {
      A_reduced.write(i, j, A.D(i + 1, j + 1));
    }
  }

  // LDLt decomposition for symmetric complex matrices
  LONG_DOUBLE_C_Sparse_Matrix D;
  LDLt_no_permutations(A_reduced, L_reduced, D);

  // Convert to flattened arrays
  m = L_reduced.m;
  D_reduced = new C_LONG_DOUBLE[L_reduced.m];

  for (unsigned int i = 0; i < m; i++) {
    D_reduced[i] = D.D(i, i);
  }

  A_swing_column = new C_LONG_DOUBLE[(m + 1)];
  for (unsigned int i = 0; i < m + 1; i++) {
    A_swing_column[i] = A.D(i, 0);
  }
}

void Static_Electrical_System::calculate_voltages(
    const FLOATING_TYPE *const currents_injected, FLOATING_TYPE *voltages_wt,
    const C_FLOATING_TYPE fixed_voltage, FLOATING_TYPE *copy_all_voltages) {
  std::unique_ptr<C_LONG_DOUBLE[]> voltages_complex(
      new C_LONG_DOUBLE[m + 1]); // zeroed

  // At the connection point, it is not known what is injected at the PCC.
  // Calculated at the end.

  // Write currents
  for (int i = 0; i < number_wt; i++) {
    voltages_complex[connections.wt_connection_points[i]] = C_FLOATING_TYPE(
        currents_injected[2 * i + 0], currents_injected[2 * i + 1]);
  }

  // Fixed voltage
  voltages_complex[0] = fixed_voltage;

  // From now on the actual calculations begin
  // First, substract first column times the fixed voltage
  for (unsigned int i = 0; i < m; i++) {
    voltages_complex[1 + i] -=
        A_swing_column[1 + i] * (C_LONG_DOUBLE)fixed_voltage;
  }

  // Calculate remaining voltages with the LDL decomposition
  solve_substitution_LDL_decomposition(L_reduced, D_reduced,
                                       &voltages_complex[1]);

  if (voltages_wt) {
    for (int i = 0; i < number_wt; i++) {
      auto wt_temp = voltages_complex[connections.wt_connection_points[i]];
      voltages_wt[i * 2 + 0] = (FLOATING_TYPE)(wt_temp.real());
      voltages_wt[i * 2 + 1] = (FLOATING_TYPE)(wt_temp.imag());
    }
  }

  if (copy_all_voltages) {
    for (unsigned int i = 0; i < m + 1; i++) {
      copy_all_voltages[2 * i + 0] =
          (FLOATING_TYPE)(voltages_complex[i].real());
      copy_all_voltages[2 * i + 1] =
          (FLOATING_TYPE)(voltages_complex[i].imag());
    }
  }
}

void Static_Electrical_System::destroy_matrices() {
  delete[] D_reduced;
  delete[] A_swing_column;
}
