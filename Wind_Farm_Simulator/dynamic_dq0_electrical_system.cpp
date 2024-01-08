#include "main.h"
#include "obtain_electrical_system.h"

void get_wt_dynamic_dq0_grid_matrix(
    const description_connections &connections, LONG_DOUBLE_C_Sparse_Matrix &A,
    C_VECTOR &admittance_list, C_VECTOR &I0_multiplier, C_VECTOR &V0_multiplier,
    const FLOATING_TYPE omega, const FLOATING_TYPE delta_t) {
  int number_buses = connections.nodes_connection;
  A = LONG_DOUBLE_C_Sparse_Matrix(number_buses, number_buses);

  admittance_list.clear();
  I0_multiplier.clear();
  V0_multiplier.clear();

  for (const auto &line : connections.cables) {
    const FLOATING_TYPE R_pu = line.R / connections.base_impedance_mt;
    const FLOATING_TYPE L_pu = line.L / connections.base_inductance_mt;
    const FLOATING_TYPE C_pu = line.C / connections.base_capacitance_mt;

    C_FLOATING_TYPE impedance;
    // Backward Euler method
    if (!std::equal_to<FLOATING_TYPE>()(C_pu, 0)) {
      impedance = C_FLOATING_TYPE(delta_t * omega, 0) / C_pu /
                  (C_FLOATING_TYPE(1, delta_t * omega));
      I0_multiplier.push_back(0);
      V0_multiplier.push_back(-(C_pu) / delta_t / omega);

    } else {
      C_FLOATING_TYPE value_(R_pu, L_pu);
      impedance = C_FLOATING_TYPE(R_pu + L_pu / delta_t / omega, L_pu);
      I0_multiplier.push_back((L_pu) / (value_ * delta_t * omega + L_pu));
      V0_multiplier.push_back(0);
    }

    C_FLOATING_TYPE admittance = C_FLOATING_TYPE(1, 0) / impedance;

    admittance_list.push_back(admittance);

    A.write_add(line.cable_node_1, line.cable_node_1, admittance);

    if (line.cable_node_2 != -1) {
      A.write_add(line.cable_node_2, line.cable_node_2, admittance);

      A.write_add(line.cable_node_1, line.cable_node_2, -admittance);
      A.write_add(line.cable_node_2, line.cable_node_1, -admittance);
    }
  }
}

void dq0_line_voltage(const description_connections &connections,
                      const C_FLOATING_TYPE *voltage_nodes,
                      C_FLOATING_TYPE *res) {
  for (unsigned int i = 0; i < connections.cables.size(); i++) {
    int n1 = connections.cables[i].cable_node_1;
    int n2 = connections.cables[i].cable_node_2;

    res[i] = voltage_nodes[n1];

    if (n2 != -1) {
      res[i] -= voltage_nodes[n2];
    }
  }
}

void Dynamic_dq0_Electrical_System::create_matrices(
    const wt_parameters_struct &parameters, int arg_number_wt,
    const FLOATING_TYPE arg_delta_t, const C_FLOATING_TYPE fixed_voltage,
    const FLOATING_TYPE *const currents_injected_initial,
    LIST_MATRIX_SOLVERS solver_type) {

  delta_t = arg_delta_t;
  number_wt = arg_number_wt;

  int number_wt_strings, length_strings, remaining_wind_turbines;
  connections = get_connections_list(parameters, number_wt, number_wt_strings,
                                     length_strings, remaining_wind_turbines);

  number_connections = connections.cables.size(); // number of transmission
                                                  // lines
  number_buses = connections.nodes_connection;

  V0_previous_buses = new C_FLOATING_TYPE[number_buses];
  V0_previous = new C_FLOATING_TYPE[number_connections];
  I0_previous = new C_FLOATING_TYPE[number_connections];
  extra_current_sources = new C_FLOATING_TYPE[number_connections];

  // Instead of solving the static system and assign the voltage to each bus,
  // the fixed voltage is assigned to all nodes, as it will be close to the real
  // value anyway
  if (currents_injected_initial) {
    FLOATING_TYPE *voltages_static_temp = new FLOATING_TYPE[2 * number_buses];
    Static_Electrical_System static_system;
    static_system.create_matrices(parameters, number_wt);
    static_system.calculate_voltages(currents_injected_initial, NULL,
                                     fixed_voltage, voltages_static_temp);

    for (int i = 0; i < connections.nodes_connection; i++) {
      V0_previous_buses[i] = C_FLOATING_TYPE(voltages_static_temp[2 * i],
                                             voltages_static_temp[2 * i + 1]);
    }
    static_system.destroy_matrices();
    delete[] voltages_static_temp;
  } else {
    for (int i = 0; i < connections.nodes_connection; i++) {
      V0_previous_buses[i] = fixed_voltage;
    }
  }

  // voltage drop for each connection
  dq0_line_voltage(connections, V0_previous_buses, V0_previous);

  // current through each line = voltage_drop*admittance
  for (unsigned int i = 0; i < number_connections; i++) {
    I0_previous[i] = create_static_admittance(connections.cables[i],
                                              connections.base_impedance_mt,
                                              connections.grid_angular_speed) *
                     V0_previous[i];
  }

  // get discretized electrical system
  LONG_DOUBLE_C_Sparse_Matrix A_long_double_dynamic;
  get_wt_dynamic_dq0_grid_matrix(
      connections, A_long_double_dynamic, admittance_list, I0_multiplier,
      V0_multiplier,
      (FLOATING_TYPE)
          parameters.Base_Electrical_angular_speed, // 2*pi*grid_frequency
      delta_t);

  A_dynamic = A_long_double_dynamic.convert_to<C_FLOATING_TYPE>();

  currents_complex = new C_FLOATING_TYPE[number_buses];

  // Create electrical system matrix inverse without swing bus
  A_reduced_LD = LONG_DOUBLE_C_Sparse_Matrix((unsigned int)number_buses - 1,
                                             (unsigned int)number_buses - 1);

  for (unsigned int i = 0; i < number_buses - 1; i++) {
    for (unsigned int j = 0; j < number_buses - 1; j++) {
      A_reduced_LD.write(i, j, A_dynamic.D(i + 1, j + 1));
    }
  }

  switch (solver_type) {
  case SOLVER_LDL:
    solver = new Solver_Matrix_LDL<C_FLOATING_TYPE>(A_reduced_LD);
    break;
  case SOLVER_INV:
    solver = new Solver_Matrix_Inverse<C_FLOATING_TYPE>(A_reduced_LD);
    break;
  case SOLVER_DIAKOPTICS_INV:
    solver = new Solver_Matrix_Diakoptics<C_FLOATING_TYPE>(A_reduced_LD);
    break;
  default:
    assert(0);
    break;
  }

  A_swing_column = new C_FLOATING_TYPE[A_dynamic.m];

  for (unsigned int i = 0; i < A_dynamic.m; i++) {
    A_swing_column[i] = A_dynamic.D(i, 0);
  }
}

void Dynamic_dq0_Electrical_System::calculate_voltages(
    const FLOATING_TYPE *const currents_injected, FLOATING_TYPE *voltages_wt,
    FLOATING_TYPE *current_injected_to_grid,

    const C_FLOATING_TYPE fixed_voltage) {

  assert(solver);

  for (unsigned int i = 0; i < number_connections;
       i++) // Injected current for each transmission line
  {
    extra_current_sources[i] =
        I0_previous[i] * I0_multiplier[i] + V0_previous[i] * V0_multiplier[i];
  }

  // Injected current for each moment = wind turbine + previous corrections

  for (unsigned int i = 0; i < A_dynamic.m; i++) {
    currents_complex[i] = 0;
  }

  for (int i = 0; i < number_wt; i++) {
    currents_complex[connections.wt_connection_points[i]] = C_FLOATING_TYPE(
        currents_injected[2 * i + 0], currents_injected[2 * i + 1]);
  }

  for (unsigned int i = 0; i < number_connections; i++) {
    currents_complex[connections.cables[i].cable_node_1] -=
        extra_current_sources[i];

    if (connections.cables[i].cable_node_2 != -1) {
      currents_complex[connections.cables[i].cable_node_2] +=
          extra_current_sources[i];
    }
  }

  // Substract the swing column
  V0_previous_buses[0] = fixed_voltage;

  for (unsigned int cc = 0; cc < number_buses - 1; cc++) {
    V0_previous_buses[cc + 1] =
        currents_complex[cc + 1] - A_swing_column[cc + 1] * fixed_voltage;
  }

  solver->solve(&V0_previous_buses[1]);

  dq0_line_voltage(connections, V0_previous_buses, V0_previous);

  for (unsigned int i = 0; i < number_connections; i++) {
    I0_previous[i] =
        admittance_list[i] * V0_previous[i] + extra_current_sources[i];
  }

  for (int i = 0; i < number_wt; i++) {
    voltages_wt[2 * i + 0] =
        V0_previous_buses[connections.wt_connection_points[i]].real();
    voltages_wt[2 * i + 1] =
        V0_previous_buses[connections.wt_connection_points[i]].imag();
  }

  if (current_injected_to_grid) {
    C_FLOATING_TYPE current_injected_complex = 0;

    for (unsigned int cc = 0; cc < A_dynamic.m; cc++) {
      // Matrix is symmetrical
      current_injected_complex += A_swing_column[cc] * V0_previous_buses[cc];
    }

    for (unsigned int i = 0; i < number_connections; i++) {
      if (connections.cables[i].cable_node_1 == 0)
        current_injected_complex += extra_current_sources[i];

      if (connections.cables[i].cable_node_2 == 0)
        current_injected_complex -= extra_current_sources[i];
    }

    current_injected_to_grid[0] = current_injected_complex.real();
    current_injected_to_grid[1] = current_injected_complex.imag();
  }
}

void Dynamic_dq0_Electrical_System::destroy_matrices() {
  delete[] V0_previous_buses;
  delete[] V0_previous;
  delete[] I0_previous;
  delete[] extra_current_sources;
  delete[] currents_complex;
  delete[] A_swing_column;

  delete solver;
}
