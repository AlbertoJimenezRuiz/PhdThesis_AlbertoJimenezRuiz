#include "linearized_wt_matrices.h"
#include "main.h"
#include "obtain_electrical_system.h"

Linearized_Matrices_WT *initialize_wt_voltage_in_grid(
    const wt_parameters_struct &parameters, int number_wt, FLOATING_TYPE v_d,
    FLOATING_TYPE v_q, FLOATING_TYPE desired_reactive, FLOATING_TYPE wind) {
  const C_FLOATING_TYPE fixed_voltage(v_d, v_q); // aligned to q-axis

  Static_Electrical_System syst;
  syst.create_matrices(parameters, number_wt);

  Linearized_Matrices_WT *matrices = new Linearized_Matrices_WT[number_wt];

  FLOATING_TYPE *current_inject = new FLOATING_TYPE[number_wt * 2];
  FLOATING_TYPE *voltage_terminals = new FLOATING_TYPE[number_wt * 2];

  for (int i = 0; i < number_wt; i++) {
    voltage_terminals[2 * i + 0] = fixed_voltage.real();
    voltage_terminals[2 * i + 1] = fixed_voltage.imag();
  }

  for (int it = 0; it < 10; it++) {

    for (int i = 0; i < number_wt; i++) {
      matrices[i] = std::move(Linearized_Matrices_WT(
          &parameters, voltage_terminals[2 * i + 0],
          voltage_terminals[2 * i + 1], desired_reactive, wind));

      current_inject[2 * i + 0] = matrices[i].MatrY0[0];
      current_inject[2 * i + 1] = matrices[i].MatrY0[1];
    }
    syst.calculate_voltages(current_inject, voltage_terminals, fixed_voltage,
                            NULL);
  }
  delete[] current_inject;
  delete[] voltage_terminals;

  syst.destroy_matrices();

  return matrices;
}
