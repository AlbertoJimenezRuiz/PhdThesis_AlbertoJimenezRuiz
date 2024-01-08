#include "electrical_system.h"
#include "main.h"
#include "matrix_functions.h"
#include "sparse_matrix.h"

bool is_value_change_justified(FLOATING_TYPE v1, FLOATING_TYPE v2) {
  // Using __isnan because std::isnan fails with -ffast-math
  if (__isnan(v1))
    return false;
  if (__isnan(v2))
    return false;

  if (std::abs(v1 - v2) > (FLOATING_TYPE)1)
    return false;

  return true;
}

inline FLOATING_TYPE &
reference_original_variable_change(basic_electric_circuit &Circuit, int i) {
  return Circuit.vector_voltages_nonlinear_currentsources_iterate[i];
}

inline long unsigned int
get_movement_vector_size(basic_electric_circuit &Circuit) {
  return Circuit.vector_voltages_nonlinear_currentsources_iterate.size();
}

inline FLOATING_TYPE
get_new_voltage_drop_value(const basic_electric_circuit &Circuit,
                           const vector<FLOATING_TYPE> &solution, int i) {
  return Circuit.voltage_drop_dc(
      solution, std::get<0>(Circuit.list_nonlinear_current_sources[i]),
      std::get<1>(Circuit.list_nonlinear_current_sources[i]));
}

void calculate_new_iteration(basic_electric_circuit &Circuit,
                             vector<FLOATING_TYPE> &solution,
                             long double &diff) {
  vector<FLOATING_TYPE> movement(get_movement_vector_size(Circuit));
  vector<FLOATING_TYPE> possible_new_temporary_values(
      get_movement_vector_size(Circuit));

  for (auto &x : movement) {
    x = (FLOATING_TYPE)1;
  }

  for (;;) {
    for (unsigned int i = 0; i < movement.size(); i++) {
      const auto &m = movement[i];
      const auto &old_value = reference_original_variable_change(Circuit, i);
      const auto &new_value = get_new_voltage_drop_value(Circuit, solution, i);

      possible_new_temporary_values[i] =
          old_value * ((FLOATING_TYPE)1 - m) + new_value * m;
    }

    for (const auto &i : movement) {
      if (i < 1e-13) {
        abortmsg("Convergence error\n");
      }
    }

    bool should_movement_be_reduced = false;
    for (unsigned int i = 0; i < movement.size(); i++) {
      if (is_value_change_justified(
              reference_original_variable_change(Circuit, i),
              possible_new_temporary_values[i])) {
        continue;
      }

      movement[i] /= (FLOATING_TYPE)2;
      should_movement_be_reduced = true;
    }

    diff = 0;
    if (!should_movement_be_reduced) {
      for (unsigned int i = 0; i < possible_new_temporary_values.size(); i++) {
        diff += std::abs(possible_new_temporary_values[i] -
                         reference_original_variable_change(Circuit, i));
        reference_original_variable_change(Circuit, i) =
            possible_new_temporary_values[i];
      }
      break;
    }
  }

  // Update current sources of inverters, if there is any
  for (unsigned int i = 0;
       i < Circuit.list_values_inverter_dc_ac_iterate.size(); i++) {
    auto &value_list = Circuit.list_values_inverter_dc_ac_iterate[i];

    auto value_list_old = value_list;

    auto &nodelist = Circuit.list_inverters_fixed_voltage[i];

    std::get<0>(value_list) = Circuit.voltage_drop_dc(
        solution, std::get<0>(nodelist), std::get<1>(nodelist));
    std::get<1>(value_list) = Circuit.voltage_drop_dc(
        solution,
        Circuit.get_imaginary_node_from_real_AC(std::get<0>(nodelist)),
        Circuit.get_imaginary_node_from_real_AC(std::get<1>(nodelist)));

    std::get<2>(value_list) = solution[std::get<2>(nodelist)];

    diff += std::abs(std::get<0>(value_list) - std::get<0>(value_list_old)) +
            std::abs(std::get<1>(value_list) - std::get<1>(value_list_old)) +
            std::abs(std::get<2>(value_list) - std::get<2>(value_list_old));
  }
}
