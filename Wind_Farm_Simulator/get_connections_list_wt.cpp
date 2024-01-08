#include "main.h"
#include "obtain_electrical_system.h"
#include "sparse_matrix.h"

void add_cable(description_connections &data, int n1, int n2, FLOATING_TYPE R,
               FLOATING_TYPE L, FLOATING_TYPE C) {
  assert(n2 != 0);

  data.cables.push_back(description_cable{n1, n2, R, L, 0});

  // pi-section line
  //-1 indicates the other connection goes to ground

  if (!std::equal_to<FLOATING_TYPE>()(C, 0)) {
    if (n1 == 0) {
      data.cables.push_back(description_cable{n2, -1, 0, 0, C});
    } else {
      data.cables.push_back(
          description_cable{n1, -1, 0, 0, C / (FLOATING_TYPE)2});

      data.cables.push_back(
          description_cable{n2, -1, 0, 0, C / (FLOATING_TYPE)2});
    }
  }
}

description_connections
get_connections_list(const wt_parameters_struct &parameters,
                     const int number_wt, int &number_wt_strings,
                     int &length_wt_strings, int &remaining_wt) {
  assert(number_wt > 0);

  description_connections result;
  result.grid_angular_speed = (FLOATING_TYPE)(2 * M_PI * 50.0);
  result.base_current_mt = (FLOATING_TYPE)(2.0 / 3.0 * parameters.Base_Power /
                                           parameters.Base_Voltage_mt);
  result.base_impedance_mt =
      (FLOATING_TYPE)(parameters.Base_Voltage_mt / result.base_current_mt);
  result.base_inductance_mt =
      (FLOATING_TYPE)(parameters.Base_Voltage_mt / result.base_current_mt) /
      result.grid_angular_speed;
  result.base_capacitance_mt =
      (FLOATING_TYPE)(result.base_current_mt / parameters.Base_Voltage_mt) /
      result.grid_angular_speed;

  length_wt_strings = 5;

  number_wt_strings = number_wt / length_wt_strings;
  remaining_wt = number_wt - number_wt_strings * length_wt_strings;

  assert(remaining_wt >= 0);

  if (remaining_wt >= length_wt_strings) {
    number_wt_strings++;
    remaining_wt -= length_wt_strings;
  }

  assert(number_wt_strings * length_wt_strings + remaining_wt == number_wt);
  assert(remaining_wt < length_wt_strings);

  // Node 0: PCC with voltage 1<0
  // Node 1: Node all WT are connected to

  // Node 2: Elevator Transformer 1-Cable
  // Node 3: WT 1-Elevator Transformer 1
  // Node 4: Elevator Transformer 2-Cable
  // Node 5: WT 2-Elevator Transformer 2
  // And so on.

  int processed_wt = 0;

  for (int string_current = 0; string_current < number_wt_strings + 1;
       string_current++) {
    for (int wt_current = 0; wt_current < ((string_current == number_wt_strings)
                                               ? remaining_wt
                                               : length_wt_strings);
         wt_current++) {
      int n1, n2;

      if (wt_current == 0) {
        n1 = 1;
        n2 = 2 + (processed_wt)*2;
      } else {
        n1 = 2 + (processed_wt - 1) * 2;
        n2 = 2 + (processed_wt)*2;
      }

      FLOATING_TYPE distance = 0.5; //(km)

      // Base change
      const FLOATING_TYPE L_elevator_transformer =
          (FLOATING_TYPE)(parameters.Impedance_percent_transformer /
                          100.0 // P.u. value

                          * parameters.Base_Power // Base change
                          / parameters.Power_transformer

                          *
                          result.base_inductance_mt); // convert to real values

      add_cable(result, n1, n2, (FLOATING_TYPE)(parameters.cable_R * distance),
                (FLOATING_TYPE)(parameters.cable_L * distance),
                (FLOATING_TYPE)(parameters.cable_C * distance));

      add_cable(result, n2, 1 + n2, 0, L_elevator_transformer, 0);

      // For Simulink, in order to compare the simulation strategies
      result.cables.push_back(
          description_cable{n2 + 1, -1, (FLOATING_TYPE)(1e10),
                            (FLOATING_TYPE)(0), (FLOATING_TYPE)(0)});

      result.wt_connection_points.push_back(1 + n2);

      processed_wt++;
    }
  }
  assert(processed_wt == number_wt);
  const FLOATING_TYPE distance_point_common_coupling =
      (FLOATING_TYPE)0.02; //(km)
  add_cable(
      result, 0, 1,
      (FLOATING_TYPE)(parameters.cable_R * distance_point_common_coupling),
      (FLOATING_TYPE)(parameters.cable_L * distance_point_common_coupling),
      (FLOATING_TYPE)(parameters.cable_C * distance_point_common_coupling));

  result.nodes_connection = 2 * number_wt + 2;

  // Reorder matrix rows
  // Node 0 must be the first one, invert the order of the rest. The result is a
  // lower fill-in.

  std::vector<int> translate_vector;
  translate_vector.reserve(result.nodes_connection);

  for (int i = 0; i < result.nodes_connection; i++) {
    if (i == 0) {
      translate_vector.push_back(0); // Do not touch first bus
      continue;
    }
    // Invert the others
    translate_vector.push_back(result.nodes_connection - i);
  }

  for (auto &current_busbar : result.wt_connection_points) {
    current_busbar = translate_vector[current_busbar];
  }

  for (auto &current_line : result.cables) {
    current_line.cable_node_1 = translate_vector[current_line.cable_node_1];
    if (current_line.cable_node_2 != -1) {
      current_line.cable_node_2 = translate_vector[current_line.cable_node_2];
    }
  }

  return result;
}
