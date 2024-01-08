#include "electrical_system.h"
#include "main.h"
#include "matrix_functions.h"

basic_electric_circuit::basic_electric_circuit() {
  vector_x_size = 0;

  admittance_list.clear();
  list_current_sources.clear();
  list_voltage_sources.clear();
  list_rupture_current_sources.clear();

  vector_voltages_nonlinear_currentsources_iterate.clear();

  correspondance_nodes_voltage_vector_x.clear();
  correspondance_voltage_sources_vector_x.clear();
  correspondance_rupture_current_sources_vector_x.clear();
}

FLOATING_TYPE
basic_electric_circuit::voltage_drop_dc(const vector<FLOATING_TYPE> &solution,
                                        int n1, int n2) const {
  FLOATING_TYPE res = 0;

  if (n1 != 0)
    res = solution[obt_f(T_VOLTAGE_DC, n1)];

  if (n2 != 0)
    res -= solution[obt_f(T_VOLTAGE_DC, n2)];

  return res;
}

std::complex<FLOATING_TYPE>
basic_electric_circuit::voltage_drop_ac(const vector<FLOATING_TYPE> &solution,
                                        int n1, int n2) const {
  std::complex<FLOATING_TYPE> res = 0;

  if (n1 != 0)
    res = solution[obt_f(T_VOLTAGE_DC, n1)] +
          solution[obt_f(T_VOLTAGE_DC, n1) + 1] * 1i;

  if (n2 != 0)
    res -= solution[obt_f(T_VOLTAGE_DC, n2)] +
           solution[obt_f(T_VOLTAGE_DC, n2) + 1] * 1i;

  return res;
}

int basic_electric_circuit::create_node() {
  int node = (int)correspondance_nodes_voltage_vector_x.size() + 1;
  correspondance_nodes_voltage_vector_x.push_back(vector_x_size);
  vector_x_size++;
  return node;
}

int basic_electric_circuit::create_node_ac() {
  int node = create_node(); // Real part of phasor
  create_node();            // Imaginary part of phasor
  return node;
}

int basic_electric_circuit::get_imaginary_node_from_real_AC(int node) const {
  if (node == 0)
    return 0;
  else
    return node + 1;
}

void basic_electric_circuit::add_rupture_current_source(int node_1,
                                                        int node_2) {
  list_rupture_current_sources.push_back(
      {node_1, node_2, correspondance_rupture_current_sources_vector_x.size()});
  correspondance_rupture_current_sources_vector_x.push_back(vector_x_size);
  vector_x_size++;
}

int basic_electric_circuit::add_linear_element(ADD_ELEMENT_TYPE type,
                                               FLOATING_TYPE value_element,
                                               int node_1, int node_2) {
  // node 1 is positive
  // node 2 is negative

  vector<value_device_and_node_number> *vector;

  int res = -1;

  switch (type) {
  case ADMITTANCE:
    vector = &admittance_list;
    break;
  case CURRENT_SOURCE:
    vector = &list_current_sources;
    break;
  case VOLTAGE_SOURCE:
    vector = &list_voltage_sources;
    res = vector_x_size;
    correspondance_voltage_sources_vector_x.push_back(vector_x_size);
    vector_x_size++;
    break;
  default:
    assert(0);
    break;
  }

  vector->push_back({node_1, node_2, value_element});
  return res;
}

int basic_electric_circuit::add_linear_element_ac(
    ADD_ELEMENT_TYPE type, complex<FLOATING_TYPE> value_element, int node_1,
    int node_2) {
  FLOATING_TYPE value_real = value_element.real();
  FLOATING_TYPE value_imag = value_element.imag();

  const int node_1_real = node_1;
  const int node_1_imag = get_imaginary_node_from_real_AC(node_1_real);
  const int node_2_real = node_2;
  const int node_2_imag = get_imaginary_node_from_real_AC(node_2_real);

  int res = -1;

  switch (type) {
  case ADMITTANCE:
    admittance_list_ac.push_back({node_1, node_2, value_element});

    break;
  case CURRENT_SOURCE:
    add_linear_element(CURRENT_SOURCE, value_real, node_1_real, node_2_real);
    add_linear_element(CURRENT_SOURCE, value_imag, node_1_imag, node_2_imag);
    break;
  case VOLTAGE_SOURCE:
    res = add_linear_element(VOLTAGE_SOURCE, value_real, node_1_real,
                             node_2_real);
    add_linear_element(VOLTAGE_SOURCE, value_imag, node_1_imag, node_2_imag);
    break;
  default:
    assert(0);
    break;
  }

  return res;
}

// One inverter control voltage in DC side and dumps all power on the AC side
void basic_electric_circuit::create_inverter_dc_ac(
    FLOATING_TYPE V_dc, std::complex<FLOATING_TYPE> V_initial_ac, int node_1_dc,
    int node_2_dc, int node_1_ac_real, int node_2_ac_real) {

  FLOATING_TYPE I1_initial = 0;

  int associated_voltage_source =
      add_linear_element(VOLTAGE_SOURCE, V_dc, node_1_dc, node_2_dc);

  int number_this_inverter = (int)list_values_inverter_dc_ac_iterate.size();
  list_values_inverter_dc_ac_iterate.push_back(
      {V_initial_ac.real(), V_initial_ac.imag(), I1_initial});

  list_inverters_fixed_voltage.push_back({node_1_ac_real, node_2_ac_real,
                                          associated_voltage_source, V_dc,
                                          number_this_inverter});
}

void basic_electric_circuit::add_nonlinear_voltage_source(
    function_pointer_type func, function_pointer_type derivative_func,
    FLOATING_TYPE V_initial, int node_1, int node_2) {

  int unknown_V = (int)list_nonlinear_current_sources.size();
  vector_voltages_nonlinear_currentsources_iterate.push_back(V_initial);

  list_nonlinear_current_sources.push_back(
      {node_1, node_2, func, derivative_func, unknown_V});
}

int basic_electric_circuit::obt_f(const MATRIX_ELEMENT_TYPE tipo_variable,
                                  const int val) const {
  if (tipo_variable == T_VOLTAGE_DC) {
    assert(val > 0);
    assert(val <= (int)correspondance_nodes_voltage_vector_x.size());
    return correspondance_nodes_voltage_vector_x[val - 1];
  } else if (tipo_variable == T_VOLTAGE_SOURCE_DC) {
    assert(val >= 0);
    assert(val <= (int)correspondance_voltage_sources_vector_x.size());
    return correspondance_voltage_sources_vector_x[val];
  } else if (tipo_variable == T_CURRENT_SOURCE_RUPTURE_DC) {
    assert(val >= 0);
    assert(val <= (int)correspondance_rupture_current_sources_vector_x.size());
    return correspondance_rupture_current_sources_vector_x[val];
  }

  assert(0);
  return -1;
}
