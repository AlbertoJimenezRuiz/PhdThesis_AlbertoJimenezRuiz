
#ifndef ELECTRICAL_SYSTEM_H
#define ELECTRICAL_SYSTEM_H

#include "main.h"
#include "sparse_matrix.h"

#include <complex>
#include <functional>

typedef std::function<FLOATING_TYPE(FLOATING_TYPE)> function_pointer_type;
typedef tuple<int, int, FLOATING_TYPE> value_device_and_node_number;
typedef tuple<int, int, complex<FLOATING_TYPE>> value_device_and_node_number_ac;
typedef tuple<int, int, function_pointer_type, function_pointer_type, int>
    personalized_admittance;
typedef tuple<int, int, int> position_virtual_current_source;
typedef tuple<int, int, int, FLOATING_TYPE, int>
    list_inverters_fixed_voltage_type;

enum ADD_ELEMENT_TYPE { ADMITTANCE, CURRENT_SOURCE, VOLTAGE_SOURCE };
enum MATRIX_ELEMENT_TYPE {
  T_VOLTAGE_DC,
  T_VOLTAGE_SOURCE_DC,
  T_CURRENT_SOURCE_RUPTURE_DC
};

class basic_electric_circuit {
public:
  basic_electric_circuit();
  ~basic_electric_circuit() {}

  int create_node();
  int create_node_ac();
  int get_imaginary_node_from_real_AC(int node) const;

  FLOATING_TYPE voltage_drop_dc(const vector<FLOATING_TYPE> &solution, int n1,
                                int n2) const;
  std::complex<FLOATING_TYPE>
  voltage_drop_ac(const vector<FLOATING_TYPE> &solution, int n1, int n2) const;

  void add_rupture_current_source(int node_1, int node_2);
  int add_linear_element(ADD_ELEMENT_TYPE type, FLOATING_TYPE value_element,
                         int node_1, int node_2);
  int add_linear_element_ac(ADD_ELEMENT_TYPE type,
                            complex<FLOATING_TYPE> value_element, int node_1,
                            int node_2);

  void add_nonlinear_voltage_source(function_pointer_type func,
                                    function_pointer_type derivative_func,
                                    FLOATING_TYPE V_initial, int node_input_old,
                                    int node_output_old);

  void create_inverter_dc_ac(FLOATING_TYPE V_dc,
                             std::complex<FLOATING_TYPE> V_initial_ac,
                             int node_1_dc, int node_2_dc, int node_1_ac_real,
                             int node_2_ac_real);

  int obt_f(const MATRIX_ELEMENT_TYPE tipo_variable, const int val) const;
  tuple<Sparse_Matrix<FLOATING_TYPE>, Sparse_Matrix<FLOATING_TYPE>>
  create_matrix_and_vector(
      FLOATING_TYPE virtual_conductances_to_facilitate_convergence = -1) const;
  void Matrix_Add_Impedance(Sparse_Matrix<FLOATING_TYPE> &A,
                            Sparse_Matrix<FLOATING_TYPE> &b, int node_1,
                            int node_2, FLOATING_TYPE val) const;
  void Matrix_Add_Impedance_ac(Sparse_Matrix<FLOATING_TYPE> &A,
                               Sparse_Matrix<FLOATING_TYPE> & /*b*/, int node_1,
                               int node_2,
                               std::complex<FLOATING_TYPE> val) const;
  void Matrix_Add_Equation_Impedance(Sparse_Matrix<FLOATING_TYPE> &A,
                                     int node_current_1, int node_current_2,
                                     int node_1, int node_2,
                                     FLOATING_TYPE val) const;

  void
  add_submatrix_2x2_complex_impedance(Sparse_Matrix<FLOATING_TYPE> &A,
                                      int node_1, int node_2,
                                      std::complex<FLOATING_TYPE> val) const;

  void Matrix_Add_Current_Source(Sparse_Matrix<FLOATING_TYPE> &A,
                                 Sparse_Matrix<FLOATING_TYPE> &b, int node_1,
                                 int node_2, FLOATING_TYPE val) const;
  void Matrix_Add_Voltage_Source(Sparse_Matrix<FLOATING_TYPE> &A,
                                 Sparse_Matrix<FLOATING_TYPE> &b, int node_1,
                                 int node_2, FLOATING_TYPE val,
                                 int row_source) const;
  void Matrix_Add_Current_Source_Unknown(Sparse_Matrix<FLOATING_TYPE> &A,
                                         int node_1, int node_2, int unknown,
                                         FLOATING_TYPE multiplier) const;

  void Matrix_Add_Current_Source_controlled_by_Voltage_half(
      Sparse_Matrix<FLOATING_TYPE> &A, int n_current, int node_1_voltage,
      int node_2_voltage, FLOATING_TYPE multiplier) const;

  void Matrix_Add_Current_Source_controlled_by_Voltage_half(
      Sparse_Matrix<FLOATING_TYPE> &A, Sparse_Matrix<FLOATING_TYPE> & /*b*/,
      int node_1_current, int node_2_current, int node_1_voltage,
      int node_2_voltage, FLOATING_TYPE multiplier) const;

  vector<personalized_admittance> list_nonlinear_current_sources;
  vector<value_device_and_node_number> admittance_list;
  vector<value_device_and_node_number_ac> admittance_list_ac;
  vector<value_device_and_node_number> list_current_sources;
  vector<value_device_and_node_number> list_voltage_sources;
  vector<position_virtual_current_source> list_rupture_current_sources;
  vector<list_inverters_fixed_voltage_type> list_inverters_fixed_voltage;
  vector<FLOATING_TYPE> vector_voltages_nonlinear_currentsources_iterate;
  vector<tuple<FLOATING_TYPE, FLOATING_TYPE, FLOATING_TYPE>>
      list_values_inverter_dc_ac_iterate;

private:
  int vector_x_size;
  vector<int> correspondance_nodes_voltage_vector_x;
  vector<int> correspondance_voltage_sources_vector_x;
  vector<int> correspondance_rupture_current_sources_vector_x;
};

#endif
