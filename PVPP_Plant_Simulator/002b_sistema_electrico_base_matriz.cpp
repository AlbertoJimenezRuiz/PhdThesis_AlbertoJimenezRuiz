#include "electrical_system.h"
#include "main.h"
#include "matrix_functions.h"

void basic_electric_circuit::Matrix_Add_Equation_Impedance(
    Sparse_Matrix<FLOATING_TYPE> &A, int node_current_1, int node_current_2,
    int node_1, int node_2, FLOATING_TYPE val) const {

  if (node_current_1 != 0) {
    if (node_1 != 0) {
      A.write_add(obt_f(T_VOLTAGE_DC, node_current_1),
                  obt_f(T_VOLTAGE_DC, node_1), val);
    }

    if (node_2 != 0) {
      A.write_add(obt_f(T_VOLTAGE_DC, node_current_1),
                  obt_f(T_VOLTAGE_DC, node_2), -val);
    }
  }

  if (node_current_2 != 0) {
    if (node_1 != 0) {
      A.write_add(obt_f(T_VOLTAGE_DC, node_current_2),
                  obt_f(T_VOLTAGE_DC, node_1), -val);
    }

    if (node_2 != 0) {
      A.write_add(obt_f(T_VOLTAGE_DC, node_current_2),
                  obt_f(T_VOLTAGE_DC, node_2), val);
    }
  }
}

void basic_electric_circuit::Matrix_Add_Impedance(
    Sparse_Matrix<FLOATING_TYPE> &A, Sparse_Matrix<FLOATING_TYPE> & /*b*/,
    int node_1, int node_2, FLOATING_TYPE val) const {
  Matrix_Add_Equation_Impedance(A, node_1, node_2, node_1, node_2, val);
}

void basic_electric_circuit::add_submatrix_2x2_complex_impedance(
    Sparse_Matrix<FLOATING_TYPE> &A, int pos_x, int pos_y,
    std::complex<FLOATING_TYPE> val) const {

  if (pos_x == 0)
    return;
  if (pos_y == 0)
    return;

  pos_x = obt_f(T_VOLTAGE_DC, pos_x);
  pos_y = obt_f(T_VOLTAGE_DC, pos_y);

  A.write_add(pos_x, pos_y, val.real());
  A.write_add(pos_x, pos_y + 1, -val.imag());
  A.write_add(pos_x + 1, pos_y, val.imag());
  A.write_add(pos_x + 1, pos_y + 1, val.real());
}
void basic_electric_circuit::Matrix_Add_Impedance_ac(
    Sparse_Matrix<FLOATING_TYPE> &A, Sparse_Matrix<FLOATING_TYPE> & /*b*/,
    int node_1, int node_2, std::complex<FLOATING_TYPE> val) const {

  add_submatrix_2x2_complex_impedance(A, node_1, node_1, val);
  add_submatrix_2x2_complex_impedance(A, node_1, node_2, -val);

  add_submatrix_2x2_complex_impedance(A, node_2, node_1, -val);
  add_submatrix_2x2_complex_impedance(A, node_2, node_2, val);
}

void basic_electric_circuit::
    Matrix_Add_Current_Source_controlled_by_Voltage_half(
        Sparse_Matrix<FLOATING_TYPE> &A, int n_current, int node_1_voltage,
        int node_2_voltage, FLOATING_TYPE multiplier) const {

  if (node_1_voltage == 0) {
    A.write_add(obt_f(T_VOLTAGE_DC, n_current),
                obt_f(T_VOLTAGE_DC, node_1_voltage), multiplier);
  }
  if (node_2_voltage == 0) {
    A.write_add(obt_f(T_VOLTAGE_DC, n_current),
                obt_f(T_VOLTAGE_DC, node_2_voltage), -multiplier);
  }
}

void basic_electric_circuit::
    Matrix_Add_Current_Source_controlled_by_Voltage_half(
        Sparse_Matrix<FLOATING_TYPE> &A, Sparse_Matrix<FLOATING_TYPE> & /*b*/,
        int node_1_current, int node_2_current, int node_1_voltage,
        int node_2_voltage, FLOATING_TYPE multiplier) const {
  // Current source connected from node_1_current to node_2_current

  if (node_1_current != 0) {
    Matrix_Add_Current_Source_controlled_by_Voltage_half(
        A, node_1_current, node_1_voltage, node_2_voltage, multiplier);
  }
  if (node_2_current != 0) {
    Matrix_Add_Current_Source_controlled_by_Voltage_half(
        A, node_2_current, node_1_voltage, node_2_voltage, -multiplier);
  }
}

void basic_electric_circuit::Matrix_Add_Current_Source(
    Sparse_Matrix<FLOATING_TYPE> & /*A*/, Sparse_Matrix<FLOATING_TYPE> &b,
    int node_1, int node_2, FLOATING_TYPE val) const {
  if (node_1 != 0) {
    b.write_add(obt_f(T_VOLTAGE_DC, node_1), 0, +val);
  }

  if (node_2 != 0) {
    b.write_add(obt_f(T_VOLTAGE_DC, node_2), 0, -val);
  }
}

void basic_electric_circuit::Matrix_Add_Voltage_Source(
    Sparse_Matrix<FLOATING_TYPE> &A, Sparse_Matrix<FLOATING_TYPE> &b,
    int node_1, int node_2, FLOATING_TYPE val, int row_source) const {

  if (node_1 != 0) {
    A.write_add(obt_f(T_VOLTAGE_DC, node_1), row_source, -1);
    A.write_add(row_source, obt_f(T_VOLTAGE_DC, node_1), 1);
  }

  if (node_2 != 0) {
    A.write_add(obt_f(T_VOLTAGE_DC, node_2), row_source, 1);
    A.write_add(row_source, obt_f(T_VOLTAGE_DC, node_2), -1);
  }

  b.write_add(row_source, 0, +val);
}

void basic_electric_circuit::Matrix_Add_Current_Source_Unknown(
    Sparse_Matrix<FLOATING_TYPE> &A, int node_1, int node_2, int unknown,
    FLOATING_TYPE multiplier) const {
  if (node_1 != 0) {
    A.write_add(obt_f(T_VOLTAGE_DC, node_1), unknown, -multiplier);
  }

  if (node_2 != 0) {
    A.write_add(obt_f(T_VOLTAGE_DC, node_2), unknown, multiplier);
  }
}

tuple<Sparse_Matrix<FLOATING_TYPE>, Sparse_Matrix<FLOATING_TYPE>>
basic_electric_circuit::create_matrix_and_vector(
    FLOATING_TYPE virtual_conductances_to_facilitate_convergence) const {

  Sparse_Matrix<FLOATING_TYPE> A(vector_x_size, vector_x_size);
  Sparse_Matrix<FLOATING_TYPE> b(vector_x_size, 1);

  for (auto const &n : admittance_list) {
    Matrix_Add_Impedance(A, b, std::get<0>(n), std::get<1>(n), std::get<2>(n));
  }

  for (auto const &n : admittance_list_ac) {
    Matrix_Add_Impedance_ac(A, b, std::get<0>(n), std::get<1>(n),
                            std::get<2>(n));
  }

  for (auto const &n : list_current_sources) {
    Matrix_Add_Current_Source(A, b, std::get<0>(n), std::get<1>(n),
                              std::get<2>(n));
  }

  for (unsigned int i = 0; i < list_voltage_sources.size(); i++) {
    const auto &n = list_voltage_sources[i];
    Matrix_Add_Voltage_Source(A, b, std::get<0>(n), std::get<1>(n),
                              std::get<2>(n), obt_f(T_VOLTAGE_SOURCE_DC, i));
  }

  for (const auto &n : list_rupture_current_sources) {
    auto node_1 = std::get<0>(n);
    auto node_2 = std::get<1>(n);
    auto corresp = obt_f(T_CURRENT_SOURCE_RUPTURE_DC, std::get<2>(n));

    Matrix_Add_Current_Source_Unknown(A, node_1, node_2, corresp, 1.0);

    if (node_1 != 0) {
      A.write_add(corresp, obt_f(T_VOLTAGE_DC, node_1), -1);
    }
    if (node_2 != 0) {
      A.write_add(corresp, obt_f(T_VOLTAGE_DC, node_2), 1);
    }
  }

  for (const auto &n : list_nonlinear_current_sources) {

    auto node_1 = std::get<0>(n);
    auto node_2 = std::get<1>(n);
    auto func = std::get<2>(n);
    auto derivative_func = std::get<3>(n);
    auto value_V =
        vector_voltages_nonlinear_currentsources_iterate[std::get<4>(n)];

    auto value_admittance = derivative_func(value_V);
    auto value_current_source = func(value_V) - value_admittance * value_V;

    Matrix_Add_Impedance(A, b, node_1, node_2, value_admittance);
    Matrix_Add_Current_Source(A, b, node_1, node_2, -value_current_source);

    if (virtual_conductances_to_facilitate_convergence > 0) {
      Matrix_Add_Impedance(
          A, b, node_1, node_2,
          (FLOATING_TYPE)virtual_conductances_to_facilitate_convergence);
    }
  }

  for (const auto &n : list_inverters_fixed_voltage) {

    int node_input_ac_real = std::get<0>(n);
    int node_input_ac_imag =
        get_imaginary_node_from_real_AC(node_input_ac_real);

    int node_output_ac_real = std::get<1>(n);
    int node_output_ac_imag =
        get_imaginary_node_from_real_AC(node_output_ac_real);

    FLOATING_TYPE V_dc = std::get<3>(n);
    int number_this_inverter = std::get<4>(n);

    auto V2_real_iteration =
        std::get<0>(list_values_inverter_dc_ac_iterate[number_this_inverter]);
    auto V2_imag_iteration =
        std::get<1>(list_values_inverter_dc_ac_iterate[number_this_inverter]);
    auto valor_I1_iteration =
        std::get<2>(list_values_inverter_dc_ac_iterate[number_this_inverter]);

    const auto denominator = V2_real_iteration * V2_real_iteration +
                             V2_imag_iteration * V2_imag_iteration;

    const auto P_0 = V_dc * valor_I1_iteration;

    const auto I2_sust_Real = P_0 * V2_real_iteration / denominator;
    const auto I2_sust_Imag = P_0 * V2_imag_iteration / denominator;

    Matrix_Add_Current_Source(A, b, node_input_ac_real, node_output_ac_real,
                              -I2_sust_Real);
    Matrix_Add_Current_Source(A, b, node_input_ac_imag, node_output_ac_imag,
                              -I2_sust_Imag);
  }

  return tie(A, b);
}
