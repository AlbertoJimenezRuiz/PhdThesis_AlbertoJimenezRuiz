#ifndef OBTAIN_ELECTRICAL_SYSTEM_H
#define OBTAIN_ELECTRICAL_SYSTEM_H

#include "sparse_matrix.h"

enum LIST_MATRIX_SOLVERS { SOLVER_LDL, SOLVER_INV, SOLVER_DIAKOPTICS_INV };

/* Matrix shape:

     Groups:

     A_1           D_1       x_1         b_1
         A_2       D_2       x_2    =    b_2
             A_3   D_3  *    x_3    =    b_3
     E_1 E_2 E_3   F_0       x_4         b_4
     */

class Matrix_Decomposer_Diakoptics {
public:
  std::vector<LONG_DOUBLE_C_Sparse_Matrix> PartA;
  std::vector<LONG_DOUBLE_C_Sparse_Matrix> PartD;
  std::vector<LONG_DOUBLE_C_Sparse_Matrix> PartE;

  LONG_DOUBLE_C_Sparse_Matrix PartF;
  std::vector<int> limit_inferior;
  std::vector<int> limit_superior;

  ~Matrix_Decomposer_Diakoptics() {}

  int number_unknowns;
  int number_groups;

  Matrix_Decomposer_Diakoptics(const LONG_DOUBLE_C_Sparse_Matrix &A);
};

template <typename T> class Solver_Matrix_Base {
public:
  virtual void solve(T *input_output) = 0;
  virtual ~Solver_Matrix_Base() {}
};

template <typename T>
class Solver_Matrix_Diakoptics : public Solver_Matrix_Base<T>,
                                 public Matrix_Decomposer_Diakoptics {
public:
  std::vector<T *> Matrices_A_inverse_dense;

  std::vector<Sparse_Matrix<T>> Decomposition_LU_Matrix_L;
  std::vector<std::vector<T>> Decomposition_LU_Diagonal;

  std::vector<T *> Matrices_D_dense;
  std::vector<T *> Matrices_E_dense;
  std::vector<int> group_sizes;
  std::vector<int> pointers_start;

  T *AD_vector;
  T *temporary_ABT;
  T *temporary_vector;

  T temporary_value_1_constant;

  ~Solver_Matrix_Diakoptics();

  Solver_Matrix_Diakoptics<T>(const LONG_DOUBLE_C_Sparse_Matrix &A);

  void solve(T *input_output);
};

template <typename T> class Solver_Matrix_LDL : public Solver_Matrix_Base<T> {
public:
  Sparse_Matrix<T> L_matrix;
  std::vector<T> D_matrix;
  int number_unknowns;

  ~Solver_Matrix_LDL();
  Solver_Matrix_LDL<T>(const LONG_DOUBLE_C_Sparse_Matrix &A);
  void solve(T *input_output);
};

template <typename T>
class Solver_Matrix_Inverse : public Solver_Matrix_Base<T> {
public:
  T *temporary_vector;
  int number_unknowns;
  T *A_inverse;

  ~Solver_Matrix_Inverse();
  Solver_Matrix_Inverse<T>(const LONG_DOUBLE_C_Sparse_Matrix &A);
  void solve(T *input_output);
};

struct description_cable {
  int cable_node_1, cable_node_2;
  FLOATING_TYPE R, L, C;
};

struct description_connections {
  std::vector<description_cable> cables;
  std::vector<int> wt_connection_points;

  FLOATING_TYPE grid_angular_speed;
  FLOATING_TYPE base_current_mt;
  FLOATING_TYPE base_impedance_mt;
  FLOATING_TYPE base_inductance_mt;
  FLOATING_TYPE base_capacitance_mt;
  int nodes_connection;
  ~description_connections() {}
};

description_connections
get_connections_list(const wt_parameters_struct &parameters,
                     const int number_wt, int &number_wt_strings,
                     int &length_wt_strings, int &remaining_wt);

// Without electrical transients
struct Static_Electrical_System {
  void create_matrices(const wt_parameters_struct &parameters,
                       int arg_number_wt);
  void calculate_voltages(const FLOATING_TYPE *const currents_injected,
                          FLOATING_TYPE *voltages_wt,
                          const C_FLOATING_TYPE fixed_voltage,
                          FLOATING_TYPE *copy_all_voltages);
  void destroy_matrices();
  ~Static_Electrical_System() {}

  LONG_DOUBLE_C_Sparse_Matrix L_reduced;
  C_LONG_DOUBLE *D_reduced;
  C_LONG_DOUBLE *A_swing_column;
  int number_wt;
  unsigned int m;
  description_connections connections;
};

C_FLOATING_TYPE create_static_admittance(const description_cable &connection,
                                         const FLOATING_TYPE base_impedance,
                                         const FLOATING_TYPE omega);

struct Dynamic_dq0_Electrical_System {
  void create_matrices(const wt_parameters_struct &parameters,
                       int arg_number_wt, const FLOATING_TYPE arg_delta_t,
                       const C_FLOATING_TYPE fixed_voltage,
                       const FLOATING_TYPE *const currents_injected_initial,
                       LIST_MATRIX_SOLVERS solver_type);
  void calculate_voltages(const FLOATING_TYPE *const currents_injected,
                          FLOATING_TYPE *voltages_wt,
                          FLOATING_TYPE *current_injected_to_grid,
                          const C_FLOATING_TYPE fixed_voltage);
  void destroy_matrices();
  Dynamic_dq0_Electrical_System() {}

  ~Dynamic_dq0_Electrical_System() {}

  FLOATING_TYPE delta_t;
  int number_wt;

  C_FLOATING_TYPE *V0_previous_buses;
  C_FLOATING_TYPE *V0_previous;
  C_FLOATING_TYPE *I0_previous;
  C_FLOATING_TYPE *extra_current_sources;

  C_Sparse_Matrix A_dynamic;
  C_VECTOR admittance_list;
  C_VECTOR I0_multiplier;
  C_VECTOR V0_multiplier;

  C_FLOATING_TYPE *currents_complex;
  C_FLOATING_TYPE *A_swing_column;

  LONG_DOUBLE_C_Sparse_Matrix A_reduced_LD;

  size_t number_connections;
  size_t number_buses;

  description_connections connections;

  Solver_Matrix_Base<C_FLOATING_TYPE> *solver;
};

#endif
