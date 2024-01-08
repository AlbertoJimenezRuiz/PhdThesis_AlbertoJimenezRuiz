#include "main.h"

class Linearized_Matrices_WT {
public:
  Linearized_Matrices_WT();
  Linearized_Matrices_WT(const wt_parameters_struct *param,
                         FLOATING_TYPE usd_current, FLOATING_TYPE usq_current,
                         FLOATING_TYPE desired_reactive_current,
                         FLOATING_TYPE uw_current);
  ~Linearized_Matrices_WT();

  Linearized_Matrices_WT &operator=(const Linearized_Matrices_WT &) = delete;

  Linearized_Matrices_WT &operator=(Linearized_Matrices_WT &&that) {
    parameters = that.parameters;

    size_matrix_A = that.size_matrix_A;
    size_matrix_B = that.size_matrix_B;
    size_matrix_C = that.size_matrix_C;
    size_matrix_D = that.size_matrix_D;
    size_matrix_X0 = that.size_matrix_X0;
    size_matrix_U0 = that.size_matrix_U0;
    size_matrix_Y0 = that.size_matrix_Y0;

    free(MatrA);
    free(MatrB);
    free(MatrC);
    free(MatrD);
    free(MatrX0);
    free(MatrU0);
    free(MatrY0);

    MatrA = that.MatrA;
    MatrB = that.MatrB;
    MatrC = that.MatrC;
    MatrD = that.MatrD;
    MatrX0 = that.MatrX0;
    MatrU0 = that.MatrU0;
    MatrY0 = that.MatrY0;

    that.MatrA = NULL;
    that.MatrB = NULL;
    that.MatrC = NULL;
    that.MatrD = NULL;
    that.MatrX0 = NULL;
    that.MatrU0 = NULL;
    that.MatrY0 = NULL;

    return *this;
  }

  FLOATING_TYPE *MatrA;
  FLOATING_TYPE *MatrB;
  FLOATING_TYPE *MatrC;
  FLOATING_TYPE *MatrD;
  FLOATING_TYPE *MatrU0;
  FLOATING_TYPE *MatrX0;
  FLOATING_TYPE *MatrY0;

  int size_matrix_A;
  int size_matrix_B;
  int size_matrix_C;
  int size_matrix_D;

  int size_matrix_U0;
  int size_matrix_X0;
  int size_matrix_Y0;

  const wt_parameters_struct *parameters;
};

Linearized_Matrices_WT *initialize_wt_voltage_in_grid(
    const wt_parameters_struct &parameters, int number_wt, FLOATING_TYPE v_d,
    FLOATING_TYPE v_q, FLOATING_TYPE desired_reactive, FLOATING_TYPE wind);
