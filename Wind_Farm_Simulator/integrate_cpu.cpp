#include "main.h"
#include <assert.h>

#include "linearized_wt_matrices.h"
#include "obtain_electrical_system.h"

// Matrix-vector product VY= A * VX
template <int rows, int columns>
static inline void matrix_vector_multiply(const FLOATING_TYPE *__restrict A,
                                          const FLOATING_TYPE *__restrict VX,
                                          FLOATING_TYPE *__restrict VY) {
  for (unsigned int counter = 0; counter < rows; counter++) {
    for (unsigned int counter2 = 0; counter2 < columns; counter2++) {
      VY[counter] += A[counter * columns + counter2] * VX[counter2];
    }
  }
}

// Integrate linearized ODE using Heun's method
template <int size_matrizx>
static void integrate_linearized_ODE(
    const FLOATING_TYPE *__restrict Local_MatrA,
    const FLOATING_TYPE *__restrict Local_MatrB,
    const FLOATING_TYPE *__restrict Local_MatrC,
    const FLOATING_TYPE *__restrict Local_MatrD,
    const FLOATING_TYPE *__restrict equilibrium_pointU,
    const FLOATING_TYPE *__restrict equilibrium_pointY,

    const FLOATING_TYPE timestep, const unsigned int number_datapoints,
    const unsigned int current_busbar,

    FLOATING_TYPE *__restrict Y_Output,

    /* temporary data*/
    FLOATING_TYPE *Local_U_voltage_dq, FLOATING_TYPE *Local_U_no_voltage,

    FLOATING_TYPE *__restrict XEuler, FLOATING_TYPE *__restrict XHeun,
    FLOATING_TYPE *__restrict XPrime, FLOATING_TYPE *__restrict XIteration,
    FLOATING_TYPE *__restrict XFinal) {

  const long unsigned int size_matrix_x_bytes =
      sizeof(FLOATING_TYPE) * size_matrizx;

  const FLOATING_TYPE U_current[NUMBER_U_VALUES] = {
      Local_U_voltage_dq[0] - equilibrium_pointU[0],
      Local_U_voltage_dq[1] - equilibrium_pointU[1],
      Local_U_no_voltage[2] - equilibrium_pointU[2],
      Local_U_no_voltage[3] - equilibrium_pointU[3]};

  const FLOATING_TYPE U_next[NUMBER_U_VALUES] = {
      Local_U_voltage_dq[0] - equilibrium_pointU[0],
      Local_U_voltage_dq[1] - equilibrium_pointU[1],
      Local_U_no_voltage[(current_busbar == number_datapoints - 1)
                             ? 2
                             : NUMBER_U_VALUES + 2] -
          equilibrium_pointU[2],
      Local_U_no_voltage[(current_busbar == number_datapoints - 1)
                             ? 3
                             : NUMBER_U_VALUES + 3] -
          equilibrium_pointU[3]};

  // It is assumed that Y_Output is zeroed
  matrix_vector_multiply<NUMBER_Y_EQUATIONS, size_matrizx>(
      Local_MatrC, XIteration, Y_Output);
  matrix_vector_multiply<NUMBER_Y_EQUATIONS, NUMBER_U_VALUES>(
      Local_MatrD, U_current, Y_Output);

  for (int j = 0; j < NUMBER_Y_EQUATIONS; j++)
    Y_Output[j] += equilibrium_pointY[j];

  memset(XPrime, 0, size_matrix_x_bytes);
  memset(XHeun, 0, size_matrix_x_bytes);

  // The result of matrix_vector_multiply is added to the existing one
  matrix_vector_multiply<size_matrizx, size_matrizx>(Local_MatrA, XIteration,
                                                     XPrime);
  matrix_vector_multiply<size_matrizx, NUMBER_U_VALUES>(Local_MatrB, U_current,
                                                        XPrime);

  for (int counter = 0; counter < size_matrizx; counter++)
    XEuler[counter] = XIteration[counter] + timestep * XPrime[counter];
  matrix_vector_multiply<size_matrizx, size_matrizx>(Local_MatrA, XEuler,
                                                     XHeun);
  matrix_vector_multiply<size_matrizx, NUMBER_U_VALUES>(Local_MatrB, U_next,
                                                        XHeun);

  for (int counter = 0; counter < size_matrizx; counter++)
    XFinal[counter] =
        XIteration[counter] + timestep / 2 * (XHeun[counter] + XPrime[counter]);

  memcpy(XIteration, XFinal, size_matrix_x_bytes);
}

#define CALL_CPU_INTEGRATOR_IF(tamX)                                           \
  else if (tamX == NUMBER_X_EQUATIONS) integrate_linearized_ODE<tamX>(         \
      mat[i].MatrA, mat[i].MatrB, mat[i].MatrC, mat[i].MatrD, mat[i].MatrU0,   \
      mat[i].MatrY0, (FLOATING_TYPE)parameters.timestep,                       \
      parameters.number_datapoints, n,                                         \
      &MatrY_CPU[i * NUMBER_Y_EQUATIONS * parameters.number_datapoints +       \
                 n * NUMBER_Y_EQUATIONS],                                      \
      &V0_electrical_system[2 * i],                                            \
      &parameters.Input_Data[n * NUMBER_U_VALUES], &XEuler[i * tamX],          \
      &XHeun[i * tamX], &XPrime[i * tamX], &XIteration[i * tamX],              \
      &XFinal[i * tamX]);

unsigned long Wind_Farm_Test_CPU(struct wt_parameters_struct &parameters,
                                 const int number_wt) {

  FLOATING_TYPE *MatrY_CPU =
      (FLOATING_TYPE *)malloc(sizeof(FLOATING_TYPE) * NUMBER_Y_EQUATIONS *
                              parameters.number_datapoints * number_wt);
  FLOATING_TYPE *global_current_injected_to_grid = (FLOATING_TYPE *)malloc(
      sizeof(FLOATING_TYPE) * 2 * parameters.number_datapoints);

  C_FLOATING_TYPE fixed_voltage(parameters.Input_Data[0],
                                parameters.Input_Data[1]);

  Linearized_Matrices_WT *mat = initialize_wt_voltage_in_grid(
      parameters, number_wt, parameters.Input_Data[0], parameters.Input_Data[1],
      parameters.Input_Data[2], parameters.Input_Data[3]);

  FLOATING_TYPE *XEuler = (FLOATING_TYPE *)malloc(
      sizeof(FLOATING_TYPE) * number_wt * NUMBER_X_EQUATIONS);
  FLOATING_TYPE *XHeun = (FLOATING_TYPE *)malloc(
      sizeof(FLOATING_TYPE) * number_wt * NUMBER_X_EQUATIONS);
  FLOATING_TYPE *XPrime = (FLOATING_TYPE *)malloc(
      sizeof(FLOATING_TYPE) * number_wt * NUMBER_X_EQUATIONS);
  FLOATING_TYPE *XIteration = (FLOATING_TYPE *)malloc(
      sizeof(FLOATING_TYPE) * number_wt * NUMBER_X_EQUATIONS);
  FLOATING_TYPE *XFinal = (FLOATING_TYPE *)malloc(
      sizeof(FLOATING_TYPE) * number_wt * NUMBER_X_EQUATIONS);

  memset(MatrY_CPU, 0,
         sizeof(FLOATING_TYPE) * NUMBER_Y_EQUATIONS *
             parameters.number_datapoints * number_wt);

  FLOATING_TYPE *I0_electrical_system = new FLOATING_TYPE[2 * number_wt];
  FLOATING_TYPE *V0_electrical_system = new FLOATING_TYPE[2 * number_wt];

  for (int i = 0; i < (int)number_wt; i++) {
    I0_electrical_system[2 * i + 0] = mat[i].MatrY0[0];
    I0_electrical_system[2 * i + 1] = mat[i].MatrY0[1];
  }

  Dynamic_dq0_Electrical_System syst;

  syst.create_matrices(parameters, number_wt,
                       (FLOATING_TYPE)parameters.timestep, fixed_voltage,
                       I0_electrical_system, SOLVER_LDL);

#pragma omp parallel for
  for (int i = 0; i < (int)number_wt; i++) {
    for (int counter = 0; counter < NUMBER_X_EQUATIONS; counter++) {
      XIteration[i * NUMBER_X_EQUATIONS + counter] =
          mat[i].MatrX0[counter] - mat[i].MatrX0[counter];
    }
    V0_electrical_system[2 * i + 0] = mat[i].MatrU0[0];
    V0_electrical_system[2 * i + 1] = mat[i].MatrU0[1];
  }

  TimeMeasurer_us time_measurer; // Starts to measure
  for (unsigned int n = 0; n < parameters.number_datapoints; n++) {
    for (int i = 0; i < (int)number_wt; i++) {

      if (0) {
      }
      CALL_CPU_INTEGRATOR_IF(11)
      else {
        abort_msg("Weird number of states X:%d \n", NUMBER_X_EQUATIONS);
      }
    }

    {
      for (int i = 0; i < (int)number_wt; i++) {
        I0_electrical_system[2 * i + 0] =
            MatrY_CPU[i * (NUMBER_Y_EQUATIONS)*parameters.number_datapoints +
                      n * (NUMBER_Y_EQUATIONS) + 0];
        I0_electrical_system[2 * i + 1] =
            MatrY_CPU[i * (NUMBER_Y_EQUATIONS)*parameters.number_datapoints +
                      n * (NUMBER_Y_EQUATIONS) + 1];
      }

      FLOATING_TYPE *pointer_global_current_injected_to_grid = NULL;
      if (global_current_injected_to_grid) {
        pointer_global_current_injected_to_grid =
            &global_current_injected_to_grid[2 * n];
      }

      syst.calculate_voltages(
          I0_electrical_system, V0_electrical_system,
          pointer_global_current_injected_to_grid,
          C_FLOATING_TYPE(parameters.Input_Data[n * NUMBER_U_VALUES + 0],
                          parameters.Input_Data[n * NUMBER_U_VALUES + 1]));
    }
  }

  unsigned long measured_time =
      time_measurer.measure("CPU"); // End of time measurement

  write_all_wt_results_csv(parameters, number_wt, MatrY_CPU, NUMBER_Y_EQUATIONS,
                           "CPU");

  if (global_current_injected_to_grid)
    write_csv_fullpath("I_Total_CPU", parameters.number_datapoints, 2,
                       global_current_injected_to_grid);

  free(XEuler);
  free(XHeun);
  free(XPrime);
  free(XIteration);
  free(XFinal);

  delete[] mat;
  delete[] I0_electrical_system;
  delete[] V0_electrical_system;

  syst.destroy_matrices();

  free(MatrY_CPU);
  free(global_current_injected_to_grid);

  return measured_time;
}
