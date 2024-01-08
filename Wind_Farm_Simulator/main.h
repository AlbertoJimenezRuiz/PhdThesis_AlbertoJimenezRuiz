#ifndef MAIN_H
#define MAIN_H

#include <assert.h>
#include <chrono>
#include <complex>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <string.h>
#include <tuple>
#include <vector>

using namespace std;

#define MAXIMUM_STRING_SIZE 10000
#define OUTPUT_PATH "./bin"

#define NUMBER_Y_EQUATIONS 5
#define NUMBER_U_VALUES 4
#define NUMBER_X_EQUATIONS 11

#define ELECTRICAL_SYSTEM_BLOCKSIZE 512

#define DIVISION_UP(n, d) (((n) + (d)-1) / (d))

#define MIN2(a, b) ((a) < (b) ? (a) : (b))
#define MAX2(a, b) ((a) > (b) ? (a) : (b))
#define MAX3(a, b, c) MAX2(a, MAX2(b, c))
#define MAX4(a, b, c, d) MAX2(d, MAX3(a, b, c))

#define WIND_TURBINES_PER_BLOCK 32

// defined (or not) in makefile
#ifdef USE_DOUBLE
typedef double FLOATING_TYPE;
#define funcion_abs abs
#else
typedef float FLOATING_TYPE;
#define funcion_abs fabs
#endif

struct wt_parameters_struct {
  wt_parameters_struct(const wt_parameters_struct &) = delete;

  wt_parameters_struct();
  ~wt_parameters_struct() { free(Input_Data); }

  unsigned int simulation_seconds;
  unsigned int timesteps_per_second;

  double Rs;
  double Rr;
  double Llr;
  double Lls;
  double Lm;
  double Friction_factor;
  double Hr;
  double Ht;
  double K_2mass;
  unsigned int Number_poles;
  double Kp_control;
  double Ki_control;
  double Kopt;
  double Blade_length;
  double Nominal_mechanical_speed_RPM;
  double Base_Power;
  double Multiplier;
  double Omega_s;
  double Base_Electrical_angular_speed;
  double Base_Voltage;
  double GSC_pole_filter;
  double Ycr;

  double Base_Voltage_mt;
  double cable_R;
  double cable_L;
  double cable_C;
  double Impedance_percent_transformer;
  double Power_transformer;

  unsigned int number_datapoints;
  double timestep;
  FLOATING_TYPE *Input_Data;
};

void __attribute__((format(printf, 1, 0))) abort_msg(const char *format, ...);

typedef std::complex<FLOATING_TYPE> C_FLOATING_TYPE;
typedef std::vector<FLOATING_TYPE> R_VECTOR;
typedef std::vector<C_FLOATING_TYPE> C_VECTOR;
typedef std::complex<long double> C_LONG_DOUBLE;

// util.cpp
void read_matrix_csv(const char *path, unsigned int *rows,
                     unsigned int *columns, FLOATING_TYPE **data);
void write_csv_fullpath(const char *path, const unsigned int rows,
                        const unsigned int columns, const FLOATING_TYPE *data);
void write_all_wt_results_csv(const wt_parameters_struct &params,
                              unsigned int number_wind_turbines,
                              FLOATING_TYPE *output_data,
                              unsigned int output_variables, const char *msg);

// measure time
class TimeMeasurer_us {
public:
  TimeMeasurer_us();
  long int measure(const char *msg);
  std::chrono::high_resolution_clock::time_point start;
};

// gpu_util
void check_memory_allocation_CUDA_possible(
    const struct wt_parameters_struct &parameters,
    unsigned int number_wind_turbines, unsigned int matrix_size_X);

// CUDA additional functions

void organize_data_vector_reduction_CUDA(
    bool groups_synchronized_using_cooperative_groups,
    const vector<vector<C_FLOATING_TYPE>> &input_full_groups_multipliers,
    const vector<vector<int>> &input_full_groups_positions,

    std::vector<C_FLOATING_TYPE> &multipliers, std::vector<int> &positions,
    std::vector<int> &element_sizes, std::vector<int> &number_elements,
    std::vector<int> &corresponding_element,

    const unsigned int blocksize);

#endif
