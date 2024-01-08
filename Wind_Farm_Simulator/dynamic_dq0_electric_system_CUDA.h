#ifndef DYNAMIC_DQ0_ELECTRIC_SYSTEM_CUDA
#define DYNAMIC_DQ0_ELECTRIC_SYSTEM_CUDA

#include "gpu_util.h"
#include "obtain_electrical_system.h"

class Dynamic_dq0_Electrical_System_CUDA {
public:
  Dynamic_dq0_Electrical_System_CUDA() {}

  void create_matrices(const wt_parameters_struct &parameters, int arg_wt,
                       const FLOATING_TYPE arg_delta_t,
                       const C_FLOATING_TYPE fixed_voltage,
                       const FLOATING_TYPE *const currents_injected_initial,
                       LIST_MATRIX_SOLVERS solver);

  void calculate_voltages(const FLOATING_TYPE *const currents_injected,
                          FLOATING_TYPE *voltages_wt,
                          FLOATING_TYPE *inject_current_to_grid,
                          const C_FLOATING_TYPE fixed_voltage);

  void destroy_matrices();

  Dynamic_dq0_Electrical_System *cpu;

  C_LONG_DOUBLE temporary_value_1_constant_high_precision;

  int number_required_threads;
  int max_elements_stage1;
  int max_elements_stage2;
  int max_elements_stage3;

  GPU_Array_complex *multipliers_stage1_unified_CUDA;
  GPU_Array_int *positions_stage1_unified_CUDA;
  GPU_Array_int *element_size_stage1_unified_CUDA;
  GPU_Array_int *number_elements_stage1_unified_CUDA;
  GPU_Array_int *corresponding_element_stage1_unified_CUDA;

  GPU_Array_complex *multipliers_stage2_unified_CUDA;
  GPU_Array_int *positions_stage2_unified_CUDA;
  GPU_Array_int *element_size_stage2_unified_CUDA;
  GPU_Array_int *number_elements_stage2_unified_CUDA;
  GPU_Array_int *corresponding_element_stage2_unified_CUDA;

  GPU_Array_complex *multipliers_stage3_unified_CUDA;
  GPU_Array_complex *multipliers_stage4_unified_CUDA;

  GPU_Array_complex *large_vector_inputs_no_padding_CUDA;

  GPU_Array_complex *temporary_vector_CUDA_1;
  GPU_Array_complex *temporary_vector_CUDA_2;
  GPU_Array_complex *temporary_vector_CUDA_3;
  GPU_Array_complex *temporary_vector_CUDA_4;

  GPU_Array_int *line_node_input_CUDA;
  GPU_Array_int *line_node_output_CUDA;
  GPU_Array_complex *I0_multiplier_CUDA;
  GPU_Array_complex *V0_multiplier_CUDA;
  GPU_Array_complex *admittance_list_CUDA;

  GPU_Array_complex *first_column_CUDA;
  GPU_Array_int *first_column_pos_CUDA;

  GPU_Array_float *injected_current_CUDA;
  GPU_Array_float *voltages_wt_CUDA;
  GPU_Array_int *wt_connection_points_CUDA;

  int first_column_size;
};

#endif
