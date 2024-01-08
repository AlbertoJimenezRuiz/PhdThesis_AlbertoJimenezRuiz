__device__ cuComplexType get_V0_previous_buses(
    int bus_number,
    cuComplexType *inputs_system, // fixed voltage

    cuComplexType *temporary_vector_2, // Reduction to obtain the value of x
                                       // before substracting the union
    cuComplexType
        *temporary_vector_3, // Reduction to obtain the first value of the union
    cuComplexType *ad_multiplier, cuComplexType temporary_value_1_constant,
    int number_buses) {

  cuComplexType u_constant = MULTIPLY_COMPLEX_CUDA(
      SUBTRACT_COMPLEX_CUDA(temporary_vector_2[number_buses - 2],
                            temporary_vector_3[0]),
      temporary_value_1_constant);

  if (bus_number == number_buses - 1) {
    return u_constant;
  } else if (bus_number == 0) {
    return inputs_system[0];
  } else {
    return SUBTRACT_COMPLEX_CUDA(
        temporary_vector_2[bus_number - 1],
        MULTIPLY_COMPLEX_CUDA(ad_multiplier[bus_number - 1], u_constant));
  }
}

#define NUMBER_INPUT_COMPONENTS_CALCULATED 2

#define SUBEXPRESSION_MEMORY_ADDRESS_INTEGRATE_WT(CHUNK)                       \
  (current_wt * (NUMBER_U_VALUES - NUMBER_INPUT_COMPONENTS_CALCULATED) *       \
       (number_timesteps + 1) +                                                \
   (NUMBER_U_VALUES - NUMBER_INPUT_COMPONENTS_CALCULATED) * (CHUNK)) +         \
      (current_state - NUMBER_INPUT_COMPONENTS_CALCULATED)

__global__
__launch_bounds__(ELECTRICAL_SYSTEM_BLOCKSIZE,
                  32) // 32 is the number of blocks per multiprocessor
    void unified_wt_simulator_kernel(
        cuComplexType *input_slack_voltage, cuComplexType *unified_input,
        cuComplexType *input_multipliers_stage1,
        cuComplexType *input_multipliers_stage2a,
        cuComplexType *input_multipliers_stage2b,
        cuComplexType *temporary_vector_1, cuComplexType *temporary_vector_2,
        cuComplexType *temporary_vector_3, int number_elements_max_stage1,
        int number_elements_max_stage2a, int number_elements_max_stage2b,

        int *unified_positions_stage1, int *element_size_stage1,
        int *number_elements_stage1, int *corresponding_element_vect_stage1,

        int *unified_positions_stage2a, int *element_size_stage2a,
        int *number_elements_stage2a, int *corresponding_element_vect_stage2a,

        int *line_node_input, int *line_node_output,

        cuComplexType *I0_multiplier, cuComplexType *V0_multiplier,
        cuComplexType *admittance_list,

        cuComplexType *first_column, int *first_column_pos,
        int first_column_size,

        int *wt_connection_points, FLOATING_TYPE *global_injected_current,
        FLOATING_TYPE *voltages_wt, cuComplexType *ad_multiplier,
        cuComplexType temporary_value_1_constant,

        int blocks_stage2a, int blocks_number_connections,

        int number_wt, int number_buses, int number_connections,
        int number_blocks_wt,

        int number_timesteps,

        FLOATING_TYPE timestep,

        FLOATING_TYPE *__restrict__ MatrU0, FLOATING_TYPE *__restrict__ MatrA,
        FLOATING_TYPE *__restrict__ MatrB, FLOATING_TYPE *__restrict__ MatrC,
        FLOATING_TYPE *__restrict__ MatrD, FLOATING_TYPE *__restrict__ dY0,
        FLOATING_TYPE *__restrict__ dU, volatile FLOATING_TYPE *__restrict__ dX,

        cuComplexType *__restrict__ dVoltages,
        FLOATING_TYPE *__restrict__ dOutputs,
        cuComplexType *__restrict__ dCurrents_Wind_Farm_Global

    ) {

  cooperative_groups::grid_group g = cooperative_groups::this_grid();

  __shared__ cuComplexType shared_memory[ELECTRICAL_SYSTEM_BLOCKSIZE];

  const int tid_stage1_stage2a_stage3a = blockIdx.x * blockDim.x + threadIdx.x;
  const int tid_stage2b =
      (blockIdx.x - blocks_stage2a) * blockDim.x + threadIdx.x;

  int current_element_size_stage1;
  int number_element_current_stage1;
  int corresponding_element_stage1;
  int position_current_stage1;
  cuComplexType input_current_multipliers_stage1;

  int current_element_size_stage2a;
  int number_element_current_stage2a;
  int corresponding_element_stage2a;
  int position_current_stage2a;
  cuComplexType input_current_multipliers_stage2a;

  cuComplexType input_current_multipliers_stage2b;

  cuComplexType I0_mult1 = I0_multiplier[tid_stage1_stage2a_stage3a];
  cuComplexType I0_mult2 =
      ADD_COMPLEX_CUDA(V0_multiplier[tid_stage1_stage2a_stage3a],
                       admittance_list[tid_stage1_stage2a_stage3a]);

  if (tid_stage1_stage2a_stage3a < number_elements_max_stage1) {
    current_element_size_stage1 =
        element_size_stage1[tid_stage1_stage2a_stage3a];
    number_element_current_stage1 =
        number_elements_stage1[tid_stage1_stage2a_stage3a];
    corresponding_element_stage1 =
        corresponding_element_vect_stage1[tid_stage1_stage2a_stage3a];
    position_current_stage1 =
        unified_positions_stage1[tid_stage1_stage2a_stage3a];
    input_current_multipliers_stage1 =
        input_multipliers_stage1[tid_stage1_stage2a_stage3a];
  }

  if (tid_stage1_stage2a_stage3a < number_elements_max_stage2a) {

    current_element_size_stage2a =
        element_size_stage2a[tid_stage1_stage2a_stage3a];
    number_element_current_stage2a =
        number_elements_stage2a[tid_stage1_stage2a_stage3a];
    corresponding_element_stage2a =
        corresponding_element_vect_stage2a[tid_stage1_stage2a_stage3a];
    position_current_stage2a =
        unified_positions_stage2a[tid_stage1_stage2a_stage3a];
    input_current_multipliers_stage2a =
        input_multipliers_stage2a[tid_stage1_stage2a_stage3a];
  }

  if (tid_stage2b < number_elements_max_stage2b) {
    input_current_multipliers_stage2b = input_multipliers_stage2b[tid_stage2b];
  }

  for (unsigned int current_timestep = 0; current_timestep < number_timesteps;
       current_timestep++) // main loop
  {
    unified_input[0] = input_slack_voltage[current_timestep];
    if (tid_stage1_stage2a_stage3a == 0) {
      temporary_vector_3[0] = CREATE_COMPLEX_CUDA(0, 0);
    }

    g.sync();
    {
#include "kernel_stage_1.cuh"
    }

    g.sync();

    {
      if (blockIdx.x < blocks_stage2a) {
#include "kernel_stage_2a.cuh"

      } else {
#include "kernel_stage_2b.cuh"
      }
    }

    g.sync();
    {
      const int tid_global = blockIdx.x * blockDim.x + threadIdx.x;

      if (tid_global < number_buses) {
        temporary_vector_1[tid_global] = CREATE_COMPLEX_CUDA(0, 0);
      }

      if (blockIdx.x < blocks_number_connections) {
#include "kernel_stage_3a.cuh"
      } else if (blockIdx.x == blocks_number_connections) {
#include "kernel_stage_3b.cuh"
      } else {
#include "integrate_standalone_wt_CUDA.cuh"
      }
    }
    g.sync();
  }
}
