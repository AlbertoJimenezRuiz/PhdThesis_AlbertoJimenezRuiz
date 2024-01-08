#include "dynamic_dq0_electric_system_CUDA.h"
#include "gpu_util.h"
#include "linearized_wt_matrices.h"
#include "main.h"
#include "obtain_electrical_system.h"
#include <unistd.h>

#include <cooperative_groups.h>
#include "unified_wt_kernel_CUDA.cuh"


// Reorganize output of the CUDA kernel so as to be written into CSV files
void reorder_CUDA_wt_output(const wt_parameters_struct &parameters,
                            unsigned int number_wt, FLOATING_TYPE *Pinned_Y,
                            FLOATING_TYPE *Matry_GPU);

unsigned long Wind_Farm_Test_CUDA(wt_parameters_struct &parameters,
                                  const int number_wt) {
  gpuErrchk(cudaSetDevice(0)); // Select CUDA Graphic Card
  gpuErrchk(
      cudaFree(0)); // Dummy call to let the driver perform its initialization
                    // here and not have strange results in the profiler.

  C_FLOATING_TYPE fixed_voltage(parameters.Input_Data[0],
                                parameters.Input_Data[1]);

  Linearized_Matrices_WT *mat = initialize_wt_voltage_in_grid(
      parameters, number_wt, parameters.Input_Data[0], parameters.Input_Data[1],
      parameters.Input_Data[2], parameters.Input_Data[3]);

  FLOATING_TYPE *I0_electrical_system_start = new FLOATING_TYPE[2 * number_wt];

  for (int i = 0; i < number_wt; i++) {
    I0_electrical_system_start[2 * i + 0] = mat[i].MatrY0[0];
    I0_electrical_system_start[2 * i + 1] = mat[i].MatrY0[1];
  }

  Dynamic_dq0_Electrical_System_CUDA syst;
  syst.create_matrices(parameters, number_wt,
                       (FLOATING_TYPE)parameters.timestep, fixed_voltage,
                       I0_electrical_system_start, SOLVER_DIAKOPTICS_INV);

  GPU_Array_float *dA =
      new GPU_Array_float(number_wt * NUMBER_X_EQUATIONS * NUMBER_X_EQUATIONS);
  GPU_Array_float *dB =
      new GPU_Array_float(number_wt * NUMBER_X_EQUATIONS * NUMBER_U_VALUES);
  GPU_Array_float *dC =
      new GPU_Array_float(number_wt * NUMBER_Y_EQUATIONS * NUMBER_X_EQUATIONS);
  GPU_Array_float *dD =
      new GPU_Array_float(number_wt * NUMBER_Y_EQUATIONS * NUMBER_U_VALUES);
  GPU_Array_float *dX0 = new GPU_Array_float(number_wt * NUMBER_X_EQUATIONS);
  GPU_Array_float *dU0 = new GPU_Array_float(number_wt * NUMBER_U_VALUES);
  GPU_Array_float *dY0 = new GPU_Array_float(number_wt * NUMBER_Y_EQUATIONS);

  for (int i = 0; i < number_wt; i++) {
    dA->copy_HtoD_range(mat[i].MatrA,
                        i * NUMBER_X_EQUATIONS * NUMBER_X_EQUATIONS,
                        NUMBER_X_EQUATIONS * NUMBER_X_EQUATIONS);
    dB->copy_HtoD_range(mat[i].MatrB, i * NUMBER_X_EQUATIONS * NUMBER_U_VALUES,
                        NUMBER_X_EQUATIONS * NUMBER_U_VALUES);
    dC->copy_HtoD_range(mat[i].MatrC,
                        i * NUMBER_Y_EQUATIONS * NUMBER_X_EQUATIONS,
                        NUMBER_Y_EQUATIONS * NUMBER_X_EQUATIONS);
    dD->copy_HtoD_range(mat[i].MatrD, i * NUMBER_Y_EQUATIONS * NUMBER_U_VALUES,
                        NUMBER_Y_EQUATIONS * NUMBER_U_VALUES);
    dX0->copy_HtoD_range(mat[i].MatrX0, i * NUMBER_X_EQUATIONS,
                         NUMBER_X_EQUATIONS);
    dU0->copy_HtoD_range(mat[i].MatrU0, i * NUMBER_U_VALUES, NUMBER_U_VALUES);
    dY0->copy_HtoD_range(mat[i].MatrY0, i * NUMBER_Y_EQUATIONS,
                         NUMBER_Y_EQUATIONS);
  }

  FLOATING_TYPE *Pinned_Y;
  FLOATING_TYPE *Pinned_U;

  const int number_values_inputs_which_arent_in_input_csv = 2;

  check_memory_allocation_CUDA_possible(parameters, number_wt,
                                        NUMBER_X_EQUATIONS);

  long long int alloc_size_u =
      ((NUMBER_U_VALUES - number_values_inputs_which_arent_in_input_csv) *
       number_wt) *
      (parameters.number_datapoints + 1);
  long long int alloc_size_y = DIVISION_UP(number_wt, WIND_TURBINES_PER_BLOCK) *
                               WIND_TURBINES_PER_BLOCK * NUMBER_Y_EQUATIONS *
                               parameters.number_datapoints;

  gpuErrchk(cudaHostAlloc(&Pinned_U, sizeof(FLOATING_TYPE) * alloc_size_u,
                          cudaHostAllocDefault));
  gpuErrchk(cudaHostAlloc(&Pinned_Y, sizeof(FLOATING_TYPE) * alloc_size_y,
                          cudaHostAllocMapped));

  FLOATING_TYPE *MatrY_GPU =
      (FLOATING_TYPE *)malloc(sizeof(FLOATING_TYPE) * NUMBER_Y_EQUATIONS *
                              parameters.number_datapoints * number_wt);
  FLOATING_TYPE *global_current_injected_to_grid = (FLOATING_TYPE *)malloc(
      sizeof(FLOATING_TYPE) * 2 * parameters.number_datapoints);

  for (int i = 0; i < number_wt; i++) {
    auto pointer_U_start = parameters.Input_Data;
    auto pointer_U_output =
        Pinned_U +
        i * (NUMBER_U_VALUES - number_values_inputs_which_arent_in_input_csv) *
            (parameters.number_datapoints + 1);

    for (unsigned int j = 0; j < parameters.number_datapoints + 1; j++) {
      memcpy(pointer_U_output,
             pointer_U_start + number_values_inputs_which_arent_in_input_csv,
             (NUMBER_U_VALUES - number_values_inputs_which_arent_in_input_csv) *
                 sizeof(FLOATING_TYPE));

      if (j != parameters.number_datapoints - 1)
        pointer_U_start += NUMBER_U_VALUES;

      pointer_U_output +=
          (NUMBER_U_VALUES - number_values_inputs_which_arent_in_input_csv);
    }
  }

  GPU_Array_float *voltage_slack;
  GPU_Array_complex *injected_current_CUDA_global;

  {
    std::vector<FLOATING_TYPE> voltage_slack_v;
    voltage_slack_v.resize(2 * (parameters.number_datapoints + 1));

    for (unsigned int j = 0; j < parameters.number_datapoints + 1; j++) {
      const int current_busbar = MIN2(j, parameters.number_datapoints - 1);

      voltage_slack_v[2 * j + 0] =
          parameters.Input_Data[current_busbar * NUMBER_U_VALUES + 0];
      voltage_slack_v[2 * j + 1] =
          parameters.Input_Data[current_busbar * NUMBER_U_VALUES + 1];
    }

    voltage_slack = new GPU_Array_float(voltage_slack_v);
  }

  injected_current_CUDA_global =
      new GPU_Array_complex(parameters.number_datapoints);

  GPU_Array_float *dU = new GPU_Array_float(alloc_size_u);
  GPU_Array_float *dX = new GPU_Array_float(NUMBER_X_EQUATIONS * number_wt);
  dX->memset(0);
  GPU_Array_float *injected_current_GPU =
      new GPU_Array_float(parameters.number_datapoints * 2);
  GPU_Array_complex *voltages_wt_CUDA;

  {
    std::vector<C_FLOATING_TYPE> voltages_wt_temp;
    for (int i = 0; i < number_wt; i++) {
      voltages_wt_temp.push_back(
          C_FLOATING_TYPE(mat[i].MatrU0[0], mat[i].MatrU0[1]));
    }
    voltages_wt_CUDA = create_GPU_Array_complex(voltages_wt_temp);
  }

  int threads_in_block = ELECTRICAL_SYSTEM_BLOCKSIZE; // at least this

  syst.large_vector_inputs_no_padding_CUDA->copy_HtoD_range(
      (cuComplexType *)(&fixed_voltage), 0, 1);

  C_FLOATING_TYPE temporary_value_1_low_precision(
      real(syst.temporary_value_1_constant_high_precision),
      imag(syst.temporary_value_1_constant_high_precision));

  int blocks_stage1 = DIVISION_UP(syst.max_elements_stage1, threads_in_block);
  int blocks_stage2a = DIVISION_UP(syst.max_elements_stage2, threads_in_block);
  int blocks_stage2b = DIVISION_UP(syst.max_elements_stage3, threads_in_block);
  int blocks_number_connections =
      DIVISION_UP(syst.cpu->number_connections, threads_in_block);
  int blocks_wt = DIVISION_UP(number_wt, WIND_TURBINES_PER_BLOCK);

  int maximum_blocks = MAX3(blocks_stage1, blocks_stage2a + blocks_stage2b,
                            blocks_number_connections + 1 + blocks_wt);

  dim3 number_blocks(maximum_blocks, 1, 1);
  dim3 number_threads(threads_in_block, 1, 1);

  const int threads_per_wt_block =
      WIND_TURBINES_PER_BLOCK *
      MAX3(NUMBER_X_EQUATIONS, NUMBER_U_VALUES, NUMBER_Y_EQUATIONS);

  cuComplexType *pointer_number_buses =
      *syst.large_vector_inputs_no_padding_CUDA + 1;

  void *args_kernel[] = {
      (void *)(&(*voltage_slack)),
      (void *)(&(*syst.large_vector_inputs_no_padding_CUDA)),

      (void *)(&(*syst.multipliers_stage1_unified_CUDA)),
      (void *)(&(*syst.multipliers_stage2_unified_CUDA)),
      (void *)(&(*syst.multipliers_stage3_unified_CUDA)),

      (void *)(&(*syst.temporary_vector_CUDA_1)),
      (void *)(&(*syst.temporary_vector_CUDA_2)),
      (void *)(&(*syst.temporary_vector_CUDA_3)),

      (void *)(&syst.max_elements_stage1),
      (void *)(&syst.max_elements_stage2),
      (void *)(&syst.max_elements_stage3),

      (void *)(&(*syst.positions_stage1_unified_CUDA)),
      (void *)(&(*syst.element_size_stage1_unified_CUDA)),
      (void *)(&(*syst.number_elements_stage1_unified_CUDA)),
      (void *)(&(*syst.corresponding_element_stage1_unified_CUDA)),

      (void *)(&(*syst.positions_stage2_unified_CUDA)),
      (void *)(&(*syst.element_size_stage2_unified_CUDA)),
      (void *)(&(*syst.number_elements_stage2_unified_CUDA)),
      (void *)(&(*syst.corresponding_element_stage2_unified_CUDA)),

      (void *)(&(*syst.line_node_input_CUDA)),
      (void *)(&(*syst.line_node_output_CUDA)),

      (void *)(&(*syst.I0_multiplier_CUDA)),
      (void *)(&(*syst.V0_multiplier_CUDA)),
      (void *)(&(*syst.admittance_list_CUDA)),

      (void *)(&(*syst.first_column_CUDA)),
      (void *)(&(*syst.first_column_pos_CUDA)),
      (void *)(&(syst.first_column_size)),

      (void *)(&(*syst.wt_connection_points_CUDA)),
      (void *)(&(*injected_current_CUDA_global)),
      (void *)(&(*voltages_wt_CUDA)),
      (void *)(&(*syst.multipliers_stage4_unified_CUDA)),
      (void *)(&(*((cuComplexType *)(&temporary_value_1_low_precision)))),

      (void *)(&blocks_stage2a),
      (void *)(&(blocks_number_connections)),

      (void *)(&(syst.cpu->number_wt)),
      (void *)(&(syst.cpu->number_buses)),
      (void *)(&(syst.cpu->number_connections)),
      (void *)(&blocks_wt),

      (void *)(&parameters.number_datapoints),

      (void *)(&(syst.cpu->delta_t)),

      (void *)(&(*dU0)),
      (void *)(&(*dA)),
      (void *)(&(*dB)),
      (void *)(&(*dC)),
      (void *)(&(*dD)),
      (void *)(&(*dY0)),
      (void *)(&(*dU)),
      (void *)(&(*dX)),

      (void *)(&(*voltages_wt_CUDA)),
      (void *)(&Pinned_Y),

      (void *)(&(pointer_number_buses))};

  size_t amount_variable_shared_memory = 0;
  cudaStream_t stream = (cudaStream_t)0;

  assert(threads_per_wt_block <= number_threads.x);

  TimeMeasurer_us time_measurer; // Start of time measurement
  dU->copy_HtoD(Pinned_U);

  gpuErrchk(cudaLaunchCooperativeKernel(
      (void *)unified_wt_simulator_kernel, number_blocks, number_threads,
      args_kernel, amount_variable_shared_memory, stream));

  injected_current_CUDA_global->copy_DtoH(
      (cuComplexType *)global_current_injected_to_grid);

  gpuErrchk(cudaDeviceSynchronize());
  unsigned long total_time = time_measurer.measure("GPU"); // End of measurement

  reorder_CUDA_wt_output(parameters, number_wt, Pinned_Y, MatrY_GPU);

  write_all_wt_results_csv(parameters, number_wt, MatrY_GPU, NUMBER_Y_EQUATIONS,
                           "GPU");
  write_csv_fullpath("I_Total_GPU", parameters.number_datapoints, 2,
                     global_current_injected_to_grid);

  gpuErrchk(cudaFreeHost(Pinned_Y));
  gpuErrchk(cudaFreeHost(Pinned_U));
  gpuErrchk(cudaDeviceSynchronize());

  syst.destroy_matrices();

  delete dA;
  delete dB;
  delete dC;
  delete dD;
  delete dX0;
  delete dU0;
  delete dY0;

  delete dU;
  delete dX;

  delete[] I0_electrical_system_start;
  delete[] mat;
  delete injected_current_GPU;

  delete voltages_wt_CUDA;
  delete voltage_slack;
  delete injected_current_CUDA_global;

  free(MatrY_GPU);
  free(global_current_injected_to_grid);

  gpuErrchk(cudaDeviceReset()); // Stop profiler

  return total_time;
}

// Reorganize output of the CUDA kernel so as to be written into CSV files
void reorder_CUDA_wt_output(const wt_parameters_struct &parameters,
                            unsigned int number_wt, FLOATING_TYPE *Pinned_Y,
                            FLOATING_TYPE *Matry_GPU) {

  int blocks_launched_wt = DIVISION_UP(number_wt, WIND_TURBINES_PER_BLOCK);

  for (unsigned int i = 0; i < number_wt; i++) {
    const int i_local = i % WIND_TURBINES_PER_BLOCK;
    const int i_start = (i - i_local) / WIND_TURBINES_PER_BLOCK;

    int position_Y_output =
        i * NUMBER_Y_EQUATIONS * parameters.number_datapoints;

    int position_Y_input =
        i_local + i_start * WIND_TURBINES_PER_BLOCK * NUMBER_Y_EQUATIONS;

    for (unsigned int j = 0; j < parameters.number_datapoints; j++) {

      for (unsigned int component = 0; component < NUMBER_Y_EQUATIONS;
           component++) {
        Matry_GPU[position_Y_output + component] =
            Pinned_Y[position_Y_input + component * WIND_TURBINES_PER_BLOCK];
      }

      position_Y_output += NUMBER_Y_EQUATIONS;
      position_Y_input +=
          blocks_launched_wt * WIND_TURBINES_PER_BLOCK * NUMBER_Y_EQUATIONS;
    }
  }
}
