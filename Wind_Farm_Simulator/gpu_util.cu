#include "gpu_util.h"
#include "main.h"

void check_memory_allocation_CUDA_possible(
    const struct wt_parameters_struct &parameters,
    unsigned int number_wind_turbines, unsigned int matrix_size_X) {
  size_t mem_tot_0 = 0;
  size_t mem_free_0 = 0;
  long long available_memory;

  const int max_size_xyu = MAX2(matrix_size_X, NUMBER_U_VALUES);

  gpuErrchk(cudaMemGetInfo(&mem_free_0, &mem_tot_0));
  available_memory =
      (long long)mem_free_0 / 100 *
      70; /* 70% of available memory so as to prevent computer freezes */

  long long memory_to_reserve = 0;
  memory_to_reserve += max_size_xyu * max_size_xyu * 4; /* Matrices A,B,C,D */
  memory_to_reserve += max_size_xyu * 3;                /*X, U0, Y0 */

  memory_to_reserve += parameters.number_datapoints * number_wind_turbines *
                       (/*Input signals */
                        NUMBER_U_VALUES);

  memory_to_reserve *= sizeof(FLOATING_TYPE);

  if (available_memory < memory_to_reserve) {
    abort_msg("Can't allocate so much data on the VRAM. Lower the number of "
              "datapoints. %d ##### %d \n",
              (int)memory_to_reserve, (int)available_memory);
  }
}

void gpuAssert(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    abort_msg("GPUassert: Error %d: %s. File: %s Line: %d \n", code,
              cudaGetErrorString(code), file, line);
  }
}
