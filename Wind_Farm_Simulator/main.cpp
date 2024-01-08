#include "main.h"
#include <cuda_runtime.h>

unsigned long Wind_Farm_Test_CPU(wt_parameters_struct &parameters,
                                 const int number_wt);
unsigned long Wind_Farm_Test_CUDA(wt_parameters_struct &parameters,
                                  const int number_wt);

int main(int /*argc*/, char ** /* argv*/) {
  cudaSetDevice(0);

  const int maximum_wt = 2464;

  wt_parameters_struct parameters;

  for (int repetitions = 0; repetitions < 3; repetitions++) {
    for (int number_wt = 1; number_wt <= maximum_wt; number_wt++) {
      Wind_Farm_Test_CPU(parameters, number_wt);
      Wind_Farm_Test_CUDA(parameters, number_wt);
    }
  }

  return 0;
}
