#include "main.h"
#include "gpu_util.h"
#include "sparse_matrix.h"

void PVPP_Simulator(Option_PVPP &options);

int main() {
  cudaDeviceReset();
  Option_PVPP options(
      1,  // number of total inverters
      50, // PV panels per inverter (direction x)
      50, // PV panels per inverter (direction y)
      23, // Voltage per PV panel (multiplied by panels in x direction)
      INTERCONNECTION_NOTHING, // interconnection
      true,                    // Uses CUDA?
      SIMULATION_NO_CLOUD,     // simulation type
      false,                   // Run only a few iterations
      1000,                    // Base Irradiance
      25                       // Temperature panels (ºC)
  );

  PVPP_Simulator(options);
  cudaDeviceReset();
  return 0;
}
