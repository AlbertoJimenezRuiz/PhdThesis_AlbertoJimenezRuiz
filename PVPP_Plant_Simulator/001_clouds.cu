#include "gpu_util.h"
#include "main.h"

list<cloud> create_random_clouds(size_t n, int width, int height) {
  list<cloud> cloud_list;

  for (size_t i = 0; i < n; i++) {

    cloud cl;

    cl.center_x = (FLOATING_TYPE)random_int(width);
    cl.center_y = (FLOATING_TYPE)random_int(height);

    cl.size_x_opaque = (FLOATING_TYPE)(10 + random_int(30));
    cl.size_y_opaque = (FLOATING_TYPE)(10 + random_int(30));

    cl.translucent_size = (FLOATING_TYPE)1.5;
    cl.opacity_cloud =
        (FLOATING_TYPE)(70 + random_int(30)) / (FLOATING_TYPE)100;

    cl.speed_x = (FLOATING_TYPE)(5 + random_int(10));
    cl.speed_y = (FLOATING_TYPE)(5 + random_int(10));

    cl.speed_x *= random_int(2) ? (FLOATING_TYPE)1 : (FLOATING_TYPE)-1;
    cl.speed_y *= random_int(2) ? (FLOATING_TYPE)1 : (FLOATING_TYPE)-1;

    cloud_list.push_back(cl);
  }

  return cloud_list;
}

list<cloud> create_one_cloud_from_left_to_right(int width, int height) {
  list<cloud> cloud_list;

  cloud cl;

  cl.center_x = -width * 15 / 100;
  cl.center_y = height / 2;

  cl.size_x_opaque = (FLOATING_TYPE)(100);
  cl.size_y_opaque = (FLOATING_TYPE)(50);

  cl.translucent_size = (FLOATING_TYPE)0.5;
  cl.opacity_cloud = (FLOATING_TYPE)0.7;

  cl.speed_x = (FLOATING_TYPE)(25);
  cl.speed_y = 0;

  cloud_list.push_back(cl);

  return cloud_list;
}

void advance_clouds(list<cloud> &list_cl, int width, int height) {
  (void)width;
  (void)height;
  for (auto &cl : list_cl) {
    cl.center_x += cl.speed_x;
    cl.center_y += cl.speed_y;
  }
}

void Option_PVPP::get_y_dimensions_canvas_cloud() {

  size_x_cloud_canvas = panels_per_block_x;
  size_y_cloud_canvas = panels_per_block_y;

  number_blocks_1D = (int)sqrt(number_blocks_total);

  if (number_blocks_1D * number_blocks_1D < number_blocks_total) {
    number_blocks_1D++;
  }

  size_x_cloud_canvas *= number_blocks_1D;
  size_y_cloud_canvas *= number_blocks_1D;

  if (simulation_type == SIMULATION_CLOUD_SWEEP) {
    cloud_list = create_one_cloud_from_left_to_right(size_x_cloud_canvas,
                                                     size_y_cloud_canvas);
    number_cloud_snapshots = 50;
    take_into_account_clouds = true;
  } else if (simulation_type == SIMULACION_1_CLOUD) {
    cloud_list =
        create_random_clouds(1, size_x_cloud_canvas, size_y_cloud_canvas);
    number_cloud_snapshots = 1;
    take_into_account_clouds = true;
  } else {
    cloud_list =
        create_random_clouds(1, size_x_cloud_canvas, size_y_cloud_canvas);
    number_cloud_snapshots = 1;
    take_into_account_clouds = false;
  }
}

__global__ void print_clouds_CUDA_kernel(FLOATING_TYPE *cloud_data,
                                         const int rows, const int columns,
                                         const int cloud_number,
                                         cloud *list_cl) {
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int x = blockIdx.x * blockDim.x + threadIdx.x;

  if (x >= columns)
    return;

  if (y >= rows)
    return;

  FLOATING_TYPE final_value = 0;

  for (int idx = 0; idx < cloud_number; idx++) {
    auto &cl = list_cl[idx];

    FLOATING_TYPE distx =
        std::abs((FLOATING_TYPE)x - cl.center_x) / cl.size_x_opaque;
    FLOATING_TYPE disty =
        std::abs((FLOATING_TYPE)y - cl.center_y) / cl.size_y_opaque;

    FLOATING_TYPE total_distance = distx * distx + disty * disty;

    total_distance = sqrt(total_distance);

    FLOATING_TYPE opacity_cloud_temporal =
        (total_distance - 1) / cl.translucent_size;
    opacity_cloud_temporal = MAX2(opacity_cloud_temporal, (FLOATING_TYPE)0);
    opacity_cloud_temporal = MIN2(opacity_cloud_temporal, (FLOATING_TYPE)1);

    opacity_cloud_temporal = 1 - opacity_cloud_temporal;

    opacity_cloud_temporal *= cl.opacity_cloud;

    final_value = final_value * (1 - opacity_cloud_temporal) +
                  (FLOATING_TYPE)1 * +opacity_cloud_temporal;
  }

  cloud_data[rows * x + y] = final_value;
}

void print_clouds_CUDA(GPU_array_f &data_clouds_CUDA, const int rows,
                       const int columns, const list<cloud> &cloud_list) {

  std::vector<cloud> list_cl_vector(cloud_list.begin(), cloud_list.end());
  GPU_array<cloud> list_cl_CUDA(list_cl_vector);

  const int threads_per_block_1D = 32;

  dim3 block_count(DIVISION_UP(rows, threads_per_block_1D),
                   DIVISION_UP(columns, threads_per_block_1D), 1);
  dim3 thread_count(threads_per_block_1D, threads_per_block_1D, 1);

  print_clouds_CUDA_kernel<<<block_count, thread_count>>>(
      data_clouds_CUDA, rows, columns, (int)cloud_list.size(), list_cl_CUDA);

  gpuErrchk(cudaPeekAtLastError());
}
