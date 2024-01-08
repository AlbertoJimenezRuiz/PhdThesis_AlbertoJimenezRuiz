
#ifndef MAIN_H
#define MAIN_H

#define FLOATING_TYPE double

const FLOATING_TYPE external_voltage_AC = 25000;

#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstring>
#include <iostream>
#include <list>
#include <set>
#include <stdarg.h>
#include <tuple>
#include <utility>
#include <vector>

using namespace std;

struct cloud {
  FLOATING_TYPE center_x;
  FLOATING_TYPE center_y;

  FLOATING_TYPE size_x_opaque;
  FLOATING_TYPE size_y_opaque;

  FLOATING_TYPE translucent_size;
  FLOATING_TYPE opacity_cloud;

  FLOATING_TYPE speed_x;
  FLOATING_TYPE speed_y;
};

enum interconnection_types {
  INTERCONNECTION_NOTHING,
  INTERCONNECTION_FULL,
  INTERCONNECTION_HONEYCOMB,
  INTERCONNECTION_BRIDGE,

  INTERCONNECTION_AGGREGATED // one large panel
};

typedef std::complex<FLOATING_TYPE> C_FLOATING_TYPE;
typedef std::vector<FLOATING_TYPE> R_VECTOR;
typedef std::vector<C_FLOATING_TYPE> C_VECTOR;
typedef std::complex<long double> C_LONG_DOUBLE;

typedef pair<int, int> int_2;

struct node_and_position_xy_panel {
  int_2 nodes_PV_panel;
  int_2 position_x_y;
  FLOATING_TYPE incoming_radiation;
  bool is_broken;
};

#define MIN2(a, b) ((a) < (b) ? (a) : (b))
#define MAX2(a, b) ((a) > (b) ? (a) : (b))
#define DIVISION_UP(n, d) (((n) + (d)-1) / (d))

enum ENUM_SIMULATION_TYPE {
  SIMULATION_NO_CLOUD,
  SIMULATION_1_SNAPSHOT,
  SIMULATION_CLOUD_SWEEP,
  SIMULACION_1_CLOUD
};

int random_int(int upper);
void advance_clouds(list<cloud> &list_cl, int width, int height);
list<cloud> create_random_clouds(size_t n, int width, int height);

const FLOATING_TYPE resistance_wire = (FLOATING_TYPE)1e-5;

void __attribute__((format(printf, 1, 0))) abortmsg(const char *format, ...);

#include <chrono>

class Time_Measurer {
public:
  Time_Measurer();
  long int measure(const char *msg = NULL);
  std::chrono::high_resolution_clock::time_point start;
};

struct Option_PVPP {
  int number_blocks_total; // 330
  int panels_per_block_x;  // 50
  int panels_per_block_y;  // 50
  FLOATING_TYPE external_voltage_per_block_x_panels;
  interconnection_types interconnection;

  bool is_CUDA;
  ENUM_SIMULATION_TYPE simulation_type;
  bool early_exit;

  FLOATING_TYPE base_irradiance;
  FLOATING_TYPE temperature_panels;

  int size_x_cloud_canvas;
  int size_y_cloud_canvas;
  int number_blocks_1D;

  int number_cloud_snapshots;
  list<cloud> cloud_list;
  bool take_into_account_clouds;

  std::set<std::tuple<int, int, int>> broken_panels;

  ~Option_PVPP() {}

  Option_PVPP() = delete;
  Option_PVPP(int m_number_blocks_total, int m_panels_per_block_x,
              int m_panels_per_block_y,
              FLOATING_TYPE m_external_voltage_per_block_x_panels,
              interconnection_types m_interconnection, bool m_is_CUDA,
              ENUM_SIMULATION_TYPE m_simulation_type, bool m_early_exit,
              FLOATING_TYPE m_base_irradiance,
              FLOATING_TYPE m_temperature_panels)
      :

        number_blocks_total(m_number_blocks_total),
        panels_per_block_x(m_panels_per_block_x),
        panels_per_block_y(m_panels_per_block_y),
        external_voltage_per_block_x_panels(
            m_external_voltage_per_block_x_panels),
        interconnection(m_interconnection), is_CUDA(m_is_CUDA),
        simulation_type(m_simulation_type), early_exit(m_early_exit),
        base_irradiance(m_base_irradiance),
        temperature_panels(m_temperature_panels) {
    get_y_dimensions_canvas_cloud();
  }

  void get_y_dimensions_canvas_cloud();
};

template <typename T> bool AreSame_floatingpoint(T a, T b) {
  return fabs(a - b) < 1e-10;
}

#endif
