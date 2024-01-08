

void create_full_electric_system(
    basic_electric_circuit &Circuit, Option_PVPP &options,
    FLOATING_TYPE *canvas_clouds, int_2 &nodes_voltage_source,
    int &position_of_current_that_flows,
    vector<node_and_position_xy_panel> &node_and_position_panels);

void calculate_new_iteration(basic_electric_circuit &Circuit,
                             vector<FLOATING_TYPE> &solution,
                             long double &diff);

void print_clouds_CUDA(GPU_array_f &data_clouds_CUDA, const int rows,
                       const int columns, const list<cloud> &cloud_list);
