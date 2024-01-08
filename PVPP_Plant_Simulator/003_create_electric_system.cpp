
#include "main.h"

#include "sparse_matrix.h"

#include "electrical_system.h"

const bool PUT_BYPASS = true;

const FLOATING_TYPE q = (FLOATING_TYPE)1.60217662E-19;
const FLOATING_TYPE k = (FLOATING_TYPE)1.38064852E-23;

const FLOATING_TYPE na = (FLOATING_TYPE)48.548;
const FLOATING_TYPE T_0 = (FLOATING_TYPE)25;
const FLOATING_TYPE G_0 = (FLOATING_TYPE)1000;

const FLOATING_TYPE R_s = (FLOATING_TYPE)0.262 * (175.0 / 400.0);
const FLOATING_TYPE R_sh = (FLOATING_TYPE)90.85;

const FLOATING_TYPE I_0 = (FLOATING_TYPE)4.86E-10 * (400.0 / 175.0);

const FLOATING_TYPE K_0_percent = (FLOATING_TYPE)-0.45;

const FLOATING_TYPE I_pv = (FLOATING_TYPE)8.11 * 400.0 / 175.0 * 1.05;
const FLOATING_TYPE Nominal_Power_PV_Panel = 400;

const FLOATING_TYPE K_0 = K_0_percent / (FLOATING_TYPE)100.0;

FLOATING_TYPE func_diode_panel(FLOATING_TYPE V,
                               FLOATING_TYPE temperature_centigrade) {
  const FLOATING_TYPE qkT =
      q / k / (FLOATING_TYPE)(273.15 + temperature_centigrade);
  auto res = I_0 * (std::exp(qkT * V / na) - (FLOATING_TYPE)1);
  return res;
}

FLOATING_TYPE
func_diode_panel_derivative(FLOATING_TYPE V,
                            FLOATING_TYPE temperature_centigrade) {
  const FLOATING_TYPE qkT =
      q / k / (FLOATING_TYPE)(273.15 + temperature_centigrade);
  auto res = I_0 * qkT / na * std::exp(V * qkT / na);

  return res;
}

FLOATING_TYPE func_diode_panel_aggregated(FLOATING_TYPE V,
                                          FLOATING_TYPE temperature_centigrade,
                                          int aggregated_x, int aggregated_y) {
  const FLOATING_TYPE qkT =
      q / k / (FLOATING_TYPE)(273.15 + temperature_centigrade);
  auto res = aggregated_y * I_0 *
             (std::exp(qkT * (V / (FLOATING_TYPE)aggregated_x) / na) -
              (FLOATING_TYPE)1);

  return res;
}

FLOATING_TYPE
func_diode_panel_derivative_aggregated(FLOATING_TYPE V,
                                       FLOATING_TYPE temperature_centigrade,
                                       int aggregated_x, int aggregated_y) {
  const FLOATING_TYPE qkT =
      q / k / (FLOATING_TYPE)(273.15 + temperature_centigrade);
  auto res = aggregated_y * I_0 * qkT / na / (FLOATING_TYPE)aggregated_x *
             std::exp((V / (FLOATING_TYPE)aggregated_x) * qkT / na);

  return res;
}

FLOATING_TYPE func_diode_bypass(FLOATING_TYPE V,
                                FLOATING_TYPE temperature_centigrade) {
  const FLOATING_TYPE qkT =
      q / k / (FLOATING_TYPE)(273.15 + temperature_centigrade);
  auto res = 0.3E-6 * (exp(qkT * V) - 1);
  return (FLOATING_TYPE)res;
}

FLOATING_TYPE
func_diode_bypass_derivative(FLOATING_TYPE V,
                             FLOATING_TYPE temperature_centigrade) {
  const FLOATING_TYPE qkT =
      q / k / (FLOATING_TYPE)(273.15 + temperature_centigrade);

  auto res = qkT * 0.3E-6 * (exp(qkT * V));
  return (FLOATING_TYPE)res;
}

void add_solar_panel(basic_electric_circuit &syst, int node_input,
                     int node_output, FLOATING_TYPE V_initial,
                     FLOATING_TYPE irradiation, FLOATING_TYPE temperature) {

  int new_internal_node = syst.create_node();

  syst.add_linear_element(ADMITTANCE, (FLOATING_TYPE)1.0 / R_s,
                          new_internal_node, node_input);
  syst.add_linear_element(ADMITTANCE, (FLOATING_TYPE)1.0 / R_sh,
                          new_internal_node, node_output);

  syst.add_nonlinear_voltage_source(
      [temperature](FLOATING_TYPE V) {
        return func_diode_panel(V, temperature);
      },
      [temperature](FLOATING_TYPE V) {
        return func_diode_panel_derivative(V, temperature);
      },
      V_initial, new_internal_node, node_output);

  const FLOATING_TYPE I_pv_irradiated_no_temperature = I_pv * irradiation / G_0;
  const FLOATING_TYPE perone_temperature_loss = K_0 * (temperature - T_0);
  const FLOATING_TYPE I_pv_final =
      I_pv_irradiated_no_temperature * (1.0 + perone_temperature_loss);

  syst.add_linear_element(CURRENT_SOURCE, I_pv_final, new_internal_node,
                          node_output);
  if (PUT_BYPASS) {
    syst.add_nonlinear_voltage_source(
        [temperature](FLOATING_TYPE V) {
          return func_diode_bypass(V, temperature);
        },
        [temperature](FLOATING_TYPE V) {
          return func_diode_bypass_derivative(V, temperature);
        },
        V_initial, node_output, node_input);
  }
}

void add_solar_panel_aggregated(basic_electric_circuit &syst, int node_input,
                                int node_output, FLOATING_TYPE V_initial,
                                FLOATING_TYPE irradiation,
                                FLOATING_TYPE temperature, int aggregation_x,
                                int aggregation_y) {

  int new_internal_node = syst.create_node();
  syst.add_linear_element(ADMITTANCE, (FLOATING_TYPE)1.0 / ((R_s)),
                          new_internal_node, node_input);

  syst.add_nonlinear_voltage_source(
      [temperature, aggregation_x, aggregation_y](FLOATING_TYPE V) {
        return func_diode_panel_aggregated(V * R_sh / (R_s + R_sh), temperature,
                                           aggregation_x, aggregation_y);
      },
      [temperature, aggregation_x, aggregation_y](FLOATING_TYPE V) {
        return func_diode_panel_derivative_aggregated(
            V * R_sh / (R_s + R_sh), temperature, aggregation_x, aggregation_y);
      },
      V_initial, new_internal_node, node_output);

  syst.add_linear_element(ADMITTANCE,
                          (FLOATING_TYPE)1.0 / (R_sh * aggregation_y),
                          new_internal_node, node_output);

  const FLOATING_TYPE I_pv_irradiated_basic =
      I_pv * irradiation / G_0 + K_0 * (temperature - T_0);
  const FLOATING_TYPE I_pv_irradiated =
      aggregation_y * I_pv_irradiated_basic * (R_sh / (R_s + R_sh));
  syst.add_linear_element(CURRENT_SOURCE, I_pv_irradiated, new_internal_node,
                          node_output);
}

void get_PV_panel_position(Option_PVPP &options, int current_block, int i,
                           int j, FLOATING_TYPE *canvas_clouds,
                           int_2 &pos_xy_panel,
                           FLOATING_TYPE &irradiation_with_clouds) {
  int current_block_x = (current_block % options.number_blocks_1D);
  int current_block_y = current_block / options.number_blocks_1D;

  int posicion_panel_x = current_block_x * options.panels_per_block_x + j;
  int posicion_panel_y = current_block_y * options.panels_per_block_y + i;

  irradiation_with_clouds = options.base_irradiance;

  if (options.take_into_account_clouds) {

    irradiation_with_clouds *=
        (FLOATING_TYPE)1.0 -
        canvas_clouds[posicion_panel_x +
                      posicion_panel_y * options.size_x_cloud_canvas];
  }

  pos_xy_panel = {posicion_panel_x, posicion_panel_y};
}

int_2 create_block_of_panels_simple(
    basic_electric_circuit &Circuit, Option_PVPP &options, int current_block,
    FLOATING_TYPE *canvas_clouds,
    vector<node_and_position_xy_panel> &node_and_position_panels) {
  if (options.interconnection == INTERCONNECTION_AGGREGATED) {
    int input_output_nodes_after_resistance_input = Circuit.create_node();
    int input_output_nodes_after_resistance_output = Circuit.create_node();

    int_2 input_output_nodes_after_resistance =
        int_2(input_output_nodes_after_resistance_input,
              input_output_nodes_after_resistance_output);

    const FLOATING_TYPE V_initial = 0;
    int_2 pos_xy_panel;
    FLOATING_TYPE irradiation_with_clouds;
    get_PV_panel_position(options, current_block,
                          options.panels_per_block_y / 2,
                          options.panels_per_block_x / 2, canvas_clouds,
                          pos_xy_panel, irradiation_with_clouds);

    add_solar_panel_aggregated(
        Circuit, input_output_nodes_after_resistance_input,
        input_output_nodes_after_resistance_output, V_initial,
        irradiation_with_clouds, options.temperature_panels,
        options.panels_per_block_x, options.panels_per_block_y);

    node_and_position_xy_panel ttt;
    ttt.nodes_PV_panel = input_output_nodes_after_resistance;
    ttt.position_x_y = pos_xy_panel;
    ttt.incoming_radiation = irradiation_with_clouds;
    ttt.is_broken = false;

    node_and_position_panels.push_back(ttt);

    return input_output_nodes_after_resistance;
  }

  int node_input_before_resistance = Circuit.create_node();
  int node_output_before_resistance = Circuit.create_node();

  vector<vector<int_2>> nodelist;

  for (int i = 0; i < options.panels_per_block_y; i++) {
    nodelist.push_back(vector<int_2>());

    int_2 nodes_solar_panel;

    for (int j = 0; j < options.panels_per_block_x; j++) {

      nodes_solar_panel = {
          (j == options.panels_per_block_x - 1) ? node_input_before_resistance
                                                : Circuit.create_node(),
          (j == 0) ? node_output_before_resistance : get<0>(nodes_solar_panel)};

      nodelist.back().push_back(nodes_solar_panel);

      const FLOATING_TYPE V_initial = 0;

      int_2 pos_xy_panel;
      FLOATING_TYPE irradiation_with_clouds;

      get_PV_panel_position(options, current_block, i, j, canvas_clouds,
                            pos_xy_panel, irradiation_with_clouds);

      const bool is_PV_panel_broken =
          options.broken_panels.find({current_block, i, j}) !=
          options.broken_panels.end();

      if (is_PV_panel_broken) {
        Circuit.add_linear_element(
            ADMITTANCE, (FLOATING_TYPE)1.0 / resistance_wire,
            std::get<0>(nodes_solar_panel), std::get<1>(nodes_solar_panel));
      } else {
        add_solar_panel(Circuit, std::get<0>(nodes_solar_panel),
                        std::get<1>(nodes_solar_panel), V_initial,
                        irradiation_with_clouds, options.temperature_panels);
      }

      node_and_position_xy_panel ttt;
      ttt.nodes_PV_panel = nodes_solar_panel;
      ttt.position_x_y = pos_xy_panel;
      ttt.incoming_radiation = irradiation_with_clouds;
      ttt.is_broken = is_PV_panel_broken;

      node_and_position_panels.push_back(ttt);
    }
  }

  for (int i = 0; i < options.panels_per_block_y - 1; i++) {
    for (int j = 0; j < options.panels_per_block_x - 1; j++) {
      bool must_add_connection = false;

      if (options.interconnection == INTERCONNECTION_NOTHING) {
        // no connection here never
      } else if (options.interconnection == INTERCONNECTION_FULL) {
        must_add_connection = true;
      } else if (options.interconnection == INTERCONNECTION_HONEYCOMB) {
        if (((i + j) % 2) == 1) {
          must_add_connection = true;
        }
      } else if (options.interconnection == INTERCONNECTION_BRIDGE) {
        must_add_connection = (i % 3) == 0;
        if ((j % 2) == 0) {
          must_add_connection = !must_add_connection;
        }
      } else {
        assert(0);
      }

      if (must_add_connection) {
        Circuit.add_linear_element(
            ADMITTANCE, (FLOATING_TYPE)1.0 / resistance_wire,
            get<0>(nodelist[i][j]), get<0>(nodelist[i + 1][j]));
      }
    }
  }

  int_2 input_output_nodes_after_resistance = {Circuit.create_node(),
                                               Circuit.create_node()};

  Circuit.add_linear_element(ADMITTANCE, (FLOATING_TYPE)1.0 / resistance_wire,
                             get<0>(input_output_nodes_after_resistance),
                             node_input_before_resistance);
  Circuit.add_linear_element(ADMITTANCE, (FLOATING_TYPE)1.0 / resistance_wire,
                             get<1>(input_output_nodes_after_resistance),
                             node_output_before_resistance);

  return input_output_nodes_after_resistance;
}

void create_full_electric_system(
    basic_electric_circuit &Circuit, Option_PVPP &options,
    FLOATING_TYPE *canvas_clouds, int_2 &nodes_voltage_source,
    int &position_of_current_that_flows,
    vector<node_and_position_xy_panel> &node_and_position_panels) {

  FLOATING_TYPE voltage_PCC =
      options.panels_per_block_x * options.external_voltage_per_block_x_panels;

  vector<int_2> external_connection;
  vector<int_2> main_nodes;

  node_and_position_panels.clear();

  for (int current_block = 0; current_block < options.number_blocks_total;
       current_block++) {
    external_connection.push_back(
        create_block_of_panels_simple(Circuit, options, current_block,
                                      canvas_clouds, node_and_position_panels)

    );

    main_nodes.push_back(
        int_2(Circuit.create_node_ac(), Circuit.create_node_ac()));

    // Path so that DC circuit is fixed to ground
    Circuit.add_linear_element(ADMITTANCE, (FLOATING_TYPE)1.0 / resistance_wire,
                               get<1>(external_connection.back()), 0);

    Circuit.create_inverter_dc_ac(
        voltage_PCC,

        external_voltage_AC + 0.0i, // AC power

        get<0>(external_connection.back()), get<1>(external_connection.back()),

        get<0>(main_nodes.back()), get<1>(main_nodes.back()));

    if (current_block != 0) {
      Circuit.add_linear_element_ac(ADMITTANCE,
                                    (FLOATING_TYPE)1.0 / resistance_wire,
                                    get<0>(main_nodes[current_block]),
                                    get<0>(main_nodes[current_block - 1]));
      Circuit.add_linear_element_ac(ADMITTANCE,
                                    (FLOATING_TYPE)1.0 / resistance_wire,
                                    get<1>(main_nodes[current_block]),
                                    get<1>(main_nodes[current_block - 1]));
    }
  }

  nodes_voltage_source = {Circuit.create_node_ac(), 0};
  position_of_current_that_flows = Circuit.add_linear_element_ac(
      VOLTAGE_SOURCE, external_voltage_AC, std::get<0>(nodes_voltage_source),
      std::get<1>(nodes_voltage_source));

  Circuit.add_linear_element_ac(
      ADMITTANCE, (FLOATING_TYPE)1.0 / resistance_wire,
      get<0>(nodes_voltage_source), get<0>(main_nodes[0]));
  Circuit.add_linear_element_ac(
      ADMITTANCE, (FLOATING_TYPE)1.0 / resistance_wire,
      get<1>(nodes_voltage_source), get<1>(main_nodes[0]));

  Circuit.add_linear_element_ac(
      ADMITTANCE, (FLOATING_TYPE)1.0 / resistance_wire,
      get<0>(nodes_voltage_source), get<0>(main_nodes.back()));
  Circuit.add_linear_element_ac(
      ADMITTANCE, (FLOATING_TYPE)1.0 / resistance_wire,
      get<1>(nodes_voltage_source), get<1>(main_nodes.back()));
}
