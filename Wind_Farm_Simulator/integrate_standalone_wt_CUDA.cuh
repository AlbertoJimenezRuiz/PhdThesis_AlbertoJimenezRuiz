
const int tid_local = threadIdx.x % WIND_TURBINES_PER_BLOCK;
const int current_state = threadIdx.x / WIND_TURBINES_PER_BLOCK;
const int current_wt =
    WIND_TURBINES_PER_BLOCK * (blockIdx.x - blocks_number_connections - 1) +
    tid_local;
const int tid = threadIdx.x;

FLOATING_TYPE __shared__ X_s[WIND_TURBINES_PER_BLOCK * NUMBER_X_EQUATIONS];
FLOATING_TYPE __shared__ X_temp[WIND_TURBINES_PER_BLOCK * NUMBER_X_EQUATIONS];
FLOATING_TYPE __shared__ Unext_s[WIND_TURBINES_PER_BLOCK * NUMBER_U_VALUES];
FLOATING_TYPE __shared__ U_s[WIND_TURBINES_PER_BLOCK * NUMBER_U_VALUES];

if (current_wt < number_wt) {
  if (current_state < NUMBER_X_EQUATIONS) {
    X_s[tid] = dX[current_wt * NUMBER_X_EQUATIONS + current_state];
  }
}

__syncthreads();

if (current_wt < number_wt) {
  const auto wt_voltage = get_V0_previous_buses(
      wt_connection_points[current_wt], unified_input,
      temporary_vector_2, // Reduction to obtain the value of x before
                          // substracting the union
      temporary_vector_3, // Reduction to obtain the first value of the union
      ad_multiplier, temporary_value_1_constant, number_buses);

  if (current_state == 0) {
    U_s[tid] = Unext_s[tid] =
        wt_voltage.x - MatrU0[current_wt * NUMBER_U_VALUES + current_state];
  } else if (current_state == 1) {
    U_s[tid] = Unext_s[tid] =
        wt_voltage.y - MatrU0[current_wt * NUMBER_U_VALUES + current_state];
  } else if (current_state < NUMBER_U_VALUES) {
    U_s[tid] = dU[SUBEXPRESSION_MEMORY_ADDRESS_INTEGRATE_WT(
                   MIN2(current_timestep, number_timesteps - 1))] -
               MatrU0[current_wt * NUMBER_U_VALUES +
                      current_state]; // The value of the current input is the
                                      // value of the input in the next instant
                                      // during the next iteration

    Unext_s[tid] = dU[SUBEXPRESSION_MEMORY_ADDRESS_INTEGRATE_WT(
                       MIN2(current_timestep + 1, number_timesteps - 1))] -
                   MatrU0[current_wt * NUMBER_U_VALUES + current_state];
  }
}

__syncthreads();
FLOATING_TYPE Xprime_local;

if (current_wt < number_wt) {
  if (current_state < NUMBER_Y_EQUATIONS) {
    FLOATING_TYPE Var1 = 0;
    FLOATING_TYPE Var2 = 0;

#pragma unroll
    for (int i = 0;
         i < (NUMBER_X_EQUATIONS > NUMBER_U_VALUES ? NUMBER_X_EQUATIONS
                                                   : NUMBER_U_VALUES);
         i++) {

      if (i < NUMBER_X_EQUATIONS)
        Var1 += MatrC[current_wt * NUMBER_Y_EQUATIONS * NUMBER_X_EQUATIONS +
                      current_state * NUMBER_X_EQUATIONS + i] *
                X_s[tid_local + i * WIND_TURBINES_PER_BLOCK];
      if (i < NUMBER_U_VALUES)
        Var2 += MatrD[current_wt * NUMBER_Y_EQUATIONS * NUMBER_U_VALUES +
                      current_state * NUMBER_U_VALUES + i] *
                U_s[tid_local + i * WIND_TURBINES_PER_BLOCK];
    }

    FLOATING_TYPE Y0_local = dY0[current_state];
    FLOATING_TYPE output_temp = Y0_local + Var1 + Var2;

    if (current_state == 0) {
      dCurrents_Wind_Farm_Global[current_wt].x = output_temp;
    } else if (current_state == 1) {
      dCurrents_Wind_Farm_Global[current_wt].y = output_temp;
    }

    dOutputs[tid + (current_timestep * number_blocks_wt +
                    (blockIdx.x - blocks_number_connections - 1)) *
                       WIND_TURBINES_PER_BLOCK * NUMBER_Y_EQUATIONS] =
        Y0_local + Var1 + Var2;
  }

  if (current_state < NUMBER_X_EQUATIONS) {
    FLOATING_TYPE Var1 = 0;
    FLOATING_TYPE Var2 = 0;

#pragma unroll
    for (int i = 0;
         i < (NUMBER_X_EQUATIONS > NUMBER_U_VALUES ? NUMBER_X_EQUATIONS
                                                   : NUMBER_U_VALUES);
         i++) {

      if (i < NUMBER_X_EQUATIONS)
        Var1 += MatrA[current_wt * NUMBER_X_EQUATIONS * NUMBER_X_EQUATIONS +
                      current_state * NUMBER_X_EQUATIONS + i] *
                X_s[tid_local + i * WIND_TURBINES_PER_BLOCK];
      if (i < NUMBER_U_VALUES)
        Var2 += MatrB[current_wt * NUMBER_X_EQUATIONS * NUMBER_U_VALUES +
                      current_state * NUMBER_U_VALUES + i] *
                U_s[tid_local + i * WIND_TURBINES_PER_BLOCK];
    }

    Xprime_local = timestep * (Var1 + Var2);
    X_temp[tid] = X_s[tid] + Xprime_local;
  }
}

__syncthreads();

if (current_wt < number_wt) {
  if (current_state < NUMBER_X_EQUATIONS) {
    FLOATING_TYPE Var1 = 0;
    FLOATING_TYPE Var2 = 0;

#pragma unroll
    for (int i = 0;
         i < (NUMBER_X_EQUATIONS > NUMBER_U_VALUES ? NUMBER_X_EQUATIONS
                                                   : NUMBER_U_VALUES);
         i++) {

      if (i < NUMBER_X_EQUATIONS)
        Var1 += MatrA[current_wt * NUMBER_X_EQUATIONS * NUMBER_X_EQUATIONS +
                      current_state * NUMBER_X_EQUATIONS + i] *
                X_temp[tid_local + i * WIND_TURBINES_PER_BLOCK];
      if (i < NUMBER_U_VALUES)
        Var2 += MatrB[current_wt * NUMBER_X_EQUATIONS * NUMBER_U_VALUES +
                      current_state * NUMBER_U_VALUES + i] *
                Unext_s[tid_local + i * WIND_TURBINES_PER_BLOCK];
    }
    X_s[tid] += (timestep * (Var1 + Var2) + Xprime_local) / 2.0f;
    dX[current_wt * NUMBER_X_EQUATIONS + current_state] = X_s[tid];
  }
}

__syncthreads();
