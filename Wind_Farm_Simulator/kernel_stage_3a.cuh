
if (tid_stage1_stage2a_stage3a < number_connections) {
  cuComplexType *I0_previous = unified_input + 1 + number_wt;
  cuComplexType *V0_previous =
      unified_input + 1 + number_wt + number_connections;

  int n1 = line_node_input[tid_stage1_stage2a_stage3a];
  int n2 = line_node_output[tid_stage1_stage2a_stage3a];

  V0_previous[tid_stage1_stage2a_stage3a] = get_V0_previous_buses(
      n1, unified_input,
      temporary_vector_2, // Reduction to obtain the value of x before
                          // substracting the union
      temporary_vector_3, // Reduction to obtain the first value of the union
      ad_multiplier, temporary_value_1_constant, number_buses);

  if (n2 != -1) {
    V0_previous[tid_stage1_stage2a_stage3a] = SUBTRACT_COMPLEX_CUDA(
        V0_previous[tid_stage1_stage2a_stage3a],

        get_V0_previous_buses(
            n2, unified_input,
            temporary_vector_2, // Reduction to obtain the value of x before
                                // substracting the union
            temporary_vector_3, // Reduction to obtain the first value of the
                                // union
            ad_multiplier, temporary_value_1_constant, number_buses));
  }

  I0_previous[tid_stage1_stage2a_stage3a] = ADD_COMPLEX_CUDA(
      MULTIPLY_COMPLEX_CUDA(I0_previous[tid_stage1_stage2a_stage3a], I0_mult1),
      MULTIPLY_COMPLEX_CUDA(V0_previous[tid_stage1_stage2a_stage3a], I0_mult2));
}
