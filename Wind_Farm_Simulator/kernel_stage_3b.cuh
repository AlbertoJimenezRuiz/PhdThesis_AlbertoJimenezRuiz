if (threadIdx.x == 0) {
  cuComplexType current_injected_complex_temp = CREATE_COMPLEX_CUDA(0, 0);

  for (unsigned int i = 0; i < first_column_size; i++) {
    current_injected_complex_temp = ADD_COMPLEX_CUDA(
        current_injected_complex_temp,
        MULTIPLY_COMPLEX_CUDA(
            first_column[i],
            get_V0_previous_buses(
                first_column_pos[i], unified_input,
                temporary_vector_2, // Reduction to obtain the value of x before
                                    // substracting the union
                temporary_vector_3, // Reduction to obtain the first value of
                                    // the union
                ad_multiplier, temporary_value_1_constant, number_buses)));
  }

  current_injected_complex_temp = ADD_COMPLEX_CUDA(
      current_injected_complex_temp, temporary_vector_2[number_buses - 1]);
  global_injected_current[2 * current_timestep + 0] =
      current_injected_complex_temp.x;
  global_injected_current[2 * current_timestep + 1] =
      current_injected_complex_temp.y;
}
