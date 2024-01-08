#include "main.h"

vector<int>
get_max_numbers_per_column(const vector<vector<list<int>>> &levels_with_strings,
                           const vector<vector<int>> &nonzero_values) {
  vector<int> max_elements_per_column;

  for (unsigned int i = 0; i < levels_with_strings.size(); i++) {
    const auto &current_level = levels_with_strings[i];

    long unsigned int current_max_number_elements_per_columns = 0;

    for (const auto &current_chain : current_level) {
      for (const auto &current_column : current_chain) {

        // It is symmetrical. This will work
        long unsigned int possible_current_max_number_elements_per_columns =
            nonzero_values[current_column].size();

        if (possible_current_max_number_elements_per_columns >
            current_max_number_elements_per_columns)
          current_max_number_elements_per_columns =
              possible_current_max_number_elements_per_columns;
      }
    }

    max_elements_per_column.push_back(
        (unsigned int)current_max_number_elements_per_columns);
  }
  return max_elements_per_column;
}
