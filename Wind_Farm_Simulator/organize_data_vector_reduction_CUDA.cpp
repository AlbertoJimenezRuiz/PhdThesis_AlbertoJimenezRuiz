#include "main.h"
#include <algorithm>

int next_power_of_two(int x) {
  int power = 1;
  while (power < x)
    power *= 2;

  return power;
}

template <typename T>
std::vector<std::vector<T>> split_vector(const std::vector<T> &vec, size_t n) {
  std::vector<std::vector<T>> outVec;

  for (size_t i = 0; i < vec.size(); i += n) {
    auto end = std::min(i + n, vec.size());
    outVec.push_back(std::vector<T>(vec.begin() + i, vec.begin() + end));
  }

  return outVec;
}

void organize_data_vector_reduction_CUDA(
    bool groups_synchronized_using_cooperative_groups,
    const vector<vector<C_FLOATING_TYPE>> &input_full_groups_multipliers,
    const vector<vector<int>> &input_full_groups_positions,

    std::vector<C_FLOATING_TYPE> &multipliers, std::vector<int> &positions,
    std::vector<int> &element_sizes, std::vector<int> &number_elements,
    std::vector<int> &corresponding_element,

    const unsigned int blocksize) {
  vector<vector<C_FLOATING_TYPE>> input_group_multipliers;
  vector<vector<int>> input_group_position;
  vector<int> original_group_index;

  for (unsigned int i = 0; i < input_full_groups_positions.size(); i++) {
    if (!groups_synchronized_using_cooperative_groups) {
      assert(input_full_groups_positions[i].size() <= blocksize);
    }

    for (const auto &vec :
         split_vector(input_full_groups_multipliers[i], blocksize)) {
      input_group_multipliers.push_back(vec);
      original_group_index.push_back(i);
    }

    for (const auto &vec :
         split_vector(input_full_groups_positions[i], blocksize)) {
      input_group_position.push_back(vec);
    }
  }

  multipliers.clear();
  positions.clear();
  element_sizes.clear();
  number_elements.clear();
  corresponding_element.clear();

  while (input_group_position.size() != 0) {
    auto it =
        find_if(input_group_position.begin(), input_group_position.end(),
                [multipliers, blocksize](const vector<int> &tt) {
                  const int extended_size = next_power_of_two((int)tt.size());
                  const bool starts_aligned =
                      (multipliers.size() % extended_size) == 0;
                  const bool ends_in_same_block =
                      ((multipliers.size() / blocksize) ==
                       ((multipliers.size() + extended_size - 1) / blocksize));
                  return starts_aligned && ends_in_same_block;
                });

    long int partial_position = 0;

    if (it != input_group_position.end()) {
      partial_position = it - input_group_position.begin();
    } else {
      partial_position = 0;
    }

    std::vector<int> selected_positions =
        input_group_position[partial_position];
    int corresponding_group_number = original_group_index[partial_position];

    input_group_position.erase(input_group_position.begin() + partial_position);
    original_group_index.erase(original_group_index.begin() + partial_position);

    const unsigned int group_length = (unsigned int)selected_positions.size();
    const unsigned int group_length_ext = next_power_of_two(group_length);

    assert(group_length <= blocksize);
    assert(group_length_ext <= blocksize);

    while (multipliers.size() % group_length_ext != 0) {
      multipliers.push_back(C_FLOATING_TYPE(0));
      positions.push_back(-1);
      element_sizes.push_back(-1);
      number_elements.push_back(-1);
      corresponding_element.push_back(-1);
    }

    std::vector<C_FLOATING_TYPE> multipliers_selected_group =
        input_group_multipliers[partial_position];
    input_group_multipliers.erase(input_group_multipliers.begin() +
                                  partial_position);
    for (unsigned int i = 0; i < multipliers_selected_group.size(); i++) {
      multipliers.push_back(multipliers_selected_group[i]);
      positions.push_back(selected_positions[i]);
      element_sizes.push_back(group_length_ext);
      number_elements.push_back(i);
      corresponding_element.push_back((i == 0) ? corresponding_group_number
                                               : (-1));
    }

    for (unsigned int i = 0; i < group_length_ext - group_length; i++) {
      multipliers.push_back(C_FLOATING_TYPE(0));
      positions.push_back(-1);
      element_sizes.push_back(-1);
      number_elements.push_back(-1);
      corresponding_element.push_back(-1);
    }
  }

  if (blocksize > 0) {
    while ((multipliers.size() % blocksize) != 0) {
      multipliers.push_back(C_FLOATING_TYPE(0));
      positions.push_back(-1);
      element_sizes.push_back(-1);
      number_elements.push_back(-1);
      corresponding_element.push_back(-1);
    }
  }
}
