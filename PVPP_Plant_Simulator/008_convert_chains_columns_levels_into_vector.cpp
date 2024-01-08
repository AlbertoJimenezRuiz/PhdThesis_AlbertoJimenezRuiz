#include "main.h"
#include <map>

void convert_chains_columns_levels_into_vector(
    const vector<vector<list<int>>> &levels, vector<int> &start_levels_unified,
    vector<int> &start_chains_columns_unified,
    vector<int> &chains_columns_unified) {
  start_levels_unified.clear();
  start_chains_columns_unified.clear();
  chains_columns_unified.clear();

  for (const vector<list<int>> &current_level : levels) {
    // 1.
    start_levels_unified.push_back((int)start_chains_columns_unified.size());

    for (const list<int> &current_chain : current_level) {
      // 2.
      start_chains_columns_unified.push_back(
          (int)chains_columns_unified.size());

      for (const int &current_column : current_chain) {
        chains_columns_unified.push_back(current_column);
      }
    }
  }
  // 1.
  start_levels_unified.push_back((int)start_chains_columns_unified.size());
  // 2.
  start_chains_columns_unified.push_back((int)chains_columns_unified.size());
}
