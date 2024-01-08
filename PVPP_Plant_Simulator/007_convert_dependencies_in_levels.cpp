#include "main.h"
#include "sparse_matrix.h"

static inline vector<vector<int>>
get_basic_levels(const vector<set<int>> &list_column_dependences) {

  vector<vector<int>> list_parallelizable_columns_by_level;

  // Vector containing, for each column, in which level it is.
  vector<int> level_where_column_is(list_column_dependences.size());

  // If it is not at any level, we set it to -1.
  std::fill(level_where_column_is.begin(), level_where_column_is.end(), -1);

  int classified_columns = 0;
  int current_level = 0; // Start by first level

  // If all columns have already been sorted, stop.
  while (classified_columns != (int)list_column_dependences.size()) {

    vector<int> new_level;

    // Go through all columns
    for (int i = 0; i < (int)list_column_dependences.size(); i++) {
      // Was it already classified?
      if (level_where_column_is[i] != -1)
        continue;

      bool has_unresolved_dependencies = true;
      for (const auto &l : list_column_dependences[i]) {

        if ((level_where_column_is[l] == -1) ||
            (level_where_column_is[l] == current_level)) {
          // If its dependence has not been classified.
          // (No columns to complete)
          // or
          // If the dependencies are already part of this level, this item
          // cannot be parallelized ye. Then this element depends on another
          // column which is not from a previous level.

          has_unresolved_dependencies = false;
          break;
        }
      }

      // If this item has no column that depends on it, then classify the column
      if (has_unresolved_dependencies) {
        level_where_column_is[i] = current_level;
        new_level.push_back(i);
        classified_columns++;
      }
    }

    // Add level to list of levels
    list_parallelizable_columns_by_level.push_back(new_level);
    current_level++;
  }

  return list_parallelizable_columns_by_level;
}

int chain_column_belongs_to(
    const vector<int> &position_level_each_dependence_belongs_to,
    const set<int> &set_dependencies) {

  int found_position = -1;

  for (const auto &dependence_candidate : set_dependencies) {
    int possible_position =
        position_level_each_dependence_belongs_to[dependence_candidate];

    if (possible_position == -1)
      continue;

    if (found_position == possible_position)
      continue;

    if (found_position != -1)
      return -1;

    found_position = possible_position;
  }

  assert(found_position != -1);

  return found_position;
}

bool get_resulting_chain_of_columns(
    const int current_level, const vector<int> &current_Basic_Level,
    const vector<set<int>> &list_column_dependences,
    const vector<int> &position_level_each_dependence_belongs_to,
    vector<int> &positions_hook_vector) {

  /*
   For each element of this basic level, which we will assume to be basic level
   2 (Elements 11 5 6) [See example comments]. Which rows should they be joined
   to (Result: 2 0 1). This is returned in positions_hook_vector. It also
   returns a boolean that indicates if a new level should be created(false), or
   if you can take the columns of this level and add it to the rows of the
   previous level. Reasons for not being able to merge the level with the rows
   of the previous level: One element depends on two rows. Two elements of the
   basic level depend on the same row. Special case: If it is the first level,
   it has no level below to merge with.
  */

  if (current_level ==
      0) // Special case, first level cannot join with previous level
  {
    return false;
  }

  // This vector stores which string should each column adhere to
  positions_hook_vector = vector<int>(current_Basic_Level.size());

  // For each column
  for (int idx = 0; idx < (int)current_Basic_Level.size(); idx++) {

    // Can this column be hooked to one and only one chain? If so, great, if
    // not, then this basic level cannot be fused

    int found_position = chain_column_belongs_to(
        position_level_each_dependence_belongs_to,
        list_column_dependences[current_Basic_Level[idx]]);

    if (found_position == -1) {
      return false;
    }

    // Are any other columns already hooked to this chain? If yes, then the
    // level cannot be fused.
    for (int idx2 = 0; idx2 < idx; idx2++) {
      if (found_position == positions_hook_vector[idx2]) {
        return false;
      }
    }

    // Add index of the string to corresponding positions_hook_vector
    positions_hook_vector[idx] = found_position;
  }

  // The level can be connected with the previous one
  return true;
}

vector<vector<list<int>>> convert_dependence_list_in_chains_of_columns(
    const vector<set<int>> &list_column_dependences) {

  // The entries tell, for each column, which columns are to be processed first
  // (parent column I call it).

  size_t n = list_column_dependences.size();

  vector<vector<int>> Basic_Levels = get_basic_levels(list_column_dependences);

  // chains of columns
  vector<vector<list<int>>> custom_levels;

  vector<int> position_level_each_dependence_belongs_to(n);
  for (int current_level = 0; current_level < (int)Basic_Levels.size();
       current_level++) {

    const auto &current_Basic_Level = Basic_Levels[current_level];

    vector<int> positions_hook_vector;

    bool can_fuse_level_with_previous_one = get_resulting_chain_of_columns(
        current_level, current_Basic_Level, list_column_dependences,
        position_level_each_dependence_belongs_to, positions_hook_vector);

    if (can_fuse_level_with_previous_one) {
      auto &level_previous_chain = custom_levels.back();

      for (int idx = 0; idx < (int)positions_hook_vector.size(); idx++) {

        const int current_column = current_Basic_Level[idx];
        const int hook_current_position = positions_hook_vector[idx];

        level_previous_chain[hook_current_position].push_back(current_column);

        position_level_each_dependence_belongs_to[current_column] =
            hook_current_position;
      }
    } else {

      // New level. Restore the vector That is, no column is in level.

      std::fill(position_level_each_dependence_belongs_to.begin(),
                position_level_each_dependence_belongs_to.end(), -1);

      vector<list<int>> res(current_Basic_Level.size());

      for (int sub_current_line = 0;
           sub_current_line < (int)current_Basic_Level.size();
           sub_current_line++) {
        const int &elem = current_Basic_Level[sub_current_line];

        // First element of each chain
        res[sub_current_line].push_back(elem);

        // Leave the array ready for the next iteration
        position_level_each_dependence_belongs_to[elem] = sub_current_line;
      }

      custom_levels.push_back(res);
    }
  }

  return custom_levels;
}
