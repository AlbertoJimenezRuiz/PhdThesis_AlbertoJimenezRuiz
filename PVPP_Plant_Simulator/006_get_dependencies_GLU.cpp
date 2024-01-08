#include "main.h"
#include "sparse_matrix.h"

// https://arxiv.org/abs/1908.00204
vector<set<int>>
get_dependencies_GLU_3_0(int n,
                         const vector<vector<int>> &fillin_dependencies) {
  vector<vector<int>> fillin_dependencies_transposed(n);

  for (unsigned int i = 0; i < fillin_dependencies.size(); i++) {
    for (const auto &j : fillin_dependencies[i]) {
      fillin_dependencies_transposed[j].push_back(i);
    }
  }

  vector<set<int>> dependencies_glu(n);

  for (int k = 0; k < n; k++) {
    set<int> &set_act = dependencies_glu[k];

    // Look up for all nonzeros in column k of U
    for (const auto &i : fillin_dependencies_transposed[k]) {
      if (i >= k)
        continue;
      // if column i of L is not empty:
      // Diagonal of matrix corresponds to U
      for (const auto &z : fillin_dependencies_transposed[i]) {
        if (i < z)
          continue;
        if (i >= n)
          continue;

        set_act.insert(i);
        break;
      }
    }
    // Look left for all non-zeros in row k of L
    for (const auto &i : fillin_dependencies_transposed[k]) {
      if (i >= k)
        continue;

      set_act.insert(i);
    }
  }

  return dependencies_glu;
}

vector<set<int>>
get_dependencies_matrix_L(int n,
                          const vector<vector<int>> &fillin_dependencies) {
  vector<set<int>> dependencies(n);

  for (int i = 0; i < (int)fillin_dependencies.size(); i++) {
    for (const auto &j : fillin_dependencies[i]) {
      if (j < i) {
        dependencies[i].insert(j);
      }
    }
  }
  return dependencies;
}

vector<set<int>>
get_dependencies_matrix_U(int n,
                          const vector<vector<int>> &fillin_dependencies) {
  vector<set<int>> dependencies(n);

  for (int i = 0; i < (int)fillin_dependencies.size(); i++) {
    for (const auto &j : fillin_dependencies[i]) {
      if (j > i) {
        dependencies[i].insert(j);
      }
    }
  }
  return dependencies;
}
