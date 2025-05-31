/*
 Initialize n-dimensional array with n_bins per dimension
 the lowest vertex given by p_min and the highest vertex by p_max
 */

#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

void initialize_ndarray_recursive(vector<vector<double>>& array, const vector<double>& x_min, const vector<double>& x_max, const vector<int>& indices, int dim, int n) {
    if (dim == x_min.size()) {
        vector<double> point(x_min.size());
        for (int i = 0; i < x_min.size(); ++i) {
            double bin_size = (x_max[i] - x_min[i]) / (n - 1);
            point[i] = x_min[i] + indices[i] * bin_size;
        }
        array.push_back(point);
        return;
    }

    for (int i = 0; i < n; ++i) {
        vector<int> new_indices = indices;
        new_indices[dim] = i;
        initialize_ndarray_recursive(array, x_min, x_max, new_indices, dim + 1, n);
    }
}

vector<vector<double>> initialize_ndarray(const vector<double>& x_min, const vector<double>& x_max, int n) {
    vector<vector<double>> array;
    vector<int> indices(x_min.size(), 0);
    initialize_ndarray_recursive(array, x_min, x_max, indices, 0, n);
    return array;
}

int main() {
    // Example data
    vector<double> x_min = {0, 3, -2};
    vector<double> x_max = {5, 7, 2};
    int n = 6;

    vector<vector<double>> array = initialize_ndarray(x_min, x_max, n);

    for (const auto& point : array) {
        cout << "[";
        for (const auto& val : point) {
            cout << val << " ";
        }
        cout << "]" << endl;
    }

    return 0;
}
