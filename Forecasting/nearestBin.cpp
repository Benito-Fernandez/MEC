/*
Co-Pilot prompt:
Given a vector, p, of n dimensions, and an n-dimensional NDarray U, which lower index U[0,0,..,0] represents a minimum vector, pMin, and the last index U[n,n,...,n] represents a maximum vector, pMax, determine the index where p will fit the in nearest bin. Each dimension is divided into bins of equal size, so U[0,0,...,0] = pMin, U[n,n,...,n] = pMax, and U[n,0,...,0] = [pMax[0], pMin[1:n]] 

Co-Pilot response:
Here's a C++ code snippet to determine the index where a vector p will fit in the nearest bin in an n-dimensional NDarray U, given that each dimension is divided into bins of equal size: 

 <code>

This code defines a findNearestBin function that calculates the nearest bin indices for a given vector p based on the minimum vector pMin, maximum vector pMax, and the number of bins nBins. The main function demonstrates how to use this function with example data. 

 */

#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

vector<int> findNearestBin(const vector<double>& p, const vector<double>& pMin, const vector<double>& pMax, int nBins) {
    int n = p.size();
    vector<int> indices(n);

    for (int i = 0; i < n; ++i) {
        double binSize = (pMax[i] - pMin[i]) / (nBins - 1);
        indices[i] = round((p[i] - pMin[i]) / binSize);
        indices[i] = max(0, min(indices[i], nBins - 1)); // Ensure indices are within bounds
    }

    return indices;
}

int main() {
    // Example data
    vector<double> p = {2.5, 3.5, 4.5};
    vector<double> pMin = {0.0, 0.0, 0.0};
    vector<double> pMax = {5.0, 5.0, 5.0};
    int nBins = 6;

    vector<int> indices = findNearestBin(p, pMin, pMax, nBins);

    cout << "Nearest bin indices: ";
    for (const auto& index : indices) {
        cout << index << " ";
    }
    cout << endl;

    return 0;
}
