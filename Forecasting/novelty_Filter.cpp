#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace std;

struct Point {
    vector<double> coordinates;
};

double euclideanDistance(const Point& p1, const Point& p2) {
    double sum = 0.0;
    for (size_t i = 0; i < p1.coordinates.size(); ++i) {
        sum += pow(p1.coordinates[i] - p2.coordinates[i], 2);
    }
    return sqrt(sum);
}

vector<Point> generateCenters(const vector<Point>& dataSet, int nCenters) {
    vector<Point> centers;
    size_t dimensions = dataSet[0].coordinates.size();
    vector<double> minVals(dimensions, numeric_limits<double>::max());
    vector<double> maxVals(dimensions, numeric_limits<double>::lowest());

    // Find the min and max values for each dimension
    for (const auto& point : dataSet) {
        for (size_t i = 0; i < dimensions; ++i) {
            minVals[i] = min(minVals[i], point.coordinates[i]);
            maxVals[i] = max(maxVals[i], point.coordinates[i]);
        }
    }

    // Generate centers
    for (int i = 0; i < nCenters; ++i) {
        Point center;
        for (size_t j = 0; j < dimensions; ++j) {
            double step = (maxVals[j] - minVals[j]) / (nCenters - 1);
            center.coordinates.push_back(minVals[j] + i * step);
        }
        centers.push_back(center);
    }

    return centers;
}

bool isNovel(const Point& newPoint, const vector<Point>& centers, double threshold) {
    for (const auto& center : centers) {
        if (euclideanDistance(newPoint, center) < threshold) {
            return false;
        }
    }
    return true;
}

int main() {
    vector<Point> dataSet = {
        {{1.0, 2.0, 3.0}},
        {{4.0, 5.0, 6.0}},
        {{7.0, 8.0, 9.0}}
    };

    int nCenters = 3;
    double threshold = 5.0;

    vector<Point> centers = generateCenters(dataSet, nCenters);

    Point newPoint = {{10.0, 11.0, 12.0}};

    if (isNovel(newPoint, centers, threshold)) {
        cout << "The new point is novel." << endl;
    } else {
        cout << "The new point is not novel." << endl;
    }

    return 0;
}
