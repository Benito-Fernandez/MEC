#include <iostream>
#include <vector>
#include <cmath>
#include <random>

using namespace std;

class MLP {
public:
    MLP(int inputDim, int hiddenDim, int outputDim);
    vector<double> forward(const vector<double>& input);
    double costFunction(const vector<double>& output, const vector<double>& target, double m, double delta, bool isTop, bool isBottom);

private:
    vector<vector<double>> weights1;
    vector<vector<double>> weights2;
    vector<double> tanh(const vector<double>& x);
    vector<double> dot(const vector<vector<double>>& matrix, const vector<double>& vec);
};

MLP::MLP(int inputDim, int hiddenDim, int outputDim) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(-1.0, 1.0);

    weights1.resize(hiddenDim, vector<double>(inputDim));
    weights2.resize(outputDim, vector<double>(hiddenDim));

    for (auto& row : weights1) {
        for (auto& val : row) {
            val = dis(gen);
        }
    }

    for (auto& row : weights2) {
        for (auto& val : row) {
            val = dis(gen);
        }
    }
}

vector<double> MLP::forward(const vector<double>& input) {
    vector<double> hidden = tanh(dot(weights1, input));
    vector<double> output = dot(weights2, hidden);
    return output;
}

double MLP::costFunction(const vector<double>& output, const vector<double>& target, double m, double delta, bool isTop, bool isBottom) {
    double cost = 0.0;
    for (size_t i = 0; i < output.size(); ++i) {
        double e = output[i] - target[i];
        double factor = 1.0;
        if (isTop) {
            factor = pow(m, tanh(e / delta));
        } else if (isBottom) {
            factor = pow(1.0 / m, tanh(e / delta));
        }
        cost += factor * log(cosh(e));
    }
    return cost;
}

vector<double> MLP::tanh(const vector<double>& x) {
    vector<double> result(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = std::tanh(x[i]);
    }
    return result;
}

vector<double> MLP::dot(const vector<vector<double>>& matrix, const vector<double>& vec) {
    vector<double> result(matrix.size());
    for (size_t i = 0; i < matrix.size(); ++i) {
        result[i] = 0.0;
        for (size_t j = 0; j < vec.size(); ++j) {
            result[i] += matrix[i][j] * vec[j];
        }
    }
    return result;
}

int main() {
    int inputDim = 3;
    int hiddenDim = 5;
    int outputDim = 2;

    MLP mlpMiddle(inputDim, hiddenDim, outputDim);
    MLP mlpTop(inputDim, hiddenDim, outputDim);
    MLP mlpBottom(inputDim, hiddenDim, outputDim);

    vector<double> input = {1.0, 2.0, 3.0};
    vector<double> target = {0.5, -0.5};

    double m = 2.0;
    double delta = 1.0;

    vector<double> outputMiddle = mlpMiddle.forward(input);
    vector<double> outputTop = mlpTop.forward(input);
    vector<double> outputBottom = mlpBottom.forward(input);

    double costMiddle = mlpMiddle.costFunction(outputMiddle, target, m, delta, false, false);
    double costTop = mlpTop.costFunction(outputTop, target, m, delta, true, false);
    double costBottom = mlpBottom.costFunction(outputBottom, target, m, delta, false, true);

    cout << "Output Middle: ";
    for (const auto& val : outputMiddle) {
        cout << val << " ";
    }
    cout << endl;

    cout << "Cost Middle: " << costMiddle << endl;

    cout << "Output Top: ";
    for (const auto& val : outputTop) {
        cout << val << " ";
    }
    cout << endl;

    cout << "Cost Top: " << costTop << endl;

    cout << "Output Bottom: ";
    for (const auto& val : outputBottom) {
        cout << val << " ";
    }
    cout << endl;

    cout << "Cost Bottom: " << costBottom << endl;

    return 0;
}
