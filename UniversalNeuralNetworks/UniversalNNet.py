# universal_neural_network.py

import numpy as np
import json
import os
import csv
import pickle
from scipy.stats import hypsecant
from numpy.linalg import norm

# Activation Functions and Derivatives

def identity(x): return x

def identity_(x): return np.ones_like(x)

def sigmoid(x): return 1./(1. + np.exp(-x))

def sigmoid_(x): return sigmoid(x)*(1-sigmoid(x))

def Tanh(x): return np.tanh(x)

def Tanh_(x): return (1+np.tanh(x))*(1-np.tanh(x))

def RBF(x): return np.exp(-np.power(np.array(x), 2))

def RBF_(x): return -2*(x*np.exp(-np.power(np.array(x), 2)))

def rbf2(x, c=0): return np.exp(-np.power((np.array(x)-np.array(c)), 2))

def rbf2_(x, c=0): return 2*(x-c)*np.exp(-np.power((np.array(x)-np.array(c)), 2))

def ReLU(x): return x * (x > 0)

def ReLU_(x): return 1. * (x > 0)

def softPlus(x, s=1): return np.log(1. + np.exp(s*x))

def softPlus_(x, s=1): return s/(1. + np.exp(-s*x))

def softReLU(x): return softPlus(x)

def softReLU_(x): return softPlus_(x)

def saturation(x):
    r = x.copy()
    r[abs(r) > 1] = np.sign(r[abs(r) > 1])
    return r

def saturation_(x):
    r = np.ones_like(x)
    r[abs(x) > 1] = 0
    return r

def hardlimit(x): return 1 * (x > 0)

def hardlimit_(x): return 0*x

def Sign(x): return np.sign(x)

def Sign_(x): return x*0

def Sin(x):
    r = np.sin(np.pi*x/2)
    for i, v in enumerate(x):
        if abs(v) > 1:
            r[i] = np.sign(v)
    return r

def Sin_(x):
    r = np.pi*np.cos(np.pi*x/2)/2
    for i, v in enumerate(x):
        if abs(v) > 1:
            r[i] = 0
    return r

def Cos(x):
    r = 0.5*(1.+np.cos(np.pi*x))
    for i, v in enumerate(x):
        if abs(v) > 1:
            r[i] = 0
    return r

def Cos_(x):
    r = 0.5*np.pi*np.sin(-np.pi*x)
    for i, v in enumerate(x):
        if abs(v) > 1:
            r[i] = 0
    return r

ACTIVATION_FUNCTION = {
    'linear': (identity, identity_, 'identity', 'identity_'),
    'tanh': (Tanh, Tanh_, 'tanh', 'tanh_'),
    'sigmoid': (sigmoid, sigmoid_, 'sigmoid', 'sigmoid_'),
    'relu': (ReLU, ReLU_, 'ReLU', 'ReLU_'),
    'softplus': (softPlus, softPlus_, 'softPlus', 'softPlus_'),
    'softReLU': (softReLU, softReLU_, 'softReLU', 'softReLU_'),
    'rbf': (RBF, RBF_, 'rbf', 'rbf_'),
    'saturation': (saturation, saturation_, 'saturation', 'saturation_'),
    'hardlimit': (hardlimit, hardlimit_, 'hardlimit', 'hardlimit_'),
    'sign': (Sign, Sign_, 'sign', 'sign_'),
    'sin': (Sin, Sin_, 'sin', 'sin_'),
    'cos': (Cos, Cos_, 'cos', 'cos_'),
}

# Loss Functions and Derivatives

def LpNorm(e, p=2, axis=1): return norm(e, p, axis=axis)

def LpNorm_(e, p=2, axis=None): return norm(np.sign(e)/norm(e, p), p-1)*np.sign(e)

def quadratic(e): return e*e/2.

def quadratic_(e): return e

def absolute(e): return np.abs(e)

def absolute_(e): return np.sign(e)

def LogCosh(e): return np.log(np.cosh(e))

def LogCosh_(e): return np.tanh(e)

def skewSoft(x, s=1): return np.log(1. + np.exp(s*x))

def skewSoft_(x, s=1): return s/(1. + np.exp(-s*x))

def skLogCosh(e, slope=5, gamma=2):
    gain = np.power(slope, np.tanh(gamma*e))
    cost = gain * LogCosh(e)
    return cost

def skLogCosh_(e, slope=5, gamma=2):
    gain = np.power(slope, np.tanh(gamma*e))
    gradient = gain * (np.tanh(e) + gamma * np.log(slope) * np.power(hypsecant.pdf(gamma*e), 2.))
    return gradient

def Max(e): return max(e)

def Max_(e): return [int(abs(i) < 1) for i in e]

def log_likelihood(features, target, weights):
    scores = np.dot(features, weights)
    ll = np.sum(target*scores - np.log(1. + np.exp(scores)))
    return ll

def crossEntropy(y, t):
    return -np.sum(np.multiply(t, np.log(y)) + np.multiply((1.-t), np.log(1.-y)))

LOSS_FUNCTION = {
    'quadratic': (quadratic, quadratic_, 'quadratic', 'quadratic_'),
    'logcosh': (LogCosh, LogCosh_, 'logcosh', 'logcosh_'),
    'sklogcosh': (skLogCosh, skLogCosh_, 'sklogcosh', 'sklogcosh_'),
    'softplus': (softPlus, softPlus_, 'softPlus', 'softPlus_'),
    'skewSoft': (skewSoft, skewSoft_, 'skewSoft', 'skewSoft_'),
    'absolute': (absolute, absolute_, 'absolute', 'absolute_'),
    # 'max': (Max, Max_, 'Max', 'Max_'),
    # 'LpNorm': (LpNorm, LpNorm_, 'LpNorm', 'LpNorm_'),
}

#-------------------------------------------------------------------------------
class UNN:
    def __init__(self, nodes,    connectivity  = None, activations = None, \
                 input_ops=None, cost_function = None, max_epochs  = 1000, \
                 window = 100):
        self.nodes         = nodes
        self.n_layers      = len(nodes) - 1
        self.connectivity  = connectivity if connectivity is not None else self.default_connectivity()
        self.weights       = {}
        self.biases        = {}
        self.past_states   = {}
        self.activations   = activations if activations else ['linear'] + ['tanh'] * (self.n_layers - 1) + ['linear']
        self.input_ops     = input_ops if input_ops else ['default'] * (self.n_layers + 1)
        self.cost_function = cost_function if cost_function else self.default_cost
        self.gradients     = {}
        self.epoch         = 0
        self.max_epochs    = max_epochs
        self.window        = window
        self.total_error   = float('inf')
        self.initialize_network()

    def default_connectivity(self):
        Connectivity = np.zeros((len(self.nodes), len(self.nodes)), dtype=bool)
        # default connectivity is an MLP (Multi-Layer Perceptron)
        # That is only feed-forward connections to the next layer only
        for i in range(len(self.nodes) - 1):
            Connectivity[i][i + 1] = True
        return Connectivity

    def initialize_network(self):
        for i in range(len(self.nodes)):
            self.biases[i] = np.random.randn(self.nodes[i])
        for i in range(len(self.nodes)):
            for j in range(len(self.nodes)):
                if self.connectivity[i][j]:
                    self.weights[(i, j)] = np.random.randn(self.nodes[j], self.nodes[i])
        for j in range(len(self.nodes)):
            self.past_states[j] = np.zeros(self.nodes[j])

    def _activation(self, x, func='linear'):
        if func == 'linear': return x
        if func == 'tanh': return np.tanh(x)
        if func == 'relu': return np.maximum(0, x)
        if func == 'cos': return np.cos(x)
        return x

    def _activation_derivative(self, x, func='linear'):
        if func == 'linear': return np.ones_like(x)
        if func == 'tanh': return 1 - np.tanh(x) ** 2
        if func == 'relu': return (x > 0).astype(float)
        if func == 'cos': return -np.sin(x)
        return np.ones_like(x)

    def _input_op(self, W, z, b, op='default'):
        if op == 'default': return np.dot(W, z) + b
        if op == 'product': return np.prod(W * z) / b
        if op == 'RBF':     return np.exp(-np.square(W - z).sum(axis=1) / b)
        if op == 'max':     return np.max(np.dot(W, z) + b)
        if op == 'min':     return np.min(np.dot(W, z) + b)
        if op == 'norm':    return np.linalg.norm(np.dot(W, z) + b)
        return np.dot(W, z) + b

    def forward(self, input_data):
        is_single_sample = len(input_data.shape) == 1
        if is_single_sample:
            input_data = input_data.reshape(-1, 1)
        batch_size = input_data.shape[1]

        self.states = {0: input_data}
        self.outputs = {0: input_data}

        for j in range(1, self.n_layers + 1):
            incoming = np.zeros((self.nodes[j], batch_size))
            for i in range(len(self.nodes)):
                if self.connectivity[i][j]:
                    z_prev = self.outputs[i] if i < j else self.past_states[i].reshape(-1, 1)
                    incoming += self._input_op(self.weights[(i, j)], z_prev, self.biases[j].reshape(-1, 1), self.input_ops[j])
            self.states[j] = incoming
            self.outputs[j] = self._activation(incoming, self.activations[j])

        for j in range(len(self.nodes)):
            self.past_states[j] = self.outputs[j][:, 0]  # Update with last time step
        return self.outputs[self.n_layers]

    def default_cost(self, prediction, target):
        error = prediction - target
        return np.sum(np.log(np.cosh(error))), error

    def compute_cost_and_gradient(self, X, Y):
        prediction = self.forward(X)
        cost, error = self.cost_function(prediction, Y)
        self.backpropagate(error)
        return cost

    def backpropagate(self, error):
        deltas = {}
        self.gradients = {'weights': {}, 'biases': {}}

        # Start with output layer
        j = self.n_layers
        delta = error * self._activation_derivative(self.states[j], self.activations[j])
        deltas[j] = delta

        for i in range(len(self.nodes)):
            if self.connectivity[i][j]:
                self.gradients['weights'][(i, j)] = np.dot(delta, self.outputs[i].T)
        self.gradients['biases'][j] = np.sum(delta, axis=1)

        # Backpropagate through hidden layers
        for j in reversed(range(1, self.n_layers)):
            delta = np.zeros_like(self.outputs[j])
            for k in range(j + 1, self.n_layers + 1):
                if self.connectivity[j][k]:
                    W_jk = self.weights[(j, k)]
                    delta += np.dot(W_jk.T, deltas[k])
            delta *= self._activation_derivative(self.states[j], self.activations[j])
            deltas[j] = delta

            for i in range(len(self.nodes)):
                if self.connectivity[i][j]:
                    self.gradients['weights'][(i, j)] = np.dot(delta, self.outputs[i].T)
            self.gradients['biases'][j] = np.sum(delta, axis=1)

    def update_weights(self, learning_rate=0.01):
        for (i, j), grad in self.gradients['weights'].items():
            self.weights[(i, j)] -= learning_rate * grad
        for j, grad in self.gradients['biases'].items():
            self.biases[j] -= learning_rate * grad

    def should_terminate(self, cost):
        if self.epoch >= self.max_epochs:
            print("Terminating: reached max epochs")
            return True
        if cost < 1e-4:
            print("Terminating: cost below threshold")
            return True
        return False

    def set_training_data(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def set_testing_data(self, x_test, y_test):
        self.x_test = x_test
        self.y_test = y_test

    def save_training_set(self, filename='training_set.csv', directory='.'):
        """Save training set (x_train + y_train) to a single CSV file."""
        if hasattr(self, 'x_train') and hasattr(self, 'y_train'):
            x = np.array(self.x_train)
            y = np.array(self.y_train)
            data = np.hstack((x, y))
            pd.DataFrame(data).to_csv(os.path.join(directory, filename), index=False)
            print(f"Training set saved to {filename}")
        else:
            print("Training data not found.")

    def save_testing_set(self, filename='testing_set.csv', directory='.'):
        """Save test set (x_test + y_test) to a single CSV file."""
        if hasattr(self, 'x_test') and hasattr(self, 'y_test'):
            x = np.array(self.x_test)
            y = np.array(self.y_test)
            data = np.hstack((x, y))
            pd.DataFrame(data).to_csv(os.path.join(directory, filename), index=False)
            print(f"Testing set saved to {filename}")
        else:
            print("Test data not found.")

    def train_from_file(self, training_file, testing_file=None):
        training_data = np.loadtxt(training_file, delimiter=',')
        X_train = training_data[:, :-self.nodes[-1]].T
        Y_train = training_data[:, -self.nodes[-1]:].T

        if testing_file is None:
            X_test, Y_test = X_train, Y_train
        else:
            testing_data = np.loadtxt(testing_file, delimiter=',')
            X_test = testing_data[:, :-self.nodes[-1]].T
            Y_test = testing_data[:, -self.nodes[-1]:].T

        self.train_with_data(X_train, Y_train)

    def train_with_data(self, X_train, Y_train):
        cost = self.compute_cost_and_gradient(X_train, Y_train)
        self.update_weights()
        self.total_error = cost
        self.epoch += 1

    def run(self, input_vector):
        return self.forward(input_vector)

    def save_network(self, filename):
        network_data = {
            'nodes': self.nodes,
            'connectivity': self.connectivity.tolist(),
            'activations': self.activations,
            'input_ops': self.input_ops,
            'weights': {str(k): v.tolist() for k, v in self.weights.items()},
            'biases': {str(k): v.tolist() for k, v in self.biases.items()}
        }
        with open(filename, 'w') as f:
            json.dump(network_data, f)

    def load_network(self, filename):
        with open(filename, 'r') as f:
            network_data = json.load(f)
        self.nodes = network_data['nodes']
        self.connectivity = np.array(network_data['connectivity'], dtype=bool)
        self.activations = network_data['activations']
        self.input_ops = network_data['input_ops']
        self.weights = {eval(k): np.array(v) for k, v in network_data['weights'].items()}
        self.biases = {int(k): np.array(v) for k, v in network_data['biases'].items()}
        self.past_states = {j: np.zeros(self.nodes[j]) for j in range(len(self.nodes))}

    def test(self):
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        def f(x):
            return -0.7 + 1.5 * np.sin(np.pi * x) + 0.5 * np.sin(3 * np.pi * x) + 2.5 * x**3

        def yt(x):
            return f(x) + (0.15 * np.ones_like(x) + 0.45 * np.abs(x) * np.cos(np.pi * x / 3))

        def yb(x):
            return f(x) - (0.25 * np.ones_like(x) + 0.15 * np.abs(x) * np.sin(np.pi * x / 2))

        # Create training data
        x_train = np.random.uniform(-1, 1, 1000)
        y_train = f(x_train) + np.random.normal(0, 1, len(x_train)) * (yt(x_train) - yb(x_train))
        x_train = x_train.reshape(1, -1)
        y_train = y_train.reshape(1, -1)

        # Create testing data
        x_test = np.random.uniform(-1, 1, 1000)
        y_test = f(x_test) + np.random.normal(0, 1, len(x_test)) * (yt(x_test) - yb(x_test))
        x_test = x_test.reshape(1, -1)
        y_test = y_test.reshape(1, -1)

        # Graph input/output
        x_graph = np.linspace(-1, 1, 101)
        y_graph = f(x_graph)

        # Set data and hyperparameters
        self.set_training_data(x_train, y_train)
        self.set_testing_data(x_test, y_test)
        self.n_epochs = 25000
        self.learning_rate = 0.001

        # Setup plot
        fig, ax = plt.subplots(figsize=(12, 8))
        line_true,           = ax.plot(x_graph, y_graph, 'r-', label='True Function f(x)', linewidth=2)
        scatter_points,      = ax.plot(x_train.flatten(), y_train.flatten(), 'go', markersize=2, label='Training Data')
        line_network_output, = ax.plot(x_graph, np.zeros_like(x_graph), 'k-', label='Network Output')
        ax.set_xlim([-1.1, 1.1])
        ax.set_ylim([-5, 5])
        ax.legend(loc='upper left')
        ax.set_title("UNN Output During Training")

        costs = []

        def update(frame):
            for _ in range(100):
                cost = self.train_one_epoch()
            costs.append(cost)
            y_pred = self.run(x_graph.reshape(1, -1)).flatten()
            line_network_output.set_ydata(y_pred)
            return line_network_output,

        anim = FuncAnimation(fig, update, frames=self.n_epochs // 100, interval=100, blit=True)
        plt.show()

        # Plot cost function in log-log
        plt.figure(figsize=(10, 6))
        plt.loglog(range(1, len(costs) + 1), costs, 'b-')
        plt.xlabel('Epochs')
        plt.ylabel('Cost')
        plt.title('Training Cost vs Epochs (log-log)')
        plt.grid(True)
        plt.show()

    def test_me(self, n_epochs=25000, learning_rate=0.001, n_steps=100):
        def f(x):
            return -0.7 + 1.5 * np.sin(np.pi * x) + 0.5 * np.sin(3 * np.pi * x) + 2.5 * x**3

        def yt(x): return f(x) + 0.5
        def yb(x): return f(x) - 0.5

        # Training and testing data
        x_train = np.random.uniform(-1, 1, 1000)
        y_train = f(x_train) + np.random.normal(0, 1, len(x_train)) * (yt(x_train) - yb(x_train))
        x_test  = np.random.uniform(-1, 1, 1000)
        y_test  = f(x_test)  + np.random.normal(0, 1, len(x_test))  * (yt(x_test) - yb(x_test))

        # Function graph
        x_graph = np.linspace(-1, 1, 101)
        y_graph = f(x_graph)

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        line_true, = ax.plot(x_graph, y_graph, 'r-', label='True Function f(x)', linewidth=3)
        scatter_points, = ax.plot(x_train, y_train, 'go', markersize=2, label='Training Data')
        line_network_output, = ax.plot(x_graph, np.zeros_like(x_graph), 'k-', label='Network Output')

        ax.set_xlim([-1.1, 1.1])
        ax.set_ylim([-5, 5])
        ax.legend(loc='upper left')

        # Create network
        n_input = 1
        n_hidden = 7
        n_output = 1

        self.initialize_network(n_input=n_input, n_hidden=n_hidden, n_output=n_output)

        cost_history = []

        def update(frame):
            nonlocal cost_history
            # Train one epoch
            x_input = x_train.reshape(1, -1)
            y_target = y_train.reshape(1, -1)

            self.forward(x_input)
            cost, grad = self.compute_cost_and_gradient(x_input, y_target)
            self.backpropagate(grad)
            self.update_weights(learning_rate=learning_rate)
            cost_history.append(cost)

            # Show network output on function graph
            if frame % n_steps == 0:
                y_pred = self.run(x_graph.reshape(1, -1))
                line_network_output.set_ydata(y_pred.flatten())
            return line_network_output,

        anim = FuncAnimation(fig, update, frames=n_epochs, interval=1, blit=True)
        plt.show()

        # Plot cost vs epochs after training
        plt.figure(figsize=(10, 6))
        plt.loglog(range(1, len(cost_history)+1), cost_history)
        plt.xlabel('Epochs (log scale)')
        plt.ylabel('Cost (log scale)')
        plt.title('Training Cost vs Epochs')
        plt.grid(True)
        plt.show()

    @staticmethod
    def test_UNN():
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        def f(x):
            return -0.7 + 1.5 * np.sin(np.pi * x) + 0.5 * np.sin(3 * np.pi * x) + 2.5 * x**3

        def yt(x): return 1. + 0*x
        def yb(x): return -1. - 0*x

        # Create training data
        x_train = np.random.uniform(-1, 1, 1000)
        y_train = f(x_train) + np.random.normal(0, 1, len(x_train)) * (yt(x_train) - yb(x_train))
        x_train = x_train.reshape(1, -1)
        y_train = y_train.reshape(1, -1)

        # Create testing data
        x_test = np.random.uniform(-1, 1, 1000)
        y_test = f(x_test) + np.random.normal(0, 1, len(x_test)) * (yt(x_test) - yb(x_test))
        x_test = x_test.reshape(1, -1)
        y_test = y_test.reshape(1, -1)

        # Graph input/output
        x_graph = np.linspace(-1, 1, 101)
        y_graph = f(x_graph)

        # Network configuration
        n_inputs = 1
        n_hidden = [7]
        n_outputs = 1
        nodes = [n_inputs] + n_hidden + [n_outputs]
        n_layers = len(nodes) - 1
        Connectivity = np.zeros((n_layers + 1, n_layers + 1), dtype=bool)

        # Full feedforward
        for i in range(n_layers):
            Connectivity[i][i+1] = True

        # Create network instance
        net = UNN(nodes=nodes, Connectivity=Connectivity)
        net.initialize_network()
        net.set_training_data(x_train, y_train)
        net.set_testing_data(x_test, y_test)
        net.n_epochs = 25000
        net.learning_rate = 0.001

        # Setup plot
        fig, ax = plt.subplots(figsize=(12, 8))
        line_true,           = ax.plot(x_graph, y_graph, 'r-', label='True Function f(x)', linewidth=2)
        scatter_points,      = ax.plot(x_train.flatten(), y_train.flatten(), 'go', markersize=2, label='Training Data')
        line_network_output, = ax.plot(x_graph, np.zeros_like(x_graph), 'k-', label='Network Output')
        ax.set_xlim([-1.1, 1.1])
        ax.set_ylim([-5, 5])
        ax.legend(loc='upper left')
        ax.set_title("UNN Output During Training")

        costs = []

        def update(frame):
            for _ in range(100):
                cost = net.train_one_epoch()
            costs.append(cost)
            y_pred = net.run(x_graph.reshape(1, -1)).flatten()
            line_network_output.set_ydata(y_pred)
            return line_network_output,

        anim = FuncAnimation(fig, update, frames=net.n_epochs // 100, interval=100, blit=True)
        plt.show()

        # Plot cost function in log-log
        plt.figure(figsize=(10, 6))
        plt.loglog(range(1, len(costs) + 1), costs, 'b-')
        plt.xlabel('Epochs')
        plt.ylabel('Cost')
        plt.title('Training Cost vs Epochs (log-log)')
        plt.grid(True)
        plt.show()

if __name__ == '__main__':
    # Network configuration
    n_inputs = 1
    n_hidden = [7]
    n_outputs = 1
    nodes = [n_inputs] + n_hidden + [n_outputs]
    n_layers = len(nodes) - 1

    # Connectivity matrix: simple feedforward
    Connectivity = np.zeros((n_layers + 1, n_layers + 1), dtype=bool)
    for i in range(n_layers):
        Connectivity[i][i + 1] = True

    # Create the UNN instance
    net = UNN(nodes=nodes, connectivity=Connectivity)
    net.initialize_network()

    # Run the test
    net.test()

