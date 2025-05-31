import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import math
#from   scipy.linalg  import norm, pinv
from   scipy.stats   import logistic, hypsecant

# Set random seed for reproducibility
np.random.seed(42)

#-------------------------------------------------------------------------------

class ActivationFunction:
    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def tanh_derivative(x):
        return 1 - np.tanh(x) ** 2

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return ActivationFunction.sigmoid(x) * (1 - ActivationFunction.sigmoid(x))

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x):
        return np.where(x > 0, 1, 0)

    @staticmethod
    def linear(x):
        return x

    @staticmethod
    def linear_derivative(x):
        return np.ones_like(x)

    @staticmethod
    def rbf (x,c=0):
        return np.exp(-np.power((np.array(x)-np.array(c)),2))

    @staticmethod
    def rbf_derivative(x,c=0):
        return 2*(x-c)*np.exp(-np.power((np.array(x)-np.array(c)),2))

    @staticmethod
    def softPlus (x, s=1):
        return np.log(1. + np.exp( s*x))

    @staticmethod
    def softPlus_derivative(x, s=1):
        return    s/(1. + np.exp(-s*x))


#-------------------------------------------------------------------------------

class CostFunction:
    @staticmethod
    def sum_of_squares(e):
        return 0.5 * np.mean(e ** 2)

    @staticmethod
    def sum_of_squares_derivative(e):
        return e

    @staticmethod
    def log_cosh(e):
        return np.mean(np.log(np.cosh(e)))

    @staticmethod
    def log_cosh_derivative(e):
        return np.tanh(e)

    @staticmethod
    def skew_log_cosh(e, m=5, g=2):
        return np.mean(m ** (np.tanh(g * e)) * np.log(np.cosh(e)))

    @staticmethod
    def skew_log_cosh_derivative(e, m=5, g=2):
        return m * (np.tanh(e) + g*np.log(m)*np.power(hypsecant.pdf(g*e),2.))

    @staticmethod
    def  skLogCosh (e, slope = 5, gamma = 2):
                        gain = np.power(slope,np.tanh(gamma*e))
                        cost = gain*LogCosh(e)
                        return cost                    # cost function: skLogCosh
    @staticmethod
    def  skLogCosh_(e, slope = 5, gamma = 2):
                        gain = np.power(slope,np.tanh(gamma*e))
                        gradient = gain*(np.tanh(e) \
                                 + gamma*np.log(slope)*np.power(hypsecant.pdf(gamma*e),2.))
                        return gradient                # gradient of    skLogCosh

#-------------------------------------------------------------------------------

class NeuralNetwork:
    def __init__(self, n, xmin, xmax, ymin, ymax):
        self.n = n
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

        # Initialize hidden biases
        self.hidden_biases = np.linspace(xmin, xmax, n)

        # Active range determines input-to-hidden weights
        self.active_range = (xmax - xmin) / n
        self.input_to_hidden_weights = 1 / self.active_range * np.ones(n)[:,None]

        # Hidden-to-output weights equally spaced
        self.hidden_to_output_weights = np.linspace(ymin, ymax, n)

        # Output bias
        self.output_bias = (ymin + ymax) / 2

    def __init__(self, n_input, n_hidden, n_output, \
        activation_fn="tanh", cost_fn="log_cosh", learning_rate=0.001, \
        whichCase = "mean", xmin = -1, xmax = 1, ymin = -3, ymax = 4):
        sigmaNets ={"lower": 0.1, "mean": 1, "upper": 10}
        print(f"cost_fn parameter: {cost_fn}")
        self.W1 = np.random.randn(n_hidden, n_input)
        self.b1 = np.zeros(n_hidden)
        self.W2 = np.random.randn(n_output, n_hidden)
        self.b2 = np.zeros(n_output)
        self.learning_rate = learning_rate
        self.set_activation_function(activation_fn)
        self.set_cost_function(cost_fn)
        self.m = sigmaNets[whichCase]
        print(f"Creating MLP NeuralNetwork ( {n_input} X {n_hidden} X {n_output} )")
        print(f"with cost_fn = {self.cost} and whichCase = '{whichCase}', m= {self.m}")

    def set_activation_function(self, name):
        if name == "sigmoid":
            self.activation = ActivationFunction.sigmoid
            self.activation_derivative = ActivationFunction.sigmoid_derivative
        elif name == "relu":
            self.activation = ActivationFunction.relu
            self.activation_derivative = ActivationFunction.relu_derivative
        elif name == "rbf":
            self.activation = ActivationFunction.rbf
            self.activation_derivative = ActivationFunction.rbf_derivative
        elif name == "softPlus":
            self.activation = ActivationFunction.softPlus
            self.activation_derivative = ActivationFunction.softPlus_derivative
        else:
            self.activation = ActivationFunction.tanh
            self.activation_derivative = ActivationFunction.tanh_derivative

    def set_cost_function(self, name):
        name = name[0]
        if name == "sum_of_squares":
            self.cost = CostFunction.sum_of_squares
            self.cost_derivative = CostFunction.sum_of_squares_derivative
        elif name == "skew_log_cosh":
            self.cost = CostFunction.skew_log_cosh
            self.cost_derivative = CostFunction.skew_log_cosh_derivative
        elif name == "skLogCosh":
            self.cost = CostFunction.skew_log_cosh
            self.cost_derivative = CostFunction.skew_log_cosh_derivative
        else:
            self.cost = CostFunction.log_cosh
            self.cost_derivative = CostFunction.skew_log_cosh_derivative

    def forward(self, x):
        # Need to update for rbf and softPlus
        z1 = np.dot(self.W1, x[:, None].T) + self.b1[:, None] * np.ones_like(x[:, None].T)
        a1 = self.activation(z1)
        z2 = np.dot(self.W2, a1) + self.b2
        a2 = ActivationFunction.linear(z2)
        return a1, a2

    def backward(self, x, y, a1, a2, m=2, g=0.5):
        # Need to update for rbf and softPlus
        e = a2 - y
        if self.cost == CostFunction.skew_log_cosh:
            cost = self.cost(e, self.m, g)
        else:
            cost = np.mean(self.cost(e))

        d2 = self.cost_derivative(e)
        dW2 = np.outer(d2, a1)
        db2 = d2

        d1 = np.dot(self.W2.T, d2) * self.activation_derivative(a1)
        dW1 = np.outer(d1, x)
        db1 = d1

        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1[:, 0]
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2[:, 0]

        return cost

#-------------------------------------------------------------------------------

def f(x):
    #return 1.7 * np.ones_like(x)
    return -0.7 + 1.5 * np.sin(np.pi * x) + 0.5 * np.sin(3 * np.pi * x) + 2.5 * x**3

def yt(x):
    return f(x) + (1.25 * np.ones_like(x) + 0.45 * np.abs(x) * np.cos(np.pi * x / 3))

def yb(x):
    return f(x) - (2.35 * np.ones_like(x) + 0.15 * np.abs(x) * np.sin(np.pi * x / 2))

#-------------------------------------------------------------------------------

def animate(epoch):
    global printCost, learningDF
    ax.clear()
    # randomize training points
    idx = np.random.randint(0, len(x_train))
    x_batch = np.array([x_train[idx]])
    y_batch = np.array([y_train[idx]])

    #----------------------------------------------------
    for _ in range(epoch_steps):
        #-------------------------------------------------
        a1, a2 = network_upper.forward(x_batch)
        cost   = network_upper.backward(x_batch, y_batch, a1, a2)
        #-------------------------------------------------
        a1, a2 = network_lower.forward(x_batch)
        cost   = network_lower.backward(x_batch, y_batch, a1, a2)
        #-------------------------------------------------
        a1, a2 = network_mean.forward(x_batch)
        cost   = network_mean.backward(x_batch, y_batch, a1, a2)
        #-------------------------------------------------

    learningDF = pd.concat([learningDF, \
                 pd.DataFrame({'epochs': [epoch], 'cost': [cost]})], ignore_index=True)

    #----------------------------------------------------
    _, y_mean = network_mean.forward(x_test)
    _, y_upper = network_upper.forward(x_test)
    _, y_lower = network_lower.forward(x_test)
    ax.plot(x_graph, y_graph, 'r-', label='True Function f(x)', linewidth=3)
    ax.plot(x_graph, yt(x_graph), 'b--', label='Upper Bound yt(x)', linewidth=1)
    ax.plot(x_graph, yb(x_graph), 'b--', label='Lower Bound yb(x)', linewidth=1)
    ax.plot(x_train, y_train, 'go', markersize=2, label='Training Data')
    ax.plot(x_test, y_mean.T,  'k-', label='Network Mean',  linewidth=2)
    ax.plot(x_test, y_upper.T, 'g-', label='Network Upper', linewidth=2)
    ax.plot(x_test, y_lower.T, 'm-', label='Network Lower', linewidth=2)
    #----------------------------------------------------
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-5, 5])
    ax.legend(loc='upper left')

    if printCost: print(f'Epoch: {epoch:5d}, Cost: {cost}')

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

if __name__ == '__main__':
    global printCost, learningDF

    #--------------------------------------------------------------------------#
    # Create data set for training the NNet
    x_train = np.random.uniform(-1, 1, 1000)
    y_train = f(x_train) + np.random.normal(0, 1, len(x_train)) * (yt(x_train) - yb(x_train))
    # Create data to plot the target function
    x_graph = np.linspace(-1, 1, 101)
    y_graph = f(x_graph)

    # Create plot to show target function, with confidence bounds,
    # training points, and network output during animation
    fig, ax = plt.subplots()
    line_true, = ax.plot(x_graph, y_graph, 'r-', label='True Function f(x)', linewidth=3)
    line_bounds_upper, = ax.plot(x_graph, yt(x_graph), 'b--', label='Upper Bound yt(x)', linewidth=1)
    line_bounds_lower, = ax.plot(x_graph, yb(x_graph), 'b--', label='Lower Bound yb(x)', linewidth=1)
    scatter_points, = ax.plot(x_train, y_train, 'go', markersize=2, label='Training Data')
    line_network_output, = ax.plot(x_graph, np.zeros_like(x_graph), 'k-', label='Network Output')

    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-5, 5])
    ax.legend(loc='upper left')

    #--------------------------------------------------------------------------#
    # Network parameters
    n_input  = 1
    n_hidden = 9
    n_output = 1

    # Training loop and animation
    n_epochs = 1000000        # Total number of training epochs
    learning_rate = 0.00001   # learning rate
    epoch_steps = 1000        # how often to update plot

    printCost = True        # Print intermediate results?
    samples = math.ceil(n_epochs/epoch_steps)
    learningDF = pd.DataFrame(columns=['epochs', 'cost'])

    #--------------------------------------------------------------------------#
    # Create 3LP NNet.
    # Hidden layer uses tanh activation function
    # Output layer is linear
    # Cost function is logcosh(e)
    activation_fn="tanh"
    cost_fn="skew_log_cosh",   # m = {0.1, 1, 10}
    cost_fn_mean="skew_log_cosh",   # m = 1
    cost_fn_upper="skew_log_cosh",  # m = 10
    cost_fn_lower="skew_log_cosh",  # m = 0.1

    #-- mean network
    network_mean = NeuralNetwork(n_input, n_hidden, n_output, \
    activation_fn, cost_fn_mean, learning_rate, "mean")

    #-- upper network
    network_upper = NeuralNetwork(n_input, n_hidden, n_output, \
    activation_fn, cost_fn_upper, learning_rate, "upper")

    #-- lower network
    network_lower = NeuralNetwork(n_input, n_hidden, n_output, \
    activation_fn, cost_fn_lower, learning_rate, "lower")

    x_test = x_graph
    _, y_test = network_mean.forward(x_test)


    #--------------------------------------------------------------------------#
    ani = animation.FuncAnimation(fig, animate, frames=range(0, n_epochs+1, epoch_steps), repeat=False)
    plt.show()

    # Plot Learning Cost evolution
    plt.semilogy(learningDF['epochs'], learningDF['cost'], label='Cost Function')
    plt.xlabel('epochs')
    plt.ylabel('cost')
    plt.title('Plot of Cost Function')
    plt.legend()
    plt.show()
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
