#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Filename: BRFnnet.py
#-------------------------------------------------------------------------------
# Name:        BRFnnet
# Purpose:
#
# Author:      Benito R. Fernandez
#
# Created:     03/04/2025
# Copyright:   (c) benito fernandez 2025
# Licence:     MIT
#-------------------------------------------------------------------------------

'''-------------------------------------------------------------------------'''
import numpy as np
import csv
import matplotlib.pyplot as plt
from   scipy.stats   import logistic, hypsecant
import math, time
"""-------------------------------------------------------------------------"""

"""
    ---------------------------------------------------------------------------
    ACTIVATION FUNTIONS
    ---------------------------------------------------------------------------
"""
def     identity (x): return (x)
def     identity_(x): return (np.ones_like(x))

def      sigmoid (x): return 1./(1. + np.exp(-x))        # activation function: sigmoid
def      sigmoid_(x): return sigmoid(x)*(1-sigmoid(x))   # derivative of        sigmoid

def         Tanh (x): return np.tanh(x)                  # activation function: Tanh
def         Tanh_(x): return (1+np.tanh(x))*(1-np.tanh(x))# derivative of       Tanh

def          RBF (x): return np.exp(-np.power(np.array(x),2))
def          RBF_(x): return -2*(x*np.exp(-np.power(np.array(x),2)))

def     rbf2 (x,c=0): return np.exp(-np.power((np.array(x)-np.array(c)),2))
def     rbf2_(x,c=0): return 2*(x-c)*np.exp(-np.power((np.array(x)-np.array(c)),2))

def         ReLU (x): return x  * (x > 0)                # activation function: ReLU
def         ReLU_(x): return 1. * (x > 0)                # derivative of        ReLU

def softPlus (x, s=1): return np.log(1. + np.exp( s*x))
def softPlus_(x, s=1): return    s/(1. + np.exp(-s*x))

def    softReLU (x): return softPlus (x)
def    softReLU_(x): return softPlus_(x)

def  saturation (x):                                    # activation function: saturation
                     r = x
                     r[abs(r)>1] = np.sign(r[abs(r)>1])
                     return r
def  saturation_(x):                                    # derivative of        saturation
                     r = np.ones_like(x)
                     r[abs(x)>1] = 0
                     return r

def   hardlimit (x): return 1 * (x > 0)                 # activation function: hardlimit
def   hardlimit_(x): return 0*x                         # derivative of        hardlimit

def        Sign (x): return np.sign(x)                  # activation function: Sign
def        Sign_(x): return x*0                         # derivative of        Sign

def         Sin (x):                                    # activation function: Sin
                     r = np.sin(np.pi*x/2)
                     for i,v in enumerate(x):
                         if abs(v)>1:
                             r[i] = np.sign(v)
                     return r
def         Sin_(x):                                    # derivative of        Sin
                     r = np.pi*np.cos(np.pi*x/2)/2
                     for i,v in enumerate(x):
                         if abs(v)>1:
                             r[i] = 0
                     return r

def         Cos (x):                                    # activation function: Cos
                     r = 0.5*(1.+np.cos(np.pi*x))
                     for i,v in enumerate(x):
                         if abs(v)>1:
                             r[i] = 0
                     return r
def         Cos_(x):                                    # derivative of        Cos
                     r = 0.5*np.pi*np.sin(-np.pi*x)
                     for i,v in enumerate(x):
                         if abs(v)>1:
                             r[i] = 0
                     return r

ACTIVATION_FUNCTION = {
        'linear'    :  (identity   ,   identity_, 'identity'  , 'identity_'  ),
        'tanh'      :  (Tanh       ,       Tanh_, 'tanh'      ,     'tanh_'  ),
        'sigmoid'   :  (sigmoid    ,    sigmoid_, 'sigmoid '  ,  'sigmoid_'  ),
        'relu'      :  (ReLU       ,       ReLU_, 'ReLU'      ,     'ReLU_'  ),
        'softplus'  :  (softPlus   ,   softPlus_, 'softPlus'  ,   'softPlus_'),
        'softReLU'  :  (softReLU   ,   softReLU_, 'softReLU'  ,   'softReLU_'),
        'rbf'       :  (RBF        ,        RBF_, 'rbf'       ,      'rbf_'  ),
        'saturation':  (saturation , saturation_, 'saturation', 'saturation_'),
        'hardlimit' :  (hardlimit  ,  hardlimit_, 'hardlimit' , 'hardlimit_' ),
        'sign'      :  (Sign       ,       Sign_, 'sign'      , 'sign_'      ),
        'sin'       :  (Sin        ,        Sin_, 'sin'       , 'sin_'       ),
        'cos'       :  (Cos        ,        Cos_, 'cos'       , 'cos_'       ),
        }

"""
    ---------------------------------------------------------------------------
    LOSS/COST FUNTIONS
    ---------------------------------------------------------------------------
"""
def  LpNorm (e, p=2, axis = 1):
                    return norm(e,p, axis = axis)
def  LpNorm_(e, p=2, axis = None):
                    return norm(np.sign(e)/norm(e,p),p-1)*np.sign(e)

def quadratic (e):  return e*e/2. # LpNorm(e,2)
def quadratic_(e):  return e

def  absolute (e):  return np.abs(e)
def  absolute_(e):  return np.sign(e)

def   LogCosh (e):  return np.log(np.cosh(e))      # cost function: LogCosh
def   LogCosh_(e):  return np.tanh(e)              # gradient of    LogCosh

def skewSoft (x, s=1): return np.log(1. + np.exp( s*x))
def skewSoft_(x, s=1): return    s/(1. + np.exp(-s*x))

def  skLogCosh (e, slope = 5, gamma = 2):
                    gain = np.power(slope,np.tanh(gamma*e))
                    cost = gain*LogCosh(e)
                    return cost                    # cost function: skLogCosh
def  skLogCosh_(e, slope = 5, gamma = 2):
                    gain = np.power(slope,np.tanh(gamma*e))
                    gradient = gain*(np.tanh(e) \
                             + gamma*np.log(slope)*np.power(hypsecant.pdf(gamma*e),2.))
                    return gradient                # gradient of    skLogCosh

def       Max (e):  return max(e) # LpNorm(e,np.inf)
def       Max_(e):  return [int(abs(i)<1) for i in e]

def log_likelihood(features, target, weights):
    scores = np.dot(features, weights)
    ll = np.sum( target*scores - np.log(1. + np.exp(scores)) )
    return ll

def crossEntropy(y, t):
    return - np.sum(np.multiply(t, np.log(y)) + np.multiply((1.-t), np.log(1.-y)))

LOSS_FUNCTION = {
        'quadratic':(quadratic , quadratic_, 'quadratic' , 'quadratic_'),
        'logcosh'  :(LogCosh   ,   LogCosh_, 'logcosh'   ,   'logcosh_'),
        'sklogcosh':(skLogCosh , skLogCosh_, 'sklogcosh' , 'sklogcosh_'),
        'softplus' :(softPlus  ,  softPlus_, 'softPlus'  ,  'softPlus_'),
        'skewSoft' :(skewSoft  ,  skewSoft_, 'skewSoft'  ,  'skewSoft_'),
        'absolute' :(absolute  ,  absolute_, 'absolute'  ,  'absolute_'),
#        'max'      :(Max       ,       Max_, 'Max'       ,       'Max_'),
#        'LpNorm'   :(LpNorm    ,    LpNorm_, 'LpNorm'    ,    'LpNorm_'),
        }
'''-------------------------------------------------------------------------'''

plt.rc_context({'axes.edgecolor'  :'red',
                'figure.facecolor':'white',
                'font.family'     :'Parallax',
                'font.weight'     :'heavy',
              })

"""
    ---------------------------------------------------------------------------
    SAMPLING METHOD
    ---------------------------------------------------------------------------
"""
def next_batch(X, Y, batchSize):
    for i in np.arange(0, X.shape[0], batchSize):
        yield (X[i:i + batchSize], Y[i:i + batchSize])


"""
    ---------------------------------------------------------------------------
    Weights Class
    ---------------------------------------------------------------------------
"""
class Weights:
    #-------------------------------------------------------------
    def __init__(self) -> None:
        self.data = {}

    #-------------------------------------------------------------
    def __getitem__(self, indices: tuple[int, int]) -> np.ndarray:
        """Enable dictionary-like access: weights[index1, index2]."""
        value = self.data.get(indices, None)
        if value is None:
            raise KeyError(f"No weights found for indices {indices}.")
        return value

    #-------------------------------------------------------------
    def __call__(self, index1: int, index2: int) -> np.ndarray:
        """Allow callable access: weights(index1, index2)."""
        value = self.data.get((index1, index2), None)
        if value is None:
            raise KeyError(f"No weights found for indices ({index1}, {index2}).")
        return value

    #-------------------------------------------------------------
    def set_weights(self, index1: int, index2: int, value: np.ndarray) -> None:
        self.data[(index1, index2)] = value

    #-------------------------------------------------------------
    def remove_weights(self, index1: int, index2: int) -> None:
        try:
            del self.data[(index1, index2)]
        except KeyError:
            raise KeyError(f"No existing weights for indices ({index1}, {index2}) to remove.")

    #-------------------------------------------------------------
    def print_weights(self) -> None:
        print("Current weights:")
        for (index1, index2), value in self.data.items():
            print(f"From layer {index1} to layer {index2}: {value}")

    #-------------------------------------------------------------
    def load_from_csv(self, filename: str) -> None:
        """
        Expected CSV format:
            index1,index2,value1,value2,...
        """
        try:
            with open(filename, 'r') as file:
                reader = csv.reader(file)
                self.data = {
                    (int(row[0]), int(row[1])): np.array([float(x) for x in row[2:]], dtype=float)
                    for row in reader if row and len(row) >= 2
                }
        except Exception as e:
            raise ValueError(f"Error loading CSV file '{filename}': {e}")

    #-------------------------------------------------------------
    def save_to_csv(self, filename: str) -> None:
        try:
            with open(filename, 'w', newline='') as file:
                writer = csv.writer(file)
                for (index1, index2), value in self.data.items():
                    row = [index1, index2] + value.tolist()
                    writer.writerow(row)
        except Exception as e:
            raise ValueError(f"Error saving CSV file '{filename}': {e}")

"""
    ---------------------------------------------------------------------------
    NNet Class
    ---------------------------------------------------------------------------
"""
class NNet:
    #-------------------------------------------------------------
    def __init__(self, n_inputs: int = None,
                 n_hidden: list[int] = None,
                 n_outputs: int = None,
                 nNodes: list[int] = None, # [n_inputs]+n_hidden+[n_outputs]
                 connectivity=None,
                 activation_functions: list[str] = None,
                 training_data_file: str = 'trainingData.csv',
                 testing_data_file: str = 'testingData.csv',
                 learning_rate: float = 0.01,
                 optimizer: str = "gd", momentum: float = 0.9) -> None:
        # Determine network architecture based on provided parameters
        if nNodes is not None:
            self.layers = nNodes
            self.n_layers = len(nNodes)
            self.n_inputs = nNodes[0]
            self.n_hidden = nNodes[1:-2]
            self.n_outputs = nNodes[-1]
        else:
            self.layers = [n_inputs] + n_hidden + [n_outputs]
            self.n_layers  = len(self.layers)
            self.n_hidden  = n_hidden
            self.n_inputs  = n_inputs
            self.n_outputs = n_outputs

        # Establish activation functions: default is linear for input and output, tanh for hidden layers.
        if activation_functions is not None:
##            for layer, activ_fn in activation_functions:
##                if activ_fn in ACTIVATION_FUNCTION:
##                    self.activation_functions[layer] = ACTIVATION_FUNCTION[activ_fn][0]
##                    self.gradient_functions[layer]   = ACTIVATION_FUNCTION[activ_fn][1]
##                else:
##                    self.activation_functions = activation_functions
            self.activation_functions = activation_functions
        else:
            self.activation_functions = [
                'linear' if i == 0 or i == self.n_layers - 1 else 'tanh'
                for i in range(self.n_layers)
            ]

        # Create Weights
        self.weights = Weights()

##        print(f"... Creating NNet.\nnNodes = {self.layers}")
##        print(f"... activations = {self.activation_functions}")
##        print(f"... Weights = {self.weights.print_weights()}")
##        print(f"  self.n_layers  = {self.n_layers}")
##        print(f"  self.n_inputs  = {self.n_inputs}")
##        print(f"  self.n_hidden  = {self.n_hidden}")
##        print(f"  self.n_outputs = {self.n_outputs}")

        # Initialize connectivity matrix: default (if not provided) is only adjacent layers connected.
        self.connectivity = self.initialize_connectivity(connectivity)
        print(f"... Connectivity = \n{self.connectivity}")
        self.initialize_weights()
##        print(f"... Weights = {self.weights.print_weights()}")

        # Load training and testing data (assumes CSV with rows of numbers)
        self.training_data = self.load_data(training_data_file)
        self.testing_data = self.load_data(testing_data_file)
##        print(f"... training_data = {self.training_data}")

        # Initialize neuron states (for activation) for each layer.
        self.states      = [np.zeros(n) for n in self.layers]
        self.prev_states = [np.zeros(n) for n in self.layers]
        # Initialize carried gradients for feedback propagation (BPTT) – one per layer.
        self.carry_grads = {l: np.zeros(n) for l, n in enumerate(self.layers)}
        # Dictionary for weight velocities when using momentum.
        self.weight_velocities = {}

        # Training hyperparameters
        self.learning_rate = learning_rate
        self.optimizer = optimizer  # Options: 'gd' for gradient descent, 'momentum' for GD with momentum.
        self.momentum = momentum

    #-------------------------------------------------------------
    def initialize_connectivity(self, connectivity):
        """
        If no connectivity is provided, default to sequential (feedforward) – only adjacent layers are connected.
        """
        if connectivity is None:
            conn = np.zeros((self.n_layers, self.n_layers))
            for i in range(self.n_layers - 1):
                conn[i][i + 1] = 1  # Connect each layer to the next (feedforward)
            return conn
        return np.array(connectivity)

    #-------------------------------------------------------------
    def initialize_weights(self) -> None:
        """
        For every connection (i, j) in the connectivity matrix, initialize a weight matrix.
        For a connection from layer i (source) to layer j (destination), the weight matrix shape is:
          (neurons in destination, neurons in source)
        """
        for i in range(self.n_layers):
            for j in range(self.n_layers):
                if self.connectivity[i, j] == 1:
                    weight_matrix = np.random.rand(self.layers[j], self.layers[i])
                    self.weights.set_weights(i, j, weight_matrix)
                    print("Weights({i},{j}).shape = {self.weights(i,j).shape}")

    #-------------------------------------------------------------
    def load_data(self, filename: str) -> np.ndarray:
        """
        Load a CSV file and return the numerical data as a NumPy array.
        """
        try:
            with open(filename, 'r') as file:
                reader = csv.reader(file)
                data_list = []
                for row in reader:
                    if row:  # Skip empty rows
                        try:
                            data_list.append([float(val) for val in row])
                        except ValueError:
                            continue
                return np.array(data_list, dtype=float)
        except Exception as e:
            raise ValueError(f"Error loading data from '{filename}': {e}")

    #-------------------------------------------------------------
    def save_weights(self, filename: str) -> None:
        self.weights.save_to_csv(filename)

    #-------------------------------------------------------------
    def load_weights(self, filename: str) -> None:
        self.weights.load_from_csv(filename)

    #-------------------------------------------------------------
    def activate(self, func: str, x: np.ndarray) -> np.ndarray:
        """
        Apply the activation function.
          - 'linear' returns x.
          - 'tanh' returns np.tanh(x).
          Extend this mapping as needed.
        """
        if func == 'linear':
            return x
        elif func == 'tanh':
            return np.tanh(x)
        else:
            raise NotImplementedError(f"Activation function '{func}' is not implemented.")

    #-------------------------------------------------------------
    def activation_derivative(self, func: str, activated: np.ndarray) -> np.ndarray:
        """
        Given the activated value (e.g. the output of tanh(x)), return the derivative.
          - For linear, the derivative is 1.
          - For tanh, the derivative is 1 - activated**2.
        """
        if func == 'linear':
            return np.ones_like(activated)
        elif func == 'tanh':
            return 1 - np.power(activated, 2)
        else:
            raise NotImplementedError(f"Derivative for activation function '{func}' is not implemented.")

    #-------------------------------------------------------------
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward propagation.

        * Sets the input layer to x.
        * For each subsequent layer l, the net input is the sum over all incoming connections:
            - If the connection is feedforward (i < l), then use the current state of layer i.
            - If the connection is feedback (i >= l), then use the previous state of layer i.
        * After summing all contributions, the activation function for layer l is applied.
        * Finally, the output layer activation is returned.
        """
        # Backup current states into prev_states for usage in delayed (FB) connections.
        self.prev_states = [s.copy() for s in self.states]
        # Set the input layer activation.
        self.states[0] = x
        # Process layers 1..n_layers-1.
        for l in range(1, self.n_layers):
            net_input = np.zeros(self.layers[l])
            for i in range(self.n_layers):
                if self.connectivity[i, l] == 1:
                    # Choose which state to use based on connection type.
                    if l > i:  # Feedforward: use the current state.
                        source = self.states[i]
                    else:      # Feedback: use the previous state.
                        source = self.prev_states[i]
                    W = self.weights.data.get((i, l))
                    if W is not None:
                        # Weight matrix W has shape (neurons in layer l, neurons in layer i).
                        net_input += np.dot(W, source)
            # Apply the activation function for layer l.
            self.states[l] = self.activate(self.activation_functions[l], net_input)
        return self.states[-1]

    #-------------------------------------------------------------
    def compute_cost(self, yNet: np.ndarray, target: np.ndarray, cost_type: str = 'SE', aggregate: str = 'sum'):
        """
        Compute the per-neuron cost and overall cost (loss) while also generating
        the gradient of the cost with respect to the network output.

        Cost function options (per element):
          - 'SE' (Squared Error): cost = (target - yNet)**2, gradient = 2*(yNet - target)
          - 'abs': Absolute Error, cost = abs(target - yNet), gradient = sign(yNet - target)
          - 'logcosh': cost = log(cosh(target - yNet)), gradient = tanh(yNet - target)

        Aggregation options for total_cost:
          - 'sum': total_cost = sum(cost)
          - 'max': total_cost = max(cost)
          - 'norm': total_cost = norm(cost)
        """
        error = target - yNet
        if cost_type == 'SE':
            cost = np.power(error, 2)
            grad = 2 * (yNet - target)
        elif cost_type == 'abs':
            cost = np.abs(error)
            grad = np.sign(yNet - target)
        elif cost_type == 'logcosh':
            cost = np.log(np.cosh(error))
            grad = np.tanh(yNet - target)
        else:
            raise NotImplementedError(f"Cost type '{cost_type}' is not implemented.")

        if aggregate == 'sum':
            total_cost = np.sum(cost)
        elif aggregate == 'max':
            total_cost = np.max(cost)
        elif aggregate == 'norm':
            total_cost = np.linalg.norm(cost)
        else:
            raise NotImplementedError(f"Aggregation method '{aggregate}' is not implemented.")

        return total_cost, cost, grad

    #-------------------------------------------------------------
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backpropagation (including delayed feedback using carried gradients).

        Steps:
          1. Compute the delta (local gradient) for the output layer:
                delta = (dCost/d_output) * (activation derivative).
          2. For each hidden (and input) layer, compute:
                delta[l] = derivative(activation at l) * sum_{m in children} (W[l, m]^T * delta[m]),
            where for feedforward connections (m > l) we use the current delta,
            and for feedback connections (m <= l) we incorporate the carried gradient from the previous epoch.
          3. Save these deltas as the “carried” gradients for next epoch.
          4. Compute the gradient for each weight with:
                dW[i,j] = outer(delta[j], source_vector),
            where the source_vector comes from either the current state (if feedforward) or previous state (if feedback).
          5. Also compute the gradient of the cost with respect to the input x.
        """
        deltas = [None] * self.n_layers
        # Output layer (layer L) delta:
        L = self.n_layers - 1
        deriv = self.activation_derivative(self.activation_functions[L], self.states[L])
        deltas[L] = grad_output * deriv

        # For layers L-1 down to 0:
        for l in range(self.n_layers - 2, -1, -1):
            sum_term = np.zeros(self.layers[l])
            for m in range(self.n_layers):
                if self.connectivity[l, m] == 1:
                    W = self.weights.data.get((l, m))
                    if W is None:
                        continue
                    if m > l:  # For feedforward, use the delta computed in this epoch.
                        sum_term += np.dot(W.T, deltas[m])
                    else:      # For feedback, use the carried gradient from the previous epoch.
                        carry = self.carry_grads[m] if m in self.carry_grads else 0
                        sum_term += np.dot(W.T, carry)
            deriv = self.activation_derivative(self.activation_functions[l], self.states[l])
            deltas[l] = deriv * sum_term

        # Save the current deltas as carried gradients for the next epoch.
        self.carry_grads = {l: deltas[l] for l in range(self.n_layers)}

        # Compute gradients for weights.
        self.weight_grads = {}
        for i in range(self.n_layers):
            for j in range(self.n_layers):
                if self.connectivity[i, j] == 1:
                    # For feedforward links, use the current state; for FB, use the previous state.
                    if j > i:
                        source = self.states[i]
                    else:
                        source = self.prev_states[i]
                    grad_w = np.outer(deltas[j], source)
                    self.weight_grads[(i, j)] = grad_w

        # Compute gradient with respect to input x.
        grad_input = np.zeros(self.layers[0])
        for m in range(1, self.n_layers):
            if self.connectivity[0, m] == 1:
                W = self.weights.data.get((0, m))
                if W is not None:
                    grad_input += np.dot(W.T, deltas[m])
        return grad_input

    #-------------------------------------------------------------
    def update_weights(self) -> None:
        """
        Update each weight based on its computed gradient.
          - For 'gd' (gradient descent): W = W - lr * grad.
          - For 'momentum': a velocity is maintained and updated:
                v = momentum * v - lr * grad, then W = W + v.
          Extend this method as needed for other update rules.
        """
        for key, grad in self.weight_grads.items():
            if self.optimizer == "gd":
                self.weights.data[key] -= self.learning_rate * grad
            elif self.optimizer == "momentum":
                if key not in self.weight_velocities:
                    self.weight_velocities[key] = np.zeros_like(grad)
                self.weight_velocities[key] = (self.momentum * self.weight_velocities[key] -
                                               self.learning_rate * grad)
                self.weights.data[key] += self.weight_velocities[key]
            else:
                # Default: simple gradient descent.
                self.weights.data[key] -= self.learning_rate * grad

    #-------------------------------------------------------------
    def train(self, n_epochs: int = 1000, display_epoch: int = 100,
              cost_type: str = 'SE', aggregate: str = 'sum') -> None:
        """
        Training loop:
          1. For each sample, split the record into the input part and target output part.
          2. Run the forward pass.
          3. Compute the cost and the output gradient.
          4. Run backward propagation (this computes weight gradients and gradient with
             respect to the input—but that latter value can be ignored or used for further analysis).
          5. Update weights accordingly.
          6. Optionally, display the average cost per epoch.

        It is assumed that each row in self.training_data is of the form:
             [x (inputs)..., target (outputs)...]
        """
        n_samples = self.training_data.shape[0]
        for epoch in range(n_epochs):
            total_epoch_cost = 0.0
            for sample in self.training_data:
                # Split sample into input and target parts.
                x = sample[:self.n_inputs]
                target = sample[self.n_inputs:self.n_inputs + self.n_outputs]

                # Forward pass.
                yNet = self.forward(x)

                # Compute cost and gradient at output.
                total_cost, _, grad_output = self.compute_cost(yNet, target, cost_type, aggregate)
                total_epoch_cost += total_cost

                # Backward pass.
                _ = self.backward(grad_output)

                # Update weights.
                self.update_weights()

            if (epoch + 1) % display_epoch == 0:
                avg_cost = total_epoch_cost / n_samples
                print(f"Epoch {epoch+1}/{n_epochs}: Average Cost = {avg_cost}")

    #-------------------------------------------------------------
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """Perform a forward pass to generate a prediction."""
        return self.forward(input_data)

    #-------------------------------------------------------------
    def __repr__(self) -> str:
        return f"NNet(n_layers={self.n_layers}, layers={self.layers}, " \
               f"activation_functions={self.activation_functions})"
