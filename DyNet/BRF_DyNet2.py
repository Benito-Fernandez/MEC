# File: 'BRF_DyNet2.py'
# (c) Benito R. Fernandez
# Recurrent Neural Network to simulate Dynamic Systems (DyNet)
import numpy as np
import pandas as pd
import json

# Utilty Methods
def print_dict(dictionary):
    width = max(len(key) for key in dictionary.keys() if isinstance(key, str))
    for key, value in dictionary.items():
        print(f"{key}".ljust(width),f": {value.shape}")

# DyNet class
# -----------
# To realize the Dynamic Network
#                proposed by Dr. Benito Fernandez, PhD
#                while he was a Professor at the NERDLab:
#                Neuro-Engineering Research & Development Laboratory
#                The University of Texas at Austin
#
# The network attempts to capture the nonlinear dynamics of the form:
#
#       dx/dt  = f(x(t), u(t), w(t))
#       y(t)   = h(x(t), u(t), v(t))
#
# When discretizing the continuous system at time steps T,
# there are two generic models: The shift and delta models:
#
#       [shift]
#               x(k+1) = f_s(x(k), u(k), w(k))
#               y(k)   = h(x(k), u(k), v(k))
#
#       [delta]
#               x(k+1) = x(k) + T*f_d(x(k), u(k), w(k))
#               y(k)   = h(x(k), u(k), v(k))
#
# where,
#       x(k)    is the state vector of dimension R^n_states at time k
#       u(k)    is the input vector of dimension R^n_inputs
#       w(k)    is the process noise vector of dimension R^n_process_noise
#       y(k)    is the output vector of dimension R^n_outputs
#       v(k)    is the sensor noise vector of dimension R^n_sensors_noise
#       f(.)    is a function that maps x(k), u(k), w(k) to x(k+1)
#       f(.)    is a function that maps x(k), u(k), v(k) to y(k)
#       T       is the time step
#       k       represents the index of time, t = k*T
#
#   The preferred model is the delta.
#   When T changes, f() needs to be re-calculated,
#   but F() in the limit, when T-->0 recovers the continuous f()
#
#       f_d() = (x(k+1)-x(k))/T
#
#       lim_{T-->0} f_d() = lim_{T-->0} (x(k+1)-x(k))/T = dx/dt = f()
#
# The network consists of 5 layers:
#   1st layer:  This is the input layer, that has
#                   - n_inputs nodes, that hold the network's inputs
#                   - n_states nodes, that hold a delayed memory
#                                     of the system's state vector
#                   - n_process_noise nodes, that hold Gaussian noise
#                                     that is fed as process noise
#                                     to the plant dynamics
#               The layer doesn't have biases and
#               the activation function is linear.
#               It just holds the current values of u(k) and
#               the previous value of x(k-1), and
#               the process noise, w(k), that is generated from
#               a Gaussian, normal distribution with zero meand and
#               standard deviation, process_stddev
#
#   2nd layer:  This is the hidden layer of the first 3LP (three-layer
#               perceptron that maps f_d() or f_s().  It has
#                   - n_hidden_states nodes
#               The layer has biases and
#               the default activation function is tanh().
#
#   3rd layer:  This is the states layer. It has
#                   - n_states nodes, that hold the current state derivative,
#                           f(x(k), u(k), w(k))
#                     It could be f_s() or f_d() (default)
#                     These nodes are connected from the other n_states that
#                     with a delay with identity matrix (each on a 1:1)
#                   - n_states nodes, that hold the current state, x(k+1)
#                   - n_sensors_noise nodes, that hold Gaussian noise
#                                     that is fed as sensor noise
#                                     to the output mapping (sensors)
#                   - n_inputs nodes, that hold the network's inputs,
#                     These are connected from the input nodes with identity
#                     without delays.  These may not be necessary if we allow
#                     direct connection from the u(k) nodes of the 1st layer.
#               The layer n_states that capture f() have biases and
#               the default activation function is tanh().
#               The other nodes (n_states that hold x(k) and n_sensors_noise
#               that generate the sensor noise) have no biases and
#               their activation is linear
#               One last thing, the weights that feed the n_states nodes
#               that compute f(), may have an embedded T in them, when computing
#               f_d().  This is only relevant if we are interested in
#               finding the network estimate of f().
#
#   4th layer:  This is the hidden layer of the second 3LP (three-layer
#               perceptron that maps h().  It has
#                   - n_hidden_sensors nodes
#               The layer has biases and
#               the default activation function is tanh().
#
#   5th layer:  This is the output layer of the first 3LP (three-layer
#               perceptron that maps h().  It has
#                   - n_outputs nodes
#               The layer has biases and
#               the default activation function is linear.
#
#
#
#
class DyNet:
    def __init__(self, n_inputs=1, n_outputs=1, n_states=2,
                 n_process_noise=None, n_sensors_noise=None,
                 # hidden layers nodes
                 n_hidden_states=5, n_hidden_sensors=3,
                 # process and sensor noises
                 process_stddev=0.02, sensors_stddev=0.01,
                 # termination conditions
                 loss_termination=1, gradient_termination=0.01, maxEpochs=1000,
                 animate_learning=True,
                 # training parameters
                 updateRule = 'gradientDescend', learningRate = 0.01,
                 # files with data and network
                 training_file='training_data.csv',
                 testing_file='testing_data.csv',
                 model_file='dynet_model.net',
                 # dynamics parameters
                 mode = 'delta',
                 T = 0.1):

        self.n_inputs  = n_inputs
        self.n_outputs = n_outputs
        self.n_states  = n_states
        self.n_process_noise  = n_process_noise if n_process_noise is not None else n_states   # Default to n_states
        self.n_sensors_noise  = n_sensors_noise if n_sensors_noise is not None else n_outputs  # Default to n_outputs
        self.n_hidden_states  = n_hidden_states
        self.n_hidden_sensors = n_hidden_sensors
        self.process_stddev   = process_stddev
        self.sensors_stddev   = sensors_stddev
        self.loss_termination = loss_termination
        self.gradient_termination = gradient_termination
        self.maxEpochs = maxEpochs
        self.animate_learning = animate_learning
        self.training_file = training_file
        self.testing_file = testing_file
        self.model_file = model_file
        self.updateRule = updateRule
        self.lr =learningRate
        self.mode = mode
        self.T = T

        # Initialize network parameters
        self.initialize_weights()

    def initialize_weights(self):
        """Initialize network weights and biases."""
        # Weights
        self.weights = {
            "inputs_to_hidden_states": np.random.randn(self.n_inputs, self.n_hidden_states),
            "states_to_hidden_states": np.random.randn(self.n_states, self.n_hidden_states),
            "noise_to_hidden_states": np.random.randn(self.n_process_noise, self.n_hidden_states),
            "hidden_states_to_states": np.random.randn(self.n_hidden_states, self.n_states),
            "states_to_hidden_sensors": np.random.randn(self.n_states, self.n_hidden_sensors),
            "noise_to_hidden_sensors": np.random.randn(self.n_sensors_noise, self.n_hidden_sensors),
            "hidden_sensors_to_outputs": np.random.randn(self.n_hidden_sensors, self.n_outputs),
            "inputs_to_states": np.eye(self.n_inputs, self.n_inputs)  # Fixed, 1:1 connections
##            "inputs_to_hidden_states": np.random.randn(self.n_inputs + self.n_states + self.n_process_noise, self.n_hidden_states),
##            "hidden_states_to_states": np.random.randn(self.n_hidden_states, self.n_states),
##            "states_to_hidden_sensors": np.random.randn(self.n_states + self.n_sensors_noise, self.n_hidden_sensors),
##            "hidden_sensors_to_outputs": np.random.randn(self.n_hidden_sensors, self.n_outputs),
##            "inputs_to_states": np.eye(self.n_inputs, self.n_inputs)  # Fixed, 1:1 connections
        }

        # Biases
        self.biases = {
            "hidden_states": np.random.randn(self.n_hidden_states),
            "states": np.random.randn(self.n_states),
            "hidden_sensors": np.random.randn(self.n_hidden_sensors),
            "outputs": np.random.randn(self.n_outputs),
        }

    def linear (x): return (x)
    def linear_(x): return (np.ones_like(x))

    def Tanh (x): return np.tanh(x)                   # activation function: Tanh
    def Tanh_(x): return (1+np.tanh(x))*(1-np.tanh(x))# derivative of        Tanh

    def RBF (x): return np.exp(-np.power(np.array(x),2))
    def RBF_(x): return -2*(x*np.exp(-np.power(np.array(x),2)))

    def loss(self, e, gain=1, gamma=2):
        """Skewed LogCosh Loss function."""
        return np.power(gain, np.tanh(gamma * e)) * np.log(np.cosh(e))

    def loss_derivative(self, e, gain=1, gamma=2):
        """Skewed LogCosh Loss function gradient."""
        return np.power(gain, np.tanh(gamma * e)) * np.tanh(e)

    def save_network(self):
        """Save network topology and weights."""
        with open(self.model_file, 'w') as f:
            json.dump({"weights": self.weights, "biases": self.biases}, f)

    def load_network(self):
        """Load network topology and weights."""
        with open(self.model_file, 'r') as f:
            model_data = json.load(f)
            self.weights = model_data["weights"]
            self.biases = model_data["biases"]

    def forward(self, input_k, state_k_minus_1, process_noise_k, sensor_noise_k):
        """Forward pass: propagate inputs through the network."""
        """We could/should make the noise generated within and
        the states_k_minus_1 be updated from the network memory."""
        # Layer 1-2: inputs, delayed states, process noise -> hidden_states
        hidden_states  = np.dot(input_k, self.weights["inputs_to_hidden_states"])
        hidden_states += np.dot(state_k_minus_1, self.weights["states_to_hidden_states"])
        hidden_states += np.dot(process_noise_k, self.weights["noise_to_hidden_states"])
        hidden_states += self.biases["hidden_states"]
        hidden_states = np.tanh(hidden_states)

        # Layer 2-3: Compute states derivatives & next_states
        state_derivatives = np.dot(hidden_states, \
        self.weights["hidden_states_to_states"] + self.biases["states"])
        next_states = state_k_minus_1 + self.T*state_derivatives

        # Layer 3-4: Generate sensor noise and stack with states
        hidden_sensors = np.dot(next_states, \
        self.weights["states_to_hidden_sensors"])
        hidden_sensors += np.dot(sensor_noise_k, \
        self.weights["noise_to_hidden_sensors"])
        hidden_sensors += self.biases["hidden_sensors"]
        hidden_sensors = np.tanh(hidden_sensors)

        # Layer 4-5: Compute outputs
        output_y = np.dot(hidden_sensors, \
        self.weights["hidden_sensors_to_outputs"]) + self.biases["outputs"]

        return next_states, output_y, hidden_states, hidden_sensors

    def compute_loss(self, y_true, y_pred):
        """Compute the loss using the Skewed LogCosh Loss function."""
        error = y_true - y_pred
        return self.loss(error)

    def backward(self, error, hidden_sensors, hidden_states):
        """Backward pass: compute gradients."""
        # Loss gradient to outputs
        grad = self.loss_derivative(error)  # Gradient of the loss function

        gradients = {
            "outputs": grad,
            "hidden_sensors_to_outputs": np.outer(hidden_sensors, grad),
            "hidden_states_to_states"  : np.outer(hidden_states,  grad),
        }
        return gradients

    ###---------------------------------------------------------------------###

    def BackPropagation(self, Error):
        # - Backward Propagation
        self.Grad[self.nLayers-1] = -self.lossGrad_function(Error)*self.gradient_function[self.nLayers-1](self.z[self.nLayers-1])
        self.Biases_grad[self.nLayers-1]  = self.Grad[self.nLayers-1].sum(axis=0)/Ntrain
        self.Weights_grad[self.nLayers-1] = np.dot(self.z[self.nLayers-2].T,self.Grad[self.nLayers-1])/Ntrain
        for layer in reversed(range(self.nLayers-1)):
            self.Grad[layer] = np.dot(self.Grad[layer+1],self.Weights[layer+1].T)*self.gradient_function[layer](self.z[layer])
            self.Biases_grad[layer]  = self.Grad[layer].sum(axis=0).reshape(self.Biases[layer].shape)/Ntrain
            if layer == 0:
                self.Weights_grad[layer] = np.dot(self.Udata.T,self.Grad[layer])/Ntrain
                #- Note: grad[-1] is the network gradient to the input!
                # gradNet = np.dot(grad[layer],Weights[layer].T)*gradient_function[layer](xi[-1])
                self.gradNet = np.dot(self.Grad[layer],self.Weights[layer].T)/Ntrain # since at this point layer=0
            else:
                self.Weights_grad[layer] = np.dot(self.z[layer-1].T,self.Grad[layer])/Ntrain
#        return (self.Weights_grad, self.Biases_grad, self.gradNet)

    ###---------------------------------------------------------------------###

    def UpdateWeights(self, updateRule = 'gradientDescend'):
        if updateRule == 'gradientDescend':
            for layer in range(self.nLayers):
                self.Weights[layer] -= self.learningRate*self.Weights_grad[layer]
                temp = self.Biases_grad[layer]
#                print(temp, temp.shape)
                self.Biases_grad[layer] = np.ndarray(shape=self.Biases[layer].shape)
#                print(self.Biases_grad[layer],self.Biases_grad[layer].shape)
                self.Biases_grad[layer].copy(temp.all())
#                print(self.Biases_grad[layer],self.Biases_grad[layer].shape)
#                print("self.Weights[layer]---------------------------> ",self.Weights[layer].shape,'\n',self.Weights[layer],type(self.Weights[layer]))
#                print("(self.Biases[layer].shape)---------------------------> ",self.Biases[layer].shape,'\n',self.Biases[layer],type(self.Biases[layer].shape))
#                print("(self.Biases_grad[layer].shape)----------------------> ",self.Biases_grad[layer].shape,'\n',self.Biases_grad[layer],type(self.Biases_grad[layer].shape))
                self.Biases[layer]  -= self.learningRate*self.Biases_grad[layer].reshape(self.Biases[layer].shape)
        return (self.Weights, self.Biases)

    ###---------------------------------------------------------------------###

    def checkTermination(self):
        if self.epochs > self.maxEpochs:
            self.TerminateRason = 'Maximum Epochs Reached'
            return True
        if self.Loss['totalSum'] < self.accuracyTest:
            self.TerminateRason = 'Error below accuaracyTest'
            return True
        return False

    ###---------------------------------------------------------------------###
    def update_weights(self, gradients, learning_rate=0.01):
        """Update weights using computed gradients."""
        for key in gradients:
            self.weights[key] += learning_rate * gradients[key]

    def train(self, accuracyTest=0.01):
        """Train the DyNet model using the training dataset."""
        # Load training data
        data = pd.read_csv(self.training_file)
        inputs = data.iloc[:, 1:self.n_inputs + 1].values
        outputs = data.iloc[:, self.n_inputs + 1:self.n_inputs + self.n_outputs + 1].values

        # Initialize states and loss tracking
        states = np.zeros((len(inputs), self.n_states))
        loss_history = []

        for epoch in range(self.maxEpochs):
            total_loss = 0

            for k in range(len(inputs)):
                # Generate process and sensor noise
                w_k = np.random.normal(0, self.process_stddev, self.n_states)
                v_k = np.random.normal(0, self.sensors_stddev, self.n_outputs)

                # Get previous state
                state_k_minus_1 = states[k - 1] if k > 0 else np.zeros(self.n_states)

                # Forward pass
                next_states, y_pred, hidden_states, hidden_sensors = self.forward(
                    inputs[k], state_k_minus_1, w_k, v_k
                )
                states[k] = next_states  # Update states

                # Compute loss
                loss_k = self.compute_loss(outputs[k], y_pred)
                total_loss += np.sum(loss_k)

                # Backward pass
                error = outputs[k] - y_pred
                gradients = self.backward(error, hidden_sensors, layer3_input, hidden_states)

                # Update weights
                self.update_weights(gradients)

            loss_history.append(total_loss)

            # Check for stopping criteria
            if total_loss < self.loss_termination:
                print(f"Training stopped early at epoch {epoch + 1} due to loss threshold.")
                break

        # Save loss history
        pd.DataFrame({"epoch": range(1, len(loss_history) + 1), "loss": loss_history}).to_csv("learning_loss.csv", index=False)

        # Optionally plot loss history
        if self.animate_learning:
            self.plot_learning_loss(loss_history)

        print(f"Training complete. Final loss: {total_loss}")

    def plot_learning_loss(self, loss_history):
        """Plot and optionally save the learning loss."""
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 5))
        plt.plot(loss_history, label="Training Loss", color="blue")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Learning Loss Over Epochs")
        plt.legend()
        plt.grid(True)
        plt.savefig("learning_loss.jpg")
        plt.show()

    def simulate(self, input_stream, initial_state=None):
        """
        Simulate the system using a stream of input data.

        Args:
            input_stream (pd.DataFrame): Input data as a Pandas DataFrame.
            initial_state (np.ndarray): Initial state vector, optional.

        Returns:
            pd.DataFrame: Simulation results including inputs, states, and outputs.
        """
        inputs = input_stream.values
        num_samples = len(inputs)
        states = np.zeros((num_samples, self.n_states))
        outputs = np.zeros((num_samples, self.n_outputs))

        # Use provided initial state or initialize to zero
        if initial_state is not None:
            states[0] = initial_state
        else:
            states[0] = np.zeros(self.n_states)

        for k in range(num_samples):
            # Generate process and sensor noise
            w_k = np.random.normal(0, self.process_stddev, self.n_states)
            v_k = np.random.normal(0, self.sensors_stddev, self.n_outputs)

            # Get previous state
            state_k_minus_1 = states[k - 1] if k > 0 else states[0]

            # Forward pass
            next_states, y_pred, _, _, _ = self.forward(inputs[k], state_k_minus_1, w_k, v_k)
            states[k] = next_states
            outputs[k] = y_pred

        # Return simulation results as a DataFrame
        results = pd.DataFrame({
            "time": input_stream.index,
            "inputs": [inputs[i].tolist() for i in range(num_samples)],
            "states": [states[i].tolist() for i in range(num_samples)],
            "outputs": [outputs[i].tolist() for i in range(num_samples)],
        })

        return results

    def predict(self, current_input, current_state, n_horizon=5):
        """
        Predict future states and outputs for a fixed horizon.

        Args:
            current_input (np.ndarray): Current input vector.
            current_state (np.ndarray): Current state vector.
            n_horizon (int): Prediction horizon (default=5).

        Returns:
            np.ndarray: Predicted outputs over the horizon.
        """
        predictions = []
        input_k = current_input
        state_k = current_state

        for _ in range(n_horizon):
            # Generate process and sensor noise
            w_k = np.random.normal(0, self.process_stddev, self.n_states)
            v_k = np.random.normal(0, self.sensors_stddev, self.n_outputs)

            # Forward pass
            next_states, y_pred, _, _, _ = self.forward(input_k, state_k, w_k, v_k)
            predictions.append(y_pred)

            # Update state and input for the next time step
            state_k = next_states
            input_k = np.zeros_like(input_k)  # Assume inputs beyond current step are zero

        return np.array(predictions)

    def print_network(self):
        """Display the network's topology, activation functions, weights, and biases."""
        print("=== DyNet Topology ===")
        print(f"Input layer: {self.n_inputs} nodes")
        print(f"State layer (feedback): {self.n_states} nodes")
        print(f"Process noise: {self.n_process_noise} nodes")
        print(f"Sensors noise: {self.n_sensors_noise} nodes")
        print(f"Hidden states  layer: {self.n_hidden_states} nodes | Activation: 'tanh'")
        print(f"Hidden sensors layer: {self.n_hidden_sensors} nodes | Activation: 'tanh'")
        print(f"Output layer:         {self.n_outputs} nodes | Activation: 'linear'")
        print("\n=== Weights ===")
        width = max(len(key) for key in self.weights.keys() if isinstance(key, str))
        for key, value in self.weights.items():
            print(f"{key}".ljust(width),f": {value.shape}")
        print("\n=== Biases ===")
        width = max(len(key) for key in self.biases.keys() if isinstance(key, str))
        print(width)
        for key, value in self.biases.items():
            print(f"{key}".ljust(width),f": {value.shape}")
        print("======================")

    def __str__(self):
        """String representation of the DyNet class."""
        return (
            f"DyNet Model\n"
            f"Inputs: {self.n_inputs} | Outputs: {self.n_outputs} | States: {self.n_states}\n"
            f"Hidden States: {self.n_hidden_states} | Hidden Sensors: {self.n_hidden_sensors}\n"
            f"Process Noise STDDEV: {self.process_stddev} | Sensor Noise STDDEV: {self.sensors_stddev}\n"
            f"Training File: {self.training_file} | Testing File: {self.testing_file}\n"
            f"Model File: {self.model_file}"
        )

    # Getters for __init__ parameters
    def get_n_inputs(self):
        return self.n_inputs

    def get_n_outputs(self):
        return self.n_outputs

    def get_n_states(self):
        return self.n_states

    def get_n_hidden_states(self):
        return self.n_hidden_states

    def get_process_stddev(self):
        return self.process_stddev

    def get_sensors_stddev(self):
        return self.sensors_stddev

    # Setters for __init__ parameters
    def set_n_inputs(self, value):
        self.n_inputs = value

    def set_n_outputs(self, value):
        self.n_outputs = value

    def set_n_states(self, value):
        self.n_states = value

    def set_n_hidden_states(self, value):
        self.n_hidden_states = value

    def set_process_stddev(self, value):
        self.process_stddev = value

    def set_sensors_stddev(self, value):
        self.sensors_stddev = value
"""-------------------------------------------------------------------------"""
"""-------------------------------------------------------------------------"""
"""-------------------------------------------------------------------------"""


if __name__ == '__main__':
    print("Creating Network")
    print(">>> mlp = MLP(3,[4],2,['linear','tanh','linear'],'logcosh')")
    dynet = DyNet()
    dynet.print_network()

    # Test forward method
    #------------------------------------------------------------
    print("\n Testing forward method...")
    input_k, state_k_minus_1, process_noise_k, sensor_noise_k = \
       0.25,      [0.1, 0.2],    [0.01, .012],          0.005

    print(f"input_k         = {input_k}")
    print(f"state_k_minus_1 = {state_k_minus_1}")
    print(f"process_noise_k = {process_noise_k}")
    print(f"sensor_noise_k  = {sensor_noise_k}")
    print(30*'-')#-----------------------------------------------

    next_states, output_y, hidden_states, hidden_sensors = \
    dynet.forward(input_k, state_k_minus_1, process_noise_k, sensor_noise_k)

    print(f"next_states    = {next_states}")
    print(f"output_y       = {output_y}")
    print(f"hidden_states  = {hidden_states}")
    print(f"hidden_sensors = {hidden_sensors}")

    # Test loss method
    #------------------------------------------------------------
    print("\n Testing loss method...")
    y_true, y_pred = 0.3, output_y[0]
    print(f"y_true = {y_true}")
    print(f"y_pred = {y_pred}")
    error = y_true - y_pred
    print(f"error = {error}")
    print(30*'-')#-----------------------------------------------

    loss = dynet.compute_loss(y_true, y_pred)

    print(f"loss = {loss}")

    gradients = dynet.backward(error, hidden_sensors, \
                hidden_states)

    print(f"gradients =")
    print(f"gradients['hidden_sensors_to_outputs'] = {gradients['hidden_sensors_to_outputs']}")
   # print(f"gradients['states_to_hidden_sensors']  = {gradients['states_to_hidden_sensors']}")
    print(f"gradients['hidden_states_to_states']   = {gradients['hidden_states_to_states']}")


