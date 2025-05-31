import numpy as np
import matplotlib.pyplot as plt
import json

class NARMAXNetwork:
    #---------------------------------
    def __init__(self, n_inputs, n_outputs, n_hidden=None,
                 n_inputs_delays=2, n_outputs_delays=3,
                 n_horizon=5, n_history=5,
                 outputs_ic=None, activation=np.tanh):
        self.n_inputs         = n_inputs
        self.n_outputs        = n_outputs
        self.n_inputs_delays  = n_inputs_delays
        self.n_outputs_delays = n_outputs_delays
        if n_hidden == None:
            self.n_hidden     = 2 * n_inputs * n_outputs\
                              + n_inputs_delays * n_outputs\
                              + n_inputs * n_outputs_delays
        else:
            self.n_hidden     = n_hidden
        self.n_history        = n_history
        self.n_horizon        = n_horizon
        self.activation       = activation
        self.weights_hidden   = np.random.randn(self.n_hidden, \
                              self.n_inputs + self.n_history * self.n_outputs)
        self.bias_hidden      = np.random.randn(self.n_hidden)
        self.weights_output   = np.random.randn(self.n_outputs, self.n_hidden)
        self.bias_output      = np.random.randn(self.n_outputs)
        self.inputs           = np.zeros(self.n_inputs).reshape(self.n_inputs,1)
        if outputs_ic==None:
            self.outputs      = np.zeros(self.n_outputs).reshape(self.n_outputs,1)
        else:
            self.outputs      = outputs_ic.reshape(self.n_outputs).reshape(self.n_outputs,1)

        #print(f"self.inputs.shape = {self.inputs.shape}\n{self.inputs}")
        #print(f"self.outputs.shape = {self.outputs.shape}\n{self.outputs}")
        self.states = np.concatenate((self.inputs.reshape(len(self.inputs),1),
                                      self.outputs.reshape(len(self.outputs),1)),
                                      axis=0)
        #print(f"self.states.shape = {self.states.shape}\n{self.states}")

    #---------------------------------
    def loss(self, e, gain=1, gamma=2):  # Skewed LogCosh Loss function
        return np.power(gain, np.tanh(gamma * e)) * np.log(np.cosh(e))

    #---------------------------------
    def loss_gradient(self, e, gain=1, gamma=2):  # Gradient approximation
        return np.power(gain, np.tanh(gamma * e)) * np.tanh(e)

    #---------------------------------
    def train(self, inputs, targets, n_epochs=1000, n_show=100):
        losses = []
        for epoch in range(n_epochs):
            predictions = self.predict(inputs)
            errors = targets - predictions
            epoch_loss = np.sum(self.loss(errors))
            losses.append(epoch_loss)

            # Backpropagation updates
            gradient = self.loss_gradient(errors)
            # Update weights and biases logic goes here...

            if epoch % n_show == 0:
                self.plot_training(inputs, targets, predictions)

        plt.plot(range(n_epochs), losses)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training Loss Over Epochs")
        plt.savefig("training_loss.pdf")

    #---------------------------------
    def predict(self, inputs):
        print("\n@NARMAXNNet.predict...")

        self.inputs = inputs[:,1].reshape(len(inputs),1)
        #self.inputs = np.stack(self.inputs,collect_sample())
        self.bias_hidden = self.bias_hidden.reshape(len(self.bias_hidden),1)
        self.outputs = self.outputs.reshape(len(self.outputs),1)

        print(f"self.inputs = {self.inputs.shape} \n{self.inputs}")
        print(f"self.weights_hidden = {self.weights_hidden.shape} \n{self.weights_hidden}")
        print(f"self.bias_hidden = {self.bias_hidden.shape} \n{self.bias_hidden}")
        print(f"self.outputs = {self.outputs.shape} \n{self.outputs}")

        self.states = np.concatenate(([self.inputs],
                      self.n_history * [self.outputs.reshape(len(self.outputs),1)]),
                      axis=0)
        print(f"self.states.shape = {self.states.shape}")
        print(f"self.states=\n{self.states}")
        hidden = self.activation(np.outer(self.states, self.weights_hidden)
               + self.bias_hidden.reshape(len(self.bias_hidden),1))
        self.outputs = np.dot(hidden, self.weights_output) + self.bias_output
        return self.outputs

    #---------------------------------
    def save_model(self, file_path):
        # Save topology and weights/biases to a .net file
        data = {
            "n_inputs": self.n_inputs,
            "n_outputs": self.n_outputs,
            "n_inputs_delays": self.n_inputs_delays,
            "n_outputs_delays": self.n_outputs_delays,
            "n_hidden": self.n_hidden,
            "n_horizon": self.n_horizon,
            "weights_hidden": self.weights_hidden.tolist(),
            "bias_hidden": self.bias_hidden.tolist(),
            "weights_output": self.weights_output.tolist(),
            "bias_output": self.bias_output.tolist()
        }
        with open(file_path, "w") as file:
            json.dump(data, file, indent=4)

    #---------------------------------
    def load_model(self, file_path):
        # Load the network topology and weights from .net file
        with open(file_path, "r") as file:
            data = eval(file.read())  # Be cautious with eval; ensure trusted inputs
        self.weights_hidden = np.array(data["weights_hidden"])
        self.bias_hidden = np.array(data["bias_hidden"])
        self.weights_output = np.array(data["weights_output"])
        self.bias_output = np.array(data["bias_output"])

    #---------------------------------
    def __str__(self):
        # Pretty-print network topology and weights/biases
        network_info = [
            45*"=",
            f"Network Topology:",
            f"  Inputs:  {self.n_inputs}",
            f"  Outputs: {self.n_outputs}",
            f"  Inputs  delays: {self.n_inputs_delays}",
            f"  Outputs delays: {self.n_outputs_delays}",
            f"  Hidden  Nodes: {self.n_hidden}",
            f"  Horizon Steps:  {self.n_horizon}",
            f"  History Steps:  {self.n_history}",
            "",
            f"Weights Hidden Layer: {self.weights_hidden.shape}",
            f"{np.array_str(self.weights_hidden, precision=2, suppress_small=True)}",
            "",
            f"Bias Hidden Layer: {self.bias_hidden.shape}",
            f"{np.array_str(self.bias_hidden, precision=2, suppress_small=True)}",
            "",
            f"Weights Output Layer: {self.weights_output.shape}",
            f"{np.array_str(self.weights_output, precision=2, suppress_small=True)}",
            "",
            f"Bias Output Layer: {self.bias_output.shape}",
            f"{np.array_str(self.bias_output, precision=2, suppress_small=True)}",
            45*"="
        ]
        return "\n".join(network_info)

    #---------------------------------
    def plot_training(self, inputs, targets, predictions):
        plt.figure()
        plt.scatter(inputs[:, 0], targets[:, 0], color="blue", label="Training Data")
        plt.plot(inputs[:, 0], predictions[:, 0], color="red", label="Network Output")
        plt.legend()
        plt.title("Training Progress")
        plt.savefig("training_progress.pdf")

#-------------------------------------------------------------------------------

def arrange_samples(Y, k, n_history, all=False):
    """
    Arrange samples by stacking columns from matrix Y.

    Parameters:
        Y (numpy.ndarray): Input array with shape (n_outputs, n_samples).
        k (int): Starting column index (ignored if all=True).
        n_history (int): Number of past columns to concatenate.
        all (bool): If False, stack only for column k.
                    If True, stack for k ranging from n_history to n_samples.

    Returns:
        numpy.ndarray: Stacked samples with adjusted dimensions.
    """
    n_outputs, n_samples = Y.shape

    if all:
        # Generate stacked samples for k from n_history to n_samples-1
        stacked_list = [np.vstack([Y[:, i - j] for j in range(n_history)]).reshape(n_outputs * n_history, 1)
                        for i in range(n_history, n_samples)]
        return np.hstack(stacked_list)  # Correctly stacks along the second axis
    else:
        # Stack columns only for given k
        stacked = np.vstack([Y[:, k - i] for i in range(n_history)])
        return stacked.reshape(n_outputs * n_history, 1)  # Ensure correct shape

#-------------------------------------------------------------------------------

def update_buffer(buffer, next_sample, n_history, n_outputs):
    """
    Updates buffer by shifting previous values downward,
    inserting the new column next_sample at the top.

    Parameters:
        buffer (numpy.ndarray):
            Current Buffer array of shape (n_outputs * n_history, 1).
        next_sample (numpy.array):
            Next Input array of shape (n_outputs, 1)

    Returns:
        numpy.ndarray: Updated buffer of shape (n_outputs * n_history, 1).
    """

    # Shift buffer down
    buffer[n_outputs:, :] = buffer[:(n_history-1)*n_outputs, :]

    # Insert new column Y[:, i+1] at the top
    buffer[:n_outputs, :] = next_sample.reshape(n_outputs, 1)

    return buffer  # Return one stacked column at a time

#-------------------------------------------------------------------------------

def collect_sample(Y, k, n_history, n_outputs):
    """
    Maintains a buffer and updates it by shifting previous values downward,
    inserting the new column Y[:, k+1] at the top.

    Parameters:
        Y (numpy.ndarray): Input array of shape (n_outputs, n_samples).
        k (int): Current column index.
        n_history (int): Number of past columns stored.

    Returns:
        numpy.ndarray: Updated buffer of shape (n_outputs * n_history, 1).
    """
    n_outputs, n_samples = Y.shape

    # Initialize buffer with the first stacked sample
    buffer = arrange_samples(Y, n_history-2, n_history, all=False)

    #print(f"scrolling k, from {k-1} to {n_samples - 1}")
    for i in range(k-1, n_samples - 1):
        # Shift buffer down
        buffer[n_outputs:, :] = buffer[:(n_history-1)*n_outputs, :]

        # Insert new column Y[:, i+1] at the top
        buffer[:n_outputs, :] = Y[:, i].reshape(n_outputs, 1)

        yield buffer  # Return one stacked column at a time

#-------------------------------------------------------------------------------
