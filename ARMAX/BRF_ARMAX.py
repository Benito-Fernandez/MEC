import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
#import matplotlib.animation as animation
#print(animation.FFMpegWriter.command)
from matplotlib.animation import FFMpegWriter, PillowWriter
import os
import pickle
import json
import inspect
import collections.abc

def display(var):
    # Try to get variable name from caller's local variables
    callers_locals = inspect.currentframe().f_back.f_locals
    var_names = [name for name, val in callers_locals.items() if val is var]
    var_name = var_names[0] if var_names else "<unknown>"

    print("-" * 40)
    print(f"Variable Name: {var_name}")
    print(f"Type: {type(var)}")

    # Try to print shape if applicable
    if hasattr(var, 'shape'):
        print(f"Shape: {var.shape}")
    elif isinstance(var, (list, tuple)):
        print(f"Length: {len(var)}")

    print("Value:")
    print(var)
    print("-" * 40)

class NonlinearARMAX_NN:
    def __init__(self, n_inputs=1, n_outputs=1, n_delayed_inputs=5, n_delayed_outputs=5,
                 n_delayed_noise=5, n_hidden=10, n_horizon=5, n_history=10,
                 weights=None, biases=None, DEBUG=False):
        # Network Parameters
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_delayed_inputs = n_delayed_inputs
        self.n_delayed_outputs = n_delayed_outputs
        self.n_delayed_noise = n_delayed_noise
        self.n_hidden = n_hidden
        self.n_horizon = n_horizon
        self.n_history = n_history

        if DEBUG:
            print(35*'-')
            print("    Creating a NonlinearARMAX_NN")
            print(" -> @NARMAX.init:: n_inputs= ",self.n_inputs)
            print(" -> @NARMAX.init:: n_outputs=",self.n_outputs)
            print(" -> @NARMAX.init:: n_hidden= ",self.n_hidden)
            print(" -> @NARMAX.init:: n_horizon= ",self.n_horizon)
            print(" -> @NARMAX.init:: n_history= ",self.n_history)
            print(" -> @NARMAX.init:: n_delayed_inputs= ",self.n_delayed_inputs)
            print(" -> @NARMAX.init:: n_delayed_outputs=",self.n_delayed_outputs)
            print(" -> @NARMAX.init:: n_delayed_noise=  ",self.n_delayed_noise)
            print(" -> @NARMAX.init:: weights=\n",weights)
            #print(" -> @NARMAX.init:: weights==None?",weights==None)
            print(" -> @NARMAX.init:: biases=\n",biases)
            #print(" -> @NARMAX.init:: biases==None?",biases==None)
            print(35*'-')

        # Initialize weights and biases
        input_size = n_inputs * (1 + n_delayed_inputs) + n_outputs * (1 + n_delayed_outputs) + n_delayed_noise        # Verify sizes
        self.weights = {
            "input_hidden":  np.random.randn(input_size, n_hidden),
            "hidden_output": np.random.randn(n_hidden, n_outputs * n_horizon),
        }

        # Load weights
        if weights==None:
            print(f" -> @NARMAX.init:: When weights == None")
        else: # weights != None
            print(f" -> @NARMAX.init:: When weights != None")
#-----------------------------------------
            # Extract weights from weights_array
            input_elements = input_size * n_hidden  # 110
            print(50*'*')
            print(f"input_elements = input_size * n_hidden: {input_elements}")
            print(f"weights.shape: {weights.shape}, weights.size: {weights.size} \n{weights}")

            print(50*'*')
            self.weights['input_hidden'] = np.array(weights.item()["input_hidden"])  # Extract the dictionary and access 'input_hidden'
            print(f"self.weights['input_hidden'] = \n{self.weights['input_hidden']}")
            self.weights['hidden_output'] = np.array(weights.item()["hidden_output"])  # Extract the dictionary and access 'input_hidden'
            print(f"self.weights['hidden_output'] = \n{self.weights['hidden_output']}")
            print(50*'*')

            # Update with validation
            if isinstance(weights, (list, tuple)) and len(weights) == 2:
                if weights_array[0].shape == (input_size, n_hidden) and weights_array[1].shape == (n_hidden, n_outputs * n_horizon):
                    self.weights['input'] = weights[0]
                    self.weights['hidden'] = weights[1]
                else:
                    print("ERROR:: Shape mismatch in weights_array elements")
                    raise ValueError("Shape mismatch in weights_array elements")
            elif weights.size == (input_size * n_hidden + n_hidden * n_outputs * n_horizon):
                input_elements = input_size * n_hidden
                self.weights['input'] = weights[:input_elements].reshape(input_size, n_hidden)
                self.weights['hidden'] = weights[input_elements:].reshape(n_hidden, n_outputs * n_horizon)
            else:
                raise ValueError(f"weights incompatible with shapes ({input_size}, {n_hidden}) and ({n_hidden}, {n_outputs * n_horizon})")
#-----------------------------------------
            print(f" -> @NARMAX.init:: When weights != None")
            display(self.weights)
            print(50*'*')
            if weights.shape == (2, input_size, n_hidden):
                self.weights = {
                    "input_hidden":  weights[0],# np.zeros((input_size, n_hidden)),
                    "hidden_output": weights[1] # np.zeros((n_hidden, n_outputs * n_horizon))
                }
            print(f"Type(weights): {type(weights)}; Size: {weights.size}")
            print(weights)
            if verify_weights(weights, n_inputs, n_hidden, n_outputs):
                print(" -> @NARMAX.init:: weights Dimensions verified")
            self.weights = weights
            print("weights['input_hidden'] = ", weights['input_hidden'])
            for i, weight_matrix in enumerate(weights):
                expected_shape = (n_inputs if i == 0 else n_hidden, n_hidden if i < len(weights) - 1 else n_outputs)
                if weight_matrix.shape != expected_shape:
                    raise ValueError(f"Weight matrix {i} has incorrect shape: {weight_matrix.shape}. Expected: {expected_shape}")

         # Load biases
        if biases==None:
            self.biases = {
                "hidden_bias": np.random.randn(n_hidden),
                "output_bias": np.random.randn(n_outputs * n_horizon),
            }
        else:
            if verify_biases(biases, n_inputs, n_hidden, n_outputs):
                print(" -> @NARMAX.init:: biases Dimensions verified")
            for i, bias_vector in enumerate(biases):
                expected_shape = (n_hidden if i < len(biases) - 1 else n_outputs,)
                if bias_vector.shape != expected_shape:
                    raise ValueError(f"Bias vector {i} has incorrect shape: {bias_vector.shape}. Expected: {expected_shape}")
            self.biases = biases

    # Activation Function and its Derivative
    def activation_function(self, x):
        return np.tanh(x)

    # Activation Function and its Derivative
    def activation_derivative(self, x):
        return 1 - np.tanh(x) ** 2

    # Loss Function
    def loss(self, e, gain=1, gamma=2):
        return np.power(gain, np.tanh(gamma * e)) * np.log(np.cosh(e))

    def loss_(self, e, gain=1, gamma=2):
        return np.power(gain, np.tanh(gamma * e)) * np.tanh(e)

    # Forward Pass
    def forward(self, x):
        self.hidden = self.activation_function(np.dot(x, self.weights["input_hidden"]) + self.biases["hidden_bias"])
        self.output = np.dot(self.hidden, self.weights["hidden_output"]) + self.biases["output_bias"]
        return self.output

    # Backward Pass
    def backward(self, x, y, lr, momentum, prev_deltas):
        error = self.output - y
        output_grad = self.loss_(error).reshape(-1, self.n_outputs * self.n_horizon)
        hidden_grad = np.dot(output_grad, self.weights["hidden_output"].T) * self.activation_derivative(self.hidden)

        deltas = {
            "hidden_output": momentum * prev_deltas["hidden_output"] - lr * np.dot(self.hidden.T, output_grad),
            "input_hidden":  momentum * prev_deltas["input_hidden"]  - lr * np.dot(x.T, hidden_grad),
            "output_bias":   momentum * prev_deltas["output_bias"]   - lr * np.sum(output_grad, axis=0),
            "hidden_bias":   momentum * prev_deltas["hidden_bias"]   - lr * np.sum(hidden_grad, axis=0),
        }

        # Update weights and biases
        self.weights["hidden_output"] += deltas["hidden_output"]
        self.weights["input_hidden"]  += deltas["input_hidden"]
        self.biases["output_bias"]    += deltas["output_bias"]
        self.biases["hidden_bias"]    += deltas["hidden_bias"]

        return deltas

    # SDisplays Network
    def __str__(self):
        # Retrieve the variable name
        #name = [name for name, value in globals().items() if value is model]
        #print(variable_name)  # Output: ['model']
        print(f"NARMAX:: model: {self.name}\n",
        f"        n_inputs:          {model.n_inputs}\n",
        f"        n_outputs:         {model.n_outputs}\n",
        f"        n_hidden:          {model.n_hidden}\n",
        f"        n_horizon:         {model.n_horizon}\n",
        f"        n_delayed_inputs:  {model.n_delayed_inputs}\n",
        f"        n_delayed_outputs: {model.n_delayed_outputs}\n",
        f"        n_delayed_noise:   {model.n_delayed_noise}\n",
        f"        input_hidden.wts:  {model.weights['input_hidden']}\n",
        f"        hidden_output.wts: {model.weights['hidden_output']}\n",
        f"        hidden_biases:     {model.biases['hidden_bias']}\n",
        f"        output_biases:     {model.biases['output_bias']}\n")

    def save_network(self, filename, mode='json'):
        """
        Saves the ARMAX neural network topology, weights, biases, and delayed parameters to a file.

        Args:
            filename (str): Path to the file where the network should be saved.
        """
        # Gather all the parameters in a dictionary
        if mode=='json':
            # Convert weights dictionary values to lists
            serialized_weights = {key: value.tolist() if hasattr(value, "tolist") else value for key, value in self.weights.items()}

            # Convert biases similarly if they are a dictionary
            serialized_biases = {key: value.tolist() if hasattr(value, "tolist") else value for key, value in self.biases.items()}

            # Create a dictionary for network parameters
            network_data = {
                "n_inputs": self.n_inputs,
                "n_outputs": self.n_outputs,
                "n_hidden": self.n_hidden,
                "n_horizon": self.n_horizon,
                "n_delayed_inputs": self.n_delayed_inputs,
                "n_delayed_outputs": self.n_delayed_outputs,
                "n_delayed_noise": self.n_delayed_noise,
                "weights": serialized_weights,
                "biases": serialized_biases
            }
            # Save the dictionary to a JSON file
            with open(filename, 'w') as f:
                json.dump(network_data, f, indent=4)

        else: # <- mode=='pickle'
            # Gather all data into a dictionary
            network_data = {
                "n_inputs": self.n_inputs,
                "n_outputs": self.n_outputs,
                "n_hidden": self.n_hidden,
                "n_horizon": self.n_horizon,
                "n_delayed_inputs": self.n_delayed_inputs,
                "n_delayed_outputs": self.n_delayed_outputs,
                "n_delayed_noise": self.n_delayed_noise,
                "weights": self.weights,
                "biases": self.biases
            }

            # Save the dictionary to a pickle file
            with open(filename, 'wb') as f:
                pickle.dump(network_data, f)

        if DEBUG:
            print(f"Network saved successfully to '{filename}'")

    # Load Network Topology and Weights
    @classmethod
    def load_network(self, filename, mode='json'):
        """
        Loads the ARMAX neural network parameters from a JSON file.

        Args:
            filename (str): Path to the saved network file.

        Returns:
            dict: The loaded network parameters.
        """
        if DEBUG:
            print(f"@ load_network:: Filename: {filename}, Type: {type(filename)}")
        if mode=='json':
            with open(filename, 'r') as f:
                network_data = json.load(f)

            # Convert lists back to NumPy arrays
            network_data['weights'] = np.array(network_data['weights'])
            network_data['biases'] = np.array(network_data['biases'])

        else: #<- mode=='pickle':
            # Ensure filename is a valid type
            if not isinstance(filename, (str, bytes, os.PathLike)):
                raise TypeError(f"Invalid filename: {filename}. Expected str, bytes, or os.PathLike.")
            with open(filename, 'rb') as f:
                network_data = pickle.load(f)

        if DEBUG:
            print(f"@ load_network:: Network loaded successfully from '{filename}'")
            print(network_data)
        # Create an instance of the class
        if DEBUG:
            print(f"@ load_network:: creating model instance:")
            print("'model_instance = NonlinearARMAX_NN(...'")
            print(f'                 n_inputs={network_data["n_inputs"]},')
            print(f'                 n_outputs={network_data["n_outputs"]},')
            print(f'                 n_hidden={network_data["n_hidden"]},')
            print(f'                 n_horizon={network_data["n_horizon"]},')
            print(f'                 n_delayed_inputs={network_data["n_delayed_inputs"]},')
            print(f'                 n_delayed_outputs={network_data["n_delayed_outputs"]},')
            print(f'                 n_delayed_noise=network_data["n_delayed_noise"],')
            print(f'                 weights={network_data["weights"]},')
            print(f'                 biases={network_data["biases"]},')
            print(f'                 DEBUG={DEBUG}')
        model_instance = NonlinearARMAX_NN(
            n_inputs=network_data["n_inputs"],
            n_outputs=network_data["n_outputs"],
            n_hidden=network_data["n_hidden"],
            n_horizon=network_data["n_horizon"],
            n_delayed_inputs=network_data["n_delayed_inputs"],
            n_delayed_outputs=network_data["n_delayed_outputs"],
            n_delayed_noise=network_data["n_delayed_noise"],
            weights=network_data["weights"],
            biases=network_data["biases"],
            DEBUG=DEBUG
        )

        print("@ load_network:: type(network_data['weights'] = ", type(network_data['weights']))
        if DEBUG:
            print("@ load_network:: Instance of NonlinearARMAX_NN created successfully!")
        return model_instance

def create_delayed_features(data, n_inputs, n_outputs, n_delayed_inputs, n_delayed_outputs, n_delayed_noise, noise_std=0.2):
    # Immediate Inputs
    inputs = data['force'].values

    # Immediate Outputs
    outputs = data[['position', 'velocity']].values

    # Noise
    noise = np.random.normal(0, noise_std, len(inputs))

    # Create Delayed Inputs
    delayed_inputs = np.zeros((len(inputs), n_delayed_inputs * n_inputs))
    for i in range(1, n_delayed_inputs + 1):
        delayed_inputs[i:, (i - 1) * n_inputs:i * n_inputs] = inputs[:-i].reshape(-1, n_inputs)

    # Create Delayed Outputs
    delayed_outputs = np.zeros((len(inputs), n_delayed_outputs * n_outputs))
    for i in range(1, n_delayed_outputs + 1):
        delayed_outputs[i:, (i - 1) * n_outputs:i * n_outputs] = outputs[:-i, :]

    # Create Delayed Noise
    delayed_noise = np.zeros((len(inputs), n_delayed_noise))
    for i in range(1, n_delayed_noise + 1):
        delayed_noise[i:, i - 1] = noise[:-i]

    # Combine Features
    combined_features = np.hstack([
        inputs.reshape(-1, n_inputs),
        delayed_inputs,
        outputs,
        delayed_outputs,
        delayed_noise
    ])
    return combined_features

def prepare_output_matrix(data, n_outputs, n_horizon):
    """
    Prepares the output matrix for comparison with multi-step predictions.

    Args:
        data (pd.DataFrame): DataFrame containing columns for outputs (e.g., 'position', 'velocity').
        n_outputs (int): Number of output signals.
        n_horizon (int): Number of prediction steps ahead.

    Returns:
        np.ndarray: Output matrix with shape (num_samples, n_outputs * n_horizon).
    """
    outputs = data[['position', 'velocity']].values
    num_samples = len(outputs) - n_horizon + 1
    horizon_outputs = np.zeros((num_samples, n_outputs * n_horizon))

    for t in range(num_samples):
        for h in range(n_horizon):
            horizon_outputs[t, h * n_outputs:(h + 1) * n_outputs] = outputs[t + h]

    return horizon_outputs

def prepare_horizon_outputs(data, n_outputs, n_horizon):
    """
    Prepares the output matrix for multi-step predictions.

    Args:
        data (pd.DataFrame): DataFrame containing columns for outputs (e.g., 'position', 'velocity').
        n_outputs (int): Number of output signals.
        n_horizon (int): Number of prediction steps ahead.

    Returns:
        np.ndarray: Output matrix with shape (num_samples, n_outputs * n_horizon).
    """
    outputs = data[['position', 'velocity']].values
    num_samples = len(outputs) - n_horizon + 1  # Adjust for horizon
    horizon_outputs = np.zeros((num_samples, n_outputs * n_horizon))

    for t in range(num_samples):  # Prepare horizon outputs
        for h in range(n_horizon):
            horizon_outputs[t, h * n_outputs:(h + 1) * n_outputs] = outputs[t + h]

    return horizon_outputs

# Verify weights and biases structure and sizes
def verify_weights(weights, input_size, n_hidden, n_outputs, n_horizon):
    """
    Validates the dimensions of the provided weights.

    Args:
        weights   (dict): Dictionary containing weight matrices.
        input_size (int): Size of the input layer.
        n_hidden   (int): Number of hidden neurons.
        n_outputs  (int): Number of output neurons.
        n_horizon  (int): Prediction horizon.

    Raises:
        ValueError: If any weight matrix has an incorrect shape.
    """
    expected_shapes = {
        "input_hidden":  (input_size, n_hidden),
        "hidden_output": (n_hidden, n_outputs * n_horizon)
    }

    for key, matrix in weights.items():
        if key not in expected_shapes:
            raise ValueError(f"Unexpected key in weights: {key}")

        if matrix.shape != expected_shapes[key]:
            raise ValueError(f"Weight matrix '{key}' has incorrect shape: {matrix.shape}. "
                             f"Expected: {expected_shapes[key]}")

    return True

# Verify weights and biases structure and sizes
def verify_biases(biases, n_hidden, n_outputs, n_horizon):
    """
    Validates the dimensions of the provided biases.

    Args:
        biases (dict): Dictionary containing bias vectors.
        n_hidden (int): Number of hidden neurons.
        n_outputs (int): Number of output neurons.
        n_horizon (int): Prediction horizon.

    Raises:
        ValueError: If any bias vector has an incorrect shape.
    """
    expected_shapes = {
        "hidden_bias": (n_hidden,),
        "output_bias": (n_outputs * n_horizon,)
    }

    for key, vector in biases.items():
        if key not in expected_shapes:
            raise ValueError(f"Unexpected key in biases: {key}")

        if vector.shape != expected_shapes[key]:
            raise ValueError(f"Bias vector '{key}' has incorrect shape: {vector.shape}. "
                             f"Expected: {expected_shapes[key]}")

    return True

# Training Function
def train_network(model, training_file, n_epochs=1000, n_show=100, noise_std=0.2):
    """
    Trains the ARMAX neural network using the training dataset.

    Args:
        model (NonlinearARMAXNN): The ARMAX neural network model.
        training_file      (str): Path to the training CSV file.
        n_epochs           (int): Number of epochs for training.
        n_show             (int): Interval for displaying training progress.
        noise_std        (float): Standard deviation of Gaussian noise for delayed noise terms.

    Returns:
        list: Loss values for each epoch.
    """
    # Load data
    data = pd.read_csv(training_file)
    time = data['time']

    # Prepare input features and actual outputs
    inputs = create_delayed_features(data, model.n_inputs, model.n_outputs, model.n_delayed_inputs, model.n_delayed_outputs, model.n_delayed_noise, noise_std)
    outputs = prepare_horizon_outputs(data, model.n_outputs, model.n_horizon)

    # Align inputs with outputs size
    inputs = inputs[:outputs.shape[0]]

    # Debugging shapes
    if DEBUG:
        print(f"Shape of inputs:  {inputs.shape}")
        print(f"Shape of outputs: {outputs.shape}")

    losses = []
    prev_deltas = {key: np.zeros_like(value) for key, value in model.weights.items()}
    prev_deltas.update({key: np.zeros_like(value) for key, value in model.biases.items()})

    for epoch in range(1, n_epochs + 1):
        y_pred = model.forward(inputs)
        error = y_pred - outputs
        loss = np.mean(model.loss(error))
        losses.append(loss)

        model.backward(inputs, outputs, lr=0.01, momentum=0.9, prev_deltas=prev_deltas)

        # Plot every n_show epochs
        if epoch % n_show == 0:
            plt.figure()
            plt.clf()
            plt.plot(time[:outputs.shape[0]], outputs[:, 0], 'k.', label="Position (Training Data)")
            plt.plot(time[:outputs.shape[0]], y_pred[:, 0], 'r', label="Position (Predicted Outputs)")
            plt.title(f"Epoch {epoch}")
            plt.legend()
            plt.pause(1)

    # Save loss plot
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.figure()
    plt.plot(range(1, n_epochs + 1), losses, label="Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss vs. Epochs")
    plt.legend()
    plt.savefig(f"Loss_Plot@{timestamp}.pdf")
    plt.savefig(f"Loss_Plot@{timestamp}.jpeg")

    return losses

def train_network_with_animation(trn_model, training_file, n_epochs=1000, n_show=100, noise_std=0.2, animation_file="training_animation.mp4"):
    """
    Trains the ARMAX neural network and generates an animation of training progress.

    Args:
        trn_model (NonlinearARMAX_NN): The Nonlinear ARMAX neural network model.
        training_file (str): Path to the training CSV file.
        n_epochs (int): Number of epochs for training.
        n_show (int): Interval for displaying training progress.
        noise_std (float): Standard deviation of Gaussian noise for delayed noise terms.
        animation_file (str): Path to save the animation file.
    """
    from matplotlib.animation import FFMpegWriter  # Use ffmpeg for MP4
    from matplotlib.animation import PillowWriter  # Use Pillow for GIFs

    # Load data
    data = pd.read_csv(training_file)
    time = data['time']

    # Prepare input features and actual outputs
    inputs = create_delayed_features(
        data,
        trn_model.n_inputs,
        trn_model.n_outputs,
        trn_model.n_delayed_inputs,
        trn_model.n_delayed_outputs,
        trn_model.n_delayed_noise,
        noise_std
    )
    outputs = prepare_horizon_outputs(data, trn_model.n_outputs, trn_model.n_horizon)

    # Align input size with horizon outputs
    inputs = inputs[:outputs.shape[0]]

    # Debugging shapes
    if DEBUG:
        print(f"Shape of inputs: {inputs.shape}")
        print(f"Shape of outputs: {outputs.shape}")

    losses = []
    prev_deltas = {key: np.zeros_like(value) for key, value in trn_model.weights.items()}
    prev_deltas.update({key: np.zeros_like(value) for key, value in trn_model.biases.items()})

    # Setup animation writer
    # animation_file = "training_animation.mp4" #<-  parameter in method
    # Attempt FFmpeg writer first, fallback to GIF
    try:
        writer = FFMpegWriter(fps=5, metadata=dict(artist='ARMAX'), bitrate=1800)
        if DEBUG:
            print("FFmpeg found. writer created.")
    except FileNotFoundError:
        if DEBUG:
            print("FFmpeg not found. Falling back to GIF...")
        writer = PillowWriter(fps=5)
        animation_file = animation_file.replace(".mp4", ".gif")
    writer = PillowWriter(fps=5)
    animation_file = animation_file.replace(".mp4", ".gif")

    # Debug:
    if DEBUG:
        print(f"writer '{writer}' created. \nIs writer Available? {writer.isAvailable()}")

    fig, ax = plt.subplots()
    with writer.saving(fig, animation_file, dpi=100):  # Save frames to an GIF file
        for epoch in range(1, n_epochs + 1):
            y_pred = trn_model.forward(inputs)
            error = y_pred - outputs
            loss = np.mean(trn_model.loss(error))
            losses.append(loss)

            trn_model.backward(inputs, outputs, lr=0.01, momentum=0.9, prev_deltas=prev_deltas)

            if epoch % n_show == 0:
                ax.clear()  # Clear the previous plot
                ax.plot(time[:outputs.shape[0]], outputs[:, 0], 'k.', label="Position (Training Data)")
                ax.plot(time[:outputs.shape[0]], y_pred[:, 0],  'r',  label="Position (Predicted Outputs)")
                ax.set_title(f"Epoch {epoch}")
                ax.legend()

                # Save the current frame to the animation
                writer.grab_frame()
                plt.pause(0.1)  # Pause for a brief moment to visualize
        plt.show()

    # Save loss plot
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.figure()
    plt.plot(range(1, n_epochs + 1), losses, label="Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss vs. Epochs")
    plt.legend()
    plt.savefig(f"Loss_Plot@{timestamp}.pdf")
    plt.savefig(f"Loss_Plot@{timestamp}.jpeg")

    if DEBUG:
        print(f"Animation saved as {animation_file}")
    return losses

# Testing Function
def test_network(tst_model, testing_file, n_horizon, n_skip=5, noise_std=0.2):
    # Load test data
    data = pd.read_csv(training_file)
    time = data['time']
    if DEBUG:
        print(f"Testing network {tst_model}")

    n_inputs           = tst_model['n_inputs']
    n_hidden           = tst_model['n_hidden']
    n_outputs          = tst_model['n_outputs']
    n_delayed_inputs   = tst_model['n_delayed_inputs']
    n_delayed_outputs  = tst_model['n_delayed_outputs']
    n_delayed_noise    = tst_model['n_delayed_noise']

    if True:
        print(f"Testing network parameters {n_inputs}x{n_hidden}x{n_outputs}")

    # Prepare input features and actual outputs
    inputs = create_delayed_features(
        data,
        n_inputs,
        n_outputs,
        n_delayed_inputs,
        n_delayed_outputs,
        n_delayed_noise,
        noise_std
    )
    outputs = prepare_horizon_outputs(data, n_outputs, n_horizon)

    predictions = []
    for t_k in range(0, len(inputs) - n_horizon, n_skip):
        input_data = inputs[t_k:t_k + n_horizon]
        forecast = tst_model.forward(input_data)
        predictions.append(forecast)

        plt.figure()
        plt.plot(time[:t_k + n_horizon], outputs[:t_k + n_horizon].flatten(), 'k.', label="Historic Data")
        plt.plot(time[t_k:t_k + n_horizon], forecast.flatten(), 'r', label="Forecast")
        plt.fill_between(time[t_k:t_k + n_horizon], forecast.flatten() - 0.1, forecast.flatten() + 0.1, alpha=0.2)
        plt.title(f"Forecast at t = {t_k}")
        plt.legend()
        plt.pause(0.1)

    # Save predictions as CSV
    pd.DataFrame(predictions).to_csv("Simulation_Results.csv", index=False)

# When run as file
if __name__ == "__main__":
    global DEBUG
    DEBUG = True   #<-  For debugging purposes

    # Initialize ARMAX Neural Network
    #---------------------------------------------------------------------------
    if DEBUG:
        print(65*'-')
        print(f"NARMAX Creating instance of 'model = NonlinearARMAX_NN(...)'.")
    model = NonlinearARMAX_NN(
        n_inputs=1,               # One input: [Force]
        n_outputs=2,              # Two outputs: [Position, Velocity]
        n_delayed_inputs=2,       # Delayed inputs for X in NARMAX
        n_delayed_outputs=2,      # Delayed outputs for AR in NARMAX
        n_delayed_noise=2,        # Delayed noise for MA in NARMAX
        n_hidden=10,              # Hidden layer size, for N in NARMAX
        n_horizon=3               # Predict 3 steps ahead
    )

    if DEBUG:
        print(65*'-')
        print(f"Nonlinear ARMAX NeuralNet Model created. Dictionary: \n{dir(model)}")
        print(65*'-')


    # Train the network
    #---------------------------------------------------------------------------
    training_file="training_file.csv"
    animation_file="training_animation.mp4"
    training_epochs = 100  # <- total number of training epochs
    display_epochs  = 25   # <- how often intermediate results are shown
    mode = 'json'           # <- what method is use to save and load network

    #train_network(model, training_file, n_epochs=1000, n_show=100)
    train_network_with_animation(model, training_file,
                                 n_epochs=training_epochs,
                                 n_show=display_epochs,
                                 animation_file=animation_file)
    # Save the trained network
    #---------------------------------------------------------------------------
    model_file="trained_network.net"
    model.save_network(model_file)

##    # Save the object to a file
##    with open(f"{model_file}.pkl", "wb") as file:
##        pickle.dump(model, file)

    if DEBUG:
        print(65*'-')
        print(f"NARMAX NNet saved successfully to '{model_file}' using '{mode}'.")
    if DEBUG:
        print(65*'-')
        print(f"NARMAX:: model saved to:    {model_file}\n",
              f"        n_inputs:           {model.n_inputs}\n",
              f"        n_outputs:          {model.n_outputs}\n",
              f"        n_hidden:           {model.n_hidden}\n",
              f"        n_horizon:          {model.n_horizon}\n",
              f"        n_delayed_inputs:   {model.n_delayed_inputs}\n",
              f"        n_delayed_outputs:  {model.n_delayed_outputs}\n",
              f"        n_delayed_noise:    {model.n_delayed_noise}\n",
              f"        input_hidden.wts:\n {model.weights['input_hidden']}\n",
              f"        hidden_output.wts:\n{model.weights['hidden_output']}\n",
              f"        hidden_biases:\n    {model.biases['hidden_bias']}\n",
              f"        output_biases:\n    {model.biases['output_bias']}\n")
        print(65*'-')

    # Load the trained network
    #---------------------------------------------------------------------------
    recovered_model = NonlinearARMAX_NN.load_network(filename=model_file)
##    # Load the object from the file
##    with open(f"{model_file}.pkl", "rb") as file:
##        loaded_model = pickle.load(file)

    if DEBUG:
        print(65*'-')
        print(f"NARMAX:: NNet loaded successfully from '{model_file}' using '{mode}'.")
        print(f"NARMAX:: Recovered model: {recovered_model}")

    if DEBUG:
        print(65*'-')
        print(f"NARMAX:: model loaded from:  {model_file}\n",
              f"         n_inputs:           {recovered_model.n_inputs}\n",
              f"         n_outputs:          {recovered_model.n_outputs}\n",
              f"         n_hidden:           {recovered_model.n_hidden}\n",
              f"         n_horizon:          {recovered_model.n_horizon}\n",
              f"         n_delayed_inputs:   {recovered_model.n_delayed_inputs}\n",
              f"         n_delayed_outputs:  {recovered_model.n_delayed_outputs}\n",
              f"         n_delayed_noise:    {recovered_model.n_delayed_noise}\n",
              f"         input_hidden.wts:\n {recovered_model.weights['input_hidden']}\n",
              f"         hidden_output.wts:\n{recovered_model.weights['hidden_output']}\n",
              f"         hidden_biases:\n    {recovered_model.biases['hidden_bias']}\n",
              f"         output_biases:\n    {recovered_model.biases['output_bias']}\n")
        print(65*'-')

    # Test the network
    #---------------------------------------------------------------------------
    testing_file="testing_file.csv"
    test_network(recovered_model, testing_file, n_horizon=3, n_skip=5)
    if DEBUG:
        print(recovered_model.n_inputs, loaded_model.n_outputs, loaded_model.n_hidden)

