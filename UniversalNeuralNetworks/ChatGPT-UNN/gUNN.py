import numpy     as     np
import pandas    as     pd
from   typing    import Union, Callable, Dict, List, Tuple, Optional
from   functools import partial
import pickle
from   pathlib   import Path

# Custom types for clarity
Array        = np.ndarray
WeightsDict  = Dict[Tuple[int, int], Array]
BiasDict     = Dict[int, Array]
ActivationFn = Callable[[Array], Array]
InputOpFn    = Callable[[Array, Array, Array], Array]

#-------------------------------------------------------------------------------
class UNN:
    """Universal Neural Network with feedforward and feedback connections."""

    #---------------------------------------------------------------------------
    def __init__(
        self,
        n_inputs: int,
        n_hidden: Union[int, List[int]],
        n_outputs: int,
        connectivity: Optional[Array] = None,
        activations: Optional[Dict[int, str]] = None,
        input_ops: Optional[Dict[int, str]] = None,
        max_epochs: int = 1000,
        learning_rate: float = 0.01
    ) -> None:
        """Initialize UNN with topology and functionality."""
        # Handle n_hidden as scalar or list
        self.nodes = [n_inputs]
        self.nodes.extend(n_hidden if isinstance(n_hidden, list) else [n_hidden])
        self.nodes.append(n_outputs)
        self.n_layers = len(self.nodes)

        # Connectivity matrix: 1/True for connection, 0/False otherwise
        self.connectivity = connectivity if connectivity is not None else np.ones((self.n_layers, self.n_layers), dtype=bool)
        np.fill_diagonal(self.connectivity, False)  # No self-loops

        # Initialize weights and biases
        self.weights: WeightsDict = {}
        self.biases: BiasDict = {j: np.zeros(self.nodes[j]) for j in range(self.n_layers)}
        self.initialize_network()

        # Activation functions per layer
        self._activations: Dict[int, ActivationFn] = {}
        self._input_ops: Dict[int, InputOpFn] = {}
        self._set_functions(activations, input_ops)

        # Training parameters
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.momentum = 0.9
        self.prev_grads: WeightsDict = {}
        self.prev_bias_grads: BiasDict = {}

        # State for recurrent connections
        self.states: List[Array] = [np.zeros(self.nodes[j]) for j in range(self.n_layers)]
        self.outputs: List[Array] = [np.zeros(self.nodes[j]) for j in range(self.n_layers)]

        # Training history
        self.train_errors: List[float] = []
        self.test_errors: List[float] = []
        self.epochs_without_improvement = 0
        self.best_test_error = float('inf')

    #---------------------------------------------------------------------------
    def initialize_network(self) -> None:
        """Initialize weights and biases based on connectivity."""
        for i in range(self.n_layers):
            for j in range(self.n_layers):
                if self.connectivity[i, j]:
                    # Xavier initialization for weights
                    scale = np.sqrt(2.0 / (self.nodes[i] + self.nodes[j]))
                    self.weights[(i, j)] = np.random.normal(0, scale, (self.nodes[j], self.nodes[i]))
        # Biases initialized to zero
        self.biases = {j: np.zeros(self.nodes[j]) for j in range(self.n_layers)}

    #---------------------------------------------------------------------------
    def _set_functions(self, activations: Optional[Dict[int, str]], input_ops: Optional[Dict[int, str]]) -> None:
        """Set activation and input operator functions per layer."""
        # Default activation functions
        default_activations = {0: 'linear', self.n_layers-1: 'linear'}
        for j in range(1, self.n_layers-1):
            default_activations[j] = 'tanh'

        # Available activation functions
        activation_map = {
            'linear': lambda x: x,
            'tanh': np.tanh,
            'relu': lambda x: np.maximum(0, x),
            'cos': np.cos
        }

        # Available input operators
        input_op_map = {
            'default': lambda w, z, b: w @ z + b,
            'rbf': lambda w, z, b: np.exp(-np.sum((w - z)**2, axis=1) / (b + 1e-6)),
            'product': lambda w, z, b: np.prod(w * z, axis=1) / (b + 1e-6)
        }

        # Assign activations
        for j in range(self.n_layers):
            act_name = activations.get(j, default_activations.get(j, 'tanh')) if activations else default_activations.get(j, 'tanh')
            self._activations[j] = activation_map[act_name]

        # Assign input operators
        for j in range(self.n_layers):
            op_name = input_ops.get(j, 'default') if input_ops else 'default'
            self._input_ops[j] = input_op_map[op_name]

    #---------------------------------------------------------------------------
    def forward(self, inputs: Array, return_all: bool = False) -> Union[Array, Tuple[List[Array], List[Array]]]:
        """Forward pass for single or multiple samples."""
        # Debugging: Print raw input shape
        print(f"forward: Raw inputs shape: {inputs.shape}")

        # Handle single sample or batch
        is_batch = inputs.ndim > 1
        if is_batch:
            assert inputs.shape[1] == self.nodes[0], f"Expected {self.nodes[0]} inputs, got {inputs.shape[1]}"
            inputs = inputs.T  # Shape: (nodes[0], n_samples)
        else:
            assert inputs.shape[0] == self.nodes[0], f"Expected {self.nodes[0]} inputs, got {inputs.shape[0]}"
            inputs = inputs[:, None]  # Shape: (nodes[0], 1)

        # Debugging: Print transposed input shape
        print(f"forward: Transposed inputs shape: {inputs.shape}")

        # Verify transposed shape
        assert inputs.shape[0] == self.nodes[0], f"Expected {self.nodes[0]} input features, got {inputs.shape[0]}"

        n_samples = inputs.shape[1]

        # Initialize storage for states and outputs
        states = [np.zeros((self.nodes[j], n_samples)) for j in range(self.n_layers)]
        outputs = [np.zeros((self.nodes[j], n_samples)) for j in range(self.n_layers)]

        # Process each timestep
        for t in range(n_samples):
            # Set input layer
            states[0][:, t] = inputs[:, t]  # Shape: (nodes[0],)
            outputs[0][:, t] = self._activations[0](states[0][:, t])

            # Process each layer
            for j in range(1, self.n_layers):
                state = np.zeros(self.nodes[j])
                # Feedforward connections
                for i in range(j):
                    if self.connectivity[i, j]:
                        # Use outputs from current timestep
                        z = outputs[i][:, t]
                        state += self._input_ops[j](self.weights[(i, j)], z, self.biases[j])
                # Feedback connections (use previous timestep if available)
                for i in range(j, self.n_layers):
                    if self.connectivity[i, j]:
                        # Use state from previous timestep (t-1) if t > 0, else zero
                        z = states[i][:, t-1] if t > 0 else np.zeros(self.nodes[i])
                        state += self._input_ops[j](self.weights[(i, j)], z, self.biases[j])
                states[j][:, t] = state
                outputs[j][:, t] = self._activations[j](state)

        # Update instance states for consistency
        self.states = [states[j][:, -1] for j in range(self.n_layers)]
        self.outputs = [outputs[j][:, -1] for j in range(self.n_layers)]

        if return_all:
            return states, outputs
        return outputs[-1][:, 0] if n_samples == 1 else outputs[-1].T

    #---------------------------------------------------------------------------
    def cost_function(self, outputs: Array, targets: Array) -> float:
        """Compute cost: sum of log(cosh(error))."""
        errors = outputs - targets
        return np.sum(np.log(np.cosh(errors)))

    #---------------------------------------------------------------------------
    def compute_gradients(self, inputs: Array, targets: Array, timesteps: int = 1) -> Tuple[WeightsDict, BiasDict]:
        """Compute gradients using backpropagation through time (BPTT)."""
        is_batch = inputs.ndim > 1
        inputs = inputs.T if is_batch else inputs[:, None]
        targets = targets.T if is_batch else targets[:, None]

        # Forward pass with all states
        states, outputs = self.forward(inputs, return_all=True)

        # Initialize gradients
        grad_weights: WeightsDict = {(i, j): np.zeros_like(self.weights[(i, j)]) for i, j in self.weights}
        grad_biases: BiasDict = {j: np.zeros_like(self.biases[j]) for j in range(self.n_layers)}

        # Backprop through time
        n_samples = inputs.shape[1]
        for t in range(n_samples-1, -1, -1):
            # Compute output layer delta
            error = outputs[-1][:, t] - targets[:, t]
            delta = error * self._derivative(self._activations[-1], states[-1][:, t])

            # Backprop to hidden layers
            deltas = [np.zeros(self.nodes[j]) for j in range(self.n_layers)]
            deltas[-1] = delta

            for j in range(self.n_layers-2, -1, -1):
                delta_j = np.zeros(self.nodes[j])
                for k in range(j+1, self.n_layers):
                    if self.connectivity[j, k]:
                        delta_j += self.weights[(j, k)].T @ deltas[k]
                deltas[j] = delta_j * self._derivative(self._activations[j], states[j][:, t])

            # Compute gradients
            for j in range(self.n_layers):
                for i in range(self.n_layers):
                    if self.connectivity[i, j]:
                        grad_weights[(i, j)] += np.outer(deltas[j], outputs[i][:, t])
                grad_biases[j] += deltas[j]

        # Average gradients
        for key in grad_weights:
            grad_weights[key] /= n_samples
        for j in grad_biases:
            grad_biases[j] /= n_samples

        return grad_weights, grad_biases

    #---------------------------------------------------------------------------
    def _derivative(self, func: ActivationFn, x: Array) -> Array:
        """Approximate derivative of activation function."""
        if func == np.tanh:
            return 1 - np.tanh(x)**2
        elif func == (lambda x: x):
            return np.ones_like(x)
        elif func == (lambda x: np.maximum(0, x)):
            return np.where(x > 0, 1, 0)
        elif func == np.cos:
            return -np.sin(x)
        return np.ones_like(x)  # Fallback

    #---------------------------------------------------------------------------
    def update_weights(self, grad_weights: WeightsDict, grad_biases: BiasDict, method: str = 'sgd') -> None:
        """Update weights and biases using specified method."""
        if method == 'sgd':
            for (i, j), gw in grad_weights.items():
                self.weights[(i, j)] -= self.learning_rate * gw
            for j, gb in grad_biases.items():
                self.biases[j] -= self.learning_rate * gb
        elif method == 'momentum':
            for (i, j), gw in grad_weights.items():
                self.prev_grads[(i, j)] = self.momentum * self.prev_grads.get((i, j), 0) + self.learning_rate * gw
                self.weights[(i, j)] -= self.prev_grads[(i, j)]
            for j, gb in grad_biases.items():
                self.prev_bias_grads[j] = self.momentum * self.prev_bias_grads.get(j, 0) + self.learning_rate * gb
                self.biases[j] -= self.prev_bias_grads[j]

    #---------------------------------------------------------------------------
    def should_terminate(self, epoch: int, train_error: float, test_error: float) -> Tuple[bool, str]:
        """Check termination conditions."""
        if epoch >= self.max_epochs:
            return True, "Maximum epochs reached."
        if train_error < 1e-5:
            return True, "Training error below threshold."
        if test_error < self.best_test_error:
            self.best_test_error = test_error
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
        if self.epochs_without_improvement >= 50:
            return True, "Test error not improving for 50 epochs."
        return False, ""

    #---------------------------------------------------------------------------
    def train(self, train_file: str, test_file: Optional[str] = None, batch_size: int = 32) -> None:
        """Train the network using CSV files."""
        train_data = pd.read_csv(train_file).values
        test_data = pd.read_csv(test_file).values if test_file else train_data

        # Extract inputs and targets
        X_train = train_data[:, :self.nodes[0]]
        y_train = train_data[:, -self.nodes[-1]:]
        X_test = test_data[:, :self.nodes[0]]
        y_test = test_data[:, -self.nodes[-1]:]

        # Debugging: Print data shapes
        print(f"train: X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

        # Verify shapes
        assert X_train.shape[1] == self.nodes[0], f"Expected {self.nodes[0]} inputs, got {X_train.shape[1]}"
        assert y_train.shape[1] == self.nodes[-1], f"Expected {self.nodes[-1]} outputs, got {y_train.shape[1]}"

        for epoch in range(self.max_epochs):
            # Mini-batch training
            indices = np.random.permutation(len(X_train))
            for start in range(0, len(X_train), batch_size):
                batch_indices = indices[start:start + batch_size]
                X_batch = X_train[batch_indices]
                y_batch = y_train[batch_indices]

                # Debugging: Print batch shape
                print(f"train: X_batch shape: {X_batch.shape}, y_batch shape: {y_batch.shape}")

                # Verify batch shape
                assert X_batch.shape[1] == self.nodes[0], f"X_batch shape {X_batch.shape} incorrect"

                # Forward and compute gradients
                outputs = self.forward(X_batch)
                grad_weights, grad_biases = self.compute_gradients(X_batch, y_batch)
                self.update_weights(grad_weights, grad_biases, method='momentum')

            # Compute errors
            train_outputs = self.forward(X_train)
            train_error = self.cost_function(train_outputs, y_train)
            test_outputs = self.forward(X_test)
            test_error = self.cost_function(test_outputs, y_test)

            self.train_errors.append(train_error)
            self.test_errors.append(test_error)

            # Check termination
            should_stop, reason = self.should_terminate(epoch, train_error, test_error)
            if should_stop:
                print(f"Training stopped: {reason}")
                break
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Train Error: {train_error:.4f}, Test Error: {test_error:.4f}")

    #---------------------------------------------------------------------------
    def run(self, inputs: Array) -> Array:
        """Run the network for deployment (same as forward)."""
        return self.forward(inputs)

    #---------------------------------------------------------------------------
    def save_network(self, filename: str) -> None:
        """Save network topology, functionality, and parameters."""
        network = {
            'nodes': self.nodes,
            'connectivity': self.connectivity,
            'weights': self.weights,
            'biases': self.biases,
            'activations': {j: str(func) for j, func in self._activations.items()},
            'input_ops': {j: str(func) for j, func in self._input_ops.items()},
            'train_errors': self.train_errors,
            'test_errors': self.test_errors
        }
        with open(filename, 'wb') as f:
            pickle.dump(network, f)

    #---------------------------------------------------------------------------
    def load_network(self, filename: str) -> None:
        """Load network from file."""
        with open(filename, 'rb') as f:
            network = pickle.load(f)
        self.nodes = network['nodes']
        self.n_layers = len(self.nodes)
        self.connectivity = network['connectivity']
        self.weights = network['weights']
        self.biases = network['biases']
        self._set_functions(network['activations'], network['input_ops'])
        self.train_errors = network['train_errors']
        self.test_errors = network['test_errors']
        self.states = [np.zeros(self.nodes[j]) for j in range(self.n_layers)]
        self.outputs = [np.zeros(self.nodes[j]) for j in range(self.n_layers)]

    #---------------------------------------------------------------------------
    @staticmethod
    def add_activation(name: str, func: Callable[[Array], Array]) -> None:
        """Add custom activation function."""
        globals()['activation_map'][name] = func

    #---------------------------------------------------------------------------
    @staticmethod
    def add_input_op(name: str, func: Callable[[Array, Array, Array], Array]) -> None:
        """Add custom input operator."""
        globals()['input_op_map'][name] = func

# Example usage

# Assuming UNN class is defined as before
# Add this to the example usage section or a separate script

#-------------------------------------------------------------------------------
def create_train_csv(filename: str, n_samples: int, n_inputs: int, n_outputs: int) -> None:
    """Create a synthetic training dataset if train.csv doesn't exist."""
    if not Path(filename).exists():
        # Generate random inputs
        X = np.random.rand(n_samples, n_inputs)
        # Generate outputs (e.g., sum of inputs for simplicity)
        y = np.sum(X, axis=1, keepdims=True) if n_outputs == 1 else np.random.rand(n_samples, n_outputs)
        # Combine into a DataFrame
        data = pd.DataFrame(np.hstack([X, y]))
        # Save to CSV
        data.to_csv(filename, index=False)
        print(f"Created {filename} with {n_samples} samples.")
        # Debugging: Print CSV shape
        print(f"create_train_csv: Data shape: {data.shape}")

#-------------------------------------------------------------------------------
if __name__ == "__main__":
    # Network parameters
    n_inputs = 2
    n_hidden = 3
    n_outputs = 1
    n_samples = 100

    # Create train.csv if it doesn't exist
    create_train_csv("train.csv", n_samples, n_inputs, n_outputs)

    # Initialize and train the network
    unn = UNN(n_inputs=n_inputs, n_hidden=n_hidden, n_outputs=n_outputs)
    unn.train("train.csv")

    # Run example
    print(unn.run(np.array([0.5, 0.5])))

"""

#-------------------------------------------------------------------------------
if __name__ == "__main__":
    # Simple network: 2 inputs, 3 hidden, 1 output
    unn = UNN(n_inputs=2, n_hidden=3, n_outputs=1)

    # Generate dummy data
    X = np.random.rand(100, 2)
    y = np.sum(X, axis=1, keepdims=True)
    pd.DataFrame(np.hstack([X, y])).to_csv("train.csv", index=False)

    # Train
    unn.train("train.csv")

    # Save and load
    unn.save_network("unn.pkl")
    unn.load_network("unn.pkl")

    # Run
    print(unn.run(np.array([0.5, 0.5])))

#-------------------------------------------------------------------------------
if __name__ == "__main__":
    # Simple network: 2 inputs, 3 hidden, 1 output
    unn = UNN(n_inputs=2, n_hidden=3, n_outputs=1)

    # Generate dummy data
    X = np.random.rand(100, 2)
    y = np.sum(X, axis=1, keepdims=True)
    pd.DataFrame(np.hstack([X, y])).to_csv("train.csv", index=False)

    # Train
    unn.train("train.csv")

    # Run
    print(unn.run(np.array([0.5, 0.5])))

"""