import numpy as np
import csv

class Weights:
    def __init__(self) -> None:
        self.data = {}

    def __getitem__(self, indices: tuple[int, int]) -> np.ndarray:
        """Allow retrieval of weight matrix as weights[index1, index2]."""
        value = self.data.get(indices, None)
        if value is None:
            raise KeyError(f"No weights found for indices {indices}.")
        return value

    def __call__(self, index1: int, index2: int) -> np.ndarray:
        """Allow callable access to weight matrix."""
        value = self.data.get((index1, index2), None)
        if value is None:
            raise KeyError(f"No weights found for indices ({index1}, {index2}).")
        return value

    def set_weights(self, index1: int, index2: int, value: np.ndarray) -> None:
        """Store a weight matrix given source and destination layer indices."""
        self.data[(index1, index2)] = value

    def remove_weights(self, index1: int, index2: int) -> None:
        """Remove weight matrix between given layers, if it exists."""
        try:
            del self.data[(index1, index2)]
        except KeyError:
            raise KeyError(f"No existing weights for indices ({index1}, {index2}) to remove.")

    def print_weights(self) -> None:
        """Print all stored weight matrices."""
        print("Current weights:")
        for (index1, index2), value in self.data.items():
            print(f"From layer {index1} to layer {index2}: {value}")

    def load_from_csv(self, filename: str) -> None:
        """Load weight matrices from a CSV file.

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

    def save_to_csv(self, filename: str) -> None:
        """Save weight matrices to a CSV file."""
        try:
            with open(filename, 'w', newline='') as file:
                writer = csv.writer(file)
                for (index1, index2), value in self.data.items():
                    row = [index1, index2] + value.tolist()
                    writer.writerow(row)
        except Exception as e:
            raise ValueError(f"Error saving CSV file '{filename}': {e}")


class NNet:
    def __init__(self, n_inputs: int = None, n_hidden: list[int] = None, n_outputs: int = None,
                 nNodes: list[int] = None, connectivity=None,
                 activation_functions: list[str] = None,
                 training_data_file: str = 'trainingData.csv', testing_data_file: str = 'testingData.csv') -> None:
        # Determine network architecture based on provided parameters
        if nNodes is not None:
            self.layers = nNodes
            self.n_layers = len(nNodes)
            self.n_inputs = nNodes[0]
            self.n_outputs = nNodes[-1]
        else:
            self.layers = [n_inputs] + n_hidden + [n_outputs]
            self.n_layers = len(self.layers)

        # Establish default activation functions if not provided
        if activation_functions is not None:
            self.activation_functions = activation_functions
        else:
            self.activation_functions = [
                'linear' if i == 0 or i == self.n_layers - 1 else 'tanh'
                for i in range(self.n_layers)
            ]

        self.weights = Weights()

        # Initialize connectivity matrix and weight matrices
        self.connectivity = self.initialize_connectivity(connectivity)
        self.initialize_weights()

        # Load training and testing data
        self.training_data = self.load_data(training_data_file)
        self.testing_data = self.load_data(testing_data_file)

    def initialize_connectivity(self, connectivity):
        """Initialize a default sequential connectivity if none is provided.

        Default: Only adjacent layers are connected.
        """
        if connectivity is None:
            conn = np.zeros((self.n_layers, self.n_layers))
            for i in range(self.n_layers - 1):
                conn[i][i + 1] = 1  # Connect to next layer
            return conn
        return np.array(connectivity)

    def initialize_weights(self) -> None:
        """Initialize weight matrices for all connected layers.

        The weight matrix for a connection from layer i (source) to layer j (destination)
        is of size (neurons in destination, neurons in source).
        """
        for i in range(self.n_layers):
            for j in range(self.n_layers):
                if self.connectivity[i, j] == 1:
                    weight_matrix = np.random.rand(self.layers[j], self.layers[i])
                    self.weights.set_weights(i, j, weight_matrix)

    def load_data(self, filename: str) -> np.ndarray:
        """Load training or testing data from a CSV file."""
        try:
            with open(filename, 'r') as file:
                reader = csv.reader(file)
                data_list = []
                for row in reader:
                    if row:  # Skip empty rows
                        try:
                            data_list.append([float(val) for val in row])
                        except ValueError:
                            # Skip rows with conversion errors
                            continue
                return np.array(data_list, dtype=float)
        except Exception as e:
            raise ValueError(f"Error loading data from '{filename}': {e}")

    def save_weights(self, filename: str) -> None:
        """Save current weight matrices to a CSV file using our Weights class."""
        self.weights.save_to_csv(filename)

    def load_weights(self, filename: str) -> None:
        """Load weight matrices from a CSV file using our Weights class."""
        self.weights.load_from_csv(filename)

    def train(self, n_epochs: int = 1000, learning_rate: float = 0.01, display_epoch: int = 100) -> None:
        """Placeholder method for training the neural network."""
        for epoch in range(n_epochs):
            # Here you would implement the forward pass, loss calculation, and backpropagation.
            if (epoch + 1) % display_epoch == 0:
                print(f'Epoch {epoch + 1}/{n_epochs}: Loss = ...')  # Replace with actual loss calculation

    def predict(self, input_data):
        """Placeholder method for prediction using forward propagation."""
        # Implement your forward propagation here using self.weights.
        pass

    def __repr__(self) -> str:
        return f"NNet(n_layers={self.n_layers}, layers={self.layers}, activation_functions={self.activation_functions})"
