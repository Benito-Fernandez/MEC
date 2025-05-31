from Weights import *
class NNet:
    def __init__(self, n_inputs=None, n_hidden=None, n_outputs=None, nNodes=None, connectivity=None,
                 activation_functions=None, training_data_file='trainingData.csv', testing_data_file='testingData.csv'):
        if nNodes is not None:
            self.n_layers = len(nNodes)
            self.layers = nNodes
            self.n_inputs = nNodes[0]
            self.n_outputs = nNodes[-1]
        else:
            self.layers = [n_inputs] + n_hidden + [n_outputs]
            self.n_layers = len(self.layers)

        self.activation_functions = activation_functions or ['linear' if i == 0 or i == self.n_layers - 1 else 'tanh' for i in range(self.n_layers)]
        self.weights = Weights()

        # Initialize connectivity matrix
        self.connectivity = self.initialize_connectivity(connectivity)
        self.initialize_weights()

        # Load training and testing data
        self.training_data = self.load_data(training_data_file)
        self.testing_data = self.load_data(testing_data_file)

    def initialize_connectivity(self, connectivity):
        if connectivity is None: # if no connectivity, create MLP (FF) Net
            conn = np.zeros((self.n_layers, self.n_layers))
            for i in range(self.n_layers - 1):
                conn[i][i + 1] = 1  # Connect to next layer
            return conn
        return np.array(connectivity)

    def initialize_weights(self):
        for i in range(self.n_layers):
            for j in range(self.n_layers):
                if self.connectivity[i, j] == 1:
                    # The weight matrix should have shape
                    # (neurons in destination, neurons in source)
                    weight_matrix = np.random.rand(self.layers[j], self.layers[i])
                    self.weights.set_weights(i, j, weight_matrix)

##    def load_data(self, filename):
##        with open(filename, 'r') as file:
##            reader = csv.reader(file)
##            return np.array([row for row in reader], dtype=float)

    def load_data(self, filename: str) -> np.ndarray:
        try:
            with open(filename, 'r') as file:
                reader = csv.reader(file)
                return np.array([row for row in reader], dtype=float)
        except Exception as e:
            raise ValueError(f"Message error loading data from {filename}: {e}")

    def save_weights(self, filename):
        self.weights.save_to_csv(filename)

    def load_weights(self, filename):
        self.weights.load_from_csv(filename)

    def train(self, n_epochs=1000, learning_rate=0.01, display_epoch=100):
        for epoch in range(n_epochs):
            # Create a forward pass and backpropagation for training logic
            # Here you'd compute the outputs with the current weights, calculate the loss, and update
            # ...

            if (epoch + 1) % display_epoch == 0:
                print(f'Epoch {epoch + 1}/{n_epochs}: Loss = ...')  # Replace with actual loss calculation

    def predict(self, input_data):
        # Method for prediction based on the trained weights
        # Here you would implement forward propagation based on current weights
        pass

    def __repr__(self):
        return f"NNet(n_layers={self.n_layers}, layers={self.layers}, activation_functions={self.activation_functions})"

