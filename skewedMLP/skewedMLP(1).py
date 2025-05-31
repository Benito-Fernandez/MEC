import numpy as np
import matplotlib.pyplot as plt
from   scipy.stats   import logistic, hypsecant


class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights and biases
        self.weights = {
            "input_hidden": np.random.randn(input_size, hidden_size),
            "hidden_output": np.random.randn(hidden_size, output_size),
        }
        self.biases = {
            "hidden": np.random.randn(hidden_size),
            "output": np.random.randn(output_size),
        }

    # Activation function
    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivative(self, x):
        return 1 - np.tanh(x) ** 2

    # Custom Skewed Log-Cosh Loss Function
    def loss(self, e, slope=1, gamma=2):
        gain = np.power(slope,np.tanh(gamma*e))
        cost = gain*np.log(np.cosh(e))
        return cost                    # cost function: skLogCosh

    def loss_gradient(self, e, slope=1, gamma=2):
        gain = np.power(slope,np.tanh(gamma*e))
        gradient = gain*(np.tanh(e) \
                 + gamma*np.log(slope)*np.power(hypsecant.pdf(gamma*e),2.))
        return gradient                # gradient of    skLogCosh

    # Forward propagation
    def forward(self, x):
        self.hidden = self.tanh(np.dot(x, self.weights["input_hidden"]) + self.biases["hidden"])
        self.output = np.dot(self.hidden, self.weights["hidden_output"]) + self.biases["output"]
        return self.output

    # Backward propagation
    def backward(self, x, y, lr, momentum, prev_deltas):
        error = self.output - y
        output_grad = self.loss_gradient(error).reshape(-1, self.output_size)
        hidden_grad = np.dot(output_grad, self.weights["hidden_output"].T) * self.tanh_derivative(self.hidden)

        deltas = {
            "hidden_output": momentum * prev_deltas["hidden_output"] - lr * np.dot(self.hidden.T, output_grad),
            "input_hidden": momentum * prev_deltas["input_hidden"] - lr * np.dot(x.T, hidden_grad),
            "output_bias": momentum * prev_deltas["output_bias"] - lr * np.sum(output_grad, axis=0),
            "hidden_bias": momentum * prev_deltas["hidden_bias"] - lr * np.sum(hidden_grad, axis=0),
        }

        # Update weights and biases
        self.weights["hidden_output"] += deltas["hidden_output"]
        self.weights["input_hidden"] += deltas["input_hidden"]
        self.biases["output"] += deltas["output_bias"]
        self.biases["hidden"] += deltas["hidden_bias"]

        return deltas


# Generate synthetic data
def generate_data(samples=1000, x_min=-4, x_max=7, noise=0.2):
    x = np.random.uniform(x_min, x_max, samples)
    y = 1 + 0.4 * np.sin(2 * x) - 0.2 * np.tanh(2 - 2 * x)
    x += np.random.normal(0, noise, x.shape)
    y += np.random.normal(0, noise, y.shape)
    return x.reshape(-1, 1), y.reshape(-1, 1)


# Training and visualization
def train_mlp():
    input_size = 1
    hidden_size = 10
    output_size = 1

    # Initialize MLP
    mlp = MLP(input_size, hidden_size, output_size)

    # Generate training data
    x_train, y_train = generate_data()
    epochs = 500
    lr = 0.01
    momentum = 0.9

    prev_deltas = {
        "hidden_output": np.zeros_like(mlp.weights["hidden_output"]),
        "input_hidden": np.zeros_like(mlp.weights["input_hidden"]),
        "output_bias": np.zeros_like(mlp.biases["output"]),
        "hidden_bias": np.zeros_like(mlp.biases["hidden"]),
    }

    # Track loss over epochs
    losses = []

    for epoch in range(epochs):
        # Forward pass
        y_pred = mlp.forward(x_train)

        # Calculate and save loss
        error = y_pred - y_train
        epoch_loss = np.mean(mlp.loss(error))
        losses.append(epoch_loss)

        # Backward pass
        prev_deltas = mlp.backward(x_train, y_train, lr, momentum, prev_deltas)

        # Print loss every 50 epochs
        if epoch % 50 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

    # Plot loss
    plt.figure()
    plt.plot(range(1, epochs + 1), losses, label="Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss vs. Epochs")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    train_mlp()
