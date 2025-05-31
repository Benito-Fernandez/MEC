import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.animation as animation
from   scipy.stats   import logistic, hypsecant

class MLP:
    def __init__(self, p=1, m=1, n=10):
        self.p = p
        self.m = m
        self.n = n
        self.weights = {
            "input_hidden": np.random.randn(p, n),
            "hidden_output": np.random.randn(n, m),
        }
        self.biases = {
            "hidden": np.random.randn(n),
            "output": np.random.randn(m),
        }

    def tanh(self, x):
        return np.tanh(x)

    def tanh_(self, x):  # tanh derivative
        return 1 - np.tanh(x) ** 2

    def loss(self, e, gain=1, gamma=2):  # Skewed LogCosh Loss function
        return np.power(gain, np.tanh(gamma * e)) * np.log(np.cosh(e))

    def loss_(self, e, gain=1, gamma=2):  # Skewed LogCosh Loss function gradient (approx)
        return np.power(gain,np.tanh(gamma*e)) * np.tanh(e)

    def forward(self, x):
        # Calculate hidden layer activations
        self.hidden = self.tanh(np.dot(x, self.weights["input_hidden"]) + self.biases["hidden"])
        # Ensure hidden layer is shaped (batch_size, n)
        if self.hidden.ndim == 1:
            self.hidden = self.hidden.reshape(-1, self.n)

        # Calculate output layer activations
        self.output = np.dot(self.hidden, self.weights["hidden_output"]) + self.biases["output"]
        # Ensure output is shaped (batch_size, m)
        if self.output.ndim == 1:
            self.output = self.output.reshape(-1, self.m)

        return self.output

    def backward(self, x, y, lr, momentum, prev_deltas):
        error = self.output - y
        # Reshape output_grad to (batch_size, m)
        output_grad = self.loss_(error).reshape(-1, self.m)

        # Ensure self.hidden is reshaped correctly
        if self.hidden.ndim == 1:
            self.hidden = self.hidden.reshape(-1, self.n)

        hidden_grad = np.dot(output_grad, self.weights["hidden_output"].T) * self.tanh_(self.hidden)

        # Debugging: Ensure alignment of shapes
        print("Shape of self.hidden.T:", self.hidden.T.shape)  # (n, batch_size)
        print("Shape of output_grad:", output_grad.shape)      # (batch_size, m)

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

def generate_data(x_min=-4, x_max=7, nTrain=1000, noise=0.2):
    x = np.random.uniform(x_min, x_max, nTrain)
    y = 1 + 0.4 * np.sin(2 * x) - 0.2 * np.tanh(2 - 2 * x)
    noise_x = np.random.normal(0, noise, x.shape)
    noise_y = np.random.normal(0, noise, y.shape)
    return x + noise_x, y + noise_y, x, y


def train_and_plot(mlp=MLP()):
    # Default network is MLP
    x_train, y_train, x_true, y_true = generate_data()
    x_train = x_train.reshape(-1, 1)
    y_train = y_train.reshape(-1, 1)
    x_true  = x_true.reshape(-1, 1)
    y_true  = y_true.reshape(-1, 1)
    n_epochs = 1000
    lr = 0.01
    momentum = 0.05
    prev_deltas = {
        "hidden_output": np.zeros_like(mlp.weights["hidden_output"]),
        "input_hidden": np.zeros_like(mlp.weights["input_hidden"]),
        "output_bias": np.zeros_like(mlp.biases["output"]),
        "hidden_bias": np.zeros_like(mlp.biases["hidden"]),
    }

    losses = []  # List to store loss values

    fig, ax = plt.subplots()

    for epoch in range(1, n_epochs + 1):
        # Perform forward and backward passes
        y_pred = np.array([mlp.forward(xi) for xi in x_train])
        error = mlp.output - y_train
        print("Shape of error:", error.shape)
        current_loss = np.mean(mlp.loss(error))
        print(f"Loss[{epoch}] = {current_loss}.Shape of current_loss:", current_loss.shape)
        losses.append(current_loss)
        mlp.backward(x_train, y_train, lr, momentum, prev_deltas)

        # Update the plot in real-time
        ax.clear()
        ax.set_xlim(-4, 7)
        ax.set_ylim(-1, 2)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"Epoch {epoch}")
        ax.plot(x_true.flatten(), y_true.flatten(), 'r.', linewidth=2)  # True function
        ax.scatter(x_train.flatten(), y_train.flatten(), s=10, c='b')   # Training data
        ax.plot(x_train.flatten(), y_pred.flatten(), 'g.', linewidth=2) # MLP prediction

        plt.pause(0.001)  # Render updates

    # Finalize the animation by saving as MP4
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename_mp4 = f"MLP@{timestamp}.mp4"
    ani = animation.FuncAnimation(fig, lambda _: None, frames=[0])  # Dummy to save the last frame
    ani.save(filename_mp4, writer='ffmpeg')

    # Plot the loss vs. epochs
    plt.figure()
    plt.plot(range(1, n_epochs + 1), losses, label="Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss vs. Epochs")
    plt.legend()
    plt.savefig(f"Loss_Plot@{timestamp}.png")
    plt.show()


if __name__ == "__main__":
    train_and_plot()
