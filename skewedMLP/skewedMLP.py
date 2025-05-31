import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.animation as animation


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
        self.loss = 0.0

    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivative(self, x):
        return 1 - np.tanh(x) ** 2

    def forward(self, x):
        self.hidden = self.tanh(np.dot(x, self.weights["input_hidden"]) + self.biases["hidden"])
        self.output = np.dot(self.hidden, self.weights["hidden_output"]) + self.biases["output"]
        return self.output

    def backward(self, x, y, lr, momentum, prev_deltas):
        error = self.output - y
        loss = SkewedLogCosh.loss(error)
        output_grad = SkewedLogCosh.gradient(error)
        hidden_grad = np.dot(output_grad, self.weights["hidden_output"].T) * self.tanh_derivative(self.hidden)

        deltas = {
            "hidden_output": momentum * prev_deltas["hidden_output"] - lr * np.dot(self.hidden.T, output_grad),
            "input_hidden": momentum * prev_deltas["input_hidden"] - lr * np.dot(x.T, hidden_grad),
            "output_bias": momentum * prev_deltas["output_bias"] - lr * np.sum(output_grad, axis=0),
            "hidden_bias": momentum * prev_deltas["hidden_bias"] - lr * np.sum(hidden_grad, axis=0),
        }

        self.weights["hidden_output"] += deltas["hidden_output"]
        self.weights["input_hidden"] += deltas["input_hidden"]
        self.biases["output"] += deltas["output_bias"]
        self.biases["hidden"] += deltas["hidden_bias"]

        return deltas


class SkewedLogCosh:
    @staticmethod
    def loss(e, m=1, g=2):
        return np.power(m, np.tanh(g * e)) * np.log(np.cosh(e))

    @staticmethod
    def gradient(e, gain=1):
        return gain * np.tanh(e)


def save_model(mlp, filename):
    np.savez(filename, weights=mlp.weights, biases=mlp.biases)


def load_model(filename):
    data = np.load(filename, allow_pickle=True)
    mlp = MLP()
    mlp.weights = data["weights"].item()
    mlp.biases = data["biases"].item()
    return mlp


def generate_data():
    x = np.random.uniform(-4, 7, 1000)
    y = 1 + 0.4 * np.sin(2 * x) - 0.2 * np.tanh(2 - 2 * x)
    noise_x = np.random.normal(0, 0.2, x.shape)
    noise_y = np.random.normal(0, 0.2, y.shape)
    return x + noise_x, y + noise_y, y


def train_and_plot():
    mlp = MLP()
    x_train, y_train, y_true = generate_data()
    x_train = x_train.reshape(-1, 1)
    y_train = y_train.reshape(-1, 1)
    y_true = y_true.reshape(-1, 1)
    n_epochs = 1000
    lr = 0.01
    momentum = 0.05
    prev_deltas = {
        "hidden_output": np.zeros_like(mlp.weights["hidden_output"]),
        "input_hidden": np.zeros_like(mlp.weights["input_hidden"]),
        "output_bias": np.zeros_like(mlp.biases["output"]),
        "hidden_bias": np.zeros_like(mlp.biases["hidden"]),
    }

    fig, ax = plt.subplots()

    def update_plot(epoch):
        ax.clear()  # Clear the previous frame
        ax.set_xlim(-4, 7)
        ax.set_ylim(-1, 2)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"Epoch {epoch}")
        ax.legend(["True Function", "Training Data", "MLP Prediction"], loc="upper right")

        y_pred = np.array([mlp.forward(xi) for xi in x_train])

        ax.plot(x_train.flatten(), y_true.flatten(), 'r-', linewidth=2)  # True function
        ax.scatter(x_train.flatten(), y_train.flatten(), s=10, c='b')   # Training data
        ax.plot(x_train.flatten(), y_pred.flatten(), 'g-', linewidth=2) # MLP prediction

    ani = animation.FuncAnimation(fig, update_plot, frames=n_epochs, interval=50)
    filename_gif = f"MLP@{datetime.now().strftime('%Y%m%d')}.gif"
    filename_jpg = f"MLP@{datetime.now().strftime('%Y%m%d')}.jpg"
    ani.save(filename_gif, writer='imagemagick')
    plt.savefig(filename_jpg)


if __name__ == "__main__":
    train_and_plot()
