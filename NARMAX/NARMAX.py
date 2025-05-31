import numpy as np

class NARMA:
    def __init__(self, n_hidden, n_x, n_u, dt=0.1, t_now=10.0, t_horizon=2.0, lr=0.01):
        self.n_hidden = n_hidden
        self.n_x = n_x
        self.n_u = n_u
        self.dt = dt
        self.t_now = t_now
        self.t_horizon = t_horizon
        self.lr = lr

        self.weights_input_hidden = np.random.randn(n_hidden, n_x + n_u)
        self.bias_hidden = np.zeros(n_hidden)
        self.weights_hidden_output = np.random.randn(n_hidden)
        self.bias_output = 0.0

    def tanh(self, z):
        return np.tanh(z)

    def tanh_derivative(self, z):
        return 1 - np.tanh(z)**2

    def forward(self, x_input, u_input):
        z_hidden = np.dot(self.weights_input_hidden, np.concatenate((x_input, u_input))) + self.bias_hidden
        a_hidden = self.tanh(z_hidden)
        output = np.dot(self.weights_hidden_output, a_hidden) + self.bias_output
        return output, a_hidden

    def train(self, x):
        X, Y = self.create_lagged_data(x)
        for epoch in range(1000):
            total_loss = 0.0
            for x_input, y_true in zip(X, Y):
                u_input = np.random.normal(scale=0.1, size=self.n_u)
                y_pred, a_hidden = self.forward(x_input, u_input)
                loss = (y_pred - y_true)**2
                total_loss += loss

                # Backpropagation
                dL_dy_pred = 2 * (y_pred - y_true)
                dL_dweights_hidden_output = dL_dy_pred * a_hidden
                dL_dbias_output = dL_dy_pred
                dL_da_hidden = dL_dy_pred * self.weights_hidden_output
                dL_dz_hidden = dL_da_hidden * self.tanh_derivative(a_hidden)

                self.weights_hidden_output -= self.lr * dL_dweights_hidden_output
                self.bias_output -= self.lr * dL_dbias_output
                self.weights_input_hidden -= self.lr * np.outer(dL_dz_hidden, np.concatenate((x_input, u_input)))
                self.bias_hidden -= self.lr * dL_dz_hidden

            print(f"Epoch {epoch+1}, Loss: {total_loss}")

    def create_lagged_data(self, x):
        N = len(x)
        X = []
        Y = []
        for i in range(self.n_x + self.n_u, N):
            x_past = x[i - self.n_x:i]
            X.append(x_past)
            Y.append(x[i])
        return np.array(X), np.array(Y)

    def predict_future(self, x):
        t_future = np.arange(self.t_now, self.t_now + self.t_horizon, self.dt)
        x_pred = list(x[-self.n_x:])  # Start with the last n_x values of x

        for _ in t_future:
            x_past = np.array(x_pred[-self.n_x:])
            u_past = np.random.normal(scale=0.1, size=self.n_u)
            x_next, _ = self.forward(x_past, u_past)
            x_pred.append(x_next)

        return x_pred

    def save_model(self, filename):
        with open(filename, 'w') as file:
            file.write(f"n_hidden: {self.n_hidden}\n")
            file.write(f"n_x: {self.n_x}\n")
            file.write(f"n_u: {self.n_u}\n")
            file.write(f"Weights Input to Hidden: {self.weights_input_hidden.tolist()}\n")
            file.write(f"Biases Hidden: {self.bias_hidden.tolist()}\n")
            file.write(f"Weights Hidden to Output: {self.weights_hidden_output.tolist()}\n")
            file.write(f"Bias Output: {self.bias_output}\n")

# Example usage
n_hidden = 50
n_x = 5
n_u = 5

# Create and train the NARMA model
narma = NARMA(n_hidden, n_x, n_u)
t = np.arange(0, narma.t_now, narma.dt)
x = np.sin(t) + np.random.normal(scale=0.1, size=len(t))

narma.train(x)

# Save the model topology and weights to a file
narma.save_model("NARMA.net")

# Save the model topology and weights to a file
#narma.load_model("NARMA.net")

# Predict future values
x_pred = narma.predict_future(x)

# Plot the results
import matplotlib.pyplot as plt

plt.plot(t, x, label='Original')
plt.plot(np.arange(narma.t_now, narma.t_now + narma.t_horizon, narma.dt), x_pred[n_x:], label='Predicted')
plt.xlabel('Time')
plt.ylabel('x(t)')
plt.legend()
plt.show()
