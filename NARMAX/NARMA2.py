#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      benit
#
# Created:     05/03/2025
# Copyright:   (c) benit 2025
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

class NARMA:
    def __init__(self, n_hidden, n_x, n_u, dt=0.1, t_now=10.0, t_horizon=2.0):
        self.n_hidden = n_hidden
        self.n_x = n_x
        self.n_u = n_u
        self.dt = dt
        self.t_now = t_now
        self.t_horizon = t_horizon

        self.model = self.create_model()

    def create_model(self):
        model = Sequential([
            Dense(self.n_hidden, activation='relu', input_shape=(self.n_x + self.n_u,)),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def train(self, x):
        X, Y = self.create_lagged_data(x)
        self.model.fit(X, Y, epochs=100, verbose=1)

    def create_lagged_data(self, x):
        N = len(x)
        X = []
        Y = []
        for i in range(self.n_x + self.n_u, N):
            x_past = x[i - self.n_x:i].reshape(-1)
            u_past = np.random.normal(scale=0.1, size=self.n_u)
            X.append(np.concatenate([x_past, u_past]))
            Y.append(x[i])
        return np.array(X), np.array(Y)

    def predict_future(self, x):
        t_future = np.arange(self.t_now, self.t_now + self.t_horizon, self.dt)
        x_pred = list(x[-self.n_x:])  # Start with the last n_x values of x

        for _ in t_future:
            x_past = np.array(x_pred[-self.n_x:]).reshape(1, -1)
            u_past = np.random.normal(scale=0.1, size=(1, self.n_u))
            X_input = np.concatenate([x_past, u_past], axis=1)
            x_next = self.model.predict(X_input)
            x_pred.append(x_next[0, 0])

        return x_pred

    def save_model(self, filename):
        with open(filename, 'w') as file:
            file.write(f"n_hidden: {self.n_hidden}\n")
            file.write(f"n_x: {self.n_x}\n")
            file.write(f"n_u: {self.n_u}\n")
            for layer in self.model.layers:
                weights, biases = layer.get_weights()
                file.write(f"Weights: {weights.tolist()}\n")
                file.write(f"Biases: {biases.tolist()}\n")

def main():

    # Example usage
    n_hidden = 50
    n_x = 5
    n_u = 5

    # Create and train the NARMA model
    narma = NARMA(n_hidden, n_x, n_u)
    t = np.arange(0, narma.t_now, narma.dt)
    x = np.sin(t) + np.random.normal(scale=0.1, size=len(t))

    narma.train(x)

    # Predict future values
    x_pred = narma.predict_future(x)

    # Save the model topology and weights to a file
    narma.save_model("NARMA.net")

    # Plot the results
    import matplotlib.pyplot as plt

    plt.plot(t, x, label='Original')
    plt.plot(np.arange(narma.t_now, narma.t_now + narma.t_horizon, narma.dt), x_pred[n_x:], label='Predicted')
    plt.xlabel('Time')
    plt.ylabel('x(t)')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
