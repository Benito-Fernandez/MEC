import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import log, cosh, tanh
import random

# Nonlinear spring and damper functions
def damper(p):  # Damper force f_d(p = momentum)
    return 0.1 * p + 0.2 * log(cosh(p)) * tanh(p)

def spring(x):  # Spring force f_s(x = position)
    return x * (0.2 + 0.05 * abs(x))

# Data Generation for Mass-Spring-Dashpot System
def generate_mass_spring_data(mass=1.0, T=0.1, total_time=50, history=5, force_range=(-5, 5), force_duration=(1, 5)):
    time = np.arange(0, total_time, T)
    n_points = len(time)

    Fa = np.zeros(n_points)
    x = np.zeros(n_points)
    p = np.zeros(n_points)

    # Generate random force signals Fa
    current_time = 0
    while current_time < total_time:
        force_value = np.random.uniform(force_range[0], force_range[1])
        force_duration_range = np.random.randint(force_duration[0], force_duration[1] + 1)

        force_time_indices = np.arange(current_time, min(current_time + force_duration_range, total_time), 1)
        Fa[force_time_indices] = force_value
        current_time += force_duration_range

    # Integrate system with force Fa
    for t in range(1, n_points):
        spring_force = spring(x[t-1])
        damper_force = damper(p[t-1])

        # Net force = External force - spring_force - damper_force
        F_net = Fa[t-1] - spring_force - damper_force

        # Update momentum (p) and position (x)
        p[t] = p[t-1] + T * F_net / mass
        x[t] = x[t-1] + T * p[t] / mass

    # Create DataFrame for inputs (Fa, x, p)
    inputs_df = pd.DataFrame({
        'time': time,
        'Fa': Fa,
        'x': x,
        'p': p
    })

    # Create DataFrame for true data (x, p)
    true_data_df = pd.DataFrame({
        'time': time,
        'x': x,
        'p': p
    })

    # Save to CSV files
    inputs_df.to_csv('inputs.csv', index=False)
    true_data_df.to_csv('true_data.csv', index=False)

    return inputs_df, true_data_df

# Forecasting Neural Network Class.  Make a derivative of generic NNet (DyNet)
class ForecastNet:
    def __init__(self, LayersSizes, activation_function):
        self.LayersSizes = LayersSizes
        self.activation_function = activation_function
        self.nLayers = len(LayersSizes)

        # Initialize weights and biases with proper dimensions
        self.Weights = [np.random.randn(self.LayersSizes[i], self.LayersSizes[i+1]) for i in range(self.nLayers - 1)]
        self.Biases  = [np.random.randn(1, self.LayersSizes[i+1]) for i in range(self.nLayers - 1)]
        self.xi      = [None] * self.nLayers
        self.z       = [None] * self.nLayers

    def forwardPropagation(self, U=None, horizon=1):
        """
        Evaluate (propagate forward) MLP NNet states.
        Arguments:
            U: Inputs, should be a DataFrame with columns Fa, x, p
            horizon: Number of steps (future horizon)
        """
        if U.empty and self.LayersSizes[0] != 0:
            raise ValueError('BRFNet:: Need Input Data to propagate forward!')

        nPoints, nInputs = U.shape

        for layer in range(self.nLayers-1):
            if layer == 0:
                # For the first layer, use U directly
                self.xi[layer] = np.dot(U, self.Weights[layer]) + np.repeat(self.Biases[layer], nPoints, axis=0)
                self.z[layer] = self.activation_function[layer](self.xi[layer])
            else:
                # For subsequent layers, use output from previous layer (z[layer-1])
                self.xi[layer] = np.dot(self.z[layer-1], self.Weights[layer]) + np.repeat(self.Biases[layer], nPoints, axis=0)
                self.z[layer] = self.activation_function[layer](self.xi[layer])

        Yout = pd.DataFrame(self.z[self.nLayers - 1], columns=['x_dot', 'p_dot'], index=U.index)
        return Yout

    def forecast(self, inputsDF, true_data, T, horizon=1, history=5, alpha=0.01):
        """
        Forecast the future values of x and p using the trained neural network.

        Arguments:
            inputsDF: DataFrame containing the input features (Fa, x, p)
            true_data: DataFrame containing the true values of x and p
            T: Time step (delta time between samples)
            horizon: Number of future steps to forecast
            history: Number of past time steps to use as input history
            alpha: Weighting factor for the momentum error in the cost function

        Returns:
            cost_list: List of cost function values for each forecasted point
            all_forecasts: List of all forecasted x and p values
        """
        nPoints       = len(inputsDF)
        cost_list     = []
        all_forecasts = []

        # Iterate over the data, starting from 'history' to (nPoints - horizon)
        for k in range(history, nPoints - horizon):
            # Prepare the input for the network (current time step)
            input_now = pd.DataFrame({
                'Fa': [inputsDF.loc[k, 'Fa']],
                'x' : [true_data.loc[k, 'x']],
                'p' : [true_data.loc[k, 'p']]
            }, index=[0])
            print(f"t={T*k}, (Fa, x, p)={input_now.values}")

            # Forward propagation to predict x_dot and p_dot
            output_now = self.forwardPropagation(U=input_now, horizon=1)

            # Perform integration for the forecast
            x_hat = true_data.loc[k, 'x']
            p_hat = true_data.loc[k, 'p']

            # Forecast x and p for the specified horizon
            x_forecast = [x_hat]
            p_forecast = [p_hat]
            for i in range(horizon):
                x_hat = x_forecast[i] + T * output_now.loc[0, 'x_dot']
                p_hat = p_forecast[i] + T * output_now.loc[0, 'p_dot']
                x_forecast.append(x_hat)
                p_forecast.append(p_hat)
                print(f"x[{T*k}+{T*i}] = {x_hat}, p[{T*k}+{T*i}] = {p_hat}")

            # Calculate the error and cost function
            e = (true_data.loc[k+1:k+horizon, 'x'].values - np.array(x_forecast[1:])) \
                + alpha * (true_data.loc[k+1:k+horizon, 'p'].values - np.array(p_forecast[1:]))
            cost = np.sum(np.log(np.cosh(e)))
            cost_list.append(cost)

            # Store the forecasted values
            all_forecasts.append((x_forecast, p_forecast))

        return cost_list, all_forecasts

#-------------------------------------------------------------------------------
# Create the neural network
LayersSizes = [3, 15, 2]  # Input layer (3 features), hidden layer (15 neurons), output layer (2 values: x_dot, p_dot)
activation_function = [np.tanh, np.tanh]  # Using tanh activation function for both layers

net = ForecastNet(LayersSizes, activation_function)

# Generate Data
T_time     = 50       # Total time
delta_t    = 0.1      # delta time step
MSD_mass   = 1.0      # System's mass
history    = 5        # History length
F_range    = (-5, 5)  # Range of input force
F_duration = (1, 5)   # Force Duration limits
inputsDF, true_data = generate_mass_spring_data(mass           = MSD_mass,
                                                T              = delta_t,
                                                total_time     = T_time,
                                                history        = 5,
                                                force_range    = F_range,
                                                force_duration = F_duration)
#-------------------------
print(true_data)
# Forecasting
T       = delta_t # Time step
horizon = 5       # Forecast horizon
cost_list, all_forecasts = net.forecast(inputsDF, true_data, T, horizon)
#-------------------------
print(all_forecasts)

# Plot the forecasted vs true values
plt.figure(figsize=(12, 6))
plt.plot(true_data['time'], true_data['x'], label='True Position (x)', color='blue')
plt.plot(true_data['time'], true_data['p'], label='True Momentum (p)', color='green')

for i in range(horizon):
    x_forecast = [forecast[0][i] for forecast in all_forecasts]
    plt.plot(true_data['time'][history:-horizon], x_forecast, label=f'Forecasted Position (x) Horizon {i+1}', linestyle='--')

plt.legend()
plt.xlabel('Time [s]')
plt.ylabel('Value')
plt.title(f'Mass-Spring-Dashpot Forecasting with Horizon={horizon}')
plt.show()
