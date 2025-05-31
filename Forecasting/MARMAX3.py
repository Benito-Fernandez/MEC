import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

def display(test_to_display):
    print(f"TWC-NARMAX::> {test_to_display}")

# Custom skewed log-cosh loss function and gradient approximation
def LogCosh(e):
    return np.log(np.cosh(e))

def skewedLogCosh(e, m=1, g=2):
    return np.power(m, np.tanh(g * e)) * LogCosh(e)

def compute_gradient(e, gain=1):
    # Approximate gradient: gain * tanh(e)
    return gain * np.tanh(e)

# Load CSV data
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

# Load data
def load_data(file_name, nInputs, nOutputs):
    print("TWC-NARMAX::> Loading data...")
    # Load the CSV file
    data = np.loadtxt(file_name, delimiter=',', skiprows=1)  # Adjust skiprows based on header presence

    # Assuming the first column is time and the last `nOutputs` columns are outputs
    time = data[:, 0]  # Time column (used for sequencing and reference)
    inputs = data[:, 1:nInputs + 1]  # Inputs (Force, Velocity, etc.) - nInputs columns
    outputs = data[:, -nOutputs:]  # Outputs (e.g., Position) - nOutputs columns

    print(f"TWC-NARMAX::> Loaded data, inputs shape: {inputs.shape}, outputs shape: {outputs.shape}")

    return time, inputs, outputs

# Create sequences for training/testing data
def create_sequences(time, inputs, outputs, horizonSteps, nInputs, nOutputs):
    X = []
    y = []
    time_sequences = []

    print("TWC-NARMAX::> Creating sequences...")
    # Generate sequences
    for i in range(horizonSteps, len(inputs) - nOutputs + 1):
        X_batch = inputs[i - horizonSteps:i]  # Sequence of inputs (shape: horizonSteps x nInputs)
        X.append(X_batch)

        # Output corresponding to the sequence (shape: nOutputs)
        y.append(outputs[i + nOutputs - 1])  # Predict the output after the sequence of `horizonSteps`

        # Add corresponding times for the sequence
        time_sequences.append(time[i - horizonSteps:i + nOutputs])  # Time sequence for each input-output pair

    X = np.array(X)
    y = np.array(y)
    time_sequences = np.array(time_sequences)

    print(f"TWC-NARMAX::> Created sequences, X shape: {X.shape}, y shape: {y.shape}, time_sequences shape: {time_sequences.shape}")

    return X, y, time_sequences

# Build RNN model
def build_rnn_model(horizonSteps, nInputs, nOutputs):
    print("TWC-NARMAX::> Building RNN model...")
    model = Sequential()
    model.add(LSTM(64, activation='tanh', input_shape=(horizonSteps, nInputs)))
    model.add(Dense(nOutputs))

    model.compile(optimizer=Adam(), loss='mean_squared_error')
    print("TWC-NARMAX::> Model built successfully.")
    return model

# Custom loss function with skewedLogCosh
def skewedLogCosh(e, m=1, g=2):
    # Calculate skewed log cosh loss
    return np.power(m, np.tanh(g * e)) * np.log(np.cosh(e))

# Custom training step with modified gradient calculation
def custom_training_step(model, X_batch, y_batch, m=1, g=2, gain=1):
    with tf.GradientTape() as tape:
        print(f"@ custom_training_step: {X_batch.shape}\n ... calling model")
        # input_shape=(horizonSteps, nInputs)
        y_pred = model(X_batch, training=True)
        print(f"@ custom_training_step: model called. \ny_pred.shape = {y_pred.shape}")

        # Compute the error
        error = y_pred - y_batch

        # Calculate skewedLogCosh loss
        loss = tf.reduce_mean(skewedLogCosh(error.numpy(), m, g))

    print(f"loss, {loss}, {model.trainable_variables}")
    for var in model.trainable_variables:
        print(f"{var.shape} <- var.")
    gradients = tape.gradient(loss, model.trainable_variables)
    print(f"gradients = {gradients}")
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss

# Training function
def train_rnn_model(model, X_train, y_train, horizonSteps, nEpochs, m=1, g=2, gain=1):
    print("TWC-NARMAX::> Training the model...")
    print(f"X_train.shape = {X_train.shape}")
    for epoch in range(nEpochs):
        epoch_loss = 0
        for i in range(len(X_train)):
##            X_batch = X_train[i:i+1,:]
##            y_batch = y_train[i:i+1,:]
            X_batch = X_train[i:i+horizonSteps,:]
            y_batch = y_train[i:i+horizonSteps,:]
            print(f"X_batch.shape = {X_batch.shape}")
            print(f"y_batch.shape = {y_batch.shape}")

            # Reshape X_batch to be (1, horizonSteps, nInputs)
##            X_batch = np.reshape(X_batch, (1, horizonSteps, X_batch.shape[1]))
            X_batch = np.reshape(X_batch, (horizonSteps, X_batch.shape[1]))
            print(f"X_batch.reshaped = {X_batch.shape}")

            # Perform custom training step
            loss = custom_training_step(model, X_batch, y_batch, m, g, gain)
            print("loss[{epoch},{i}] = {loss}")
            epoch_loss += loss

        print(f"TWC-NARMAX::> Epoch {epoch+1}/{nEpochs}, Loss: {epoch_loss.numpy()}")
    print("TWC-NARMAX::> Training complete.")

# Test and plot results
def test_rnn_model(test_data_file, model, horizonSteps, nInputs, nOutputs):
    display("Testing the model...")

    # Load test data, including time, inputs, and outputs
    time, test_inputs, test_outputs = load_data(test_data_file, nInputs, nOutputs)

    # Create sequences for testing data
    X_test, y_test, time_sequences = create_sequences(time, test_inputs, test_outputs, horizonSteps, nInputs, nOutputs)

    # Load model weights
    model.load_weights('narmax_rnn_model_weights.h5')

    predictions = []
    for i in range(len(X_test)):
        X_batch = X_test[i:i+1]

        # Reshape X_batch to be (1, horizonSteps, nInputs)
        X_batch = np.reshape(X_batch, (1, horizonSteps, X_batch.shape[1]))

        y_pred = model(X_batch, training=False)
        predictions.append(y_pred.numpy().flatten())

    # Plotting the results: forecast vs actual with time on the x-axis
    predictions = np.array(predictions)

    display(" Plotting forecast vs actual for test data...")
    plt.figure(figsize=(10, 6))
    for i in range(nOutputs):
        plt.plot(time_sequences[:, -nOutputs + i], test_outputs[:len(predictions), i], label=f"Actual Output {i+1}")
        plt.plot(time_sequences[:, -nOutputs + i], predictions[:, i], label=f"Forecasted Output {i+1}", linestyle='--')

    plt.legend()
    plt.xlabel('Time Steps')
    plt.ylabel('Output Values')
    plt.title(f'Forecasted vs Actual Outputs (Test Data) - {nOutputs} Output(s)')
    plt.grid(True)
    plt.show()  # Ensure this is called to display the plot

    # Compute cost
    total_cost = np.sum(np.abs(predictions - test_outputs[:len(predictions)]))
    display(f"Total cost on test data: {total_cost}")

# Main function
def main(training_data_file, test_data_file, nInputs, nOutputs, dDelayInputs,
         nDelayOutputs, nDelayNoise, horizonSteps, nEpochs):
    display("Starting NARMAX model training...")

    # Load training data
    time, train_inputs, train_outputs = load_data(training_data_file, nInputs, nOutputs)

    # Create training sequences
    X_train, y_train, _ = create_sequences(time, train_inputs, train_outputs, horizonSteps, nInputs, nOutputs)

    # Build and train the model
    model = build_rnn_model(horizonSteps, nInputs, nOutputs)
    train_rnn_model(model, X_train, y_train, horizonSteps, nEpochs)

    # Test the model and plot results
    test_rnn_model(test_data_file, model, horizonSteps, nInputs, nOutputs)

    display("NARMAX model training and testing complete.")

# Example usage
training_data_file = 'training_data.csv'  # Your training data CSV file
test_data_file     = 'test_data.csv'  # Your test data CSV file
nInputs            = 1  # For example, 1 input (force)
nOutputs           = 1  # For example, 1 output (position)
dDelayInputs       = 10
nDelayOutputs      = 5
nDelayNoise        = 5
horizonSteps       = 10
nEpochs            = 10

main(training_data_file, test_data_file, nInputs, nOutputs, dDelayInputs, nDelayOutputs, nDelayNoise, horizonSteps, nEpochs)
