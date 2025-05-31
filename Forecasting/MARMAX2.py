import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Custom skewed log-cosh loss function and gradient approximation
def LogCosh(e):
    return np.log(np.cosh(e))

def skewedLogCosh(e, m=1, g=2):
    return np.power(m, np.tanh(g * e)) * LogCosh(e)

def compute_gradient(e, gain=1):
    # Approximate gradient: gain * tanh(e)
    return gain * np.tanh(e)

# Load CSV data
def load_data(file_name, nInputs, nOutputs):
    # Load the CSV file
    data = np.loadtxt(file_name, delimiter=',', skiprows=1)

    # Extract columns
    time = data[:, 0]  # Time column
    force = data[:, 1:nInputs]  # Force (input) column
    position = data[:, nInputs+1:]  # Position (output) column

    # Assuming the first column is time,
    # the next nInputs  colums are inputs and
    # the last nOutputs columns are outputs
    time = data[:, 0]  # Time column (used for sequencing, not a feature)
    inputs = data[:, 1:nInputs + 1]  # Inputs (Force, Velocity, etc.) - nInputs columns
    outputs = data[:, -nOutputs:]  # Outputs (e.g., Position) - nOutputs columns

    print(f"Loaded data, inputs shape: {inputs.shape}, outputs shape: {outputs.shape}")

    return time, inputs, outputs

#def create_sequences(train_inputs, train_outputs, nInputs, nOutputs, nDelayInputs, nDelayOutputs, nDelayNoise, horizonSteps)
def create_sequences(time, inputs, outputs, horizonSteps, nInputs, nOutputs):
    X = []
    Y = []
    time_sequences = []

    # Generate sequences
    for i in range(horizonSteps, len(inputs) - nOutputs + 1):
        X_batch = inputs[i - horizonSteps:i]  # Sequence of inputs (shape: horizonSteps x nInputs)
        X.append(X_batch)

        # Output corresponding to the sequence (shape: nOutputs)
        Y.append(outputs[i + nOutputs - 1])  # Predict the output after the sequence of `horizonSteps`

        # Add corresponding times for the sequence
        time_sequences.append(time[i - horizonSteps:i + nOutputs])  # Time sequence for each input-output pair

    X = np.array(X)
    Y = np.array(Y)
    time_sequences = np.array(time_sequences)

    print(f"Created sequences, X shape: {X.shape}, Y shape: {Y.shape}, time_sequences shape: {time_sequences.shape}")

    return X, Y, time_sequences

# Define the RNN model
def create_rnn_model(nInputs, nOutputs, nDelayInputs, nDelayOutputs, nDelayNoise, horizonSteps):
    model = models.Sequential()
    model.add(layers.LSTM(64, input_shape=(horizonSteps, nInputs + nOutputs), return_sequences=True))
    model.add(layers.LSTM(64))
    model.add(layers.Dense(nOutputs))
    model.compile(optimizer='adam', loss='mse')  # We'll override the loss function with our custom one
    return model

# Custom training loop with skewedLogCosh loss and gradient approximation
def custom_loss(y_true, y_pred, m=1, g=2, gain=1):
    e = y_true - y_pred
    loss = skewedLogCosh(e, m, g)
    return np.mean(loss)

def custom_training_step(model, X_batch, y_batch, m=1, g=2, gain=1):
    # Perform forward pass
    with tf.GradientTape() as tape:
        y_pred = model(X_batch, training=True)
        loss = custom_loss(y_batch, y_pred, m, g, gain)

    # Calculate the gradients
    gradients = tape.gradient(loss, model.trainable_variables)

    # Apply custom gradient approximation (using the computed loss)
    adjusted_gradients = []
    for grad in gradients:
        # Assuming the gradient approximation involves multiplying by gain * tanh(e)
        adjusted_gradients.append(grad * gain * np.tanh(loss))

    # Apply gradients
    model.optimizer.apply_gradients(zip(adjusted_gradients, model.trainable_variables))

    return loss

# Train the model with the custom training loop
def train_rnn_model(model, X_train, y_train, nEpochs, horizonSteps, m=1, g=2, gain=1):
    loss_per_epoch = []
    print(f"Shape of X_train: {X_train.shape}")
    print(f"Shape of y_train: {y_train.shape}")
    for epoch in range(nEpochs):
        epoch_loss = 0
        for i in range(len(X_train)):
            X_batch = X_train[i:i+1]
            y_batch = y_train[i:i+1]
            print(f"Shape of X_batch: {X_batch.shape}")
            print(f"Shape of y_batch: {y_batch.shape}")
            print("horizonSteps, X_batch.shape[1] = ",horizonSteps, X_batch.shape[1])

            # Reshape X_batch to be (1, horizonSteps, nInputs + nOutputs)
            X_batch = np.reshape(X_batch, (1, horizonSteps, X_batch.shape[1]))

            loss = custom_training_step(model, X_batch, y_batch, m, g, gain)
            epoch_loss += loss
        avg_epoch_loss = epoch_loss / len(X_train)
        loss_per_epoch.append(avg_epoch_loss)
        print(f"Epoch {epoch+1}/{nEpochs}, Loss: {avg_epoch_loss}")

    # Save model weights
    model.save_weights('narmax_rnn_model_weights.h5')

    # Plot training loss over epochs
    print("TWC-NARMAX::> Plotting training loss...")
    plt.figure(figsize=(10, 6))
    plt.plot(range(nEpochs), loss_per_epoch, label="Training Loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.show()

# Run the model with test data and forecast
def test_rnn_model(test_data_file, model, horizonSteps, nInputs, nOutputs):
    # Load test data, including time, inputs, and outputs
    time, test_inputs, test_outputs = load_data(test_data_file, nInputs, nOutputs)
    print(f"TWC-NARMAX::> Test data loaded, inputs shape: {test_inputs.shape}, outputs shape: {test_outputs.shape}")

    # Create sequences for testing data
    X_test, y_test, time_sequences = create_sequences(time, test_inputs, test_outputs, horizonSteps, nInputs, nOutputs)
    print(f"TWC-NARMAX::> Test sequences created, X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

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

    print("TWC-NARMAX::> Plotting forecast vs actual for test data...")
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
    print(f"TWC-NARMAX::> Total cost on test data: {total_cost}")


# Main function
def main(training_data_file, test_data_file, nInputs, nOutputs, \
         nDelayInputs, nDelayOutputs, nDelayNoise, horizonSteps, nEpochs):
    # Load training data
    time, train_inputs, train_outputs = load_data(training_data_file, nInputs, nOutputs)
    print(f"TWC-NARMAX::> Training data loaded, inputs shape: {train_inputs.shape}, outputs shape: {train_outputs.shape}")

    # Prepare training sequences
    # (time, inputs, outputs, horizonSteps, nInputs, nOutputs)
    X_train, y_train = create_sequences(train_inputs, train_outputs, nInputs, nOutputs, nDelayInputs, nDelayOutputs, nDelayNoise, horizonSteps)
    print(f"TWC-NARMAX::> Training sequences created, X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

    # Create and train the model
    model = create_rnn_model(nInputs, nOutputs, nDelayInputs, nDelayOutputs, nDelayNoise, horizonSteps)
    train_rnn_model(model, X_train, y_train, nEpochs, horizonSteps)

    # Test the model on the test data
    test_rnn_model(test_data_file, model, horizonSteps, nInputs, nOutputs)

# Example usage
if __name__ == "__main__":
    training_data_file = 'training_data.csv'  # Input your training CSV file path
    test_data_file = 'test_data.csv'  # Input your test CSV file path
    nInputs = 3  # Number of input features
    nOutputs = 2  # Number of output features
    nDelayInputs = 5  # Delay in inputs
    nDelayOutputs = 3  # Delay in outputs
    nDelayNoise = 2  # Delay in noise
    horizonSteps = 10  # Forecasting horizon
    nEpochs = 50  # Number of training epochs

    main(training_data_file, test_data_file, nInputs, nOutputs, nDelayInputs, nDelayOutputs, nDelayNoise, horizonSteps, nEpochs)
