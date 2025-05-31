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
def load_data(filename, nInputs, nOutputs):
    print("TWC-NARMAX:-> running 'load_data'")
    data    = pd.read_csv(filename)
    time    = data.iloc[:,0]
    inputs  = data.iloc[:, 1:nInputs+1].values
    outputs = data.iloc[:, nInputs+1:nInputs+1+nOutputs].values
    return time, inputs, outputs

# Prepare input-output sequences for training
def create_sequences(inputs, outputs, nInputs, nOutputs, dDelayInputs, nDelayOutputs, nDelayNoise, horizonSteps):
    print("TWC-NARMAX:-> running 'create_sequences'")
    X = []
    y = []
    for i in range(len(inputs) - horizonSteps):
        X_seq = np.hstack([
            inputs[i:i+horizonSteps, :],  # delayed inputs
            outputs[i:i+horizonSteps, :]  # delayed outputs
        ])
        y_seq = outputs[i+horizonSteps, :]
        X.append(X_seq)
        y.append(y_seq)
    return np.array(X), np.array(y)

# Define the RNN model
def create_rnn_model(nInputs, nOutputs, dDelayInputs, nDelayOutputs, nDelayNoise, horizonSteps):
    print("TWC-NARMAX:-> running 'create_rnn_model'")
    model = models.Sequential()
    model.add(layers.LSTM(64, input_shape=(horizonSteps, nInputs + nOutputs), return_sequences=True))
    model.add(layers.LSTM(64))
    model.add(layers.Dense(nOutputs))

##    model = models.Sequential(
##        [
##            layers.LSTM(64, input_shape=(horizonSteps, nInputs + nOutputs),
##                        return_sequences=True, name="input_layer")
##            layers.LSTM(64, activation="tanh", name="hidden_layer"),
##            layers.Dense(nOutputs, name="output_layer"),
##        ]
##    )
    model.compile(optimizer='adam', loss='mse')  # We'll override the loss function with our custom one
    print(f"TWC-NARMAX:-> model created with {len(model.layers)} layers")
    return model

# Custom training loop with skewedLogCosh loss and gradient approximation
def custom_loss(y_true, y_pred, m=1, g=2, gain=1):
    e = y_true - y_pred
    loss = skewedLogCosh(e, m, g)
    return np.mean(loss)

def custom_training_step(model, X_batch, y_batch, m=1, g=2, gain=1):
    print("TWC-NARMAX:-> running 'custom_training_step'")
    # Perform forward pass
    with tf.GradientTape() as tape:
        print(f"@ custom_training_step:: X_batch.shape: {X_batch.shape}, y_batch.shape:{y_batch.shape}")
        print("TWC-NARMAX:-> next: 'y_pred = model(X_batch, training=True)'")
        y_pred = model(X_batch, training=True)
        loss = custom_loss(y_batch, y_pred, m, g, gain)
        print(loss, y_pred.shape)
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
    print("TWC-NARMAX:-> running 'train_rnn_model'")
    loss_per_epoch = []
    for epoch in range(nEpochs):
        epoch_loss = 0
        for i in range(len(X_train)):
            X_batch = X_train[i:i+1]
            y_batch = y_train[i:i+1]
            # Reshape X_batch to be (1, horizonSteps, nInputs + nOutputs)
            X_batch = np.reshape(X_batch, (1, horizonSteps, X_batch.shape[2]))
            print(X_train.shape, X_batch.shape, y_train.shape, y_batch.shape, \
                  m, g, gain, model)

            loss = custom_training_step(model, X_batch, y_batch, m, g, gain)
            epoch_loss += loss
        avg_epoch_loss = epoch_loss / len(X_train)
        loss_per_epoch.append(avg_epoch_loss)
        print(f"Epoch {epoch+1}/{nEpochs}, Loss: {avg_epoch_loss}")

    # Save model weights
    model.save_weights('narmax_rnn_model_weights.h5')

    # Plot training loss over epochs
    plt.figure(figsize=(10, 6))
    plt.plot(range(nEpochs), loss_per_epoch, label="Training Loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.show()

# Run the model with test data and forecast
def test_rnn_model(test_data_file, model, horizonSteps, nInputs, nOutputs):
    print("TWC-NARMAX:-> running 'test_rnn_model'")
    test_inputs, test_outputs = load_data(test_data_file, nInputs, nOutputs)

    X_test, y_test = create_sequences(test_inputs, test_outputs, nInputs, nOutputs, dDelayInputs, nDelayOutputs, nDelayNoise, horizonSteps)

    # Load model weights
    model.load_weights('narmax_rnn_model_weights.h5')

    predictions = []
    for i in range(len(X_test)):
        X_batch = X_test[i:i+1]

        # Reshape X_batch to be (1, horizonSteps, nInputs + nOutputs)
        X_batch = np.reshape(X_batch, (1, horizonSteps, X_batch.shape[2]))
        print(X_train.shape, X_batch.shape, y_train.shape, y_batch.shape, \
              model, horizonSteps, nInputs, nOutputs)

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
    print("TWC-NARMAX::> Running 'main' function ...")

    # Load training data
    print("TWC-NARMAX::> Loading the data sets ...")
    time, train_inputs, train_outputs = load_data(training_data_file, nInputs, nOutputs)
    print(f"TWC-NARMAX::> Training data loaded: \n",
          f"        inputs shape: {train_inputs.shape}, \n",
          f"       outputs shape: {train_outputs.shape}")

    # Prepare training sequences
    # (time, inputs, outputs, horizonSteps, nInputs, nOutputs)
    print("TWC-NARMAX::> Creating sequences (with delays and forecast) ...")
    X_train, y_train = create_sequences(train_inputs, train_outputs, nInputs, nOutputs, nDelayInputs, nDelayOutputs, nDelayNoise, horizonSteps)
    print(f"TWC-NARMAX::> Training sequences created, \n",
          f"       X_train shape: {X_train.shape}, \n",
          f"       y_train shape: {y_train.shape}")

    print("TWC-NARMAX::> Creating 'model' ...")
    # Create and train the model
    model = create_rnn_model(nInputs, nOutputs, nDelayInputs, nDelayOutputs, nDelayNoise, horizonSteps)
    print(f"TWC-NARMAX::> model created:\n{model.summary()}")

    print("TWC-NARMAX::> Training 'model' ...")
    train_rnn_model(model, X_train, y_train, nEpochs, horizonSteps)

    # Test the model on the test data
    print("TWC-NARMAX::> Testing 'model' ...")
    test_rnn_model(test_data_file, model, horizonSteps, nInputs, nOutputs)

    print("TWC-NARMAX::> Done!")


# Example usage
if __name__ == "__main__":
    print("TWC-NARMAX::> Running 'NARMAX.py' ...")
    training_data_file = 'training_data.csv'  # Input your training CSV file path
    test_data_file     = 'test_data.csv'  # Input your test CSV file path
    nInputs            = 3  # Number of input features
    nOutputs           = 2  # Number of output features
    nDelayInputs       = 5  # Delay in inputs
    nDelayOutputs      = 3  # Delay in outputs
    nDelayNoise        = 2  # Delay in noise
    horizonSteps       = 10  # Forecasting horizon
    nEpochs            = 50  # Number of training epochs

    main(training_data_file, test_data_file, nInputs, nOutputs, nDelayInputs, nDelayOutputs, nDelayNoise, horizonSteps, nEpochs)
