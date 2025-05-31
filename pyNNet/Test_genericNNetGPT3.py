import numpy as np
import matplotlib.pyplot as plt
from genericNNetGPT3 import NNet

# -----------------------------------------------------------------------------
# Step 1: Generate the dataset
# -----------------------------------------------------------------------------
def generate_data(n_points=1000, noise_std=0.1):
    """
    Generate a synthetic dataset.

    For each data point:
      - x is drawn uniformly from [0, 2*pi]
      - y is computed as: sin(2*x - 1) * cos(x) plus Gaussian noise.

    Returns:
      x as an (n_points, 1) array and y as an (n_points, 1) array.
    """
    x = np.random.uniform(0, 2 * np.pi, n_points)
    y = 2 * np.sin(1.3 * x - 1) - np.cos(0.7 * x) + np.random.normal(0, noise_std, n_points)
    return x.reshape(-1, 1), y.reshape(-1, 1)

# Generate training and testing data
training_points = 1000
testing_points = 500
noise_std=0.1
x_train, y_train = generate_data(n_points=training_points, noise_std=noise_std)
x_test, y_test   = generate_data(n_points=testing_points,  noise_std=noise_std)

# -----------------------------------------------------------------------------
# Step 2: Prepare the training data in the format expected by the network.
# -----------------------------------------------------------------------------
# The updated NN expects each sample to be formatted as:
#    [ input1, input2, ..., target1, target2, ... ]
# For our case (n_inputs = 1, n_outputs = 1), we horizontally stack x and y.
training_data = np.hstack([x_train, y_train])
# -----------------------------------------------------------------------------
# Step 2b: Visualize the results
# -----------------------------------------------------------------------------
##plt.figure(figsize=(12, 6))
##plt.scatter(x_train, y_train, label='Training Data', color='blue', alpha=0.5)
##plt.title('Actual Training Data')
##plt.xlabel('x')
##plt.ylabel('y')
##plt.legend()
##plt.grid(True)
##plt.show()

# -----------------------------------------------------------------------------
# Step 3: Initialize and configure the neural network
# -----------------------------------------------------------------------------
n_inputs  = 1
n_hidden  = [9, 11]  # two hidden layers of 5 neurons each
n_outputs = 1

# Create an instance of NNet imported from your module.
myNN = NNet(n_inputs=n_inputs, n_hidden=n_hidden, n_outputs=n_outputs)

# Override the training data loaded from CSV with our generated data.
myNN.training_data = training_data
print(f"--> training_data generated {training_data.shape}")

# -----------------------------------------------------------------------------
# Step 4: Train the network
# -----------------------------------------------------------------------------
# The train method of the updated NN class uses the learning rate
# provided at initialization. (In our implementation, train does not accept a learning_rate parameter.)
n_epochs = 25000
myNN.train(n_epochs=n_epochs, display_epoch=100, cost_type='SE', aggregate='sum')
print(f"--> training done {training_data.shape}")

# -----------------------------------------------------------------------------
# Step 5: Predict on the test set
# -----------------------------------------------------------------------------
predictions = []
# Each entry in x_test is an array of shape (1,). We call predict with a 1D array.
for xi in x_test:
    # Ensure xi is 1D using ravel (so that its shape is (n_inputs,))
    pred = myNN.predict(xi.ravel())
    # predict returns a 1D array (of length n_outputs); extract its first element.
    predictions.append(pred[0])
predictions = np.array(predictions)
print(f"--> predictions done {predictions.shape}")

# -----------------------------------------------------------------------------
# Step 6: Visualize the results
# -----------------------------------------------------------------------------
plt.figure(figsize=(12, 6))
plt.scatter(x_train, y_train, label='Training Data', color='blue', alpha=0.5)
plt.scatter(x_test, y_test, label='Testing Data', color='red', alpha=0.5)
plt.scatter(x_test, predictions, label='Predictions', color='green', marker='x', s=80, alpha=0.7)
plt.title('Neural Network Predictions vs Actual Data')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.legend()
plt.show()
print(f"--> plotting is done!")
