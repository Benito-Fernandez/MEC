import numpy as np
import matplotlib.pyplot as plt
from genericNNetGPT import NNet

# Step 1: Generate the dataset
def generate_data(n_points=1000, noise_std=0.1):
    # Generate random x values
    x_train = np.random.uniform(0, 2 * np.pi, n_points)
    # Compute y values
    y_train = np.sin(2 * x_train - 1) * np.cos(x_train) + np.random.normal(0, noise_std, n_points)

    # Prepare the data in the format expected by the network
    return x_train.reshape(-1, 1), y_train.reshape(-1, 1)  # Reshape for compatibility

# Generate training and testing data
x_train, y_train = generate_data()
x_test, y_test = generate_data(n_points=100)  # Smaller set for testing

# Step 2: Initialize the neural network
n_inputs = 1
n_hidden = [5, 5]  # Example hidden layers
n_outputs = 1

myNN = NNet(n_inputs=n_inputs, n_hidden=n_hidden, n_outputs=n_outputs)

# Step 3: Train the network (this part assumes a proper `train` method is implemented)
n_epochs = 1000
learning_rate = 0.01
myNN.train(n_epochs=n_epochs, learning_rate=learning_rate, display_epoch=100)

# Step 4: Predict using the trained network
predictions = []
for x in x_test:
    pred = myNN.predict(x.reshape(1, -1))  # Make it 2D for input compatibility
    predictions.append(pred[0][0])  # Obtain the scalar output

# Convert predictions to a NumPy array for easy plotting
predictions = np.array(predictions)

# Step 5: Visualize the results
plt.figure(figsize=(12, 6))
plt.scatter(x_train, y_train, label='Training Data', color='blue', alpha=0.5)
plt.scatter(x_test, y_test, label='Testing Data', color='red', alpha=0.5)
plt.scatter(x_test, predictions, label='Predictions', color='green', alpha=0.7)
plt.title('Neural Network Predictions vs Actual Data')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
