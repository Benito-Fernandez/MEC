import csv
import numpy as np

def generate_data(filename: str, n_samples: int, input_dim: int, output_dim: int) -> None:
    """
    Generate a CSV file containing synthetic data.

    Each row contains (input_dim + output_dim) float values:
      - The first input_dim values represent randomly generated inputs.
      - The following output_dim values represent the target outputs, which here are also randomly generated.
    """
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for _ in range(n_samples):
            # Generate random inputs and outputs within the range [0, 1).
            inputs = np.random.rand(input_dim)
            outputs = np.random.rand(output_dim)
            # Concatenate inputs and outputs into a single row.
            row = np.concatenate([inputs, outputs])
            writer.writerow(row.tolist())
    print(f"Data written to {filename}")

if __name__ == "__main__":
    # Configuration parameters
    n_samples_train = 100  # Number of training samples
    n_samples_test = 30    # Number of testing samples
    input_dim = 3          # Number of input features, for example from a 3-neuron input layer
    output_dim = 2         # Number of output labels, for example for a 2-neuron output layer

    # File names consistent with the NNet class defaults
    training_filename = 'trainingData.csv'
    testing_filename = 'testingData.csv'

    # Generate the synthetic training and testing data files
    generate_data(training_filename, n_samples_train, input_dim, output_dim)
    generate_data(testing_filename, n_samples_test, input_dim, output_dim)