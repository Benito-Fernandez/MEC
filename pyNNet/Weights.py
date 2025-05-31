import numpy as np
import csv

class Weights:
    def __init__(self):
        self.data = {}

    def __call__(self, index1, index2):
        return np.array(self.data.get((index1, index2), None))

##    def set_weights(self, index1, index2, value):
##        self.data[(index1, index2)] = value

    def set_weights(self, index1: int, index2: int, value: np.ndarray) -> None:
        self.data[(index1, index2)] = value

    def remove_weights(self, index1, index2):
        try:
            del self.data[(index1, index2)]
        except KeyError:
            raise KeyError(f"No existing weights for indices ({index1}, {index2}) to remove.")

    def print_weights(self):
        print("Current weights:")
        for (index1, index2), value in self.data.items():
            print(f"From layer {index1} to layer {index2}: {value}")

##    def load_from_csv(self, filename):
##        with open(filename, 'r') as file:
##            reader = csv.reader(file)
##            self.data = { (int(row[0]), int(row[1])): np.array([float(x) for x in row[2:]], dtype=float) for row in reader }

    def load_from_csv(self, filename: str) -> None:
        try:
            data = np.loadtxt(filename, delimiter=',')
            self.data = { (int(row[0]), int(row[1])): np.array(row[2:], dtype=float) for row in data }
        except Exception as e:
            raise ValueError(f"Error loading CSV file: {e}")

    def save_to_csv(self, filename):
        with open(filename, 'w') as file:
            writer = csv.writer(file)
            for (index1, index2), value in self.data.items():
                writer.writerow([index1, index2] + value.tolist())

    def __getitem__(self, indices):
        value = self.data.get(indices, None)
        if value is None:
            raise KeyError(f"No weights found for indices {indices}.")
        return value