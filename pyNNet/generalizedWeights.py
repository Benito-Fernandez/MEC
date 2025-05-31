from collections import defaultdict

class Weights():
    def __init__(self, n1, n2):
        self.source = n1
        self.sink   = n2
        self.data = {}
        for i in range(n1):
            self.data[i]={}
            for j in range(n2):
                self.data[i][j] = [[]]

    def add_array(self, i, j, _array, debug=False):
        if i not in self.data:
            self.data[i] = {}
            if debug: print(f"|---> self.source inceased from {self.source} to {i+1}")
            self.source = i+1
        self.data[i][j] = _array
        if debug: print(f"updated self.data[{i}][{j}] = {_array}")
        if j > self.sink:
            if debug: print(f"|---> self.sink inceased from {self.sink} to {j+1}")
            self.sink = j+1

    def get_array(self, i, j):
        if i in self.data and j in self.data[i]:
            return self.data[i][j]
        else:
            return None

    def __call__(self, source=None, sink=None):
        if source is None:
            if sink is None:
                return None
            return self.get_sources(sink)
        if source in self.data:
            if sink is None:
                return self.get_sinks(source)
            if sink in self.data[source]:
                return self.get_array(source, sink)
        else:
            return None

    def get_sources(self, sink):
        result = {}
        for (index1, arr) in self.data.items():
            for index2 in range(len(arr)):
                if index2 == sink:
                    result[index1] = arr[sink]
        return result

    def get_sinks(self, source):
        if source in self.data:
            result = {}
            for j in self.data[source]:
                result[j] = self.data[source][j]
            return result
        return None

    def __repr__(self):
       repr = f"W({W.source},{W.sink}): "
       for i in range(self.source):
            for j in range(self.sink):
                repr += f"\n W[{i}][{j}] -> {W.get_array(i,j)}"
       return repr

    def is_connected(self, i, j):
       return i in self.data and j in self.data[i] # self.get_array(i,j) != None


if __name__ == "__main__":
    n1, n2 = 2, 3
    W = Weights(n1, n2) # <- empty
    print(50*'|')
    print(50*'v')
    print(f"Just created Weights: 'W = Weights({W.source},{W.sink}))")
    print(f"W({W.source},{W.sink}).data = \n{W.data}")
    print(f"Printing Full Weights: 'print(W)'n{W}")

    print(50*'-')
    # Adding weights
    w1 = [2]
    w2 = [[2, 3],[5, 7]]
    w3 = [-1,[1,2]]
    W.add_array(0,1,w2)
    W.add_array(1,1,w1)
    W.add_array(1,0,w3)

    print("Just added a couple of weights within limits")
    print(f"W({W.source},{W.sink}).data = \n{W.data}")
    print(f"Printing Full Weights: 'print(W)'n{W}")

    print(50*'-')
    W.add_array(2,0,[[[2],[3]],[1]])
    print("Just added a couple of weights outside limits")
    print(f"W({W.source},{W.sink}).data = \n{W.data}")
    print(f"Printing Full Weights: 'print(W)'n{W}")

    print(50*'-')
    print(f"Is layer 2, connected to layer 1? : W.is_connected(2,1) = {W.is_connected(2,1)}")
    print(f"W.get_array(2,0) = {W.get_array(2,0)}")
    print(f"W(2,0) = {W(2,0)}")
    print(f"W(1,) = {W(1,)}")
    print(f"W(,1) DOENS'T WROK")
    print(f"W(None,1) or W.get_sources(1) = {W(None,1)}")

    print(50*'-')
##    print("----------")
##    print(f"Weights.__doc__ = ")
##    print(W.data.__doc__)
##    print("----------")
##    print(W.data.keys)
    for i in range(len(W.data)):
        print(f"W.get_sources({i}) = {W.get_sources(i)}")
    print("-----------")
    for i in range(len(W.data)):
        print(f"W.get_sinks({i}) = {W.get_sinks(i)}")


##
##def find_arrays_with_same_second_index(data):
##    """
##    Finds arrays with the same second index in a dictionary.
##
##    Args:
##        data (dict): A dictionary where keys are tuples (index1, index2)
##                     and values are arrays.
##
##    Returns:
##        dict: A dictionary where keys are second indices and values are lists
##              of arrays that share that second index.
##    """
##    result = defaultdict(list)
##    for (index1, index2), arr in data.items():
##        result[index2].append(arr)
##    return dict(result)
##
### Example Usage
##data = {
##    (1, 1): [1, 2, 3],
##    (1, 2): [4, 5, 6],
##    (2, 1): [7, 8, 9],
##    (2, 2): [10, 11, 12],
##    (3, 3): [13, 14, 15]
##}
##
##arrays_with_same_second_index = find_arrays_with_same_second_index(data)
##print(arrays_with_same_second_index)
### Expected Output:
### {1: [[1, 2, 3], [7, 8, 9]], 2: [[4, 5, 6], [10, 11, 12]], 3: [[13, 14, 15]]}

print(50*'-')#--------------------------------------------
def flexible_function(*args, **kwargs):
    if not args and not kwargs:
        print("No arguments provided.")
    else:
        print("Positional arguments:", args)
        print("Keyword arguments:", kwargs)

flexible_function()  # Output: No arguments provided.
flexible_function(1, 2, 3, name="Alice", age=30)
# Output:
# Positional arguments: (1, 2, 3)
# Keyword arguments: {'name': 'Alice', 'age': 30}

print(50*'-')#--------------------------------------------

def print_args(*args, **kargs):
    for arg in args:
        print(arg)

def print_kwargs(*args, **kwargs):
    for key, value in kwargs.items():
        print(f"{key}: {value}")

print_args(1, 2, 3, name="Alice", age=30)
print("----")
print_kwargs(1, 2, 3, name="Alice", age=30)
print("--------")
print_args(1, 2, 3)
print("----")
print_kwargs(name="Alice", age=30)

print(50*'-')#--------------------------------------------
def do_something(data):
    if not data:
        pass # Do nothing if data is empty
    else:
        print("Processing data")

do_something(None) # Does nothing
do_something("some data") # Prints "Processing data"

print(50*'-')#--------------------------------------------
def show(arg1, arg2):
    print(f"arg1: {arg1}, arg2: {arg2}")

show(1, 2)  # Valid
#show(, 2)  # Invalid syntax
show(arg2=2, arg1=1) # Valid
def show_with_default(arg1=None, arg2=None):
    print(f"arg1: {arg1}, arg2: {arg2}")

show_with_default(1, 2)  # Valid
print("----")
show_with_default(arg2=2)  # Valid, arg1 will be None
print("----")
show_with_default() # Valid, both args will be None

print(50*'^')
print(50*'|')

import numpy as np

import numpy as np

class DoubleIndexedArray:
    def __init__(self, data):
        self.data = data

    def __call__(self, index1, index2):
        # Access the 2D array for the given indices
        return np.array(self.data.get((index1, index2), None))  # Return None if key does not exist

    def dot(self, index1, index2, x):
        # Get the 2D array and calculate the dot product with x
        array = self(index1, index2)
        if array is None:
            raise ValueError(f"No array found for indices ({index1}, {index2})")
        return np.dot(array, x)

    def list_index_pairs(self):
        # Return a set of existing index pairs
        return set(self.data.keys())

    def exist(self, index1, index2):
        # Check if the index pair exists in the data
        return (index1, index2) in self.data

    def get_element(self, index1, index2):
        """Get the element (2D array) at the specified index pair."""
        return self(index1, index2)

    def set_element(self, index1, index2, value):
        """Set the element (2D array) at the specified index pair."""
        self.data[(index1, index2)] = value

    def modify_element(self, index1, index2, new_value):
        """Modify the element (2D array) at the specified index pair."""
        if (index1, index2) in self.data:
            self.data[(index1, index2)] = new_value
        else:
            raise KeyError(f"No existing index pair ({index1}, {index2}) to modify.")

    def remove_element(self, index1, index2):
        """Remove the element (2D array) at the specified index pair."""
        try:
            del self.data[(index1, index2)]
        except KeyError:
            raise KeyError(f"No existing index pair ({index1}, {index2}) to remove.")

    def print_entries(self):
        """Print each entry in data on its own line."""
        print("Current entries in DoubleIndexedArray:")
        for (index1, index2), value in self.data.items():
            print(f"Index Pair: ({index1}, {index2}) => Value: {value}")

    def __repr__(self):
        # String representation of the object
        index_pairs = self.list_index_pairs()
        repr = 32*'_'\
             + f"\n{self.__class__.__name__}(\n  data_pairs = {len(index_pairs)} pairs, "\
             + f"\n  existing pairs = {list(index_pairs)}"\
             +  f"\n  data = "
        for (index1, index2), value in self.data.items():
            repr += f"\n         ({index1}, {index2}) : {value}"
        repr += '\n)'
        return repr

    def categorize_connections(self):
        """Categorize connections into Feed-Forward (FF) and Feedback (FB)."""
        ff_connections = []
        fb_connections = []
        for (index1, index2) in self.data:
            if index2 > index1:
                ff_connections.append((index1, index2))
            elif index2 <= index1:
                fb_connections.append((index1, index2))
        return ff_connections, fb_connections

    def print_connection_types(self):
        """Print out the types of connections (FF and FB)."""
        ff, fb = self.categorize_connections()
        print("Feed-Forward Connections:")
        for conn in ff:
            print(f" - From Layer {conn[0]} to Layer {conn[1]}")
        print("Feedback Connections:")
        for conn in fb:
            print(f" - From Layer {conn[0]} to Layer {conn[1]}")

#------------------------------------------------------------
# Example usage
data = {
    (1, 2): [[1, 2, 3], [4, 5, 6]],   # FF connection
    (2, 3): [[7, 8, 9], [10, 11, 12]], # FF connection
    (2, 4): [[7, 3, 4], [9, -1, 0]], # FF connection
    (3, 1): [[13, 14, 15], [16, 17, 18]], # FB connection
}
# Instantiate the class
A = DoubleIndexedArray(data)

# Display the representation of the object
print(f"print(A = DoubleIndexedArray(data)) =\n{A}")

# List existing index pairs
existing_pairs = A.list_index_pairs()
print("Existing index pairs = A.list_index_pairs():", existing_pairs)

# Check if specific pairs exist
print("Does A.exist(1, 3) exist? ", A.exist(1, 3))  # Should return True
print("Does A.exist(1, 2) exist? ", A.exist(1, 2))  # Should return False
print("Does A.exist(2, 4) exist? ", A.exist(2, 4))  # Should return True
print("Does A.exist(3, 2) exist? ", A.exist(3, 2))  # Should return False

# Get element at (1, 3)
print("Element at (1, 3), A.get_element(1, 3): ", A.get_element(1, 3))

# Print all entries
A.print_entries()

# Set new element at (1, 2)
print("Setting new element at (1, 2)...")
A.set_element(2, 2, [[19, 20, 21], [22, 23, 24]])
print("New element at (1, 2):", A.get_element(1, 2))

# Print all entries
A.print_entries()

# Print connection types
A.print_connection_types()

# Modify element at (2, 4)
print("Modifying element at (2, 4)...")
A.modify_element(2, 4, [[13, 14, 15], [16, 17, 18]])
print("New element at (2, 4):", A.get_element(2, 4))

# Display the updated representation of the object
print(A)

# Remove an existing element
print("Removing element at (1, 2)...")
A.remove_element(1, 2)
print(f"A.remove_element(1, 2) creates new A as:\n{A}")

# Extract specific array
print(f"A(2,3) = \n{A(2,3)}")

# Use element in dot product
x = np.array([1, 2, -1])
print(f"x = np.array([1, 2, -1]) = {x}")

# Perform dot product
y = A.dot(2, 3, x)
print(f"y = A.dot(2, 3, x) = {y}")

z = np.dot(A(2,4), x)
print(f"z = np.dot(A(2,4), x) = {np.dot(A(2,4), x)}")

# Try to remove a non-existing index pair to see the error handling
print("Trying to remove a non-existing index pair\n>>>A.remove_element(1, 2)")
try:
    A.remove_element(1, 2)
except KeyError as e:
    print(e)

