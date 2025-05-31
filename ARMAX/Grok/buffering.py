import numpy as np

#-------------------------------------------------------------------------------

def arrange_samples(Y, k, n_history, all=False):
    """
    Arrange samples by stacking columns from matrix Y.

    Parameters:
        Y (numpy.ndarray): Input array with shape (n_outputs, n_samples).
        k (int): Starting column index (ignored if all=True).
        n_history (int): Number of past columns to concatenate.
        all (bool): If False, stack only for column k.
                    If True, stack for k ranging from n_history to n_samples.

    Returns:
        numpy.ndarray: Stacked samples with adjusted dimensions.
    """
    n_outputs, n_samples = Y.shape

    if all:
        # Generate stacked samples for k from n_history to n_samples-1
        stacked_list = [np.vstack([Y[:, i - j] for j in range(n_history)]).reshape(n_outputs * n_history, 1)
                        for i in range(n_history, n_samples)]
        return np.hstack(stacked_list)  # Correctly stacks along the second axis
    else:
        # Stack columns only for given k
        stacked = np.vstack([Y[:, k - i] for i in range(n_history)])
        return stacked.reshape(n_outputs * n_history, 1)  # Ensure correct shape

#-------------------------------------------------------------------------------

# Example usage
n_outputs = 5
n_samples = 10
Y = np.random.rand(5, 10) # (n_outputs=5, n_samples=10)
print(f"Y = {Y.shape}\n{Y}\n")

k = 7
n_history = 3

print("Executing 'arrange_samples'")

result1 = arrange_samples(Y, k, n_history, all=False)
print(f"result1 = arrange_samples(Y, k={k}, n_history={n_history}, all=False)\n{result1.shape}")  # Expected shape: ((n_history+1)*n_outputs,)
print(result1)

result2 = arrange_samples(Y, k, n_history, all=True)
print(f"result2 = arrange_samples(Y, k={k}, n_history={n_history}, all=True)\n{result2.shape}")  # Expected shape: ((n_history+1)*n_outputs,)
print(result2)

#-------------------------------------------------------------------------------

def update_buffer(buffer, next_sample, n_history, n_outputs):
    """
    Updates buffer by shifting previous values downward,
    inserting the new column next_sample at the top.

    Parameters:
        buffer (numpy.ndarray):
            Current Buffer array of shape (n_outputs * n_history, 1).
        next_sample (numpy.array):
            Next Input array of shape (n_outputs, 1)

    Returns:
        numpy.ndarray: Updated buffer of shape (n_outputs * n_history, 1).
    """

    # Shift buffer down
    buffer[n_outputs:, :] = buffer[:(n_history-1)*n_outputs, :]

    # Insert new column Y[:, i+1] at the top
    buffer[:n_outputs, :] = next_sample.reshape(n_outputs, 1)

    return buffer  # Return one stacked column at a time

#-------------------------------------------------------------------------------

def collect_sample(Y, k, n_history, n_outputs):
    """
    Maintains a buffer and updates it by shifting previous values downward,
    inserting the new column Y[:, k+1] at the top.

    Parameters:
        Y (numpy.ndarray): Input array of shape (n_outputs, n_samples).
        k (int): Current column index.
        n_history (int): Number of past columns stored.

    Returns:
        numpy.ndarray: Updated buffer of shape (n_outputs * n_history, 1).
    """
    n_outputs, n_samples = Y.shape

    # Initialize buffer with the first stacked sample
    buffer = arrange_samples(Y, n_history-2, n_history, all=False)

    print(f"scrolling k, from {k-1} to {n_samples - 1}")
    for i in range(k, n_samples):
        # Shift buffer down
        buffer[n_outputs:, :] = buffer[:(n_history-1)*n_outputs, :]

        # Insert new column Y[:, i+1] at the top
        buffer[:n_outputs, :] = Y[:, i].reshape(n_outputs, 1)

        yield buffer  # Return one stacked column at a time

#-------------------------------------------------------------------------------

##print("Executing 'collect_sample'")
# Example usage
##print(f"Y = {Y.shape}\n{Y}\n")
n_outputs, n_samples = Y.shape
n_history = 3
k=0
##result3 = np.array(result2[:,0]*(n_samples-n_history)).reshape(n_outputs * n_history, 1)
result3 = result2
print(result3.shape)
##print(result3[:,1].reshape(n_outputs * n_history, 1))

# Loop through k from n_history to n_samples
for stacked_column in collect_sample(Y, n_history, n_history, n_outputs):
##    print(f"y_sample[{k}] = {stacked_column.shape}\n {stacked_column}")  # Expected shape: (n_outputs*n_history, 1)
    result3[:,k] = stacked_column.reshape(n_outputs * n_history, 1).T
    k+=1

print(f"result3 = {result3.shape}\n{result3}")
##print(f"result3[:,0] = {result3[:,0].shape}\n{result3[:,0]}")

print(f"result2 == result3? : \n{result2 == result3}")

print("\n .... DONE")