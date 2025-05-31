import numpy as np

# === REGRESSION LOSSES ===

def mean_squared_error(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def huber_loss(y_true, y_pred, delta=1.0):
    error = y_true - y_pred
    is_small_error = np.abs(error) <= delta
    squared_loss = 0.5 * np.square(error)
    linear_loss = delta * (np.abs(error) - 0.5 * delta)
    return np.mean(np.where(is_small_error, squared_loss, linear_loss))

# === CLASSIFICATION LOSSES ===

def binary_crossentropy(y_true, y_pred, epsilon=1e-15):
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def categorical_crossentropy(y_true, y_pred, epsilon=1e-15):
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.sum(y_true * np.log(y_pred), axis=-1).mean()

def sparse_categorical_crossentropy(y_true, y_pred, epsilon=1e-15):
    """`y_true` is a 1D array of class indices, `y_pred` is a 2D softmax probability array"""
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    indices = y_true.astype(int)
    return -np.mean(np.log(y_pred[np.arange(len(y_pred)), indices]))

# === ADDITIONAL METRICS ===

def hinge_loss(y_true, y_pred):
    return np.mean(np.maximum(0, 1 - y_true * y_pred))

def squared_hinge_loss(y_true, y_pred):
    return np.mean(np.square(np.maximum(0, 1 - y_true * y_pred)))

# === LOG LOSS (alias for binary crossentropy) ===

def log_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred)

if __name__ == '__main__':
    import numpy as np
    from loss_functions import mean_squared_error, binary_crossentropy

    y_true = np.array([1, 0, 1])
    y_pred = np.array([0.9, 0.1, 0.8])

    print("MSE:", mean_squared_error(y_true, y_pred))
    print("Binary Crossentropy:", binary_crossentropy(y_true, y_pred))
