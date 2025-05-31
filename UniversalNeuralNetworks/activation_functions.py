import numpy as np

# === Standard Activations ===

def identity(x):
    return x

def binary_step(x):
    return np.where(x >= 0, 1.0, 0.0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def elu(x, alpha=1.0):
    return np.where(x >= 0, x, alpha * (np.exp(x) - 1))

def swish(x):
    return x * sigmoid(x)

def mish(x):
    return x * np.tanh(np.log1p(np.exp(x)))  # softplus inside tanh

def softplus(x):
    return np.log1p(np.exp(x))

# === Normalization / Gating Activations ===

def softmax(x, axis=-1):
    """Numerically stable softmax"""
    shiftx = x - np.max(x, axis=axis, keepdims=True)
    exps = np.exp(shiftx)
    return exps / np.sum(exps, axis=axis, keepdims=True)

def log_softmax(x, axis=-1):
    """Log of softmax for better numerical stability"""
    shiftx = x - np.max(x, axis=axis, keepdims=True)
    logsumexp = np.log(np.sum(np.exp(shiftx), axis=axis, keepdims=True))
    return shiftx - logsumexp

# === Derivatives for Backpropagation ===

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh_derivative(x):
    t = tanh(x)
    return 1 - t**2

def relu_derivative(x):
    return np.where(x > 0, 1.0, 0.0)

def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1.0, alpha)

def softplus_derivative(x):
    return sigmoid(x)

if __name__ == '__main__':
    import numpy as np
    from activation_functions import relu, softmax

    x = np.array([-1.0, 0.0, 1.0, 2.0])
    print("ReLU:", relu(x))
    print("Softmax:", softmax(x))
