import numpy as np

def normalize(x, mean=None, std=None):
    """Standard score normalization: (x - mean) / std"""
    if mean is None:
        mean = np.mean(x, axis=0)
    if std is None:
        std = np.std(x, axis=0)
    std[std == 0] = 1  # avoid divide by zero
    return (x - mean) / std

def min_max_scale(x, feature_range=(0, 1)):
    """Rescale features to a range (default [0, 1])"""
    min_val = np.min(x, axis=0)
    max_val = np.max(x, axis=0)
    scale = (feature_range[1] - feature_range[0]) / (max_val - min_val + 1e-8)
    return feature_range[0] + (x - min_val) * scale

def add_bias(x):
    """Append a bias (1.0) to each input vector."""
    if x.ndim == 1:
        return np.append(x, 1.0)
    else:
        bias = np.ones((x.shape[0], 1))
        return np.hstack((x, bias))

def dropout_input(x, dropout_rate=0.1, training=True):
    """Apply dropout to the input (set some elements to zero during training)."""
    if not training or dropout_rate <= 0.0:
        return x
    mask = np.random.binomial(1, 1 - dropout_rate, size=x.shape)
    return x * mask

def clip_input(x, min_val=-1.0, max_val=1.0):
    """Clip input values to a given range."""
    return np.clip(x, min_val, max_val)

def one_hot_encode(labels, num_classes=None):
    """Convert class labels to one-hot encoded vectors."""
    labels = np.array(labels, dtype=int).flatten()
    if num_classes is None:
        num_classes = np.max(labels) + 1
    return np.eye(num_classes)[labels]

def binarize(x, threshold=0.5):
    """Convert input to binary (0 or 1) based on threshold."""
    return (x >= threshold).astype(float)

if __name__ == '__main__':
    #import numpy as np
    #from input_operators import normalize, add_bias, one_hot_encode

    x = np.random.rand(5, 3) * 10
    x_norm = normalize(x)
    x_bias = add_bias(x_norm)
    print(f"x = \n{x}")
    print(f"\nx_norm = normalize(x) = \n{x_norm}")
    print(f"\nx_bias = add_bias(x_norm) = \n{x_bias}")

    labels = [0, 2, 1, 0, 1]
    y = one_hot_encode(labels, num_classes=3)
    print(f"\nlabels = {labels}")
    print(f"\ny = one_hot_encode(labels, num_classes=3) = \n{y}")
