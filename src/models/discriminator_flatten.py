from tensorflow.keras.layers import Flatten

def flatten_features(x):
    """
    Converts 7x7x128 spatial feature map into a flat vector
    of size 6272 for the final classification layer.

    Args:
        x: Input tensor of shape (None, 7, 7, 128).

    Returns:
        Tensor: Flattened feature vector.
    """
    x = Flatten(name="flatten_features")(x)
    
    # Expected output shape: (None, 6272)
    return x
