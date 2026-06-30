from tensorflow.keras.layers import Reshape

def reshape_to_feature_map(x, target_shape=(7, 7, 256)):
    """
    Converts flat dense output into a spatial 7x7x256 tensor,
    the starting point for transposed convolutions.

    Args:
        x: Input tensor (flat dense projection).
        target_shape (tuple): The target shape to reshape to (default: (7, 7, 256)).

    Returns:
        Tensor: The reshaped spatial tensor of shape target_shape.
    """
    # Assert comment: expected input flat size = 7*7*256 = 12544
    return Reshape(target_shape, name="reshape_7x7x256")(x)
