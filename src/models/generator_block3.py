from tensorflow.keras.layers import Conv2DTranspose
from setup.train_config import CHANNELS

def generator_block3(x, channels=CHANNELS):
    """
    final upsample 14x14 -> 28x28, tanh keeps output in [-1, 1]
    matching the normalization range used in src/data/normalize.py.

    Args:
        x: Input tensor.
        channels (int): Number of output channels (default: CHANNELS).

    Returns:
        Tensor: Output layer tensor.
    """
    outputs = Conv2DTranspose(channels, (5, 5), strides=(2, 2), padding='same',
                              use_bias=False, activation='tanh',
                              name="convT_output")(x)
    
    # Expected output shape: (None, 28, 28, 1)
    return outputs
