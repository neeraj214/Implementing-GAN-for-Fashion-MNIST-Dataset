from tensorflow.keras.layers import Conv2D, LeakyReLU, Dropout

def discriminator_block2(x, filters=128, dropout_rate=0.3):
    """
    Second convolutional block of the discriminator.
    stride=2 downsamples 14x14 → 7x7, extracts higher-level features.

    Args:
        x: Input tensor.
        filters (int): Number of filters for Conv2D (default: 128).
        dropout_rate (float): Dropout probability (default: 0.3).

    Returns:
        Tensor: Downsampled and processed feature map.
    """
    x = Conv2D(filters, (5, 5), strides=(2, 2), padding='same',
               name="conv_block2")(x)
    x = LeakyReLU(name="leakyrelu_block2")(x)
    x = Dropout(dropout_rate, name="dropout_block2")(x)
    
    # Expected output shape: (None, 7, 7, 128)
    return x
