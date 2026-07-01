from tensorflow.keras.layers import Conv2D, LeakyReLU, Dropout

def discriminator_block1(x, filters=64, dropout_rate=0.3):
    """
    First convolutional block of the discriminator.
    stride=2 downsamples 28x28 → 14x14, extracts low-level features.
    Per DCGAN best practices, there is no BatchNormalization in this first layer.

    Args:
        x: Input tensor.
        filters (int): Number of filters for Conv2D (default: 64).
        dropout_rate (float): Dropout probability (default: 0.3).

    Returns:
        Tensor: Downsampled and processed feature map.
    """
    x = Conv2D(filters, (5, 5), strides=(2, 2), padding='same',
               name="conv_block1")(x)
    x = LeakyReLU(name="leakyrelu_block1")(x)
    x = Dropout(dropout_rate, name="dropout_block1")(x)
    
    # Expected output shape: (None, 14, 14, 64)
    return x
