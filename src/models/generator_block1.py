from tensorflow.keras.layers import Conv2DTranspose, BatchNormalization, LeakyReLU

def generator_block1(x, filters=128):
    """
    stride=1 keeps spatial size at 7x7, increases feature depth processing.

    Args:
        x: Input tensor.
        filters (int): Number of filters for Conv2DTranspose (default: 128).

    Returns:
        Tensor: Processed tensor.
    """
    x = Conv2DTranspose(filters, (5, 5), strides=(1, 1), padding='same',
                        use_bias=False, name="convT_block1")(x)
    x = BatchNormalization(name="bn_block1")(x)
    x = LeakyReLU(name="leakyrelu_block1")(x)
    
    # Expected output shape: (None, 7, 7, 128)
    return x
