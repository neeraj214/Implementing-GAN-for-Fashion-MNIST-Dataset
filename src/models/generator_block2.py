from tensorflow.keras.layers import Conv2DTranspose, BatchNormalization, LeakyReLU

def generator_block2(x, filters=64):
    """
    stride=2 upsamples spatial resolution 7x7 -> 14x14.

    Args:
        x: Input tensor.
        filters (int): Number of filters for Conv2DTranspose (default: 64).

    Returns:
        Tensor: Processed tensor.
    """
    x = Conv2DTranspose(filters, (5, 5), strides=(2, 2), padding='same',
                        use_bias=False, name="convT_block2")(x)
    x = BatchNormalization(name="bn_block2")(x)
    x = LeakyReLU(name="leakyrelu_block2")(x)
    
    # Expected output shape: (None, 14, 14, 64)
    return x
