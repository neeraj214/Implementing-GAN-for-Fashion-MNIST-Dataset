from tensorflow.keras.layers import Dense

def discriminator_output(x):
    """
    Outputs a single raw logit (no sigmoid activation is used here).
    BinaryCrossentropy(from_logits=True) will be used in the loss function
    for numerical stability during training (applied in src/losses/discriminator_loss.py).

    Args:
        x: Input flattened features tensor of shape (None, 6272).

    Returns:
        Tensor: Single output logit representing real vs fake classification.
    """
    outputs = Dense(1, name="real_fake_logit")(x)
    return outputs
