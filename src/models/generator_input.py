from tensorflow.keras.layers import Input, Dense, BatchNormalization, LeakyReLU
from setup.train_config import LATENT_DIM

def build_generator_input(latent_dim=LATENT_DIM):
    """
    Projects a latent vector into a 7x7x256 feature map.

    Args:
        latent_dim (int): The dimension of the latent space (default: LATENT_DIM).

    Returns:
        tuple: A tuple containing:
            - inputs: The Input tensor of shape (latent_dim,).
            - x: The dense projection tensor, batch normalized and activated with LeakyReLU,
                 of shape (7 * 7 * 256,).
    """
    inputs = Input(shape=(latent_dim,), name="latent_input")
    x = Dense(7*7*256, use_bias=False, name="dense_projection")(inputs)
    x = BatchNormalization(name="bn_input")(x)
    x = LeakyReLU(name="leakyrelu_input")(x)
    return inputs, x
