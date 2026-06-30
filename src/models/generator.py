import os
import sys

# Add the project root to python path to run script directly
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from tensorflow.keras import Model
from setup.train_config import LATENT_DIM
from src.models.generator_input import build_generator_input
from src.models.generator_reshape import reshape_to_feature_map
from src.models.generator_block1 import generator_block1
from src.models.generator_block2 import generator_block2
from src.models.generator_block3 import generator_block3

def build_generator(latent_dim=LATENT_DIM):
    """
    Assembles the full generator network from input to output blocks.

    Args:
        latent_dim (int): Dimension of the latent input space (default: LATENT_DIM).

    Returns:
        Model: A Keras Model instance representing the generator.
    """
    inputs, x = build_generator_input(latent_dim)
    x = reshape_to_feature_map(x)
    x = generator_block1(x)
    x = generator_block2(x)
    outputs = generator_block3(x)
    model = Model(inputs, outputs, name="generator")
    return model

if __name__ == "__main__":
    model = build_generator()
    model.summary()
    print(f"Output shape: {model.output_shape}")
