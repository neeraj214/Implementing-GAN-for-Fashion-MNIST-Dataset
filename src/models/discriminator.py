import os
import sys

# Add the project root to python path to run script directly
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from tensorflow.keras import Model
from setup.train_config import IMAGE_SIZE, CHANNELS
from src.models.discriminator_input import build_discriminator_input
from src.models.discriminator_block1 import discriminator_block1
from src.models.discriminator_block2 import discriminator_block2
from src.models.discriminator_flatten import flatten_features
from src.models.discriminator_output import discriminator_output

def build_discriminator(image_size=IMAGE_SIZE, channels=CHANNELS):
    """
    Assembles the full discriminator network from input to output blocks.

    Args:
        image_size (int): Height and width of input images (default: IMAGE_SIZE).
        channels (int): Number of input image channels (default: CHANNELS).

    Returns:
        Model: A Keras Model instance representing the discriminator.
    """
    inputs = build_discriminator_input(image_size, channels)
    x = discriminator_block1(inputs)
    x = discriminator_block2(x)
    x = flatten_features(x)
    outputs = discriminator_output(x)
    model = Model(inputs, outputs, name="discriminator")
    return model

if __name__ == "__main__":
    model = build_discriminator()
    model.summary()
    print(f"Output shape: {model.output_shape}")
