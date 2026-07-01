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
    # 1. Instantiate the Input Layer (default shape: 28x28x1)
    inputs = build_discriminator_input(image_size, channels)
    
    # 2. First Conv Block: downsample 28x28 -> 14x14, extracts 64 filters
    x = discriminator_block1(inputs)
    
    # 3. Second Conv Block: downsample 14x14 -> 7x7, extracts 128 filters
    x = discriminator_block2(x)
    
    # 4. Flatten spatial feature maps (7x7x128 = 6272 flat size)
    x = flatten_features(x)
    
    # 5. Output Layer: single dense classification logit (real vs fake)
    outputs = discriminator_output(x)
    
    model = Model(inputs, outputs, name="discriminator")
    return model

if __name__ == "__main__":
    model = build_discriminator()
    model.summary()
    print(f"Output shape: {model.output_shape}")
