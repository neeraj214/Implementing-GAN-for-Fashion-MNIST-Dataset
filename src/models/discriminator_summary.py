import os
import sys

# Add the project root to python path to run script directly
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

import tensorflow as tf
import numpy as np
from pathlib import Path
from setup.train_config import PLOTS_DIR, LATENT_DIM
from src.models.discriminator import build_discriminator
from src.models.generator import build_generator

def main():
    # Build discriminator model
    discriminator = build_discriminator()

    # Print full model.summary()
    discriminator.summary()

    # Print total trainable parameters count
    trainable_count = int(np.sum([np.prod(w.shape) for w in discriminator.trainable_weights]))
    print(f"Total Trainable Parameters: {trainable_count:,}")

    # Save architecture diagram
    plots_dir_path = Path(PLOTS_DIR)
    plots_dir_path.mkdir(parents=True, exist_ok=True)
    
    try:
        tf.keras.utils.plot_model(
            discriminator,
            to_file=plots_dir_path / 'discriminator_architecture.png',
            show_shapes=True,
            show_layer_names=True
        )
        print("Saved discriminator architecture plot to discriminator_architecture.png")
    except Exception as e:
        print(f"Could not plot model architecture: {e}")

    # Sanity test using a generated fake image from the Generator
    generator = build_generator()
    noise = tf.random.normal([1, LATENT_DIM])
    fake_image = generator(noise, training=False)
    
    # Pass the generated image to the discriminator
    decision = discriminator(fake_image, training=False)
    print(f"Sanity Check: Discriminator raw logit output for fake image: {decision.numpy()}")

    # Print success message with detailed metadata info
    print(f"Discriminator successfully built!")
    print(f"Expected Input Shape : {discriminator.input_shape}")
    print(f"Expected Output Shape: {discriminator.output_shape}")

if __name__ == "__main__":
    main()
