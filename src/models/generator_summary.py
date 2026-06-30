import os
import sys

# Add the project root to python path to run script directly
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from setup.train_config import PLOTS_DIR, LATENT_DIM
from src.models.generator import build_generator

def main():
    # Build the model
    model = build_generator()

    # Print full model.summary()
    model.summary()

    # Print total trainable parameters count
    trainable_count = int(np.sum([np.prod(w.shape) for w in model.trainable_weights]))
    print(f"Total Trainable Parameters: {trainable_count:,}")

    # Save architecture diagram
    plots_dir_path = Path(PLOTS_DIR)
    plots_dir_path.mkdir(parents=True, exist_ok=True)
    
    try:
        tf.keras.utils.plot_model(
            model,
            to_file=plots_dir_path / 'generator_architecture.png',
            show_shapes=True,
            show_layer_names=True
        )
        print("Saved generator architecture plot to generator_architecture.png")
    except Exception as e:
        print(f"Could not plot model architecture: {e}")

    # Generate one test image
    noise = tf.random.normal([1, LATENT_DIM])
    test_output = model(noise, training=False)

    # Plot the raw test output (random weights, just to confirm shape works)
    plt.figure(figsize=(4, 4))
    plt.imshow(test_output[0, :, :, 0], cmap='gray')
    plt.title("Random Generator Output (Untrained)")
    plt.axis('off')
    
    test_output_plot_path = plots_dir_path / 'generator_random_output_test.png'
    plt.savefig(test_output_plot_path, bbox_inches='tight')
    plt.close()
    print(f"Saved random generator test output to {test_output_plot_path}")

    # Print validation message
    print(f"Generator built successfully - output shape: {model.output_shape}")

if __name__ == "__main__":
    main()
