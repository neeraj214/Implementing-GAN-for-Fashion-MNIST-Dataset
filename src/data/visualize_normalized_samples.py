import os
import sys
import numpy as np
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt

# Add the project root to python path to import setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from setup import train_config as config

def main():
    # Load data/processed/X_train_normalized.npy
    processed_path = os.path.join(config.BASE_DIR, 'data', 'processed', 'X_train_normalized.npy')
    if not os.path.exists(processed_path):
        print(f"Error: Processed file not found at {processed_path}. Please run download_data.py and normalize.py first.")
        return

    X_train_normalized = np.load(processed_path)

    # Check min/max of the loaded array
    min_val = X_train_normalized.min()
    max_val = X_train_normalized.max()
    print(f"Normalized array stats: min = {min_val:.4f}, max = {max_val:.4f}")
    if np.isclose(min_val, -1.0, atol=0.1) and np.isclose(max_val, 1.0, atol=0.1):
        print("Confirmation: Normalization is correct (range is roughly [-1, 1]).")
    else:
        print("Warning: Normalized values are outside the expected [-1, 1] range!")

    # Set seed for reproducibility
    np.random.seed(config.SEED)

    # Select 16 random samples
    indices = np.random.choice(len(X_train_normalized), 16, replace=False)
    samples = X_train_normalized[indices]

    # Plot 4x4 grid
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    fig.suptitle("Normalized Samples (Range: -1 to 1)", fontsize=14, y=0.95)

    for idx, ax in enumerate(axes.ravel()):
        # Use imshow with cmap='gray', vmin=-1, vmax=1
        ax.imshow(samples[idx].squeeze(), cmap='gray', vmin=-1, vmax=1)
        ax.axis('off')

    plt.tight_layout()

    # Save as outputs/plots/normalized_samples_grid.png
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    save_path = os.path.join(config.PLOTS_DIR, 'normalized_samples_grid.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()

    print(f"Saved normalized samples grid to {save_path}")

if __name__ == "__main__":
    main()
