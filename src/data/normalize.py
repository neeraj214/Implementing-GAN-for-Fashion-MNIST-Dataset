import os
import sys
# pyrefly: ignore [missing-import]
import numpy as np

# Add the project root to python path to import train_config
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from setup.train_config import BASE_DIR as CONFIG_BASE_DIR

def normalize_for_gan(images: np.ndarray) -> np.ndarray:
    """
    Normalizes images for GAN training:
    - Cast to float32
    - Reshape to (-1, 28, 28, 1)
    - Scale to [-1, 1] range (GAN convention, since generator uses tanh)
    
    Args:
        images: Input images numpy array.
        
    Returns:
        Normalized images numpy array.
    """
    images = images.astype(np.float32)
    images = np.reshape(images, (-1, 28, 28, 1))
    images = (images - 127.5) / 127.5
    return images

def main():
    raw_path = os.path.join(CONFIG_BASE_DIR, 'data', 'raw', 'X_train.npy')
    processed_dir = os.path.join(CONFIG_BASE_DIR, 'data', 'processed')
    os.makedirs(processed_dir, exist_ok=True)

    if not os.path.exists(raw_path):
        print(f"Error: Raw file {raw_path} not found. Make sure to run download_data.py first.")
        return

    # Load raw data
    X_train = np.load(raw_path)

    # Print stats before
    print(f"Before normalization: min = {X_train.min()}, max = {X_train.max()}")

    # Normalize
    X_train_normalized = normalize_for_gan(X_train)

    # Save processed array
    processed_path = os.path.join(processed_dir, 'X_train_normalized.npy')
    np.save(processed_path, X_train_normalized)

    # Print stats after
    print(f"After normalization: min = {X_train_normalized.min()}, max = {X_train_normalized.max()}")
    print(f"Final shape: {X_train_normalized.shape}")

if __name__ == "__main__":
    main()
