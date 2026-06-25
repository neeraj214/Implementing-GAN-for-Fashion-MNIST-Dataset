import os
import sys
# pyrefly: ignore [missing-import]
import numpy as np
import tensorflow as tf

# Add the project root to python path to import train_config
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from setup.train_config import BASE_DIR as CONFIG_BASE_DIR

def download_and_save_data():
    raw_data_dir = os.path.join(CONFIG_BASE_DIR, 'data', 'raw')
    os.makedirs(raw_data_dir, exist_ok=True)

    # Load from keras
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    # Save as .npy files
    np.save(os.path.join(raw_data_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(raw_data_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(raw_data_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(raw_data_dir, 'y_test.npy'), y_test)

    # Print shapes
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    # Confirmation
    print("Fashion-MNIST downloaded and cached")

if __name__ == "__main__":
    download_and_save_data()
