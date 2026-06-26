import os
import sys
import tensorflow as tf
import numpy as np

# Add the project root to python path to import train_config
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from setup.train_config import BATCH_SIZE, BASE_DIR as CONFIG_BASE_DIR

def get_train_dataset() -> tf.data.Dataset:
    """
    Loads the normalized Fashion-MNIST training images and creates a tf.data.Dataset pipeline.
    
    The pipeline performs the following steps:
    1. Loads the pre-processed numpy array of images from data/processed/X_train_normalized.npy.
    2. Converts it into a tf.data.Dataset.
    3. Shuffles the dataset.
    4. Batches the dataset, dropping remainder batches to ensure consistent batch sizes.
    5. Prefetches batches to optimize training performance.
    
    Returns:
        tf.data.Dataset: The configured pipeline for training batches.
    """
    # Path to the processed numpy array
    processed_path = os.path.join(CONFIG_BASE_DIR, 'data', 'processed', 'X_train_normalized.npy')
    
    # Load data/processed/X_train_normalized.npy
    images = np.load(processed_path)
    
    # Create tf.data.Dataset.from_tensor_slices(images)
    dataset = tf.data.Dataset.from_tensor_slices(images)
    
    # .shuffle(buffer_size=60000)
    dataset = dataset.shuffle(buffer_size=60000)
    
    # .batch(BATCH_SIZE, drop_remainder=True)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    
    # .prefetch(tf.data.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

if __name__ == "__main__":
    # Print number of batches: 60000 // BATCH_SIZE
    num_batches = 60000 // BATCH_SIZE
    print(f"Number of batches: {num_batches}")
    
    # Get dataset
    dataset = get_train_dataset()
    
    # Print one batch shape as sanity check using dataset.take(1)
    for batch in dataset.take(1):
        print(f"Batch shape: {batch.shape}")
