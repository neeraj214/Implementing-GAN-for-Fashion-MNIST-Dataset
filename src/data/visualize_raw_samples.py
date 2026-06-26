import os
import sys
import numpy as np
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt

# Add the project root to python path to import setup and src modules
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from setup import train_config as config
from src.data.class_names import CLASS_NAMES

def visualize_raw_samples():
    # Load data/raw/X_train.npy and y_train.npy
    x_train_path = os.path.join(config.BASE_DIR, 'data', 'raw', 'X_train.npy')
    y_train_path = os.path.join(config.BASE_DIR, 'data', 'raw', 'y_train.npy')
    
    if not os.path.exists(x_train_path) or not os.path.exists(y_train_path):
        print(f"Error: Raw files not found at {x_train_path} or {y_train_path}. Please run download_data.py first.")
        return

    X_train = np.load(x_train_path)
    y_train = np.load(y_train_path)

    # Set seed for reproducibility
    np.random.seed(config.SEED)

    # Plot 5x10 grid: 5 random samples per class, grayscale cmap
    fig, axes = plt.subplots(5, 10, figsize=(15, 8))
    
    for class_idx in range(10):
        # Find indices of samples belonging to class_idx
        class_indices = np.where(y_train == class_idx)[0]
        # Randomly select 5 indices
        selected_indices = np.random.choice(class_indices, 5, replace=False)
        
        for row_idx, sample_idx in enumerate(selected_indices):
            ax = axes[row_idx, class_idx]
            img = X_train[sample_idx]
            
            # Plot grayscale cmap
            ax.imshow(img, cmap='gray')
            ax.axis('off')
            
            # Title each subplot with class name
            ax.set_title(CLASS_NAMES[class_idx], fontsize=8)

    plt.tight_layout()
    
    # Save as outputs/plots/raw_samples_grid.png
    save_path = os.path.join(config.PLOTS_DIR, 'raw_samples_grid.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    # Print "Saved raw sample visualization"
    print("Saved raw sample visualization")

if __name__ == "__main__":
    visualize_raw_samples()
