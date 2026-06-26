import os
import sys
import json
import numpy as np
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt

# Add the project root to python path to import setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from setup import train_config as config

def main():
    # Load data/raw/X_train.npy
    x_train_path = os.path.join(config.BASE_DIR, 'data', 'raw', 'X_train.npy')
    if not os.path.exists(x_train_path):
        print(f"Error: Raw file not found at {x_train_path}. Please run download_data.py first.")
        return

    X_train = np.load(x_train_path)

    # Compute stats
    mean_val = float(np.mean(X_train))
    std_val = float(np.std(X_train))
    min_val = int(np.min(X_train))
    max_val = int(np.max(X_train))

    # Histogram of pixel intensities (0-255)
    counts, bin_edges = np.histogram(X_train, bins=256, range=(0, 256))

    # Print stats
    print(f"Mean pixel value: {mean_val:.4f}")
    print(f"Std pixel value: {std_val:.4f}")
    print(f"Min pixel value: {min_val}")
    print(f"Max pixel value: {max_val}")
    print("\nHistogram of pixel intensities (first 10 bins):")
    for i in range(10):
        print(f"Intensity {i}: {counts[i]}")
    print("...")
    print(f"Histogram array (all 256 bins):\n{counts}\n")

    # Plot pixel intensity histogram
    plt.figure(figsize=(10, 5))
    plt.hist(X_train.ravel(), bins=256, range=(0, 256), color='gray', alpha=0.75, edgecolor='black', linewidth=0.2)
    plt.xlabel('Pixel Intensity (0-255)')
    plt.ylabel('Frequency')
    plt.title('Fashion-MNIST Raw Pixel Intensity Histogram')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    # Save plot as outputs/plots/pixel_histogram.png
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    plot_path = os.path.join(config.PLOTS_DIR, 'pixel_histogram.png')
    plt.savefig(plot_path, bbox_inches='tight', dpi=150)
    plt.close()

    # Save stats dict as outputs/metrics/pixel_stats.json
    os.makedirs(config.METRICS_DIR, exist_ok=True)
    stats_dict = {
        "mean": mean_val,
        "std": std_val,
        "min": min_val,
        "max": max_val
    }
    json_path = os.path.join(config.METRICS_DIR, 'pixel_stats.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(stats_dict, f, indent=4)

    print(f"Saved histogram plot to {plot_path}")
    print(f"Saved pixel stats JSON to {json_path}")

if __name__ == "__main__":
    main()
