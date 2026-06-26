import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

# Add the project root to python path to import setup and src modules
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from setup import train_config as config
from src.data.class_names import get_class_distribution

def main():
    # Load data/raw/y_train.npy
    y_train_path = os.path.join(config.BASE_DIR, 'data', 'raw', 'y_train.npy')
    if not os.path.exists(y_train_path):
        print(f"Error: Raw file not found at {y_train_path}. Please run download_data.py first.")
        return

    y_train = np.load(y_train_path)

    # Compute distribution dict using get_class_distribution()
    dist_dict = get_class_distribution(y_train)

    # Print distribution table in terminal
    print("\nClass Distribution Table:")
    print("-" * 35)
    print(f"{'Class Name':<20} | {'Count':<10}")
    print("-" * 35)
    for class_name, count in dist_dict.items():
        print(f"{class_name:<20} | {count:<10}")
    print("-" * 35 + "\n")

    # Plot horizontal bar chart: class name vs count
    classes = list(dist_dict.keys())
    counts = list(dist_dict.values())

    plt.figure(figsize=(10, 6))
    # Horizontal bar chart
    bars = plt.barh(classes, counts, color='skyblue', edgecolor='black')
    plt.xlabel('Count')
    plt.ylabel('Class Name')
    plt.title('Fashion-MNIST Class Distribution (y_train)')
    
    # Add values on the bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width - 500 if width > 500 else width + 50, bar.get_y() + bar.get_height()/2, 
                 f'{width}', 
                 va='center', ha='right' if width > 500 else 'left', fontsize=10, fontweight='bold')

    plt.gca().invert_yaxis()  # top-down order
    plt.tight_layout()

    # Save as outputs/plots/class_distribution.png
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    plot_path = os.path.join(config.PLOTS_DIR, 'class_distribution.png')
    plt.savefig(plot_path, bbox_inches='tight', dpi=150)
    plt.close()

    # Save distribution dict as outputs/metrics/class_distribution.json
    os.makedirs(config.METRICS_DIR, exist_ok=True)
    json_path = os.path.join(config.METRICS_DIR, 'class_distribution.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(dist_dict, f, indent=4)

    print(f"Saved distribution chart to {plot_path}")
    print(f"Saved distribution JSON to {json_path}")

if __name__ == "__main__":
    main()
