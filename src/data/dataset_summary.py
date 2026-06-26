import os
import sys
import json

# Add the project root to python path to import setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from setup import train_config as config

def main():
    class_dist_path = os.path.join(config.METRICS_DIR, 'class_distribution.json')
    pixel_stats_path = os.path.join(config.METRICS_DIR, 'pixel_stats.json')

    # Read outputs/metrics/class_distribution.json if exists
    if os.path.exists(class_dist_path):
        with open(class_dist_path, 'r', encoding='utf-8') as f:
            class_distribution = json.load(f)
    else:
        print(f"Warning: Class distribution metrics not found at {class_dist_path}")
        class_distribution = {}

    # Read outputs/metrics/pixel_stats.json if exists
    if os.path.exists(pixel_stats_path):
        with open(pixel_stats_path, 'r', encoding='utf-8') as f:
            pixel_stats = json.load(f)
    else:
        print(f"Warning: Pixel stats metrics not found at {pixel_stats_path}")
        pixel_stats = {}

    # Build combined summary dict
    summary_dict = {
        "total_train_samples": 60000,
        "total_test_samples": 10000,
        "image_shape": [28, 28, 1],
        "num_classes": 10,
        "normalization_range": [-1, 1],
        "class_distribution": class_distribution,
        "pixel_stats": pixel_stats
    }

    # Save as outputs/metrics/dataset_summary.json
    os.makedirs(config.METRICS_DIR, exist_ok=True)
    summary_path = os.path.join(config.METRICS_DIR, 'dataset_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_dict, f, indent=4)

    # Print formatted summary report in terminal
    print("\n" + "=" * 50)
    print("             DATASET METADATA SUMMARY             ")
    print("=" * 50)
    print(f"Total Train Samples  : {summary_dict['total_train_samples']:,}")
    print(f"Total Test Samples   : {summary_dict['total_test_samples']:,}")
    print(f"Image Shape          : {summary_dict['image_shape']}")
    print(f"Number of Classes    : {summary_dict['num_classes']}")
    print(f"Normalization Range  : {summary_dict['normalization_range']}")
    print("-" * 50)
    
    print("Pixel Stats (Raw):")
    if pixel_stats:
        print(f"  Mean               : {pixel_stats.get('mean', 'N/A'):.4f}")
        print(f"  Std Dev            : {pixel_stats.get('std', 'N/A'):.4f}")
        print(f"  Min                : {pixel_stats.get('min', 'N/A')}")
        print(f"  Max                : {pixel_stats.get('max', 'N/A')}")
    else:
        print("  No pixel stats available (run pixel_stats.py first)")
        
    print("-" * 50)
    print("Class Distribution:")
    if class_distribution:
        for cls_name, count in class_distribution.items():
            print(f"  {cls_name:<18} : {count:,}")
    else:
        print("  No class distribution available (run plot_class_distribution.py first)")
    print("=" * 50 + "\n")

    print(f"Saved dataset summary JSON to {summary_path}")

if __name__ == "__main__":
    main()
