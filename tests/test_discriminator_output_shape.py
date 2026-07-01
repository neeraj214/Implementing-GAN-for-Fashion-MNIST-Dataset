import os
import sys

# Add the project root to python path to run test directly
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

import tensorflow as tf
import numpy as np
from setup.train_config import IMAGE_SIZE, CHANNELS
from src.models.discriminator import build_discriminator

def test_output_shape():
    model = build_discriminator()
    images = tf.random.normal([4, IMAGE_SIZE, IMAGE_SIZE, CHANNELS])
    output = model(images, training=False)
    assert output.shape == (4, 1), f"Expected shape (4, 1), got {output.shape}"

def test_trainable_params():
    model = build_discriminator()
    assert model.count_params() > 0, "Model parameter count should be greater than 0"

if __name__ == "__main__":
    tests = [
        ("test_output_shape", test_output_shape),
        ("test_trainable_params", test_trainable_params)
    ]
    
    all_passed = True
    for name, test_func in tests:
        try:
            test_func()
            print(f"PASS: {name}")
        except Exception as e:
            print(f"FAIL: {name} - {e}")
            all_passed = False
            
    if all_passed:
        print("\nAll discriminator tests passed")
    else:
        print("\nSome tests failed")
        sys.exit(1)
