import os
import sys

# Add the project root to python path to run test directly
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

import tensorflow as tf
import numpy as np
from setup.train_config import LATENT_DIM
from src.models.generator import build_generator

def test_output_shape():
    model = build_generator()
    noise = tf.random.normal([4, LATENT_DIM])
    output = model(noise, training=False)
    assert output.shape == (4, 28, 28, 1), f"Expected shape (4, 28, 28, 1), got {output.shape}"

def test_output_range():
    model = build_generator()
    noise = tf.random.normal([4, LATENT_DIM])
    output = model(noise, training=False).numpy()
    assert np.all(output >= -1.0) and np.all(output <= 1.0), "Output values not within [-1, 1]"

def test_trainable_params():
    model = build_generator()
    assert model.count_params() > 0, "Model parameter count should be greater than 0"

if __name__ == "__main__":
    tests = [
        ("test_output_shape", test_output_shape),
        ("test_output_range", test_output_range),
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
        print("\nAll generator tests passed")
    else:
        print("\nSome tests failed")
        sys.exit(1)
