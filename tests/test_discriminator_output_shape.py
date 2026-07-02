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

def test_real_image_input():
    model = build_discriminator()
    # Create real-shaped normalized images (e.g. range [-1, 1])
    images = tf.random.uniform([4, IMAGE_SIZE, IMAGE_SIZE, CHANNELS], minval=-1.0, maxval=1.0)
    output = model(images, training=False)
    assert output.shape == (4, 1), f"Expected shape (4, 1), got {output.shape}"

def test_trainable_params():
    model = build_discriminator()
    assert model.count_params() > 0, "Model parameter count should be greater than 0"

def test_logit_output_unbounded():
    model = build_discriminator()
    # Output layer must not have sigmoid activation (must be linear/none)
    output_layer = model.get_layer("real_fake_logit")
    activation_name = output_layer.activation.__name__
    assert activation_name == "linear", f"Output layer activation should be linear (no sigmoid), got {activation_name}"
    
    # Check that outputs can actually fall outside the [0, 1] range
    images = tf.random.normal([100, IMAGE_SIZE, IMAGE_SIZE, CHANNELS])
    output = model(images, training=False).numpy()
    has_outside_range = np.any((output < 0.0) | (output > 1.0))
    assert has_outside_range, "Logits should not be constrained to [0, 1]"

if __name__ == "__main__":
    tests = [
        ("test_output_shape", test_output_shape),
        ("test_real_image_input", test_real_image_input),
        ("test_trainable_params", test_trainable_params),
        ("test_logit_output_unbounded", test_logit_output_unbounded)
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
