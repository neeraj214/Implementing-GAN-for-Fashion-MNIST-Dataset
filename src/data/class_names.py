# pyrefly: ignore [missing-import]
import numpy as np
from typing import Union, List, Dict

CLASS_NAMES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

def get_class_name(label_idx: int) -> str:
    """
    Returns the class name for a given label index.
    
    Args:
        label_idx: Integer index of the class.
        
    Returns:
        String name of the class.
    """
    try:
        return CLASS_NAMES[int(label_idx)]
    except IndexError:
        raise ValueError(f"Label index {label_idx} is out of range.")

def get_class_distribution(y: Union[np.ndarray, List[int]]) -> Dict[str, int]:
    """
    Computes the distribution of classes in a label array.
    
    Args:
        y: A list or numpy array of class labels.
        
    Returns:
        A dictionary mapping class names to their frequencies.
    """
    unique, counts = np.unique(y, return_counts=True)
    distribution = {}
    for label, count in zip(unique, counts):
        class_name = get_class_name(int(label))
        distribution[class_name] = int(count)
    return distribution
