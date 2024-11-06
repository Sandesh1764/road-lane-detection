# utils.py

import torch
import random
import numpy as np

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device():
    """Return the available device."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
