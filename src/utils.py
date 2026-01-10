# Utility functions 
import numpy as np
import torch


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def sample_uniform(n: int, low: float, high: float, device):
    return (low + (high - low) * torch.rand(n, 1, device=device))
