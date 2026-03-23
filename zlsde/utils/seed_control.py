"""Random seed control for deterministic reproducibility."""

import random
import numpy as np
import torch
import os


def set_random_seed(seed: int) -> None:
    """
    Set random seed for all libraries to ensure reproducibility.
    
    This function sets seeds for:
    - Python's random module
    - NumPy's random number generator
    - PyTorch (CPU and CUDA)
    - Environment variables for deterministic behavior
    
    Args:
        seed: Random seed value (integer)
    
    Example:
        >>> set_random_seed(42)
        >>> # All subsequent random operations will be deterministic
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Enable deterministic behavior for CUDA operations
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Set environment variables for additional determinism
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_random_state() -> dict:
    """
    Capture current random state for all libraries.
    
    Returns:
        Dictionary containing random states for Python, NumPy, and PyTorch
    
    Example:
        >>> state = get_random_state()
        >>> # ... perform random operations ...
        >>> restore_random_state(state)  # Restore to previous state
    """
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }
    
    if torch.cuda.is_available():
        state['torch_cuda'] = torch.cuda.get_rng_state_all()
    
    return state


def restore_random_state(state: dict) -> None:
    """
    Restore random state for all libraries.
    
    Args:
        state: Dictionary containing random states (from get_random_state())
    
    Example:
        >>> state = get_random_state()
        >>> # ... perform random operations ...
        >>> restore_random_state(state)  # Restore to previous state
    """
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    
    if 'torch_cuda' in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['torch_cuda'])
