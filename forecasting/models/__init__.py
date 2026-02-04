"""
Model wrappers for non-sklearn backends.
"""

import config


def get_torch_device():
    """
    Get the appropriate torch device based on config.USE_GPU setting.
    
    Returns 'cuda' only if:
    - config.USE_GPU is True, OR
    - config.USE_GPU is None (auto-detect) AND CUDA is available
    
    This ensures all models respect the same GPU configuration and avoids
    GPU memory contention when running with joblib parallelization.
    """
    import torch
    
    use_gpu = getattr(config, "USE_GPU", None)
    
    if use_gpu is False:
        return torch.device("cpu")
    if use_gpu is True:
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            # Requested GPU but not available - fall back to CPU with warning
            import warnings
            warnings.warn("USE_GPU=True but CUDA not available, falling back to CPU")
            return torch.device("cpu")
    
    # Auto-detect (USE_GPU is None)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_accelerator_for_lightning():
    """
    Get accelerator string for PyTorch Lightning trainer.
    Respects config.USE_GPU setting.
    """
    import torch
    
    use_gpu = getattr(config, "USE_GPU", None)
    
    if use_gpu is False:
        return "cpu"
    if use_gpu is True:
        return "gpu" if torch.cuda.is_available() else "cpu"
    
    # Auto-detect
    return "gpu" if torch.cuda.is_available() else "cpu"
