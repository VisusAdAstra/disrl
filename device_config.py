"""
Device configuration and GPU utilities.
"""

import torch


def get_device(device_preference='auto'):
    """
    Get the best available device for training.
    
    Args:
        device_preference: 'auto', 'cuda', 'mps', or 'cpu'
    
    Returns:
        torch.device object
    """
    if device_preference == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
            print("🚀 Using Apple Silicon GPU (MPS)")
        else:
            device = torch.device('cpu')
            print("💻 Using CPU (no GPU detected)")
    else:
        device = torch.device(device_preference)
        if device_preference == 'cuda':
            if torch.cuda.is_available():
                print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                print("⚠️  CUDA requested but not available, falling back to CPU")
                device = torch.device('cpu')
        elif device_preference == 'mps':
            if torch.backends.mps.is_available():
                print("🚀 Using Apple Silicon GPU (MPS)")
            else:
                print("⚠️  MPS requested but not available, falling back to CPU")
                device = torch.device('cpu')
        else:
            print(f"💻 Using {device_preference.upper()}")
    
    return device


def print_gpu_memory_usage():
    """Print current GPU memory usage if CUDA is available."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"\nGPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")


def clear_gpu_memory():
    """Clear GPU cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU cache cleared")
