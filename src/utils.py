"""
Utility functions for the project
"""
import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

def save_pickle(obj, path):
    """Save object to pickle file"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    print(f"Saved: {path}")

def load_pickle(path):
    """Load object from pickle file"""
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_json(data, path):
    """Save data to JSON file"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Saved: {path}")

def load_json(path):
    """Load data from JSON file"""
    with open(path, 'r') as f:
        return json.load(f)

def save_numpy(arr, path):
    """Save numpy array"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, arr)
    print(f"Saved: {path}")

def load_numpy(path):
    """Load numpy array"""
    return np.load(path)

def save_csv(df, path, index=True):
    """Save DataFrame to CSV"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=index)
    print(f"Saved: {path}")

def save_figure(fig, path, dpi=150):
    """Save matplotlib figure"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close(fig)

def save_model(model, optimizer, epoch, loss, path):
    """Save model checkpoint"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)
    print(f"Model saved: {path}")

def load_model(model, path, optimizer=None):
    """Load model checkpoint"""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']

def print_gpu_info():
    """Print GPU information"""
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"Memory Cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    else:
        print("GPU not available, using CPU")

def format_time(seconds):
    """Format seconds to HH:MM:SS"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def count_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
