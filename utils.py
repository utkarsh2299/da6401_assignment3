import matplotlib.pyplot as plt
import torch
import random
import numpy as np
import os
from typing import Dict, List, Tuple


def set_seed(seed: int):
    """
    Set random seed for reproducibility
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in the model
    
    Args:
        model: Model to count parameters for
        
    Returns:
        count: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_loss_curves(history: Dict[str, List[float]], save_path: str = None):
    """
    Plot the training and validation loss curves
    
    Args:
        history: Dictionary containing training and validation loss history
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['valid_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()


def calculate_accuracy(predictions: List[str], targets: List[str]) -> float:
    """
    Calculate accuracy of predictions
    
    Args:
        predictions: List of predicted strings
        targets: List of target strings
        
    Returns:
        accuracy: Accuracy (exact match)
    """
    correct = sum(1 for p, t in zip(predictions, targets) if p == t)
    return correct / len(targets)


def display_sample_predictions(samples: List[Tuple[str, str, str]], n: int = 10):
    """
    Display sample predictions
    
    Args:
        samples: List of sample (source, prediction, target) tuples
        n: Number of samples to display
    """
    samples = samples[:n]
    
    for i, (src, pred, trg) in enumerate(samples):
        print(f"Sample {i+1}:")
        print(f"Source: {src}")
        print(f"Prediction: {pred}")
        print(f"Target: {trg}")
        print(f"Correct: {'Yes' if pred == trg else 'No'}")
        print("-" * 50)
