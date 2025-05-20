import matplotlib.pyplot as plt
import torch
import random
import numpy as np
import os
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
import matplotlib.font_manager as fm


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


# def visualize_attention(model: nn.Module, 
#                         test_dataset: torch.utils.data.Dataset,
#                         iterator: torch.utils.data.DataLoader, 
#                         device: torch.device,
#                         n_examples: int = 9,
#                         n_rows: int = 3,
#                         n_cols: int = 3,
#                         save_path: str = 'attention_heatmaps.png'):
#     """
#     Generate and visualize attention heatmaps for test examples
    
#     Args:
#         model: Model to use
#         test_dataset: Test dataset
#         iterator: DataLoader for test data
#         device: Device to use
#         n_examples: Number of examples to visualize
#         n_rows: Number of rows in the grid
#         n_cols: Number of columns in the grid
#         save_path: Path to save the heatmap visualization
#     """
    
#     import matplotlib as mpl
#     import matplotlib.font_manager as fm
#     # fm._rebuild()
#     font_path = "./Nirmala.ttf"  # Path to a font that supports Devanagari
#     # font_prop = fm.FontProperties(fname=font_path)
#     mpl.rcParams['font.family'] = 'Nirmala UI'

    
#     model.eval()
#     examples = []
    
#     with torch.no_grad():
#         # Get some test examples
#         for src, trg in iterator:
#             src = src.to(device)
#             trg = trg.to(device)
            
#             batch_size = src.shape[0]
            
#             for j in range(batch_size):
#                 if len(examples) >= n_examples:
#                     break
                    
#                 src_tensor = src[j].unsqueeze(0)
#                 trg_tensor = trg[j].unsqueeze(0)
                
#                 # Handle both models with and without attention
#                 try:
#                     # Try unpacking two return values (model with attention)
#                     output_indices, attention_weights = model.greedy_decode(src_tensor, max_len=100)
#                 except ValueError:
#                     # If that fails, model probably doesn't have attention
#                     output_indices = model.greedy_decode(src_tensor, max_len=100)
#                     # Create dummy uniform attention weights for visualization
#                     src_len = src_tensor.shape[1]
#                     output_len = len(output_indices) if isinstance(output_indices, list) else 1
#                     attention_weights = [torch.ones(src_len) / src_len for _ in range(output_len)]
#                     print("Using uniform attention weights (model may not have attention)")
                
#                 # Handle case where output_indices is a single integer instead of a list
#                 if isinstance(output_indices, int):
#                     output_indices = [output_indices]
                
#                 # Convert indices to text
#                 src_text = test_dataset.decode_latin(src_tensor.squeeze().cpu().numpy().tolist())
#                 trg_text = test_dataset.decode_devanagari(trg_tensor.squeeze().cpu().numpy().tolist())
                
#                 try:
#                     pred_text = test_dataset.decode_devanagari(output_indices)
#                 except TypeError as e:
#                     print(f"Error decoding indices: {e}")
#                     print(f"Output indices type: {type(output_indices)}, value: {output_indices}")
#                     # Try to handle various possible formats
#                     if hasattr(output_indices, 'tolist'):
#                         output_indices = output_indices.tolist()
#                     elif not isinstance(output_indices, list):
#                         output_indices = [output_indices]
#                     pred_text = test_dataset.decode_devanagari(output_indices)
                
#                 # Check if prediction is correct
#                 is_correct = (pred_text == trg_text)
                
#                 # Get individual characters for visualization
#                 try:
#                     pred_chars = list(pred_text)
#                 except:
#                     print(f"Error converting pred_text to characters. pred_text: {pred_text}")
#                     pred_chars = [str(idx) for idx in output_indices]
                
#                 # Store example with attention weights
#                 examples.append({
#                     'src_text': src_text,
#                     'pred_text': pred_text,
#                     'trg_text': trg_text,
#                     'is_correct': is_correct,
#                     'attention_weights': attention_weights,
#                     'src_chars': list(src_text),
#                     'pred_chars': pred_chars,
#                     'output_indices': output_indices
#                 })
                
#             if len(examples) >= n_examples:
#                 break
    
#     # Create a figure with subplots for each example
#     fig = plt.figure(figsize=(16, 12))
    
#     # Display attention for all examples in a grid
#     for i, example in enumerate(examples):
#         if i >= n_rows * n_cols:
#             break
            
#         # Display attention heatmap
#         ax = plt.subplot(n_rows, n_cols, i+1)
        
#         # Get the attention weights and characters
#         src_chars = example['src_chars']
#         pred_chars = example['pred_chars']
        
#         # Ensure attention weights is a list of tensors
#         if not isinstance(example['attention_weights'], list):
#             print(f"Warning: attention_weights is not a list, type: {type(example['attention_weights'])}")
#             # If it's a single tensor, convert to a list with one tensor
#             attention_weights = [example['attention_weights']]
#         else:
#             attention_weights = example['attention_weights']
        
#         try:
#             attention = torch.stack(attention_weights)
#             attention = attention.cpu().numpy()
#         except:
#             print(f"Error stacking attention weights. Using dummy attention.")
#             attention = np.ones((len(pred_chars), len(src_chars))) / len(src_chars)
        
#         # Take only valid part of attention matrix
#         att_height = min(len(pred_chars), attention.shape[0])
#         att_width = min(len(src_chars), attention.shape[1])
#         attention_plot = attention[:att_height, :att_width]
        
#         # Create heatmap
#         im = ax.imshow(attention_plot, cmap='viridis', aspect='auto')
        
#         # Set labels
#         # For latin (source) characters 
#         ax.set_xticks(range(len(src_chars[:att_width])))
#         ax.set_xticklabels(src_chars[:att_width], fontsize=10, rotation=45)
        
#         # For Devanagari (prediction) characters
#         ax.set_yticks(range(len(pred_chars[:att_height])))
#         ax.set_yticklabels(pred_chars[:att_height], fontsize=10)
        
#         # Set title - green for correct, red for incorrect
#         color = 'green' if example['is_correct'] else 'red'
        
#         # Use simpler title for compatibility with different fonts
#         ax.set_title(f"Source: {example['src_text']} → Pred: {example['pred_text']}", 
#                    fontsize=10, color=color)
        
#         # Add colorbar
#         plt.colorbar(im, ax=ax, shrink=0.8)
    
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     plt.show()
    
#     print(f"Attention heatmaps saved to: {save_path}")
#     return examples



# # def visualize_attention_connectivity(model: nn.Module, 
#                         test_dataset: torch.utils.data.Dataset,
#                         iterator: torch.utils.data.DataLoader, 
#                         device: torch.device,
#                         n_examples: int = 4,
#                         save_path: str = 'attention_connectivity.png',
#                         ):
#     """
#     Generate and visualize attention heatmaps for test examples
#     showing character-level connectivity
    
#     Args:
#         model: Model to use
#         test_dataset: Test dataset
#         iterator: DataLoader for test data
#         device: Device to use
#         n_examples: Number of examples to visualize
#         save_path: Path to save the heatmap visualization
#         log_to_wandb: Whether to log the image to wandb
#     """
    
#     import matplotlib as mpl
#     import matplotlib.font_manager as fm
#     import numpy as np
    
#     # Font setup
#     mpl.rcParams['font.family'] = 'Nirmala UI'
    
#     model.eval()
#     examples = []
    
#     with torch.no_grad():
#         # Get some test examples
#         for src, trg in iterator:
#             src = src.to(device)
#             trg = trg.to(device)
            
#             batch_size = src.shape[0]
            
#             for j in range(batch_size):
#                 if len(examples) >= n_examples:
#                     break
                    
#                 src_tensor = src[j].unsqueeze(0)
#                 trg_tensor = trg[j].unsqueeze(0)
                
#                 # Handle both models with and without attention
#                 try:
#                     # Try unpacking two return values (model with attention)
#                     output_indices, attention_weights = model.greedy_decode(src_tensor, max_len=100)
#                 except ValueError:
#                     # If that fails, model probably doesn't have attention
#                     output_indices = model.greedy_decode(src_tensor, max_len=100)
#                     # Create dummy uniform attention weights for visualization
#                     src_len = src_tensor.shape[1]
#                     output_len = len(output_indices) if isinstance(output_indices, list) else 1
#                     attention_weights = [torch.ones(src_len) / src_len for _ in range(output_len)]
#                     print("Using uniform attention weights (model may not have attention)")
                
#                 # Handle case where output_indices is a single integer instead of a list
#                 if isinstance(output_indices, int):
#                     output_indices = [output_indices]
                
#                 # Convert indices to text
#                 src_text = test_dataset.decode_latin(src_tensor.squeeze().cpu().numpy().tolist())
#                 trg_text = test_dataset.decode_devanagari(trg_tensor.squeeze().cpu().numpy().tolist())
                
#                 try:
#                     pred_text = test_dataset.decode_devanagari(output_indices)
#                 except TypeError as e:
#                     print(f"Error decoding indices: {e}")
#                     if hasattr(output_indices, 'tolist'):
#                         output_indices = output_indices.tolist()
#                     elif not isinstance(output_indices, list):
#                         output_indices = [output_indices]
#                     pred_text = test_dataset.decode_devanagari(output_indices)
                
#                 # Get individual characters for visualization
#                 try:
#                     pred_chars = list(pred_text)
#                 except:
#                     print(f"Error converting pred_text to characters. pred_text: {pred_text}")
#                     pred_chars = [str(idx) for idx in output_indices]
                
#                 # Ensure attention weights is a list of tensors
#                 if not isinstance(attention_weights, list):
#                     attention_weights = [attention_weights]
                
#                 # Stack attention weights and convert to numpy
#                 try:
#                     attention = torch.stack(attention_weights)
#                     attention = attention.cpu().numpy()
#                 except:
#                     print(f"Error stacking attention weights. Using dummy attention.")
#                     attention = np.ones((len(pred_chars), len(src_text))) / len(src_text)
                
#                 # Check if prediction is correct
#                 is_correct = (pred_text == trg_text)
                
#                 # Store example
#                 examples.append({
#                     'src_text': src_text,
#                     'pred_text': pred_text,
#                     'trg_text': trg_text,
#                     'is_correct': is_correct,
#                     'attention': attention,
#                     'src_chars': list(src_text),
#                     'pred_chars': pred_chars
#                 })
                
#             if len(examples) >= n_examples:
#                 break
    
#     if not examples:
#         print("No examples found.")
#         return []
    
#     # Create a figure with the desired style
#     fig, axes = plt.subplots(n_examples, 1, figsize=(12, 5*n_examples))
#     if n_examples == 1:
#         axes = [axes]  # Make axes iterable if only one example
    
#     # Use the viridis colormap
#     cmap = plt.cm.viridis
    
#     # Function to find max attention for each output character
#     def find_max_attention(att_matrix, src_chars):
#         max_indices = np.argmax(att_matrix, axis=1)
#         return [src_chars[i] if i < len(src_chars) else "?" for i in max_indices]
    
#     for i, example in enumerate(axes):
#         if i >= len(examples):
#             example.axis('off')
#             continue
        
#         # Get data for this example
#         src_chars = examples[i]['src_chars']
#         pred_chars = examples[i]['pred_chars']
#         attention = examples[i]['attention']
        
#         # Take only valid part of attention matrix
#         att_height = min(len(pred_chars), attention.shape[0])
#         att_width = min(len(src_chars), attention.shape[1])
#         attention_plot = attention[:att_height, :att_width]
        
#         # Find the input character with max attention for each output character
#         max_attention_chars = find_max_attention(attention_plot, src_chars)
        
#         # Plot attention matrix
#         im = axes[i].imshow(attention_plot, cmap=cmap, aspect='auto')
        
#         # Set source characters as x-axis labels
#         axes[i].set_xticks(range(len(src_chars[:att_width])))
#         axes[i].set_xticklabels(src_chars[:att_width], fontsize=12)
        
#         # Set target characters as y-axis labels
#         axes[i].set_yticks(range(len(pred_chars[:att_height])))
#         axes[i].set_yticklabels(pred_chars[:att_height], fontsize=12)
        
#         # Add a color bar
#         cbar = plt.colorbar(im, ax=axes[i])
#         cbar.set_label('Attention weight', fontsize=10)
        
#         # Highlight max attention for each output character (connectivity)
#         for y in range(att_height):
#             x = np.argmax(attention_plot[y])
#             if x < att_width:
#                 # Draw a rectangle around the max attention
#                 rect = plt.Rectangle((x-0.5, y-0.5), 1, 1, fill=False, 
#                                     edgecolor='lime', linewidth=2)
#                 axes[i].add_patch(rect)
                
#                 # Draw text showing max attention value
#                 axes[i].text(x, y, f'{attention_plot[y, x]:.2f}',
#                            ha='center', va='center', color='white', fontsize=8)
        
#         # Add model info at top
#         model_type = 'RNN' if not hasattr(model.decoder, 'attention') else 'RNN+Attention'
        
#         # Red for incorrect, green for correct
#         color = 'green' if examples[i]['is_correct'] else 'red'
#         axes[i].set_title(f"{model_type}: '{examples[i]['src_text']}' → '{examples[i]['pred_text']}'", 
#                         fontsize=14, color=color)
        
#         # Add connectivity info at bottom
#         connectivity_text = "Connectivity: "
#         for y, char in enumerate(pred_chars[:att_height]):
#             if y < len(max_attention_chars):
#                 connectivity_text += f"{char}←{max_attention_chars[y]}, "
        
#         # Add text below plot
#         axes[i].text(0, att_height + 0.5, connectivity_text[:-2], 
#                    fontsize=10, ha='left', va='center')
    
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
       
#     plt.show()
    
#     print(f"Connectivity heatmaps saved to: {save_path}")
#     return examples