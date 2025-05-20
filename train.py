import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb
import time
from typing import Dict, List, Tuple, Optional


def train(model: nn.Module, 
          iterator: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer, 
          criterion: nn.Module, 
          clip: float,
          device: torch.device,
          teacher_forcing_ratio: float = 0.5) -> float:
    """
    Train the model for one epoch
    
    Args:
        model: Model to train
        iterator: DataLoader for training data
        optimizer: Optimizer to use
        criterion: Loss function
        clip: Gradient clipping value
        device: Device to use
        teacher_forcing_ratio: Probability of using teacher forcing
        
    Returns:
        epoch_loss: Average loss over the epoch
    """
    model.train()
    epoch_loss = 0
    
    for i, (src, trg) in enumerate(tqdm(iterator, desc="Training", leave=False)):
        src = src.to(device)
        trg = trg.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        output = model(src, trg, teacher_forcing_ratio)
        
        # Reshape output and target for loss calculation
        # output: [batch_size, trg_len-1, output_dim]
        # trg: [batch_size, trg_len]
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)
        
        # Calculate loss
        loss = criterion(output, trg)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        # Update parameters
        optimizer.step()
        
        # Update total loss
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)


def evaluate(model: nn.Module, 
             iterator: torch.utils.data.DataLoader, 
             criterion: nn.Module, 
             device: torch.device) -> float:
    """
    Evaluate the model
    
    Args:
        model: Model to evaluate
        iterator: DataLoader for evaluation data
        criterion: Loss function
        device: Device to use
        
    Returns:
        epoch_loss: Average loss over the evaluation set
    """
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for i, (src, trg) in enumerate(tqdm(iterator, desc="Evaluating", leave=False)):
            src = src.to(device)
            trg = trg.to(device)
            
            # Forward pass (no teacher forcing)
            output = model(src, trg, 0)
            
            # Reshape output and target for loss calculation
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)
            
            # Calculate loss
            loss = criterion(output, trg)
            
            # Update total loss
            epoch_loss += loss.item()
            
    return epoch_loss / len(iterator)


def test(model: nn.Module, 
         test_data: torch.utils.data.Dataset,
         iterator: torch.utils.data.DataLoader, 
         device: torch.device,
         beam_size: Optional[int] = None) -> Tuple[float, List[Tuple[str, str, str]]]:
    """
    Test the model and calculate accuracy
    
    Args:
        model: Model to test
        test_data: Test dataset
        iterator: DataLoader for test data
        device: Device to use
        beam_size: Size of beam for beam search (None for greedy)
        
    Returns:
        accuracy: Accuracy of the model
        samples: List of sample translations (source, prediction, target)
    """
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    samples = []
    
    with torch.no_grad():
        for i, (src, trg) in enumerate(tqdm(iterator, desc="Testing", leave=False)):
            src = src.to(device)
            trg = trg.to(device)
            
            batch_size = src.shape[0]
            
            for j in range(batch_size):
                src_tensor = src[j].unsqueeze(0)
                trg_tensor = trg[j].unsqueeze(0)
                
                # Get indices with beam search or greedy decoding
                # Handle both models with and without attention
                try:
                    if beam_size:
                        # For attention models
                        output_indices, _ = model.beam_search(src_tensor, max_len=100, beam_size=beam_size)
                    else:
                        # For attention models
                        output_indices, _ = model.greedy_decode(src_tensor, max_len=100)
                except ValueError:
                    # For non-attention models
                    if beam_size:
                        output_indices = model.beam_search(src_tensor, max_len=100, beam_size=beam_size)
                    else:
                        output_indices = model.greedy_decode(src_tensor, max_len=100)
                except Exception as e:
                    print(f"Error in decoding: {e}")
                    continue
                
                # Handle case where output_indices is a single integer
                if isinstance(output_indices, int):
                    output_indices = [output_indices]
                
                # Convert indices to text
                src_text = test_data.decode_latin(src_tensor.squeeze().cpu().numpy().tolist())
                trg_text = test_data.decode_devanagari(trg_tensor.squeeze().cpu().numpy().tolist())
                pred_text = test_data.decode_devanagari(output_indices)
                
                # Check if prediction is correct
                if pred_text == trg_text:
                    correct_predictions += 1
                
                total_predictions += 1
                
                # Add sample to list
                samples.append((src_text, pred_text, trg_text))
    
    # Calculate accuracy
    accuracy = correct_predictions / total_predictions
    
    return accuracy, samples


def train_model(model: nn.Module, 
                train_iterator: torch.utils.data.DataLoader,
                valid_iterator: torch.utils.data.DataLoader,
                optimizer: torch.optim.Optimizer,
                criterion: nn.Module,
                n_epochs: int,
                clip: float,
                device: torch.device,
                patience: int = 5,
                min_delta: float = 0.0,
                save_path: str = 'best_model.pt',
                log_wandb: bool = True) -> Dict:
    """
    Train the model for multiple epochs with early stopping
    
    Args:
        model: Model to train
        train_iterator: DataLoader for training data
        valid_iterator: DataLoader for validation data
        optimizer: Optimizer to use
        criterion: Loss function
        n_epochs: Number of epochs to train for
        clip: Gradient clipping value
        device: Device to use
        patience: Number of epochs to wait for improvement
        min_delta: Minimum change in validation loss to be considered as improvement
        save_path: Path to save the best model
        log_wandb: Whether to log metrics to wandb
        
    Returns:
        history: Dictionary containing training and validation loss history
    """
    best_valid_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'valid_loss': []}
    
    for epoch in range(n_epochs):
        start_time = time.time()
        
        # Training
        train_loss = train(model, train_iterator, optimizer, criterion, clip, device)
        
        # Validation
        valid_loss = evaluate(model, valid_iterator, criterion, device)
        
        # Calculate accuracy on just a small batch for logging (won't affect training time significantly)
        if log_wandb:
            train_acc = calculate_sample_accuracy(model, train_iterator, device, max_batches=2)
            valid_acc = calculate_sample_accuracy(model, valid_iterator, device, max_batches=2)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['valid_loss'].append(valid_loss)
        
        # Calculate elapsed time
        end_time = time.time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
        
        # Log metrics
        if log_wandb:
            wandb.log({
                'train_loss': train_loss,
                'valid_loss': valid_loss,
                'train_accuracy': train_acc,
                'valid_accuracy': valid_acc,
                'epoch': epoch
            })
        
        # Print progress
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs:.2f}s')
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\tVal. Loss: {valid_loss:.3f}')
        if log_wandb:
            print(f'\tTrain Acc: {train_acc:.3f} | Val. Acc: {valid_acc:.3f} (sample)')
        
        # Check if this is the best model
        if valid_loss < best_valid_loss - min_delta:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), save_path)
            print(f'\tBest validation loss: {valid_loss:.3f}')
            patience_counter = 0
        else:
            patience_counter += 1
            print(f'\tNo improvement in validation loss for {patience_counter} epochs')
            
        # Early stopping
        if patience_counter >= patience:
            print(f'Early stopping after {epoch+1} epochs')
            break
    
    return history

def calculate_sample_accuracy(model, iterator, device, max_batches=2):
    """
    Calculate accuracy on a small sample of batches
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i, (src, trg) in enumerate(iterator):
            if i >= max_batches:
                break
                
            src = src.to(device)
            trg = trg.to(device)
            
            # Forward pass
            output = model(src, trg, 0)
            predictions = output.argmax(dim=2)
            target = trg[:, 1:]
            
            # Create mask to ignore padding
            mask = (target != 0)
            
            # Count correct predictions
            correct += ((predictions == target) * mask).sum().item()
            total += mask.sum().item()
    
    return correct / total if total > 0 else 0

def predict_batch(model: nn.Module, 
                 dataset: torch.utils.data.Dataset,
                 latin_texts: List[str], 
                 device: torch.device,
                 beam_size: Optional[int] = None) -> List[str]:
    """
    Predict Devanagari text from Latin texts
    
    Args:
        model: Model to use
        dataset: Dataset containing vocabulary
        latin_texts: List of Latin texts
        device: Device to use
        beam_size: Size of beam for beam search (None for greedy)
        
    Returns:
        devanagari_texts: List of predicted Devanagari texts
    """
    model.eval()
    devanagari_texts = []
    
    with torch.no_grad():
        for text in latin_texts:
            # Convert text to indices
            indices = [dataset.latin_char2idx.get(char, dataset.latin_char2idx['<UNK>']) 
                       for char in text]
            src_tensor = torch.tensor([indices]).to(device)
            
            # Get indices with beam search or greedy decoding
            if beam_size:
                output_indices = model.beam_search(src_tensor, max_len=100, beam_size=beam_size)
            else:
                output_indices = model.greedy_decode(src_tensor, max_len=100)
            
            # Convert indices to text
            pred_text = dataset.decode_devanagari(output_indices)
            devanagari_texts.append(pred_text)
    
    return devanagari_texts