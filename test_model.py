# test_model.py
import os
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import pandas as pd
from typing import List, Tuple

from data import get_dataloaders
from model import create_model
from train import train_model, test
from utils import set_seed, display_sample_predictions
from config import ModelConfig


def train_and_test_model(config: ModelConfig, 
                        output_file: str = None,
                        display_samples: int = 10,
                        n_epochs: int = 20):
    """
    Train a model with the given configuration and test it
    
    Args:
        config: Model configuration
        output_file: Path to save prediction results (optional)
        display_samples: Number of samples to display
        n_epochs: Number of epochs to train for
        
    Returns:
        accuracy: Test accuracy
    """
    device = config.device
    print(f"Training and testing model with configuration:")
    print(f"  Embedding size: {config.embedding_size}")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Encoder layers: {config.num_encoder_layers}")
    print(f"  Decoder layers: {config.num_decoder_layers}")
    print(f"  Cell type: {config.cell_type}")
    print(f"Using device: {device}")
    
    # Get dataloaders
    dataloaders = get_dataloaders(config.data_path, config.batch_size)
    train_loader = dataloaders['train']
    dev_loader = dataloaders['dev']
    test_loader = dataloaders['test']
    train_dataset = dataloaders['train_dataset']
    test_dataset = dataloaders['test_dataset']
    
    # Get vocabulary sizes
    src_vocab_size, trg_vocab_size = train_dataset.get_vocab_size()
    
    # Get special token indices
    sos_idx = train_dataset.devanagari_char2idx['<SOS>']
    eos_idx = train_dataset.devanagari_char2idx['<EOS>']
    
    # Create model
    model = create_model(
        src_vocab_size=src_vocab_size,
        trg_vocab_size=trg_vocab_size,
        embedding_size=config.embedding_size,
        hidden_size=config.hidden_size,
        num_encoder_layers=config.num_encoder_layers,
        num_decoder_layers=config.num_decoder_layers,
        encoder_dropout=config.encoder_dropout,
        decoder_dropout=config.decoder_dropout,
        cell_type=config.cell_type,
        device=device,
        sos_idx=sos_idx,
        eos_idx=eos_idx
    )
    
    # Move model to device
    model = model.to(device)
    
    # Print model information
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model has {num_params:,} trainable parameters')
    
    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore padding index
    
    # Define model save path
    os.makedirs(config.save_path, exist_ok=True)
    model_save_path = os.path.join(config.save_path, config.model_name)
    
    # Train model
    print(f"Training model for {n_epochs} epochs...")
    history = train_model(
        model=model,
        train_iterator=train_loader,
        valid_iterator=dev_loader,
        optimizer=optimizer,
        criterion=criterion,
        n_epochs=n_epochs,
        clip=config.clip,
        device=device,
        patience=config.patience,
        min_delta=config.min_delta,
        save_path=model_save_path,
        log_wandb=False  # No need to log to wandb for this test
    )
    
    # Load best model
    model.load_state_dict(torch.load(model_save_path))
    
    # Test model
    print(f"Running test with beam size: {config.beam_size}")
    accuracy, samples = test(
        model=model,
        test_data=test_dataset,
        iterator=test_loader,
        device=device,
        beam_size=config.beam_size
    )
    
    print(f"\nTest Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    
    # Display sample predictions
    if display_samples > 0:
        print(f"\nSample Predictions (showing {min(display_samples, len(samples))} of {len(samples)}):")
        display_sample_predictions(samples, n=display_samples)
    
    # Save results to file if requested
    if output_file:
        save_results(samples, accuracy, output_file, config)
        print(f"Results saved to: {output_file}")
        
    return accuracy, samples


def save_results(samples: List[Tuple[str, str, str]], 
                accuracy: float, 
                output_file: str,
                config: ModelConfig):
    """
    Save test results to a file
    
    Args:
        samples: List of (source, prediction, target) tuples
        accuracy: Test accuracy
        output_file: Path to save results
        config: Model configuration
    """
    # Convert samples to DataFrame
    results_df = pd.DataFrame({
        'source': [s[0] for s in samples],
        'prediction': [s[1] for s in samples],
        'target': [s[2] for s in samples],
        'correct': [s[1] == s[2] for s in samples]
    })
    
    # Add summary at the top
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Test Accuracy: {accuracy:.4f}\n")
        f.write(f"Total samples: {len(samples)}\n")
        f.write(f"Correct predictions: {int(accuracy * len(samples))}\n\n")
        f.write(f"Model Configuration:\n")
        f.write(f"  Embedding size: {config.embedding_size}\n")
        f.write(f"  Hidden size: {config.hidden_size}\n")
        f.write(f"  Encoder layers: {config.num_encoder_layers}\n")
        f.write(f"  Decoder layers: {config.num_decoder_layers}\n")
        f.write(f"  Cell type: {config.cell_type}\n")
        f.write(f"  Beam size: {config.beam_size}\n\n")
    
    # Append DataFrame
    results_df.to_csv(output_file, mode='a', index=False, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description='Train and test a transliteration model')
    parser.add_argument('--data_path', type=str, default='dakshina_dataset_v1.0/hi/lexicons/',
                       help='Path to the dataset')
    parser.add_argument('--beam_size', type=int, default=5,
                       help='Beam size for decoding (None for greedy)')
    parser.add_argument('--output_file', type=str, default='prediction_out.csv',
                       help='Path to save prediction results (optional)')
    parser.add_argument('--display_samples', type=int, default=10,
                       help='Number of sample predictions to display')
    parser.add_argument('--embedding_size', type=int, default=128,
                       help='Size of the embeddings')
    parser.add_argument('--hidden_size', type=int, default=256,
                       help='Size of the hidden states')
    parser.add_argument('--num_encoder_layers', type=int, default=3,
                       help='Number of layers in the encoder')
    parser.add_argument('--num_decoder_layers', type=int, default=3,
                       help='Number of layers in the decoder')
    parser.add_argument('--cell_type', type=str, default='lstm',
                       help='Type of RNN cell (rnn, lstm, gru)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for testing')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate for training')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of epochs to train for')
    parser.add_argument('--encoder_dropout', type=float, default=0.3,
                       help='Dropout for encoder')
    parser.add_argument('--decoder_dropout', type=float, default=0.2,
                       help='Dropout for decoder')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create config from command line args
    config = ModelConfig(
        data_path=args.data_path,
        embedding_size=args.embedding_size,
        hidden_size=args.hidden_size,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        cell_type=args.cell_type,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        encoder_dropout=args.encoder_dropout,
        decoder_dropout=args.decoder_dropout,
        beam_size=args.beam_size,
        device=device
    )
    
    # Train and test model
    train_and_test_model(
        config=config, 
        output_file=args.output_file,
        display_samples=args.display_samples,
        n_epochs=args.epochs
    )


if __name__ == "__main__":
    main()