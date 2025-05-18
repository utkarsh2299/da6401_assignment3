# test_model.py
import os
import torch
import argparse
import pandas as pd
from typing import List, Tuple

from data import get_dataloaders
from model import create_model
from train import test
from utils import set_seed, display_sample_predictions
from config import ModelConfig


def load_model(model_path: str, config: ModelConfig, device: torch.device):
    """
    Load a trained model from a saved checkpoint
    
    Args:
        model_path: Path to the saved model weights
        config: Model configuration
        device: Device to load the model on
        
    Returns:
        model: Loaded model
    """
    # Get dataloaders to get vocabulary sizes
    dataloaders = get_dataloaders(config.data_path, config.batch_size)
    train_dataset = dataloaders['train_dataset']
    
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
    
    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    
    return model, dataloaders


def test_model(model_path: str, 
              config: ModelConfig, 
              output_file: str = None,
              display_samples: int = 10):
    """
    Test a trained model and compute accuracy
    
    Args:
        model_path: Path to the saved model weights
        config: Model configuration
        output_file: Path to save prediction results (optional)
        display_samples: Number of samples to display
        
    Returns:
        accuracy: Test accuracy
    """
    device = config.device
    print(f"Testing model: {model_path}")
    print(f"Using device: {device}")
    
    # Load model
    model, dataloaders = load_model(model_path, config, device)
    test_loader = dataloaders['test']
    test_dataset = dataloaders['test_dataset']
    
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
        save_results(samples, accuracy, output_file)
        print(f"Results saved to: {output_file}")
        
    return accuracy, samples


def save_results(samples: List[Tuple[str, str, str]], 
                accuracy: float, 
                output_file: str):
    """
    Save test results to a file
    
    Args:
        samples: List of (source, prediction, target) tuples
        accuracy: Test accuracy
        output_file: Path to save results
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
        f.write(f"Correct predictions: {int(accuracy * len(samples))}\n")
        f.write("\n")
    
    # Append DataFrame
    results_df.to_csv(output_file, mode='a', index=False, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description='Test a trained transliteration model')
    parser.add_argument('--model_path', type=str, default='models/best_model.pt',
                       help='Path to the saved model weights')
    parser.add_argument('--config_path', type=str, default=None,
                       help='Path to a saved config file (optional)')
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
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load config from file or use command line args
    if args.config_path and os.path.exists(args.config_path):
        # Load config from file if provided
        config_dict = torch.load(args.config_path)
        config = ModelConfig(**config_dict)
        print(f"Loaded configuration from: {args.config_path}")
    else:
        # Create config from command line args
        config = ModelConfig(
            data_path=args.data_path,
            embedding_size=args.embedding_size,
            hidden_size=args.hidden_size,
            num_encoder_layers=args.num_encoder_layers,
            num_decoder_layers=args.num_decoder_layers,
            cell_type=args.cell_type,
            batch_size=args.batch_size,
            beam_size=args.beam_size,
            device=device
        )
    
    # Override beam size if explicitly provided
    if args.beam_size is not None:
        config.beam_size = args.beam_size
    
    # Test model
    test_model(
        model_path=args.model_path, 
        config=config, 
        output_file=args.output_file,
        display_samples=args.display_samples
    )


if __name__ == "__main__":
    main()