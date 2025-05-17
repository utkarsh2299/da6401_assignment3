import torch
import argparse
import os
from data import get_dataloaders
from model import create_model
from train import test
from config import ModelConfig


def test_best_model(config_path: str = 'best_config.pt'):
    """
    Test the best model from a previous training run or sweep
    
    Args:
        config_path: Path to the saved model configuration
    """
    # Load configuration
    if os.path.exists(config_path):
        config_dict = torch.load(config_path)
        
        # Create config object
        config = ModelConfig(
            data_path=config_dict.get('data_path', 'dakshina_dataset_v1.0/hi/lexicons/'),
            embedding_size=config_dict.get('embedding_size', 64),
            hidden_size=config_dict.get('hidden_size', 128),
            num_encoder_layers=config_dict.get('num_encoder_layers', 1),
            num_decoder_layers=config_dict.get('num_decoder_layers', 1),
            encoder_dropout=config_dict.get('encoder_dropout', 0.3),
            decoder_dropout=config_dict.get('decoder_dropout', 0.3),
            cell_type=config_dict.get('cell_type', 'gru'),
            beam_size=config_dict.get('beam_size', None)
        )
    else:
        print(f"Config file {config_path} not found. Using default configuration.")
        config = ModelConfig()
    
    # Load model path
    model_path = os.path.join(config.save_path, config.model_name)
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found. Please train a model first.")
        return
    
    # Get dataloaders
    dataloaders = get_dataloaders(config.data_path, config.batch_size)
    test_loader = dataloaders['test']
    test_dataset = dataloaders['test_dataset']
    
    # Get vocabulary sizes
    src_vocab_size, trg_vocab_size = test_dataset.get_vocab_size()
    
    # Get special token indices
    sos_idx = test_dataset.devanagari_char2idx['< SOS >']
    eos_idx = test_dataset.devanagari_char2idx['<EOS>']
    
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
        device=config.device,
        sos_idx=sos_idx,
        eos_idx=eos_idx
    )
    
    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=config.device))
    model = model.to(config.device)
    
    # Test model
    accuracy, samples = test(
        model=model,
        test_data=test_dataset,
        iterator=test_loader,
        device=config.device,
        beam_size=config.beam_size
    )
    
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Display sample predictions
    print("\nSample Predictions:")
    for i, (src, pred, trg) in enumerate(samples[:10]):
        print(f"Sample {i+1}:")
        print(f"Source (Latin): {src}")
        print(f"Prediction (Devanagari): {pred}")
        print(f"Target (Devanagari): {trg}")
        print(f"Correct: {'✓' if pred == trg else '✗'}")
        print("-" * 50)
    
    return accuracy, samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test a trained transliteration model')
    parser.add_argument('--config', type=str, default='best_config.pt',
                        help='Path to the saved model configuration')
    
    args = parser.parse_args()
    
    test_best_model(args.config)