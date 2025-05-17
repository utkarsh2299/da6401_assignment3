
# main.py
import torch
import torch.nn as nn
import torch.optim as optim
import os
import wandb
import argparse
from data import get_dataloaders
from model import create_model
from train import train_model, test
from utils import set_seed, count_parameters, plot_loss_curves, display_sample_predictions
from config import ModelConfig


def main(config: ModelConfig):
    """
    Main function to train and evaluate the model
    
    Args:
        config: Model configuration
    """
    # Set random seed
    set_seed(config.seed)
    
    # Initialize wandb if enabled
    if config.log_wandb:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_name,
            config=config.to_dict()
        )
    
    # Get dataloaders
    dataloaders = get_dataloaders(config.data_path, config.batch_size)
    train_loader = dataloaders['train']
    dev_loader = dataloaders['dev']
    test_loader = dataloaders['test']
    train_dataset = dataloaders['train_dataset']
    dev_dataset = dataloaders['dev_dataset']
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
        device=config.device,
        sos_idx=sos_idx,
        eos_idx=eos_idx
    )
    
    # Move model to device
    model = model.to(config.device)
    
    # Print model information
    print(f'The model has {count_parameters(model):,} trainable parameters')
    
    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore padding index
    
    # Define model save path
    model_save_path = os.path.join(config.save_path, config.model_name)
    
    # Train model
    history = train_model(
        model=model,
        train_iterator=train_loader,
        valid_iterator=dev_loader,
        optimizer=optimizer,
        criterion=criterion,
        n_epochs=config.n_epochs,
        clip=config.clip,
        device=config.device,
        patience=config.patience,
        min_delta=config.min_delta,
        save_path=model_save_path,
        log_wandb=config.log_wandb
    )
    
    # Plot loss curves
    # plot_loss_curves(history, save_path=os.path.join(config.save_path, 'loss_curves.png'))
    
    # Load best model
    model.load_state_dict(torch.load(model_save_path))
    
    # Test model
    accuracy, samples = test(
        model=model,
        test_data=test_dataset,
        iterator=test_loader,
        device=config.device,
        beam_size=config.beam_size
    )
    
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Log test accuracy to wandb
    if config.log_wandb:
        wandb.log({'test_accuracy': accuracy})
    
    # Display sample predictions
    display_sample_predictions(samples, n=10)
    
    # Close wandb run
    if config.log_wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a transliteration model')
    parser.add_argument('--data_path', type=str, default='dakshina_dataset_v1.0/hi/lexicons/',
                        help='Path to the Dakshina dataset')
    parser.add_argument('--embedding_size', type=int, default=64,
                        help='Size of the embeddings')
    parser.add_argument('--hidden_size', type=int, default=128,
                        help='Size of the hidden states')
    parser.add_argument('--num_encoder_layers', type=int, default=1,
                        help='Number of layers in the encoder')
    parser.add_argument('--num_decoder_layers', type=int, default=1,
                        help='Number of layers in the decoder')
    parser.add_argument('--encoder_dropout', type=float, default=0.3,
                        help='Dropout probability for the encoder')
    parser.add_argument('--decoder_dropout', type=float, default=0.3,
                        help='Dropout probability for the decoder')
    parser.add_argument('--cell_type', type=str, default='lstm',
                        help='Type of RNN cell (rnn, lstm, gru)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for the optimizer')
    parser.add_argument('--n_epochs', type=int, default=2,
                        help='Number of epochs to train for')
    parser.add_argument('--clip', type=float, default=1.0,
                        help='Gradient clipping value')
    parser.add_argument('--beam_size', type=int, default=None,
                        help='Beam size for beam search (None for greedy)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--log_wandb', action='store_true', default=True,
                        help='Whether to log metrics to wandb')
    
    args = parser.parse_args()
    
    # Create config
    config = ModelConfig(
        data_path=args.data_path,
        embedding_size=args.embedding_size,
        hidden_size=args.hidden_size,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        encoder_dropout=args.encoder_dropout,
        decoder_dropout=args.decoder_dropout,
        cell_type=args.cell_type,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        n_epochs=args.n_epochs,
        clip=args.clip,
        beam_size=args.beam_size,
        seed=args.seed,
        log_wandb=args.log_wandb
    )
    
    main(config)



