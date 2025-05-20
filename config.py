import torch
from dataclasses import dataclass
from typing import Optional, Dict, Any
import os



@dataclass
class ModelConfig:
    # Data paths
    data_path: str = 'dakshina_dataset_v1.0/hi/lexicons/'
    
    # Model hyperparameters
    embedding_size: int = 64
    hidden_size: int = 128
    num_encoder_layers: int = 1
    num_decoder_layers: int = 1
    encoder_dropout: float = 0.3
    decoder_dropout: float = 0.3
    cell_type: str = 'gru'
    
    # Training hyperparameters
    batch_size: int = 64
    learning_rate: float = 0.001
    n_epochs: int = 20
    clip: float = 1.0
    teacher_forcing_ratio: float = 0.5
    
    # Early stopping
    patience: int = 5
    min_delta: float = 0.0
    
    # Decoding
    beam_size: Optional[int] = None  # None for greedy decoding
    
    # Paths
    save_path: str = 'models_true/'
    model_name: str = 'best_model.pt'
    
    # Reproducibility
    seed: int = 42
    
    # Device
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Wandb
    log_wandb: bool = True
    # wandb_project: str = 'hindi-transliteration'
    wandb_project: str = 'da6401_assignment3'
    wandb_name: Optional[str] = None
    
    use_attention: bool = False
    visualize_connectivity: bool = False
    
    
    def __post_init__(self):
        """Create model save directory if it doesn't exist"""
        os.makedirs(self.save_path, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for wandb"""
        return {k: v for k, v in self.__dict__.items() 
                if not k.startswith('_') and k != 'device'}
        
        
        
def setup_sweep_configuration():
    """
    Setup wandb sweep configuration
    
    Returns:
        sweep_config: Sweep configuration
    """
    sweep_config = {
        'method': 'bayes',  
        'metric': {
            'name': 'valid_loss',
            'goal': 'minimize'
        },
        'parameters': {
            'embedding_size': {
                'values': [16, 64, 128, 256]
            },
            'hidden_size': {
                'values': [64, 128, 256]
            },
            
            'num_layers': {
            'values': [1, 2, 3]
            },
            'cell_type': {
                'values': ['rnn', 'lstm', 'gru']
            },
            'encoder_dropout': {
                'values': [ 0.2, 0.5]
            },
            'decoder_dropout': {
                'values': [ 0.2, 0.3, 0.5]
            },
            'batch_size': {
                'values': [32, 64, 128]
            },
            'learning_rate': {
                'values': [0.0001, 0.001, 0.0005]
            },
            'beam_size': {
                'values': [None, 3, 5, 7]
            },
            'use_attention': {
                'values': [True]
            }
        }
    }
    
    return sweep_config