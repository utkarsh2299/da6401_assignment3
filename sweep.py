import wandb
import os
import argparse
from config import ModelConfig, setup_sweep_configuration


def sweep_train(config=None):
    """
    Training function for wandb sweep
    
    Args:
        config: Wandb sweep configuration
    """
    
    
    # Initialize wandb
    with wandb.init(config=config)as run:
        # Get sweep config
        config = wandb.config
        run_name = f"{config.cell_type}-ec_{config.num_layers}-dc_{config.num_layers}-hs_{config.hidden_size}-emb_{config.embedding_size}-bs_{config.batch_size}-lr_{config.learning_rate}"
        run.name = run_name
        run.save()

        # Create model config
        model_config = ModelConfig(
            data_path='dakshina_dataset_v1.0/hi/lexicons/',
            embedding_size=config.embedding_size,
            hidden_size=config.hidden_size,
            num_encoder_layers=config.num_layers,
            num_decoder_layers=config.num_layers,
            encoder_dropout=config.encoder_dropout,
            decoder_dropout=config.decoder_dropout,
            cell_type=config.cell_type,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            n_epochs=20,  # Fixed for all runs
            clip=1.0,  # Fixed for all runs
            beam_size=config.beam_size,
            seed=42,  # Fixed for all runs
            log_wandb=True,
            # wandb_project='da6401_assignment3',
            # wandb_name=f"sweep-{wandb.run.id}"
            wandb_name=run_name
        )
        
        # Import here to avoid circular imports
        from main import main
        
        # Run training
        main(model_config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a wandb sweep for transliteration model')
    parser.add_argument('--count', type=int, default=10,
                        help='Number of runs to perform in the sweep')
    
    args = parser.parse_args()
    
    # Setup sweep
    sweep_config = setup_sweep_configuration()
    
    # Initialize sweep
    sweep_id = wandb.sweep(sweep_config, project='da6401_assignment3')
    
    # Run sweep
    wandb.agent(sweep_id, function=sweep_train, count=args.count)