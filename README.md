# da6401_assignment3

# Hindi Transliteration Model

This project implements a sequence-to-sequence (seq2seq) RNN-based model for transliterating Latin script (romanized) to Devanagari script for Hindi using the Google Dakshina dataset.

## Project Structure

```
.
├── config.py          # Configuration and hyperparameters
├── data.py            # Data loading and preprocessing
├── main.py            # Main script to run experiments
├── model.py           # Seq2Seq model architecture
├── sweep.py           # Hyperparameter sweep configuration
├── test_model.py      # Script to test trained models
├── train.py           # Training and testing functions
└── utils.py           # Utility functions
```

## Model Architecture

The model follows a standard encoder-decoder architecture:

1. **Encoder**: Processes the input Latin script word character by character
   - Character embedding layer
   - RNN/LSTM/GRU for sequential encoding
   - Configurable number of layers and hidden dimensions

2. **Decoder**: Generates the Devanagari output character by character
   - Character embedding layer
   - RNN/LSTM/GRU for sequential decoding
   - Configurable number of layers and hidden dimensions
   - Supports both greedy decoding and beam search

## Setting Up

### Requirements

- Python 3.7+
- PyTorch
- pandas
- tqdm
- wandb (for hyperparameter tuning)
- matplotlib (for visualization)

Install dependencies:
```bash
pip install torch pandas tqdm wandb matplotlib
```

### Dataset

The code assumes the Dakshina dataset is organized in the following structure:
```
dakshina_dataset_v1.0/
└── hi/
    └── lexicons/
        ├── train.tsv
        ├── dev.tsv
        └── test.tsv
```

Each TSV file should contain two columns: Latin script (romanized) and Devanagari script.

## Usage

### Training a Model

Train a model with default parameters:
```bash
python main.py
```

Train with custom parameters:
```bash
python main.py --embedding_size 128 --hidden_size 256 --cell_type lstm
```

### Hyperparameter Optimization

Run a standard hyperparameter sweep:
```bash
python sweep.py --count 10 
```

Test the best model from a sweep:
```bash
python sweep.py --test_best --sweep_id <sweep_id>
```

### Testing a Trained Model

Test a trained model:
```bash
python test_model.py --config best_config.pt
```


## Model Complexity

### Computational Complexity

The computational complexity depends on the RNN cell type:

- **Basic RNN**: O(T × (2m + 2(m×k + k×k) + k×V))
- **GRU**: O(T × (2m + 6(m×k + k×k) + k×V))
- **LSTM**: O(T × (2m + 8(m×k + k×k) + k×V))

Where:
- T = sequence length
- m = embedding size
- k = hidden state size
- V = vocabulary size

### Parameter Count

The total number of parameters also depends on the RNN cell type:

- **Basic RNN**: 2Vm + 2(m+k+1)k + (k+1)V
- **GRU**: 2Vm + 6(m+k+1)k + (k+1)V
- **LSTM**: 2Vm + 8(m+k+1)k + (k+1)V



## Evaluation

The model is evaluated based on exact match accuracy - the transliteration is correct only if it exactly matches the reference Devanagari text.

