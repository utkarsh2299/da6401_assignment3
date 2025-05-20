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

### You can use the Run that gave bests score : gru-ec_3-dc_3-hs_256-emb_256-bs_128-lr_0.0005 :

```
batch_size:
    value: 128

beam_size:
    value: 3

cell_type:
    value: gru

decoder_dropout:
    value: 0.5

embedding_size:
    value: 256

encoder_dropout:
    value: 0.5

hidden_size:
    value: 256

learning_rate:
    value: 0.0005

num_layers:
    value: 3

use_attention:
    value: false
```

### Hyperparameter Optimization

Run a standard hyperparameter sweep:
```bash
python sweep.py --count 10 
```


### Testing a Trained Model

Test a trained model:
```bash
python test_model.py 
```

## Evaluation

The model is evaluated based on exact match accuracy - the transliteration is correct only if it exactly matches the reference Devanagari text.

### Notes

I have used Nirmala fonts as Devanagari and latin fonts together were not rendered properly in Matplotlib.

You can open the .html files to see the attention visualisaition which is interactive.


### Feedback

Let me know if there is any issue with the code. Thank you.
