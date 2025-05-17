# data.py
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import Dict, List, Tuple, Any, Union


class TransliterationDataset(Dataset):
    def __init__(self, data_path: str, split: str = 'train'):
        """
        Dataset class for transliteration data with enhanced error handling
        """
        self.data_path = data_path
        self.split = split
        
        # Add special tokens
        special_tokens = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']
        
        # Initialize with default values
        self.latin_texts = []
        self.devanagari_texts = []
        self.latin_vocab = special_tokens.copy()
        self.devanagari_vocab = special_tokens.copy()
        
        # Set up character-to-index and index-to-character mappings
        self.latin_char2idx = {token: i for i, token in enumerate(special_tokens)}
        self.devanagari_char2idx = {token: i for i, token in enumerate(special_tokens)}
        self.latin_idx2char = {i: token for i, token in enumerate(special_tokens)}
        self.devanagari_idx2char = {i: token for i, token in enumerate(special_tokens)}
        
        # Try loading data
        try:
            # Find correct file path
            file_path = self._find_file_path(data_path, split)
            if not file_path:
                print(f"Warning: Could not find file for split '{split}' in '{data_path}'")
                return
                
            print(f"Loading data from: {file_path}")
            
            # Load data with robust parsing
            data_pairs = self._load_data_safely(file_path)
            if not data_pairs:
                print("Warning: No valid data pairs loaded")
                return
                
            print(f"Loaded {len(data_pairs)} valid data pairs")
            
            # Extract source and target texts
            self.latin_texts, self.devanagari_texts = zip(*data_pairs)
            
            # Now build vocabularies from clean data
            latin_chars = self._extract_characters(self.latin_texts)
            devanagari_chars = self._extract_characters(self.devanagari_texts)
            
            # Create vocabularies
            self.latin_vocab = special_tokens + sorted(list(latin_chars))
            self.devanagari_vocab = special_tokens + sorted(list(devanagari_chars))
            
            # Update mappings
            self.latin_char2idx = {char: idx for idx, char in enumerate(self.latin_vocab)}
            self.devanagari_char2idx = {char: idx for idx, char in enumerate(self.devanagari_vocab)}
            self.latin_idx2char = {idx: char for char, idx in self.latin_char2idx.items()}
            self.devanagari_idx2char = {idx: char for char, idx in self.devanagari_char2idx.items()}
            
            print(f"Latin vocabulary size: {len(self.latin_vocab)}")
            print(f"Devanagari vocabulary size: {len(self.devanagari_vocab)}")
            
        except Exception as e:
            print(f"Error initializing dataset: {e}")
    
    def _find_file_path(self, base_path: str, split: str) -> str:
        """Find the correct file path with various naming conventions"""
        possible_extensions = ['.tsv']
        possible_prefixes = ['', 'hi.translit.sampled.']
        
        # Check if directory exists
        if not os.path.exists(base_path):
            print(f"Warning: Directory {base_path} does not exist")
            return ""
            
        # List files in directory
        try:
            files = os.listdir(base_path)
            print(f"Files in directory: {files}")
        except Exception as e:
            print(f"Error listing directory: {e}")
            files = []
        
        # Try different combinations
        for prefix in possible_prefixes:
            for ext in possible_extensions:
                filename = f"{prefix}{split}{ext}"
                full_path = os.path.join(base_path, filename)
                
                # Check exact match
                if os.path.exists(full_path):
                    return full_path
                
                # Check in directory listing
                if filename in files:
                    return os.path.join(base_path, filename)
        
        # Try to find a partial match
        split_base = split.split('.')[0]  # Handle cases like 'hi.translit.sampled.train'
        for filename in files:
            if split_base in filename:
                return os.path.join(base_path, filename)
                
        return ""
    
    def _load_data_safely(self, file_path: str) -> List[Tuple[str, str]]:
        """Load data with multiple fallback methods"""
        valid_pairs = []
        
        # First attempt: pandas with tab separator
        try:
            df = pd.read_csv(file_path, sep='\t', header=None, 
                             na_values=['nan', 'NaN', 'NULL', 'None', 'NA'],
                             dtype=str)  # Force string type
            print(f"Loaded with pandas (tab): {df.shape}")
            
            # Check and clean data
            valid_pairs = self._extract_valid_pairs(df)
            if valid_pairs:
                return valid_pairs
        except Exception as e:
            print(f"Failed to load with pandas (tab): {e}")
        
        # Second attempt: pandas with comma separator
        try:
            df = pd.read_csv(file_path, sep=',', header=None, 
                             na_values=['nan', 'NaN', 'NULL', 'None', 'NA'],
                             dtype=str)
            print(f"Loaded with pandas (comma): {df.shape}")
            
            valid_pairs = self._extract_valid_pairs(df)
            if valid_pairs:
                return valid_pairs
        except Exception as e:
            print(f"Failed to load with pandas (comma): {e}")
        
        # Third attempt: manual file reading
        try:
            valid_pairs = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                        
                    # Try tab separator
                    parts = line.split('\t')
                    if len(parts) != 2:
                        # Try space separator
                        parts = line.split()
                    
                    if len(parts) >= 2:
                        src, tgt = parts[0], parts[1]
                        # Verify both are strings and not empty
                        if (isinstance(src, str) and isinstance(tgt, str) and 
                            src.strip() and tgt.strip()):
                            valid_pairs.append((src.strip(), tgt.strip()))
            
            print(f"Loaded manually: {len(valid_pairs)} pairs")
            return valid_pairs
        except Exception as e:
            print(f"Failed to load manually: {e}")
            
        return []
    
    def _extract_valid_pairs(self, df: pd.DataFrame) -> List[Tuple[str, str]]:
        """Extract valid string pairs from DataFrame"""
        valid_pairs = []
        
        # Ensure we have at least 2 columns
        if len(df.columns) < 2:
            print(f"Not enough columns: {len(df.columns)}")
            return valid_pairs
            
        # Force string conversion
        df[0] = df[0].astype(str)
        df[1] = df[1].astype(str)
        
        # Check for invalid values
        df = df[(df[0] != 'nan') & (df[1] != 'nan') & 
                (df[0] != 'None') & (df[1] != 'None') &
                (df[0] != '') & (df[1] != '')]
        
        # Extract pairs
        for i, row in df.iterrows():
            src, tgt = str(row[0]), str(row[1])
            
            # Skip any numeric-only values as they're likely errors
            if src.replace('.', '').isdigit() or tgt.replace('.', '').isdigit():
                print(f"Skipping numeric value at row {i}: {src} -> {tgt}")
                continue
                
            # Skip NaN string representations
            if src.lower() in ('nan', 'none', 'null') or tgt.lower() in ('nan', 'none', 'null'):
                print(f"Skipping NaN string at row {i}: {src} -> {tgt}")
                continue
            
            valid_pairs.append((src, tgt))
            
        print(f"Extracted {len(valid_pairs)} valid pairs from DataFrame")
        return valid_pairs
    
    def _extract_characters(self, texts: List[str]) -> set:
        """Safely extract unique characters from texts"""
        chars = set()
        for text in texts:
            if isinstance(text, str):
                for char in text:
                    chars.add(char)
        return chars
    
    def __len__(self) -> int:
        return len(self.latin_texts)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get an item with extra safety checks"""
        # Verify index is valid
        if idx < 0 or idx >= len(self.latin_texts):
            # Default to empty strings if index out of range
            latin_text = ""
            devanagari_text = ""
        else:
            # Get texts with safety checks
            latin_text = self.latin_texts[idx] if isinstance(self.latin_texts[idx], str) else ""
            devanagari_text = self.devanagari_texts[idx] if isinstance(self.devanagari_texts[idx], str) else ""
        
        # Convert characters to indices with safety
        latin_indices = []
        for char in latin_text:
            latin_indices.append(self.latin_char2idx.get(char, self.latin_char2idx['<UNK>']))
        
        # Add SOS and EOS tokens for target
        devanagari_indices = [self.devanagari_char2idx['<SOS>']]
        for char in devanagari_text:
            devanagari_indices.append(self.devanagari_char2idx.get(char, self.devanagari_char2idx['<UNK>']))
        devanagari_indices.append(self.devanagari_char2idx['<EOS>'])
        
        return torch.tensor(latin_indices), torch.tensor(devanagari_indices)
    
    def get_vocab_size(self) -> Tuple[int, int]:
        """Returns the vocabulary size for source and target languages"""
        return len(self.latin_vocab), len(self.devanagari_vocab)
    
    def decode_latin(self, indices: List[int]) -> str:
        """Convert a list of indices to Latin text"""
        return ''.join([self.latin_idx2char.get(idx, '<UNK>') for idx in indices 
                        if idx not in [self.latin_char2idx['<PAD>'], 
                                       self.latin_char2idx['<SOS>'],
                                       self.latin_char2idx['<EOS>']]])
    
    def decode_devanagari(self, indices: List[int]) -> str:
        """Convert a list of indices to Devanagari text"""
        return ''.join([self.devanagari_idx2char.get(idx, '<UNK>') for idx in indices 
                        if idx not in [self.devanagari_char2idx['<PAD>'], 
                                       self.devanagari_char2idx['<SOS>'],
                                       self.devanagari_char2idx['<EOS>']]])


def collate_fn(batch):
    """Custom collate function for padding sequences in a batch"""
    # Handle empty batch
    if not batch:
        return torch.tensor([]), torch.tensor([])
    
    latin_seqs, devanagari_seqs = zip(*batch)
    
    # Pad sequences
    latin_seqs_padded = pad_sequence(latin_seqs, batch_first=True, padding_value=0)
    devanagari_seqs_padded = pad_sequence(devanagari_seqs, batch_first=True, padding_value=0)
    
    return latin_seqs_padded, devanagari_seqs_padded


def get_dataloaders(data_path: str, batch_size: int) -> Dict[str, Any]:
    """Create DataLoaders for train, dev, and test sets"""
    print(f"Creating dataloaders for path: {data_path}")
    
    # Create datasets with proper error handling
    try:
        train_dataset = TransliterationDataset(data_path, 'train')
    except Exception as e:
        print(f"Error creating train dataset: {e}")
        raise
        
    try:
        dev_dataset = TransliterationDataset(data_path, 'dev')
    except Exception as e:
        print(f"Error creating dev dataset: {e}")
        raise
        
    try:
        test_dataset = TransliterationDataset(data_path, 'test')
    except Exception as e:
        print(f"Error creating test dataset: {e}")
        raise
    
    # Ensure dev and test use the same vocab as train (if valid)
    if hasattr(train_dataset, 'latin_vocab') and train_dataset.latin_texts:
        dev_dataset.latin_vocab = train_dataset.latin_vocab
        dev_dataset.devanagari_vocab = train_dataset.devanagari_vocab
        dev_dataset.latin_char2idx = train_dataset.latin_char2idx
        dev_dataset.devanagari_char2idx = train_dataset.devanagari_char2idx
        dev_dataset.latin_idx2char = train_dataset.latin_idx2char
        dev_dataset.devanagari_idx2char = train_dataset.devanagari_idx2char
        
        test_dataset.latin_vocab = train_dataset.latin_vocab
        test_dataset.devanagari_vocab = train_dataset.devanagari_vocab
        test_dataset.latin_char2idx = train_dataset.latin_char2idx
        test_dataset.devanagari_char2idx = train_dataset.devanagari_char2idx
        test_dataset.latin_idx2char = train_dataset.latin_idx2char
        test_dataset.devanagari_idx2char = train_dataset.devanagari_idx2char
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    
    dev_loader = DataLoader(
        dev_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    print(f"Created dataloaders - Train: {len(train_dataset)} samples, "
          f"Dev: {len(dev_dataset)} samples, Test: {len(test_dataset)} samples")
    
    return {
        'train': train_loader,
        'dev': dev_loader,
        'test': test_loader,
        'train_dataset': train_dataset,
        'dev_dataset': dev_dataset,
        'test_dataset': test_dataset
    }


    # train_dataset = TransliterationDataset(data_path, 'hi.translit.sampled.train')
    # dev_dataset = TransliterationDataset(data_path, 'hi.translit.sampled.dev')
    # test_dataset = TransliterationDataset(data_path, 'hi.translit.sampled.test')