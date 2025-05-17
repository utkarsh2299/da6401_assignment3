import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from typing import Tuple, List, Optional


class Encoder(nn.Module):
    def __init__(self, 
                 input_size: int, 
                 embedding_size: int, 
                 hidden_size: int, 
                 num_layers: int = 1, 
                 dropout: float = 0.0, 
                 cell_type: str = 'gru'):
        """
        Encoder for the Seq2Seq model
        
        Args:
            input_size: Size of the input vocabulary
            embedding_size: Size of the embeddings
            hidden_size: Size of the hidden state
            num_layers: Number of layers in the RNN
            dropout: Dropout probability (only applied between layers)
            cell_type: Type of RNN cell (rnn, lstm, gru)
        """
        super(Encoder, self).__init__()
        
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell_type = cell_type.lower()
        
        # Embedding layer
        self.embedding = nn.Embedding(input_size, embedding_size)
        
        # RNN layer
        if self.cell_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=embedding_size, 
                hidden_size=hidden_size, 
                num_layers=num_layers, 
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True
            )
        elif self.cell_type == 'gru':
            self.rnn = nn.GRU(
                input_size=embedding_size, 
                hidden_size=hidden_size, 
                num_layers=num_layers, 
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True
            )
        else:  # default to RNN
            self.rnn = nn.RNN(
                input_size=embedding_size, 
                hidden_size=hidden_size, 
                num_layers=num_layers, 
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True
            )
        
        # Dropout for input
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, 
                src: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor] or Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of the encoder
        
        Args:
            src: Source sequence [batch_size, seq_len]
            
        Returns:
            outputs: Outputs of the RNN for each time step [batch_size, seq_len, hidden_size]
            hidden: Final hidden state [num_layers, batch_size, hidden_size]
                   or (hidden, cell) for LSTM
        """
        # Apply embedding and dropout
        embedded = self.dropout(self.embedding(src))  # [batch_size, seq_len, embedding_size]
        
        # Run through RNN
        outputs, hidden = self.rnn(embedded)
        
        return outputs, hidden


class Decoder(nn.Module):
    def __init__(self, 
                 output_size: int, 
                 embedding_size: int, 
                 hidden_size: int, 
                 num_layers: int = 1, 
                 dropout: float = 0.0, 
                 cell_type: str = 'gru'):
        """
        Decoder for the Seq2Seq model
        
        Args:
            output_size: Size of the output vocabulary
            embedding_size: Size of the embeddings
            hidden_size: Size of the hidden state
            num_layers: Number of layers in the RNN
            dropout: Dropout probability (only applied between layers)
            cell_type: Type of RNN cell (rnn, lstm, gru)
        """
        super(Decoder, self).__init__()
        
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.cell_type = cell_type.lower()
        
        # Embedding layer
        self.embedding = nn.Embedding(output_size, embedding_size)
        
        # RNN layer
        if self.cell_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=embedding_size, 
                hidden_size=hidden_size, 
                num_layers=num_layers, 
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True
            )
        elif self.cell_type == 'gru':
            self.rnn = nn.GRU(
                input_size=embedding_size, 
                hidden_size=hidden_size, 
                num_layers=num_layers, 
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True
            )
        else:  # default to RNN
            self.rnn = nn.RNN(
                input_size=embedding_size, 
                hidden_size=hidden_size, 
                num_layers=num_layers, 
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True
            )
        
        # Output layer
        self.fc_out = nn.Linear(hidden_size, output_size)
        
        # Dropout for input and between layers
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, 
                input: torch.Tensor, 
                hidden: torch.Tensor or Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor or Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of the decoder for a single time step
        
        Args:
            input: Input tensor [batch_size, 1]
            hidden: Hidden state from the encoder or previous decoder step
                    [num_layers, batch_size, hidden_size] or (h, c) for LSTM
            
        Returns:
            output: Output probabilities [batch_size, output_size]
            hidden: Hidden state for next time step
        """
        # Apply embedding and dropout
        embedded = self.dropout(self.embedding(input))  # [batch_size, 1, embedding_size]
        
        # Run through RNN
        output, hidden = self.rnn(embedded, hidden)
        
        # Pass through linear layer
        prediction = self.fc_out(output.squeeze(1))  # [batch_size, output_size]
        
        return prediction, hidden


class Seq2Seq(nn.Module):
    def __init__(self, 
                 encoder: Encoder, 
                 decoder: Decoder, 
                 device: torch.device,
                 sos_idx: int,
                 eos_idx: int):
        """
        Sequence-to-Sequence model
        
        Args:
            encoder: Encoder module
            decoder: Decoder module
            device: Device to use
            sos_idx: Start of sequence index
            eos_idx: End of sequence index
        """
        super(Seq2Seq, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
    
    def forward(self, 
                src: torch.Tensor, 
                trg: torch.Tensor, 
                teacher_forcing_ratio: float = 0.5) -> torch.Tensor:
        """
        Forward pass of the Seq2Seq model
        
        Args:
            src: Source sequence [batch_size, src_len]
            trg: Target sequence [batch_size, trg_len]
            teacher_forcing_ratio: Probability of using teacher forcing
            
        Returns:
            outputs: Decoder outputs [batch_size, trg_len-1, output_size]
        """
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_size
        
        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, trg_len - 1, trg_vocab_size).to(self.device)
        
        # Encode source sequence
        _, hidden = self.encoder(src)
        
        # First input to the decoder is the <SOS> token
        input = trg[:, 0].unsqueeze(1)  # [batch_size, 1]
        
        # Teacher forcing - use target as next input vs. use own prediction
        for t in range(1, trg_len):
            # Pass through decoder
            output, hidden = self.decoder(input, hidden)
            
            # Save output
            outputs[:, t-1] = output
            
            # Decide whether to use teacher forcing
            teacher_force = random.random() < teacher_forcing_ratio
            
            # Get the highest predicted token
            top1 = output.argmax(1).unsqueeze(1)
            
            # Use teacher forcing or own prediction
            input = trg[:, t].unsqueeze(1) if teacher_force else top1
        
        return outputs
    
    def beam_search(self, 
                     src: torch.Tensor, 
                     max_len: int, 
                     beam_size: int = 3) -> List[List[int]]:
        """
        Perform beam search decoding
        
        Args:
            src: Source sequence [1, src_len]
            max_len: Maximum length of the output sequence
            beam_size: Size of the beam
            
        Returns:
            best_sequence: Best decoded sequence as a list of indices
        """
        if src.shape[0] != 1:
            raise ValueError("Beam search can only be used with batch size 1")
        
        # Encode source sequence
        _, hidden = self.encoder(src)
        
        # Start with <SOS> token
        input = torch.tensor([[self.sos_idx]]).to(self.device)
        
        # Initialize beam with the first prediction
        output, hidden_state = self.decoder(input, hidden)
        log_probs = F.log_softmax(output, dim=1)
        
        # Get top beam_size candidates
        topk_log_probs, topk_indices = log_probs.topk(beam_size, dim=1)
        
        # Initialize beam candidates
        # Each candidate is (log_prob, sequence, hidden_state)
        candidates = []
        
        for i in range(beam_size):
            candidates.append((
                topk_log_probs[0, i].item(),
                [self.sos_idx, topk_indices[0, i].item()],
                hidden_state
            ))
        
        # Perform beam search
        for _ in range(2, max_len):
            next_candidates = []
            
            # Expand each current candidate
            for log_prob, seq, hidden_state in candidates:
                # If sequence ended with <EOS>, keep it as is
                if seq[-1] == self.eos_idx:
                    next_candidates.append((log_prob, seq, hidden_state))
                    continue
                
                # Get next prediction
                input = torch.tensor([[seq[-1]]]).to(self.device)
                output, new_hidden = self.decoder(input, hidden_state)
                log_probs = F.log_softmax(output, dim=1)
                
                # Get top beam_size candidates
                topk_log_probs, topk_indices = log_probs.topk(beam_size, dim=1)
                
                # Add new candidates
                for i in range(beam_size):
                    next_log_prob = log_prob + topk_log_probs[0, i].item()
                    next_seq = seq + [topk_indices[0, i].item()]
                    next_candidates.append((next_log_prob, next_seq, new_hidden))
            
            # Select beam_size best candidates
            next_candidates.sort(key=lambda x: x[0], reverse=True)
            candidates = next_candidates[:beam_size]
            
            # Check if all sequences have ended with <EOS>
            if all(seq[-1] == self.eos_idx for _, seq, _ in candidates):
                break
        
        # Return the best sequence
        best_seq = candidates[0][1]
        
        # Remove <SOS> and <EOS> tokens if present
        if best_seq[0] == self.sos_idx:
            best_seq = best_seq[1:]
        if best_seq[-1] == self.eos_idx:
            best_seq = best_seq[:-1]
            
        return best_seq

    def greedy_decode(self, 
                      src: torch.Tensor, 
                      max_len: int) -> List[int]:
        """
        Perform greedy decoding
        
        Args:
            src: Source sequence [1, src_len]
            max_len: Maximum length of the output sequence
            
        Returns:
            sequence: Decoded sequence as a list of indices
        """
        if src.shape[0] != 1:
            raise ValueError("Greedy decoding can only be used with batch size 1")
        
        # Encode source sequence
        _, hidden = self.encoder(src)
        
        # Start with <SOS> token
        input = torch.tensor([[self.sos_idx]]).to(self.device)
        
        # List to store output sequence
        sequence = [self.sos_idx]
        
        for _ in range(1, max_len):
            # Get next prediction
            output, hidden = self.decoder(input, hidden)
            
            # Get the highest predicted token
            top1 = output.argmax(1).item()
            
            # Add token to sequence
            sequence.append(top1)
            
            # Break if <EOS> is predicted
            if top1 == self.eos_idx:
                break
            
            # Use the predicted token as next input
            input = torch.tensor([[top1]]).to(self.device)
        
        # Remove <SOS> and <EOS> tokens if present
        if sequence[0] == self.sos_idx:
            sequence = sequence[1:]
        if sequence and sequence[-1] == self.eos_idx:
            sequence = sequence[:-1]
            
        return sequence


def create_model(src_vocab_size: int, 
                 trg_vocab_size: int,
                 embedding_size: int,
                 hidden_size: int,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 encoder_dropout: float,
                 decoder_dropout: float,
                 cell_type: str,
                 device: torch.device,
                 sos_idx: int,
                 eos_idx: int) -> Seq2Seq:
    """
    Create a Seq2Seq model
    
    Args:
        src_vocab_size: Size of the source vocabulary
        trg_vocab_size: Size of the target vocabulary
        embedding_size: Size of the embeddings
        hidden_size: Size of the hidden states
        num_encoder_layers: Number of layers in the encoder
        num_decoder_layers: Number of layers in the decoder
        encoder_dropout: Dropout probability for the encoder
        decoder_dropout: Dropout probability for the decoder
        cell_type: Type of RNN cell (rnn, lstm, gru)
        device: Device to use
        sos_idx: Start of sequence index
        eos_idx: End of sequence index
        
    Returns:
        model: Seq2Seq model
    """
    encoder = Encoder(
        input_size=src_vocab_size,
        embedding_size=embedding_size,
        hidden_size=hidden_size,
        num_layers=num_encoder_layers,
        dropout=encoder_dropout,
        cell_type=cell_type
    )
    
    decoder = Decoder(
        output_size=trg_vocab_size,
        embedding_size=embedding_size,
        hidden_size=hidden_size,
        num_layers=num_decoder_layers,
        dropout=decoder_dropout,
        cell_type=cell_type
    )
    
    model = Seq2Seq(
        encoder=encoder,
        decoder=decoder,
        device=device,
        sos_idx=sos_idx,
        eos_idx=eos_idx
    )
    
    return model