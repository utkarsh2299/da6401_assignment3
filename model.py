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
                batch_first=True,
                bidirectional=True  # Bidirectional for better context capture
            )
        elif self.cell_type == 'gru':
            self.rnn = nn.GRU(
                input_size=embedding_size, 
                hidden_size=hidden_size, 
                num_layers=num_layers, 
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
                bidirectional=True  # Bidirectional for better context capture
            )
        else:  # default to RNN
            self.rnn = nn.RNN(
                input_size=embedding_size, 
                hidden_size=hidden_size, 
                num_layers=num_layers, 
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
                bidirectional=True  # Bidirectional for better context capture
            )
        
        # Projection layer to convert bidirectional hidden states to decoder hidden size
        self.fc = nn.Linear(hidden_size * 2, hidden_size)
        
        # Dropout for input
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the encoder
        
        Args:
            src: Source sequence [batch_size, seq_len]
            
        Returns:
            outputs: Outputs of the RNN for each time step [batch_size, seq_len, hidden_size*2]
            hidden: Final hidden state [num_layers, batch_size, hidden_size]
                   or (hidden, cell) for LSTM
        """
        # Apply embedding and dropout
        embedded = self.dropout(self.embedding(src))  # [batch_size, seq_len, embedding_size]
        
        # Run through RNN
        outputs, hidden = self.rnn(embedded)
        
        # Process hidden state for the decoder (combine bidirectional)
        if self.cell_type == 'lstm':
            # For LSTM, hidden is a tuple (h, c)
            h, c = hidden
            
            # Combine forward and backward directions
            # h shape: [num_layers*2, batch_size, hidden_size]
            h_forward = h[0::2]  # Forward direction: layers 0, 2, 4, ...
            h_backward = h[1::2]  # Backward direction: layers 1, 3, 5, ...
            h_combined = torch.cat((h_forward, h_backward), dim=2)  # [num_layers, batch_size, hidden_size*2]
            
            # Process combined hidden for decoder
            h_decoder = torch.tanh(self.fc(h_combined))  # [num_layers, batch_size, hidden_size]
            
            # Same for cell state
            c_forward = c[0::2]
            c_backward = c[1::2]
            c_combined = torch.cat((c_forward, c_backward), dim=2)
            c_decoder = torch.tanh(self.fc(c_combined))
            
            # Final hidden state
            hidden = (h_decoder, c_decoder)
        else:
            # For GRU/RNN, hidden is a tensor
            # hidden shape: [num_layers*2, batch_size, hidden_size]
            h_forward = hidden[0::2]  # Forward direction
            h_backward = hidden[1::2]  # Backward direction
            h_combined = torch.cat((h_forward, h_backward), dim=2)  # [num_layers, batch_size, hidden_size*2]
            
            # Process combined hidden for decoder
            hidden = torch.tanh(self.fc(h_combined))  # [num_layers, batch_size, hidden_size]
        
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

class Attention(nn.Module):
    def __init__(self, encoder_hidden_dim: int, decoder_hidden_dim: int):
        """
        Bahdanau Attention mechanism
        
        Args:
            encoder_hidden_dim: Size of encoder hidden state (hidden_size*2 for bidirectional)
            decoder_hidden_dim: Size of decoder hidden state
        """
        super(Attention, self).__init__()
        
        self.attn = nn.Linear(encoder_hidden_dim + decoder_hidden_dim, decoder_hidden_dim)
        self.v = nn.Linear(decoder_hidden_dim, 1, bias=False)
    
    def forward(self, hidden: torch.Tensor, encoder_outputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of attention mechanism
        
        Args:
            hidden: Current decoder hidden state [batch_size, decoder_hidden_dim]
            encoder_outputs: All encoder outputs [batch_size, src_len, encoder_hidden_dim]
            
        Returns:
            attention_weights: Attention weights [batch_size, src_len]
        """
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        
        # Repeat decoder hidden state for each source token
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)  # [batch_size, src_len, decoder_hidden_dim]
        
        # Combine encoder outputs and decoder hidden state
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))  # [batch_size, src_len, decoder_hidden_dim]
        
        # Get attention weights
        attention = self.v(energy).squeeze(2)  # [batch_size, src_len]
        
        # Return normalized attention weights
        return F.softmax(attention, dim=1)  # [batch_size, src_len]


class AttentionDecoder(nn.Module):
    def __init__(self, 
                 output_size: int, 
                 embedding_size: int, 
                 encoder_hidden_size: int,
                 decoder_hidden_size: int, 
                 num_layers: int = 1, 
                 dropout: float = 0.0, 
                 cell_type: str = 'gru'):
        """
        Decoder with attention for the Seq2Seq model
        
        Args:
            output_size: Size of the output vocabulary
            embedding_size: Size of the embeddings
            encoder_hidden_size: Size of encoder hidden states (hidden_size*2 for bidirectional)
            decoder_hidden_size: Size of the decoder hidden state
            num_layers: Number of layers in the RNN
            dropout: Dropout probability (only applied between layers)
            cell_type: Type of RNN cell (rnn, lstm, gru)
        """
        super(AttentionDecoder, self).__init__()
        
        self.embedding_size = embedding_size
        self.decoder_hidden_size = decoder_hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.cell_type = cell_type.lower()
        
        # Embedding layer
        self.embedding = nn.Embedding(output_size, embedding_size)
        
        # Attention mechanism
        self.attention = Attention(encoder_hidden_size, decoder_hidden_size)
        
        # Input to RNN will be embedding + weighted encoder outputs
        rnn_input_size = embedding_size + encoder_hidden_size
        
        # RNN layer
        if self.cell_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=rnn_input_size, 
                hidden_size=decoder_hidden_size, 
                num_layers=num_layers, 
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True
            )
        elif self.cell_type == 'gru':
            self.rnn = nn.GRU(
                input_size=rnn_input_size, 
                hidden_size=decoder_hidden_size, 
                num_layers=num_layers, 
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True
            )
        else:  # default to RNN
            self.rnn = nn.RNN(
                input_size=rnn_input_size, 
                hidden_size=decoder_hidden_size, 
                num_layers=num_layers, 
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True
            )
        
        # Output layer
        self.fc_out = nn.Linear(decoder_hidden_size + encoder_hidden_size + embedding_size, output_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, 
                input: torch.Tensor, 
                hidden: torch.Tensor, 
                encoder_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the attention decoder for a single time step
        
        Args:
            input: Input tensor [batch_size, 1]
            hidden: Hidden state from the previous step 
                  [num_layers, batch_size, hidden_size] or (h, c) for LSTM
            encoder_outputs: All encoder outputs [batch_size, src_len, hidden_size*2]
            
        Returns:
            prediction: Output probabilities [batch_size, output_size]
            hidden: Updated hidden state
            attention_weights: Attention weights for visualization
        """
        # Get embedding of input
        embedded = self.dropout(self.embedding(input))  # [batch_size, 1, embedding_size]
        
        # Get top layer hidden state for attention
        if self.cell_type == 'lstm':
            top_hidden = hidden[0][-1].unsqueeze(0)  # [1, batch_size, hidden_size]
            attn_hidden = top_hidden.permute(1, 0, 2).squeeze(1)  # [batch_size, hidden_size]
        else:
            attn_hidden = hidden[-1]  # [batch_size, hidden_size]
        
        # Calculate attention weights
        attention_weights = self.attention(attn_hidden, encoder_outputs)  # [batch_size, src_len]
        
        # Apply attention weights to encoder outputs
        attention_weights = attention_weights.unsqueeze(1)  # [batch_size, 1, src_len]
        context_vector = torch.bmm(attention_weights, encoder_outputs)  # [batch_size, 1, encoder_hidden_dim]
        
        # Combine embedding and context vector as input to RNN
        rnn_input = torch.cat((embedded, context_vector), dim=2)  # [batch_size, 1, embedding_size + encoder_hidden_dim]
        
        # Pass through RNN
        output, hidden = self.rnn(rnn_input, hidden)  # output: [batch_size, 1, decoder_hidden_dim]
        
        # Combine RNN output, context vector, and embedding for prediction
        embedded = embedded.squeeze(1)  # [batch_size, embedding_size]
        output = output.squeeze(1)  # [batch_size, decoder_hidden_dim]
        context_vector = context_vector.squeeze(1)  # [batch_size, encoder_hidden_dim]
        
        prediction_input = torch.cat((output, context_vector, embedded), dim=1)  # [batch_size, decoder_hidden_dim + encoder_hidden_dim + embedding_size]
        
        # Pass through output layer
        prediction = self.fc_out(prediction_input)  # [batch_size, output_size]
        
        return prediction, hidden, attention_weights


class AttentionSeq2Seq(nn.Module):
    def __init__(self, 
                 encoder: Encoder, 
                 decoder: AttentionDecoder, 
                 device: torch.device,
                 sos_idx: int,
                 eos_idx: int):
        """
        Sequence-to-Sequence model with Attention
        
        Args:
            encoder: Encoder module
            decoder: Attention decoder module
            device: Device to use
            sos_idx: Start of sequence index
            eos_idx: End of sequence index
        """
        super(AttentionSeq2Seq, self).__init__()
        
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
        Forward pass of the Seq2Seq model with attention
        
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
        
        # Tensor to store attention weights (for visualization)
        attentions = torch.zeros(batch_size, trg_len - 1, src.shape[1]).to(self.device)
        
        # Encode source sequence
        encoder_outputs, hidden = self.encoder(src)
        
        # First input to the decoder is the < SOS > token
        input = trg[:, 0].unsqueeze(1)  # [batch_size, 1]
        
        # Teacher forcing - use target as next input vs. use own prediction
        for t in range(1, trg_len):
            # Pass through decoder with attention
            output, hidden, attn_weights = self.decoder(input, hidden, encoder_outputs)
            
            # Save output and attention weights
            outputs[:, t-1] = output
            attentions[:, t-1] = attn_weights.squeeze(1)
            
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
                     beam_size: int = 3) -> List[int]:
        """
        Perform beam search decoding with attention
        
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
        encoder_outputs, hidden = self.encoder(src)
        
        # Start with < SOS > token
        input = torch.tensor([[self.sos_idx]]).to(self.device)
        
        # Initialize beam with the first prediction
        output, hidden_state, _ = self.decoder(input, hidden, encoder_outputs)
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
                output, new_hidden, _ = self.decoder(input, hidden_state, encoder_outputs)
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
        
        # Remove < SOS > and <EOS> tokens if present
        if best_seq[0] == self.sos_idx:
            best_seq = best_seq[1:]
        if best_seq[-1] == self.eos_idx:
            best_seq = best_seq[:-1]
            
        return best_seq

    def greedy_decode(self, src: torch.Tensor, max_len: int) -> Tuple[List[int], List[torch.Tensor]]:
        """
        Perform greedy decoding with attention and return attention weights
        
        Args:
            src: Source sequence [1, src_len]
            max_len: Maximum length of the output sequence
            
        Returns:
            sequence: Decoded sequence as a list of indices
            attention_weights: List of attention weights for each output step
        """
        if src.shape[0] != 1:
            raise ValueError("Greedy decoding can only be used with batch size 1")
        
        # Encode source sequence
        encoder_outputs, hidden = self.encoder(src)
        
        # Start with < SOS > token
        input = torch.tensor([[self.sos_idx]]).to(self.device)
        
        # Lists to store output sequence and attention weights
        sequence = [self.sos_idx]
        attention_weights = []
        
        for _ in range(1, max_len):
            # Get next prediction with attention
            output, hidden, attn_weights = self.decoder(input, hidden, encoder_outputs)
            
            # Store attention weights
            attention_weights.append(attn_weights.squeeze().detach().cpu())
            
            # Get the highest predicted token
            top1 = output.argmax(1).item()
            
            # Add token to sequence
            sequence.append(top1)
            
            # Break if <EOS> is predicted
            if top1 == self.eos_idx:
                break
            
            # Use the predicted token as next input
            input = torch.tensor([[top1]]).to(self.device)
        
        # Remove < SOS > and <EOS> tokens if present
        if sequence[0] == self.sos_idx:
            sequence = sequence[1:]
        if sequence and sequence[-1] == self.eos_idx:
            sequence = sequence[:-1]
            
        return sequence, attention_weights


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
                 eos_idx: int,
                 use_attention: bool = True) -> nn.Module:
    """
    Create a Seq2Seq model with or without attention
    
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
        use_attention: Whether to use attention mechanism
        
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
    
    if use_attention:
        # With bidirectional encoder, the outputs have 2*hidden_size dimensions
        encoder_hidden_size = hidden_size * 2
        print("inside attention")
        decoder = AttentionDecoder(
            output_size=trg_vocab_size,
            embedding_size=embedding_size,
            encoder_hidden_size=encoder_hidden_size,
            decoder_hidden_size=hidden_size,
            num_layers=num_decoder_layers,
            dropout=decoder_dropout,
            cell_type=cell_type
        )
        
        model = AttentionSeq2Seq(
            encoder=encoder,
            decoder=decoder,
            device=device,
            sos_idx=sos_idx,
            eos_idx=eos_idx
        )
    else:
        # Use original non-attention model
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