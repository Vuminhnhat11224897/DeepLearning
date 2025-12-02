"""
Encoder-Decoder (Seq2Seq) Model Architecture
"""
import torch
import torch.nn as nn
from typing import Tuple, Optional

class Encoder(nn.Module):
    """
    LSTM Encoder for sequence-to-sequence model
    
    Args:
        input_size: Number of input features
        hidden_size: Hidden state dimension
        num_layers: Number of LSTM layers
        dropout: Dropout probability
        bidirectional: Use bidirectional LSTM
    """
    def __init__(self, 
                 input_size: int,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.2,
                 bidirectional: bool = True):
        super(Encoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # If bidirectional, project back to hidden_size
        if bidirectional:
            self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size)
            self.fc_cell = nn.Linear(hidden_size * 2, hidden_size)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
        
        Returns:
            outputs: All hidden states, shape (batch, seq_len, hidden_size * num_directions)
            (hidden, cell): Final states, each shape (num_layers, batch, hidden_size)
        """
        # outputs: (batch, seq_len, hidden_size * num_directions)
        # hidden, cell: (num_layers * num_directions, batch, hidden_size)
        outputs, (hidden, cell) = self.lstm(x)
        
        if self.bidirectional:
            # Combine forward and backward hidden states
            # hidden: (num_layers * 2, batch, hidden_size) -> (num_layers, batch, hidden_size * 2)
            batch_size = hidden.shape[1]
            
            # Reshape to (num_layers, 2, batch, hidden_size)
            hidden = hidden.view(self.num_layers, 2, batch_size, self.hidden_size)
            cell = cell.view(self.num_layers, 2, batch_size, self.hidden_size)
            
            # Concatenate forward and backward: (num_layers, batch, hidden_size * 2)
            hidden = torch.cat([hidden[:, 0, :, :], hidden[:, 1, :, :]], dim=2)
            cell = torch.cat([cell[:, 0, :, :], cell[:, 1, :, :]], dim=2)
            
            # Project to hidden_size: (num_layers, batch, hidden_size)
            hidden = self.fc_hidden(hidden)
            cell = self.fc_cell(cell)
        
        return outputs, (hidden, cell)


class Decoder(nn.Module):
    """
    LSTM Decoder for sequence-to-sequence model
    
    Args:
        output_size: Number of output features (1 for single target)
        hidden_size: Hidden state dimension (must match encoder)
        num_layers: Number of LSTM layers
        dropout: Dropout probability
    """
    def __init__(self,
                 output_size: int = 1,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.2):
        super(Decoder, self).__init__()
        
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, 
                x: torch.Tensor, 
                hidden: torch.Tensor, 
                cell: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for one timestep
        
        Args:
            x: Input tensor of shape (batch, 1, output_size)
            hidden: Hidden state from previous step, shape (num_layers, batch, hidden_size)
            cell: Cell state from previous step, shape (num_layers, batch, hidden_size)
        
        Returns:
            prediction: Output prediction, shape (batch, 1, output_size)
            (hidden, cell): Updated states
        """
        output, (hidden, cell) = self.lstm(x, (hidden, cell))
        prediction = self.fc(output)
        
        return prediction, (hidden, cell)


class Seq2Seq(nn.Module):
    """
    Encoder-Decoder Sequence-to-Sequence Model
    
    Args:
        encoder: Encoder module
        decoder: Decoder module
        output_seq_len: Number of timesteps to predict
        device: Computation device
    """
    def __init__(self,
                 encoder: Encoder,
                 decoder: Decoder,
                 output_seq_len: int = 5,
                 device: torch.device = None):
        super(Seq2Seq, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.output_seq_len = output_seq_len
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def forward(self, 
                x: torch.Tensor, 
                target: Optional[torch.Tensor] = None,
                teacher_forcing_ratio: float = 0.5) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input sequence, shape (batch, input_seq_len, n_features)
            target: Target sequence for teacher forcing, shape (batch, output_seq_len)
            teacher_forcing_ratio: Probability of using teacher forcing
        
        Returns:
            outputs: Predictions, shape (batch, output_seq_len)
        """
        batch_size = x.shape[0]
        
        # Encode input sequence
        _, (hidden, cell) = self.encoder(x)
        
        # Initialize decoder input with last known value (or zeros)
        # Use the target column value from the last timestep of input
        decoder_input = x[:, -1, 0].unsqueeze(1).unsqueeze(2)  # (batch, 1, 1)
        
        # Store outputs
        outputs = []
        
        for t in range(self.output_seq_len):
            # Decode one timestep
            prediction, (hidden, cell) = self.decoder(decoder_input, hidden, cell)
            outputs.append(prediction.squeeze(2))  # (batch, 1)
            
            # Teacher forcing
            if target is not None and torch.rand(1).item() < teacher_forcing_ratio:
                decoder_input = target[:, t].unsqueeze(1).unsqueeze(2)  # (batch, 1, 1)
            else:
                decoder_input = prediction
        
        # Concatenate outputs: (batch, output_seq_len)
        outputs = torch.cat(outputs, dim=1)
        
        return outputs


def build_model(input_size: int,
                hidden_size: int = 128,
                num_layers: int = 2,
                dropout: float = 0.2,
                bidirectional: bool = True,
                output_seq_len: int = 5,
                device: torch.device = None) -> Seq2Seq:
    """
    Build Encoder-Decoder model
    
    Args:
        input_size: Number of input features
        hidden_size: Hidden dimension
        num_layers: Number of LSTM layers
        dropout: Dropout probability
        bidirectional: Use bidirectional encoder
        output_seq_len: Number of prediction steps
        device: Computation device
    
    Returns:
        Seq2Seq model
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    encoder = Encoder(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional
    )
    
    decoder = Decoder(
        output_size=1,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    )
    
    model = Seq2Seq(
        encoder=encoder,
        decoder=decoder,
        output_seq_len=output_seq_len,
        device=device
    )
    
    model = model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model built on {device}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model
