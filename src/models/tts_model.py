import torch
import torch.nn as nn

class TextEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        """
        Text encoder network
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension
        """
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
    def forward(self, x):
        return self.lstm(x)[0]

class AudioDecoder(nn.Module):
    def __init__(self, hidden_dim: int, output_dim: int):
        """
        Audio decoder network
        Args:
            hidden_dim: Hidden dimension
            output_dim: Output dimension
        """
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=hidden_dim * 2,  # bidirectional encoder
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True
        )
        self.projection = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, encoder_output):
        decoder_output, _ = self.lstm(encoder_output)
        return self.projection(decoder_output)

class TTSModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        """
        Text-to-Speech Model
        Args:
            input_dim: Input dimension for text features
            hidden_dim: Hidden dimension
            output_dim: Output dimension for audio features
        """
        super().__init__()
        self.encoder = TextEncoder(input_dim, hidden_dim)
        self.decoder = AudioDecoder(hidden_dim, output_dim)
        
    def forward(self, x):
        features = self.encoder(x)
        audio = self.decoder(features)
        return audio