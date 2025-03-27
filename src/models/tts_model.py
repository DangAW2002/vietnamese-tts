import torch
import torch.nn as nn
import torch.nn.functional as F

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
        Text-to-Speech model
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension
            output_dim: Output dimension
        """
        super().__init__()
        self.encoder = TextEncoder(input_dim, hidden_dim)
        self.decoder = AudioDecoder(hidden_dim, output_dim)
        
    def forward(self, x, target_shape=None):
        encoder_output = self.encoder(x)
        decoder_output = self.decoder(encoder_output)
        
        # Transpose the output to match the target shape: [batch_size, time, features] -> [batch_size, features, time]
        decoder_output = decoder_output.transpose(1, 2)
        
        # If target_shape is provided, resize the time dimension to match
        if target_shape is not None:
            # Assuming target_shape is a tuple with batch_size, feature_dim, time_dim
            target_time_dim = target_shape[2]
            current_time_dim = decoder_output.shape[2]
            
            if current_time_dim != target_time_dim:
                # Use interpolation to match the target time dimension
                decoder_output = F.interpolate(
                    decoder_output, 
                    size=target_time_dim, 
                    mode='linear', 
                    align_corners=False
                )
        
        return decoder_output