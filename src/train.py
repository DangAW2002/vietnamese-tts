import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import argparse
import random

# Import local modules
from models.tts_model import TTSModel
from utils.audio_utils import AudioProcessor

class ProcessedAudioDataset(Dataset):
    def __init__(self, data_dir):
        """
        Dataset for processed audio data
        Args:
            data_dir: Directory containing processed .pt files
        """
        self.data_dir = Path(data_dir)
        self.file_list = list(self.data_dir.glob('*.pt'))
        print(f"Found {len(self.file_list)} files in {data_dir}")
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        data = torch.load(self.file_list[idx])
        
        # Get audio features
        features = data['features']
        
        # Create simple text embedding (one-hot encoding)
        text = data['text']
        text_encoded = self.simple_text_encoding(text)
        
        return text_encoded, features
    
    def simple_text_encoding(self, text):
        """
        Simple text encoding method (placeholder for more sophisticated encodings)
        """
        # Convert characters to ASCII values and normalize
        char_ids = [ord(c) % 128 for c in text]
        # Create a simple one-hot like embedding (256 dim for simplicity)
        embedding = torch.zeros(len(char_ids), 256)
        for i, char_id in enumerate(char_ids):
            embedding[i, char_id] = 1.0
        return embedding

def collate_fn(batch, pad_to_fixed_length=False, max_seq_len=None):
    """
    Custom collate function to handle variable length sequences
    Args:
        batch: Batch of data
        pad_to_fixed_length: Whether to pad sequences to a fixed length
        max_seq_len: Maximum sequence length for padding
    """
    # Sort by text length in descending order
    batch.sort(key=lambda x: x[0].shape[0], reverse=True)
    
    # Get max lengths
    text_lengths = [item[0].shape[0] for item in batch]
    max_text_len = min(max(text_lengths), max_seq_len) if pad_to_fixed_length and max_seq_len else max(text_lengths)
    
    feature_lengths = [item[1].shape[1] if len(item[1].shape) > 1 else item[1].shape[0] 
                      for item in batch]
    max_feature_len = min(max(feature_lengths), max_seq_len) if pad_to_fixed_length and max_seq_len else max(feature_lengths)
    
    # Prepare batched tensors
    batch_size = len(batch)
    text_dim = batch[0][0].shape[1]
    
    # Handle different feature shapes
    if len(batch[0][1].shape) > 1:
        feature_dim = batch[0][1].shape[0]
        batched_features = torch.zeros(batch_size, feature_dim, max_feature_len)
    else:
        feature_dim = 1
        batched_features = torch.zeros(batch_size, max_feature_len)
    
    batched_text = torch.zeros(batch_size, max_text_len, text_dim)
    
    # Fill in the data
    for i, (text, features) in enumerate(batch):
        text_len = min(text.shape[0], max_text_len)
        batched_text[i, :text_len, :] = text[:text_len]
        
        if len(features.shape) > 1:
            feat_len = min(features.shape[1], max_feature_len)
            batched_features[i, :, :feat_len] = features[:, :feat_len]
        else:
            feat_len = min(features.shape[0], max_feature_len)
            batched_features[i, :feat_len] = features[:feat_len]
    
    return batched_text, batched_features, torch.tensor(text_lengths), torch.tensor(feature_lengths)

def train_model(config):
    """
    Train the TTS model
    Args:
        config: Configuration dictionary
    """
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Create directories for logs and checkpoints
    log_dir = Path(config['training']['checkpoint_dir']) / 'logs'
    checkpoint_dir = Path(config['training']['checkpoint_dir']) / 'models'
    
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize tensorboard writer
    writer = SummaryWriter(log_dir=log_dir)
    
    # Initialize datasets and dataloaders
    train_dataset = ProcessedAudioDataset(config['data']['train_data'])
    val_dataset = ProcessedAudioDataset(config['data']['val_data'])
    
    # Get padding parameters from config
    pad_to_fixed_length = config.get('data_processing', {}).get('pad_to_fixed_length', False)
    max_seq_len = config.get('data_processing', {}).get('max_seq_len', None)
    
    # Create custom collate functions with set parameters
    def train_collate(batch):
        return collate_fn(batch, pad_to_fixed_length, max_seq_len)
    
    def val_collate(batch):
        return collate_fn(batch, pad_to_fixed_length, max_seq_len)
    
    print(f"Using padding: {pad_to_fixed_length}, max_seq_len: {max_seq_len}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=train_collate,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=val_collate,
        num_workers=2
    )
    
    # Initialize model
    model = TTSModel(
        input_dim=config['model']['input_dim'],
        hidden_dim=config['model']['hidden_dim'],
        output_dim=config['model']['output_dim']
    )
    
    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = model.to(device)
    
    # Initialize optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    criterion = nn.MSELoss()
    
    # Training loop
    num_epochs = config['training']['num_epochs']
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        train_loss = 0.0
        
        for batch in progress_bar:
            optimizer.zero_grad()
            text, features, text_lengths, feature_lengths = batch
            text = text.to(device)
            features = features.to(device)
            
            # Pass the target shape to the model
            outputs = model(text, target_shape=features.shape)
            loss = criterion(outputs, features)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.2e}")
            
        avg_train_loss = train_loss / len(train_loader)
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for text, features, text_lengths, feature_lengths in val_loader:
                text = text.to(device)
                features = features.to(device)
                
                # Forward pass - pass features too for potential resizing
                outputs = model(text, target_shape=features.shape)
                
                # Compute loss
                loss = criterion(outputs, features)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        
        print(f"Epoch {epoch}/{num_epochs}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Save checkpoint if validation loss improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
    
    print("Training completed!")

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train TTS model')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config file')
    parser.add_argument('--batch_size', type=int, help='Batch size for training')
    parser.add_argument('--pad_to_fixed_length', type=lambda x: x.lower() == 'true', help='Whether to pad sequences to fixed length')
    parser.add_argument('--max_seq_len', type=int, help='Maximum sequence length for padding')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Override config with command line arguments if provided
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
        print(f"Overriding batch size with command line argument: {args.batch_size}")
    
    if args.pad_to_fixed_length is not None and args.max_seq_len is not None:
        if 'data_processing' not in config:
            config['data_processing'] = {}
        config['data_processing']['pad_to_fixed_length'] = args.pad_to_fixed_length
        config['data_processing']['max_seq_len'] = args.max_seq_len
        print(f"Setting padding to fixed length: {args.pad_to_fixed_length}, max sequence length: {args.max_seq_len}")
    
    # Train model
    train_model(config)

if __name__ == "__main__":
    main()