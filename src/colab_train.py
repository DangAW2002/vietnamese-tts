import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F  # Added import for F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import argparse
import random
import shutil
import glob
import gdown
import zipfile

# Import local modules
from models.tts_model import TTSModel

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

def collate_fn(batch):
    """
    Custom collate function to handle variable length sequences
    """
    # Sort by text length in descending order
    batch.sort(key=lambda x: x[0].shape[0], reverse=True)
    
    # Get max lengths
    text_lengths = [item[0].shape[0] for item in batch]
    max_text_len = max(text_lengths)
    
    feature_lengths = [item[1].shape[1] if len(item[1].shape) > 1 else item[1].shape[0] 
                      for item in batch]
    max_feature_len = max(feature_lengths)
    
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
        text_len = text.shape[0]
        batched_text[i, :text_len, :] = text
        
        feat_len = features.shape[1] if len(features.shape) > 1 else features.shape[0]
        
        if len(features.shape) > 1:
            batched_features[i, :, :feat_len] = features
        else:
            batched_features[i, :feat_len] = features
    
    return batched_text, batched_features, torch.tensor(text_lengths), torch.tensor(feature_lengths)

def download_vivos_processed(output_dir='data/processed'):
    """
    Download processed VIVOS data from Google Drive
    """
    # URL for the processed VIVOS data (replace with your actual Google Drive URL)
    url = "REPLACE_WITH_ACTUAL_GOOGLE_DRIVE_URL"
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Downloading processed VIVOS data...")
    zip_path = "vivos_processed.zip"
    gdown.download(url, zip_path, quiet=False)
    
    print("Extracting data...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    
    # Remove the zip file
    os.remove(zip_path)
    print(f"Data extracted to {output_dir}")

def download_config():
    """
    Download configuration file from Google Drive
    """
    # URL for the config file
    url = "REPLACE_WITH_CONFIG_GOOGLE_DRIVE_URL"
    
    # Create configs directory
    os.makedirs("configs", exist_ok=True)
    
    print("Downloading configuration file...")
    config_path = "configs/config.yaml"
    gdown.download(url, config_path, quiet=False)
    print(f"Config file downloaded to {config_path}")

def prepare_environment():
    """
    Prepare Colab environment
    """
    # Create necessary directories
    os.makedirs("checkpoints/logs", exist_ok=True)
    os.makedirs("checkpoints/models", exist_ok=True)
    
    # Check if processed data exists
    if not os.path.exists("data/processed/train") or not os.path.exists("data/processed/val"):
        download_vivos_processed()
    
    # Check if config exists
    if not os.path.exists("configs/config.yaml"):
        download_config()
    
    # Install required packages
    print("Installing required packages...")
    os.system("pip install tensorboard tqdm pyyaml gdown librosa")

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
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
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
        train_loss = 0.0
        
        # Training step
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for i, (text, features, text_lengths, feature_lengths) in enumerate(progress_bar):
            # Move data to device
            text = text.to(device)
            features = features.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass - check if model supports target_shape parameter
            try:
                outputs = model(text, target_shape=features.shape)
            except TypeError:
                # Fall back to regular forward pass without target_shape
                outputs = model(text)
                
                # Handle dimension mismatch manually if needed
                if outputs.shape[2] != features.shape[2]:
                    outputs = F.interpolate(
                        outputs, 
                        size=features.shape[2],
                        mode='linear',
                        align_corners=False
                    )
            
            # Compute loss (simple MSE for demonstration)
            loss = criterion(outputs, features)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update statistics
            train_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            
            # Log to tensorboard
            global_step = epoch * len(train_loader) + i
            writer.add_scalar('train/step_loss', loss.item(), global_step)
            
        # Calculate average training loss for the epoch
        avg_train_loss = train_loss / len(train_loader)
        writer.add_scalar('train/epoch_loss', avg_train_loss, epoch)
        
        # Validation step
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for text, features, text_lengths, feature_lengths in val_loader:
                # Move data to device
                text = text.to(device)
                features = features.to(device)
                
                # Forward pass - check if model supports target_shape parameter
                try:
                    outputs = model(text, target_shape=features.shape)
                except TypeError:
                    # Fall back to regular forward pass without target_shape
                    outputs = model(text)
                    
                    # Handle dimension mismatch manually if needed
                    if outputs.shape[2] != features.shape[2]:
                        outputs = F.interpolate(
                            outputs, 
                            size=features.shape[2],
                            mode='linear',
                            align_corners=False
                        )
                
                # Compute loss
                loss = criterion(outputs, features)
                val_loss += loss.item()
        
        # Calculate average validation loss
        avg_val_loss = val_loss / len(val_loader)
        writer.add_scalar('validation/epoch_loss', avg_val_loss, epoch)
        
        # Print statistics
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % config['training']['save_interval'] == 0:
            checkpoint_path = checkpoint_dir / f"{config['training']['model_name']}_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
            
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = checkpoint_dir / f"{config['training']['model_name']}_best.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss
            }, best_model_path)
            print(f"Best model saved to {best_model_path}")
    
    # Close tensorboard writer
    writer.close()
    print("Training completed!")

def main():
    # Prepare the Colab environment
    prepare_environment()
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train TTS model on Colab')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config file')
    parser.add_argument('--epochs', type=int, default=None, help='Override number of epochs')
    parser.add_argument('--batch_size', type=int, default=None, help='Override batch size')
    parser.add_argument('--lr', type=float, default=None, help='Override learning rate')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Override config with command-line arguments if provided
    if args.epochs:
        config['training']['num_epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.lr:
        config['training']['learning_rate'] = args.lr
    
    # Train model
    train_model(config)

if __name__ == "__main__":
    main()