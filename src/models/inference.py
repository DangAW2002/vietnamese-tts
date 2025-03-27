import os
import argparse
import torch
import numpy as np
import yaml
from pathlib import Path
from tts_model import TTSModel
import sys
import librosa
from typing import Dict

# Add src directory to path so we can import from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.audio_utils import AudioProcessor

def load_model(checkpoint_path, config_path):
    """
    Load the trained model from a checkpoint
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading model from checkpoint: {checkpoint_path}")
    
    # Load config
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Initialize model with config parameters
    model = TTSModel(
        input_dim=config['model']['input_dim'],
        hidden_dim=config['model']['hidden_dim'],
        output_dim=config['model']['output_dim']
    )
    
    # Load the saved weights
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set model to evaluation mode
    model.eval()
    
    print("Model loaded successfully")
    return model, config

def text_to_tensor(text):
    """
    Convert input text to model input tensor
    """
    # Create simple text encoding (one-hot encoding)
    char_ids = [ord(c) % 128 for c in text]
    embedding = torch.zeros(len(char_ids), 256)
    for i, char_id in enumerate(char_ids):
        embedding[i, char_id] = 1.0
    
    # Add batch dimension
    return embedding.unsqueeze(0)

def generate_speech(model, text, config, output_path="output.wav"):
    """
    Generate speech from input text using the trained model
    """
    print(f"Generating speech for: '{text}'")
    
    # Convert text to tensor
    input_tensor = text_to_tensor(text)
    print(f"Input tensor shape: {input_tensor.shape}")
    
    # Generate mel spectrogram
    with torch.no_grad():
        mel_outputs = model(input_tensor)
        print(f"Model output shape: {mel_outputs.shape}")
        
        # Ensure output is positive
        mel_outputs = torch.abs(mel_outputs)
        
        # Convert to numpy and transpose if needed
        mel_spec = mel_outputs.squeeze(0).numpy()
        if len(mel_spec.shape) == 3:
            mel_spec = mel_spec.squeeze(0)
        print(f"Mel spectrogram shape after processing: {mel_spec.shape}")
        print(f"Mel spectrogram stats - min: {mel_spec.min():.2f}, max: {mel_spec.max():.2f}, mean: {mel_spec.mean():.2f}")
    
    # Initialize audio processor
    audio_processor = AudioProcessor(config)
    
    # Convert mel spectrogram to audio
    waveform = audio_processor.mel_to_audio(mel_spec)
    
    # Save audio
    audio_processor.save_wav(waveform, output_path)
    
    return output_path

def list_checkpoints(checkpoints_dir):
    """
    List all available checkpoints
    """
    checkpoints = []
    if os.path.exists(checkpoints_dir):
        for file in os.listdir(checkpoints_dir):
            if file.endswith('.pt') or file.endswith('.pth'):
                checkpoints.append(os.path.join(checkpoints_dir, file))
    return checkpoints

def main():
    parser = argparse.ArgumentParser(description="Run inference with trained TTS model")
    parser.add_argument("--checkpoint", help="Path to model checkpoint. If not provided, will use latest checkpoint.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--text", default="Xin chào, đây là giọng nói tiếng Việt.", 
                      help="Text to synthesize")
    parser.add_argument("--output", default="output.wav", 
                      help="Output audio file path")
    parser.add_argument("--list-checkpoints", action="store_true",
                      help="List all available checkpoints")
    
    args = parser.parse_args()
    
    # Get project root directory
    root_dir = Path(__file__).parent.parent.parent
    checkpoints_dir = root_dir / "checkpoints" / "models"
    
    # List checkpoints if requested
    if args.list_checkpoints:
        print("\nAvailable checkpoints:")
        checkpoints = list_checkpoints(checkpoints_dir)
        for i, ckpt in enumerate(checkpoints, 1):
            print(f"{i}. {os.path.basename(ckpt)}")
        return
    
    # If no checkpoint specified, use the latest one
    if args.checkpoint is None:
        checkpoints = list_checkpoints(checkpoints_dir)
        if not checkpoints:
            print("No checkpoints found in checkpoints/models/")
            return
        args.checkpoint = max(checkpoints, key=os.path.getctime)
        print(f"Using latest checkpoint: {args.checkpoint}")
    
    # Load the model
    model, config = load_model(args.checkpoint, args.config)
    
    # Generate speech
    output_path = generate_speech(model, args.text, config, args.output)
    
    print(f"Speech synthesis completed. Output saved to: {output_path}")

if __name__ == "__main__":
    main()