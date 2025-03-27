import os
import yaml
import torch
import numpy as np
import re
from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple, Dict

# Fix relative import
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.audio_utils import AudioProcessor

class VivosDataPreprocessor:
    def __init__(self, config_path: str):
        """
        Initialize data preprocessor for VIVOS dataset
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.audio_processor = AudioProcessor(
            sample_rate=self.config['audio']['sample_rate']
        )

    def load_vivos_data(self, data_dir: str) -> Tuple[List[str], Dict[str, str]]:
        """
        Load VIVOS data from directory
        Args:
            data_dir: Base directory containing VIVOS dataset (e.g., data/raw/vivos)
        Returns:
            audio_files: List of audio file paths
            transcriptions: Dictionary mapping audio file IDs to transcriptions
        """
        data_dir = Path(data_dir)
        audio_files = []
        transcriptions = {}
        
        # Process both train and test sets
        for subset in ['train', 'test']:
            subset_dir = data_dir / subset
            
            # Load prompts file (transcriptions)
            prompts_file = subset_dir / "prompts.txt"
            with open(prompts_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split(' ', 1)
                    if len(parts) == 2:
                        file_id, text = parts
                        transcriptions[file_id] = text
            
            # Find all wav files
            waves_dir = subset_dir / "waves"
            for speaker_dir in waves_dir.iterdir():
                if speaker_dir.is_dir():
                    for wav_file in speaker_dir.glob("*.wav"):
                        audio_files.append(wav_file)
        
        print(f"Found {len(audio_files)} audio files and {len(transcriptions)} transcriptions")
        return audio_files, transcriptions

    def process_audio(self, audio_path: str) -> torch.Tensor:
        """
        Process audio file
        Args:
            audio_path: Path to audio file
        Returns:
            features: Mel spectrogram features
        """
        audio, _ = self.audio_processor.load_audio(audio_path)
        features = self.audio_processor.extract_features(audio)
        return features

    def process_text(self, text: str) -> str:
        """
        Basic text processing
        Args:
            text: Raw text
        Returns:
            processed_text: Processed text
        """
        # Basic text normalization (remove extra spaces)
        processed_text = re.sub(r'\s+', ' ', text).strip()
        return processed_text

    def process_and_save(self, 
                        audio_files: List[Path], 
                        transcriptions: Dict[str, str],
                        output_dir: str) -> None:
        """
        Process and save data
        Args:
            audio_files: List of audio files
            transcriptions: Dictionary mapping audio file IDs to transcriptions
            output_dir: Output directory
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create train, val, and test directories
        train_dir = output_dir / 'train'
        val_dir = output_dir / 'val'
        test_dir = output_dir / 'test'
        
        for directory in [train_dir, val_dir, test_dir]:
            directory.mkdir(exist_ok=True)
        
        processed_count = 0
        skipped_count = 0
        
        for audio_file in tqdm(audio_files, desc="Processing VIVOS data"):
            # Extract file ID from path (e.g., VIVOSDEV01_R002)
            file_id = audio_file.stem
            
            # Skip if no transcription available
            if file_id not in transcriptions:
                skipped_count += 1
                continue
            
            # Process audio
            features = self.process_audio(str(audio_file))
            
            # Process text (basic processing)
            text = self.process_text(transcriptions[file_id])
            
            # Determine output directory (test or train)
            if "test" in str(audio_file):
                # Split test data: 80% for test, 20% for validation
                if processed_count % 5 == 0:
                    out_dir = val_dir
                else:
                    out_dir = test_dir
            else:
                out_dir = train_dir
                
            # Save processed data
            save_data = {
                'features': features,
                'text': text,
                'file_id': file_id,
                'original_path': str(audio_file)
            }
            
            torch.save(save_data, out_dir / f'{file_id}.pt')
            
            processed_count += 1
            
        print(f"Processed {processed_count} files, skipped {skipped_count} files")
        print(f"Data saved to {output_dir}")

def main():
    """Main function to run VIVOS data preprocessing"""
    config_path = 'configs/config.yaml'
    preprocessor = VivosDataPreprocessor(config_path)
    
    # Load VIVOS data
    vivos_dir = 'data/raw/vivos'
    audio_files, transcriptions = preprocessor.load_vivos_data(vivos_dir)
    
    # Process and save
    preprocessor.process_and_save(audio_files, transcriptions, 'data/processed')

if __name__ == "__main__":
    main()