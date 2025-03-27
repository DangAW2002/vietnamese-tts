import torch
import librosa
import numpy as np
from typing import Tuple, Optional, Dict

class AudioProcessor:
    def __init__(self, sample_rate: int = 22050, config: Dict = None):
        """
        Initialize audio processor
        Args:
            sample_rate: Sampling frequency
            config: Configuration dictionary
        """
        self.sample_rate = sample_rate
        self.config = config or {}

    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Read audio file and resample if needed
        Args:
            file_path: Path to audio file
        Returns:
            audio: Audio data
            sr: Sample rate
        """
        audio, sr = librosa.load(file_path, sr=None)
        if sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
        return audio, self.sample_rate

    def save_audio(self, audio: np.ndarray, file_path: str) -> None:
        """
        Save audio array to file
        Args:
            audio: Audio data
            file_path: Path to save file
        """
        librosa.output.write_wav(file_path, audio, self.sample_rate)

    def extract_features(self, audio: np.ndarray) -> torch.Tensor:
        """
        Extract mel spectrogram features
        Args:
            audio: Audio data
        Returns:
            mel_specs: Mel spectrogram features
        """
        # Default parameters for mel spectrogram extraction
        n_fft = self.config.get('audio', {}).get('n_fft', 1024)
        hop_length = self.config.get('audio', {}).get('hop_length', 256)
        win_length = self.config.get('audio', {}).get('win_length', 1024)
        n_mels = self.config.get('audio', {}).get('mel_channels', 80)
        
        mel_specs = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels
        )
        mel_specs = librosa.power_to_db(mel_specs, ref=np.max)
        
        return torch.FloatTensor(mel_specs)

    def extract_all_features(self, audio: np.ndarray) -> Dict[str, torch.Tensor]:
        """
        Extract comprehensive audio features
        Args:
            audio: Audio data
        Returns:
            features: Dictionary containing various feature types
        """
        features = {}
        
        # Extract mel spectrogram
        mel_specs = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.config.get('audio', {}).get('n_fft', 1024),
            hop_length=self.config.get('audio', {}).get('hop_length', 256),
            n_mels=self.config.get('audio', {}).get('mel_channels', 80)
        )
        features['mel'] = torch.FloatTensor(librosa.power_to_db(mel_specs, ref=np.max))
        
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(
            y=audio, 
            sr=self.sample_rate,
            n_mfcc=13
        )
        features['mfcc'] = torch.FloatTensor(mfcc)
        
        # Extract energy contour
        energy = np.sum(mel_specs, axis=0)
        features['energy'] = torch.FloatTensor(energy)
        
        return features


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, audio_files: list, text_files: Optional[list] = None, config: Dict = None):
        """
        Dataset for audio and text data
        Args:
            audio_files: List of audio files
            text_files: List of corresponding text files (optional)
            config: Configuration dictionary
        """
        self.audio_files = audio_files
        self.text_files = text_files
        self.processor = AudioProcessor(config=config)

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        audio, _ = self.processor.load_audio(audio_path)
        
        # Extract features
        features = self.processor.extract_all_features(audio)
        
        if self.text_files:
            text_path = self.text_files[idx]
            with open(text_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            return features, text
        
        return features