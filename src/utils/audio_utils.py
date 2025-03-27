import torch
import librosa
import numpy as np
import soundfile as sf
from typing import Tuple, Optional, Dict

class AudioProcessor:
    def __init__(self, config: Dict = None):
        """
        Initialize audio processor
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.sample_rate = int(self.config.get('sample_rate', 22050))
        if isinstance(self.config, dict) and 'audio' in self.config:
            self.audio_config = self.config['audio']
        else:
            self.audio_config = self.config

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
        n_fft = self.audio_config.get('n_fft', 1024)
        hop_length = self.audio_config.get('hop_length', 256)
        win_length = self.audio_config.get('win_length', 1024)
        n_mels = self.audio_config.get('mel_channels', 80)
        
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
            n_fft=self.audio_config.get('n_fft', 1024),
            hop_length=self.audio_config.get('hop_length', 256),
            n_mels=self.audio_config.get('mel_channels', 80)
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

    def mel_to_audio(self, mel_spectrogram: np.ndarray) -> np.ndarray:
        """
        Convert mel spectrogram back to audio using Griffin-Lim algorithm
        Args:
            mel_spectrogram: Mel spectrogram (n_mels, time)
        Returns:
            audio: Reconstructed audio waveform
        """
        # Reverse the normalization applied in the model
        mel_spectrogram = mel_spectrogram * 3 - 12
        
        # Convert from log scale back to linear
        mel_spectrogram = np.exp(mel_spectrogram)
        
        # Get config parameters
        n_fft = int(self.audio_config.get('n_fft', 1024))
        hop_length = int(self.audio_config.get('hop_length', 256))
        win_length = int(self.audio_config.get('win_length', 1024))
        n_mels = mel_spectrogram.shape[0]

        print(f"Audio parameters: sr={self.sample_rate}, n_fft={n_fft}, hop_length={hop_length}, n_mels={n_mels}")
        print(f"Mel spectrogram shape: {mel_spectrogram.shape}")
        
        # Create mel filterbank
        mel_basis = librosa.filters.mel(
            sr=self.sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=0.0,
            fmax=self.sample_rate/2.0
        )
        
        # Convert mel spectrogram to linear frequency domain
        mel_to_linear = np.linalg.pinv(mel_basis)
        linear_spectrogram = np.maximum(1e-10, np.dot(mel_to_linear, mel_spectrogram))
        
        print(f"Linear spectrogram shape: {linear_spectrogram.shape}")
        
        # Griffin-Lim algorithm to reconstruct phase
        audio = librosa.griffinlim(
            linear_spectrogram,
            n_iter=64,  # Increased iterations for better quality
            hop_length=hop_length,
            win_length=win_length,
            window='hann',
            center=True,
            dtype=np.float32,
            momentum=0.99
        )
        
        print(f"Generated audio shape: {audio.shape}, duration: {len(audio)/self.sample_rate:.2f}s")
        return audio

    def save_wav(self, audio: np.ndarray, file_path: str) -> None:
        """
        Save audio as WAV file
        Args:
            audio: Audio data
            file_path: Output file path
        """
        if len(audio) == 0:
            raise ValueError("Audio data is empty!")
            
        # Remove DC offset
        audio = audio - np.mean(audio)
        
        # Apply fade in/out to reduce clicking
        fade_length = min(1000, len(audio) // 4)
        fade_in = np.linspace(0., 1., fade_length)
        fade_out = np.linspace(1., 0., fade_length)
        audio[:fade_length] *= fade_in
        audio[-fade_length:] *= fade_out
        
        # Normalize audio
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.9  # Leave some headroom
        else:
            raise ValueError("Audio data contains only zeros!")
        
        print(f"Saving audio: length={len(audio)}, min={np.min(audio):.2f}, max={np.max(audio):.2f}")
        
        # Save as WAV file
        import soundfile as sf
        sf.write(file_path, audio, self.sample_rate)
        print(f"Saved audio to: {file_path}")


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