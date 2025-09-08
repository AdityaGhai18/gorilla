import numpy as np
from pydub import AudioSegment
import os
from pathlib import Path
import random

class BackgroundNoiseProcessor:
    def __init__(self, noise_dir=None):
        """
        Initialize the background noise processor.
        
        Args:
            noise_dir (str): Directory containing background noise audio files (optional)
        """
        self.noise_dir = noise_dir if noise_dir else os.path.join(
            os.path.dirname(__file__), "background_noise"
        )
        self._ensure_noise_directory()
        self.noise_files = self._load_noise_files()

    def _ensure_noise_directory(self):
        """Create the background noise directory if it doesn't exist."""
        Path(self.noise_dir).mkdir(parents=True, exist_ok=True)

    def _load_noise_files(self):
        """Load all noise files from the noise directory and its subdirectories."""
        supported_formats = {".wav", ".mp3", ".ogg", ".webm"}
        noise_files = []
        
        # Search in main directory and all subdirectories
        for file in Path(self.noise_dir).rglob("*"):
            if file.suffix.lower() in supported_formats:
                noise_files.append(str(file))
        
        if not noise_files:
            print(f"Warning: No supported noise files found in {self.noise_dir} or its subdirectories")
        else:
            print(f"Found {len(noise_files)} noise files")
                
        return noise_files

    def add_background_noise(self, audio_path, output_path=None, noise_level=-20):
        """
        Add background noise to an audio file.
        
        Args:
            audio_path (str): Path to the input audio file
            output_path (str): Path to save the output audio file (optional)
            noise_level (int): Volume of background noise in dB (default: -20)
            
        Returns:
            str: Path to the output audio file
        """
        if not self.noise_files:
            raise ValueError("No background noise files found in the noise directory")

        # Load the main audio
        audio = AudioSegment.from_file(audio_path)
        
        # Randomly select a noise file
        noise_file = random.choice(self.noise_files)
        noise = AudioSegment.from_file(noise_file)
        
        # Loop or trim noise to match speech duration
        target_duration = len(audio)
        if len(noise) < target_duration:
            # Loop the noise
            loops_needed = int(np.ceil(target_duration / len(noise)))
            noise = noise * loops_needed
        
        # Trim to exact length
        noise = noise[:target_duration]
        
        # Adjust noise volume
        noise = noise + noise_level
        
        # Overlay noise with speech
        combined = audio.overlay(noise)
        
        # Generate output path if not provided
        if output_path is None:
            base_path = os.path.splitext(audio_path)[0]
            output_path = f"{base_path}_with_noise.wav"
        
        # Export the result
        combined.export(output_path, format="wav")
        return output_path

    def add_background_noise_batch(self, audio_dir, output_dir=None, noise_level=-20):
        """
        Process multiple audio files in a directory.
        
        Args:
            audio_dir (str): Directory containing input audio files
            output_dir (str): Directory to save processed files (optional)
            noise_level (int): Volume of background noise in dB (default: -20)
            
        Returns:
            list: Paths to all processed audio files
        """
        if output_dir is None:
            output_dir = os.path.join(audio_dir, "with_noise")
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        processed_files = []
        
        for audio_file in Path(audio_dir).glob("*.wav"):
            output_path = os.path.join(output_dir, f"{audio_file.stem}_with_noise.wav")
            processed_file = self.add_background_noise(
                str(audio_file), output_path, noise_level
            )
            processed_files.append(processed_file)
            
        return processed_files
