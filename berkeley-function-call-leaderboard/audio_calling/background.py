import numpy as np
from pydub import AudioSegment
import os
from pathlib import Path
import random

class BackgroundNoiseProcessor:
    def __init__(self, noise_dir="background_noise"):
        """
        Initialize the background noise processor.
        
        Args:
            noise_dir (str): Base directory containing the 'noise' subdirectory with noise files
        """
        self.noise_dir = noise_dir
        self._ensure_noise_directory()
        self.noise_files = self._load_noise_files()

    def _ensure_noise_directory(self):
        """Create the necessary directories if they don't exist."""
        # Ensure main directory exists
        Path(self.noise_dir).mkdir(parents=True, exist_ok=True)
        # Ensure noise directory exists
        Path(os.path.join(self.noise_dir, "noise")).mkdir(parents=True, exist_ok=True)
        # Ensure noisy_audio directory exists
        Path(os.path.join(self.noise_dir, "noisy_audio")).mkdir(parents=True, exist_ok=True)

    def _load_noise_files(self):
        """Load noise files from the noise directory."""
        noise_dir = os.path.join(self.noise_dir, "noise")
        noise_files = []
        
        # Only look for .webm files in the noise directory
        webm_files = list(Path(noise_dir).glob("sample-*.webm"))
        if webm_files:
            noise_files.extend([str(f) for f in webm_files])
            print(f"Found {len(webm_files)} .webm noise files")
        
        if not noise_files:
            print(f"Warning: No noise files found in {noise_dir}")
            
        return noise_files
                
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
        random.seed(20)
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
            # Create noisy_audio directory inside background_noise
            noisy_dir = os.path.join(self.noise_dir, "noisy_audio")
            Path(noisy_dir).mkdir(parents=True, exist_ok=True)
            
            # Use just the filename without full path
            audio_filename = os.path.basename(audio_path)
            base_filename = os.path.splitext(audio_filename)[0]
            output_path = os.path.join(noisy_dir, f"{base_filename}_with_noise.wav")
        
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
        # If no output directory specified, use noisy_audio in background_noise directory
        if output_dir is None:
            output_dir = os.path.join(self.noise_dir, "noisy_audio")
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        processed_files = []
        
        # Get all wav files from input directory and its subdirectories
        audio_files = list(Path(audio_dir).rglob("*.wav"))
        total_files = len(audio_files)
        
        if total_files == 0:
            print(f"No .wav files found in {audio_dir}")
            return processed_files
        
        print(f"Processing {total_files} audio files...")
        
        # Process each audio file
        for i, audio_file in enumerate(audio_files, 1):
            # Preserve directory structure
            rel_path = audio_file.relative_to(Path(audio_dir))
            output_subdir = os.path.join(output_dir, os.path.dirname(str(rel_path)))
            Path(output_subdir).mkdir(parents=True, exist_ok=True)
            
            # Create output path
            output_path = os.path.join(output_subdir, f"{audio_file.stem}_with_noise.wav")
            
            try:
                processed_file = self.add_background_noise(
                    str(audio_file), output_path, noise_level
                )
                processed_files.append(processed_file)
                print(f"Processed {i}/{total_files}: {rel_path}")
            except Exception as e:
                print(f"Error processing {rel_path}: {str(e)}")
                continue
            
        print(f"\nCompleted processing {len(processed_files)} files")
        return processed_files
