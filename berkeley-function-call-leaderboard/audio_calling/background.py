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
        """Load .wav, .webm, and .mp3 noise files from the noise directory."""
        noise_dir = os.path.join(self.noise_dir, "noise")
        noise_files = []
        # Look for .wav, .webm, and .mp3 files in the noise directory
        for ext in ("*.wav", "*.webm", "*.mp3"):
            found = list(Path(noise_dir).glob(ext))
            if found:
                noise_files.extend([str(f) for f in found])
        if noise_files:
            print(f"Found {len(noise_files)} noise files (.wav, .webm, .mp3)")
        else:
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

def overlay_audio_with_noise(speech_path, noise_path, output_path=None, noise_level_db=-20):
    """
    Overlay a speech audio file with another audio file as noise.
    
    Args:
        speech_path (str): Path to the speech audio file
        noise_path (str): Path to the noise audio file
        output_path (str): Path to save the output file (optional)
        noise_level_db (int): dB to reduce noise audio (default: -20)
        
    Returns:
        str: Path to the output file
    """
    # Load audio files
    speech = AudioSegment.from_file(speech_path)
    noise = AudioSegment.from_file(noise_path)

    # Loop or trim noise to match speech duration
    target_duration = len(speech)
    if len(noise) < target_duration:
        loops_needed = int(target_duration / len(noise)) + 1
        noise = noise * loops_needed
    noise = noise[:target_duration]

    # Adjust noise volume
    noise = noise + noise_level_db

    # Overlay
    combined = speech.overlay(noise)

    # Output path
    if output_path is None:
        base = os.path.splitext(os.path.basename(speech_path))[0]
        noise_base = os.path.splitext(os.path.basename(noise_path))[0]
        output_path = f"{base}_with_{noise_base}_noise.wav"

    combined.export(output_path, format="wav")
    return output_path

def fluctuate_audio_volume(
    audio_path,
    output_path=None,
    min_db_change=-15,
    max_db_change=10,
    min_duration_ms=500,
    max_duration_ms=3000,
    n_fluctuations=5
):
    """
    Randomly fluctuate the volume of an audio clip, lowering and raising it at random intervals.
    
    Args:
        audio_path (str): Path to the input audio file
        output_path (str): Path to save the output audio file (optional)
        min_db_change (int): Minimum dB change (negative for lowering)
        max_db_change (int): Maximum dB change (positive for increasing)
        min_duration_ms (int): Minimum duration of a fluctuation in ms
        max_duration_ms (int): Maximum duration of a fluctuation in ms
        n_fluctuations (int): Number of fluctuations to apply
            
    Returns:
        str: Path to the output audio file
    """
    audio = AudioSegment.from_file(audio_path)
    length = len(audio)
    segments = []
    last_pos = 0
    flucts = []
    for _ in range(n_fluctuations):
        start = random.randint(0, length - min_duration_ms)
        dur = random.randint(min_duration_ms, min(max_duration_ms, length - start))
        db_change = random.uniform(min_db_change, max_db_change)
        flucts.append((start, start+dur, db_change))
    flucts.sort()  # sort by start time
    for start, end, db_change in flucts:
        if last_pos < start:
            segments.append(audio[last_pos:start])
        fluct_segment = audio[start:end] + db_change
        segments.append(fluct_segment)
        last_pos = end
    if last_pos < length:
        segments.append(audio[last_pos:])
    fluctuated = sum(segments)
    if output_path is None:
        base = os.path.splitext(os.path.basename(audio_path))[0]
        output_path = f"{base}_fluctuated.wav"
    fluctuated.export(output_path, format="wav")
    print(f"Created fluctuated audio: {output_path}")
    return output_path
