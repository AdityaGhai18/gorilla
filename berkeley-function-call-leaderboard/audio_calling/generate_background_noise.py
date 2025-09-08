import numpy as np
from scipy.io import wavfile
import os
from pathlib import Path

def generate_noise_samples():
    """Generate synthetic background noise samples."""
    noise_dir = Path(__file__).parent / "background_noise"
    noise_dir.mkdir(exist_ok=True)
    
    # Parameters
    sample_rate = 44100  # Hz
    duration = 10  # seconds
    
    # Generate white noise
    white_noise = np.random.normal(0, 0.1, size=sample_rate * duration)
    wavfile.write(
        noise_dir / "white_noise.wav",
        sample_rate,
        (white_noise * 32767).astype(np.int16)
    )
    
    # Generate brown noise (deeper than white noise)
    brown_noise = np.cumsum(np.random.normal(0, 0.1, size=sample_rate * duration))
    brown_noise = brown_noise / np.max(np.abs(brown_noise)) * 0.9
    wavfile.write(
        noise_dir / "brown_noise.wav",
        sample_rate,
        (brown_noise * 32767).astype(np.int16)
    )
    
    # Generate pink noise
    f = np.fft.fftfreq(sample_rate * duration)
    f = np.abs(f)
    f[0] = 1e-6  # Prevent divide by 0
    pink = np.random.normal(0, 1, size=sample_rate * duration)
    pink_f = np.fft.fft(pink)
    pink_f = pink_f / np.sqrt(f)
    pink = np.real(np.fft.ifft(pink_f))
    pink = pink / np.max(np.abs(pink)) * 0.9
    wavfile.write(
        noise_dir / "pink_noise.wav",
        sample_rate,
        (pink * 32767).astype(np.int16)
    )
    
    print("Generated noise samples in background_noise directory:")
    for file in noise_dir.glob("*.wav"):
        print(f"- {file.name}")

if __name__ == "__main__":
    generate_noise_samples()
