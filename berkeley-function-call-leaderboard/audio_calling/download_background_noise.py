import os
import urllib.request
from pathlib import Path

def download_noise_samples():
    """
    Download sample background noise files from FreeSound or similar sources.
    Using public domain or Creative Commons licensed sounds.
    """
    # Create background_noise directory if it doesn't exist
    noise_dir = Path(__file__).parent / "background_noise"
    noise_dir.mkdir(exist_ok=True)

    # List of public domain background noise samples
    noise_samples = {
        "cafe_background.wav": "https://www.pacdv.com/sounds/ambience_sounds/cafe-1.wav",
        "office_background.wav": "https://www.pacdv.com/sounds/ambience_sounds/office-1.wav",
        "street_background.wav": "https://www.pacdv.com/sounds/ambience_sounds/city_traffic_1.wav",
        "white_noise.wav": "https://www.pacdv.com/sounds/ambience_sounds/white-noise-1.wav"
    }

    for filename, url in noise_samples.items():
        output_path = noise_dir / filename
        if not output_path.exists():
            print(f"Downloading {filename}...")
            try:
                urllib.request.urlretrieve(url, output_path)
                print(f"Successfully downloaded {filename}")
            except Exception as e:
                print(f"Error downloading {filename}: {e}")

if __name__ == "__main__":
    download_noise_samples()
