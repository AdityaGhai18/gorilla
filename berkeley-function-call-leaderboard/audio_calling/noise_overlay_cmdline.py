from pydub import AudioSegment
import sys
import os

def overlay_audio_with_noise(speech_path, noise_path, output_path="background_noise/noisy_audio", noise_level_db=-20):
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
    print(f"Created: {output_path}")
    return output_path

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python overlay_audio_with_noise.py <speech.wav> <noise.wav|webm|mp3> [output.wav] [noise_level_db]")
        sys.exit(1)
    speech_path = sys.argv[1]
    noise_path = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else None
    noise_level_db = int(sys.argv[4]) if len(sys.argv) > 4 else -20
    overlay_audio_with_noise(speech_path, noise_path, output_path, noise_level_db)
