from background import BackgroundNoiseProcessor
import os
from pathlib import Path

def test_with_kaggle_noise():
    """Test adding background noise using Kaggle noise files."""
    # Initialize the processor with the full path to background_noise directory
    noise_dir = os.path.join(os.path.dirname(__file__), "background_noise")
    processor = BackgroundNoiseProcessor(noise_dir=noise_dir)
    
    # Use a test speech file
    test_file = "/Users/imradawoodani/gorilla/berkeley-function-call-leaderboard/audio_calling/TTS/single_test_output/openai_prompted_text_function_context/simple/live_simple_106-63-0.wav"
    
    # Process with different noise levels
    noise_levels = [-30, -20, -10]  # From quieter to louder
    
    for noise_level in noise_levels:
        print(f"\nProcessing with noise level: {noise_level}dB")
        output_path = f"/Users/imradawoodani/gorilla/berkeley-function-call-leaderboard/audio_calling/background_noise/noisy_audio/test_output_kaggle_noise_{abs(noise_level)}db.wav"
        
        try:
            processed_file = processor.add_background_noise(
                test_file,
                output_path=output_path,
                noise_level=noise_level
            )
            print(f"Created file with background noise: {processed_file}")
        except Exception as e:
            print(f"Error processing file: {e}")

def test_background_noise():
    # Initialize the processor
    processor = BackgroundNoiseProcessor()
    
    # Use one of the existing speech files
    test_file = "/Users/imradawoodani/gorilla/berkeley-function-call-leaderboard/audio_calling/TTS/single_test_output/openai_prompted_text_function_context/simple/live_simple_106-63-0.wav"
    
    # Test different noise levels
    noise_levels = [-30, -20, -10]  # From quieter to louder
    
    for noise_level in noise_levels:
        output_path = f"/Users/imradawoodani/gorilla/berkeley-function-call-leaderboard/audio_calling/background_noise/noisy_audio/test_output_noise_{abs(noise_level)}db.wav"
        
        print(f"\nProcessing with noise level: {noise_level}dB")
        try:
            processed_file = processor.add_background_noise(
                test_file,
                output_path=output_path,
                noise_level=noise_level
            )
            print(f"Created file with background noise: {processed_file}")
        except Exception as e:
            print(f"Error processing file: {e}")

def test_batch_processing():
    # Initialize the processor with the full path to background_noise directory
    script_dir = os.path.dirname(__file__)
    noise_dir = os.path.join(script_dir, "background_noise")
    processor = BackgroundNoiseProcessor(noise_dir=noise_dir)
    
    # Path to directory containing speech files
    audio_dir = os.path.join(script_dir, "TTS/single_test_output/openai_prompted_text_function_context/simple")
    
    # Process files with different noise levels
    noise_levels = [-30, -20, -10]  # from quieter to louder
    
    for noise_level in noise_levels:
        print(f"\nProcessing batch with noise level: {noise_level}dB")
        output_dir = os.path.join(noise_dir, "noisy_audio", f"level_{abs(noise_level)}db")
        
        processed_files = processor.add_background_noise_batch(
            audio_dir=audio_dir,
            output_dir=output_dir,
            noise_level=noise_level
        )
        
        if processed_files:
            print(f"Successfully processed {len(processed_files)} files")
            print(f"Output directory: {output_dir}")
        else:
            print("No files were processed")

if __name__ == "__main__":
    test_background_noise()
    test_with_kaggle_noise()
    test_batch_processing()