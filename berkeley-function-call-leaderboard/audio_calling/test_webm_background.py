from background import BackgroundNoiseProcessor
import os
import shutil
from pathlib import Path

def setup_noise_files(webm_source_dir):
    """
    Copy .webm noise files to the background_noise directory.
    
    Args:
        webm_source_dir (str): Directory containing the downloaded .webm noise files
    """
    # Create background_noise directory if it doesn't exist
    noise_dir = Path(__file__).parent / "background_noise"
    noise_dir.mkdir(exist_ok=True)
    
    # Copy all .webm files to background_noise directory
    source_dir = Path(webm_source_dir)
    copied_files = []
    
    for file in source_dir.glob("*.webm"):
        dest_path = noise_dir / file.name
        shutil.copy2(file, dest_path)
        copied_files.append(dest_path)
    
    print(f"Copied {len(copied_files)} noise files to {noise_dir}")
    return copied_files

def test_webm_background_noise(webm_source_dir, test_file=None, noise_level=-20):
    """
    Test adding background noise using .webm files.
    
    Args:
        webm_source_dir (str): Directory containing the downloaded .webm noise files
        test_file (str, optional): Path to the speech file to process. If None, uses a default file
        noise_level (int): Volume of background noise in dB (default: -20)
    """
    # Setup noise files
    setup_noise_files(webm_source_dir)
    
    # Initialize the processor
    processor = BackgroundNoiseProcessor()
    
    # Use default test file if none provided
    if test_file is None:
        test_file = "/Users/imradawoodani/gorilla/berkeley-function-call-leaderboard/audio_calling/TTS/single_test_output/openai_prompted_text_function_context/simple/live_simple_106-63-0.wav"
    
    # Process the file with each noise file to test
    try:
        output_path = processor.add_background_noise(
            test_file,
            noise_level=noise_level
        )
        print(f"\nProcessed file saved at: {output_path}")
    except Exception as e:
        print(f"Error processing file: {e}")

if __name__ == "__main__":
    # Replace this with the path to your downloaded .webm files
    webm_source_dir = input("Enter the path to your .webm noise files directory: ")
    
    # You can optionally specify a different speech file to test with
    test_file = input("Enter the path to a speech file to test (or press Enter to use default): ").strip()
    if not test_file:
        test_file = None
    
    # Test with different noise levels
    noise_levels = [-30, -20, -10]
    for level in noise_levels:
        print(f"\nTesting with noise level: {level}dB")
        test_webm_background_noise(webm_source_dir, test_file, level)
