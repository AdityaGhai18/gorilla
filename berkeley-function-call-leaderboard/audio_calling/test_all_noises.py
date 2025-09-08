from background import BackgroundNoiseProcessor
import os
from pathlib import Path
import random

def test_all_noise_files():
    """Test a single audio file with all available noise files."""
    # Initialize the processor
    script_dir = os.path.dirname(__file__)
    noise_dir = os.path.join(script_dir, "background_noise")
    processor = BackgroundNoiseProcessor(noise_dir=noise_dir)
    
    # Get all noise files manually to ensure we test each one
    noise_path = os.path.join(noise_dir, "noise")
    noise_files = []
    for ext in [".wav", ".webm"]:
        noise_files.extend(list(Path(noise_path).glob(f"*{ext}")))
    
    if not noise_files:
        print("No noise files found!")
        return
    
    print(f"Found {len(noise_files)} noise files:")
    for file in noise_files:
        print(f"- {file.name}")
    
    # Use a test audio file
    test_file = os.path.join(
        script_dir,
        "TTS/single_test_output/openai_prompted_text_function_context/simple/live_simple_106-63-0.wav"
    )
    
    # Process with different noise levels
    noise_levels = [-30, -20, -10]  # from quieter to louder
    
    # Create output directory structure
    base_output_dir = os.path.join(noise_dir, "noisy_audio", "noise_comparison")
    Path(base_output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save original noise_files from processor
    original_noise_files = processor.noise_files
    
    # Test each noise file
    for noise_file in noise_files:
        print(f"\nTesting with noise file: {noise_file.name}")
        
        # Temporarily set this as the only noise file
        processor.noise_files = [str(noise_file)]
        
        # Process with different noise levels
        for noise_level in noise_levels:
            # Create descriptive output filename
            noise_name = noise_file.stem
            output_name = f"speech_with_{noise_name}_at_{abs(noise_level)}db.wav"
            output_path = os.path.join(base_output_dir, output_name)
            
            try:
                processed_file = processor.add_background_noise(
                    test_file,
                    output_path=output_path,
                    noise_level=noise_level
                )
                print(f"Created: {os.path.basename(processed_file)} (noise level: {noise_level}dB)")
            except Exception as e:
                print(f"Error processing with {noise_file.name} at {noise_level}dB: {str(e)}")
    
    # Restore original noise files
    processor.noise_files = original_noise_files
    
    print(f"\nAll processed files are in: {base_output_dir}")

if __name__ == "__main__":
    test_all_noise_files()
