from background import BackgroundNoiseProcessor
import os

def test_background_noise():
    # Initialize the processor
    processor = BackgroundNoiseProcessor()
    
    # Use one of the existing speech files
    test_file = "/Users/imradawoodani/gorilla/berkeley-function-call-leaderboard/audio_calling/TTS/single_test_output/openai_prompted_text_function_context/simple/live_simple_106-63-0.wav"
    
    # Test different noise levels
    noise_levels = [-30, -20, -10]  # From quieter to louder
    
    for noise_level in noise_levels:
        output_path = f"/Users/imradawoodani/gorilla/berkeley-function-call-leaderboard/audio_calling/background_noise/test_output_noise_{abs(noise_level)}db.wav"
        
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

if __name__ == "__main__":
    test_background_noise()
