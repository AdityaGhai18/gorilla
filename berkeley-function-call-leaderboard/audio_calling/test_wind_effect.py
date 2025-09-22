from pathlib import Path
from background import apply_heavy_wind_effect

input_file = Path('audio/BFCL_v3_irrelevance/irrelevance_0_openai.mp3')
wind_file = Path('background_noise/noise/wind_in_mic.mp3')
output_file = Path('test_windy_output.mp3')

if __name__ == "__main__":
    print(f" Applying heavy wind effect to '{input_file.name}'...")
    
    apply_heavy_wind_effect(
        audio_path=str(input_file),
        wind_noise_path=str(wind_file),
        output_path=str(output_file)
    )
    
    print(f"Done! Output saved as '{output_file.name}' in your main directory.")