from background import (
    BackgroundNoiseProcessor,
    fluctuate_audio_volume,
    apply_gradual_audio_fade,
    apply_network_cut_effect,
    apply_network_beep_effect
)
import os

def process_speechified_audio(
    input_path,
    output_dir,
    noise_dir=None,
    apply_noise=True,
    noise_level=-20,
    apply_fluctuate=True,
    apply_fade=True,
    apply_network_cut=True,
    apply_network_beep=False,
    custom_suffix="processed"
):
    """
    Apply a sequence of audio effects to a speechified audio file.
    """
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(input_path))[0]
    current_path = input_path

    # 1. Add background noise using BackgroundNoiseProcessor
    if apply_noise and noise_dir:
        noise_output = os.path.join(output_dir, f"{base}_noise.wav")
        processor = BackgroundNoiseProcessor(noise_dir=noise_dir)
        processor.add_background_noise(current_path, output_path=noise_output, noise_level=noise_level)
        current_path = noise_output

    # 2. Fluctuate volume
    if apply_fluctuate:
        fluct_output = os.path.join(output_dir, f"{base}_fluct.wav")
        fluctuate_audio_volume(current_path, output_path=fluct_output)
        current_path = fluct_output

    # 3. Gradual fade (walk away and return)
    if apply_fade:
        fade_output = os.path.join(output_dir, f"{base}_fade.wav")
        apply_gradual_audio_fade(current_path, output_path=fade_output)
        current_path = fade_output

    # 4. Network cut (mute)
    if apply_network_cut:
        cut_output = os.path.join(output_dir, f"{base}_networkcut.wav")
        apply_network_cut_effect(current_path, output_path=cut_output)
        current_path = cut_output

    # 5. Network beep (optional, can be exclusive or in addition)
    if apply_network_beep:
        beep_output = os.path.join(output_dir, f"{base}_networkbeep.wav")
        apply_network_beep_effect(current_path, output_path=beep_output)
        current_path = beep_output

    # Final output
    final_output = os.path.join(output_dir, f"{base}_{custom_suffix}.wav")
    os.rename(current_path, final_output)
    print(f"Final processed file: {final_output}")
    return final_output


process_speechified_audio(
    input_path="path/to/speechified.wav",
    output_dir="path/to/output_dir",
    noise_dir="path/to/background_noise/noise",
    apply_noise=True,
    noise_level=-20,
    apply_fluctuate=True,
    apply_fade=True,
    apply_network_cut=True,
    apply_network_beep=False,
    custom_suffix="humanized"
)