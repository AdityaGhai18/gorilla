import os
import itertools
from background import (
    BackgroundNoiseProcessor,
    fluctuate_audio_volume,
    apply_gradual_audio_fade,
    apply_network_cut_effect,
    apply_network_beep_effect
)
from pathlib import Path

def get_all_noise_files(noise_dir):
    noise_files = []
    for ext in ("*.wav", "*.webm", "*.mp3"):
        noise_files.extend(list(Path(noise_dir).glob(ext)))
    return noise_files

def apply_effects_chain(audio_path, output_path, effects_chain):
    current_path = audio_path
    temp_files = []
    noise_processor = None
    import random
    random.seed(16)
    for effect in effects_chain:
        if effect['type'] == 'background_noise':
            noise_file = effect['noise_file']
            temp_out = output_path.replace('.wav', f'_noise_{Path(noise_file).stem}.wav')
            if noise_processor is None:
                noise_processor = BackgroundNoiseProcessor(noise_dir=str(Path(noise_file).parent.parent))
            # Temporarily override noise_files to use only the selected noise_file
            original_noise_files = noise_processor.noise_files
            noise_processor.noise_files = [noise_file]
            noise_processor.add_background_noise(current_path, output_path=temp_out, noise_level=effect.get('noise_level', -20))
            noise_processor.noise_files = original_noise_files
            current_path = temp_out
            temp_files.append(temp_out)
        elif effect['type'] == 'audio_fade':
            temp_out = output_path.replace('.wav', '_fade.wav')
            apply_gradual_audio_fade(current_path, output_path=temp_out)
            current_path = temp_out
            temp_files.append(temp_out)
        elif effect['type'] == 'network_cut':
            prob = effect.get('probability', 1.0)
            n_cuts = effect.get('n_cuts', 8)
            if random.random() < prob:
                temp_out = output_path.replace('.wav', '_networkcut.wav')
                apply_network_cut_effect(current_path, output_path=temp_out, n_cuts=n_cuts)
                current_path = temp_out
                temp_files.append(temp_out)
        elif effect['type'] == 'network_beep':
            prob = effect.get('probability', 1.0)
            n_beeps = effect.get('n_beeps', 8)
            if random.random() < prob:
                temp_out = output_path.replace('.wav', '_networkbeep.wav')
                apply_network_beep_effect(current_path, output_path=temp_out, n_beeps=n_beeps)
                current_path = temp_out
                temp_files.append(temp_out)
        elif effect['type'] == 'fluctuate':
            temp_out = output_path.replace('.wav', '_fluctuate.wav')
            fluctuate_audio_volume(current_path, output_path=temp_out)
            current_path = temp_out
            temp_files.append(temp_out)
        # Add more effects here as needed
    # Final output
    os.rename(current_path, output_path)
    # Clean up temp files except the final output
    for f in temp_files:
        if f != output_path and os.path.exists(f):
            os.remove(f)
    print(f"Created: {output_path}")
    return output_path

def generate_feature_combinations(noise_files, features):
    # Each feature is a dict with 'type' and optional params
    # For background noise, create a sub-feature for each noise file
    feature_variants = []
    for f in features:
        if f['type'] == 'background_noise':
            for noise_file in noise_files:
                variant = f.copy()
                variant['noise_file'] = str(noise_file)
                feature_variants.append(variant)
        else:
            feature_variants.append(f)
    # Generate all non-empty combinations
    all_combos = []
    for r in range(1, len(feature_variants)+1):
        for combo in itertools.combinations(feature_variants, r):
            # Only one background noise per combo
            noise_count = sum(1 for e in combo if e['type'] == 'background_noise')
            if noise_count <= 1:
                all_combos.append(combo)
    return all_combos

def process_all_speechified(input_dir, output_dir, noise_dir):
    os.makedirs(output_dir, exist_ok=True)
    noise_files = get_all_noise_files(noise_dir)
    features = [
        {'type': 'background_noise', 'noise_level': -20},
        {'type': 'audio_fade'},
        {'type': 'network_cut', 'probability': 0.3, 'n_cuts': 4},
        {'type': 'network_beep', 'probability': 0.2, 'n_beeps': 2},
        {'type': 'fluctuate'}
        # Add more features here (mic rubbing, mumbles, volume level)
    ]
    combos = generate_feature_combinations(noise_files, features)
    print(f"Total combinations: {len(combos)}")
    speech_files = list(Path(input_dir).glob('*.mp3'))
    # Assign each permutation to a different audio file (one-to-one mapping)
    min_len = min(len(speech_files), len(combos))
    for i in range(min_len):
        speech_file = speech_files[i]
        combo = combos[i]
        base = Path(speech_file).stem
        combo_names = '_'.join([e['type'] + (('_' + Path(e['noise_file']).stem) if e['type']=='background_noise' else '') for e in combo])
        out_name = f"{base}_{combo_names}.wav"
        out_path = os.path.join(output_dir, out_name)
        print(f"Processing {speech_file} with {combo_names}")
        apply_effects_chain(str(speech_file), out_path, combo)
    if len(combos) > len(speech_files):
        print(f"Warning: {len(combos) - len(speech_files)} permutations were not used due to insufficient audio files.")
    elif len(speech_files) > len(combos):
        print(f"Warning: {len(speech_files) - len(combos)} audio files were not used due to insufficient permutations.")


# when all permutations of features are done, loppover them for the rest of the audio files in the directory
# e.g. if there are 10 permutations and 25 audio files, the first 10 audio files get unique permutations, the next 10 get the same as the first 10
# and the last 5 get the first 5 permutations again

process_all_speechified(
    input_dir="berkeley-function-call-leaderboard/audio_calling/clean_to_speech_text/final_results/audio/BFCL_v3_multi_turn_base",           # directory with your speechified .wav files
    output_dir="berkeley-function-call-leaderboard/audio_calling/background_noise/final",       # where to save all processed files
    noise_dir="berkeley-function-call-leaderboard/audio_calling/background_noise/noise" # directory with your noise files
)