import os
import itertools
from background import (
    BackgroundNoiseProcessor,
    fluctuate_audio_volume,
    apply_gradual_audio_fade,
    apply_network_cut_effect,
    apply_audio_mumbling_effect,
    apply_mic_rubbing_effect
)
from pathlib import Path
import random
random.seed(16)

def get_all_noise_files(noise_dir):
    noise_files = []
    for ext in ("*.wav", "*.webm", "*.mp3"):
        noise_files.extend(list(Path(noise_dir).glob(ext)))
    return noise_files

def apply_effects_chain(audio_path, output_path, effects_chain):
    current_path = audio_path
    temp_files = []
    noise_processor = None
    for effect in effects_chain:
        if effect['type'] == 'background_noise':
            noise_file = effect['noise_file']
            temp_out = output_path.replace('.wav', f'_noise_{Path(noise_file).stem}.wav')
            if noise_processor is None:
                noise_processor = BackgroundNoiseProcessor(noise_dir=str(Path(noise_file).parent.parent))
            noise_processor.add_background_noise(
                current_path,
                noise_file=noise_file, # Pass the specific noise file
                output_path=temp_out,
                noise_level=effect.get('noise_level', -20)
            )

            # # Temporarily override noise_files to use only the selected noise_file
            # original_noise_files = noise_processor.noise_files
            # noise_processor.noise_files = [noise_file]
            # noise_processor.add_background_noise(current_path, output_path=temp_out, noise_level=effect.get('noise_level', -20))
            # noise_processor.noise_files = original_noise_files
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
        elif effect['type'] == 'fluctuate':
            temp_out = output_path.replace('.wav', '_fluctuate.wav')
            fluctuate_audio_volume(current_path, output_path=temp_out)
            current_path = temp_out
            temp_files.append(temp_out)
        # Add more effects here as needed
        elif effect['type'] == 'mic_rubbing':
            prob = effect.get('probability', 1.0)
            if random.random() < prob:
                temp_out = output_path.replace('.wav', '_micrubbing.wav')
                apply_mic_rubbing_effect(current_path, output_path=temp_out)
                current_path = temp_out
                temp_files.append(temp_out)
        
        elif effect['type'] == 'audio_mumbling':
            prob = effect.get('probability', 1.0)
            if random.random() < prob:
                temp_out = output_path.replace('.wav', '_mumbling.wav')
                apply_audio_mumbling_effect(current_path, output_path=temp_out)
                current_path = temp_out
                temp_files.append(temp_out)
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
    # For background noise, create sub-features for each noise file and dB level combination
    feature_variants = []
    for f in features:
        if f['type'] == 'background_noise':
            db_levels = [-30, -25, -20, -15]  # Different decibel levels
            for noise_file in noise_files:
                for db_level in db_levels:
                    variant = f.copy()
                    variant['noise_file'] = str(noise_file)
                    variant['noise_level'] = db_level
                    feature_variants.append(variant)
        elif f['type'] == 'network_cut':
            cut_counts = [2, 4, 5, 7, 10]  # Your desired sub-features
            for count in cut_counts:
                variant = f.copy()
                variant['n_cuts'] = count
                feature_variants.append(variant)
        else:
            feature_variants.append(f)
    # Generate all non-empty combinations
    all_combos = []
    for r in range(1, len(feature_variants)+1):
        for combo in itertools.combinations(feature_variants, r):
            # Only one background noise per combo
            noise_count = sum(1 for e in combo if e['type'] == 'background_noise')
            # Prevent multiple network_cut variants in the same combo
            cut_count = sum(1 for e in combo if e['type'] == 'network_cut')
            if noise_count <= 1 and cut_count <= 1:
                all_combos.append(combo)
    return all_combos

def get_all_audio_files(input_dir):
    audio_files = []
    for ext in ("*.wav", "*.mp3"):
        for file in Path(input_dir).rglob(ext):
            audio_files.append(file)
    return audio_files

def process_all_speechified(input_dir, output_dir, noise_dir):
    os.makedirs(output_dir, exist_ok=True)
    noise_files = get_all_noise_files(noise_dir)
    features = [
        #removed ntwork beep
        {'type': 'background_noise'},  # noise_level will be added in combinations
        {'type': 'audio_fade'},
        {'type': 'network_cut', 'probability': 0.3},
        {'type': 'fluctuate'},
        # Add the mic rumbling and audio mumbling effects here too
        {'type': 'mic_rubbing', 'probability': 0.1},
        {'type': 'audio_mumbling', 'probability': 0.1}
    ]
    combos = generate_feature_combinations(noise_files, features)
    print(f"Total combinations: {len(combos)}")
    
    # Get all audio files recursively
    speech_files = get_all_audio_files(input_dir)
    print(f"Total audio files found: {len(speech_files)}")
    
    # Process each audio file with a combination
    for i, speech_file in enumerate(speech_files):
        # Use modulo to loop over combinations
        combo = combos[i % len(combos)]
        
        # Preserve directory structure in output
        rel_path = speech_file.relative_to(Path(input_dir))
        out_dir = Path(output_dir) / rel_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        
        base = speech_file.stem
        combo_names = '_'.join([
            e['type'] + 
            (('_' + Path(e['noise_file']).stem + f'_db{e["noise_level"]}') if e['type']=='background_noise' else '') 
            for e in combo
        ])
        out_name = f"{base}_{combo_names}.wav"
        out_path = str(out_dir / out_name)
        
        print(f"Processing {speech_file} with {combo_names}")
        apply_effects_chain(str(speech_file), out_path, combo)


# Update the input directory to process all audio files
process_all_speechified(
    input_dir="berkeley-function-call-leaderboard/audio_calling/audio",
    output_dir="berkeley-function-call-leaderboard/audio_calling/background_noise/noisy_FINAL",
    noise_dir="berkeley-function-call-leaderboard/audio_calling/background_noise/noise"
)