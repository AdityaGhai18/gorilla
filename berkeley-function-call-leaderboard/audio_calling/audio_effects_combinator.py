import os
from background import (
    BackgroundNoiseProcessor,
    fluctuate_audio_volume,
    apply_gradual_audio_fade,
    apply_network_cut_effect,
    apply_audio_mumbling_effect,
    apply_mic_rubbing_effect,
    apply_heavy_wind_effect
)
from pathlib import Path
import random
random.seed(16)

def get_all_noise_files(noise_dir):
    noise_files = []
    for ext in ("*.wav", "*.webm", "*.mp3"):
        noise_files.extend(list(Path(noise_dir).glob(ext)))
    return noise_files

# def apply_effects_chain(audio_path, output_path, effects_chain):
#     current_path = audio_path
#     temp_files = []
#     noise_processor = None
#     for effect in effects_chain:
#         if effect['type'] == 'background_noise':
#             noise_file = effect['noise_file']
#             temp_out = output_path.replace('.wav', f'_noise_{Path(noise_file).stem}.wav')
#             if noise_processor is None:
#                 noise_processor = BackgroundNoiseProcessor(noise_dir=str(Path(noise_file).parent.parent))
#             noise_processor.add_background_noise(
#                 current_path,
#                 noise_file=noise_file, # Pass the specific noise file
#                 output_path=temp_out,
#                 noise_level=effect.get('noise_level', -20)
#             )

#             # # Temporarily override noise_files to use only the selected noise_file
#             # original_noise_files = noise_processor.noise_files
#             # noise_processor.noise_files = [noise_file]
#             # noise_processor.add_background_noise(current_path, output_path=temp_out, noise_level=effect.get('noise_level', -20))
#             # noise_processor.noise_files = original_noise_files
#             current_path = temp_out
#             temp_files.append(temp_out)
#         elif effect['type'] == 'audio_fade':
#             temp_out = output_path.replace('.wav', '_fade.wav')
#             apply_gradual_audio_fade(current_path, output_path=temp_out)
#             current_path = temp_out
#             temp_files.append(temp_out)
#         elif effect['type'] == 'network_cut':
#             prob = effect.get('probability', 1.0)
#             n_cuts = effect.get('n_cuts', 8)
#             if random.random() < prob:
#                 temp_out = output_path.replace('.wav', '_networkcut.wav')
#                 apply_network_cut_effect(current_path, output_path=temp_out, n_cuts=n_cuts)
#                 current_path = temp_out
#                 temp_files.append(temp_out)
#         elif effect['type'] == 'fluctuate':
#             temp_out = output_path.replace('.wav', '_fluctuate.wav')
#             fluctuate_audio_volume(current_path, output_path=temp_out)
#             current_path = temp_out
#             temp_files.append(temp_out)
#         # Add more effects here as needed
#         elif effect['type'] == 'mic_rubbing':
#             prob = effect.get('probability', 1.0)
#             if random.random() < prob:
#                 temp_out = output_path.replace('.wav', '_micrubbing.wav')
#                 apply_mic_rubbing_effect(current_path, output_path=temp_out)
#                 current_path = temp_out
#                 temp_files.append(temp_out)
        
#         elif effect['type'] == 'audio_mumbling':
#             prob = effect.get('probability', 1.0)
#             if random.random() < prob:
#                 temp_out = output_path.replace('.wav', '_mumbling.wav')
#                 apply_audio_mumbling_effect(current_path, output_path=temp_out)
#                 current_path = temp_out
#                 temp_files.append(temp_out)
#         elif effect['type'] == 'heavy_wind':
#             wind_file = effect['wind_file']
#             temp_out = output_path.replace('.wav', '_heavywind.wav')
#             apply_heavy_wind_effect(
#                 current_path,
#                 wind_noise_path=wind_file,
#                 output_path=temp_out
#             )
#             current_path = temp_out
#             temp_files.append(temp_out)
#     # Final output
#     os.rename(current_path, output_path)
#     # Clean up temp files except the final output
#     for f in temp_files:
#         if f != output_path and os.path.exists(f):
#             os.remove(f)
#     print(f"Created: {output_path}")
#     return output_path

def apply_effects_chain(audio_path, output_path, effects_chain):
    current_path = audio_path
    temp_files = []
    noise_processor = None
    
    # Use pathlib for robust path manipulation
    p = Path(output_path)
    stem = p.stem
    suffix = p.suffix  # This will now be '.mp3'

    for effect in effects_chain:
        effect_type = effect['type']
        
        # This new logic creates temp names like 'original_stem_effect.mp3'
        if effect_type == 'background_noise':
            noise_file = effect['noise_file']
            temp_name = f"{stem}_noise_{Path(noise_file).stem}{suffix}"
            temp_out = str(p.with_name(temp_name))
            if noise_processor is None:
                # Assuming noise_dir is the parent of the 'noise' folder
                noise_processor = BackgroundNoiseProcessor(noise_dir=str(Path(noise_file).parent.parent))
            noise_processor.add_background_noise(current_path, noise_file=noise_file, output_path=temp_out, noise_level=effect.get('noise_level', -20))
        else:
            temp_name = f"{stem}_{effect_type}{suffix}"
            temp_out = str(p.with_name(temp_name))

            if effect_type == 'audio_fade':
                apply_gradual_audio_fade(current_path, output_path=temp_out)
            elif effect_type == 'network_cut':
                if random.random() < effect.get('probability', 1.0):
                    apply_network_cut_effect(current_path, output_path=temp_out, n_cuts=effect.get('n_cuts'))
                else: continue # Skip to next effect if probability check fails
            elif effect_type == 'fluctuate':
                fluctuate_audio_volume(current_path, output_path=temp_out)
            elif effect_type == 'mic_rubbing':
                if random.random() < effect.get('probability', 1.0):
                    apply_mic_rubbing_effect(current_path, output_path=temp_out)
                else: continue
            elif effect_type == 'audio_mumbling':
                if random.random() < effect.get('probability', 1.0):
                    apply_audio_mumbling_effect(current_path, output_path=temp_out)
                else: continue
            elif effect_type == 'heavy_wind':
                 apply_heavy_wind_effect(current_path, wind_noise_path=effect['wind_file'], output_path=temp_out)

        current_path = temp_out
        temp_files.append(temp_out)

    import shutil
    # Final output and cleanup
    if not temp_files:
        # If no effects were applied, copy the original file instead of moving it
        shutil.copy(audio_path, output_path)
    else:
        # Otherwise, rename the final temporary file as planned
        os.rename(current_path, output_path)
        # And clean up the intermediate files
        for f in temp_files:
            if f != output_path and os.path.exists(f):
                os.remove(f)

    print(f"Created: {output_path}")
    return output_path
    # # Final output and cleanup
    # os.rename(current_path, output_path)
    # for f in temp_files:
    #     if f != output_path and os.path.exists(f):
    #         os.remove(f)
    # print(f"Created: {output_path}")
    # return output_path
    

# def generate_feature_combinations(noise_files, features):
#     # Each feature is a dict with 'type' and optional params
#     # For background noise, create sub-features for each noise file and dB level combination
#     feature_variants = []
#     for f in features:
#         if f['type'] == 'background_noise':
#             db_levels = [-30, -25, -20, -15]  # Different decibel levels
#             for noise_file in noise_files:
#                 for db_level in db_levels:
#                     variant = f.copy()
#                     variant['noise_file'] = str(noise_file)
#                     variant['noise_level'] = db_level
#                     feature_variants.append(variant)
#         elif f['type'] == 'network_cut':
#             cut_counts = [2, 4, 5, 7, 10]  # Your desired sub-features
#             for count in cut_counts:
#                 variant = f.copy()
#                 variant['n_cuts'] = count
#                 feature_variants.append(variant)
#         else:
#             feature_variants.append(f)
#     # Generate all non-empty combinations
#     all_combos = []
#     for r in range(1, len(feature_variants)+1):
#         for combo in itertools.combinations(feature_variants, r):
#             # Only one background noise per combo
#             noise_count = sum(1 for e in combo if e['type'] == 'background_noise')
#             # Prevent multiple network_cut variants in the same combo
#             cut_count = sum(1 for e in combo if e['type'] == 'network_cut')
#             if noise_count <= 1 and cut_count <= 1:
#                 all_combos.append(combo)
#     return all_combos

def get_all_audio_files(input_dir):
    audio_files = []
    for ext in ("*.wav", "*.mp3"):
        for file in Path(input_dir).rglob(ext):
            audio_files.append(file)
    return audio_files

# def process_all_speechified(input_dir, output_dir, noise_dir):
#     os.makedirs(output_dir, exist_ok=True)
#     noise_files = get_all_noise_files(noise_dir)
#     wind_file_path = str(Path(noise_dir) / 'wind_in_mic.mp3')
#     features = [
#         #removed ntwork beep
#         {'type': 'background_noise'},  # noise_level will be added in combinations
#         {'type': 'audio_fade'},
#         {'type': 'network_cut', 'probability': 0.3},
#         {'type': 'fluctuate'},
#         # Add the mic rumbling and audio mumbling effects here too
#         {'type': 'mic_rubbing', 'probability': 0.1},
#         {'type': 'audio_mumbling', 'probability': 0.1},
#         {'type': 'heavy_wind', 'wind_file': wind_file_path} # Example wind noise file
#     ]
#     combos = generate_feature_combinations(noise_files, features)
#     print(f"Total combinations: {len(combos)}")
    
#     # Get all audio files recursively
#     speech_files = get_all_audio_files(input_dir)
#     print(f"Total audio files found: {len(speech_files)}")
    
#     # Process each audio file with a combination
#     for i, speech_file in enumerate(speech_files):
#         # Use modulo to loop over combinations
#         combo = combos[i % len(combos)]
        
#         # Preserve directory structure in output
#         rel_path = speech_file.relative_to(Path(input_dir))
#         out_dir = Path(output_dir) / rel_path.parent
#         out_dir.mkdir(parents=True, exist_ok=True)
        
#         base = speech_file.stem
#         combo_names = '_'.join([
#             e['type'] + 
#             (('_' + Path(e['noise_file']).stem + f'_db{e["noise_level"]}') if e['type']=='background_noise' else '') 
#             for e in combo
#         ])
#         out_name = f"{base}_{combo_names}.mp3"
#         out_path = str(out_dir / out_name)
        
#         print(f"Processing {speech_file} with {combo_names}")
#         apply_effects_chain(str(speech_file), out_path, combo)

def process_all_speechified(input_dir, output_dir, noise_dir):
    os.makedirs(output_dir, exist_ok=True)
    noise_files = get_all_noise_files(noise_dir)
    wind_file_path = str(Path(noise_dir) / 'wind_in_mic.mp3')

    # 1. Define all possible effect variants in separate pools
    db_levels = [-15, -10, -5, 0, 5]
    noise_variants = [{'type': 'background_noise', 'noise_file': str(nf), 'noise_level': db} for nf in noise_files for db in db_levels]

    cut_counts = [2, 4, 5, 7, 10]
    cut_variants = [{'type': 'network_cut', 'probability': 0.3, 'n_cuts': n} for n in cut_counts]

    simple_variants = [
        {'type': 'audio_fade'},
        {'type': 'fluctuate'},
        {'type': 'mic_rubbing', 'probability': 0.1},
        {'type': 'audio_mumbling', 'probability': 0.1},
        {'type': 'heavy_wind', 'wind_file': wind_file_path}
    ]
    all_variants = noise_variants + cut_variants + simple_variants

    speech_files = get_all_audio_files(input_dir)
    print(f"Total audio files found: {len(speech_files)}")

    # 2. Build the final list of combinations in two phases
    final_playlist = []

    # --- PHASE 1: Single effects to guarantee coverage ---
    # Create a combination with only one effect for each unique variant
    print(f"Phase 1: Generating {len(all_variants)} single-effect combinations to ensure coverage...")
    for variant in all_variants:
        if len(final_playlist) < len(speech_files):
            final_playlist.append([variant])
        else:
            break # Stop if we already have enough combos for the number of files

    # --- PHASE 2: Mixed effects for remaining files ---
    remaining_files = len(speech_files) - len(final_playlist)
    if remaining_files > 0:
        print(f"Phase 2: Generating {remaining_files} mixed-effect combinations...")
        for _ in range(remaining_files):
            # Start with a random base effect
            combo = [random.choice(all_variants)]
            
            # Add up to two EXTRA effects
            for _ in range(2): # Loop twice for two potential additions
                # 50% chance to add an extra effect
                if random.random() < 0.5:
                    # Get current effect types in the combo
                    current_types = [e['type'] for e in combo]
                    
                    # Find a new effect that doesn't violate constraints
                    potential_additions = [
                        v for v in all_variants if not (
                            (v['type'] == 'background_noise' and 'background_noise' in current_types) or
                            (v['type'] == 'network_cut' and 'network_cut' in current_types)
                        )
                    ]
                    if potential_additions:
                        combo.append(random.choice(potential_additions))
            final_playlist.append(combo)

    # 3. Shuffle the entire playlist to randomize assignment
    print("Shuffling the final playlist of combinations...")
    random.shuffle(final_playlist)

    # 4. Process all audio files with the generated playlist
    print("Starting audio processing...")
    for i, speech_file in enumerate(speech_files):
        combo = final_playlist[i]
        
        # --- The rest of your file processing logic remains the same ---
        rel_path = speech_file.relative_to(Path(input_dir))
        out_dir = Path(output_dir) / rel_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        base = speech_file.stem
        combo_names = '_'.join(sorted([
            e['type'] +
            (f"_n{e['n_cuts']}" if e['type'] == 'network_cut' else '') +
            (f"_{Path(e['noise_file']).stem}_db{e['noise_level']}" if e['type']=='background_noise' else '')
            for e in combo
        ]))
        out_name = f"{base}_{combo_names}.mp3"
        out_path = str(out_dir / out_name)

        print(f"Processing {speech_file.name} with effects: {combo_names}")
        apply_effects_chain(str(speech_file), out_path, combo)

# Update the input directory to process all audio files
process_all_speechified(
    input_dir="audio",
    output_dir="background_noise/noisy_FINAL",
    noise_dir="background_noise/noise"
)