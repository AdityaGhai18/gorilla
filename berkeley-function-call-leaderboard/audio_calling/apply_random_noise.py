"""
Simple Audio Noise Injection Script

Takes audio files from input directory, applies random noise/effects,
outputs to a new directory with same filenames.

PARAMETER VALUES: Aligned with experimental pipeline values from:
- background.py defaults (e.g., 120ms echo delay, 0.6 decay, 8 cuts/beeps)
- generate_variants.py experimental configs (-15, -5, 0 dB for noise)
- test files common values (-30, -20, -10 dB for noise)
This ensures compatibility with systematic experiments in pipeline.py.

Usage:
    python apply_random_noise.py --input_dir ./audio/my_files --output_dir ./noisy_output
    python apply_random_noise.py --input_dir ./audio/my_files --output_dir ./noisy_output --workers 4
    python apply_random_noise.py --input_file single.mp3 --output_dir ./noisy_output
"""

import os
import random
import argparse
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import threading

# Import effect functions from background.py
from background import (
    overlay_audio_with_noise,
    fluctuate_audio_volume,
    apply_gradual_audio_fade,
    apply_network_cut_effect,
    apply_network_beep_effect,
    apply_mic_rubbing_effect,
    apply_audio_mumbling_effect,
    apply_reverb,
    apply_clipping_distortion,
    apply_gain_variation,
    apply_mechanical_interference,
    BackgroundNoiseProcessor
)

# ============================================================================
# EFFECT DEFINITIONS - Each returns a callable that takes (input_path, output_path)
# ============================================================================

def get_noise_files() -> List[str]:
    """Get available noise files from background_noise/noise directory."""
    noise_dir = Path(__file__).parent / "background_noise" / "noise"
    noise_files = []
    for ext in ("*.wav", "*.mp3", "*.webm"):
        noise_files.extend([str(f) for f in noise_dir.glob(ext)])
    return noise_files

NOISE_FILES = None  # Lazy loaded

def _get_noise_files():
    global NOISE_FILES
    if NOISE_FILES is None:
        NOISE_FILES = get_noise_files()
    return NOISE_FILES


# Intensity levels for random selection
INTENSITY_LEVELS = {
    "light": 0,
    "medium": 1, 
    "heavy": 2
}

def random_intensity():
    """Return random intensity: light, medium, or heavy."""
    return random.choice(["light", "medium", "heavy"])


# Define effect generators - each returns random params with random intensity
EFFECT_CONFIGS = {
    "background_noise": {
        "weight": 3,  # Higher weight = more likely to be selected as additional
        "params_fn": lambda: {
            "noise_file": random.choice(_get_noise_files()) if _get_noise_files() else None,
            # EXPERIMENTAL VALUES: generate_variants.py uses exactly [-15, -5, 0]
            # Tight ranges around these 3 experimental values
            "noise_level_db": {
                "light": random.choice([-17, -15, -13]),      # Tight around -15
                "medium": random.choice([-7, -5, -3]),        # Tight around -5
                "heavy": random.choice([-2, 0, 2])            # Tight around 0
            }[random_intensity()],
            "_intensity": random_intensity()
        }
    },
    "reverb_echo": {
        "weight": 2,
        "params_fn": lambda: {
            "mode": "echo",
            # EXPERIMENTAL: generate_variants uses 150ms, 0.6 decay, 3 echoes (via defaults)
            # background.py default: 120ms, 0.6 decay, 3 echoes
            # Tight ranges around 120-150ms, 0.6 decay, 3 echoes
            **{
                "light": {"echo_delay_ms": random.choice([110, 120, 130]), "echo_decay": random.choice([0.55, 0.6, 0.65]), "echo_n": 2},   # Tight around 120ms, 0.6, fewer echoes
                "medium": {"echo_delay_ms": random.choice([140, 150, 160]), "echo_decay": random.choice([0.55, 0.6, 0.65]), "echo_n": 3},  # Tight around 150ms, 0.6, 3 echoes
                "heavy": {"echo_delay_ms": random.choice([150, 160, 170]), "echo_decay": random.choice([0.6, 0.65, 0.7]), "echo_n": 3}     # Tight above 150ms, 0.6-0.7
            }[random_intensity()]
        }
    },
    "reverb_cave": {
        "weight": 2,
        "params_fn": lambda: {
            "mode": "cave",
            # EXPERIMENTAL: background.py default n_reflections=40, max_delay=120, decay=0.6
            # Very tight ranges around these defaults
            **{
                "light": {"n_reflections": random.choice([35, 38, 40]), "max_reflection_delay_ms": random.choice([110, 120, 130]), "decay_mean": random.choice([0.55, 0.6, 0.65])},    # Just below default
                "medium": {"n_reflections": random.choice([38, 40, 42]), "max_reflection_delay_ms": random.choice([115, 120, 125]), "decay_mean": random.choice([0.58, 0.6, 0.62])},   # Tight around default (40, 120, 0.6)
                "heavy": {"n_reflections": random.choice([40, 42, 45]), "max_reflection_delay_ms": random.choice([120, 125, 130]), "decay_mean": random.choice([0.6, 0.62, 0.65])}     # Just above default
            }[random_intensity()]
        }
    },
    "volume_fluctuation": {
        "weight": 2,
        "params_fn": lambda: {
            # EXPERIMENTAL: background.py default min=-15, max=+10, n=5
            # Very tight ranges around these single defaults
            **{
                "light": {"min_db_change": random.choice([-12, -10, -8]), "max_db_change": random.choice([7, 8, 9]), "n_fluctuations": random.choice([3, 4, 5])},       # Just above default
                "medium": {"min_db_change": random.choice([-17, -15, -13]), "max_db_change": random.choice([9, 10, 11]), "n_fluctuations": random.choice([4, 5, 6])},    # Tight around default (-15, +10, 5)
                "heavy": {"min_db_change": random.choice([-18, -16, -15]), "max_db_change": random.choice([10, 12, 14]), "n_fluctuations": random.choice([5, 6, 7])}     # Just below default
            }[random_intensity()]
        }
    },
    "network_cuts": {
        "weight": 1,
        "params_fn": lambda: {
            # EXPERIMENTAL: background.py default n_cuts=8, min=100, max=600
            # Tight around single default values
            **{
                "light": {"n_cuts": random.choice([6, 7, 8]), "min_cut_ms": random.choice([90, 100, 110]), "max_cut_ms": random.choice([500, 550, 600])},      # Just below default
                "medium": {"n_cuts": random.choice([7, 8, 9]), "min_cut_ms": random.choice([95, 100, 105]), "max_cut_ms": random.choice([580, 600, 620])},     # Tight around default (8, 100, 600)
                "heavy": {"n_cuts": random.choice([8, 9, 10]), "min_cut_ms": random.choice([100, 110, 120]), "max_cut_ms": random.choice([600, 650, 700])}     # Just above default
            }[random_intensity()]
        }
    },
    "network_beeps": {
        "weight": 1,
        "params_fn": lambda: {
            # EXPERIMENTAL: background.py default n_beeps=8, freq=1000, db=-10
            # Very tight around single defaults
            **{
                "light": {"n_beeps": random.choice([6, 7, 8]), "beep_freq": random.choice([950, 1000, 1050]), "beep_db": random.choice([-12, -11, -10])},     # Just below default
                "medium": {"n_beeps": random.choice([7, 8, 9]), "beep_freq": random.choice([980, 1000, 1020]), "beep_db": random.choice([-11, -10, -9])},     # Tight around default (8, 1000, -10)
                "heavy": {"n_beeps": random.choice([8, 9, 10]), "beep_freq": random.choice([1000, 1050, 1100]), "beep_db": random.choice([-10, -9, -8])}      # Just above default
            }[random_intensity()]
        }
    },
    "mic_rubbing": {
        "weight": 1,
        "params_fn": lambda: {}  # No params, fixed effect
    },
    "mumbling": {
        "weight": 1,
        "params_fn": lambda: {}  # No params, fixed effect (low-pass filter)
    },
    "walkaway_fade": {
        "weight": 1,
        "params_fn": lambda: {
            # EXPERIMENTAL: background.py default min_db=-30, max_db=0
            # Very tight around single default -30
            **{
                "light": {"min_db": random.choice([-25, -23, -20]), "max_db": 0},      # Just above default (less fade)
                "medium": {"min_db": random.choice([-32, -30, -28]), "max_db": 0},     # Tight around default (-30)
                "heavy": {"min_db": random.choice([-35, -32, -30]), "max_db": 0}       # Just below default (more fade)
            }[random_intensity()]
        }
    },
    "clipping": {
        "weight": 1,
        "params_fn": lambda: {
            # EXPERIMENTAL: background.py default threshold=0.8, drive=0
            # Very tight around single defaults
            **{
                "light": {"clip_threshold": random.choice([0.82, 0.85, 0.88]), "drive_db": random.choice([0, 0, 1])},       # Just above default (less clipping)
                "medium": {"clip_threshold": random.choice([0.78, 0.8, 0.82]), "drive_db": random.choice([0, 1, 2])},       # Tight around default (0.8, 0)
                "heavy": {"clip_threshold": random.choice([0.75, 0.78, 0.8]), "drive_db": random.choice([1, 2, 3])}         # Just below default (more clipping)
            }[random_intensity()]
        }
    },
    "gain_variation": {
        "weight": 1,
        "params_fn": lambda: {
            # EXPERIMENTAL: background.py default min=-12, max=+6, segment=400ms
            # Very tight around single defaults
            **{
                "light": {"min_gain_db": random.choice([-10, -9, -8]), "max_gain_db": random.choice([5, 5, 6]), "segment_ms": random.choice([400, 420, 450])},     # Just above default
                "medium": {"min_gain_db": random.choice([-13, -12, -11]), "max_gain_db": random.choice([5, 6, 7]), "segment_ms": random.choice([380, 400, 420])},   # Tight around default (-12, +6, 400)
                "heavy": {"min_gain_db": random.choice([-14, -13, -12]), "max_gain_db": random.choice([6, 7, 8]), "segment_ms": random.choice([350, 380, 400])}     # Just below default
            }[random_intensity()]
        }
    },
    "mechanical": {
        "weight": 1,
        "params_fn": lambda: {
            # EXPERIMENTAL: background.py default n_rubs=3, n_clicks=8
            # Very tight around single defaults
            **{
                "light": {"n_rubs": random.choice([2, 3, 3]), "n_clicks": random.choice([6, 7, 8])},         # Just below default
                "medium": {"n_rubs": random.choice([2, 3, 4]), "n_clicks": random.choice([7, 8, 9])},        # Tight around default (3 rubs, 8 clicks)
                "heavy": {"n_rubs": random.choice([3, 4, 4]), "n_clicks": random.choice([8, 9, 10])}         # Just above default
            }[random_intensity()]
        }
    }
}


def apply_effect(effect_name: str, input_path: str, output_path: str, params: Dict[str, Any]) -> bool:
    """Apply a single effect to an audio file."""
    try:
        if effect_name == "background_noise":
            if not params.get("noise_file"):
                print(f"  ⚠️  No noise files available, skipping background_noise")
                return False
            overlay_audio_with_noise(
                speech_path=input_path,
                noise_path=params["noise_file"],
                output_path=output_path,
                noise_level_db=params["noise_level_db"]
            )
        elif effect_name == "reverb_echo":
            apply_reverb(input_path, output_path, mode="echo", 
                        echo_delay_ms=params["echo_delay_ms"],
                        echo_decay=params["echo_decay"],
                        echo_n=params["echo_n"])
        elif effect_name == "reverb_cave":
            apply_reverb(input_path, output_path, mode="cave",
                        n_reflections=params["n_reflections"],
                        max_reflection_delay_ms=params["max_reflection_delay_ms"],
                        decay_mean=params["decay_mean"])
        elif effect_name == "volume_fluctuation":
            fluctuate_audio_volume(input_path, output_path,
                                   min_db_change=params["min_db_change"],
                                   max_db_change=params["max_db_change"],
                                   n_fluctuations=params["n_fluctuations"])
        elif effect_name == "network_cuts":
            apply_network_cut_effect(input_path, output_path,
                                     n_cuts=params["n_cuts"],
                                     min_cut_ms=params["min_cut_ms"],
                                     max_cut_ms=params["max_cut_ms"])
        elif effect_name == "network_beeps":
            apply_network_beep_effect(input_path, output_path,
                                      n_beeps=params["n_beeps"],
                                      beep_freq=params["beep_freq"],
                                      beep_db=params["beep_db"])
        elif effect_name == "mic_rubbing":
            apply_mic_rubbing_effect(input_path, output_path)
        elif effect_name == "mumbling":
            apply_audio_mumbling_effect(input_path, output_path)
        elif effect_name == "walkaway_fade":
            apply_gradual_audio_fade(input_path, output_path,
                                     min_db=params["min_db"],
                                     max_db=params["max_db"])
        elif effect_name == "clipping":
            apply_clipping_distortion(input_path, output_path,
                                      clip_threshold=params["clip_threshold"],
                                      drive_db=params["drive_db"])
        elif effect_name == "gain_variation":
            apply_gain_variation(input_path, output_path,
                                min_gain_db=params["min_gain_db"],
                                max_gain_db=params["max_gain_db"],
                                segment_ms=params["segment_ms"])
        elif effect_name == "mechanical":
            apply_mechanical_interference(input_path, output_path,
                                          n_rubs=params["n_rubs"],
                                          n_clicks=params["n_clicks"])
        else:
            print(f"  ⚠️  Unknown effect: {effect_name}")
            return False
        return True
    except Exception as e:
        print(f"  ❌ Error applying {effect_name}: {e}")
        return False


def select_random_effects(n_additional: int = None) -> List[str]:
    """
    Select random effects.
    
    ALWAYS includes background_noise first, then adds 0-2 random additional effects.
    """
    # Always start with background noise
    selected = ["background_noise"]
    
    # Determine how many additional effects (0-2)
    if n_additional is None:
        n_additional = random.randint(0, 2)
    
    # Build weighted list for additional effects (exclude background_noise)
    weighted_effects = []
    for name, config in EFFECT_CONFIGS.items():
        if name != "background_noise":
            weighted_effects.extend([name] * config["weight"])
    
    # Sample additional effects without replacement
    available = [name for name in EFFECT_CONFIGS.keys() if name != "background_noise"]
    for _ in range(min(n_additional, len(available))):
        if not weighted_effects:
            break
        choice = random.choice([e for e in weighted_effects if e in available])
        selected.append(choice)
        available.remove(choice)
        weighted_effects = [e for e in weighted_effects if e != choice]
    
    return selected


def process_single_file(
    input_path: str,
    output_dir: str,
    effect_sequence: List[str] = None,
    preserve_name: bool = True
) -> Dict[str, Any]:
    """
    Process a single audio file with random effects.
    
    Returns dict with results and metadata.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine output filename - preserves original filename, just in new directory
    # Example: input_dir/file.mp3 -> output_dir/file.mp3 (same name, different location)
    if preserve_name:
        output_path = output_dir / input_path.name  # Same filename, different directory
    else:
        output_path = output_dir / f"{input_path.stem}_noisy{input_path.suffix}"
    
    # Select effects if not provided
    if effect_sequence is None:
        effect_sequence = select_random_effects()
    
    result = {
        "input": str(input_path),
        "output": str(output_path),
        "effects": [],
        "success": False
    }
    
    # Apply effects in chain
    current_input = str(input_path)
    temp_files = []
    
    for i, effect_name in enumerate(effect_sequence):
        # Get random params for this effect
        params = EFFECT_CONFIGS[effect_name]["params_fn"]()
        
        # Determine output for this step
        if i == len(effect_sequence) - 1:
            # Last effect - output to final path
            step_output = str(output_path)
        else:
            # Intermediate - use temp file
            temp_path = output_dir / f".temp_{input_path.stem}_{i}{input_path.suffix}"
            step_output = str(temp_path)
            temp_files.append(step_output)
        
        success = apply_effect(effect_name, current_input, step_output, params)
        
        if success:
            result["effects"].append({
                "name": effect_name,
                "params": params
            })
            current_input = step_output
        else:
            # If effect fails, skip to next
            continue
    
    # Clean up temp files
    for temp_file in temp_files:
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        except:
            pass
    
    # If no effects were applied successfully, copy original
    if not result["effects"]:
        import shutil
        shutil.copy(str(input_path), str(output_path))
        result["effects"] = [{"name": "none", "params": {}}]
    
    result["success"] = os.path.exists(str(output_path))
    return result


def get_audio_files(input_dir: str) -> List[str]:
    """Get all audio files from directory."""
    input_dir = Path(input_dir)
    audio_files = []
    for ext in ("*.mp3", "*.wav", "*.flac", "*.m4a", "*.ogg"):
        audio_files.extend([str(f) for f in input_dir.glob(ext)])
    return sorted(audio_files)


def read_jsonl(jsonl_path: str, audio_path_field: str = "audio_path") -> List[Dict[str, Any]]:
    """Read JSONL file and extract records with audio paths."""
    records = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError as e:
                print(f"⚠️  Warning: Skipping invalid JSON at line {line_num}: {e}")
    return records


def write_jsonl_record(jsonl_path: str, record: Dict[str, Any], lock: threading.Lock = None):
    """Append a single record to JSONL file (thread-safe with lock)."""
    if lock:
        with lock:
            with open(jsonl_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
    else:
        with open(jsonl_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


def process_directory(
    input_dir: str,
    output_dir: str,
    workers: int = 1,
    max_additional_effects: int = 2,
    seed: int = None
) -> List[Dict[str, Any]]:
    """
    Process all audio files in a directory.
    
    Each file gets:
    - ALWAYS: background_noise (with random intensity)
    - PLUS: 0-2 additional random effects (with random intensity)
    
    Args:
        input_dir: Directory with input audio files
        output_dir: Directory for output files
        workers: Number of parallel workers (1 = sequential)
        max_additional_effects: Max additional effects on top of background_noise (default: 2)
        seed: Random seed for reproducibility
    """
    # Safety check: prevent overwriting input directory
    input_dir_path = Path(input_dir).resolve()
    output_dir_path = Path(output_dir).resolve()
    
    if input_dir_path == output_dir_path:
        raise ValueError(
            f"❌ ERROR: Input and output directories are the same!\n"
            f"   Input:  {input_dir_path}\n"
            f"   Output: {output_dir_path}\n"
            f"   This would overwrite your original files. Please use a different output directory."
        )
    
    if seed is not None:
        random.seed(seed)
    
    audio_files = get_audio_files(input_dir)
    
    if not audio_files:
        print(f"❌ No audio files found in {input_dir}")
        return []
    
    print(f"🎵 Found {len(audio_files)} audio files")
    print(f"📂 Output directory: {output_dir}")
    print(f"⚙️  Workers: {workers}")
    print(f"🎛️  Effects: background_noise (always) + 0-{max_additional_effects} random effects")
    print(f"🎲 Intensity: randomly sampled (light/medium/heavy) per effect")
    print()
    
    # Pre-generate effect sequences - each file gets random selection
    # ALWAYS background_noise + 0-2 random additional effects
    effect_sequences = []
    for _ in audio_files:
        n_additional = random.randint(0, max_additional_effects)
        sequence = select_random_effects(n_additional)
        effect_sequences.append(sequence)
    
    results = []
    
    if workers == 1:
        # Sequential processing with progress bar
        for audio_file, effects in tqdm(zip(audio_files, effect_sequences), 
                                         total=len(audio_files), 
                                         desc="Processing"):
            result = process_single_file(audio_file, output_dir, effects)
            results.append(result)
            if result["success"]:
                effect_names = [e["name"] for e in result["effects"]]
                tqdm.write(f"  ✅ {Path(audio_file).name} → {effect_names}")
    else:
        # Parallel processing
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(process_single_file, audio_file, output_dir, effects): audio_file
                for audio_file, effects in zip(audio_files, effect_sequences)
            }
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
                audio_file = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    if result["success"]:
                        effect_names = [e["name"] for e in result["effects"]]
                        tqdm.write(f"  ✅ {Path(audio_file).name} → {effect_names}")
                except Exception as e:
                    tqdm.write(f"  ❌ {Path(audio_file).name}: {e}")
                    results.append({
                        "input": audio_file,
                        "output": None,
                        "effects": [],
                        "success": False,
                        "error": str(e)
                    })
    
    # Summary
    successful = sum(1 for r in results if r["success"])
    print(f"\n✅ Processed {successful}/{len(audio_files)} files successfully")
    
    return results


def process_jsonl(
    input_jsonl: str,
    output_jsonl: str,
    output_audio_dir: str,
    audio_path_field: str = "audio_path",
    workers: int = 1,
    max_additional_effects: int = 2,
    seed: int = None
) -> int:
    """
    Process audio files from JSONL, update paths, save incrementally.
    
    Args:
        input_jsonl: Path to input JSONL file
        output_jsonl: Path to output JSONL file (will be created/appended)
        output_audio_dir: Directory for output audio files
        audio_path_field: Field name containing audio path in JSONL records
        workers: Number of parallel workers
        max_additional_effects: Max additional effects beyond background_noise
        seed: Random seed for reproducibility
        
    Returns:
        Number of successfully processed records
    """
    # Safety checks
    input_jsonl_path = Path(input_jsonl).resolve()
    output_jsonl_path = Path(output_jsonl).resolve()
    
    if input_jsonl_path == output_jsonl_path:
        raise ValueError(
            f"❌ ERROR: Input and output JSONL paths are the same!\n"
            f"   Input:  {input_jsonl_path}\n"
            f"   Output: {output_jsonl_path}\n"
            f"   This would overwrite your original file. Please use a different output path."
        )
    
    # Read input JSONL
    print(f"📄 Reading JSONL from: {input_jsonl}")
    records = read_jsonl(input_jsonl, audio_path_field)
    
    if not records:
        print(f"❌ No records found in {input_jsonl}")
        return 0
    
    print(f"🎵 Found {len(records)} records")
    print(f"📂 Output audio directory: {output_audio_dir}")
    print(f"📄 Output JSONL: {output_jsonl}")
    print(f"⚙️  Workers: {workers}")
    print(f"🎛️  Effects: background_noise (always) + 0-{max_additional_effects} random effects")
    print(f"💾 Saving incrementally after each file")
    print()
    
    # Create output audio directory
    Path(output_audio_dir).mkdir(parents=True, exist_ok=True)
    
    # Create/clear output JSONL file
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        pass  # Just create empty file
    
    # Set seed
    if seed is not None:
        random.seed(seed)
    
    # Pre-generate effect sequences
    effect_sequences = []
    for _ in records:
        n_additional = random.randint(0, max_additional_effects)
        sequence = select_random_effects(n_additional)
        effect_sequences.append(sequence)
    
    # Thread-safe file writing lock
    write_lock = threading.Lock()
    successful = 0
    
    def process_and_save_record(record, effects):
        """Process a single record and save to JSONL immediately."""
        nonlocal successful
        
        # Get original audio path
        audio_path = record.get(audio_path_field)
        if not audio_path or not os.path.exists(audio_path):
            print(f"  ⚠️  Skipping record: audio file not found at '{audio_path}'")
            return
        
        # Process audio file
        result = process_single_file(audio_path, output_audio_dir, effects)
        
        # Update record with new audio path and processing info
        updated_record = record.copy()
        if result["success"]:
            updated_record[f"{audio_path_field}_noisy"] = result["output"]
            updated_record["noise_effects_applied"] = [
                {
                    "name": e["name"],
                    "intensity": e["params"].get("_intensity", "n/a")
                }
                for e in result["effects"]
            ]
            updated_record["noise_processing_success"] = True
            successful += 1
            
            effect_names = [e["name"] for e in result["effects"]]
            tqdm.write(f"  ✅ {Path(audio_path).name} → {effect_names}")
        else:
            updated_record["noise_processing_success"] = False
            updated_record["noise_processing_error"] = result.get("error", "Unknown error")
            tqdm.write(f"  ❌ {Path(audio_path).name}: {result.get('error', 'Failed')}")
        
        # Write to JSONL immediately (incremental save)
        write_jsonl_record(output_jsonl, updated_record, lock=write_lock)
    
    if workers == 1:
        # Sequential processing
        for record, effects in tqdm(zip(records, effect_sequences), 
                                    total=len(records), 
                                    desc="Processing"):
            process_and_save_record(record, effects)
    else:
        # Parallel processing
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(process_and_save_record, record, effects): record
                for record, effects in zip(records, effect_sequences)
            }
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
                try:
                    future.result()
                except Exception as e:
                    record = futures[future]
                    audio_path = record.get(audio_path_field, "unknown")
                    tqdm.write(f"  ❌ {Path(audio_path).name}: {e}")
    
    print(f"\n✅ Processed {successful}/{len(records)} records successfully")
    print(f"📄 Results saved to: {output_jsonl}")
    
    return successful


def main():
    parser = argparse.ArgumentParser(
        description="Apply random noise/effects to audio files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process entire directory
  python apply_random_noise.py --input_dir ./audio/clean --output_dir ./audio/noisy

  # Process with 4 parallel workers
  python apply_random_noise.py --input_dir ./audio/clean --output_dir ./audio/noisy --workers 4

  # Process single file
  python apply_random_noise.py --input_file ./audio/test.mp3 --output_dir ./audio/noisy

  # Reproducible with seed
  python apply_random_noise.py --input_dir ./audio/clean --output_dir ./audio/noisy --seed 42

  # Process from JSONL (reads audio paths from JSONL, saves incrementally)
  python apply_random_noise.py --input_jsonl ./data.jsonl --output_jsonl ./data_noisy.jsonl --output_dir ./audio/noisy
  
  # JSONL with custom audio path field and parallel processing
  python apply_random_noise.py --input_jsonl ./data.jsonl --output_jsonl ./data_noisy.jsonl --output_dir ./audio/noisy --audio_path_field "file_path" --workers 4

How it works:
  - ALWAYS applies background_noise (random intensity: light/medium/heavy)
  - PLUS 0-2 additional random effects (also random intensity each)
  - Total effects per file: 1-3

Available additional effects:
  - reverb_echo: Echo effect (light/medium/heavy delay & decay)
  - reverb_cave: Cave-like reverb (light/medium/heavy reflections)
  - volume_fluctuation: Random volume dips/spikes
  - network_cuts: Silence dropouts (light=few short, heavy=many long)
  - network_beeps: Beep dropouts  
  - mic_rubbing: Low-frequency rumble
  - mumbling: Low-pass filter (muffled sound)
  - walkaway_fade: Volume fade out/in
  - clipping: Distortion/overload (light=subtle, heavy=harsh)
  - gain_variation: Segment-wise gain changes
  - mechanical: Clicks and thumps

Intensity levels (randomly sampled per effect):
  - light: Subtle, barely noticeable
  - medium: Noticeable but not overwhelming  
  - heavy: Strong effect, clearly audible
        """
    )
    
    parser.add_argument("--input_dir", type=str, help="Directory with input audio files")
    parser.add_argument("--input_file", type=str, help="Single input audio file")
    parser.add_argument("--input_jsonl", type=str, help="Input JSONL file (each line has audio path)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for audio files")
    parser.add_argument("--output_jsonl", type=str, help="Output JSONL file (updated with new audio paths, saved incrementally)")
    parser.add_argument("--audio_path_field", type=str, default="audio_path", help="Field name in JSONL containing audio path (default: audio_path)")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers (default: 1)")
    parser.add_argument("--max_additional", type=int, default=2, help="Max additional effects beyond background_noise (default: 2, so 1-3 total)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Validate input arguments
    input_sources = sum([
        args.input_dir is not None,
        args.input_file is not None,
        args.input_jsonl is not None
    ])
    
    if input_sources == 0:
        parser.error("Must specify one of: --input_dir, --input_file, or --input_jsonl")
    elif input_sources > 1:
        parser.error("Cannot specify multiple input sources (choose one: --input_dir, --input_file, or --input_jsonl)")
    
    # JSONL mode requires output_jsonl
    if args.input_jsonl and not args.output_jsonl:
        parser.error("--input_jsonl requires --output_jsonl to be specified")
    
    if args.input_jsonl:
        # JSONL mode
        process_jsonl(
            input_jsonl=args.input_jsonl,
            output_jsonl=args.output_jsonl,
            output_audio_dir=args.output_dir,
            audio_path_field=args.audio_path_field,
            workers=args.workers,
            max_additional_effects=args.max_additional,
            seed=args.seed
        )
    elif args.input_file:
        # Single file mode
        # Safety check: prevent overwriting input file
        input_file_path = Path(args.input_file).resolve()
        output_dir_path = Path(args.output_dir).resolve()
        output_file_path = output_dir_path / input_file_path.name
        
        if input_file_path == output_file_path:
            parser.error(
                f"❌ ERROR: Output file would overwrite input file!\n"
                f"   Input:  {input_file_path}\n"
                f"   Output: {output_file_path}\n"
                f"   Please use a different output directory."
            )
        
        if args.seed:
            random.seed(args.seed)
        result = process_single_file(args.input_file, args.output_dir)
        if result["success"]:
            effect_names = [e["name"] for e in result["effects"]]
            print(f"✅ {Path(args.input_file).name} → {effect_names}")
            print(f"   Output: {result['output']}")
            # Show intensity info
            for eff in result["effects"]:
                intensity = eff["params"].get("_intensity", "n/a")
                print(f"   • {eff['name']}: intensity={intensity}")
        else:
            print(f"❌ Failed to process {args.input_file}")
    else:
        # Directory mode
        process_directory(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            workers=args.workers,
            max_additional_effects=args.max_additional,
            seed=args.seed
        )


if __name__ == "__main__":
    main()

