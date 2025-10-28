"""
Pipeline adapter for using your existing background.py functions
with the audio comparison evaluation system.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any
import random
random.seed(42)

# Import your existing background functions
try:
    from background import (
        BackgroundNoiseProcessor,
        overlay_audio_with_noise,
        fluctuate_audio_volume,
        apply_gradual_audio_fade,
        apply_network_cut_effect,
        apply_network_beep_effect,
        apply_mic_rubbing_effect,
        apply_audio_mumbling_effect,
        apply_heavy_wind_effect,
        apply_reverb,
        apply_clipping_distortion,
        apply_gain_variation,
        apply_mechanical_interference
    )
    BACKGROUND_FUNCTIONS_AVAILABLE = True
    print("✅ Successfully imported your background.py functions")
except ImportError as e:
    print(f"⚠️  Could not import some background.py functions: {e}")
    # Try importing just the core functions
    try:
        from background import (
            BackgroundNoiseProcessor,
            overlay_audio_with_noise,
            fluctuate_audio_volume,
            apply_reverb
        )
        BACKGROUND_FUNCTIONS_AVAILABLE = True
        print("✅ Successfully imported core background.py functions")
    except ImportError as e:
        print(f"❌ Could not import background.py functions: {e}")
        BACKGROUND_FUNCTIONS_AVAILABLE = False


def generate_variants(audio_paths: List[str], effects_spec: Dict[str, Dict], output_root: str) -> List[Dict[str, Any]]:
    """
    Generate audio variants using your existing background.py functions.
    
    Args:
        audio_paths: List of original audio file paths
        effects_spec: Dictionary defining effects and parameters from config
        output_root: Root directory for output files
        
    Returns:
        List of variant records with metadata
    """
    if not BACKGROUND_FUNCTIONS_AVAILABLE:
        raise RuntimeError("Background functions not available. Check background.py import.")

    variant_records = []
    output_root = Path(output_root)

    # Initialize noise processor for background noise effects
    noise_processor = BackgroundNoiseProcessor()
    # Get all noise files for background noise effects
    noise_files = noise_processor.noise_files

    # Only use a random 1/3rd subset of the audio files for all effects
    if len(audio_paths) > 3:
        random.seed(42)
        audio_paths = random.sample(audio_paths, max(1, len(audio_paths) // 3))

    print(f"🎛️  Generating variants for {len(audio_paths)} audio files...")

    for audio_path in audio_paths:
        audio_base = Path(audio_path).stem
        print(f"   Processing: {audio_base}")

        for effect_key, spec in effects_spec.items():
            fn_name = spec.get("fn")
            variants = spec.get("variants", [{}])

            # Special handling for background noise and heavy wind effects
            if fn_name in ("apply_background_noise"):
                # For each noise file and each dB level, create a variant
                db_levels = [-15, -5, 0]
                for noise_file in noise_files:
                    for db in db_levels:
                        subdir = output_root / audio_base / effect_key
                        subdir.mkdir(parents=True, exist_ok=True)
                        noise_name = Path(noise_file).stem
                        variant_fname = f"{audio_base}_{effect_key}_{noise_name}_db{db}.mp3"
                        variant_path = subdir / variant_fname
                        params = {"noise_level": db, "noise_file": noise_file}
                        try:
                            success = apply_effect_by_name(
                                fn_name=fn_name,
                                audio_path=audio_path,
                                output_path=str(variant_path),
                                params=params,
                                noise_processor=noise_processor
                            )
                            if success:
                                record = {
                                    "audio_path": audio_path,
                                    "audio_base": audio_base,
                                    "effect_key": effect_key,
                                    "variant_idx": f"{noise_name}_db{db}",
                                    "variant_path": str(variant_path),
                                    "params": params,
                                    "function_name": fn_name
                                }
                                variant_records.append(record)
                                print(f"      ✅ {effect_key}_{noise_name}_db{db}")
                            else:
                                print(f"      ❌ Failed: {effect_key}_{noise_name}_db{db}")
                        except Exception as e:
                            print(f"      ❌ Error {effect_key}_{noise_name}_db{db}: {str(e)}")
                            continue
                continue  # skip normal variant loop for this effect

            if fn_name == "apply_heavy_wind_effect":
                # For each noise file and each dB level, create a variant
                db_levels = [-15, -5, 0]
                for noise_file in noise_files:
                    for db in db_levels:
                        subdir = output_root / audio_base / effect_key
                        subdir.mkdir(parents=True, exist_ok=True)
                        noise_name = Path(noise_file).stem
                        variant_fname = f"{audio_base}_{effect_key}_{noise_name}_db{db}.mp3"
                        variant_path = subdir / variant_fname
                        params = dict(spec.get("base_params", {}))
                        params.update({"wind_noise_path": noise_file, "noise_level_db": db})
                        try:
                            # Call the actual background.py function directly
                            apply_heavy_wind_effect(
                                audio_path=audio_path,
                                wind_noise_path=noise_file,
                                output_path=str(variant_path),
                                noise_level_db=db,
                                n_hits=params.get("n_hits", 5),
                                hit_min_freq=params.get("hit_min_freq", 30),
                                hit_max_freq=params.get("hit_max_freq", 60),
                                hit_min_duration_ms=params.get("hit_min_duration_ms", 80),
                                hit_max_duration_ms=params.get("hit_max_duration_ms", 250),
                                hit_db=params.get("hit_db", 10)
                            )
                            record = {
                                "audio_path": audio_path,
                                "audio_base": audio_base,
                                "effect_key": effect_key,
                                "variant_idx": f"{noise_name}_db{db}",
                                "variant_path": str(variant_path),
                                "params": params,
                                "function_name": fn_name
                            }
                            variant_records.append(record)
                            print(f"      ✅ {effect_key}_{noise_name}_db{db}")
                        except Exception as e:
                            print(f"      ❌ Error {effect_key}_{noise_name}_db{db}: {str(e)}")
                            continue
                continue  # skip normal variant loop for this effect

            # Double overlay: competing speech (always use fixed file)
            if fn_name == "competing_speech_overlay":
                db_levels = spec.get("db_levels", [-15, -5, 0])
                competing_noise_file = "./audio/BFCL_v3_java/java_0_openai.mp3"
                if not os.path.exists(competing_noise_file):
                    print(f"⚠️  Competing speech file not found: {competing_noise_file}")
                    continue
                # Never same as audio_path (by definition)
                noise_name = Path(competing_noise_file).stem
                for db in db_levels:
                    subdir = output_root / audio_base / effect_key
                    subdir.mkdir(parents=True, exist_ok=True)
                    variant_fname = f"{audio_base}_{effect_key}_{noise_name}_db{db}.mp3"
                    variant_path = subdir / variant_fname
                    params = {"noise_file": competing_noise_file, "noise_level": db}
                    try:
                        success = apply_effect_by_name(
                            fn_name=fn_name,
                            audio_path=audio_path,
                            output_path=str(variant_path),
                            params=params
                        )
                        if success:
                            record = {
                                "audio_path": audio_path,
                                "audio_base": audio_base,
                                "effect_key": effect_key,
                                "variant_idx": f"{noise_name}_db{db}",
                                "variant_path": str(variant_path),
                                "params": params,
                                "function_name": fn_name
                            }
                            variant_records.append(record)
                            print(f"      ✅ {effect_key}_{noise_name}_db{db}")
                        else:
                            print(f"      ❌ Failed: {effect_key}_{noise_name}_db{db}")
                    except Exception as e:
                        print(f"      ❌ Error {effect_key}_{noise_name}_db{db}: {str(e)}")
                        continue
                continue  # skip normal variant loop for this effect

            # Normal effect handling
            for i, params in enumerate(variants):
                subdir = output_root / audio_base / effect_key
                subdir.mkdir(parents=True, exist_ok=True)
                variant_fname = f"{audio_base}_{effect_key}_v{i}.mp3"
                variant_path = subdir / variant_fname
                try:
                    success = apply_effect_by_name(
                        fn_name=fn_name,
                        audio_path=audio_path,
                        output_path=str(variant_path),
                        params=params,
                        noise_processor=noise_processor
                    )
                    if success:
                        record = {
                            "audio_path": audio_path,
                            "audio_base": audio_base,
                            "effect_key": effect_key,
                            "variant_idx": i,
                            "variant_path": str(variant_path),
                            "params": params,
                            "function_name": fn_name
                        }
                        variant_records.append(record)
                        print(f"      ✅ {effect_key}_v{i}")
                    else:
                        print(f"      ❌ Failed: {effect_key}_v{i}")
                except Exception as e:
                    print(f"      ❌ Error {effect_key}_v{i}: {str(e)}")
                    continue

    print(f"✅ Generated {len(variant_records)} audio variants")
    return variant_records


def apply_effect_by_name(fn_name: str, audio_path: str, output_path: str, params: Dict, noise_processor=None) -> bool:
    """
    Apply an audio effect by function name using your existing background.py functions.
    
    Args:
        fn_name: Name of the effect function
        audio_path: Input audio file path
        output_path: Output audio file path
        params: Effect parameters
        noise_processor: BackgroundNoiseProcessor instance for noise effects
        
    Returns:
        True if successful, False otherwise
    """
    
    try:
        # Remove apply_white_noise, not needed
        # if fn_name == "apply_white_noise":
        #     return apply_background_noise_effect(
        #         audio_path, output_path, params, noise_processor, noise_type="white"
        #     )
        if fn_name == "apply_background_noise":
            # Use the specific noise file from params (no indirection)
            noise_file = params.get("noise_file")
            noise_level = params.get("noise_level", -20)
            if not noise_file:
                print(f"⚠️  No noise_file specified in params for background noise")
                return False
            if not os.path.exists(noise_file):
                print(f"⚠️  Noise file not found: {noise_file}")
                return False
            noise_processor.add_background_noise(
                audio_path=audio_path,
                noise_file=noise_file,
                output_path=output_path,
                noise_level=noise_level
            )
            return True
            
        elif fn_name == "apply_audio_overlay":
            # Use your overlay_audio_with_noise function
            noise_level = params.get("noise_level", -20)
            noise_file = params.get("noise_file", "default_noise.mp3")  # You'd specify this
            overlay_audio_with_noise(audio_path, noise_file, output_path, noise_level)
            return True
            
        elif fn_name == "apply_volume_fluctuation":
            # Use your fluctuate_audio_volume function
            fluctuate_audio_volume(
                audio_path=audio_path,
                output_path=output_path,
                min_db_change=params.get("min_db_change", -15),
                max_db_change=params.get("max_db_change", 10),
                min_duration_ms=params.get("min_duration_ms", 500),
                max_duration_ms=params.get("max_duration_ms", 3000),
                n_fluctuations=params.get("n_fluctuations", 5)
            )
            return True
            
        elif fn_name == "apply_filter" or fn_name == "apply_frequency_filter":
            # Use your mumbling effect as a simple lowpass filter
            apply_audio_mumbling_effect(
                audio_path=audio_path,
                output_path=output_path
            )
            return True
            
        elif fn_name == "apply_time_stretch" or fn_name == "apply_speed_change":
            # Map to gradual audio fade as closest equivalent
            apply_gradual_audio_fade(
                speech_path=audio_path,
                output_path=output_path,
                min_db=params.get("min_db", -20),
                max_db=params.get("max_db", 0)
            )
            return True
            
        elif fn_name == "apply_pitch_shift":
            # Use mic rubbing effect as substitute
            apply_mic_rubbing_effect(
                audio_path=audio_path,
                output_path=output_path
            )
            return True
            
        elif fn_name == "apply_reverb":
            # Use your apply_reverb function
            apply_reverb(
                audio_path=audio_path,
                output_path=output_path,
                mode=params.get("mode", "echo"),
                echo_delay_ms=params.get("echo_delay_ms", 150),
                echo_decay=params.get("echo_decay", 0.6),
                echo_n=params.get("echo_n", 3)
            )
            return True
            
        elif fn_name == "apply_clipping" or fn_name == "apply_distortion":
            # Use your apply_clipping_distortion function if available
            try:
                apply_clipping_distortion(
                    audio_path=audio_path,
                    output_path=output_path,
                    clip_threshold=params.get("clip_threshold", 0.8),
                    drive_db=params.get("drive_db", 0.0)
                )
                return True
            except NameError:
                # Fall back to network beep effect
                apply_network_beep_effect(
                    audio_path=audio_path,
                    output_path=output_path
                )
                return True
            
        elif fn_name == "apply_gain_variation" or fn_name == "apply_compression":
            # Use your apply_gain_variation function if available
            try:
                apply_gain_variation(
                    audio_path=audio_path,
                    output_path=output_path,
                    min_gain_db=params.get("min_gain_db", -12),
                    max_gain_db=params.get("max_gain_db", 6),
                    segment_ms=params.get("segment_ms", 400)
                )
                return True
            except NameError:
                # Fall back to volume fluctuation
                fluctuate_audio_volume(
                    audio_path=audio_path,
                    output_path=output_path,
                    min_db_change=params.get("min_db_change", -10),
                    max_db_change=params.get("max_db_change", 5),
                    n_fluctuations=params.get("n_fluctuations", 3)
                )
                return True
            
        elif fn_name == "apply_mechanical_interference":
            # Use your apply_mechanical_interference function if available
            try:
                apply_mechanical_interference(
                    audio_path=audio_path,
                    output_path=output_path,
                    n_rubs=params.get("n_rubs", 3),
                    rub_freq_range=params.get("rub_freq_range", (40, 120)),
                    rub_db=params.get("rub_db", -6),
                    n_clicks=params.get("n_clicks", 8),
                    click_db=params.get("click_db", 0)
                )
                return True
            except NameError:
                # Fall back to mic rubbing effect
                apply_mic_rubbing_effect(
                    audio_path=audio_path,
                    output_path=output_path
                )
                return True
            
        # Compatibility mappings for common effect names
        elif fn_name == "apply_network_cuts":
            # Use your network cut effect
            apply_network_cut_effect(
                audio_path=audio_path,
                output_path=output_path,
                min_cut_ms=params.get("min_cut_ms", 100),
                max_cut_ms=params.get("max_cut_ms", 600),
                n_cuts=params.get("n_cuts", 8)
            )
            return True
            
        elif fn_name == "apply_echo":
            # Map to reverb with echo mode
            apply_reverb(
                audio_path=audio_path,
                output_path=output_path,
                mode="echo",
                echo_delay_ms=params.get("delay_ms", 200),
                echo_decay=params.get("decay", 0.5),
                echo_n=params.get("echo_n", 2)
            )
            return True
            
        elif fn_name == "apply_volume_change":
            # Use gradual audio fade
            volume_factor = params.get("volume_factor", 0.5)
            if volume_factor < 1.0:
                min_db = 20 * log10(volume_factor) if volume_factor > 0 else -60
                max_db = 0
            else:
                min_db = 0
                max_db = 20 * log10(volume_factor)
            
            apply_gradual_audio_fade(
                speech_path=audio_path,
                output_path=output_path,
                min_db=min_db,
                max_db=max_db
            )
            return True
            
        elif fn_name == "apply_dropout":
            # Use network cut effect to simulate dropouts
            apply_network_cut_effect(
                audio_path=audio_path,
                output_path=output_path,
                min_cut_ms=50,
                max_cut_ms=200,
                n_cuts=int(params.get("dropout_probability", 0.1) * 20)
            )
            return True
            
        elif fn_name == "apply_gradual_audio_fade":
            # Directly call the background function
            apply_gradual_audio_fade(
                speech_path=audio_path,
                output_path=output_path,
                min_db=params.get("min_db", -30),
                max_db=params.get("max_db", 0)
            )
            return True

        elif fn_name == "apply_mic_rubbing_effect":
            apply_mic_rubbing_effect(
                audio_path=audio_path,
                output_path=output_path
            )
            return True

        elif fn_name == "apply_audio_mumbling_effect":
            apply_audio_mumbling_effect(
                audio_path=audio_path,
                output_path=output_path
            )
            return True
            
        elif fn_name == "competing_speech_overlay":
            # Overlay another clean audio file as noise
            noise_file = params.get("noise_file")
            noise_level = params.get("noise_level", -20)
            if not noise_file or not os.path.exists(noise_file):
                print(f"⚠️  Competing speech noise file not found: {noise_file}")
                return False
            overlay_audio_with_noise(audio_path, noise_file, output_path, noise_level)
            return True
            
        else:
            print(f"⚠️  Unknown effect function: {fn_name}")
            return False
            
    except Exception as e:
        print(f"❌ Error applying {fn_name}: {str(e)}")
        return False


def apply_background_noise_effect(audio_path: str, output_path: str, params: Dict, 
                                noise_processor, noise_type: str = "random") -> bool:
    """
    Apply background noise using your BackgroundNoiseProcessor.
    
    Args:
        audio_path: Input audio file
        output_path: Output audio file  
        params: Effect parameters
        noise_processor: BackgroundNoiseProcessor instance
        noise_type: Type of noise to apply
        
    Returns:
        True if successful
    """
    try:
        noise_level = params.get("noise_level", -20)
        
        # Get available noise files
        if not noise_processor.noise_files:
            print("⚠️  No noise files available in BackgroundNoiseProcessor")
            return False
        
        # Select noise file
        if noise_type == "white":
            # Look for white noise file or use first available
            noise_file = noise_processor.noise_files[0]  # Fallback
            for nf in noise_processor.noise_files:
                if "white" in Path(nf).name.lower():
                    noise_file = nf
                    break
        else:
            # Random selection
            noise_file = random.choice(noise_processor.noise_files)
        
        # Apply the noise
        noise_processor.add_background_noise(
            audio_path=audio_path,
            noise_file=noise_file,
            output_path=output_path,
            noise_level=noise_level
        )
        return True
        
    except Exception as e:
        print(f"❌ Background noise error: {str(e)}")
        return False


# Math import for volume calculations
def log10(x):
    """Simple log10 implementation"""
    import math
    return math.log10(x)


def validate_pipeline():
    """Validate that the pipeline can work with your background.py functions."""
    
    if not BACKGROUND_FUNCTIONS_AVAILABLE:
        return False, "Background functions not available"
    
    # Test noise processor initialization
    try:
        processor = BackgroundNoiseProcessor()
        if not processor.noise_files:
            return False, "No noise files found in BackgroundNoiseProcessor"
    except Exception as e:
        return False, f"Could not initialize BackgroundNoiseProcessor: {e}"
    
    return True, "Pipeline validation successful"


if __name__ == "__main__":
    print("🎛️  Audio Pipeline Adapter for Your Background.py")
    print("=" * 60)
    
    # Validate pipeline
    valid, message = validate_pipeline()
    if valid:
        print(f"✅ {message}")
        
        # Show available functions
        available_functions = [
            "apply_background_noise", "apply_audio_overlay",
            "apply_volume_fluctuation", "apply_reverb", "apply_gradual_audio_fade",
            "apply_network_cuts", "apply_mic_rubbing_effect", "apply_audio_mumbling_effect",
            # Compatibility mappings
            "apply_echo", "apply_volume_change", "apply_dropout",
            "apply_filter", "apply_frequency_filter", "apply_time_stretch", 
            "apply_speed_change", "apply_pitch_shift"
        ]
        
        print(f"\n🎯 Available effect functions: {len(available_functions)}")
        for func in available_functions:
            print(f"   - {func}")
            
    else:
        print(f"❌ {message}")
        print("💡 Make sure your background.py file is in the same directory")