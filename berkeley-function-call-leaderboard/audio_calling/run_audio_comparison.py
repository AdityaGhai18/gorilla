#!/usr/bin/env python3
"""
Run Audio-to-Audio Transcription Comparison Evaluation

This script compares original audio transcriptions vs noisy audio transcriptions
to analyze how noise affects transcription quality and intent recognition.

No ground truth text needed - just your original audio files!

Usage:
    python run_audio_comparison.py                    # Run full evaluation
    python run_audio_comparison.py --quick            # Quick test
    python run_audio_comparison.py --medium           # Medium test
    python run_audio_comparison.py --effects effect1,effect2  # Specific effects
"""

import argparse
import sys
import os
from pathlib import Path
import json

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

# use the new pipeline orchestration
from audio_calling.pipeline import run_full_experiment, example_llm_judge_wrapper, generate_dataset_for_llm
from audio_calling.background import BackgroundNoiseProcessor

from env_config import (
    AudioComparisonEnvConfig,
    validate_env_config,
    estimate_costs_from_env,
    get_full_effects,
    create_env_file
)


def main():
    """Main function to run the audio comparison evaluation."""
    
    parser = argparse.ArgumentParser(
        description="Run audio-to-audio transcription comparison evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_audio_comparison.py                     # Use config from .env
    python run_audio_comparison.py --quick             # Override to quick test
    python run_audio_comparison.py --medium            # Override to medium test
    python run_audio_comparison.py --effects white_noise,reverb
    python run_audio_comparison.py --validate          # Validate .env setup
    python run_audio_comparison.py --create-env        # Create .env from template
        """
    )
    
    parser.add_argument("--quick", action="store_true", 
                       help="Override to quick test (ignores .env effect setting)")
    parser.add_argument("--medium", action="store_true",
                       help="Override to medium test (ignores .env effect setting)")
    parser.add_argument("--full", action="store_true",
                       help="Override to full test (ignores .env effect setting)")
    parser.add_argument("--effects", type=str,
                       help="Override effects (comma-separated, ignores .env)")
    parser.add_argument("--validate", action="store_true",
                       help="Validate .env configuration")
    parser.add_argument("--create-env", action="store_true",
                       help="Create .env file from template")
    parser.add_argument("--estimate-cost", action="store_true",
                       help="Estimate API costs from .env config")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be done without running")
    
    args = parser.parse_args()
    
    print("🎵 Audio-to-Audio Transcription Comparison")
    print("=" * 60)
    print("Compares original audio transcripts vs noisy audio transcripts")
    print("Configuration loaded from .env file\n")
    
    # Handle special modes first
    if args.create_env:
        create_env_file()
        return 0
    
    if args.validate:
        if validate_env_config():
            print("✅ .env configuration is valid!")
        return 0
    
    if args.estimate_cost:
        print("\n💰 Estimating API costs from .env config...")
        costs = estimate_costs_from_env()
        if "error" in costs:
            print(f"❌ {costs['error']}")
            return 1
        
        print(f"   📁 Audio files: {costs['audio_files']}")
        print(f"   🔊 Effects: {costs['effects']}")
        print(f"   🎛️  Total variants: {costs['total_variants']}")
        print(f"   🎤 Total transcriptions: {costs['total_transcriptions']}")
        print(f"   🧠 Total analyses: {costs['total_analyses']}")
        print(f"   💵 Estimated cost: ${costs['total_estimated_cost']:.2f}")
        return 0
    
    # Load configuration from .env
    try:
        config = AudioComparisonEnvConfig.from_env()
    except Exception as e:
        print(f"❌ Error loading .env configuration: {e}")
        print("💡 Try: python run_audio_comparison.py --create-env")
        return 1
    
    # Validate configuration
    print("🔍 Validating .env configuration...")
    errors = config.validate()
    if errors:
        print("❌ Configuration errors:")
        for error in errors:
            print(f"   - {error}")
        return 1
    
    print("✅ Configuration validated!")
    
    # Override effect set if specified
    if args.quick:
        config.default_effect_set = "quick"
        test_type = "quick (override)"
    elif args.medium:
        config.default_effect_set = "medium"
        test_type = "medium (override)"
    elif args.full:
        config.default_effect_set = "full"
        test_type = "full (override)"
    elif args.effects:
        effect_list = [e.strip() for e in args.effects.split(",")]
        
        # Validate effects exist
        full_effects = get_full_effects()
        invalid_effects = [e for e in effect_list if e not in full_effects]
        if invalid_effects:
            print(f"❌ Invalid effects: {invalid_effects}")
            print(f"Available effects: {list(full_effects.keys())}")
            return 1
        
        config.custom_effects = effect_list
        config.default_effect_set = "custom"
        test_type = f"custom ({len(effect_list)} effects)"
    else:
        test_type = f"{config.default_effect_set} (from .env)"
    
    # Get final configuration
    final_config = config.to_dict()
    effects_spec = config.get_effects_spec()
    
    # Calculate what will be done
    effects_count = len(effects_spec)
    total_variants = sum(len(spec['variants']) for spec in effects_spec.values())
    
    # Estimate based on audio files
    audio_dir = Path(config.audio_directory)
    audio_files = []
    for ext in config.audio_extensions:
        audio_files.extend(list(audio_dir.glob(f"*{ext}")))
    
    total_transcriptions = len(audio_files) * (1 + total_variants)
    
    print(f"""
📋 Evaluation Plan ({test_type}):
   📁 Audio files: {len(audio_files)}
   🔊 Effects to test: {effects_count}
   🎛️  Total variants: {total_variants}
   🎤 Total transcriptions: {total_transcriptions}
   📊 Output: {config.results_dir}
   📈 Visualizations: {config.results_dir}/visualizations/
""")
    
    if args.dry_run:
        print("🔍 Dry run complete. Remove --dry-run to execute.")
        return 0
    
    # Confirm for large runs
    if not args.quick and total_transcriptions > 100:
        print(f"⚠️  This will make {total_transcriptions} transcription API calls.")
        costs = estimate_costs_from_env()
        if "error" not in costs:
            print(f"💰 Estimated cost: ~${costs['total_estimated_cost']:.2f}")
        
        response = input("Continue? (y/N): ")
        if response.lower() != 'y':
            print("Evaluation cancelled.")
            return 0
    
    try:
        print("\n🚀 Starting audio comparison evaluation (controlled-noise pipeline)...")
        
        # Build a simple queries_map: look for sidecar .txt files with same stem, else empty string
        audio_dir_path = Path(config.audio_directory)
        audio_files = []
        for ext in config.audio_extensions:
            audio_files.extend(list(audio_dir_path.glob(f"*{ext}")))
        queries_map = {}
        for p in audio_files:
            stem = p.stem
            # look for same-stem text file in the same dir
            txt_candidates = [p.with_suffix(".txt"), Path(p.parent) / f"{stem}.txt"]
            transcript = ""
            for c in txt_candidates:
                if c.exists():
                    try:
                        transcript = c.read_text(encoding="utf-8").strip()
                    except Exception:
                        transcript = ""
                    break
            queries_map[stem] = transcript
        
        # choose judge: user can replace this with their llm integration
        llm_judge_fn = example_llm_judge_wrapper
        
        # Run the full experiment orchestration that generates variants, runs judge, and aggregates
        summary_path = run_full_experiment(
            audio_dir=str(audio_dir_path),
            queries_map=queries_map,
            effects_spec=effects_spec,
            llm_judge_fn=llm_judge_fn,
            output_root=config.results_dir + "/pipeline_outputs",
            results_dir=config.results_dir + "/pipeline_results",
            audio_glob="*"
        )
        
        print("\n🎉 Controlled-noise experiment completed!")
        print("=" * 60)
        print(f"📊 Summary (experiment): {summary_path}")
        
        # try to print simple aggregated stats from experiment_summary.json
        try:
            with open(summary_path, 'r') as f:
                summary = json.load(f)
            # print top-level aggregation
            by_effect = summary.get("by_effect", {})
            by_error = summary.get("by_error_type", {})
            print(f"\n📈 Aggregated results:")
            print(f"  Effects tested: {len(by_effect)}")
            for ef, data in sorted(by_effect.items()):
                total = data.get("total", 0)
                errs = data.get("error_counts", {})
                top_errs = ", ".join([f"{k}:{v}" for k, v in sorted(errs.items(), key=lambda x: -x[1])[:3]])
                print(f"   - {ef}: total={total}, top_errors={top_errs}")
            print(f"\n  Top error types: {len(by_error)}")
            for et, data in sorted(by_error.items()):
                print(f"   - {et}: total={data.get('total',0)}, effects_count={len(data.get('effects',{}))}")
        except Exception as e:
            print(f"⚠️  Could not load experiment summary: {e}")
        
        print("\n🔗 Next steps:")
        print("   1. Inspect pipeline_results/flat_judgments.csv for per-variant judgments")
        print("   2. Upload the pipeline_outputs folder to your LLM/ASR pipeline if needed")
        print("   3. Replace example_llm_judge_wrapper with your LLM prompt function for richer labels")
        return 0
         
    except KeyboardInterrupt:
        print("\n⏹️  Evaluation interrupted by user.")
        return 1
    except Exception as e:
        print(f"\n❌ Evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


def show_config_info():
    """Show current configuration information."""
    from audio_comparison_config import AUDIO_DIRECTORY, EFFECTS_SPEC, OPENAI_API_KEY
    
    print("📋 Current Configuration:")
    print(f"   📁 Audio directory: {AUDIO_DIRECTORY}")
    print(f"   🔊 Total effects defined: {len(EFFECTS_SPEC)}")
    
    # Check if audio files exist
    audio_dir = Path(AUDIO_DIRECTORY)
    if audio_dir.exists():
        audio_files = []
        for ext in ['.wav', '.mp3', '.flac', '.m4a']:
            audio_files.extend(list(audio_dir.glob(f"*{ext}")))
        print(f"   🎵 Audio files found: {len(audio_files)}")
    else:
        print(f"   ❌ Audio directory not found")
    
    # Show .env and API key status
    print(f"   🔑 OpenAI API key: {'✅ Set' if OPENAI_API_KEY else '❌ Not found in .env'}")
    if Path(".env").exists():
        print(f"   📄 .env file: ✅ Found")
    else:
        print(f"   📄 .env file: ❌ Not found")


def show_available_effects():
    """Show available effects that can be used."""
    from audio_comparison_config import EFFECTS_SPEC
    
    print("🔊 Available Effects:")
    print("-" * 30)
    
    categories = {
        "noise": ["white_noise_light", "white_noise_heavy"],
        "background": ["background_cafe", "background_traffic"], 
        "reverb": ["reverb_small_room", "reverb_large_room"],
        "echo": ["echo_short", "echo_long"],
        "compression": ["compression_light", "compression_heavy"],
        "volume": ["volume_reduction", "volume_boost"],
        "distortion": ["distortion_light", "distortion_heavy"],
        "filter": ["lowpass_filter", "highpass_filter"],
        "speed": ["speed_slow", "speed_fast", "pitch_shift"],
        "dropout": ["audio_dropout", "clipping"]
    }
    
    for category, effects in categories.items():
        print(f"\n{category.title()}:")
        for effect in effects:
            if effect in EFFECTS_SPEC:
                variants = len(EFFECTS_SPEC[effect].get('variants', []))
                print(f"  - {effect} ({variants} variants)")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Show info if no arguments
        show_config_info()
        print("\n🔊 This evaluation compares:")
        print("   Original Audio → Transcription")
        print("   vs")
        print("   Noisy Audio → Transcription") 
        print("\n   No ground truth text needed!")
        print("   Configuration loaded from .env file")
        print("\nUse --help for usage information")
        print("Use --create-env to setup .env file")
        print("Use --validate to check .env setup")
        print("Use --estimate-cost to see API costs")
    elif len(sys.argv) == 2 and sys.argv[1] == "--list-effects":
        show_available_effects()
    else:
        sys.exit(main())