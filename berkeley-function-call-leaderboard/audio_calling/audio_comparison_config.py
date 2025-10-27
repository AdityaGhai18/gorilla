"""
Configuration for Audio-to-Audio Transcription Comparison

This configuration is for comparing original audio transcriptions 
vs noisy audio transcriptions (no ground truth text needed).
"""

import os
from pathlib import Path

# =============================================================================
# SETUP CONFIGURATION
# =============================================================================

# Load environment variables from .env file
def load_env_file(env_path=".env"):
    """Load environment variables from .env file."""
    env_vars = {}
    try:
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip()
    except FileNotFoundError:
        print(f"Warning: {env_path} file not found. Please create one with OPENAI_API_KEY=your-key")
    return env_vars

# Load .env variables
_env_vars = load_env_file()

# OpenAI API Configuration
OPENAI_API_KEY = _env_vars.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
TRANSCRIPTION_MODEL = "whisper-1"              # OpenAI Whisper model
ANALYSIS_MODEL = "gpt-4-turbo-preview"         # GPT model for intent analysis

# Paths
AUDIO_DIRECTORY = "./audio/BFCL_v3_live_simple"   # Your original audio files
OUTPUT_ROOT = "./audio_comparison_outputs"     # Where noisy variants will be saved
RESULTS_DIR = "./audio_comparison_results"     # Where results and reports will be saved

# Audio file extensions to process
AUDIO_EXTENSIONS = [".wav", ".mp3", ".flac", ".m4a"]

# =============================================================================
# NOISE EFFECTS CONFIGURATION (Using Your Actual Functions)
# =============================================================================

# Define noise effects to apply to your original audio
# These map to functions that actually exist in your background.py file

EFFECTS_SPEC = {
    # Background Noise Effects (using your BackgroundNoiseProcessor)
    "background_noise_light": {
        "fn": "apply_background_noise",
        "variants": [
            {"noise_level": -30},  # Very quiet background
            {"noise_level": -25},  # Light background
            {"noise_level": -20}   # Moderate background
        ]
    },
    "background_noise_heavy": {
        "fn": "apply_background_noise", 
        "variants": [
            {"noise_level": -15},  # Heavy background
            {"noise_level": -10},  # Very heavy background
            {"noise_level": -5}    # Competing audio level
        ]
    },
    
    # White Noise (using your background noise system)
    "white_noise_light": {
        "fn": "apply_white_noise",
        "variants": [
            {"noise_level": -30},
            {"noise_level": -25},
            {"noise_level": -20}
        ]
    },
    
    # Volume Fluctuations (using your fluctuate_audio_volume)
    "volume_fluctuation_mild": {
        "fn": "apply_volume_fluctuation",
        "variants": [
            {
                "min_db_change": -8,
                "max_db_change": 5,
                "min_duration_ms": 800,
                "max_duration_ms": 2000,
                "n_fluctuations": 3
            },
            {
                "min_db_change": -12,
                "max_db_change": 8,
                "min_duration_ms": 500,
                "max_duration_ms": 1500,
                "n_fluctuations": 5
            }
        ]
    },
    
    # Reverb Effects (using your apply_reverb)
    "reverb_echo": {
        "fn": "apply_reverb",
        "variants": [
            {
                "mode": "echo",
                "echo_delay_ms": 100,
                "echo_decay": 0.4,
                "echo_n": 2
            },
            {
                "mode": "echo", 
                "echo_delay_ms": 200,
                "echo_decay": 0.6,
                "echo_n": 3
            }
        ]
    },
    "reverb_cave": {
        "fn": "apply_reverb",
        "variants": [
            {
                "mode": "cave",
                "n_reflections": 8,
                "max_reflection_delay_ms": 200,
                "decay_mean": 0.3,
                "decay_std": 0.1,
                "lowpass_freq": 4000
            }
        ]
    },
    
    # Network Effects (using your network functions)
    "network_cuts": {
        "fn": "apply_network_cuts",
        "variants": [
            {
                "min_cut_ms": 100,
                "max_cut_ms": 400,
                "n_cuts": 5
            },
            {
                "min_cut_ms": 200,
                "max_cut_ms": 800,
                "n_cuts": 8
            }
        ]
    },
    
    # Audio Fade Effects (using your apply_gradual_audio_fade)
    "audio_fade": {
        "fn": "apply_gradual_audio_fade",
        "variants": [
            {"min_db": -20, "max_db": 0},
            {"min_db": -30, "max_db": 0}
        ]
    },
    
    # Mic Effects (using your mic functions)
    "mic_rubbing": {
        "fn": "apply_mic_rubbing_effect",
        "variants": [{}]  # No parameters needed
    },
    
    # Mumbling Effect (using your apply_audio_mumbling_effect)
    "mumbling_effect": {
        "fn": "apply_audio_mumbling_effect", 
        "variants": [{}]  # No parameters needed
    }
}

# =============================================================================
# SUBSET CONFIGURATIONS FOR TESTING (Using Your Actual Functions)
# =============================================================================

# Quick test with minimal effects (using functions that exist in your background.py)
QUICK_TEST_EFFECTS = {
    "background_noise_test": {
        "fn": "apply_background_noise",
        "variants": [{"noise_level": -20}]
    },
    "reverb_test": {
        "fn": "apply_reverb",
        "variants": [{"mode": "echo", "echo_delay_ms": 150, "echo_decay": 0.5, "echo_n": 2}]
    }
}

# Medium test with moderate coverage (using functions that exist in your background.py)
MEDIUM_TEST_EFFECTS = {
    "background_noise": {
        "fn": "apply_background_noise",
        "variants": [{"noise_level": -25}, {"noise_level": -15}]
    },
    "volume_fluctuation": {
        "fn": "apply_volume_fluctuation",
        "variants": [{"min_db_change": -10, "max_db_change": 5, "n_fluctuations": 3}]
    },
    "reverb_echo": {
        "fn": "apply_reverb",
        "variants": [{"mode": "echo", "echo_delay_ms": 200, "echo_decay": 0.6, "echo_n": 3}]
    },
    "mic_rubbing": {
        "fn": "apply_mic_rubbing_effect",
        "variants": [{}]
    }
}

# =============================================================================
# ANALYSIS CONFIGURATION
# =============================================================================

# Intent analysis categories that the LLM will identify
EXPECTED_INTENT_CATEGORIES = [
    "command",           # "Turn on the lights"
    "question",          # "What's the weather?"
    "dictation",         # "Send email to John"
    "navigation",        # "Navigate to home"
    "search",           # "Find Italian restaurants"
    "information",      # "What time is it?"
    "communication",    # "Call mom"
    "entertainment",    # "Play music"
    "smart_home",       # "Set thermostat"
    "scheduling"        # "Set reminder"
]

# Error type categories for analysis
ERROR_TYPES = [
    "transcription_degradation",    # Overall quality decrease
    "word_substitution",           # Wrong words used
    "word_deletion",               # Missing words  
    "word_insertion",              # Extra words added
    "phonetic_confusion",          # Similar-sounding words confused
    "entity_confusion",            # Names/numbers/places wrong
    "command_confusion",           # Action/intent changed
    "semantic_drift",              # Meaning changed
    "background_confusion",        # Background noise causing confusion
    "volume_distortion",           # Volume-related distortion effects
    "frequency_distortion",        # Frequency filtering effects
    "reverb_distortion",           # Echo/reverb causing issues
    "partial_loss",                # Some content lost
    "complete_failure",            # Completely unintelligible
    "no_significant_impact"        # Minimal changes
]

# Quality thresholds
WER_THRESHOLD_LOW = 0.2         # Below this = low impact
WER_THRESHOLD_HIGH = 0.5        # Above this = high impact
INTENT_CONFIDENCE_THRESHOLD = 0.7  # Below this = low confidence

# =============================================================================
# VISUALIZATION CONFIGURATION
# =============================================================================

# Chart settings
CHART_STYLE = "default"         # matplotlib style
COLOR_PALETTE = "husl"          # seaborn palette
FIGURE_DPI = 300               # high resolution
FIGURE_SIZE_STANDARD = (12, 6)  # standard chart size
FIGURE_SIZE_LARGE = (14, 8)     # large chart size

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_quick_test_config():
    """Get configuration for quick testing."""
    return {
        "openai_api_key": OPENAI_API_KEY,
        "audio_dir": AUDIO_DIRECTORY,
        "effects_spec": QUICK_TEST_EFFECTS,
        "output_root": f"{OUTPUT_ROOT}_test",
        "results_dir": f"{RESULTS_DIR}_test"
    }

def get_medium_test_config():
    """Get configuration for medium testing."""
    return {
        "openai_api_key": OPENAI_API_KEY,
        "audio_dir": AUDIO_DIRECTORY,
        "effects_spec": MEDIUM_TEST_EFFECTS,
        "output_root": f"{OUTPUT_ROOT}_medium",
        "results_dir": f"{RESULTS_DIR}_medium"
    }

def get_full_config():
    """Get configuration for full evaluation."""
    return {
        "openai_api_key": OPENAI_API_KEY,
        "audio_dir": AUDIO_DIRECTORY,
        "effects_spec": EFFECTS_SPEC,
        "output_root": OUTPUT_ROOT,
        "results_dir": RESULTS_DIR
    }

def get_custom_config(effect_keys: list):
    """Get configuration for specific effects only."""
    custom_effects = {k: v for k, v in EFFECTS_SPEC.items() if k in effect_keys}
    
    return {
        "openai_api_key": OPENAI_API_KEY,
        "audio_dir": AUDIO_DIRECTORY,
        "effects_spec": custom_effects,
        "output_root": f"{OUTPUT_ROOT}_custom",
        "results_dir": f"{RESULTS_DIR}_custom"
    }

def validate_audio_comparison_config():
    """Validate configuration for audio comparison."""
    errors = []
    
    if not OPENAI_API_KEY:
        if not Path(".env").exists():
            errors.append("No .env file found. Please create .env with OPENAI_API_KEY=your-key")
        else:
            errors.append("OPENAI_API_KEY not found in .env file")
    
    if not Path(AUDIO_DIRECTORY).exists():
        errors.append(f"Audio directory not found: {AUDIO_DIRECTORY}")
    
    if not EFFECTS_SPEC:
        errors.append("No effects defined in EFFECTS_SPEC")
    
    # Check for audio files
    audio_dir = Path(AUDIO_DIRECTORY)
    if audio_dir.exists():
        audio_files = []
        for ext in AUDIO_EXTENSIONS:
            audio_files.extend(list(audio_dir.glob(f"*{ext}")))
        
        if not audio_files:
            errors.append(f"No audio files found in {AUDIO_DIRECTORY}")
    
    return errors

def estimate_api_costs():
    """Estimate OpenAI API costs for the evaluation."""
    audio_dir = Path(AUDIO_DIRECTORY)
    
    if not audio_dir.exists():
        return "Cannot estimate - audio directory not found"
    
    # Count audio files
    audio_files = []
    for ext in AUDIO_EXTENSIONS:
        audio_files.extend(list(audio_dir.glob(f"*{ext}")))
    
    # Calculate total variants
    total_variants = sum(len(spec['variants']) for spec in EFFECTS_SPEC.values())
    total_transcriptions = len(audio_files) * (1 + total_variants)  # original + variants
    total_analyses = len(audio_files) * total_variants  # LLM analyses
    
    # Rough cost estimates (as of 2024)
    whisper_cost_per_minute = 0.006  # $0.006 per minute
    gpt4_cost_per_analysis = 0.02    # ~$0.02 per analysis (rough estimate)
    
    # Assume average 30 seconds per audio file
    avg_duration_minutes = 0.5
    
    transcription_cost = total_transcriptions * avg_duration_minutes * whisper_cost_per_minute
    analysis_cost = total_analyses * gpt4_cost_per_analysis
    total_estimated_cost = transcription_cost + analysis_cost
    
    return {
        "audio_files": len(audio_files),
        "total_variants": total_variants,
        "total_transcriptions": total_transcriptions,
        "total_analyses": total_analyses,
        "estimated_transcription_cost": transcription_cost,
        "estimated_analysis_cost": analysis_cost,
        "total_estimated_cost": total_estimated_cost
    }

# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("Audio-to-Audio Comparison Configuration")
    print("=" * 50)
    
    # Validate configuration
    errors = validate_audio_comparison_config()
    if errors:
        print("❌ Configuration errors:")
        for error in errors:
            print(f"   - {error}")
    else:
        print("✅ Configuration looks good!")
    
    # Show configuration summary
    print(f"\n📁 Audio directory: {AUDIO_DIRECTORY}")
    print(f"🔊 Effects defined: {len(EFFECTS_SPEC)}")
    
    # Show .env and API key status
    print(f"🔑 OpenAI API key: {'✅ Set' if OPENAI_API_KEY else '❌ Not found in .env'}")
    if Path(".env").exists():
        print(f"📄 .env file: ✅ Found")
    else:
        print(f"📄 .env file: ❌ Not found (create with OPENAI_API_KEY=your-key)")
    
    # Estimate costs
    cost_estimate = estimate_api_costs()
    if isinstance(cost_estimate, dict):
        print(f"\n💰 Estimated API Costs:")
        print(f"   📁 Audio files: {cost_estimate['audio_files']}")
        print(f"   🎛️  Total variants: {cost_estimate['total_variants']}")
        print(f"   🎤 Transcriptions: {cost_estimate['total_transcriptions']}")
        print(f"   🧠 LLM analyses: {cost_estimate['total_analyses']}")
        print(f"   💵 Estimated cost: ${cost_estimate['total_estimated_cost']:.2f}")
        print(f"      (Transcription: ${cost_estimate['estimated_transcription_cost']:.2f}, Analysis: ${cost_estimate['estimated_analysis_cost']:.2f})")
    else:
        print(f"\n💰 Cost estimate: {cost_estimate}")
    
    print(f"\n🚀 Available configurations:")
    print(f"   - Quick test: {len(QUICK_TEST_EFFECTS)} effects")
    print(f"   - Medium test: {len(MEDIUM_TEST_EFFECTS)} effects") 
    print(f"   - Full evaluation: {len(EFFECTS_SPEC)} effects")