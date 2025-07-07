#!/usr/bin/env python3
"""
Simple TTS runner script.
Choose your provider, input file, and number of cases to process.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from tts_generator_elevenlabs import ElevenLabsTTSGenerator
from tts_generator_cartesia import CartesiaTTSGenerator
from tts_generator_qwen import QwenTTSGenerator

def get_api_key(varname):
    api_key = os.getenv(varname)
    if not api_key:
        raise EnvironmentError(f"{varname} environment variable not set.")
    return api_key

def main():
    # Configuration - modify these as needed
    PROVIDER = "elevenlabs"  # Options: "elevenlabs", "cartesia", "qwen"
    INPUT_FILE = "latest_results/BFCL_v3_live_simple_granular_spoken_final2.json"
    NUM_CASES = 1
    OUTPUT_ROOT = "single_test_output"  # Specific directory for this single test

    # Provider mapping
    providers = {
        "elevenlabs": (ElevenLabsTTSGenerator, "ELEVENLABS_API_KEY"),
        "cartesia": (CartesiaTTSGenerator, "CARTESIA_API_KEY"),
        "qwen": (QwenTTSGenerator, "QWEN_API_KEY"),
    }

    if PROVIDER not in providers:
        print(f"Invalid provider: {PROVIDER}")
        print(f"Available providers: {list(providers.keys())}")
        sys.exit(1)

    GeneratorClass, env_var = providers[PROVIDER]
    api_key = get_api_key(env_var)

    # Run the generator
    generator = GeneratorClass(api_key, output_root=OUTPUT_ROOT)
    generator.run(INPUT_FILE, NUM_CASES)

if __name__ == "__main__":
    main() 