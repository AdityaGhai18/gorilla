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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

from tts_generator_elevenlabs import ElevenLabsTTSGenerator
from tts_generator_cartesia import CartesiaTTSGenerator
from tts_generator_qwen import QwenTTSGenerator
from tts_generator_openai import OpenAITTSGenerator

def get_api_key(varname):
    api_key = os.getenv(varname)
    if not api_key:
        raise EnvironmentError(f"{varname} environment variable not set.")
    return api_key

def main():
    # Configuration - modify these as needed
    PROVIDER = "openai"  # Options: "elevenlabs", "cartesia", "qwen"
    INPUT_FILE = os.path.join(SCRIPT_DIR, "../latest_results/BFCL_v3_live_simple_granular_spoken_final2.json")
    NUM_CASES = 10
    OUTPUT_ROOT = "single_test_output/openai_prompted_text_function_context"  # Specific directory for this single test

    # Provider mapping
    providers = {
        "elevenlabs": (ElevenLabsTTSGenerator, "ELEVENLABS_API_KEY"),
        "cartesia": (CartesiaTTSGenerator, "CARTESIA_API_KEY"),
        "qwen": (QwenTTSGenerator, "QWEN_API_KEY"),
        "openai": (OpenAITTSGenerator, "OPENAI_API_KEY"),
    }

    if PROVIDER not in providers:
        print(f"Invalid provider: {PROVIDER}")
        print(f"Available providers: {list(providers.keys())}")
        sys.exit(1)

    GeneratorClass, env_var = providers[PROVIDER]
    api_key = get_api_key(env_var)

    if PROVIDER == "openai":
        voice = os.getenv("OPENAI_TTS_VOICE", "coral")
        model = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
        instructions = os.getenv("OPENAI_TTS_INSTRUCTIONS", "Speak in a natural, clear tone.")
        # No need to load or pass data anymore
        generator = GeneratorClass(api_key, output_root=OUTPUT_ROOT, model=model, voice=voice, instructions=instructions)
    else:
        generator = GeneratorClass(api_key, output_root=OUTPUT_ROOT)

    generator.run(INPUT_FILE, NUM_CASES)

if __name__ == "__main__":
    main() 