import json
import os
from audio_calling.clean_to_speech_text.granular_speech_pipeline import GranularSpeechPipeline, PipelineConfig
import re
from langdetect import detect
import argparse
import time

def load_bfcl_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            return json.load(f)
        else:
            return [json.loads(line) for line in f if line.strip()]

def save_transformed_data(data, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to input data JSON')
    parser.add_argument('--output_file', type=str, required=True, help='Path to output JSON')
    parser.add_argument('--sample_count', type=str, default=None, help='Number of samples to process (default: all). Use None or omit for all.')
    parser.add_argument('--audio_dir', type=str, default=None, help='Directory to save audio files (required)')
    args = parser.parse_args()

    data_path = args.data_path
    output_file = args.output_file
    # Convert sample_count to int or None
    if args.sample_count is None or args.sample_count == 'None':
        sample_count = None
    else:
        sample_count = int(args.sample_count)

    # Use user-specified audio_dir, or default to previous hardcoded value
    if args.audio_dir:
        audio_dir = args.audio_dir
    else:
        audio_dir = "audio_calling/clean_to_speech_text/final_results/audio/missing"
    import os
    os.makedirs(audio_dir, exist_ok=True)

    # === Clean text to speech-like text ===
    config = PipelineConfig(
        max_features=6,
        confidence_threshold=0.5,
        temperature=0.8,
        max_retries=3,
        retry_delay=1.0,
        asr=False
    )
    pipeline = GranularSpeechPipeline(config)

    print(f"Loading data from: {data_path}")
    data = load_bfcl_data(data_path)
    print(f"Loaded {len(data)} test cases.")

    # Flatten all turns for provider assignment
    all_turns = []  # (case_idx, turn_idx, turn_dict)
    for case_idx, case in enumerate(data):
        for turn_idx, turn in enumerate(case["question"]):
            for utter_idx, utter in enumerate(turn):
                if utter.get("role") == "user":
                    all_turns.append((case_idx, turn_idx, utter_idx, utter))

    print(f"Total user turns to process: {len(all_turns)}")

    # Optionally subsample
    if sample_count is not None and sample_count < len(all_turns):
        import random
        all_turns = random.sample(all_turns, sample_count)
        print(f"Sampled down to {len(all_turns)} turns.")

    # === TTS setup ===
    from audio_calling.TTS.scripts.tts_generator_openai import OpenAITTSGenerator
    from audio_calling.TTS.scripts.tts_generator_cartesia import CartesiaTTSGenerator
    from audio_calling.TTS.scripts.tts_generator_elevenlabs import ElevenLabsTTSGenerator

    tts_providers = [
        ("openai", OpenAITTSGenerator, "OPENAI_API_KEY"),
        ("cartesia", CartesiaTTSGenerator, "CARTESIA_API_KEY"),
        ("elevenlabs", ElevenLabsTTSGenerator, "ELEVENLABS_API_KEY"),
    ]

    generators = {}
    for name, cls, env_var in tts_providers:
        api_key = os.environ.get(env_var)
        if not api_key:
            raise RuntimeError(f"Missing API key for {name}: set {env_var}")
        ext = "mp3" if name == "cartesia" else "wav"
        generators[name] = (cls(api_key, output_root=audio_dir), ext)

    provider_names = list(generators.keys())
    n_providers = len(provider_names)

    # Assign providers round-robin to all turns
    for idx, (case_idx, turn_idx, utter_idx, utter) in enumerate(all_turns):
        provider_name = provider_names[idx % n_providers]
        generator, ext = generators[provider_name]
        transcript_result = pipeline.transform_text(utter["content"])
        transcript = transcript_result["final"]
        # Remove all quotes from transcript
        transcript = transcript.replace('"', '').replace("'", '')
        utter["transcript"] = transcript
        audio_filename = f"{data[case_idx]['id']}_turn{turn_idx}_{provider_name}.{ext}"
        audio_path = os.path.join(audio_dir, audio_filename)
        # Generate audio with Cartesia retry logic
        if provider_name == "cartesia":
            while True:
                try:
                    audio_bytes = generator._generate_audio(transcript)
                    break
                except Exception as e:
                    print(f"[Cartesia ERROR] Failed to generate audio for {data[case_idx]['id']} turn {turn_idx}: {e}. Retrying in 30 seconds...")
                    time.sleep(30)
        elif provider_name == "openai":
            audio_bytes = generator._generate_audio(transcript, data[case_idx])
        else:
            audio_bytes = generator._generate_audio(transcript)
        with open(audio_path, "wb") as f:
            f.write(audio_bytes)
        utter["audio_path"] = audio_path
        print(f"Generated audio for {data[case_idx]['id']} turn {turn_idx} ({provider_name}) -> {audio_path}")
        # Save the updated JSON with transcripts and audio paths after each turn
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    # Save the updated JSON with transcripts and audio paths
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Output written to {output_file}") 