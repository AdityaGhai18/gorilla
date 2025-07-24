import json
import os
from audio_calling.clean_to_speech_text.granular_speech_pipeline import GranularSpeechPipeline, PipelineConfig
import re
from langdetect import detect
import argparse

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

    # # --- Uncomment to filter for a specific test case by ID ---
    # specific_id = "live_simple_161-96-0"
    # data = [case for case in data if case["id"] == specific_id]
    # print(f"Filtered to {len(data)} test case(s) with id {specific_id}.")

    # Filter for cases where at least one turn in the first group has role == 'user'
    def has_user_role(test_case):
        return any(turn.get("role") == "user" for turn in test_case["question"][0])

    user_role_data = [case for case in data if has_user_role(case)]
    print(f"Filtered to {len(user_role_data)} cases with at least one 'user' role. Excluded {len(data) - len(user_role_data)} cases.")

    # Now filter for English test cases using the pipeline
    filtered_data = GranularSpeechPipeline.filter_english_test_cases(pipeline, user_role_data)
    print(f"Filtered to {len(filtered_data)} English test cases. Excluded {len(user_role_data) - len(filtered_data)} cases.")

    print("Processing with granular speech pipeline...")
    processed = pipeline.transform_dataset_for_clean_output(filtered_data, sample_count=sample_count)

    # Remove all quotes from transcripts
    for case in processed:
        for turn in case["question"]:
            for utterance in turn:
                if "transcript" in utterance:
                    utterance["transcript"] = utterance["transcript"].replace('"', '').replace("'", '')

    # Save the transcript JSON
    with open(output_file, 'w') as f:
        json.dump(processed, f, indent=2, ensure_ascii=False)

    # Reload processed data from output_file for TTS
    with open(output_file) as f:
        processed = json.load(f)

    # === TTS batch generation for all test cases ===
    import os
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

    providers = list(generators.items())
    n = len(processed)
    base = n // 3
    remainder = n % 3

    # Assign all OpenAI, then Cartesia, then ElevenLabs, then remainders in order
    provider_indices = []
    for i in range(3):
        provider_indices.extend([i] * base)
    for i in range(remainder):
        provider_indices.append(i)  # Remainders: 0, 1, 2

    def is_english(text):
        try:
            return detect(text) == 'en'
        except:
            return False

    filtered_processed = []
    excluded_ids = []
    for idx, case in enumerate(processed):
        # Find the first user turn with a transcript
        user_turn = next((turn for turn in case["question"][0] if turn.get("role") == "user" and "transcript" in turn), None)
        if not user_turn:
            print(f"Skipping {case['id']} (no user turn with transcript)")
            continue
        transcript = user_turn["transcript"]
        if not is_english(transcript):
            print(f"Excluding {case['id']} (transcript not English)")
            excluded_ids.append(case['id'])
            continue
        provider_idx = provider_indices[idx]
        provider_name, (generator, ext) = providers[provider_idx]
        audio_filename = f"{case['id']}_{provider_name}.{ext}"
        audio_path = os.path.join(audio_dir, audio_filename)
        if provider_name == "openai":
            audio_bytes = generator._generate_audio(transcript, case)
        else:
            audio_bytes = generator._generate_audio(transcript)
        with open(audio_path, "wb") as f:
            f.write(audio_bytes)
        user_turn["audio_path"] = audio_path
        filtered_processed.append(case)

    # Save the updated JSON with audio paths, only for included cases
    with open(output_file, 'w') as f:
        json.dump(filtered_processed, f, indent=2, ensure_ascii=False)

    # Print excluded test case IDs
    if excluded_ids:
        print("\nTest cases excluded due to non-English transcripts:")
        for eid in excluded_ids:
            print(f"  - {eid}")
    else:
        print("\nNo test cases were excluded due to language.") 