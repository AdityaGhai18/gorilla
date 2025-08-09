import json
import os
import re
import argparse
import random
import time
from langdetect import detect
from typing import List, Tuple

from audio_calling.clean_to_speech_text.granular_speech_pipeline import (
    GranularSpeechPipeline,
    PipelineConfig,
)
from audio_calling.TTS.scripts.tts_generator_openai import OpenAITTSGenerator
from audio_calling.TTS.scripts.tts_generator_cartesia import CartesiaTTSGenerator
from audio_calling.TTS.scripts.tts_generator_elevenlabs import ElevenLabsTTSGenerator


# Primary: langdetect
# Secondary: Unicode-based script check
# Special handling: short phrases (< 7 words)
# Fallback: Unicode script check

_NON_LATIN_RE = re.compile(r"[^\u0000-\u024F\u1E00-\u1EFF]")
_WORD_RE = re.compile(r"\w+", re.UNICODE)


def contains_non_latin(text: str) -> bool:
    if not text:
        return False
    return bool(_NON_LATIN_RE.search(text))


def word_count(text: str) -> int:
    if not text:
        return 0
    return len(_WORD_RE.findall(text))


def ascii_only(text: str) -> bool:
    return all(ord(c) < 128 for c in text)


def is_non_english(text: str) -> bool:
    """Return True if text is non-English, using hybrid detection.

    Strategy:
    - Try langdetect first.
    - For short phrases (<7 words) predicted as English, verify there are no non-Latin characters; if found, treat as non-English.
    - For short phrases (<7 words) predicted as non-English but Latin-only, treat as English to avoid false positives.
    - Numeric-heavy short ASCII phrases are treated as English.
    - If langdetect predicts non-English, trust that (except short Latin-only override above).
    - On detection failure, fall back to non-Latin script signal.
    """
    if not text or not text.strip():
        return False
    wc = word_count(text)
    ascii_txt = ascii_only(text)
    try:
        lang = detect(text)
        if lang == "en":
            # If very short and contains non-Latin, consider non-English
            if wc < 7 and contains_non_latin(text):
                return True
            return False
        else:
            # Avoid false positives for very short Latin-only phrases
            if wc < 7 and not contains_non_latin(text):
                # Numeric-heavy ASCII short commands -> English
                digits = sum(ch.isdigit() for ch in text)
                letters = sum(ch.isalpha() for ch in text)
                if ascii_txt and digits >= letters:
                    return False
                return False
            return True
    except Exception:
        # Fallback: if it contains non-Latin characters, consider non-English
        return contains_non_latin(text)


def load_bfcl_data(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            return json.load(f)
        else:
            return [json.loads(line) for line in f if line.strip()]


def save_json(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def collect_non_english_user_turns(data) -> List[Tuple[int, int, int]]:
    """Return list of (case_idx, turn_idx, utter_idx) where the user query is non-English.

    Detection prefers raw 'content' (original user text); falls back to 'transcript'.
    """
    indices = []
    for case_idx, case in enumerate(data):
        for turn_idx, turn in enumerate(case.get("question", [])):
            for utter_idx, utter in enumerate(turn):
                if utter.get("role") != "user":
                    continue
                detection_text = utter.get("content") or utter.get("transcript") or ""
                if is_non_english(detection_text):
                    indices.append((case_idx, turn_idx, utter_idx))
    return indices


def main():
    parser = argparse.ArgumentParser(description="Multilingual speechifying pipeline: process only non-English queries.")
    parser.add_argument('--data_path', type=str, required=True, help='Path to input data JSON (array or JSONL)')
    parser.add_argument('--output_file', type=str, required=True, help='Path to output JSON')
    parser.add_argument('--sample_count', type=str, default=None, help='Number of turns to process (default: all). Use None for all.')
    parser.add_argument('--audio_dir', type=str, default=None, help='Directory to save audio files')
    args = parser.parse_args()

    data_path = args.data_path
    output_file = args.output_file

    if args.sample_count is None or args.sample_count == 'None':
        sample_count = None
    else:
        sample_count = int(args.sample_count)

    # Audio output directory
    if args.audio_dir:
        audio_dir = args.audio_dir
    else:
        audio_dir = "audio_calling/clean_to_speech_text/final_results/audio/multilingual"
    os.makedirs(audio_dir, exist_ok=True)

    # Configure the granular speech pipeline (same as English pipeline)
    config = PipelineConfig(
        max_features=6,
        confidence_threshold=0.5,
        temperature=0.8,
        max_retries=3,
        retry_delay=1.0,
        asr=False,
    )
    pipeline = GranularSpeechPipeline(config)

    print(f"Loading data from: {data_path}")
    data = load_bfcl_data(data_path)
    print(f"Loaded {len(data)} test cases.")

    # Identify non-English user turns
    non_en_turns = collect_non_english_user_turns(data)
    print(f"Found {len(non_en_turns)} non-English user turns.")

    # Optionally subsample
    if sample_count is not None and sample_count < len(non_en_turns):
        non_en_turns = random.sample(non_en_turns, sample_count)
        print(f"Sampled down to {len(non_en_turns)} non-English turns.")

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

    # Process only the non-English user turns
    for idx, (case_idx, turn_idx, utter_idx) in enumerate(non_en_turns):
        utter = data[case_idx]["question"][turn_idx][utter_idx]

        # Apply the same disfluency-adding method as English via transform_text
        transcript_result = pipeline.transform_text(utter.get("content", ""))
        transcript = transcript_result.get("final", "")
        transcript = transcript.replace('"', '').replace("'", '')
        utter["transcript"] = transcript

        # Generate audio (Round-robin provider assignment for now; TODO: fix to one provider which works best)
        provider_name = provider_names[idx % n_providers]
        generator, ext = generators[provider_name]
        audio_filename = f"{data[case_idx]['id']}_turn{turn_idx}_{provider_name}.{ext}"
        audio_path = os.path.join(audio_dir, audio_filename)

        if provider_name == "cartesia":
            while True:
                try:
                    audio_bytes = generator._generate_audio(transcript)
                    break
                except Exception as e:
                    print(f"[Cartesia ERROR] {data[case_idx]['id']} turn {turn_idx}: {e}. Retrying in 30 seconds...")
                    time.sleep(30)
        elif provider_name == "openai":
            audio_bytes = generator._generate_audio(transcript, data[case_idx])
        else:
            audio_bytes = generator._generate_audio(transcript)

        with open(audio_path, "wb") as f:
            f.write(audio_bytes)
        utter["audio_path"] = audio_path

        print(f"Processed multilingual turn -> case={data[case_idx]['id']} turn={turn_idx} provider={provider_name} path={audio_path}")

        save_json(data, output_file)

    # Final save
    save_json(data, output_file)
    print(f"\nDone. Wrote multilingual transcripts + audio to {output_file}")


if __name__ == "__main__":
    main()
