import json
import os
import re
import argparse
import random
import time
from typing import List, Tuple, Dict, Any, Optional

# Import Lingua for improved language detection
from lingua import Language, LanguageDetectorBuilder

from audio_calling.clean_to_speech_text.granular_speech_pipeline import (
    GranularSpeechPipeline,
    PipelineConfig,
)
from audio_calling.TTS.scripts.tts_generator_openai import OpenAITTSGenerator
from audio_calling.TTS.scripts.tts_generator_cartesia import CartesiaTTSGenerator
from audio_calling.TTS.scripts.tts_generator_elevenlabs import ElevenLabsTTSGenerator

# Mapping of Lingua language objects to human-readable names
LANGUAGE_MAPPING = {
    Language.ENGLISH: "English",
    Language.HINDI: "Hindi",
    Language.URDU: "Urdu",
    Language.BENGALI: "Bengali",
    Language.PUNJABI: "Punjabi",
    Language.MARATHI: "Marathi",
    Language.GUJARATI: "Gujarati",
    Language.TAMIL: "Tamil",
    Language.TELUGU: "Telugu",
    Language.FRENCH: "French",
    Language.GERMAN: "German",
    Language.SPANISH: "Spanish",
    Language.PORTUGUESE: "Portuguese",
    Language.ITALIAN: "Italian",
    Language.DUTCH: "Dutch",
    Language.POLISH: "Polish",
    Language.DANISH: "Danish",
    Language.SWEDISH: "Swedish",
    Language.FINNISH: "Finnish",
    Language.HUNGARIAN: "Hungarian",
    Language.ICELANDIC: "Icelandic",
    Language.CZECH: "Czech",
    Language.SLOVAK: "Slovak",
    Language.RUSSIAN: "Russian",
    Language.UKRAINIAN: "Ukrainian",
    Language.CHINESE: "Chinese",
    Language.JAPANESE: "Japanese",
    Language.KOREAN: "Korean",
    Language.ARABIC: "Arabic",
    Language.PERSIAN: "Persian",
    Language.THAI: "Thai",
    Language.TURKISH: "Turkish"
}

# Initialize the Lingua language detector with all languages
SUPPORTED_LANGUAGES = [
    Language.ENGLISH, Language.HINDI, Language.URDU, Language.BENGALI, 
    Language.PUNJABI, Language.MARATHI, Language.GUJARATI, Language.TAMIL,
    Language.TELUGU, Language.FRENCH, Language.GERMAN, Language.SPANISH, 
    Language.PORTUGUESE, Language.ITALIAN, Language.DUTCH, Language.POLISH, 
    Language.DANISH, Language.SWEDISH, Language.FINNISH, Language.HUNGARIAN, 
    Language.ICELANDIC, Language.CZECH, Language.SLOVAK, Language.RUSSIAN, 
    Language.UKRAINIAN, Language.CHINESE, Language.JAPANESE, Language.KOREAN, 
    Language.ARABIC, Language.PERSIAN, Language.THAI, Language.TURKISH
]

# Initialize the detector once
DETECTOR = LanguageDetectorBuilder.from_languages(*SUPPORTED_LANGUAGES).build()

def detect_language(text: str) -> Tuple[str, float]:
    """Detect the primary language of a text using Lingua.
    
    Args:
        text: The text to detect the language of
        
    Returns:
        Tuple of (language_name, confidence_score)
    """
    if not text or not text.strip():
        return ("English", 0.0)  # Default to English for empty text
    
    try:
        detected_language = DETECTOR.detect_language_of(text)
        if detected_language:
            language_name = LANGUAGE_MAPPING.get(detected_language, str(detected_language))
            confidence = DETECTOR.compute_language_confidence(text, detected_language)
            return (language_name, confidence)
        return ("English", 0.0)  # Default to English if detection fails
    except Exception as e:
        print(f"Error detecting language: {e}")
        return ("English", 0.0)  # Default to English on error

def detect_multiple_languages(text: str) -> List[Tuple[str, float]]:
    """Detect multiple languages in a text using Lingua.
    
    Args:
        text: The text to detect languages in
        
    Returns:
        List of tuples (language_name, confidence_score)
    """
    if not text or not text.strip():
        return []
    
    try:
        # Get all detected languages with their probabilities
        detected_languages = DETECTOR.detect_multiple_languages_of(text)
        results = []
        
        # Extract the language from each result and then compute confidence
        for result in detected_languages:
            # Access the language property of the DetectionResult
            lang = result.language
            language_name = LANGUAGE_MAPPING.get(lang, str(lang))
            confidence = DETECTOR.compute_language_confidence(text, lang)
            results.append((language_name, confidence))
            
        return results
    except Exception as e:
        print(f"Error detecting multiple languages: {e}")
        return []

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

def extract_question_text(entry: Dict[str, Any]) -> str:
    """Extract question text from an entry.
    
    Args:
        entry: Dictionary containing the entry data
        
    Returns:
        The extracted question text
    """
    try:
        # Handle MS MARCO format (direct question string)
        if isinstance(entry.get("question", ""), str):
            return entry.get("question", "")
            
        # Handle BFCL format (nested structure)
        question_data = entry.get("question", [])
        if question_data and isinstance(question_data, list) and question_data[0]:
            for message in question_data[0]:
                if message.get("role") == "user" and "content" in message:
                    return message["content"]
        return ""
    except Exception as e:
        print(f"Error extracting question text: {e}")
        return ""

def collect_all_user_turns(data) -> List[Tuple[int, int, int, str, str]]:
    """Return list of (case_idx, turn_idx, utter_idx, text, detected_language) for all user queries.
    
    Handles both BFCL format (nested structure) and MS MARCO format (direct question string).
    """
    entries = []
    for case_idx, case in enumerate(data):
        # Handle MS MARCO format (direct question string)
        if isinstance(case.get("question", ""), str):
            text = case.get("question", "")
            language, confidence = detect_language(text)
            entries.append((case_idx, 0, 0, text, language))  # Use (0,0) as placeholder indices for turn/utter
            continue
            
        # Handle original BFCL format (nested structure)
        for turn_idx, turn in enumerate(case.get("question", [])):
            # Handle case where turn is directly a list of utterances
            if isinstance(turn, list):
                for utter_idx, utter in enumerate(turn):
                    if isinstance(utter, dict) and utter.get("role") != "user":
                        continue
                    text = utter.get("content") or utter.get("transcript") or ""
                    language, confidence = detect_language(text)
                    entries.append((case_idx, turn_idx, utter_idx, text, language))
            # Handle case where turn might be a dict with utterances
            elif isinstance(turn, dict):
                text = turn.get("content") or turn.get("transcript") or ""
                language, confidence = detect_language(text)
                entries.append((case_idx, turn_idx, 0, text, language))
    return entries

def main():
    parser = argparse.ArgumentParser(description="Multilingual speechifying pipeline with Lingua language detection")
    parser.add_argument('--data_path', type=str, required=True, help='Path to input data JSON (array or JSONL)')
    parser.add_argument('--output_file', type=str, required=True, help='Path to output JSON')
    parser.add_argument('--sample_count', type=str, default=None, help='Number of turns to process (default: all). Use None for all.')
    parser.add_argument('--audio_dir', type=str, default=None, help='Directory to save audio files')
    parser.add_argument('--all_languages', action='store_true', help='Process all languages, not just non-English')
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

    # Identify all user turns with their detected languages
    all_turns = collect_all_user_turns(data)
    print(f"Found {len(all_turns)} total user turns.")
    
    # Count languages
    language_counts = {}
    for _, _, _, _, lang in all_turns:
        language_counts[lang] = language_counts.get(lang, 0) + 1
    
    print("Language distribution:")
    for lang, count in sorted(language_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {lang}: {count} ({count/len(all_turns)*100:.1f}%)")
    
    # Filter turns if not processing all languages
    if not args.all_languages:
        all_turns = [(case_idx, turn_idx, utter_idx, text, lang) 
                     for case_idx, turn_idx, utter_idx, text, lang in all_turns 
                     if lang != "English"]
        print(f"Filtered to {len(all_turns)} non-English user turns.")

    # Optionally subsample
    if sample_count is not None and sample_count < len(all_turns):
        all_turns = random.sample(all_turns, sample_count)
        print(f"Sampled down to {len(all_turns)} turns.")

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

    # Process the selected user turns
    for idx, (case_idx, turn_idx, utter_idx, text, detected_language) in enumerate(all_turns):
        print(f"Processing query {idx+1}/{len(all_turns)}: case={case_idx}, turn={turn_idx}, utter={utter_idx}, language={detected_language}")
        case = data[case_idx]
        
        # Handle MS MARCO format
        if isinstance(case.get("question", ""), str):
            transcript = case["question"]
            audio_path = os.path.join(audio_dir, f"{case['id']}_audio.wav")
        # Handle original BFCL format
        else:
            turn = case["question"][turn_idx]
            if isinstance(turn, list):
                utter = turn[utter_idx]
                transcript = utter.get("content") or utter.get("transcript")
                utter_id = utter.get("id", f"{utter_idx}")
                audio_path = os.path.join(audio_dir, f"{case['id']}_turn{turn_idx}_utter{utter_id}_audio.wav")
            else:
                transcript = turn.get("content") or turn.get("transcript")
                audio_path = os.path.join(audio_dir, f"{case['id']}_turn{turn_idx}_audio.wav")
        
        # Apply disfluency-adding method via transform_text
        transcript_result = pipeline.transform_text(transcript)
        transcript = transcript_result.get("final", "")
        transcript = transcript.replace('"', '').replace("'", '')
        
        # Store the detected language in the case
        if isinstance(case.get("question", ""), str):
            case["detected_language"] = detected_language
            case["question"] = transcript
        else:
            if isinstance(case["question"][turn_idx], list):
                case["question"][turn_idx][utter_idx]["detected_language"] = detected_language
            else:
                case["question"][turn_idx]["detected_language"] = detected_language
        
        # Generate audio (Round-robin provider assignment for now)
        provider_name = provider_names[idx % n_providers]
        generator, ext = generators[provider_name]
        
        # Update audio path with language info
        audio_path = audio_path.replace(".wav", f"_{detected_language.lower()}.{ext}")
        
        if provider_name == "cartesia":
            while True:
                try:
                    # Pass language code to Cartesia TTS
                    audio_bytes = generator._generate_audio(transcript, language=detected_language)
                    break
                except Exception as e:
                    print(f"[Cartesia ERROR] {data[case_idx]['id']} turn {turn_idx}: {e}. Retrying in 30 seconds...")
                    time.sleep(30)
        elif provider_name == "openai":
            # OpenAI TTS should handle multilingual text automatically
            audio_bytes = generator._generate_audio(transcript, case, language=detected_language)
        else:
            # ElevenLabs uses multilingual model by default
            audio_bytes = generator._generate_audio(transcript, language=detected_language)

        with open(audio_path, "wb") as f:
            f.write(audio_bytes)
            
        # Store the audio path in the case
        if isinstance(case.get("question", ""), str):
            case["audio_path"] = audio_path
        else:
            if isinstance(case["question"][turn_idx], list):
                case["question"][turn_idx][utter_idx]["audio_path"] = audio_path
            else:
                case["question"][turn_idx]["audio_path"] = audio_path

        print(f"Processed multilingual turn -> case={data[case_idx]['id']} turn={turn_idx} language={detected_language} provider={provider_name} path={audio_path}")

        # Save after each turn to preserve progress
        save_json(data, output_file)

    # Final save
    save_json(data, output_file)
    print(f"\nDone. Wrote multilingual transcripts + audio to {output_file}")


if __name__ == "__main__":
    main()
