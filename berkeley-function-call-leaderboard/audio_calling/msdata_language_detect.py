#!/usr/bin/env python3
import json
import os
import argparse
from typing import Dict, List, Tuple, Any
from collections import Counter

# Import Lingua for language detection
from lingua import Language, LanguageDetectorBuilder

# Define language mapping for better readability
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
    Language.TURKISH: "Turkish",
    Language.VIETNAMESE: "Vietnamese",
    Language.INDONESIAN: "Indonesian",
    Language.MALAY: "Malay",
    Language.ROMANIAN: "Romanian",
    Language.GREEK: "Greek",
}

def load_jsonl(file_path: str) -> List[Dict]:
    """Load data from a JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def detect_language(text: str, detector) -> Tuple[str, float]:
    """Detect the language of a text using Lingua.
    
    Args:
        text: The text to detect the language of
        detector: Lingua language detector instance
        
    Returns:
        Tuple of (language_name, confidence_score)
    """
    if not text or not text.strip():
        return "Unknown", 0.0
    
    try:
        # Get the most likely language and its confidence
        detected = detector.detect_language_of(text)
        if detected:
            language_name = LANGUAGE_MAPPING.get(detected, str(detected))
            confidence = detector.compute_language_confidence(text, detected)
            return language_name, confidence
        return "Unknown", 0.0
    except Exception as e:
        print(f"Error detecting language: {e}")
        return "Error", 0.0

def detect_multiple_languages(text: str, detector) -> List[Tuple[str, float]]:
    """Detect multiple languages in a text using Lingua.
    
    Args:
        text: The text to detect languages in
        detector: Lingua language detector instance
        
    Returns:
        List of tuples (language_name, confidence_score)
    """
    if not text or not text.strip():
        return []
    
    try:
        # Get all detected languages with their probabilities
        detected_languages = detector.detect_multiple_languages_of(text)
        results = []
        
        # Instead of trying to use DetectionResult objects directly,
        # extract the language from each result and then compute confidence
        for result in detected_languages:
            # Access the language property of the DetectionResult
            lang = result.language
            language_name = LANGUAGE_MAPPING.get(lang, str(lang))
            confidence = detector.compute_language_confidence(text, lang)
            results.append((language_name, confidence))
            
        return results
    except Exception as e:
        print(f"Error detecting multiple languages: {e}")
        return []

def analyze_dataset(data_path: str, output_path: str = None, verbose: bool = False):
    """Analyze the language distribution in a dataset.
    
    Args:
        data_path: Path to the JSONL file
        output_path: Path to save the results (optional)
        verbose: Whether to print detailed results
    """
    # Initialize the Lingua language detector with all languages
    languages = [
        Language.ENGLISH, Language.HINDI, Language.URDU, Language.BENGALI, 
        Language.PUNJABI, Language.MARATHI, Language.GUJARATI, Language.TAMIL,
        Language.TELUGU, Language.FRENCH, Language.GERMAN, Language.SPANISH, 
        Language.PORTUGUESE, Language.ITALIAN, Language.DUTCH, Language.POLISH, 
        Language.DANISH, Language.SWEDISH, Language.FINNISH, Language.HUNGARIAN, 
        Language.ICELANDIC, Language.CZECH, Language.SLOVAK, Language.RUSSIAN, 
        Language.UKRAINIAN, Language.CHINESE, Language.JAPANESE, Language.KOREAN, 
        Language.ARABIC, Language.PERSIAN, Language.THAI, Language.TURKISH, Language.VIETNAMESE, 
        Language.INDONESIAN, Language.MALAY, Language.ROMANIAN, Language.GREEK
    ]
    
    detector = LanguageDetectorBuilder.from_languages(*languages).build()
    
    # Load the dataset
    print(f"Loading data from {data_path}...")
    data = load_jsonl(data_path)
    print(f"Loaded {len(data)} entries")
    
    # Analyze each entry
    language_counts = Counter()
    results = []
    
    print("Detecting languages...")
    for i, entry in enumerate(data):
        if i % 10 == 0 and i > 0:
            print(f"Processed {i}/{len(data)} entries")
            
        question = entry.get("question", "")
        
        # Detect the primary language
        language, confidence = detect_language(question, detector)
        
        # Detect multiple languages if present
        multiple_langs = detect_multiple_languages(question, detector)
        
        # Store the results
        language_counts[language] += 1
        
        result = {
            "id": entry.get("id", i),
            "question": question,
            "primary_language": language,
            "confidence": confidence,
            "detected_languages": multiple_langs
        }
        results.append(result)
        
        if verbose:
            print(f"Entry {i}:")
            print(f"  Question: {question}")
            print(f"  Primary language: {language} (confidence: {confidence:.4f})")
            print(f"  All detected languages: {multiple_langs}")
            print()
    
    # Print summary
    print("\nLanguage Distribution:")
    for lang, count in language_counts.most_common():
        print(f"  {lang}: {count} ({count/len(data)*100:.2f}%)")
    
    # Save results if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                "summary": {
                    "total_entries": len(data),
                    "language_distribution": {
                        lang: {
                            "count": count,
                            "percentage": count/len(data)*100
                        } for lang, count in language_counts.items()
                    }
                },
                "entries": results
            }, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Detect languages in MS MARCO dataset using Lingua")
    parser.add_argument('--data_path', type=str, required=True, 
                        help='Path to the JSONL file (e.g., massive-10k-20samples-train-converted-all-langs-fixed/hi-IN/input.jsonl)')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Path to save the results (optional)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed results for each entry')
    
    args = parser.parse_args()
    
    analyze_dataset(args.data_path, args.output_path, args.verbose)

if __name__ == "__main__":
    main()