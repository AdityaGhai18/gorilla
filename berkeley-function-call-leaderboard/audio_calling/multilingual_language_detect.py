
#!/usr/bin/env python3
"""
BFCL Irrelevance Language Detection Script

This script uses the Lingua library to detect languages in the BFCL live irrelevance dataset.
It analyzes each question entry and provides language detection statistics.
"""

import argparse
import json
from collections import Counter
from typing import Dict, List, Tuple, Any, Optional

from lingua import Language, LanguageDetectorBuilder

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

def load_json(file_path: str) -> List[Dict[str, Any]]:
    """Load JSON data from a file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        List of dictionaries containing the JSON data
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        # Try loading as JSONL if JSON fails
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return []

def extract_question_text(entry: Dict[str, Any]) -> str:
    """Extract question text from a BFCL irrelevance entry.
    
    Args:
        entry: Dictionary containing the entry data
        
    Returns:
        The extracted question text
    """
    try:
        # BFCL irrelevance format has nested question structure
        question_data = entry.get("question", [])
        if question_data and isinstance(question_data, list) and question_data[0]:
            for message in question_data[0]:
                if message.get("role") == "user" and "content" in message:
                    return message["content"]
        return ""
    except Exception as e:
        print(f"Error extracting question text: {e}")
        return ""

def detect_language(text: str, detector) -> Tuple[str, float]:
    """Detect the primary language of a text using Lingua.
    
    Args:
        text: The text to detect the language of
        detector: Lingua language detector instance
        
    Returns:
        Tuple of (language_name, confidence_score)
    """
    if not text or not text.strip():
        return ("Unknown", 0.0)
    
    try:
        detected_language = detector.detect_language_of(text)
        if detected_language:
            language_name = LANGUAGE_MAPPING.get(detected_language, str(detected_language))
            confidence = detector.compute_language_confidence(text, detected_language)
            return (language_name, confidence)
        return ("Unknown", 0.0)
    except Exception as e:
        print(f"Error detecting language: {e}")
        return ("Unknown", 0.0)

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
        
        # Extract the language from each result and then compute confidence
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
    """Analyze the language distribution in a BFCL irrelevance dataset.
    
    Args:
        data_path: Path to the BFCL irrelevance JSON file
        output_path: Optional path to save the results
        verbose: Whether to print verbose output
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
        Language.ARABIC, Language.PERSIAN, Language.THAI, Language.TURKISH
    ]
    
    detector = LanguageDetectorBuilder.from_languages(*languages).build()
    
    # Load the dataset
    print(f"Loading data from {data_path}...")
    data = load_json(data_path)
    print(f"Loaded {len(data)} entries")
    
    # Analyze each entry
    language_counts = Counter()
    results = []
    
    print("Detecting languages...")
    for i, entry in enumerate(data):
        if i % 10 == 0 and i > 0:
            print(f"Processed {i}/{len(data)} entries")
            
        question = extract_question_text(entry)
        
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
            print(f"  ID: {entry.get('id', i)}")
            print(f"  Question: {question}")
            print(f"  Primary language: {language} (confidence: {confidence:.4f})")
            print(f"  All detected languages: {multiple_langs}")
            print()
    
    # Add a final progress message
    print(f"Processed all {len(data)} entries")
    
    # Print summary
    print("\nLanguage Distribution:")
    for lang, count in language_counts.most_common():
        percentage = (count / len(data)) * 100
        print(f"  {lang}: {count} ({percentage:.2f}%)")
    
    # Save results if output path is provided
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                "total_entries": len(data),
                "language_distribution": {lang: count for lang, count in language_counts.items()},
                "detailed_results": results
            }, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Detect languages in BFCL irrelevance dataset")
    parser.add_argument("--data_path", required=True, help="Path to the BFCL irrelevance JSON file")
    parser.add_argument("--output_path", help="Path to save the results")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")
    
    args = parser.parse_args()
    analyze_dataset(args.data_path, args.output_path, args.verbose)

if __name__ == "__main__":
    main()
