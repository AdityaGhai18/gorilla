"""
Manual Speech Pipeline - Manual Selection of Speech Features and ASR Errors

This script provides a manual interface for applying speech noise features and ASR errors
to text, allowing you to choose exactly which features to apply and in what order.

SPEECH NOISE FEATURES (26 total):
    These features simulate natural speech patterns and disfluencies:

    1. disfluencies: Filler words (um, uh, like), hesitations, natural pauses
    2. repetitions: Repeat words or phrases
    3. self_corrections: Correct oneself with an actual correction, e.g., 'the file... no, the folder'
    4. false_starts: Start to say something, then restart or change direction
    5. thinking_aloud: Express thinking or searching for words, e.g., 'let me see'
    6. backchanneling: Conversational markers (yeah, so, right, okay)
    7. emotional_markers: Emotion or attitude (ugh, wow, oh, right, seriously)
    8. restarts_repairs: Restart or repair a sentence, e.g., 'what I mean is...'
    9. ellipsis_proforms: Use ellipsis or pro-forms (do it, get it, that thing)
    10. spelling_noise: Spell out names/terms that might be misunderstood
    11. numbers_noise: Say numbers/addresses as a real person would
    12. contractions: Use wanna, gonna, lemme
    13. casual_pronouns: ya, em, imma
    14. slang_terms: grab, check out, look up
    15. symbol_pronunciation: Say symbols out loud (slash, dash, at)
    16. article_dropping: Drop the, a, an where natural
    17. preposition_dropping: Drop in, on, at, for where natural
    18. subject_dropping: Drop subject pronouns where natural
    19. fragment_sentences: Break into shorter fragments
    20. word_reordering: Slightly reorder words naturally
    21. vague_references: Use that thing, the stuff, some info
    22. approximate_quantifiers: like 10 minutes, around 5 files
    23. simplified_verbs: get instead of retrieve, check instead of verify
    24. polite_hedges: maybe, could you, if possible, would you
    25. confidence_markers: I think, probably, should be
    26. contextual_references: the one we talked about, that repo, that day, that file

ASR ERROR TYPES (5 total):
    These simulate common Automatic Speech Recognition errors:

    1. word_substitution: Replace words with homophones, similar-sounding words, or common ASR confusions
    2. word_deletion: Omit short function words, especially articles, prepositions, or pronouns
    3. word_insertion: Add filler words or repeated short words
    4. punctuation_error: Remove or misplace punctuation marks
    5. capitalization_error: Lowercase proper nouns or start sentences without capitalization

USAGE:
    # Create a manual pipeline
    pipeline = ManualSpeechPipeline()
    
    # Apply specific speech features
    result = pipeline.apply_speech_features(
        text="Can you retrieve the user details?",
        features=["contractions", "numbers_noise", "casual_pronouns"],
        intensities=["moderate", "light", "heavy"]
    )
    
    # Apply specific ASR errors
    asr_result = pipeline.apply_asr_errors(
        text=result["transformed"],
        error_types=["word_substitution", "punctuation_error"]
    )


    
"""

import json
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from granular_speech_pipe import GranularSpeechPipeline, ASRErrors, PipelineConfig, FeatureSelection


@dataclass
class ManualConfig:
    """Configuration for manual speech pipeline."""
    default_intensity: str = "moderate"
    track_stages: bool = True
    verbose: bool = True


class ManualSpeechPipeline:
    """
    Manual interface for applying speech features and ASR errors.
    
    This class allows you to manually select which speech noise features and ASR errors
    to apply to text, giving you full control over the transformation process.
    """
    
    def __init__(self, config: Optional[ManualConfig] = None):
        """
        Initialize the manual pipeline.
        
        Args:
            config: Configuration object for the pipeline
        """
        self.config = config or ManualConfig()
        self.speech_pipeline = GranularSpeechPipeline(PipelineConfig())
        self.asr_pipeline = ASRErrors()
        
        # Available speech features
        self.speech_features = [
            "disfluencies", "repetitions", "self_corrections", "false_starts",
            "thinking_aloud", "backchanneling", "emotional_markers", "restarts_repairs",
            "ellipsis_proforms", "spelling_noise", "numbers_noise", "contractions",
            "casual_pronouns", "slang_terms", "symbol_pronunciation", "article_dropping",
            "preposition_dropping", "subject_dropping", "fragment_sentences",
            "word_reordering", "vague_references", "approximate_quantifiers",
            "simplified_verbs", "polite_hedges", "confidence_markers", "contextual_references"
        ]
        
        # Available ASR error types
        self.asr_errors = [
            "word_substitution", "word_deletion", "word_insertion",
            "punctuation_error", "capitalization_error"
        ]

    
    def validate_features(self, features: List[str]) -> List[str]:
        """
        Validate and return only valid speech feature names.
        
        Args:
            features: List of feature names to validate
            
        Returns:
            List of valid feature names
        """
        valid_features = []
        for feature in features:
            if feature in self.speech_features:
                valid_features.append(feature)
            elif self.config.verbose:
                print(f"Warning: Unknown speech feature '{feature}' - skipping")
        return valid_features
    
    def validate_asr_errors(self, error_types: List[str]) -> List[str]:
        """
        Validate and return only valid ASR error types.
        
        Args:
            error_types: List of ASR error types to validate
            
        Returns:
            List of valid ASR error types
        """
        valid_errors = []
        for error in error_types:
            if error in self.asr_errors:
                valid_errors.append(error)
            elif self.config.verbose:
                print(f"Warning: Unknown ASR error type '{error}' - skipping")
        return valid_errors
    
    def apply_speech_features(
        self, 
        text: str, 
        features: List[str]) -> Dict[str, Union[str, Dict[str, str]]]:
        """
        Apply specific speech features to the input text, always using 'moderate' intensity.
        
        Args:
            text: Input text to transform
            features: List of speech feature names to apply
        Returns:
            Dictionary with original text, transformed text, and stages (if enabled)
        """

        valid_features = self.validate_features(features)
        if not valid_features:
            return {"original": text, "transformed": text, "stages": {}}
        
        # Always use 'moderate' intensity
        intensity = "moderate"
        
        # Apply features
        current_text = text
        stages = {"original": text} if self.config.track_stages else {}
        
        if self.config.verbose:
            print(f"Applying {len(valid_features)} speech features to: '{text}'")
        
        for i, feature in enumerate(valid_features):
            if self.config.verbose:
                print(f"  {i+1}. Applying {feature} (intensity: {intensity})")
            
            # Get the feature function
            feature_func = getattr(self.speech_pipeline, f"apply_{feature}", None)
            if feature_func:
                current_text = feature_func(current_text, intensity)
                if self.config.track_stages:
                    stages[f"stage_{i+1}_{feature}"] = current_text
                
                if self.config.verbose:
                    print(f"     Result: '{current_text}'")
            else:
                if self.config.verbose:
                    print(f"     Warning: No function found for feature '{feature}'")
        
        result = {
            "original": text,
            "transformed": current_text,
            "features_applied": valid_features
        }
        
        if self.config.track_stages:
            result["stages"] = stages
        
        return result
    
    def apply_asr_errors(
        self, 
        text: str, 
        error_types: List[str]
    ) -> Dict[str, Union[str, Dict[str, str]]]:
        """
        Apply specific ASR errors to the input text.
        
        Args:
            text: Input text to transform
            error_types: List of ASR error types to apply
            
        Returns:
            Dictionary with original text, transformed text, and stages (if enabled)
        """
        # Validate error types
        valid_errors = self.validate_asr_errors(error_types)
        if not valid_errors:
            return {"original": text, "transformed": text, "stages": {}}
        
        # Apply errors
        current_text = text
        stages = {"original": text} if self.config.track_stages else {}
        
        if self.config.verbose:
            print(f"Applying {len(valid_errors)} ASR errors to: '{text}'")
        
        for i, error_type in enumerate(valid_errors):
            if self.config.verbose:
                print(f"  {i+1}. Applying {error_type}")
            
            # Get the error function
            error_func = getattr(self.asr_pipeline, f"apply_{error_type}", None)
            if error_func:
                current_text = error_func(current_text)
                if self.config.track_stages:
                    stages[f"asr_{i+1}_{error_type}"] = current_text
                
                if self.config.verbose:
                    print(f"     Result: '{current_text}'")
            else:
                if self.config.verbose:
                    print(f"     Warning: No function found for ASR error '{error_type}'")
        
        result = {
            "original": text,
            "transformed": current_text,
            "asr_errors_applied": valid_errors
        }
        
        if self.config.track_stages:
            result["stages"] = stages
        
        return result 

def demo():
    """
    Demonstration of the manual pipeline usage. just apply the 2 functions. 
    Apply speech features and apply asr errors
    """
    # init pipeline
    pipeline = ManualSpeechPipeline(ManualConfig(verbose=False, track_stages=True))
    # Load BFCL data
    data_path = "../data/BFCL_v3_live_simple.json"
    bfcl_data = load_bfcl_data(data_path)
    print(f"Loaded {len(bfcl_data)} test cases from {data_path}")
    num_cases = 5  # Change this to process a different number of test cases
    # For each test case, apply speech features and ASR errors
    for i, test_case in enumerate(bfcl_data[:num_cases]):
        print(f"\n{'='*60}")
        print(f"Test case {i+1}/{min(num_cases, len(bfcl_data))}")
        # Extract user content (assuming same structure as granular_speech_pipe)
        user_content = test_case['question'][0][0]['content']
        print(f"Original text: {user_content}")
        # Apply speech features
        result1 = pipeline.apply_speech_features(
            text=user_content,
            features=["contractions", "numbers_noise", "casual_pronouns"] # specify features here
        )
        print(f"Speech-like result: {result1['transformed']}")
        # Apply ASR errors to the speech-like output
        result2 = pipeline.apply_asr_errors(
            text=result1["transformed"],
            error_types=["word_substitution", "punctuation_error"] # specify asr errors here
        )
        print(f"Final_result_ASR: {result2['transformed']}")

def load_bfcl_data(file_path: str) -> list:
    """
    Load BFCL test cases from a JSON or JSONL file.
    Supports both array and line-delimited formats.
    """
    with open(file_path, "r") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            return json.load(f)
        else:
            return [json.loads(line) for line in f if line.strip()]

if __name__ == "__main__":
    demo() 