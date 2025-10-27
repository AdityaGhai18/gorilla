"""
LLM Judge for Audio Query Evaluation with Controlled Noise and Disruptions

This module provides a comprehensive LLM-based judge for evaluating audio transcription
quality under various noise conditions and correlating noise types with error patterns.
"""

import json
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import difflib
from pathlib import Path


class ErrorType(Enum):
    """Categories of errors that can occur in audio transcription."""
    TRANSCRIPTION_ERROR = "transcription_error"
    WORD_SUBSTITUTION = "word_substitution"
    WORD_DELETION = "word_deletion"
    WORD_INSERTION = "word_insertion"
    PHONETIC_CONFUSION = "phonetic_confusion"
    BACKGROUND_CONFUSION = "background_confusion"
    VOLUME_DISTORTION = "volume_distortion"
    FREQUENCY_DISTORTION = "frequency_distortion"
    REVERB_DISTORTION = "reverb_distortion"
    ENTITY_MISHEARING = "entity_mishearing"
    SEMANTIC_DRIFT = "semantic_drift"
    PARTIAL_LOSS = "partial_loss"
    COMPLETE_FAILURE = "complete_failure"
    NO_ERROR = "no_error"


class NoiseType(Enum):
    """Types of noise and disruptions that can be applied to audio."""
    BACKGROUND_NOISE = "background_noise"
    WHITE_NOISE = "white_noise"
    REVERB = "reverb"
    ECHO = "echo"
    COMPRESSION = "compression"
    DISTORTION = "distortion"
    VOLUME_CHANGE = "volume_change"
    FREQUENCY_FILTER = "frequency_filter"
    SPEED_CHANGE = "speed_change"
    PITCH_SHIFT = "pitch_shift"
    CLIPPING = "clipping"
    DROPOUT = "dropout"


@dataclass
class JudgmentResult:
    """Result structure for LLM judge evaluation."""
    error_types: List[str]
    confidence_score: float
    transcript: Optional[str]
    wer_score: Optional[float]
    semantic_similarity: Optional[float]
    detailed_analysis: Dict[str, Any]
    reasoning: str
    noise_correlation: Dict[str, float]


class AudioQualityJudge:
    """
    LLM-based judge for evaluating audio transcription quality and
    correlating noise types with specific error patterns.
    """
    
    def __init__(self, llm_client=None, transcribe_fn=None):
        """
        Initialize the judge with optional LLM client and transcription function.
        
        Args:
            llm_client: Client for calling LLM (e.g., OpenAI, Anthropic)
            transcribe_fn: Function to transcribe audio files
        """
        self.llm_client = llm_client
        self.transcribe_fn = transcribe_fn
        self.error_patterns = self._initialize_error_patterns()
        
    def _initialize_error_patterns(self) -> Dict[str, Dict]:
        """Initialize patterns that correlate noise types with expected errors."""
        return {
            "background_noise": {
                "likely_errors": ["background_confusion", "word_substitution", "partial_loss"],
                "indicators": ["similar sounding words", "environment confusion", "masking"]
            },
            "reverb": {
                "likely_errors": ["reverb_distortion", "word_deletion", "phonetic_confusion"],
                "indicators": ["echo effects", "temporal smearing", "clarity loss"]
            },
            "compression": {
                "likely_errors": ["frequency_distortion", "phonetic_confusion", "entity_mishearing"],
                "indicators": ["frequency artifacts", "dynamic range loss", "clarity degradation"]
            },
            "volume_change": {
                "likely_errors": ["volume_distortion", "partial_loss", "complete_failure"],
                "indicators": ["amplitude issues", "signal-to-noise problems", "audibility"]
            },
            "distortion": {
                "likely_errors": ["transcription_error", "phonetic_confusion", "entity_mishearing"],
                "indicators": ["harmonic distortion", "signal clipping", "quality degradation"]
            }
        }

    def calculate_wer(self, reference: str, hypothesis: str) -> float:
        """Calculate Word Error Rate between reference and hypothesis."""
        if not reference or not hypothesis:
            return 1.0
            
        ref_words = reference.lower().split()
        hyp_words = hypothesis.lower().split()
        
        if not ref_words:
            return 1.0 if hyp_words else 0.0
            
        # Use SequenceMatcher for alignment
        matcher = difflib.SequenceMatcher(None, ref_words, hyp_words)
        operations = matcher.get_opcodes()
        
        substitutions = deletions = insertions = 0
        
        for op, i1, i2, j1, j2 in operations:
            if op == 'replace':
                substitutions += max(i2 - i1, j2 - j1)
            elif op == 'delete':
                deletions += i2 - i1
            elif op == 'insert':
                insertions += j2 - j1
                
        total_errors = substitutions + deletions + insertions
        return total_errors / len(ref_words)

    def analyze_error_patterns(self, reference: str, hypothesis: str) -> List[str]:
        """Analyze specific error patterns between reference and hypothesis."""
        errors = []
        
        if not hypothesis:
            return ["complete_failure"]
            
        ref_words = reference.lower().split()
        hyp_words = hypothesis.lower().split()
        
        # Check for common error patterns
        matcher = difflib.SequenceMatcher(None, ref_words, hyp_words)
        operations = matcher.get_opcodes()
        
        for op, i1, i2, j1, j2 in operations:
            if op == 'replace':
                errors.append("word_substitution")
                # Check for phonetic confusion
                ref_segment = ' '.join(ref_words[i1:i2])
                hyp_segment = ' '.join(hyp_words[j1:j2])
                if self._are_phonetically_similar(ref_segment, hyp_segment):
                    errors.append("phonetic_confusion")
            elif op == 'delete':
                errors.append("word_deletion")
            elif op == 'insert':
                errors.append("word_insertion")
        
        # Check for entity recognition errors
        if self._has_entity_errors(reference, hypothesis):
            errors.append("entity_mishearing")
            
        # Check for semantic drift
        if self._has_semantic_drift(reference, hypothesis):
            errors.append("semantic_drift")
            
        return list(set(errors)) if errors else ["no_error"]

    def _are_phonetically_similar(self, word1: str, word2: str) -> bool:
        """Check if two words/phrases are phonetically similar."""
        # Simple phonetic similarity check based on edit distance
        if not word1 or not word2:
            return False
            
        # Remove spaces and convert to lowercase
        w1 = re.sub(r'[^a-z]', '', word1.lower())
        w2 = re.sub(r'[^a-z]', '', word2.lower())
        
        if not w1 or not w2:
            return False
            
        # Calculate normalized edit distance
        max_len = max(len(w1), len(w2))
        edit_dist = difflib.SequenceMatcher(None, w1, w2).ratio()
        
        return edit_dist > 0.6  # Threshold for phonetic similarity

    def _has_entity_errors(self, reference: str, hypothesis: str) -> bool:
        """Check for entity recognition errors (names, places, etc.)."""
        # Simple pattern matching for common entities
        entity_patterns = [
            r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # Names
            r'\b[A-Z][a-z]+\s+(Street|Avenue|Road|Boulevard)\b',  # Addresses
            r'\b[A-Z][a-z]+\s+(City|State|Country)\b',  # Places
        ]
        
        ref_entities = set()
        hyp_entities = set()
        
        for pattern in entity_patterns:
            ref_entities.update(re.findall(pattern, reference))
            hyp_entities.update(re.findall(pattern, hypothesis))
            
        # Check if entities are missing or different
        return len(ref_entities.symmetric_difference(hyp_entities)) > 0

    def _has_semantic_drift(self, reference: str, hypothesis: str) -> bool:
        """Check for semantic drift using simple keyword analysis."""
        # Extract key content words (excluding common stop words)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
        
        ref_words = set(word.lower() for word in reference.split() if word.lower() not in stop_words)
        hyp_words = set(word.lower() for word in hypothesis.split() if word.lower() not in stop_words)
        
        if not ref_words:
            return False
            
        # Calculate semantic overlap
        overlap = len(ref_words.intersection(hyp_words))
        semantic_similarity = overlap / len(ref_words) if ref_words else 0
        
        return semantic_similarity < 0.5  # Threshold for semantic drift

    def correlate_noise_with_errors(self, noise_type: str, error_types: List[str]) -> Dict[str, float]:
        """Correlate specific noise types with observed error patterns."""
        correlations = {}
        
        if noise_type in self.error_patterns:
            expected_errors = self.error_patterns[noise_type]["likely_errors"]
            
            for error in error_types:
                if error in expected_errors:
                    correlations[error] = 1.0  # Strong correlation
                else:
                    correlations[error] = 0.3  # Weak correlation
                    
        return correlations

    def generate_llm_prompt(self, original_query: str, transcript: str, 
                          noise_metadata: Dict[str, Any]) -> str:
        """Generate a prompt for LLM analysis of transcription quality."""
        prompt = f"""
You are an expert audio transcription evaluator. Analyze the quality of this audio transcription and identify specific error types caused by noise/distortion.

ORIGINAL QUERY: "{original_query}"
TRANSCRIBED TEXT: "{transcript}"

NOISE/DISTORTION APPLIED:
- Effect Type: {noise_metadata.get('effect_key', 'unknown')}
- Parameters: {json.dumps(noise_metadata.get('params', {}), indent=2)}

Please analyze this transcription and provide:

1. ERROR CLASSIFICATION: Choose all applicable error types from:
   - transcription_error: General transcription mistakes
   - word_substitution: Wrong words used instead of correct ones
   - word_deletion: Words missing from transcription
   - word_insertion: Extra words added to transcription
   - phonetic_confusion: Similar-sounding words confused
   - background_confusion: Background noise causing confusion
   - volume_distortion: Volume-related distortion effects
   - frequency_distortion: Frequency filtering effects
   - reverb_distortion: Echo/reverb causing issues
   - entity_mishearing: Names, places, etc. misheard
   - semantic_drift: Meaning significantly changed
   - partial_loss: Some content lost but some preserved
   - complete_failure: Transcription completely failed
   - no_error: No significant errors detected

2. CONFIDENCE SCORE: Rate your confidence in this analysis (0.0-1.0)

3. NOISE CORRELATION: How likely is each error type to be caused by the specific noise/distortion applied?

4. DETAILED REASONING: Explain your analysis, focusing on:
   - How the specific noise type likely contributed to errors
   - Patterns you observe in the mistakes
   - Relationship between noise parameters and error severity

Respond in this JSON format:
{{
    "error_types": ["list", "of", "error", "types"],
    "confidence_score": 0.95,
    "noise_correlation": {{"error_type": correlation_score, ...}},
    "reasoning": "Detailed explanation of your analysis",
    "word_error_rate_estimate": 0.15,
    "semantic_preservation": 0.85
}}
"""
        return prompt

    def judge_transcription(self, variant_audio_path: str, original_query: str, 
                          metadata: Dict[str, Any]) -> JudgmentResult:
        """
        Main judging function that evaluates transcription quality.
        
        Args:
            variant_audio_path: Path to the audio file with applied noise
            original_query: Original text that should have been transcribed
            metadata: Information about the noise/effect applied
            
        Returns:
            JudgmentResult with comprehensive analysis
        """
        # Step 1: Transcribe the audio
        transcript = None
        if self.transcribe_fn:
            try:
                transcript = self.transcribe_fn(variant_audio_path)
            except Exception as e:
                transcript = f"[TRANSCRIPTION_FAILED: {str(e)}]"
        else:
            transcript = "[NO_TRANSCRIPTION_FUNCTION_PROVIDED]"
        
        # Step 2: Calculate basic metrics
        wer_score = None
        if transcript and not transcript.startswith("["):
            wer_score = self.calculate_wer(original_query, transcript)
        
        # Step 3: Analyze error patterns
        error_types = []
        if transcript and not transcript.startswith("["):
            error_types = self.analyze_error_patterns(original_query, transcript)
        else:
            error_types = ["complete_failure"]
        
        # Step 4: Correlate with noise type
        noise_type = metadata.get('effect_key', 'unknown')
        noise_correlations = self.correlate_noise_with_errors(noise_type, error_types)
        
        # Step 5: LLM analysis (if available)
        llm_analysis = {}
        confidence_score = 0.8  # Default confidence
        reasoning = "Analysis based on pattern matching and WER calculation"
        
        if self.llm_client and transcript and not transcript.startswith("["):
            try:
                prompt = self.generate_llm_prompt(original_query, transcript, metadata)
                llm_response = self._call_llm(prompt)
                llm_analysis = self._parse_llm_response(llm_response)
                
                # Update results with LLM insights
                if llm_analysis:
                    error_types = llm_analysis.get("error_types", error_types)
                    confidence_score = llm_analysis.get("confidence_score", confidence_score)
                    reasoning = llm_analysis.get("reasoning", reasoning)
                    noise_correlations.update(llm_analysis.get("noise_correlation", {}))
                    
            except Exception as e:
                llm_analysis["error"] = f"LLM analysis failed: {str(e)}"
        
        # Step 6: Calculate semantic similarity
        semantic_similarity = None
        if transcript and not transcript.startswith("[") and original_query:
            semantic_similarity = self._calculate_semantic_similarity(original_query, transcript)
        
        # Step 7: Compile detailed analysis
        detailed_analysis = {
            "noise_metadata": metadata,
            "transcript_length": len(transcript) if transcript else 0,
            "original_length": len(original_query),
            "llm_analysis": llm_analysis,
            "error_pattern_analysis": {
                "detected_patterns": error_types,
                "noise_specific_patterns": self.error_patterns.get(noise_type, {})
            }
        }
        
        return JudgmentResult(
            error_types=error_types,
            confidence_score=confidence_score,
            transcript=transcript,
            wer_score=wer_score,
            semantic_similarity=semantic_similarity,
            detailed_analysis=detailed_analysis,
            reasoning=reasoning,
            noise_correlation=noise_correlations
        )

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM with the given prompt."""
        if not self.llm_client:
            raise ValueError("No LLM client provided")
        
        # This would depend on your specific LLM client
        # Example for OpenAI:
        # response = self.llm_client.chat.completions.create(
        #     model="gpt-4",
        #     messages=[{"role": "user", "content": prompt}],
        #     temperature=0.1
        # )
        # return response.choices[0].message.content
        
        # Placeholder implementation
        return '{"error_types": ["transcription_error"], "confidence_score": 0.8, "reasoning": "LLM analysis"}'

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured format."""
        try:
            # Try to extract JSON from the response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
        
        return {"error": "Failed to parse LLM response", "raw_response": response}

    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts."""
        # Simple implementation using word overlap
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
            
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0


def create_llm_judge_function(llm_client=None, transcribe_fn=None) -> callable:
    """
    Factory function to create an LLM judge function compatible with the pipeline.
    
    Args:
        llm_client: Client for LLM calls (optional)
        transcribe_fn: Function to transcribe audio files
        
    Returns:
        Function with signature: (variant_audio_path, original_query_text, metadata) -> dict
    """
    judge = AudioQualityJudge(llm_client=llm_client, transcribe_fn=transcribe_fn)
    
    def llm_judge_function(variant_audio_path: str, original_query_text: str, 
                          metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        LLM judge function compatible with the pipeline.
        
        Returns:
            Dictionary with keys expected by the pipeline:
            - error_types: List of error type strings
            - transcript: Transcribed text (optional)
            - notes: Additional analysis notes (optional)
            - confidence_score: Confidence in the analysis
            - wer_score: Word Error Rate
            - detailed_analysis: Comprehensive analysis results
        """
        try:
            result = judge.judge_transcription(variant_audio_path, original_query_text, metadata)
            
            return {
                "error_types": result.error_types,
                "transcript": result.transcript,
                "notes": result.reasoning,
                "confidence_score": result.confidence_score,
                "wer_score": result.wer_score,
                "semantic_similarity": result.semantic_similarity,
                "noise_correlation": result.noise_correlation,
                "detailed_analysis": result.detailed_analysis
            }
            
        except Exception as e:
            return {
                "error_types": ["judge_failed"],
                "transcript": None,
                "notes": f"Judge failed with error: {str(e)}",
                "confidence_score": 0.0,
                "error": str(e)
            }
    
    return llm_judge_function


# Example usage and configuration
def setup_example_judge():
    """Example setup for the LLM judge with common configurations."""
    
    # Example transcription function (replace with your actual implementation)
    def example_transcribe_fn(audio_path: str) -> str:
        """
        Example transcription function.
        Replace this with your actual ASR implementation.
        """
        # This would typically call Whisper, Google Speech-to-Text, etc.
        return f"[MOCK_TRANSCRIPTION_OF_{Path(audio_path).name}]"
    
    # Example LLM client setup (replace with your actual implementation)
    class ExampleLLMClient:
        """Example LLM client. Replace with your actual LLM client."""
        def generate(self, prompt: str) -> str:
            return '{"error_types": ["transcription_error"], "confidence_score": 0.8}'
    
    llm_client = ExampleLLMClient()
    
    # Create the judge function
    judge_fn = create_llm_judge_function(
        llm_client=llm_client,
        transcribe_fn=example_transcribe_fn
    )
    
    return judge_fn


# Configuration for common noise types and their expected error correlations
NOISE_ERROR_CORRELATIONS = {
    "white_noise": {
        "high_correlation": ["background_confusion", "phonetic_confusion"],
        "medium_correlation": ["word_substitution", "partial_loss"],
        "low_correlation": ["entity_mishearing"]
    },
    "reverb": {
        "high_correlation": ["reverb_distortion", "word_deletion"],
        "medium_correlation": ["phonetic_confusion", "semantic_drift"],
        "low_correlation": ["word_insertion"]
    },
    "compression": {
        "high_correlation": ["frequency_distortion", "entity_mishearing"],
        "medium_correlation": ["phonetic_confusion", "word_substitution"],
        "low_correlation": ["word_deletion"]
    },
    "volume_low": {
        "high_correlation": ["partial_loss", "complete_failure"],
        "medium_correlation": ["word_deletion", "volume_distortion"],
        "low_correlation": ["word_insertion"]
    },
    "distortion": {
        "high_correlation": ["transcription_error", "phonetic_confusion"],
        "medium_correlation": ["entity_mishearing", "word_substitution"],
        "low_correlation": ["semantic_drift"]
    }
}


if __name__ == "__main__":
    # Example of how to use the judge
    judge_fn = setup_example_judge()
    
    # Test the judge
    result = judge_fn(
        variant_audio_path="/path/to/noisy_audio.wav",
        original_query_text="What is the weather like today?",
        metadata={
            "effect_key": "white_noise",
            "params": {"noise_level": 0.3},
            "variant_idx": 0
        }
    )
    
    print("Judge Result:")
    print(json.dumps(result, indent=2))