"""
Speech Noise Pipeline for BFCL Research

This module implements a cascading pipeline to transform clean BFCL test data
into realistic spoken language variants, simulating the distribution shift
between typed text and ASR-transcribed conversational speech.

Two pipelines are implemented:
1. Spoken Language Only: Captures true linguistic shift (speech-like)
2. ASR-Style Simulation: Includes ASR errors on top of spoken-style input
"""

import json
import os
import random
import time
from typing import Dict, List, Optional, Tuple
import openai
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-4" #any model can be used wasnt sure so just went with 4 for now

@dataclass
class PipelineConfig:

    """Configuration for the speech noise pipeline."""

    include_asr_noise: bool = False
    max_retries: int = 3
    retry_delay: float = 1.0
    temperature: float = 0.7

class SpeechNoisePipeline:
    """Main class for transforming clean text into spoken language variants."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
        
    def conversationalize(self, text: str) -> str:
        """
        Stage 1: Make text sound like natural spoken language.
        Adds disfluencies, spontaneity, informality.
        """
        prompt = f"""
        Transform this clean, written text into natural spoken language as if someone is saying it out loud.
        
        Add:
        - Filler words (uh, um, like, you know, I mean)
        - Informal contractions (gonna, wanna, gotta)
        - Natural disfluencies and hesitations
        - Conversational tone and spontaneity
        - Contextual omissions and references
        
        Keep the core meaning and intent exactly the same.
        
        Input: "{text}"
        
        Output (spoken version):
        """
        
        return self._call_openai(prompt)
    
    def degrade_syntax(self, text: str) -> str:
        """
        Stage 2: Degrade syntax and structure.
        Removes grammar, creates fragments, reorders words.
        """
        prompt = f"""
        Transform this spoken text to have degraded syntax and structure, as if it came from ASR or casual speech.
        
        Apply:
        - Drop articles (a, an, the)
        - Remove prepositions where possible
        - Create sentence fragments
        - Use telegraphic speech
        - Simplify complex structures
        - Reorder words naturally
        
        Keep the essential information and function call intent.
        
        Input: "{text}"
        
        Output (syntax degraded):
        """
        
        return self._call_openai(prompt)
    
    def semantic_simplify(self, text: str) -> str:
        """
        Stage 3: Semantic paraphrasing and simplification.
        Replace words with similar ones, use vague references.
        """
        prompt = f"""
        Simplify and paraphrase this text semantically, as if someone is using simpler words or vague references.
        
        Apply:
        - Replace complex words with simpler synonyms
        - Use vague references (that thing, this stuff)
        - Simplify technical terms
        - Use more common vocabulary
        - Maintain the same function call intent
        
        Input: "{text}"
        
        Output (semantically simplified):
        """
        
        return self._call_openai(prompt)
    
    def add_domain_specific_noise(self, text: str) -> str:
        """
        Stage 4: Domain-specific alterations for function calling.
        Add spelling errors, abbreviations, slot confusion.
        """
        prompt = f"""
        Add domain-specific noise that might occur in function calling scenarios.
        
        Apply:
        - Common spelling errors (H for hibernate, etc.)
        - Abbreviations and shortcuts
        - Slot confusion (wrong parameter names)
        - Misspoken entities
        - Function name variations
        
        Keep the function call recognizable but add realistic errors.
        
        Input: "{text}"
        
        Output (domain-specific noise added):
        """
        
        return self._call_openai(prompt)
    
    def inject_asr_errors(self, text: str) -> str:
        """
        Stage 5: Simulate ASR errors (substitutions, deletions, insertions).
        Only used in the ASR pipeline.
        """
        prompt = f"""
        Simulate ASR (Automatic Speech Recognition) errors that commonly occur in speech-to-text systems.
        
        Apply:
        - Word substitutions (homophones, similar sounding words)
        - Word deletions (missing articles, prepositions)
        - Word insertions (extra filler words)
        - Punctuation errors
        - Capitalization issues
        
        Make it sound like it came from a real ASR system with typical error rates.
        
        Input: "{text}"
        
        Output (ASR errors added):
        """
        
        return self._call_openai(prompt)
    
    def _call_openai(self, prompt: str) -> str:
        """Make API call to OpenAI with structured output."""
        response = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.config.temperature,
            max_tokens=500,
            response_format={"type": "text"}  # Structured output for cleaner responses
        )
        return response.choices[0].message.content.strip()
    
    def transform_text(self, text: str) -> Dict[str, str]:
        """
        Apply the full cascading transformation pipeline.
        
        Returns a dictionary with intermediate and final results.
        """
        results = {
            "original": text,
            "conversational": "",
            "syntax_degraded": "",
            "semantic_simplified": "",
            "domain_noise": "",
            "final": ""
        }
        
        print(f"Original: {text}")
        
        # Stage 1: Conversationalization
        results["conversational"] = self.conversationalize(text)
        print(f"Stage 1 (Conversational): {results['conversational']}")
        
        # Stage 2: Syntax degradation
        results["syntax_degraded"] = self.degrade_syntax(results["conversational"])
        print(f"Stage 2 (Syntax): {results['syntax_degraded']}")
        
        # Stage 3: Semantic simplification
        results["semantic_simplified"] = self.semantic_simplify(results["syntax_degraded"])
        print(f"Stage 3 (Semantic): {results['semantic_simplified']}")
        
        # Stage 4: Domain-specific noise
        results["domain_noise"] = self.add_domain_specific_noise(results["semantic_simplified"])
        print(f"Stage 4 (Domain): {results['domain_noise']}")
        
        # Stage 5: ASR errors (optional)
        if self.config.include_asr_noise:
            results["final"] = self.inject_asr_errors(results["domain_noise"])
            print(f"Stage 5 (ASR): {results['final']}")
        else:
            results["final"] = results["domain_noise"]
            print(f"Final (No ASR): {results['final']}")
        
        return results

def load_bfcl_data(file_path: str) -> List[Dict]:
    """Load BFCL test data from JSON file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def save_transformed_data(data: List[Dict], output_path: str):
    """Save transformed data to another JSON file for review."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')

def main():

    """
    Main function to run the speech noise pipeline and cascading stages. 
    Easy to remove stages from here if needed.
    
    """
    
    # Configuration
    config = PipelineConfig(
        include_asr_noise=False,  # Set to True for ASR pipeline
        max_retries=3,
        retry_delay=1.0,
        temperature=0.7
    )
    
    # Initialize pipeline
    pipeline = SpeechNoisePipeline(config)
    
    # Load BFCL data
    data_path = "data/BFCL_v3_live_simple.json"
    bfcl_data = load_bfcl_data(data_path)
    
    print(f"Loaded {len(bfcl_data)} test cases from {data_path}")
    
    # transform a subset for testing (first 3 examples to begin with can be extended)
    test_subset = bfcl_data[:3]
    transformed_data = []
    
    for i, test_case in enumerate(test_subset):
        print(f"\n{'='*50}")
        print(f"Processing test case {i+1}/{len(test_subset)}")
        print(f"ID: {test_case['id']}")
        
        # Extract the user question from the bfcl data file
        user_content = test_case['question'][0][0]['content']
        
        transformation_results = pipeline.transform_text(user_content)
        
        transformed_case = {
            "id": f"{test_case['id']}_spoken",
            "original_id": test_case['id'],
            "original_question": user_content,
            "transformed_question": transformation_results["final"],
            "transformation_stages": transformation_results,
            "function": test_case['function']  # Keep the same function definition
        }
        
        transformed_data.append(transformed_case)
        
        print(f"Transformation complete for {test_case['id']}")
    
    # results
    output_path = "data/BFCL_v3_live_simple_spoken.json"
    save_transformed_data(transformed_data, output_path)
    print(f"\nSaved {len(transformed_data)} transformed test cases to {output_path}")
    
    # summary
    print(f"\n{'='*50}")
    print("TRANSFORMATION SUMMARY")
    print(f"{'='*50}")
    for i, case in enumerate(transformed_data):
        print(f"\n{i+1}. {case['original_id']}")
        print(f"   Original: {case['original_question']}")
        print(f"   Transformed: {case['transformed_question']}")

if __name__ == "__main__":
    main() 