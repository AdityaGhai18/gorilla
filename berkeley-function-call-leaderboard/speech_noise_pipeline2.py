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
        prompt = f'''
        Rewrite the input as if someone is speaking casually and naturally to a voice assistant or a friend. Keep the meaning intact, use a relaxed tone, and make it sound spoken — but **avoid overusing filler words**.

        Do:
        - Add **1–2** light disfluencies or informal cues: "uh", "I guess", "like", "you know"
        - Use contractions and colloquial phrasing: "wanna", "gonna", "lemme", "could you maybe"
        - Slightly rephrase to sound natural out loud but maintain key details and do not change meaning

        Don't:
        - Use more than **2** filler/disfluency words
        - Ramble, pad with vague language, or repeat yourself
        - Break sentence flow too much

        Input: "{text}"

        Output (spoken version):
'''
        
        return self._call_openai(prompt)
    
    def degrade_syntax(self, text: str) -> str:
        """
        Stage 2: Degrade syntax and structure.
        Removes grammar, creates fragments, reorders words.
        """
        prompt = f'''
        Rewrite the sentence as if someone is speaking quickly or casually, maybe while multitasking — so their syntax is slightly messy or incomplete. Think real human speech: short cuts, dropped small words, but still understandable and **clearly pointing to what they want**.

        Do:
        - Drop or shorten minor connecting words (like "the", "to", "that") **where natural**
        - Use light sentence fragments, mild reordering
        - Keep **all key entities, names, and references**
        - Sound like natural clipped speech — not broken text

        Don't:
        - Drop the subject of the request
        - Omit important nouns like what the person wants or where
        - Mangle or confuse the core meaning
        - Make it sound like a bad transcript

        Input: "{text}"

        Output (lightly degraded syntax):
'''
        
        return self._call_openai(prompt)
    
    def semantic_simplify(self, text: str) -> str:
        """
        Stage 3: Semantic paraphrasing and simplification.
        Replace words with similar ones, use vague references.
        """
        prompt = f'''
        Rewrite the sentence as if someone is describing it casually or lazily — using slightly vague, imprecise language. Imagine they understand the request but are explaining it informally or off-the-cuff.

        Do:
        - Use casual or fuzzy terms ("some info", "that repo", "like 10 minutes")
        - Replace rigid structures with looser phrases ("retrieve the details" → "get the info")
        - Preserve all **key identifiers** (IDs, names, repo paths, colors, numbers) — **but you can reframe them lightly**
        - Sound like a human simplifying, not distorting

        Don't:
        - Change or omit numbers, IDs, or key request parts
        - Turn clear attributes into open-ended guesses ("something about black?")
        - Drift into speculation or uncertainty unless the original implies it

        Input: "{text}"

        Output (semantically simplified):
'''
        return self._call_openai(prompt)
    
    def add_domain_specific_noise(self, text: str) -> str:
        """
        Stage 4: Domain-specific alterations for function calling.
        Add spelling errors, abbreviations, slot confusion.
        """
        prompt = f'''
        Rewrite the sentence with **very light domain-style paraphrasing** based on your understanding of the world, like how a person might casually talk about a task aloud — but only if it's completely clear and safe to do so.

        Keep it grounded and accurate. Keep all repo names, file paths, addresses, numbers, and named entities **exactly the same**.

        Do:
        - Loosen phrasing slightly (e.g., "grab a comfy Uber" instead of "request a Comfort Uber")
        - Refer to tasks casually if meaning is obvious ("check that repo" instead of "retrieve the star history")

        Don't:
        - Change, shorten, or replace names, addresses, or technical terms
        - Add any filler words, hesitations, or informal noises
        - Invent new references or make cultural jokes
        - Introduce ambiguity or shift meaning in any way

        Input: "{text}"

        Output (minimal domain-style paraphrase):
'''


        
        return self._call_openai(prompt)
    
    def inject_asr_errors(self, text: str) -> str:
        """
        Stage 5: Simulate ASR errors (substitutions, deletions, insertions).
        Only used in the ASR pipeline.
        """
        prompt = f"""
        You are simulating the output of a real Automatic Speech Recognition (ASR) system, as described in the paper "Are LLMs Robust for Spoken Dialogues?".
        
        Apply realistic ASR errors as statistically observed in real ASR transcripts:
        - **Word substitutions** (~7%): Replace words with homophones, similar-sounding words, or common ASR confusions (e.g., "to" ↔ "too", "there" ↔ "their", "know" ↔ "no").
        - **Word deletions** (~4%): Omit short function words, especially articles (a, an, the), prepositions (in, on, at, for), and sometimes pronouns.
        - **Word insertions** (~2%): Add filler words or repeated short words (e.g., "um", "uh", "like", "you know").
        - **Punctuation errors**: Remove or misplace punctuation marks, especially periods and commas.
        - **Capitalization errors**: Lowercase proper nouns or start sentences without capitalization.
        - **Slot confusion** (if applicable): Occasionally confuse similar slot/entity names in the domain.
        - **Misspellings or phonetic errors**: For rare or unusual words, introduce plausible misspellings or phonetic substitutions.
        
        The overall error rate should be moderate and reflect a typical ASR system (not excessive, but noticeable).
        Do NOT change the core intent or meaning of the utterance.
        Output should look like a realistic ASR transcript, not a parody or exaggerated error case.
        
        Input (spoken-style text): "{text}"
        
        Output (ASR-transcribed with realistic errors):
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
        
        # Stage 5: ASR errors only if second pipeline is used we will set this to true in the config
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
    """Save transformed data to another JSON file for review and characterising shift later."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')

def main():

    """
    Main function to run the speech noise pipeline and cascading stages. 
    Easy to remove stages from here if needed.
    
    """
    
    config = PipelineConfig(
        include_asr_noise=True,  # Set to True for ASR pipeline
        max_retries=3,
        retry_delay=1.0,
        temperature=0.7
    )
    
    #init pipeline
    pipeline = SpeechNoisePipeline(config)
    
    # BFCL data
    data_path = "data/BFCL_v3_live_simple.json"
    bfcl_data = load_bfcl_data(data_path)
    
    print(f"Loaded {len(bfcl_data)} test cases from {data_path}")
    
    test_subset = bfcl_data[:15] # examples number
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
    output_path = "data/BFCL_v3_live_simple_spoken2.json"
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