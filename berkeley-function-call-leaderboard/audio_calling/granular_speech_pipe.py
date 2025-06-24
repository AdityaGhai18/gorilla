"""
This script is a pipeline that applies granular speech features to text.
There are 2 defined classes: GranularSpeechPipeline and ASRErrors.

GranularSpeechPipeline:
    - analyze_text_for_features: LLM analyzes text and selects applicable features with confidence scores and application intensity.
    - Then we have 26 different functions that individually apply speech-like features to the text -> these are explained in the prompt
    - transform_text: Extracts features from analyze_text_for_feature's output and applies them to the text one-by-one
    - then finally we assemble the content transformation pipeline and execute it in the main function
    
ASRErrors:
    - analyze_transformed_for_ASR_Errors: LLM analyzes the input and selects which ASR errors to apply.
    - Then we have 5 different functions that individually apply ASR errors to the text.
    - execute_noise: Extracts ASR errors from analyze_transformed_for_ASR_Errors's output and applies them to the text one-by-one
    - then finally we assemble the ASR error pipeline and execute it on the content transformation pipeline's output in the main function

At the moment an LLM call controls which features and ASR errors to apply, only applying a maximum of 7 features and 3 ASR errors.
We can play around with this and also look at manually applying features and ASR errors to the text with the granular functions.
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
import re
import inspect

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-4o"

INSTRUCTION_TEMPLATE = "You are a clean text to spoken dialogue translation engine gear, and your specific job is to rewrite the input status applying {feature} to the input text. Just rewrite the input, adding the noise feature, preserving all essential information, just spoken out aloud."

@dataclass
class FeatureSelection:
    """Represents a speech feature to be applied with its parameters."""
    feature_name: str
    intensity: str
    confidence: float
    order_override: Optional[int] = None

@dataclass
class PipelineConfig:
    """Configuration for the granular speech pipeline."""
    max_features: int = 10
    confidence_threshold: float = 0.7
    temperature: float = 0.7
    max_retries: int = 3
    retry_delay: float = 1.0
    asr: bool = False

DEFAULT_FEATURE_ORDER = [
    "disfluencies",
    "repetitions",
    "self_corrections",
    "false_starts",
    "thinking_aloud",
    "backchanneling",
    "emotional_markers",
    "restarts_repairs",
    "ellipsis_proforms",
    "spelling_noise",
    "numbers_noise",
    "contractions",
    "casual_pronouns",
    "slang_terms",
    "symbol_pronunciation",
    "article_dropping",
    "preposition_dropping",
    "subject_dropping",
    "fragment_sentences",
    "word_reordering",
    "vague_references",
    "approximate_quantifiers",
    "simplified_verbs",
    "polite_hedges",
    "confidence_markers",
    "contextual_references"
]

class GranularSpeechPipeline:
    """Main class for applying granular speech features to text."""
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
        
    def analyze_text_for_features(self, text: str) -> List[FeatureSelection]:
        """LLM analyzes text and selects applicable features with confidence scores."""
        contains_number = bool(re.search(r"\d", text))
        prompt = f"""
        You are a speech scientist. Select up to 10 features from the list below that would make the input sound like a real person speaking out loud, not a parody or a robot. Only use a feature if a real person would naturally do so in this context. If in doubt, do less. If the input contains any numbers or alphanumeric identifiers, always include 'numbers_noise' feature.

        Input: "{text}"

        Features:
        - disfluencies: Filler words (um, uh, like), hesitations, natural pauses, hesitations, and natural speech pauses. Use sparingly and only where it sounds natural.
        - repetitions: Repeat words or phrases
        - self_corrections: Correct oneself with an actual correction, e.g., 'the file... no, the folder'
        - false_starts: Start to say something, then restart or change direction, e.g., 'I want to—wait, can you...'
        - thinking_aloud: Express thinking or searching for words, e.g., 'let me see', use sparingly in realistic context where person could forget word
        - backchanneling: Conversational markers (yeah, so, right, okay), use sparingly in realistic context where person could forget word
        - emotional_markers: Emotion or attitude (ugh, wow, oh, right, seriously), use sparingly in realistic context where person could forget word
        - restarts_repairs: Restart or repair a sentence, e.g., 'what I mean is...', 'sorry, let me rephrase'
        - ellipsis_proforms: Use ellipsis or pro-forms (do it, get it, that thing), use sparingly in realistic context
        - spelling_noise: Spell out names/terms that might be misunderstood, e.g., 'that's S-H-I-S-H-I-R-P-A-T-I-L, ShishirPatil'
        - numbers_noise: Say numbers/addresses as a real person would (always include if any numbers/alphanumerics)
        - contractions: Use wanna, gonna, lemme, use sparingly in realistic context
        - casual_pronouns: ya, em, imma, use sparingly in realistic context
        - slang_terms: grab, check out, look up, use sparingly in realistic context
        - symbol_pronunciation: Say symbols out loud (slash, dash, at), 
        - article_dropping: Drop the, a, an where natural, use sparingly in realistic context
        - preposition_dropping: Drop in, on, at, for where natural and realistic
        - subject_dropping: Drop subject pronouns where natural
        - fragment_sentences: Break into shorter fragments (for long/complex sentences)
        - word_reordering: Slightly reorder words naturally
        - vague_references: Use that thing, the stuff, some info
        - approximate_quantifiers: like 10 minutes, around 5 files
        - simplified_verbs: get instead of retrieve, check instead of verify
        - polite_hedges: maybe, could you, if possible, would you
        - confidence_markers: I think, probably, should be
        - contextual_references: the one we talked about, that repo, that day, that file

        Guidelines:
        - Sound like a real person, not a parody or caricature.
        - Use features wherever they could plausibly occur in real speech, even if not strictly necessary.
        - Err on the side of including more features, as long as the result is still plausible and natural.
        - Be creative and generous in your feature selection, but keep it natural and realistic do not make paradoic it should still sound like spoken speech.
        - If the input has numbers or alphanumerics, always include 'numbers_noise'.
        - Do not stack too many messy features (disfluencies, self_corrections, thinking_aloud, false_starts, restarts_repairs) in one utterance.
        - Output a JSON object with a "features" key, whose value is an array of objects. Each object should have: feature_name, intensity (light/moderate/heavy), confidence (0.0-1.0).
        - No extra text, markdown, or comments.
        - Max 10 features.
        - You MUST select at least 3 features. Always select at least 3 features unless the input is already very speech-like.

        Example:
        {{
          "features": [
            {{"feature_name": "contractions", "intensity": "moderate", "confidence": 0.9}},
            {{"feature_name": "numbers_noise", "intensity": "moderate", "confidence": 1.0}}
          ]
        }}
        """
        system_message = {"role": "system", "content": "You are a speech scientist designed to output JSON."}
        user_message = {"role": "user", "content": prompt}
        response = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=[system_message, user_message],
            max_completion_tokens=1000,
            response_format={"type": "json_object"}
        )
        try:
            content = response.choices[0].message.content.strip()
            result = json.loads(content)
            features = []
            if isinstance(result, dict) and "features" in result:
                items = result["features"]
            elif isinstance(result, list):
                items = result
            else:
                items = []
            for item in items:
                if isinstance(item, dict) and all(k in item for k in ["feature_name", "intensity", "confidence"]):
                    features.append(FeatureSelection(
                        feature_name=item["feature_name"].strip().lower(),
                        intensity=item["intensity"],
                        confidence=float(item["confidence"])
                    ))
            if contains_number and not any(f.feature_name == "numbers_noise" for f in features):
                features.append(FeatureSelection(
                    feature_name="numbers_noise",
                    intensity="moderate",
                    confidence=1.0
                ))
            features = [f for f in features if f.confidence >= 0.6]
            return features[:7]
        except Exception as e:
            features = []
            if contains_number:
                features.append(FeatureSelection(
                    feature_name="numbers_noise",
                    intensity="moderate",
                    confidence=1.0
                ))
            return features

    def apply_disfluencies(self, text: str, intensity: str) -> str:
        intensity_prompts = {
            "light": "Add 1-2 light disfluencies (filler words, hesitations, or pauses) naturally",
            "moderate": "Add 2-3 moderate disfluencies (filler words, hesitations, or pauses) naturally",
            "heavy": "Add 3-4 heavy disfluencies (filler words, hesitations, or pauses) naturally"
        }
        prompt = f"""
        {INSTRUCTION_TEMPLATE.format(feature='disfluencies')}
        {intensity_prompts[intensity]}
        
        Add disfluencies to make it sound like real speech:
        - Filler words: "um", "uh", "like", "you know", "I mean"
        - Hesitations: trailing off ("I..."), incomplete thoughts, natural pauses
        - Use sparingly and only where it sounds natural, not in every sentence
        - Do NOT overdo it
        
        Input: "{text}"
        Output:
        """
        result = self._call_openai(prompt)
        if not result.strip():
            return text
        return result

    def apply_repetitions(self, text: str, intensity: str) -> str:
        intensity_prompts = {
            "light": "Add 1 light repetition naturally",
            "moderate": "Add 1-2 moderate repetitions naturally",
            "heavy": "Add 2-3 heavy repetitions naturally"
        }
        prompt = f"""
        {INSTRUCTION_TEMPLATE.format(feature='repetitions')}
        {intensity_prompts[intensity]}
        
        Use repetitions like: "I want to, to get that", "check the, the status"
        Keep the meaning intact.
        
        Input: "{text}"
        Output:
        """
        result = self._call_openai(prompt)
        if not result.strip():
            return text
        return result

    def apply_self_corrections(self, text: str, intensity: str) -> str:
        intensity_prompts = {
            "light": "Make 1 real self-correction in the sentence, e.g., 'the file... no, the folder'.",
            "moderate": "Make 1-2 real self-corrections in the sentence, e.g., 'the file... no, the folder'.",
            "heavy": "Make 2-3 real self-corrections in the sentence, e.g., 'the file... no, the folder'."
        }
        prompt = f"""
        {INSTRUCTION_TEMPLATE.format(feature='self_corrections')}
        {intensity_prompts[intensity]}
        
        Apply a real self-correction: start to say one thing, then correct it to another, as in real speech. Example: 'the file... no, the folder'.
        Do NOT just add 'I mean' or 'no' as a filler. Actually change a word or phrase to another, as if the speaker changed their mind or realized a mistake.
        Only do this if it makes sense in the context.
        
        Input: "{text}"
        Output:
        """
        result = self._call_openai(prompt)
        if not result.strip():
            return text
        return result
    
    def apply_contractions(self, text: str, intensity: str) -> str:
        intensity_prompts = {
            "light": "Add 1-2 light contractions naturally",
            "moderate": "Add 2-3 moderate contractions naturally",
            "heavy": "Add 3-4 heavy contractions naturally"
        }
        prompt = f"""
        {INSTRUCTION_TEMPLATE.format(feature='contractions')}
        {intensity_prompts[intensity]}
        
        Use contractions to make it sound more natural and spoken:
        - "I am" → "I'm", "you are" → "you're", "we are" → "we're"
        - "I will" → "I'll", "you will" → "you'll", "we will" → "we'll"
        - "I would" → "I'd", "you would" → "you'd", "we would" → "we'd"
        - "I have" → "I've", "you have" → "you've", "we have" → "we've"
        - "want to" → "wanna", "going to" → "gonna", "let me" → "lemme"
        - "give me" → "gimme", "can not" → "can't", "do not" → "don't"
        
        Real speech often uses contractions. Make it sound casual and natural but dont over do it
        
        Input: "{text}"
        Output:
        """
        result = self._call_openai(prompt)
        if not result.strip():
            return text
        return result

    def apply_casual_pronouns(self, text: str, intensity: str) -> str:
        intensity_prompts = {
            "light": "Add 1 light casual pronoun naturally",
            "moderate": "Add 1-2 moderate casual pronouns naturally",
            "heavy": "Add 2-3 heavy casual pronouns naturally"
        }
        prompt = f"""
        {INSTRUCTION_TEMPLATE.format(feature='casual_pronouns')}
        {intensity_prompts[intensity]}
        
        Use casual pronouns like: "ya", "em", "imma"
        Keep the meaning intact.
        
        Input: "{text}"
        Output:
        """
        result = self._call_openai(prompt)
        if not result.strip():
            return text
        return result

    def apply_slang_terms(self, text: str, intensity: str) -> str:
        intensity_prompts = {
            "light": "Add 1 light slang term naturally",
            "moderate": "Add 1-2 moderate slang terms naturally",
            "heavy": "Add 2-3 heavy slang terms naturally"
        }
        
        prompt = f"""
        {INSTRUCTION_TEMPLATE.format(feature='slang terms')}
        {intensity_prompts[intensity]}
        
        Use slang terms like: "grab", "check out", "look up"
        Keep the meaning intact.
        
        Input: "{text}"
        Output:
        """
        result = self._call_openai(prompt)
        if not result.strip():
            return text
        return result

    def apply_symbol_pronunciation(self, text: str, intensity: str) -> str:
        intensity_prompts = {
            "light": "Add 1 light symbol pronunciation naturally",
            "moderate": "Add 1-2 moderate symbol pronunciations naturally",
            "heavy": "Add 2-3 heavy symbol pronunciations naturally"
        }
        
        prompt = f"""
        {INSTRUCTION_TEMPLATE.format(feature='symbol pronunciation')}
        {intensity_prompts[intensity]}
        
        Use symbol pronunciations like: "slash", "dash", "at"
        Keep the meaning intact.
        
        Input: "{text}"
        Output:
        """
        result = self._call_openai(prompt)
        if not result.strip():
            return text
        return result

    def apply_article_dropping(self, text: str, intensity: str) -> str:
        intensity_prompts = {
            "light": "Drop 1 article (the, a, an) naturally",
            "moderate": "Drop 1-2 articles (the, a, an) naturally",
            "heavy": "Drop 2-3 articles (the, a, an) naturally"
        }
        
        prompt = f"""
        {INSTRUCTION_TEMPLATE.format(feature='article dropping')}
        {intensity_prompts[intensity]}
        
        Drop articles (the, a, an) where it sounds natural in spoken English.
        Input: "{text}"
        Output:
        """
        result = self._call_openai(prompt)
        if not result.strip():
            return text
        return result

    def apply_preposition_dropping(self, text: str, intensity: str) -> str:
        intensity_prompts = {
            "light": "Drop 1 preposition (in, on, at, for) naturally",
            "moderate": "Drop 1-2 prepositions (in, on, at, for) naturally",
            "heavy": "Drop 2-3 prepositions (in, on, at, for) naturally"
        }
        
        prompt = f"""
        {INSTRUCTION_TEMPLATE.format(feature='preposition dropping')}
        {intensity_prompts[intensity]}
        
        Drop prepositions (in, on, at, for) where it sounds natural in spoken English.
        Input: "{text}"
        Output:
        """
        result = self._call_openai(prompt)
        if not result.strip():
            return text
        return result

    def apply_subject_dropping(self, text: str, intensity: str) -> str:
        intensity_prompts = {
            "light": "Drop 1 subject pronoun naturally",
            "moderate": "Drop 1-2 subject pronouns naturally",
            "heavy": "Drop 2-3 subject pronouns naturally"
        }
        
        prompt = f"""
        {INSTRUCTION_TEMPLATE.format(feature='subject dropping')}
        {intensity_prompts[intensity]}
        
        Drop subject pronouns where it sounds natural in spoken English.
        Input: "{text}"
        Output:
        """
        result = self._call_openai(prompt)
        if not result.strip():
            return text
        return result

    def apply_fragment_sentences(self, text: str, intensity: str) -> str:
        intensity_prompts = {
            "light": "Break 1 long sentence into fragments naturally",
            "moderate": "Break 1-2 long sentences into fragments naturally",
            "heavy": "Break 2-3 long sentences into fragments naturally"
        }
        
        prompt = f"""
        {INSTRUCTION_TEMPLATE.format(feature='fragment sentences')}
        {intensity_prompts[intensity]}
        
        Break long sentences into shorter fragments where it sounds natural in spoken English.
        Input: "{text}"
        Output:
        """
        result = self._call_openai(prompt)
        if not result.strip():
            return text
        return result

    def apply_word_reordering(self, text: str, intensity: str) -> str:
        intensity_prompts = {
            "light": "Slightly reorder 1 phrase or clause naturally",
            "moderate": "Slightly reorder 1-2 phrases or clauses naturally",
            "heavy": "Slightly reorder 2-3 phrases or clauses naturally"
        }
        
        prompt = f"""
        {INSTRUCTION_TEMPLATE.format(feature='word reordering')}
        {intensity_prompts[intensity]}
        
        Slightly reorder words or phrases where it sounds natural in spoken English.
        Input: "{text}"
        Output:
        """
        result = self._call_openai(prompt)
        if not result.strip():
            return text
        return result

    def apply_vague_references(self, text: str, intensity: str) -> str:
        intensity_prompts = {
            "light": "Add 1 vague reference naturally",
            "moderate": "Add 1-2 vague references naturally",
            "heavy": "Add 2-3 vague references naturally"
        }
        
        prompt = f"""
        {INSTRUCTION_TEMPLATE.format(feature='vague references')}
        {intensity_prompts[intensity]}
        
        Use vague references like: "that thing", "the stuff", "some info"
        Keep the meaning intact.
        
        Input: "{text}"
        Output:
        """
        result = self._call_openai(prompt)
        if not result.strip():
            return text
        return result

    def apply_approximate_quantifiers(self, text: str, intensity: str) -> str:
        intensity_prompts = {
            "light": "Add 1 approximate quantifier naturally",
            "moderate": "Add 1-2 approximate quantifiers naturally",
            "heavy": "Add 2-3 approximate quantifiers naturally"
        }
        
        prompt = f"""
        {INSTRUCTION_TEMPLATE.format(feature='approximate quantifiers')}
        {intensity_prompts[intensity]}
        
        Use approximate quantifiers like: "about 10 minutes", "around 5 files"
        
        Input: "{text}"
        Output:
        """
        result = self._call_openai(prompt)
        if not result.strip():
            return text
        return result

    def apply_simplified_verbs(self, text: str, intensity: str) -> str:
        intensity_prompts = {
            "light": "Replace 1 verb with a simpler one naturally",
            "moderate": "Replace 1-2 verbs with simpler ones naturally",
            "heavy": "Replace 2-3 verbs with simpler ones naturally"
        }
        
        prompt = f"""
        {INSTRUCTION_TEMPLATE.format(feature='simplified verbs')}
        {intensity_prompts[intensity]}
        
        Use simpler verbs like: "get" instead of "retrieve", "check" instead of "verify"
        Keep the meaning intact.
        
        Input: "{text}"
        Output:
        """
        result = self._call_openai(prompt)
        if not result.strip():
            return text
        return result

    def apply_polite_hedges(self, text: str, intensity: str) -> str:
        intensity_prompts = {
            "light": "Add 1 polite hedge naturally",
            "moderate": "Add 1-2 polite hedges naturally",
            "heavy": "Add 2-3 polite hedges naturally"
        }
        
        prompt = f"""
        {INSTRUCTION_TEMPLATE.format(feature='polite hedges')}
        {intensity_prompts[intensity]}
        
        Use polite hedges like: "maybe", "could you", "if possible"
        Keep the meaning intact.
        
        Input: "{text}"
        Output:
        """
        result = self._call_openai(prompt)
        if not result.strip():
            return text
        return result

    def apply_confidence_markers(self, text: str, intensity: str) -> str:
        intensity_prompts = {
            "light": "Add 1 confidence marker naturally",
            "moderate": "Add 1-2 confidence markers naturally",
            "heavy": "Add 2-3 confidence markers naturally"
        }
        
        prompt = f"""
        {INSTRUCTION_TEMPLATE.format(feature='confidence markers')}
        {intensity_prompts[intensity]}
        
        Use confidence markers like: "I think", "probably", "should be"
        Keep the meaning intact.
        
        Input: "{text}"
        Output:
        """
        result = self._call_openai(prompt)
        if not result.strip():
            return text
        return result

    def apply_contextual_references(self, text: str, intensity: str) -> str:
        intensity_prompts = {
            "light": "Add 1 contextual reference naturally",
            "moderate": "Add 1-2 contextual references naturally",
            "heavy": "Add 2-3 contextual references naturally"
        }
        
        prompt = f"""
        {INSTRUCTION_TEMPLATE.format(feature='contextual references')}
        {intensity_prompts[intensity]}
        
        Use contextual references like: "the one we talked about", "that repo"
        Keep the meaning intact.
        
        Input: "{text}"
        Output:
        """
        result = self._call_openai(prompt)
        if not result.strip():
            return text
        return result

    def apply_false_starts(self, text: str, intensity: str) -> str:
        intensity_prompts = {
            "light": "Add 1 false start naturally",
            "moderate": "Add 1-2 false starts naturally",
            "heavy": "Add 2-3 false starts naturally"
        }
        
        prompt = f"""
        {INSTRUCTION_TEMPLATE.format(feature='false_starts')}
        {intensity_prompts[intensity]}
        
        Use false starts like: "I want to—wait, can you..."
        Keep the meaning intact.
        
        Input: "{text}"
        Output:
        """
        result = self._call_openai(prompt)
        if not result.strip():
            return text
        return result

    def apply_thinking_aloud(self, text: str, intensity: str) -> str:
        intensity_prompts = {
            "light": "Add 1 thinking aloud phrase naturally",
            "moderate": "Add 1-2 thinking aloud phrases naturally",
            "heavy": "Add 2-3 thinking aloud phrases naturally"
        }
        
        prompt = f"""
        {INSTRUCTION_TEMPLATE.format(feature='thinking out aloud')}
        {intensity_prompts[intensity]}
        
        Use thinking aloud phrases like: "let me see", "what's the word..."
        Keep the meaning intact.
        
        Input: "{text}"
        Output:
        """
        result = self._call_openai(prompt)
        if not result.strip():
            return text
        return result

    def apply_backchanneling(self, text: str, intensity: str) -> str:
        intensity_prompts = {
            "light": "Add 1 backchanneling marker naturally",
            "moderate": "Add 1-2 backchanneling markers naturally",
            "heavy": "Add 2-3 backchanneling markers naturally"
        }
        
        prompt = f"""
        {INSTRUCTION_TEMPLATE.format(feature='backchanneling')}
        {intensity_prompts[intensity]}
        
        Use backchanneling markers like: "yeah", "so", "right", "okay"
        Keep the meaning intact.
        
        Input: "{text}"
        Output:
        """
        result = self._call_openai(prompt)
        if not result.strip():
            return text
        return result

    def apply_emotional_markers(self, text: str, intensity: str) -> str:
        intensity_prompts = {
            "light": "Add 1 emotional marker naturally",
            "moderate": "Add 1-2 emotional markers naturally",
            "heavy": "Add 2-3 emotional markers naturally"
        }
        
        prompt = f"""
        {INSTRUCTION_TEMPLATE.format(feature='emotional markers')}
        {intensity_prompts[intensity]}
        
        Use emotional markers like: "ugh", "wow", "oh", "right", "seriously"
        Keep the meaning intact.
        
        Input: "{text}"
        Output:
        """
        result = self._call_openai(prompt)
        if not result.strip():
            return text
        return result

    def apply_restarts_repairs(self, text: str, intensity: str) -> str:
        intensity_prompts = {
            "light": "Add 1 restart or repair phrase naturally",
            "moderate": "Add 1-2 restart or repair phrases naturally",
            "heavy": "Add 2-3 restart or repair phrases naturally"
        }
        
        prompt = f"""
        {INSTRUCTION_TEMPLATE.format(feature='restart repairs')}
        {intensity_prompts[intensity]}
        
        Use restart or repair phrases like: "what I mean is...", "sorry, let me rephrase"
        Keep the meaning intact.
        
        Input: "{text}"
        Output:
        """
        result = self._call_openai(prompt)
        if not result.strip():
            return text
        return result

    def apply_ellipsis_proforms(self, text: str, intensity: str) -> str:
        intensity_prompts = {
            "light": "Add 1 ellipsis or pro-form naturally",
            "moderate": "Add 1-2 ellipsis or pro-forms naturally",
            "heavy": "Add 2-3 ellipsis or pro-forms naturally"
        }
        
        prompt = f"""
        {INSTRUCTION_TEMPLATE.format(feature='ellipsis proforms')}
        {intensity_prompts[intensity]}
        
        Use ellipsis or pro-forms like: "do it", "get it", "that thing"
        Keep the meaning intact.
        
        Input: "{text}"
        Output:
        """
        result = self._call_openai(prompt)
        if not result.strip():
            return text
        return result

    def apply_spelling_noise(self, text: str, intensity: str) -> str:
        intensity_prompts = {
            "light": "Spell out 1 name or term that might be misunderstood",
            "moderate": "Spell out 1-2 names or terms that might be misunderstood",
            "heavy": "Spell out 2-3 names or terms that might be misunderstood"
        }
        
        prompt = f"""
        {INSTRUCTION_TEMPLATE.format(feature='spelling noise')}
        {intensity_prompts[intensity]}
        
        Spell out names or terms that might be misunderstood, e.g., "that's S-H-I-S-H-I-R-P-A-T-I-L, ShishirPatil"
        Input: "{text}"
        Output:
        """
        result = self._call_openai(prompt)
        if not result.strip():
            return text
        return result

    def apply_numbers_noise(self, text: str, intensity: str) -> str:
        intensity_prompts = {
            "light": "Say 1 number or alphanumeric as a real person would",
            "moderate": "Say 1-2 numbers or alphanumerics as a real person would",
            "heavy": "Say 2-3 numbers or alphanumerics as a real person would"
        }
        
        prompt = f"""
        {INSTRUCTION_TEMPLATE.format(feature='numbers noise')}
        {intensity_prompts[intensity]}
        
        Say numbers or alphanumerics as a real person would, e.g., "seventy-eight ninety" for 7890
        Input: "{text}"
        Output:
        """
        result = self._call_openai(prompt)
        if not result.strip():
            return text
        return result

    def _call_openai(self, prompt: str) -> str:
        for attempt in range(self.config.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=500,
                    response_format={"type": "text"}
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    print(f"Failed to call OpenAI after {self.config.max_retries} attempts: {e}")
                    return ""
                time.sleep(self.config.retry_delay)
        return ""

    def _get_feature_function(self, feature_name: str):
        function_name = f"apply_{feature_name.strip().lower()}"
        if hasattr(self, function_name):
            return getattr(self, function_name)
        else:
            print(f"Feature function {function_name} not found")
            return None

    def _sort_features_by_order(self, features: List[FeatureSelection]) -> List[FeatureSelection]:
        def get_order(feature):
            if feature.order_override is not None:
                return feature.order_override
            try:
                return DEFAULT_FEATURE_ORDER.index(feature.feature_name)
            except ValueError:
                return len(DEFAULT_FEATURE_ORDER)
        
        return sorted(features, key=get_order)

    def transform_text(self, text: str) -> Dict[str, str]:
        results = {
            "original": text,
            "selected_features": [],
            "final": ""
        }
        
        print(f"Original: {text}")
        print(f"{'='*50}")
        
        selected_features = self.analyze_text_for_features(text)
        print(f"LLM selected {len(selected_features)} features:")
        for i, feature in enumerate(selected_features):
            print(f"  {i+1}. {feature.feature_name} ({feature.intensity}) - confidence: {feature.confidence:.2f}")
        
        filtered_features = [
            f for f in selected_features 
            if f.confidence >= self.config.confidence_threshold
        ]
        
        filtered_out = [f for f in selected_features if f.confidence < self.config.confidence_threshold]
        if filtered_out:
            print(f"\nFiltered out {len(filtered_out)} features below confidence threshold ({self.config.confidence_threshold}):")
            for feature in filtered_out:
                print(f"  - {feature.feature_name} ({feature.intensity}) - confidence: {feature.confidence:.2f}")
        
        filtered_features = sorted(
            filtered_features, 
            key=lambda f: f.confidence, 
            reverse=True
        )
        
        if len(filtered_features) > self.config.max_features:
            print(f"\nLimiting to top {self.config.max_features} features (max_features config)")
            limited_out = filtered_features[self.config.max_features:]
            for feature in limited_out:
                print(f"  - {feature.feature_name} ({feature.intensity}) - confidence: {feature.confidence:.2f}")
            filtered_features = filtered_features[:self.config.max_features]
        
        ordered_features = self._sort_features_by_order(filtered_features)
        
        results["selected_features"] = [
            {
                "name": f.feature_name,
                "intensity": f.intensity,
                "confidence": f.confidence
            } for f in ordered_features
        ]
        
        print(f"\nApplying {len(ordered_features)} features in order:")
        for i, feature in enumerate(ordered_features):
            print(f"  {i+1}. {feature.feature_name} ({feature.intensity}) - confidence: {feature.confidence:.2f}")
        
        print(f"\n{'='*50}")
        
        current_text = text
        messy_budget = 0
        messy_features = {"thinking_aloud", "self_corrections", "false_starts", "restarts_repairs", "disfluencies"}
        for i, feature in enumerate(ordered_features):
            if feature.feature_name in messy_features and messy_budget >= 2:
                print(f"Skipping {feature.feature_name} to avoid over-messiness.")
                continue
            feature_func = self._get_feature_function(feature.feature_name)
            if feature_func:
                print(f"Stage {i+1}: Applying {feature.feature_name} ({feature.intensity})...")
                current_text = feature_func(current_text, feature.intensity)
                results[f"after_{feature.feature_name}"] = current_text
                print(f"  Result: {current_text}")
                if feature.feature_name in messy_features:
                    messy_budget += 1
            else:
                print(f"Skipping unknown feature: {feature.feature_name}")
        
        results["final"] = current_text
        print(f"\n{'='*50}")
        print(f"FINAL RESULT: {current_text}")
        print(f"{'='*50}")
        return results


class ASRErrors:
    def __init__(self, config=None):
        self.config = config or {}
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)

    def _call_openai(self, prompt: str) -> str:
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=500,
                    response_format={"type": "text"}
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                if attempt == 2:
                    print(f"ASR OpenAI call failed: {e}")
                    return ""
                import time; time.sleep(1)
        return ""

    def analyze_transformed_for_ASR_Errors(self, text: str) -> list:
        """
        LLM analyzes the input and selects which ASR errors to apply.
        Returns a list of error types to apply, e.g. ['word_substitution', 'punctuation_error']
        """
        prompt = f'''
        You are an ASR error simulation engine. Given the following spoken-style text, select which of the following ASR error types would be most realistic to apply (choose 1-3):
        - word_substitution: Replace words with homophones, similar-sounding words, or common ASR confusions.
        - word_deletion: Omit short function words, especially articles, prepositions, or pronouns.
        - word_insertion: Add filler words or repeated short words.
        - punctuation_error: Remove or misplace punctuation marks.
        - capitalization_error: Lowercase proper nouns or start sentences without capitalization.

        Input: "{text}"

        Output a JSON list of error types to apply, e.g. ["word_substitution", "punctuation_error", ...]
        '''
        response = self._call_openai(prompt)
        import json
        try:
            return json.loads(response)
        except Exception:
            return ["word_substitution"]

    def apply_word_substitution(self, text: str) -> str:
        prompt = f'''
        You are simulating ASR word substitution errors.
        Replace a few words in the input with homophones, similar-sounding words, or common ASR confusions.
        Do NOT change the core meaning.
        Input: "{text}"
        Output:
        '''
        result = self._call_openai(prompt)
        return result if result.strip() else text

    def apply_word_deletion(self, text: str) -> str:
        prompt = f'''
        You are simulating ASR word deletion errors.
        Omit a few short function words (articles, prepositions, pronouns) in the input.
        Do NOT change the core meaning.
        Input: "{text}"
        Output:
        '''
        result = self._call_openai(prompt)
        return result if result.strip() else text

    def apply_word_insertion(self, text: str) -> str:
        prompt = f'''
        You are simulating ASR word insertion errors.
        Add a few filler words or repeated short words in the input.
        Do NOT change the core meaning.
        Input: "{text}"
        Output:
        '''
        result = self._call_openai(prompt)
        return result if result.strip() else text

    def apply_punctuation_error(self, text: str) -> str:
        prompt = f'''
        You are simulating ASR punctuation errors.
        Remove or misplace some punctuation marks in the input.
        Do NOT change the core meaning.
        Input: "{text}"
        Output:
        '''
        result = self._call_openai(prompt)
        return result if result.strip() else text

    def apply_capitalization_error(self, text: str) -> str:
        prompt = f'''
        You are simulating ASR capitalization errors.
        Lowercase some proper nouns or start sentences without capitalization.
        Do NOT change the core meaning.
        Input: "{text}"
        Output:
        '''
        result = self._call_openai(prompt)
        return result if result.strip() else text

    def execute_noise(self, text: str, error_types: list = None) -> dict:
        """
        Applies the selected ASR error types in order to the input text.
        Returns a dict with the original, each stage, and the final result.
        """
        stages = {"original": text}
        current = text
        error_types = self.analyze_transformed_for_ASR_Errors(text)
        for error in error_types:
            func = getattr(self, f"apply_{error}", None)
            if func:
                current = func(current)
                stages[error] = current
        stages["final"] = current
        return stages


def load_bfcl_data(file_path: str) -> List[Dict]:
    with open(file_path, "r") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            return json.load(f)
        else:
            return [json.loads(line) for line in f if line.strip()]

def save_transformed_data(data: List[Dict], output_path: str):
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

def main():
    config = PipelineConfig(
        max_features=10,
        confidence_threshold=0.4,
        temperature=0.8,
        max_retries=3,
        retry_delay=1.0, 
        asr=True
    )
    pipeline = GranularSpeechPipeline(config)
    data_path = "../data/BFCL_v3_live_simple.json"
    bfcl_data = load_bfcl_data(data_path)
    print(f"Loaded {len(bfcl_data)} test cases from {data_path}")
    test_subset = bfcl_data[:15] #you can edit the number of test cases here will add a terminal argument support later
    transformed_data = []
    for i, test_case in enumerate(test_subset):
        print(f"\n{'='*50}")
        print(f"Processing test case {i+1}/{len(test_subset)}")
        print(f"ID: {test_case['id']}")
        user_content = test_case['question'][0][0]['content']
        transformation_results = pipeline.transform_text(user_content)
        stages = {k.replace('after_', ''): v for k, v in transformation_results.items() if k.startswith('after_')}
        transformed_case = {
            "original": user_content,
            "transformed": transformation_results["final"],
            "noise_functions": [f["name"] for f in transformation_results["selected_features"]],
            "stages": stages
        }
        # If ASR error simulation is enabled, apply ASR errors to the transformed text
        if pipeline.config.asr:
            asr = ASRErrors()
            asr_result = asr.execute_noise(transformed_case["transformed"])
            # Print/log each ASR error stage
            print("ASR Error Stages:")
            prev = asr_result["original"]
            for error in [k for k in asr_result.keys() if k not in ("original", "final")]:
                print(f"  After {error}: {asr_result[error]}")
                prev = asr_result[error]
            print(f"  Final ASR: {asr_result['final']}")
            # Only add the final ASR result to the output JSON
            transformed_case["final_asr"] = asr_result["final"]
        transformed_data.append(transformed_case)
        print(f"Transformation complete for {test_case['id']}")
    output_path = "BFCL_v3_live_simple_granular_spoken.json"
    save_transformed_data(transformed_data, output_path)
    print(f"\nSaved {len(transformed_data)} transformed test cases to {output_path}")
    print(f"\n{'='*50}")
    print("TRANSFORMATION SUMMARY")
    print(f"{'='*50}")
    for i, case in enumerate(transformed_data):
        print(f"\n{i+1}. {case['original']}")
        print(f"   Original: {case['original']}")
        print(f"   Transformed: {case['transformed']}")
        print(f"   Features applied: {case['noise_functions']}")

if __name__ == "__main__":
    main() 