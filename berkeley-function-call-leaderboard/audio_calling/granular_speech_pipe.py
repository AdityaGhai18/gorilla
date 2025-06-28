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
    max_features: int = 8
    confidence_threshold: float = 0.5
    temperature: float = 0.7
    max_retries: int = 3
    retry_delay: float = 1.0
    asr: bool = False

DEFAULT_FEATURE_ORDER = [
    "sentence_restructuring",
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
    "detail_dropping",
    "confidence_markers",
    "contextual_references"
]

class GranularSpeechPipeline:
    """Main class for applying granular speech features to text."""
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
        
    def is_english_text(self, text: str, threshold: float = 0.9) -> bool:
        """
        Check if text is in English using character-based script detection.
            
        Returns:
            True if text is likely English, False otherwise
        """
        if not text or not text.strip():
            return False
        
        for char in text:
            if char.isspace():  #
                continue
            
            char_code = ord(char)
            
            # Chinese characters (CJK Unified Ideographs)
            if (0x4E00 <= char_code <= 0x9FFF or  # CJK Unified Ideographs
                0x3400 <= char_code <= 0x4DBF or  # CJK Unified Ideographs Extension A
                0x20000 <= char_code <= 0x2A6DF or  # CJK Unified Ideographs Extension B
                0x2A700 <= char_code <= 0x2B73F or  # CJK Unified Ideographs Extension C
                0x2B740 <= char_code <= 0x2B81F or  # CJK Unified Ideographs Extension D
                0x2B820 <= char_code <= 0x2CEAF):   # CJK Unified Ideographs Extension E
                return False
            
            # Korean Hangul
            if (0xAC00 <= char_code <= 0xD7AF or  # Hangul Syllables
                0x1100 <= char_code <= 0x11FF or  # Hangul Jamo
                0x3130 <= char_code <= 0x318F):   # Hangul Compatibility Jamo
                return False
            
            # Japanese Hiragana and Katakana
            if (0x3040 <= char_code <= 0x309F or  # Hiragana
                0x30A0 <= char_code <= 0x30FF):   # Katakana
                return False
            
            # Arabic
            if (0x0600 <= char_code <= 0x06FF or  # Arabic
                0x0750 <= char_code <= 0x077F or  # Arabic Supplement
                0x08A0 <= char_code <= 0x08FF or  # Arabic Extended-A
                0xFB50 <= char_code <= 0xFDFF or  # Arabic Presentation Forms-A
                0xFE70 <= char_code <= 0xFEFF):   # Arabic Presentation Forms-B
                return False
            
            # Thai
            if 0x0E00 <= char_code <= 0x0E7F:
                return False
            
            # Vietnamese diacritics (Latin characters with diacritics)
            if char in 'àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ':
                return False
        
        # If no non-Latin scripts found, assume it's English
        return True
    
    def filter_english_texts(self, texts: List[str], threshold: float = 0.9) -> List[str]:
        """
        Filter a list of texts to keep only English ones.
        
        Args:
            texts: List of texts to filter
            threshold: Minimum percentage of ASCII characters required
            
        Returns:
            List of English texts only
        """
        english_texts = []
        for text in texts:
            if self.is_english_text(text, threshold):
                english_texts.append(text)
        return english_texts
        
    def analyze_text_for_features(self, text: str) -> List[FeatureSelection]:
        """LLM analyzes text and selects applicable features with confidence scores."""
        contains_number = bool(re.search(r"\d", text))
        prompt = f"""
        You are a speech scientist analyzing how to convert written text into natural spoken dialogue. Your job is to select the most appropriate speech features to make the input sound like a real person speaking to a voice assistant (like Siri, Alexa, or Google Home).

        Input: "{text}"

        IMPORTANT: People talking to voice assistants are typically DIRECT, CASUAL, and COMMAND-LIKE. They want quick answers and don't waste time with formal language or excessive politeness. Think of how people actually talk to Siri / Alexa - they're direct and to the point.

        PRIORITY FEATURES (select 4-6 from these):
        - contractions: Use wanna, gonna, lemme, gimme, etc.
        - simplified_verbs: get instead of retrieve, check instead of verify
        - casual_pronouns: ya, em, imma
        - slang_terms: grab, check out, look up
        - disfluencies: Filler words (um, uh, like), hesitations, natural pauses
        - thinking_aloud: Express thinking or searching for words, e.g., 'let me see'
        - false_starts: Start to say something, then restart, e.g., 'I want to—wait, can you...'
        - self_corrections: Correct oneself with an actual correction, e.g., 'the file... no, the folder'
        - backchanneling: Conversational markers (yeah, so, right, okay)
        - emotional_markers: Emotion or attitude (oh, right, seriously)
        - article_dropping: Drop articles where natural
        - preposition_dropping: Drop prepositions where natural
        - subject_dropping: Drop subjects where natural
        - fragment_sentences: Use incomplete sentences
        - word_reordering: Natural word order changes
        - vague_references: Use pronouns and references
        - approximate_quantifiers: Use approximate numbers
        - symbol_pronunciation: Say symbols out loud
        - spelling_noise: ALWAYS include if there are names, usernames, or complex terms
        - numbers_noise: Say numbers naturally
        - detail_dropping: Drop unnecessary formal details (state abbreviations, titles, etc.)

        AVOID THESE FEATURES (they make speech too polite/formal):
        - confidence_markers: "I think", "probably", "should be" (too hedging)
        - restarts_repairs: "what I mean is...", "sorry, let me rephrase" (too apologetic)
        - contextual_references: "the one we talked about" (too conversational)
        - ellipsis_proforms: "do it", "get it" (too vague)

        FEATURE GROUPINGS - You MUST select at least 1 feature from each group:

        GROUP 1 - BASIC SPEECH PATTERNS (always include 1-2):
        - contractions: Use wanna, gonna, lemme, gimme, etc.
        - simplified_verbs: get instead of retrieve, check instead of verify
        - casual_pronouns: ya, em, imma
        - slang_terms: grab, check out, look up

        GROUP 2 - DISFLUENCIES & HESITATIONS (always include 1-2):
        - disfluencies: Filler words (um, uh, like), hesitations, natural pauses
        - thinking_aloud: Express thinking or searching for words, e.g., 'let me see'
        - false_starts: Start to say something, then restart, e.g., 'I want to—wait, can you...'
        - self_corrections: Correct oneself with an actual correction, e.g., 'the file... no, the folder'

        GROUP 3 - CONVERSATIONAL MARKERS (include 1):
        - backchanneling: Conversational markers (yeah, so, right, okay)
        - emotional_markers: Emotion or attitude (oh, right, seriously)
        - repetitions: Repeat words or phrases naturally

        GROUP 4 - STRUCTURAL CHANGES (include 1-2):
        - fragment_sentences: Use incomplete sentences
        - word_reordering: Natural word order changes
        - article_dropping: Drop articles where natural
        - preposition_dropping: Drop prepositions where natural
        - subject_dropping: Drop subjects where natural
        - detail_dropping: Drop unnecessary formal details (state in the address, titles, etc.)

        GROUP 5 - CONTENT MODIFICATIONS (include 1-2):
        - spelling_noise: Spell out names/terms that might be misunderstood
        - symbol_pronunciation: Say symbols out loud (slash, dash, at)
        - vague_references: Use that thing, the stuff, some info
        - approximate_quantifiers: like 10 minutes when something like 600 seconds is used


        Features to choose from:
        - sentence_restructuring: Completely rephrase written instructions into natural spoken language. Change sentence structure, word order, and phrasing to sound like someone actually speaking rather than reading written text.
        - disfluencies: Filler words (um, uh, like), hesitations, natural pauses. Use VERY sparingly - only 1-2 per sentence maximum.
        - repetitions: Repeat words or phrases naturally
        - self_corrections: Correct oneself with an actual correction, e.g., 'the file... no, the folder'
        - false_starts: Start to say something, then restart or change direction, e.g., 'I want to—wait, can you...'
        - thinking_aloud: Express thinking or searching for words, e.g., 'let me see', use sparingly
        - backchanneling: Conversational markers (yeah, so, right, okay), use sparingly
        - emotional_markers: Emotion or attitude (oh, right, seriously), use sparingly
        - spelling_noise: Spell out names/terms that might be misunderstood, e.g., 'that's S-H-I-S-H-I-R-P-A-T-I-L, ShishirPatil'
        - numbers_noise: Say numbers/addresses as a real person would (ALWAYS include if any numbers/alphanumerics)
        - contractions: Use wanna, gonna, lemme, use moderately
        - casual_pronouns: ya, em, imma, use sparingly
        - slang_terms: grab, check out, look up, use sparingly
        - symbol_pronunciation: Say symbols out loud (slash, dash, at)
        - article_dropping: Drop the, a, an where natural, use sparingly
        - preposition_dropping: Drop in, on, at, for where natural
        - subject_dropping: Drop subject pronouns where natural
        - fragment_sentences: Break into shorter fragments (for long/complex sentences)
        - word_reordering: Slightly reorder words naturally
        - vague_references: Use that thing, the stuff, some info
        - approximate_quantifiers: like 10 minutes when something like 600 seconds is used, around 5 files
        - simplified_verbs: get instead of retrieve, check instead of verify
        - detail_dropping: Drop unnecessary formal details (state abbreviations, titles, company suffixes)

        EXAMPLES BY INPUT TYPE:

        CALENDAR SCHEDULING:
        Written: "I would like to schedule a meeting with the marketing team for next Tuesday at 2:30 PM in the conference room, and could you please send out calendar invitations to all participants?"
        Spoken: "Schedule a meeting with marketing Tuesday at two-thirty. Send invites to everyone."
        Features: disfluencies, numbers_noise, contractions, simplified_verbs, backchanneling

        DOCUMENT EDITING:
        Written: "Please modify the quarterly report document by adding the financial data from Q3 and removing the outdated statistics from the previous version."
        Spoken: "Update the quarterly report with Q3 data. Remove the old stats."
        Features: disfluencies, simplified_verbs, contractions, thinking_aloud

        MUSIC PLAYBACK:
        Written: "I would like to play the album 'Midnight Dreams' by the artist 'Stellar Echo' and set the volume to 75% while enabling shuffle mode."
        Spoken: "Play Midnight Dreams by Stellar Echo. That's S-T-E-L-L-A-R E-C-H-O. Volume at seventy-five. Turn shuffle on."
        Features: numbers_noise, simplified_verbs, disfluencies, spelling_noise

        EMAIL COMPOSITION:
        Written: "Please compose a new email message addressed to john.smith@company.com with the subject line 'Project Update - Phase 2 Completion' and include the following content in the body."
        Spoken: "Write an email to john dot smith at company dot com. Subject is Project Update Phase 2. Add the content."
        Features: symbol_pronunciation, simplified_verbs, disfluencies, contractions, casual_pronouns

        FILE MANAGEMENT:
        Written: "I would like to access the quarterly report document located in the shared drive folder and create a backup copy in my personal directory."
        Spoken: "Get the quarterly report from shared drive. Make a backup in my directory."
        Features: disfluencies, self_corrections, simplified_verbs, contractions, thinking_aloud

        SOCIAL MEDIA:
        Written: "Please post a status update on my social media account with the message 'Excited to announce our new product launch!' and include the hashtag #innovation."
        Spoken: "Post a status excited to announce our new product launch. Add hashtag innovation."
        Features: disfluencies, simplified_verbs, contractions, emotional_markers, fragment_sentences

        SELECTION GUIDELINES:
        - sentence_restructuring is ALWAYS applied automatically (don't select it)
        - ALWAYS include numbers_noise if input has numbers/alphanumerics
        - ALWAYS include spelling_noise if input has names, usernames, or complex terms (like GitHub repos, technical names)
        - You MUST select at least 1 feature from each of the 5 groups above
        - Total of 6-8 features maximum (sentence_restructuring + your selections)
        - Be conservative with disfluencies - people don't say "um" that much to voice assistants
        - Focus on making it sound direct and efficient, not overly polite or formal
        - Choose features that make speech sound natural, not robotic or forced
        - Prioritize features that improve flow and naturalness over adding complexity
        - Avoid over-fragmentation - don't break sentences unnecessarily
        - Focus on natural conversational flow rather than artificial speech patterns
        - Ensure diversity across groups - don't stack too many features from the same group
        - With 8 features total, you can select 2 features from some groups for richer speech patterns

        Output a JSON object with a "features" key, whose value is an array of objects. Each object should have: feature_name, intensity (light/moderate/heavy), confidence (0.0-1.0).

        Example:
        {{
          "features": [
            {{"feature_name": "contractions", "intensity": "moderate", "confidence": 0.8}},
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
            
            # Always include sentence_restructuring as the first feature
            if not any(f.feature_name == "sentence_restructuring" for f in features):
                features.insert(0, FeatureSelection(
                    feature_name="sentence_restructuring",
                    intensity="moderate",
                    confidence=1.0
                ))
            
            return features[:8]
        except Exception as e:
            features = []
            if contains_number:
                features.append(FeatureSelection(
                    feature_name="numbers_noise",
                    intensity="moderate",
                    confidence=1.0
                ))
            return features

    def apply_sentence_restructuring(self, text: str, intensity: str) -> str:
        """Completely rephrase written instructions into natural spoken language."""
        intensity_prompts = {
            "light": "Make minor adjustments to sound more spoken",
            "moderate": "Significantly rephrase to sound like natural speech",
            "heavy": "Completely restructure the sentence to sound like someone actually speaking"
        }
        
        prompt = f"""
        You are converting written instructions into natural spoken dialogue. Your job is to COMPLETELY restructure the input to sound like someone actually speaking to a voice assistant, not reading written text.

        {intensity_prompts[intensity]}

        CRITICAL: Do NOT just add periods or break sentences. You must completely rephrase and restructure the content to sound natural.

        KEY PRINCIPLES:
        - Completely change the sentence structure and word order
        - Use natural conversational flow and rhythm
        - Be direct and casual - people want quick answers
        - Remove ALL formal language and politeness
        - Use natural speech patterns and word choices
        - Make it flow like someone thinking out loud
        - Be aggressive with restructuring - don't be timid
        - Change the entire approach to how the request is made

        EXAMPLES:
        Written: "I would like to schedule a meeting with the marketing team for next Tuesday at 2:30 PM in the conference room, and could you please send out calendar invitations to all participants?"
        Spoken: "Schedule a meeting with marketing Tuesday at two-thirty. Send invites to everyone."

        Written: "Please modify the quarterly report document by adding the financial data from Q3 and removing the outdated statistics from the previous version."
        Spoken: "Update the quarterly report with Q3 data. Remove the old stats."

        Written: "I would like to play the album 'Midnight Dreams' by the artist 'Stellar Echo' and set the volume to 75% while enabling shuffle mode."
        Spoken: "Play Midnight Dreams by Stellar Echo. Volume at seventy-five. Turn shuffle on."

        Written: "I would like to access the quarterly report document located in the shared drive folder and create a backup copy in my personal directory."
        Spoken: "Get the quarterly report from shared drive. Make a backup in my directory."

        
        Input: "{text}"
        Output:
        """
        result = self._call_openai(prompt)
        if not result.strip():
            return text
        return result

    def apply_disfluencies(self, text: str, intensity: str) -> str:
        intensity_prompts = {
            "light": "Add 1 light disfluency (filler word or hesitation) naturally",
            "moderate": "Add 1-2 moderate disfluencies (filler words or hesitations) naturally",
            "heavy": "Add 2 moderate disfluencies (filler words or hesitations) naturally"
        }
        prompt = f"""
        {INSTRUCTION_TEMPLATE.format(feature='disfluencies')}
        {intensity_prompts[intensity]}
        
        Add disfluencies to make it sound like real speech, but be VERY conservative:
        - Filler words: "um", "uh", "like", "you know", "I mean"
        - Hesitations: trailing off ("I..."), incomplete thoughts, natural pauses
        - Use VERY sparingly - people don't say "um" that much to voice assistants
        - Maximum 1-2 disfluencies per sentence
        - Do NOT overdo it - this should sound natural, not like someone struggling to speak
        - Place them where people naturally hesitate (before important words, when thinking)
        
        Examples:
        - "Get me... a Comfort Uber from twenty-twenty Addison Street"
        - "What's the weather in... Tel Aviv?"
        - "I need to... get the details for user ID seventy-eight ninety"
        
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
        
        IMPORTANT: Do NOT change any numbers or alphanumeric identifiers that are already in spoken form (like "seventy-eight ninety", "twenty-twenty", etc.). Keep them exactly as they are.
        
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
        Only break sentences that are actually long or complex. Do NOT over-fragment short sentences.
        Make it sound natural, not choppy.
        
        Examples:
        - "Get the details for user ID seven eight nine zero. There's a special request. It's black." (good)
        - "Get. The details. For user. ID seven. Eight nine. Zero." (bad - too choppy)
        
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
        
        IMPORTANT: Do NOT change any numbers or alphanumeric identifiers that are already in spoken form (like "seventy-eight ninety", "twenty-twenty", etc.). Keep them exactly as they are.
        
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
            "moderate": "that thing, the stuff, some info",  # <-- Add this line if missing
            "heavy": "that thing, the stuff, some info, whatever"
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
        Ensure changing units that are used in speech, for example 600 seconds should be 10 minutes.
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
        
        IMPORTANT: Do NOT change any numbers or alphanumeric identifiers that are already in spoken form (like "seventy-eight ninety", "twenty-twenty", etc.). Keep them exactly as they are.
        
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
        
        Use natural thinking aloud phrases that people actually use:
        - "let me see", "what's the word...", "I think", "maybe"
        - Use sparingly and only where it sounds natural
        - Don't overdo it - people don't constantly think out loud to voice assistants
        
        Examples:
        - "Let me see... get the details for user ID seventy-eight ninety"
        - "What's the weather in Tel Aviv? I think... in Fahrenheit"
        
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
        
        Use natural backchanneling markers that people actually use with voice assistants:
        - "okay", "right", "yeah", "sure"
        - Use sparingly and only where it sounds natural
        - Don't overdo it - people don't constantly say "yeah" to voice assistants
        
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
        
        Use natural emotional markers that people actually use with voice assistants:
        - "oh" (realization), "right" (agreement), "yeah" (confirmation)
        - "okay" (acknowledgment), "sure" (agreement)
        - Use sparingly and only where it sounds natural
        - Avoid forced emotions like "seriously", "ugh", "wow" unless contextually appropriate
        
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
        
        Spell out names, usernames, or complex terms that might be misunderstood:
        - GitHub usernames or repository names: "JohnDoe" → "JohnDoe, thats spelt J-O-H-N-D-O-E, JohnDoe¬"
        - Complex names: "TechCorp" → "T-E-C-H-C-O-R-P, TechCorp"
        - Technical terms: "api-client" → "api dash client"
        - Any name that's not immediately obvious how to pronounce
        
        Examples:
        - "User123/repo" → "U-S-E-R-1-2-3 slash repo, User123 slash repo"
        - "my-app/client" → "my dash app slash client"
        - "TechCorp" → "T-E-C-H-C-O-R-P, TechCorp"
        
        Be aggressive about spelling out names - people do this when talking to voice assistants to avoid confusion.
        
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
        
        Say numbers or alphanumerics as a real person would, e.g., "seventy-eight ninety" for 7890, "twenty-twenty" for 2020
        ONLY convert actual numbers, addresses, or alphanumeric identifiers. Do NOT convert words like "Fahrenheit", "Celsius", etc.
        
        Examples:
        - "7890" → "seventy-eight ninety"
        - "2020" → "twenty-twenty" 
        - "221B" → "two twenty-one B"
        - "600 seconds" → "ten minutes"
        - "Fahrenheit" → "Fahrenheit" (keep as is)
        
        Input: "{text}"
        Output:
        """
        result = self._call_openai(prompt)
        if not result.strip():
            return text
        return result

    def apply_detail_dropping(self, text: str, intensity: str) -> str:
        intensity_prompts = {
            "light": "Drop 1 unnecessary detail naturally",
            "moderate": "Drop 1-2 unnecessary details naturally",
            "heavy": "Drop 2-3 unnecessary details naturally"
        }
        
        prompt = f"""
        {INSTRUCTION_TEMPLATE.format(feature='detail dropping')}
        {intensity_prompts[intensity]}
        
        Drop unnecessary formal details that people skip when speaking:
        - State abbreviations: "CA", "NY", "TX" → just skip it when you say an address
        - Formal titles: "Dr.", "Mr.", "Ms.", "Prof." → skip them
        - Company suffixes: "Inc.", "LLC", "Corp." → skip them
        
        Examples:
        - "Yosemite National Park, Mariposa, CA" → "Yosemite National Park, Mariposa"
        - "Dr. John Smith" → "John Smith"
        - "Apple Inc." → "Apple"
        
        Only drop details that don't change the core meaning and that people naturally skip when speaking.
        
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
        
        # Post-processing: clean up quotes and fix spacing
        current_text = self._post_process_text(current_text)
        
        results["final"] = current_text
        print(f"\n{'='*50}")
        print(f"FINAL RESULT: {current_text}")
        print(f"{'='*50}")
        return results

    def _post_process_text(self, text: str) -> str:
        """Clean up the final text by removing quotes and fixing spacing."""
        # Remove surrounding quotes if they exist
        text = text.strip()
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
        if text.startswith("'") and text.endswith("'"):
            text = text[1:-1]
        
        # Fix spacing around punctuation
        text = re.sub(r'\s+([,.!?])', r'\1', text)  # Remove spaces before punctuation
        text = re.sub(r'([,.!?])\s*([,.!?])', r'\1\2', text)  # Fix double punctuation
        
        # Fix spacing around dashes and slashes
        text = re.sub(r'\s*-\s*', '-', text)  # Remove spaces around single dashes
        text = re.sub(r'\s*/\s*', '/', text)  # Remove spaces around slashes
        
        # Fix multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()


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
        max_features=8,
        confidence_threshold=0.5,
        temperature=0.8,
        max_retries=3,
        retry_delay=1.0, 
        asr=True
    )
    pipeline = GranularSpeechPipeline(config)
    data_path = "../data/BFCL_v3_live_simple.json"
    bfcl_data = load_bfcl_data(data_path)
    print(f"Loaded {len(bfcl_data)} test cases from {data_path}")
    
    # Filter for English texts only
    english_test_cases = []
    for test_case in bfcl_data:
        user_content = test_case['question'][0][0]['content']
        if pipeline.is_english_text(user_content):
            english_test_cases.append(test_case)
        else:
            print(f"Skipping non-English text: {user_content[:100]}...")
    
    print(f"Found {len(english_test_cases)} English test cases out of {len(bfcl_data)} total")
    
    test_subset = english_test_cases[:25] #you can edit the number of test cases here will add a terminal argument support later
    if len(test_subset) == 0:
        print("No English test cases found. Exiting.")
        return
    
    print(f"Processing {len(test_subset)} English test cases...")
    
    results = []
    for i, test_case in enumerate(test_subset):
        print(f"\n{'='*60}")
        print(f"Test case {i+1}/{len(test_subset)}")
        user_content = test_case['question'][0][0]['content']
        print(f"Original text: {user_content}")
        
        try:
            result = pipeline.transform_text(user_content)
            print(f"Transformed text: {result['final']}")
            results.append({
                'original': user_content,
                'transformed': result['final'],
                'features_applied': result['selected_features'],
                'confidence_scores': [f['confidence'] for f in result['selected_features']]
            })
        except Exception as e:
            print(f"Error processing test case {i+1}: {e}")
            continue
    
    # Save results
    output_file = "BFCL_v3_live_simple_granular_spoken.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_file}")
    print(f"Successfully processed {len(results)} English test cases")

if __name__ == "__main__":
    main()