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
from bfcl_eval.model_handler.utils import combine_consecutive_user_prompts

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-4o"

INSTRUCTION_TEMPLATE = "You are a clean text to spoken dialogue translation engine gear, and your specific job is to rewrite the input status applying {feature} to the input text. Just rewrite the input, adding the noise feature, preserving all essential information, just spoken out aloud. CRITICAL: NEVER modify content within quotes - keep quoted text EXACTLY as it is, including punctuation and spacing. CRITICAL: NEVER modify access tokens, passwords, or key function arguments - if you need to spell them out, add formatting info like 'access token abc one two three x y z, all lowercase, no spaces'."

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
    max_features: int = 6
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
    "label_naturalization",
    "detail_dropping",
    "word_reordering",
    "simplified_verbs",
    "filepath_verbalization"
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

        CRITICAL RULE: BE CONSERVATIVE: If input is already speech-like, apply minimal changes
        CRITICAL: NEVER modify, remove, or paraphrase away links/URLs, file paths, or similar technical references. Always preserve them in the output, either verbalized or as-is.

        PRIORITY FEATURES (select 4-6 from these):
        - contractions: Use wanna, gonna, lemme, gimme, etc.
        - simplified_verbs: get instead of retrieve, check instead of verify
        - casual_pronouns: ya, em, imma
        - slang_terms: grab, check out, look up
        - disfluencies: Filler words (um, uh, like), hesitations, natural pauses
        - thinking_aloud: Express thinking or searching for words, e.g., 'let me see'
        - false_starts: Start to say something, then restart, e.g., 'I want to—wait, can you...'
        - self_corrections: Correct oneself with an actual correction, e.g., 'the file... no, the folder'
        - emotional_markers: Emotion, attitude, or conversational cues (okay, right, yeah, oh, wow, seriously, etc.)
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

        ULTRA-CONSERVATIVE RULE: If input is already direct and speech-like, apply MAXIMUM 2-3 features
        Examples of already speech-like inputs that need minimal changes:
        - "Reverse say hi" → apply 1-2 features max
        - "Play music" → apply 1 feature
        - "Get weather" → apply 1-2 features max
        - "Turn on lights" → apply 1-2 features max
        - "Call mom" → apply 1-2 features max
        
        For already speech-like inputs, prefer these minimal features:
        - contractions (light)
        - disfluencies 

        FEATURE GROUPINGS - You MUST select at least 1 feature from each group:

        GROUP 1 - BASIC SPEECH PATTERNS (always include 1-2):
        - contractions: Use wanna, gonna, lemme, gimme, etc.
        - simplified_verbs: get instead of retrieve, check instead of verify
        - casual_pronouns: ya, em, imma
        - slang_terms: grab, check out, look up

        GROUP 2 - DISFLUENCIES & HESITATIONS (always include 1-2):
        - disfluencies: Filler words (um, uh, like), hesitations, natural pauses (INCREASE FREQUENCY - 60% of cases should have disfluencies)
        - thinking_aloud: Express thinking or searching for words, e.g., 'let me see'
        - false_starts: Start to say something, then restart, e.g., 'I want to—wait, can you...'
        - self_corrections: Correct oneself with an actual correction, e.g., 'the file... no, the folder'

        GROUP 3 - CONVERSATIONAL MARKERS (include 1):
        - emotional_markers: Emotion or attitude (oh, right, seriously)
        - repetitions: Repeat words or phrases naturally

        GROUP 4 - STRUCTURAL CHANGES (include 1-2):
        - word_reordering: Natural word order changes
        - article_dropping: Drop articles where natural
        - preposition_dropping: Drop prepositions where natural
        - subject_dropping: Drop subjects where natural
        - detail_dropping: Drop unnecessary formal details (state in the address, titles, etc.)

        GROUP 5 - CONTENT MODIFICATIONS (include 1-2):
        - spelling_noise: Spell out names, usernames, or complex terms that might be misunderstood (ALWAYS include for names, usernames, complex terms)
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
        - word_reordering: Slightly reorder words naturally
        - vague_references: Use that thing, the stuff, some info
        - approximate_quantifiers: like 10 minutes when something like 600 seconds is used, around 5 files
        - simplified_verbs: get instead of retrieve, check instead of verify
        - detail_dropping: Drop unnecessary formal details (state abbreviations, titles, company suffixes)
        - label_naturalization: When a colon is used to introduce a label, field, or quoted value, insert a natural filler like 'named', 'called', or 'titled' after the colon or before the value to make the speech sound more natural. For example, "Create a to-do: 'Go for shopping at 9 PM.'" should become "Create a to-do named 'Go for shopping at 9 PM.'". Only do this when it improves the conversational flow.
        - filepath_verbalization: For any file path, verbalize it in the most natural and context-appropriate way. For short/simple paths (like 'image.png' or 'vikhyatk/moondream2'), spell out or say as-is. For long/complex paths (like 'd:/playground/pc_contoller/env/Scripts/python.exe'), use a hierarchical, natural spoken description (e.g., 'in the Scripts folder inside env, inside pc_controller, inside playground on the D drive, the file python dot exe'). CRITICAL: NEVER combine this with symbol_pronunciation or spelling_noise for the same file path. Only verbalize file paths ONCE, using the most natural method. Avoid duplication and messy outputs with any other verbalization selected.

        EXAMPLES BY INPUT TYPE:

        CALENDAR SCHEDULING:
        Written: "I would like to schedule a meeting with the marketing team for next Tuesday at 2:30 PM in the conference room, and could you please send out calendar invitations to all participants?"
        Spoken: "Schedule a meeting with marketing Tuesday at two-thirty. Send invites to everyone."
        Features: disfluencies, numbers_noise, contractions, simplified_verbs, emotional_markers

        DOCUMENT EDITING:
        Written: "Please modify the quarterly report document by adding the financial data from Q3 and removing the outdated statistics from the previous version."
        Spoken: "Update the quarterly report with Q3 data. Remove the old stats."
        Features: disfluencies, simplified_verbs, thinking_aloud

        MUSIC PLAYBACK:
        Written: "I would like to play the album 'Midnight Dreams' by the artist 'Stellar Echo' and set the volume to 75% while enabling shuffle mode."
        Spoken: "Play Midnight Dreams by Stellar Echo. That's S-T-E-L-L-A-R E-C-H-O. Volume at seventy-five. Turn shuffle on."
        Features: numbers_noise, simplified_verbs, disfluencies, spelling_noise

        EMAIL COMPOSITION:
        Written: "Please compose a new email message addressed to john.smith@company.com with the subject line 'Project Update - Phase 2 Completion' and include the following content in the body."
        Spoken: "Write an email to john dot smith at company dot com. Subject is Project Update Phase 2. Add the content."
        Features: symbol_pronunciation, simplified_verbs, disfluencies, contractions, casual_pronouns

        ACCESS TOKEN EXAMPLE:
        Written: "Utilizing my access token 'access_token_abc123', I'll cap my budget at 2000 USD for the impending journey."
        Spoken: "Use access token abc one two three, all lowercase, no spaces. Set budget at two thousand dollars for the trip."
        Features: numbers_noise, simplified_verbs, disfluencies, contractions

        QUOTED CONTENT EXAMPLE:
        Written: "Post a status update with the message 'Just filled up the tank and checked the tire pressures. Ready for the next adventure!'"
        Spoken: "Post a status with the message 'Just filled up the tank and checked the tire pressures. Ready for the next adventure!'"
        Features: simplified_verbs, disfluencies, contractions

        FILE MANAGEMENT:
        Written: "I would like to access the quarterly report document located in the shared drive folder and create a backup copy in my personal directory."
        Spoken: "Get the quarterly report from shared drive. Make a backup in my directory."
        Features: disfluencies, self_corrections, simplified_verbs, thinking_aloud

        SOCIAL MEDIA:
        Written: "Please post a status update on my social media account with the message 'Excited to announce our new product launch!' and include the hashtag #innovation."
        Spoken: "Post a status excited to announce our new product launch. Add hashtag innovation."
        Features: disfluencies, simplified_verbs, emotional_markers

        SELECTION GUIDELINES:
        - sentence_restructuring is ALWAYS applied automatically (don't select it)
        - ALWAYS include numbers_noise if input has numbers/alphanumerics
        - ALWAYS include spelling_noise if input has names, usernames, or complex terms (60% of cases should have spelling_noise)
        - DO NOT include spelling_noise for common English words, standard terms
        - You MUST select at least 1 feature from each of the 5 groups above
        - Total of 4-6 features maximum (sentence_restructuring + 3-5 others)
        - If the input is very short (e.g., 1-3 words) or a simple direct command, apply zero or at most one minimal feature (preferably none). Example: 'Order pizza.'
        - For direct, confident, or blunt commands where the user clearly knows what they want, do NOT add disfluencies (e.g., 'uh', 'um') in most cases. Only a small proportion (e.g., 30%) of such commands should have any disfluency, and the majority (70%) should be clean and direct. Example: 'Turn off the lights.' should usually remain without disfluencies.
        - INCREASE disfluencies frequency - 60% of cases should have disfluencies (um, uh, like, pauses ...)
        - Focus on making it sound direct and efficient, not overly polite or formal
        - Choose features that make speech sound natural, not robotic or forced
        - Prioritize features that improve flow and naturalness over adding complexity
        - Focus on natural conversational flow rather than artificial speech patterns
        - Ensure diversity across groups - don't stack too many features from the same group
        - IMPORTANT: Only apply changes when needed. If the input is already speech-like, apply minimal changes
        - If the input is very short or a super simple direct command (like 'Order pizza', 'Play music', 'Get weather'), apply zero or at most one minimal feature (preferably none). Do not add emotional markers or unnecessary changes to these utterances. Leave them as-is unless absolutely necessary.
        - Preserve all function-calling content completely unchanged
        - CRITICAL: NEVER modify content within quotes - keep quoted text EXACTLY as it is, including punctuation and spacing
        - CRITICAL: NEVER modify access tokens, passwords, or key information that could act as arguments for downstream function calling task - if you need to spell them out, add formatting info like "access token abc one two three x y z, all lowercase, no spaces"
        - Use proper English - some informal is okay but maintain good grammar
        - DO NOT add unnecessary filler like "let me check" after commands
        - DO NOT make technical terms vague - keep them specific and clear
        - Do NOT select both disfluencies and emotional_markers for the same utterance unless it is extremely natural. In most cases, only one conversational marker (like "uh", "oh", "okay", etc.) should appear at the start of a sentence. If the input already sounds hesitant or emotional, do not add another marker.
        - Be very conservative with conversational and emotional markers—avoid making the speech sound overly hesitant or artificial/comical by stacking multiple markers.
        - Select label_naturalization whenever a label, event, or item is referenced by name (especially after a colon, in quotes, or with a unique identifier). For example: 'move the event named Alice-One-one-One...'.

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
            
            return features[:6] #first x features being taken
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
        You are converting written instructions into natural spoken dialogue. Your job is to restructure the input to sound like someone actually speaking to a voice assistant, not reading written text.

        {intensity_prompts[intensity]}
        
        CRITICAL:
        - Do NOT change the meaning of the sentence. If in doubt, do less restructuring.
        - Do not remove the context of any order, request, or command if it is essential for the meaning.
        - NEVER modify, remove, or paraphrase away links/URLs, file paths, or similar technical references. Always preserve them in the output, either verbalized or as-is.

        GOOD Examples:
        - Original: "I would like to schedule a meeting with the marketing team for next Tuesday at 2:30 PM in the conference room, and could you please send out calendar invitations to all participants?"
          Spoken: "Schedule a meeting with marketing Tuesday at two-thirty. Send invites to everyone."
        - Original: "Please modify the quarterly report document by adding the financial data from Q3 and removing the outdated statistics from the previous version."
          Spoken: "Update the quarterly report with Q3 data. Remove the old stats."
        - Original: "add todo with content go to sleep at 9 pm"
          Spoken: "Uh, add a to-do: sleep at nine PM."
        - Original: "Could you help me classify the following customer queries into the appropriate categories?"
          Spoken: "Sure, could you sort these customer questions into categories: ..."

        BAD Examples (DO NOT DO):
        - Original: "I'd like to modify my order by making the dish called 'chicken dish' to a new spice level 'extra spicy', please."
          Bad: "Change the chicken dish to extra spicy." (Loses the 'order' context)
        - Original: "I have a list of numerical values: [2.5, 3.6, 4.1, 5.2], and I need to apply normalization to them."
          Bad: "Make these numbers normal: two point five, three point six, four point one, five point two." (Vague, loses the technical meaning of 'normalize')
        
        Input: "{text}"
        Output:
        """
        result = self._call_openai(prompt)
        if not result.strip():
            return text
        return result

    def apply_disfluencies(self, text: str, intensity: str) -> str:
        intensity_prompts = {
            "light": "Add 1 light disfluency (filler word or hesitation) naturally.",
            "moderate": "Add 1 light disfluency (filler word or hesitation) naturally.",
            "heavy": "Add 1 light disfluency (filler word or hesitation) naturally."
        }
        prompt = f"""
        {INSTRUCTION_TEMPLATE.format(feature='disfluencies')}
        {intensity_prompts[intensity]}
        
        Add disfluencies to make it sound like real speech:
        - Filler words: "um", "uh", "like", "you know", "I mean"
        - Hesitations: trailing off ("I..."), incomplete thoughts, natural pauses
        - Use naturally - people do use disfluencies when speaking to voice assistants
        - 1 disfluency per sentence is normal - sometimes 2 is okay if it sounds natural
        - Do NOT overdo it - this should sound natural, not like someone struggling to speak
        - Place them where people naturally hesitate (before important words, when thinking, not just randomly and definitely not at the end of the sentence)
        - It is natural to use "um" at the start of a sentence when the speaker is thinking, hesitating, or formulating a question or complex command.
        - Only place "um" at the start if the sentence is a question, a complex/uncertain request, or if the speaker is likely to be thinking.
        - Avoid stacking "um" with "oh" or "uh" at the start.
        - For direct, confident commands, avoid "um" at the start.
        - Prefer to insert "uh" instead of "um" between different ideas or clauses, especially where the speaker might pause to think.
        
        GOOD Examples:
        - "Get me... a Comfort Uber from twenty-twenty Addison Street"
        - "Um, what's the weather in Tel Aviv?" 
        - "What's the weather in... Tel Aviv?"
        - "I need to... get the details for user ID seventy-eight ninety"
        - "Um, get user details for ID seventy-eight ninety with black as special request."
        - "Uh, change my profile with a new email, uh, john dot doe at example dot com, and, um, age, thirty. User ID's one two three four five, one two three four five." (listing multiple changes, and might naturally hesitate)
        - "Order five burgers and, um, six chicken wings from Uber Pitada, that's U-B-E-R P-I-T-A-D-A, Uber Pitada." (perfect placement between 2 things trying to recall / think next item)



        BAD Examples (DO NOT DO):
        - "Um, uh, what's the weather in Tel Aviv?" (clustering at the start is bad)
        - "Oh, uh, what's the weather in Tel Aviv?" (clustering at the start is bad)
        - "What's the, uh, weather today in Boston in Fahrenheit?" (poorly placed: The user doesnt need to think this is not that complex a request and it is not between two ideas)
        - "Uh, just grab me a Whopper, ya know, burger." (uncomfortable, awkward, unnatural with poor flow of speech)
        
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
        
        Real speech often uses contractions. Make it sound casual and natural but dont over do it especially if it impacts meaning or ruins flow of speech
        
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

        Use slang terms like: "grab", "check out", "look up" ONLY if the request is for something casual, like reminders, food, or informal tasks. 
        DO NOT use slang for technical, search, or music requests, or when the request is for a specific named entity (like a song, app, or person).
        If in doubt, do not apply slang.

        CRITICAL: If the request is to search, play, or find a song, app, or specific item, do NOT use slang like 'grab' or 'check out'—just keep the request direct and natural.

        Examples of GOOD slang usage:
        - "Remind me to buy lunch with a friend today" → "Remind me to grab lunch with my mate today"
        - "Check out this new app" → "Check out this cool app"

        Examples of BAD slang usage (DO NOT DO):
        - "Search for 'Baby Shark'" → "Grab 'Baby Shark'" (bad)
        - "Find the weather" → "Grab the weather" (bad)
        - "Play 'Despacito'" → "Grab 'Despacito'" (bad)

        Input: "{text}"
        Output:
        """
        result = self._call_openai(prompt)
        if not result.strip():
            return text
        return result

    def apply_symbol_pronunciation(self, text: str, intensity: str) -> str:
        intensity_prompts = {
            "light": "Pronounce 1-2 symbols (like slash, dash, dot, at) in a natural way",
            "moderate": "Pronounce several symbols (like slash, dash, dot, at) in a natural way",
            "high": "Pronounce many symbols (like slash, dash, dot, at) in a natural way"
        }
        
        prompt = f"""
        {INSTRUCTION_TEMPLATE.format(feature='symbol pronunciation')}
        {intensity_prompts[intensity]}
        
        Use symbol pronunciations like: "slash", "dash", "at"
        Keep the meaning intact.
        
        IMPORTANT:
        - For decimal numbers, always say "point" (e.g., "two point five"), never "dot".
        - Use "dot" only for URLs, emails, or file names (e.g., "gmail dot com", "file dot txt").
        - For links/URLs: If the link is short (e.g., a YouTube link), spelling it out is acceptable. If the link is long, do NOT spell it out—just include it as-is. NEVER double spell or verbalize a link if another feature has already done so.
        - For other symbols, use "slash", "dash", etc., only when it makes sense in context.
        
        GOOD Examples:
        - "2.5" → "two point five"
        - "3.6" → "three point six"
        - "gmail.com" → "gmail dot com"
        - "file.txt" → "file dot txt"
        - 'using this link: h t t p s colon slash slash w w w dot youtube dot com slash watch question mark v equals d Q w four w nine W g X c Q.'
        
        BAD Examples (DO NOT DO):
        - "2.5" → "two dot five" (wrong for numbers)
        - "3.6" → "three dot six" (wrong for numbers)
        - "gmail.com" → "gmail point com" (wrong for URLs)
        - "file.txt" → "file point txt" (wrong for file names)
        - 'using this link: h t t p s colon slash slash w w w dot example dot com slash a slash very slash long slash path slash with slash lots slash of slash segments question mark query equals long.' (do NOT spell out long links)
        - 'using this link: h t t p s colon slash slash w w w dot youtube dot com slash watch question mark v equals d Q w four w nine W g X c Q. h t t p s colon slash slash w w w dot youtube dot com slash watch question mark v equals d Q w four w nine W g X c Q.' (never double spell)
        
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
        
        CRITICAL GRAMMAR RULES:
        - Maintain proper English grammar and sentence structure
        - Do NOT create ungrammatical sentences
        - Only drop articles where it sounds natural and doesn't break flow
        - If dropping an article makes the sentence unclear, do NOT apply the change
        - Output must be proper, understandable English
        
        Examples of GOOD article dropping:
        - "Show me the C drive" → "Show me C drive"
        - "Get the weather" → "Get weather"
        
        Examples of BAD article dropping (DO NOT DO):
        - "Let me do a a quick check" → "Let me quick check" (if it sounds unnatural or breaks sentence)
        
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

        CRITICAL GUIDELINES:
        - Only drop prepositions if the sentence remains clear, natural, and grammatically correct.
        - Do NOT drop prepositions if it makes the sentence sound awkward, incomplete, or confusing.
        - Do NOT drop prepositions that are required for location, time, or other essential information (e.g., "in the kitchen", "at 5pm", "on the table").
        - If dropping a preposition makes the sentence ambiguous or changes the meaning, do NOT apply the change.
        - If in doubt, do NOT drop the preposition.
        - Never drop prepositions from common phrases or idioms where they are required for natural English.

        Examples of GOOD preposition dropping:
        - "Get the weather in New York" → "Get weather in New York"
        - "Check the status on my order" → "Check status on my order"
        - "Turn on the lights in the living room" → "Turn on lights in the living room"


        Examples of BAD preposition dropping (DO NOT DO):
        - "Play music in the kitchen" → "Play music kitchen" (bad)
        - "Set alarm for 7am" → "Set alarm 7am" (bad)
        - "Put it on the table" → "Put it table" (bad)
        - "Meet me at the park" → "Meet me park" (bad)
        - "I'll be in San Ramon and want to go to a salon." →  "look up salon in San Ramon." (dont remove the a, CHANGES SHOULD NOT IMPACT MEANING)


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
        
        CRITICAL GRAMMAR RULES:
        - Maintain proper English grammar and sentence structure
        - Do NOT create ungrammatical or incomplete sentences
        - Do NOT change the meaning of conditional clauses
        - Do NOT drop subjects from complex sentences where it breaks flow
        - Only drop subjects where it sounds natural and doesn't break flow
        - If dropping the subject makes the sentence unclear, do NOT apply the change
        - Output must be proper, understandable English
        
        Examples of GOOD subject dropping:
        - "I can't answer that" → "Can't answer that"
        - "I need to check" → "Need to check"
        - "I want to go" → "Want to go"
        
        Examples of BAD subject dropping (DO NOT DO):
        - "if a user asks a question" → "if asked question" (changes meaning and breaks grammar)
        - "when the system starts" → "when start" (breaks grammar)
        - "if you need help" → "if help" (breaks grammar)
        - "while the process runs" → "while runs" (breaks grammar)
        
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
        
        CRITICAL GRAMMAR RULES:
        - Maintain proper English grammar and sentence structure
        - Do NOT break subject-verb agreement
        - Do NOT create ungrammatical sentences
        - If reordering breaks grammar, do NOT apply the change
        - Output must be proper, understandable English
        - DO NOT REORDER IN A WAY THAT BREAKS THE FLOW OF THE SENTENCE AND SOUNDS UNNATURAL
        
        IMPORTANT: Do NOT change any numbers or alphanumeric identifiers that are already in spoken form (like "seventy-eight ninety", "twenty-twenty", etc.). Keep them exactly as they are.

        
        Examples of BAD reordering (DO NOT DO):
        - "Logistic regression wasn't mentioned" → "Logistic regression didn't mention" (breaks grammar)
        - "I can't answer" → "Can't answer that, I" (incomplete)
        - "I can't answer that" → "Can't answer that, I can't" (repetition)
        - "Logistic regression isn't mentioned" → "Isn't mentioned, logistic regression" (weird flow, incorrect meaning and grammar)
        - "Oh, find a reasonably-priced coffeehouse in New York." → "Find in New York, oh, a reasonably-priced coffeehouse."

        
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
        
        CRITICAL RULES:
        - Do NOT make technical terms, names, or specific content vague
        - Do NOT make important specifications or requirements vague
        - Keep specific terms like "ATM location", "customer query", "API endpoint" as they are key to the user instruction
        - Only use vague references for general concepts, not specific technical content
        - Preserve all function-calling content completely unchanged
        
        Examples of GOOD vague references:
        - "Get the stuff from the folder" (good - folder is general)
        - "Check that thing in the settings" (good - settings is general)
        
        Examples of BAD vague references (DO NOT DO):
        - "Where do I put that ATM location thing?" (bad - ATM location is specific technical content)
        - "Classify that customer question thing" (bad - customer query is specific technical content)
        - "Check that API endpoint thing" (bad - API endpoint is specific technical content)
        - "Get that weather thing in Fahrenheit" (bad - weather in Fahrenheit is specific requirement)
        
        Keep the meaning intact and preserve all specific key content.
        
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
        
        CRITICAL:
        - Only simplify verbs if it does NOT change the core meaning or make the sentence awkward.
        - Never replace 'make' or 'change' with 'do' unless it is natural in spoken English.
        - Do NOT use 'do' for food orders or modifications (e.g., never say 'do the chicken dish extra spicy').
        - If in doubt, keep the original verb.

        GOOD Examples:
        - "retrieve" → "get"
        - "verify" → "check"
        - "purchase" → "buy"
        - "classify" → "sort" (if context is casual and meaning is preserved)

        BAD Examples (DO NOT DO):
        - "make the chicken dish extra spicy" → "do the chicken dish extra spicy" (unnatural)
        - "change the chicken dish to extra spicy" → "do the chicken dish extra spicy" (unnatural)
        - "normalize these numbers" → "make these numbers normal" (vague, loses technical meaning)
        - "classify these queries" → "sort these queries" (if context requires technical precision)
        
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
        - DO NOT add thinking phrases after commands - only before or during the command
        
        Examples:
        - "Let me see... get the details for user ID seventy-eight ninety"
        - "What's the weather in Tel Aviv? I think... in Fahrenheit"
        
        BAD EXAMPLES (DO NOT DO):
        - "Get the weather in Tel Aviv. Let me see..." (don't add after command)
        - "List C drive. Let me check..." (don't add after command)
        
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
        - "okay" (acknowledgment), "sure" (agreement), "thanks" (at the end)
        - Use sparingly and only where it sounds natural
        - Avoid forced emotions like "seriously", "ugh", "wow" unless contextually appropriate
        - Be careful at what part of the sentence you add the emotional marker...DO NOT BREAK THE FLOW OF THE SENTENCE. EXAMPLE: DO NOT ADD OKAY TO THE END OF SENTENCES
        - Don't overdo it—people don't constantly use these markers with voice assistants.
        - Do NOT remove or replace existing disfluencies (like "uh", "um", "er", "like") when adding emotional/conversational markers.
        - Only use 'oh' when the speaker is expressing realization, surprise, or correcting themselves.
        - Do NOT use 'oh' as a generic sentence starter for commands or direct requests.
        - Avoid stacking 'oh' and any other emotional markers with disfluencies (e.g., do not use 'Oh, uh,' or 'Um, oh,').
        - If the sentence is a direct command, do not add 'oh' at the start.
        
        Examples of GOOD usage:
        - "Okay, find a coffeehouse in New York."
        - "Oh, I forgot to mention, add milk."
        - "Right, play some music."
        - "Find me some movies with Brad Pitt in them, thanks."
        - "You know, I could use a coffee right now. Find a coffeehouse in New York."
        
        Examples of BAD usage (DO NOT DO):
        - "Find a coffeehouse in New York, okay."
        - "Play some music, right."
        - "Add milk to my list, sure"
        - "Find a coffeehouse in New York, you know."
        - "Oh, order pizza." (bad, simple request no realization)
        - "Oh, uh, what's the weather?" (bad, simple request no realization and stacking emotional markers with disfluencies)
        
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
            "light": "Spell out 1 truly ambiguous or complex term that might be misunderstood",
            "moderate": "Spell out 1-2 truly ambiguous or complex terms that might be misunderstood",
            "heavy": "Spell out 2-3 truly ambiguous or complex terms that might be misunderstood"
        }
        
        prompt = f"""
        {INSTRUCTION_TEMPLATE.format(feature='spelling noise')}
        {intensity_prompts[intensity]}
        
        Spell out names, usernames, or complex terms that might be misunderstood:
        - Personal names: "Montgomery" → "Montgomery, that's M-O-N-T-G-O-M-E-R-Y, Montgomery"
        - GitHub usernames or repository names: "JohnDoe" → "JohnDoe, that's J-O-H-N-D-O-E, JohnDoe"
        - Complex names: "TechCorp" → "T-E-C-H-C-O-R-P, TechCorp"
        - Any name that's not immediately obvious how to pronounce or spell
        
        Examples:
        - "Elizabeth Montgomery" → "Elizabeth Montgomery, that's M-O-N-T-G-O-M-E-R-Y, Elizabeth Montgomery"
        - "User123/repo" → "U-S-E-R-1-2-3 slash repo, User123 slash repo"
        - "my-app/client" → "my dash app slash client"
        - "TechCorp" → "T-E-C-H-C-O-R-P, TechCorp"
        
        DO NOT SPELL OUT:
        - Common English words: "user", "data", "project", "database", "connection"
        - Very simple names: "John", "Mary", "Smith" (only if they're clearly simple)
        - Standard technical terms: "Postgres", "MySQL", "HTTP", "JSON"
        - Obvious abbreviations: "DB" (database), "API" (in most contexts)
        - Numbers or simple alphanumeric: "12345", "user1"
        
        CRITICAL: Be more aggressive with spelling out names and complex terms. If in doubt, spell it out.
        
        Input: "{text}"
        Output:
        """
        result = self._call_openai(prompt)
        if not result.strip():
            return text
        return result

    def apply_numbers_noise(self, text: str, intensity: str) -> str:
        intensity_prompts = {
            "light": "Add a small amount of number noise (e.g., say numbers in a more casual or spoken way)",
            "moderate": "Add moderate number noise (e.g., say numbers in a more casual or spoken way, or spell out a few digits)",
            "high": "Add heavy number noise (e.g., say numbers in a more casual or spoken way, or spell out several digits)"
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
        
                
        IF the input contains a link/URL:
        - If the link/URL is short (e.g., a YouTube link), spelling it out is acceptable. Example: 'h t t p s colon slash slash w w w dot youtube dot com slash watch question mark v equals d Q w four w nine W g X c Q.'
        - If the link/URL is long (many path segments or long query), do NOT spell it out—just include it as-is in the output. Example: 'https://www.example.com/very/long/path/with/lots/of/segments?query=long'.
        - NEVER spell out or verbalize a link/URL if another feature has already done so (avoid double spelling/verbalization).
        GOOD Example:
        - 'start h t t p s colon slash slash w w w dot youtube dot com slash watch question mark v equals d Q w four w nine W g X c Q.'
        BAD Example:
        - 'start h t t p s colon slash slash w w w dot example dot com slash a slash very slash long slash path slash with slash lots slash of slash segments question mark query equals long.' (do NOT spell out long links)
        - 'start h t t p s colon slash slash w w w dot youtube dot com slash watch question mark v equals d Q w four w nine W g X c Q. h t t p s colon slash slash w w w dot youtube dot com slash watch question mark v equals d Q w four w nine W g X c Q.' (never double spell)
        
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
        
        Drop ONLY unnecessary formal/administrative details that people skip when speaking:
        - State/province names when city is clear: "New York, NY" → "New York"
        - Formal titles: "Dr.", "Mr.", "Ms.", "Prof." → skip them
        - Company suffixes: "Inc.", "LLC", "Corp." → skip them
        
        CRITICAL: NEVER drop, modify, or paraphrase away links/URLs, file paths, or similar technical references. Always preserve them in the output, either verbalized or as-is.
        
        CRITICAL: NEVER drop important technical specifications or units:
        - Temperature units: "Fahrenheit", "Celsius", "Kelvin" → KEEP THESE
        - Measurement units: "miles", "kilometers", "pounds" → KEEP THESE
        - Technical terms: "API", "JSON", "HTTP" → KEEP THESE
        - Specific requirements: "in Fahrenheit", "in Celsius" → KEEP THESE
        - ANY unit specification: "in fahrenheit", "in celsius", "in miles" → KEEP THESE
        
        Examples of GOOD detail dropping:
        - "Yosemite National Park, Mariposa, CA" → "Yosemite National Park, Mariposa"
        - "Dr. John Smith" → "John Smith"
        - "Apple Inc." → "Apple"
        - "123 Main Street" → "123 Main"
        
        Examples of BAD detail dropping (DO NOT DO):
        - "length in Centimeters" → "length" (removes important unit specification)
        - "API endpoint" → "endpoint" (removes important technical context)
        
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
                # Post-process after each feature application
                current_text = self._post_process_text(current_text)
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

    def transform_dataset_for_clean_output(self, test_cases, sample_count=None):
        """
        Processes a list of test cases, applying the pipeline to each user turn with role 'user', and returns a new list
        with only the original structure and 'transcript' added to user turns. Removes asr and speech_features.
        If sample_count is provided, only that many random samples are processed.
        If a turn group contains multiple user turns, they are combined into one before transformation.
        """
        import random
        if sample_count is not None:
            test_cases = random.sample(test_cases, min(sample_count, len(test_cases)))
        processed_results = []
        for test_case in test_cases:
            new_test_case = test_case.copy()
            new_test_case['question'] = []
            for turn_group in test_case['question']:
                # Combine consecutive user prompts if more than one user in the group
                user_count = sum(1 for turn in turn_group if turn.get('role') == 'user')
                new_turn_group = turn_group
                if user_count > 1:
                    new_turn_group = combine_consecutive_user_prompts(turn_group)
                # Now process as before
                processed_turn_group = []
                for turn in new_turn_group:
                    new_turn = turn.copy()
                    if new_turn.get('role') == 'user':
                        result = self.transform_text(new_turn['content'])
                        new_turn['transcript'] = result['final']
                        new_turn.pop('asr', None)
                        new_turn.pop('speech_features', None)
                    processed_turn_group.append(new_turn)
                new_test_case['question'].append(processed_turn_group)
            processed_results.append(new_test_case)
        return processed_results

    def _post_process_text(self, text: str) -> str:
        """Clean up the final text by removing quotes and fixing spacing."""
        text = text.strip()
        # Remove all leading/trailing quotes (single or double, even if repeated)
        while (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
            text = text[1:-1].strip()
        # Unescape any escaped quotes
        text = text.replace('\\"', '"').replace("\\'", "'")
        # After unescaping, remove quotes again if present
        if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
            text = text[1:-1].strip()
        # Fix spacing around punctuation
        text = re.sub(r'\s+([,.!?])', r'\1', text)  # Remove spaces before punctuation
        text = re.sub(r'([,.!?])\s*([,.!?])', r'\1\2', text)  # Fix double punctuation
        # Fix spacing around dashes and slashes
        text = re.sub(r'\s*-\s*', '-', text)  # Remove spaces around single dashes
        text = re.sub(r'\s*/\s*', '/', text)  # Remove spaces around slashes
        # Fix multiple spaces
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    @staticmethod
    def filter_english_test_cases(pipeline, test_cases):
        """
        Filters out non-English test cases. Handles both multi-turn and single-turn data.
        """
        english_test_cases = []
        for test_case in test_cases:
            is_multi_turn = (
                'question' in test_case and
                isinstance(test_case['question'], list) and
                len(test_case['question']) > 0 and
                isinstance(test_case['question'][0], list)
            )
            if is_multi_turn:
                all_english = True
                for turn in test_case['question']:
                    user_content = turn[0]['content']
                    if not pipeline.is_english_text(user_content):
                        all_english = False
                        break
                if all_english:
                    english_test_cases.append(test_case)
            else:
                user_content = test_case['question'][0][0]['content']
                if pipeline.is_english_text(user_content):
                    english_test_cases.append(test_case)
        return english_test_cases

    def apply_label_naturalization(self, text: str, intensity: str) -> str:
        """
        LLM-based function to insert a natural filler (like 'named', 'called', or 'titled') after a colon or before a quoted/named value to make the speech sound more natural.
        Only do this when a colon is used to introduce a label, field, or quoted value.
        """
        intensity_prompts = {
            "light": "Add a natural filler word (like 'named', 'called', or 'titled') after a colon or before a quoted/named value, only when appropriate.",
            "moderate": "Add a natural filler word (like 'named', 'called', or 'titled') after a colon or before a quoted/named value, only when appropriate.",
            "heavy": "Add a natural filler word (like 'named', 'called', or 'titled') after a colon or before a quoted/named value, only when appropriate."
        }
        prompt = f"""
        You are converting written instructions into natural spoken dialogue. Your job is to insert a natural filler word (like 'named', 'called', or 'titled') after a colon or before a quoted/named value to make the speech sound more natural and conversational. Only do this when a colon is used to introduce a label, field, or quoted value.
        {intensity_prompts[intensity]}
        
        GOOD Examples:
        - "Create a to-do: 'Go for shopping at 9 PM.'" → "Create a to-do named 'Go for shopping at 9 PM.'"
        - "Add a to-do: crash at nine P M." → "Add a to-do called 'crash at nine P M.'"
        - "Draft an email to Andy. Subject: 'Sales Forecast Request'. Message: 'Where's the latest sales forecast?'" → "Draft an email to Andy. Subject named 'Sales Forecast Request'. Message called 'Where's the latest sales forecast?'"
        - "Add a NewsItem: 'Julian is testing one two' to Sitefinity CMS." → "Add a NewsItem titled 'Julian is testing one two' to Sitefinity CMS."
        - "Hey, can you move the event named 'Alice-One-one-One' to November first, twenty twenty-three, for ten P M Central European Summer Time?"
        
        BAD Examples (DO NOT DO):
        - "Create a to-do named: 'Go for shopping at 9 PM.'" (don't add 'named:' after a colon)
        - "Add a to-do called: crash at nine P M." (don't add 'called:' after a colon)
        - "Subject: named 'Sales Forecast Request'." (don't use both colon and 'named')
        - "Create a to-do named called 'Go for shopping at 9 PM.'" (never stack multiple fillers)
        - "Create a to-do: 'named Go for shopping at 9 PM.'" (don't put the filler inside the quotes)
        - "Add a to-do: 'crash at nine P M. called.'" (don't put the filler at the end)
        
        Input: "{text}"
        Output:
        """
        result = self._call_openai(prompt)
        if not result.strip():
            return text
        return result

    def apply_filepath_verbalization(self, text: str, intensity: str) -> str:
        """
        LLM-based function to verbalize file paths in the most natural and context-appropriate way.
        For short/simple paths (like 'image.png' or 'vikhyatk/moondream2'), spell out or say as-is.
        For long/complex paths (like 'd:/playground/pc_contoller/env/Scripts/python.exe'), use a hierarchical, natural spoken description (e.g., 'in the Scripts folder inside env, inside pc_controller, inside playground on the D drive, the file python dot exe').
        CRITICAL: NEVER combine this with symbol_pronunciation or spelling_noise for the same file path. Only verbalize file paths ONCE, using the most natural method. Avoid duplication and messy outputs.
        """
        intensity_prompts = {
            "light": "Verbalize file paths in the most natural way. For short/simple paths, spell out or say as-is. For long/complex paths, use a hierarchical spoken description. Do NOT combine with symbol_pronunciation or spelling_noise. Only verbalize file paths once.",
            "moderate": "Verbalize file paths in the most natural way. For short/simple paths, spell out or say as-is. For long/complex paths, use a hierarchical spoken description. Do NOT combine with symbol_pronunciation or spelling_noise. Only verbalize file paths once, even if there are multiple in the text.",
            "heavy": "Verbalize file paths in the most natural way. For short/simple paths, spell out or say as-is. For long/complex paths, use a hierarchical spoken description. Do NOT combine with symbol_pronunciation or spelling_noise. Only verbalize file paths once, and never repeat or spell out every character for long paths."
        }
        prompt = f"""
        You are converting written instructions into natural spoken dialogue. Your job is to verbalize file paths in the most natural and context-appropriate way.
        {intensity_prompts[intensity]}
        
        CRITICAL:
        - NEVER combine this with symbol_pronunciation or spelling_noise for the same file path.
        - Only verbalize file paths ONCE, using the most natural method.
        - Avoid duplication and messy outputs with any other verbalization selected.
        - NEVER use file path verbalization for links or URLs (e.g., anything starting with http, https, www, or ending in .com, .org, etc.). For links, only verbalize the symbols (slash, dot, colon, etc.) as appropriate, but do NOT use hierarchical or spelling approaches.
        
        GOOD Examples:
        - 'image.png' → 'image dot p n g'
        - 'vikhyatk/moondream2' → 'vikhyatk slash moondream two'
        - 'd:/playground/pc_contoller/env/Scripts/python.exe' → 'in the Scripts folder inside env, inside pc_controller, inside playground on the D drive, the file python dot exe'
        - 'https://example.com/path/file.txt' → 'h t t p s colon slash slash example dot com slash path slash file dot t x t' (symbols only, not hierarchical)
        
        BAD Examples (DO NOT DO):
        - 'd:/playground/pc_contoller/env/Scripts/python.exe' → 'D colon slash playground slash P-C underscore controller slash env slash Scripts slash python dot E-X-E' (spelling out every segment is unnatural)
        - 'd:/playground/pc_contoller/env/Scripts/python.exe' → 'python dot e x e in d colon slash playground slash p c underscore controller slash env slash scripts' (duplication)
        - 'image.png' → 'image dot p n g image dot p n g' (never repeat)
        - 'vikhyatk/moondream2' → 'vikhyatk slash moondream two, that's v-i-k-h-y-a-t-k slash m-o-o-n-d-r-e-a-m two' (don't combine spelling and saying as-is unless the path is truly ambiguous)
        - 'd:/playground/pc_contoller/env/Scripts/python.exe' → 'in the Scripts folder inside env, inside pc_controller, inside playground on the D drive, the file python dot exe, d colon slash playground slash p c underscore controller slash env slash scripts slash python dot e x e' (messy, never combine both)
        - 'https://example.com/path/file.txt' → 'in the file path h t t p s colon slash slash example dot com slash path slash file dot t x t' (never use hierarchical or spelling for links/URLs)
        
        Input: "{text}"
        Output:
        """
        result = self._call_openai(prompt)
        if not result.strip():
            return text
        return result


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
        max_features=6,
        confidence_threshold=0.5,
        temperature=0.8,
        max_retries=3,
        retry_delay=1.0, 
        asr=False  # Set ASR to False as requested
    )
    pipeline = GranularSpeechPipeline(config)
    
    # Use the data file (path is relative to the current working directory)
    data_path = "../../data/BFCL_v3_live_simple.json"
    bfcl_data = load_bfcl_data(data_path)
    print(f"Loaded {len(bfcl_data)} test cases from {data_path}")

    # Detect if this is multi-turn data (auto-detect logic remains)
    is_multi_turn = False
    if bfcl_data and 'question' in bfcl_data[0]:
        first_question = bfcl_data[0]['question']
        if isinstance(first_question, list) and len(first_question) > 0:
            if isinstance(first_question[0], list) and len(first_question[0]) > 1:
                is_multi_turn = True
    
    print(f"Detected data format: {'Multi-turn' if is_multi_turn else 'Simple'}")

    # Filter for English texts only
    english_test_cases = []
    for test_case in bfcl_data:
        if is_multi_turn:
            # Multi-turn: check all turns
            all_english = True
            for turn in test_case['question']:
                user_content = turn[0]['content']
                if not pipeline.is_english_text(user_content):
                    all_english = False
                    print(f"Skipping non-English text: {user_content[:100]}...")
                    break
            if all_english:
                english_test_cases.append(test_case)
        else:
            # Simple: check single turn
            user_content = test_case['question'][0][0]['content']
            if pipeline.is_english_text(user_content):
                english_test_cases.append(test_case)
            else:
                print(f"Skipping non-English text: {user_content[:100]}...")

    print(f"Found {len(english_test_cases)} English test cases out of {len(bfcl_data)} total")

    # Randomly sample 25 test cases
    import random
    random.seed(5)
    test_subset = random.sample(english_test_cases, min(3, len(english_test_cases)))
    if len(test_subset) == 0:
        print("No English test cases found. Exiting.")
        return

    print(f"Processing {len(test_subset)} randomly sampled English test cases...")

    processed_results = []
    for i, test_case in enumerate(test_subset):
        print(f"\n{'='*60}")
        print(f"Test case {i+1}/{len(test_subset)}")
        new_test_case = test_case.copy()
        new_test_case['question'] = []
        for turn_idx, turn in enumerate(test_case['question']):
            user_content = turn[0]['content']
            print(f"  Turn {turn_idx+1}: {user_content}")
            try:
                result = pipeline.transform_text(user_content)
                print(f"    Transformed text: {result['final']}")
                asr_pipeline = ASRErrors()
                asr_result = asr_pipeline.execute_noise(result['final'])
                final_asr_text = asr_result.get("final", result['final'])
                if final_asr_text.startswith('"') and final_asr_text.endswith('"'):
                    final_asr_text = final_asr_text[1:-1]
                final_asr_text = final_asr_text.replace('\\"', '"').replace("\\'", "'")
                print(f"    ASR processed text: {final_asr_text}")
                features_applied = result['selected_features']
                feature_names = [f['name'] for f in features_applied]
                feature_intensities = [f['intensity'] for f in features_applied]
                feature_confidences = [f['confidence'] for f in features_applied]
                new_turn = turn.copy()
                new_turn[0]['transformed_content'] = result['final']
                new_turn[0]['asr'] = final_asr_text
                new_turn[0]['speech_features'] = {
                    'name': feature_names,
                    'intensity': feature_intensities,
                    'confidence': feature_confidences
                }
                new_test_case['question'].append(new_turn)
            except Exception as e:
                print(f"    Error processing turn {turn_idx+1} in test case {i+1}: {e}")
                continue
        processed_results.append(new_test_case)
    output_file = "new_results/BFCL_v3_multi_turn_base_granular_spoken.json"
    import os
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_file}")
    print(f"Successfully processed {len(processed_results)} English test cases")

if __name__ == "__main__":
    main()