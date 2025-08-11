import os
from openai import OpenAI
from audio_calling.TTS.scripts.tts_generator_base import TTSGeneratorBase
import openai
from typing import Optional, Dict, Any

def generate_tts_instruction_from_text(
    transformed_content: str,
    language: str = "en",
    function_name: Optional[str] = None,
    function_description: Optional[str] = None,
    openai_api_key: str = None,
    model: str = "gpt-4o"
) -> str:
    """
    Calls an LLM to generate a 1-2 line instruction for OpenAI TTS, given the text to be spoken and function context.
    The function context is provided to help the LLM understand the purpose of the speech and generate a more natural, context-aware instruction.
    
    Args:
        transformed_content: The text to be spoken
        language: ISO language code for the text
        function_name: Optional function name for context
        function_description: Optional function description for context
        openai_api_key: Optional OpenAI API key
        model: Model to use for instruction generation
        
    Returns:
        A natural-sounding instruction for the TTS model
    """
    prompt = (
        "You are helping to generate instructions for a text-to-speech model. "
        f"Given the following text in {language} that will be spoken out-loud by the TTS model, write a 1-2 line instruction for the TTS model to make the speech sound as natural, conversational, and human-like as possible. "
        "MOST IMPORTANTLY: Focus on making it sound like real world speech by incorporating little features like pauses, intonation, and other natural speech patterns as though you were just talking to or giving instructions to an assistant.\n\n"
        f"Text: {transformed_content}\n\n"
        "Instruction:"
    )
    
    print("[TTS Prompt Context]")
    print(f"Transformed Content: {transformed_content}")
    print(f"Language: {language}")
    print(f"Function: {function_name}")
    print(f"Function Description: {function_description}")
    client = openai.OpenAI(api_key=openai_api_key) if openai_api_key else openai.OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert in speech synthesis and prompt engineering."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=80,
        temperature=0.7,
    )
    instruction = response.choices[0].message.content.strip()
    print(f"[TTS Instruction Generated]: {instruction}")
    return instruction

class OpenAITTSGenerator(TTSGeneratorBase):
    """OpenAI TTS Generator implementation."""
    
    def __init__(self, api_key: str = None, output_root: str = None, model: str = "gpt-4o-mini-tts", voice: str = "coral", instructions: str = "Speak naturally and conversationally, as though you were just talking to or giving instructions to an assistant."):
        super().__init__("openai", output_root=output_root)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.voice = voice
        self.instructions = instructions
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize OpenAI client."""
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key)
        else:
            self.client = OpenAI()
        print("OpenAI client initialized")
    
    def _generate_audio(self, text: str, test_case: Optional[Dict[str, Any]] = None, language: str = "en") -> bytes:
        """Generate audio using OpenAI TTS API, with dynamic instructions per sample.
        
        Args:
            text: The text to convert to speech
            test_case: Optional test case data for context
            language: ISO language code for the text
            
        Returns:
            Audio bytes in MP3 format
        """
        try:
            function_name = ""
            function_description = ""
            if test_case is not None and "function" in test_case and isinstance(test_case["function"], list) and len(test_case["function"]) > 0:
                func = test_case["function"][0]
                function_name = func.get("name", "")
                function_description = func.get("description", "")
            instruction = generate_tts_instruction_from_text(
                transformed_content=text,
                language=language,
                function_name=function_name,
                function_description=function_description,
                openai_api_key=self.api_key,
                model="gpt-4o"  # Use chat model for instruction generation
            )
            with self.client.audio.speech.with_streaming_response.create(
                model=self.model,
                voice=self.voice,
                input=text,
                instructions=instruction,
            ) as response:
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
                    response.stream_to_file(tmp_file.name)
                    tmp_file.flush()
                    tmp_file.seek(0)
                    audio_bytes = tmp_file.read()
                os.remove(tmp_file.name)
                return audio_bytes
        except Exception as e:
            print(f"Error generating audio with OpenAI: {e}")
            raise 