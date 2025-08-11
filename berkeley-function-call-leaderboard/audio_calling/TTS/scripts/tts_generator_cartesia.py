import os
from cartesia import Cartesia
from audio_calling.TTS.scripts.tts_generator_base import TTSGeneratorBase

class CartesiaTTSGenerator(TTSGeneratorBase):
    """Cartesia TTS Generator implementation."""
    
    # List of languages supported by Cartesia
    SUPPORTED_LANGUAGES = [
        "en", "en-US", "en-GB", 
        "es", "es-ES", "es-MX",
        "fr", "fr-FR", 
        "de", "de-DE",
        "it", "it-IT",
        "pt", "pt-BR", "pt-PT",
        "zh", "zh-CN", "zh-TW",
        "ja", "ja-JP",
        "ko", "ko-KR",
        "ar", "ar-SA",
        "ru", "ru-RU",
        "hi", "hi-IN"
    ]
    
    def __init__(self, api_key: str, output_root: str = None):
        super().__init__("cartesia", output_root=output_root)
        self.api_key = api_key
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Cartesia client."""
        self.client = Cartesia(api_key=self.api_key)
        print("Cartesia client initialized")
    
    def _generate_audio(self, text: str, language: str = "en") -> bytes:
        """Generate audio using Cartesia API.
        
        Args:
            text: The text to convert to speech
            language: ISO language code (e.g., 'en', 'es', 'fr')
            
        Returns:
            Audio bytes in MP3 format
        """
        try:
            # Check if the language is supported, otherwise fall back to English
            # First try the full language code
            if language not in self.SUPPORTED_LANGUAGES:
                # Try the base language code (e.g., "en" from "en-US")
                base_language = language.split('-')[0] if '-' in language else language
                if base_language not in self.SUPPORTED_LANGUAGES:
                    print(f"Warning: Language '{language}' not supported by Cartesia, falling back to English")
                    language = "en"
                else:
                    language = base_language
            
            # Generate audio using Cartesia
            data_gen = self.client.tts.bytes(
                model_id="sonic-2",
                transcript=text,
                voice={"mode": "id", "id": "bf0a246a-8642-498a-9950-80c35e9276b5"},
                language=language,  # Use the provided language code or fallback
                output_format={
                    "container": "mp3",
                    "encoding": "mp3",
                    "sample_rate": 44100,
                },
            )
            
            # Convert generator to bytes by iterating through chunks
            audio_chunks = []
            for chunk in data_gen:
                audio_chunks.append(chunk)
            audio_bytes = b''.join(audio_chunks)
            
            return audio_bytes
        except Exception as e:
            print(f"Error generating audio with Cartesia: {e}")
            raise
