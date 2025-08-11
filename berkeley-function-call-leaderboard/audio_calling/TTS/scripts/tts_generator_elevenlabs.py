import os
from elevenlabs.client import ElevenLabs
from audio_calling.TTS.scripts.tts_generator_base import TTSGeneratorBase

class ElevenLabsTTSGenerator(TTSGeneratorBase):
    """ElevenLabs TTS Generator implementation."""
    
    def __init__(self, api_key: str, output_root: str = None):
        super().__init__("elevenlabs", output_root=output_root)
        self.api_key = api_key
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize ElevenLabs client."""
        self.client = ElevenLabs(api_key=self.api_key)
        print("ElevenLabs client initialized")
    
    def _generate_audio(self, text: str, language: str = None) -> bytes:
        """Generate audio using ElevenLabs API.
        
        Args:
            text: The text to convert to speech
            language: ISO language code (optional, ElevenLabs detects language automatically)
            
        Returns:
            Audio bytes in MP3 format
        """
        try:
            # Generate audio using ElevenLabs client
            # Note: ElevenLabs multilingual model automatically detects the language
            # from the input text, so we don't need to explicitly set the language
            audio = self.client.text_to_speech.convert(
                text=text,
                voice_id="JBFqnCBsd6RMkjVDRZzb",  # Josh voice ID
                model_id="eleven_multilingual_v2", # use v2 because v3 is not available for public use
                output_format="mp3_44100_128",
            )
            
            # Convert generator to bytes by iterating through chunks
            audio_chunks = []
            for chunk in audio:
                audio_chunks.append(chunk)
            audio_bytes = b''.join(audio_chunks)
            
            return audio_bytes
        except Exception as e:
            print(f"Error generating audio with ElevenLabs: {e}")
            raise
