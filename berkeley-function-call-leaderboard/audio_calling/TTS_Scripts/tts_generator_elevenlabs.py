import os
from elevenlabs.client import ElevenLabs
from .tts_generator_base import TTSGeneratorBase

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
    
    def _generate_audio(self, text: str) -> bytes:
        """Generate audio using ElevenLabs API."""
        try:
            # Generate audio using ElevenLabs client
            audio = self.client.text_to_speech.convert(
                text=text,
                voice_id="JBFqnCBsd6RMkjVDRZzb",  # Josh voice ID
                model_id="eleven_multilingual_v2",
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
