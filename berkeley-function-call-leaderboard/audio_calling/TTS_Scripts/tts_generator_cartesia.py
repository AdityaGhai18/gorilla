import os
from cartesia import Cartesia
from .tts_generator_base import TTSGeneratorBase

class CartesiaTTSGenerator(TTSGeneratorBase):
    """Cartesia TTS Generator implementation."""
    
    def __init__(self, api_key: str, output_root: str = None):
        super().__init__("cartesia", output_root=output_root)
        self.api_key = api_key
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Cartesia client."""
        self.client = Cartesia(api_key=self.api_key)
        print("Cartesia client initialized")
    
    def _generate_audio(self, text: str) -> bytes:
        """Generate audio using Cartesia API."""
        try:
            # Generate audio using Cartesia - exactly like the working example
            data_gen = self.client.tts.bytes(
                model_id="sonic-2",
                transcript=text,
                voice={"mode": "id", "id": "bf0a246a-8642-498a-9950-80c35e9276b5"},
                language="en",
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
