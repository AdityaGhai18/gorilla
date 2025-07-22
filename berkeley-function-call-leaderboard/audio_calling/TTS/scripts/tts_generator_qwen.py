import os
import requests
import dashscope
from audio_calling.TTS.scripts.tts_generator_base import TTSGeneratorBase

class QwenTTSGenerator(TTSGeneratorBase):
    """Qwen TTS Generator implementation using working dashscope.audio.qwen_tts.SpeechSynthesizer.call approach."""
    
    def __init__(self, api_key: str, output_root: str = None):
        super().__init__("qwen", output_root=output_root)
        self.api_key = api_key
        self._initialize_client()
    
    def _initialize_client(self):
        # No client object needed for dashscope
        print("Qwen client initialized")
    
    def _generate_audio(self, text: str) -> bytes:
        """Generate audio using Qwen API and return audio bytes."""
        try:
            response = dashscope.audio.qwen_tts.SpeechSynthesizer.call(
                model="qwen-tts-latest",
                api_key=self.api_key,
                text=text,
                voice="Cherry",
            )
            # Defensive checks as in your original code
            if response is None:
                raise RuntimeError("API call returned None response")
            if response.output is None:
                raise RuntimeError("API call failed: response.output is None")
            if not hasattr(response.output, 'audio') or response.output.audio is None:
                raise RuntimeError("API call failed: response.output.audio is None or missing")
            audio_url = response.output.audio["url"]
            # Download the audio file and return bytes
            resp = requests.get(audio_url, timeout=10)
            resp.raise_for_status()
            return resp.content
        except Exception as e:
            print(f"Error generating audio with Qwen: {e}")
            raise

