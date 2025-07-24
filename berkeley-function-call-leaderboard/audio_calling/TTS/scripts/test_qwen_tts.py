import os
from dotenv import load_dotenv
import dashscope

# Load environment variables from .env file
load_dotenv()

def get_api_key():
    api_key = os.getenv("QWEN_API_KEY")
    if not api_key:
        raise EnvironmentError("QWEN_API_KEY environment variable not set.")
    return api_key

def synthesize_speech(text, voice="Cherry", model="qwen-tts-latest"):
    api_key = get_api_key()
    try:
        response = dashscope.audio.qwen_tts.SpeechSynthesizer.call(
            model=model,
            api_key=api_key,
            text=text,
            voice=voice,
        )
        # Check if response is None
        if response is None:
            raise RuntimeError("API call returned None response")
        # Check if response.output is None
        if response.output is None:
            raise RuntimeError("API call failed: response.output is None")
        # Check if response.output.audio exists
        if not hasattr(response.output, 'audio') or response.output.audio is None:
            raise RuntimeError("API call failed: response.output.audio is None or missing")
        audio_url = response.output.audio["url"]
        return audio_url
    except Exception as e:
        raise RuntimeError(f"Speech synthesis failed: {e}")

if __name__ == "__main__":
    print("QWEN_API_KEY:", os.getenv("QWEN_API_KEY"))
    try:
        audio_url = synthesize_speech("Hello world", voice="Cherry")
        print("Audio URL:", audio_url)
    except Exception as e:
        print("Error:", e) 