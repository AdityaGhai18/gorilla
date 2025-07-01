import os
import json
import requests
import dashscope
from pathlib import Path
from dotenv import load_dotenv

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


def download_audio(audio_url, save_path):
    try:
        resp = requests.get(audio_url, timeout=10)
        resp.raise_for_status()
        with open(save_path, 'wb') as f:
            f.write(resp.content)
        print(f"Audio file saved to: {save_path}")
    except Exception as e:
        raise RuntimeError(f"Download failed: {e}")


def load_json_data(file_path):
    """Load the granular speech JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def process_simple_data(data, output_dir):
    """Process simple data format (single turn per test case), split among 4 voices as evenly as possible."""
    audio_files = []
    voices = ["Cherry", "Ethan", "Chelsie", "Serena"]
    n = len(data)
    base = n // 4
    remainder = n % 4
    split = [base] * 4
    for i in range(remainder):
        split[-(i+1)] += 1  # Give extra to the last voices

    # Create subfolder for live_simple
    base_output_dir = output_dir / "live_simple"
    base_output_dir.mkdir(parents=True, exist_ok=True)

    idx = 0
    for v, count in zip(voices, split):
        # Create a subfolder for each voice
        voice_output_dir = base_output_dir / v
        voice_output_dir.mkdir(parents=True, exist_ok=True)
        for _ in range(count):
            if idx >= n:
                break
            test_case = data[idx]
            test_id = test_case.get('id', f'test_case_{idx}')
            print(f"\nProcessing test case {idx+1}: {test_id} (voice: {v})")
            try:
                transformed_content = test_case['question'][0][0]['transformed_content']
                print(f"Transformed content: {transformed_content}")
                audio_filename = f"{test_id}.wav"
                audio_path = voice_output_dir / audio_filename
                print(f"Generating TTS for transformed content...")
                audio_url = synthesize_speech(transformed_content, voice=v)
                download_audio(audio_url, audio_path)
                audio_files.append(str(audio_path))
            except KeyError as e:
                print(f"Error: Missing key {e} in test case {test_id}")
            except Exception as e:
                print(f"Error processing test case {test_id}: {e}")
            idx += 1
    return audio_files


def process_multi_turn_data(data, output_dir):
    """Process multi-turn data format (multiple turns per test case)."""
    audio_files = []
    
    for i, test_case in enumerate(data):
        test_id = test_case.get('id', f'test_case_{i}')
        print(f"\nProcessing test case {i+1}: {test_id}")
        
        # Process each turn
        for turn_idx, turn in enumerate(test_case['question']):
            try:
                transformed_content = turn[0]['transformed_content']
                print(f"Turn {turn_idx+1} transformed content: {transformed_content}")
                
                # Generate audio filename using the test ID and turn number
                audio_filename = f"{test_id}_turn_{turn_idx+1}.wav"
                audio_path = output_dir / audio_filename
                
                # Synthesize and download audio
                print(f"Generating TTS for turn {turn_idx+1}...")
                audio_url = synthesize_speech(transformed_content)
                download_audio(audio_url, audio_path)
                audio_files.append(str(audio_path))
                
            except KeyError as e:
                print(f"Error: Missing key {e} in test case {test_id}, turn {turn_idx+1}")
            except Exception as e:
                print(f"Error processing test case {test_id}, turn {turn_idx+1}: {e}")
    
    return audio_files


def process_multi_turn_data_for_flat_audio(data, output_dir):
    """Process multi-turn data, saving each turn as a separate audio file in a speaker subfolder. Each test case is assigned to a single speaker."""
    audio_files = []
    voices = ["Cherry", "Ethan", "Chelsie", "Serena"]
    for idx, test_case in enumerate(data):
        test_id = test_case.get('id', 'unknown')
        voice = voices[idx % len(voices)]
        speaker_dir = output_dir / voice
        speaker_dir.mkdir(parents=True, exist_ok=True)
        for turn_idx, turn in enumerate(test_case['question']):
            try:
                transformed_content = turn[0]['transformed_content']
                print(f"Test case: {test_id}, Turn: {turn_idx+1}, Speaker: {voice}, Content: {transformed_content}")
                audio_filename = f"{test_id}_turn_{turn_idx+1}.wav"
                audio_path = speaker_dir / audio_filename
                print(f"Processing: {audio_filename} (voice: {voice})")
                audio_url = synthesize_speech(transformed_content, voice=voice)
                download_audio(audio_url, audio_path)
                audio_files.append(str(audio_path))
            except KeyError as e:
                print(f"Error: Missing key {e} in {test_id} turn {turn_idx+1}")
            except Exception as e:
                print(f"Error processing {test_id} turn {turn_idx+1}: {e}")
    return audio_files


def main():
    # Configuration
    json_file = "BFCL_v3_multi_turn_base_granular_spoken_final2.json"
    output_dir = Path("audio")
    subfolder = "multi_turn_base"
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    print(f"Loading data from: {json_file}")
    print(f"Output directory: {output_dir}")
    
    try:
        # Load the JSON data
        data = load_json_data(json_file)
        print(f"Loaded {len(data)} test cases")
        
        # Detect data format
        is_multi_turn = False
        if data and 'question' in data[0]:
            first_question = data[0]['question']
            if isinstance(first_question, list) and len(first_question) > 1:
                is_multi_turn = True
        
        print(f"Detected data format: {'Multi-turn' if is_multi_turn else 'Simple'}")
        
        if is_multi_turn and "multi_turn_base" in json_file:
            audio_files = process_multi_turn_data_for_flat_audio(data, output_dir / subfolder)
        elif is_multi_turn:
            audio_files = process_multi_turn_data(data, output_dir / subfolder)
        else:
            audio_files = process_simple_data(data, output_dir / subfolder)
        
        print(f"\n{'='*60}")
        print(f"TTS Generation Complete!")
        print(f"Generated {len(audio_files)} audio files:")
        for audio_file in audio_files:
            print(f"  - {audio_file}")
        print(f"All files saved in: {output_dir / subfolder}")
        
    except FileNotFoundError:
        print(f"Error: JSON file '{json_file}' not found.")
        print("Please make sure the file exists in the current directory.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main() 