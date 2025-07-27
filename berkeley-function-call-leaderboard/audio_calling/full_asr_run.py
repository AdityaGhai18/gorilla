import argparse
import os
import json
from dotenv import load_dotenv
import asyncio
from deepgram import Deepgram
from tqdm import tqdm
import time
import tempfile
import subprocess
from io import BytesIO

load_dotenv()

# Initialize Deepgram client
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
if not DEEPGRAM_API_KEY:
    print("[ERROR] DEEPGRAM_API_KEY environment variable not found")
    print("[INFO] Please set DEEPGRAM_API_KEY in your .env file")
    deepgram_client = None
else:
    deepgram_client = Deepgram(DEEPGRAM_API_KEY)
    print("[INFO] Deepgram client initialized successfully")


def reencode_wav_to_pcm16k(input_path):
    temp_fd, temp_path = tempfile.mkstemp(suffix=".wav")
    os.close(temp_fd)
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000", temp_path
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"[RE-ENCODE] {input_path} -> {temp_path}")
        return temp_path
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] ffmpeg failed to re-encode {input_path}: {e}")
        return None

async def transcribe_audio_async(audio_path):
    if deepgram_client is None:
        print(f"[ERROR] Deepgram client not available for {audio_path}")
        return None
    
    retries = 0
    max_retries = 3
    
    while retries < max_retries:
        try:
            # Open the audio file in binary mode
            with open(audio_path, 'rb') as audio:
                # Make the transcription request with version 3 API
                response = await deepgram_client.transcription.prerecorded(
                    {"buffer": audio, "mimetype": "audio/wav"},
                    {
                        "model": "nova-2",
                        "punctuate": True,
                        "language": "en-US"
                    }
                )
                
                # Extract the transcript text from the response
                transcript = response["results"]["channels"][0]["alternatives"][0]["transcript"]
                
                if transcript and transcript.strip():
                    return transcript.strip()
                else:
                    print(f"[WARN] Empty transcription for {audio_path}")
                    return None
                    
        except Exception as e:
            print(f"[ERROR] Could not transcribe {audio_path}: {e}")
            retries += 1
            if retries < max_retries:
                print(f"[RETRY] Attempting retry {retries}/{max_retries}...")
                await asyncio.sleep(5)
                continue
            else:
                print(f"[ERROR] Max retries exceeded for {audio_path}")
                return None
    
    return None

def transcribe_audio(audio_path):
    """Wrapper to run async transcription in sync context"""
    return asyncio.run(transcribe_audio_async(audio_path))

def process_file(data_path, sample_count=None):
    with open(data_path, "r") as f:
        data = json.load(f)

    # Determine which indices to process
    indices_to_process = set(range(len(data)))
    if sample_count is not None:
        import random
        if sample_count > len(data):
            sample_count = len(data)
        indices_to_process = set(random.sample(range(len(data)), sample_count))

    updated = 0
    
    # Create a filtered list for progress bar
    filtered_data = [case for idx, case in enumerate(data) if idx in indices_to_process]
    
    for idx, case in enumerate(tqdm(filtered_data, desc=f"Processing test cases (sample_count={sample_count or 'all'})")):
        case_updated = False
        
        # Process all turn groups in the question
        for turn_group_idx, turn_group in enumerate(case.get("question", [])):
            # Both single-turn and multi-turn use the same structure: turn_group is a list
            if isinstance(turn_group, list):
                for turn_idx, turn in enumerate(turn_group):
                    audio_path = turn.get("audio_path")
                    if audio_path:
                        print(f"[PROCESS] {case.get('id', '')} turn_group_{turn_group_idx} turn_{turn_idx}: Attempting Deepgram ASR for {audio_path}")
                        asr_deepgram = transcribe_audio(audio_path)
                        if asr_deepgram is not None:
                            # Add asr_output_deepgram field if it doesn't exist
                            if "asr_output_deepgram" not in turn:
                                turn["asr_output_deepgram"] = asr_deepgram
                                updated += 1
                                case_updated = True
                                print(f"[SUCCESS] {case.get('id', '')} turn_group_{turn_group_idx} turn_{turn_idx}: asr_output_deepgram added.")
                            else:
                                print(f"[SKIP] {case.get('id', '')} turn_group_{turn_group_idx} turn_{turn_idx}: asr_output_deepgram already exists")
                        else:
                            print(f"[FAIL] {case.get('id', '')} turn_group_{turn_group_idx} turn_{turn_idx}: Deepgram ASR failed for {audio_path}")
                        time.sleep(1)  # Small sleep to avoid hammering API
                    else:
                        print(f"[SKIP] {case.get('id', '')} turn_group_{turn_group_idx} turn_{turn_idx}: No audio_path found.")
        
        # Write after each case if any update was made
        if case_updated:
            with open(data_path, "w") as f:
                json.dump(data, f, indent=2)
    
    print(f"\nAdded Deepgram ASR output for {updated} audio files.")


def main():
    parser = argparse.ArgumentParser(description="Add ASR output to test cases using Deepgram.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the JSON data file.")
    parser.add_argument("--sample_count", type=int, default=None, help="Number of test cases to process (default: all)")
    parser.add_argument("--test_mode", action="store_true", help="Test mode: process only 3 random samples")
    args = parser.parse_args()

    # If test mode is enabled, override sample_count to 3
    if args.test_mode:
        args.sample_count = 3
        print("[TEST MODE] Processing only 3 random samples for testing")

    process_file(args.data_path, args.sample_count)

if __name__ == "__main__":
    main() 