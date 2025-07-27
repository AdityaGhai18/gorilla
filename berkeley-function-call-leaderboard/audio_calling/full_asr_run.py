import argparse
import os
import json
from dotenv import load_dotenv
# from openai import OpenAI  # Commented out OpenAI
from google.cloud import speech  # Added Google Cloud Speech
from tqdm import tqdm
import time
import tempfile
import subprocess

load_dotenv()

# client = OpenAI()  # Commented out OpenAI client

# Initialize Google Cloud Speech client
# Make sure to set GOOGLE_APPLICATION_CREDENTIALS environment variable or use service account key
try:
    google_client = speech.SpeechClient()
    print("[INFO] Google Cloud Speech client initialized successfully")
except Exception as e:
    print(f"[ERROR] Failed to initialize Google Cloud Speech client: {e}")
    print("[INFO] Make sure you have set up Google Cloud credentials properly")
    google_client = None


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

def transcribe_audio(audio_path):
    if google_client is None:
        print(f"[ERROR] Google Cloud Speech client not available for {audio_path}")
        return None
    
    retries = 0
    temp_path = None
    use_path = audio_path
    
    # Only re-encode if .wav
    if audio_path.lower().endswith('.wav'):
        temp_path = reencode_wav_to_pcm16k(audio_path)
        if temp_path:
            use_path = temp_path
        else:
            print(f"[WARN] Re-encode failed for {audio_path}, trying original file.")
            use_path = audio_path
    
    try:
        while True:
            try:
                # Read the audio file
                with open(use_path, "rb") as audio_file:
                    content = audio_file.read()
                
                # Configure the recognition
                audio = speech.RecognitionAudio(content=content)
                config = speech.RecognitionConfig(
                    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=16000,
                    language_code="en-US",
                    enable_automatic_punctuation=True,
                    model="latest_long",  # Use latest model for better accuracy
                )
                
                # Perform the transcription
                response = google_client.recognize(config=config, audio=audio)
                
                # Extract the transcription
                if response.results:
                    transcription = ""
                    for result in response.results:
                        transcription += result.alternatives[0].transcript + " "
                    return transcription.strip()
                else:
                    print(f"[WARN] No transcription results for {audio_path}")
                    return None
                    
            except Exception as e:
                # Check for rate limit or quota exceeded errors
                if "quota" in str(e).lower() or "rate" in str(e).lower():
                    print(f"[RATE LIMIT] Quota/rate limit error for {audio_path}. Pausing for 60 seconds...")
                    time.sleep(60)
                    retries += 1
                    if retries > 3:  # Limit retries
                        print(f"[ERROR] Max retries exceeded for {audio_path}")
                        return None
                    continue
                else:
                    print(f"[ERROR] Could not transcribe {audio_path}: {e}")
                    return None
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

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
    renamed = 0
    
    for idx, case in enumerate(tqdm(data, desc="Processing test cases")):
        case_updated = False
        if idx not in indices_to_process:
            continue
        
        # Process all turn groups in the question
        for turn_group_idx, turn_group in enumerate(case.get("question", [])):
            for turn_idx, turn in enumerate(turn_group):
                # First, rename existing asr_output to asr_output_openai if it exists
                if "asr_output" in turn and "asr_output_openai" not in turn:
                    turn["asr_output_openai"] = turn["asr_output"]
                    del turn["asr_output"]
                    renamed += 1
                    case_updated = True
                    print(f"[RENAME] {case.get('id', '')} turn_group_{turn_group_idx} turn_{turn_idx}: asr_output -> asr_output_openai")
                
                audio_path = turn.get("audio_path")
                if audio_path:
                    print(f"[PROCESS] {case.get('id', '')} turn_group_{turn_group_idx} turn_{turn_idx}: Attempting Google ASR for {audio_path}")
                    asr_google = transcribe_audio(audio_path)
                    if asr_google is not None:
                        # Add asr_output_google field if it doesn't exist
                        if "asr_output_google" not in turn:
                            turn["asr_output_google"] = asr_google
                            updated += 1
                            case_updated = True
                            print(f"[SUCCESS] {case.get('id', '')} turn_group_{turn_group_idx} turn_{turn_idx}: asr_output_google added.")
                        else:
                            print(f"[SKIP] {case.get('id', '')} turn_group_{turn_group_idx} turn_{turn_idx}: asr_output_google already exists")
                    else:
                        print(f"[FAIL] {case.get('id', '')} turn_group_{turn_group_idx} turn_{turn_idx}: Google ASR failed for {audio_path}")
                    time.sleep(1)  # Small sleep to avoid hammering API
                else:
                    print(f"[SKIP] {case.get('id', '')} turn_group_{turn_group_idx} turn_{turn_idx}: No audio_path found.")
        
        # Write after each case if any update was made
        if case_updated:
            with open(data_path, "w") as f:
                json.dump(data, f, indent=2)
    
    print(f"\nRenamed {renamed} asr_output fields to asr_output_openai")
    print(f"Added Google ASR output for {updated} audio files.")


def main():
    parser = argparse.ArgumentParser(description="Add ASR output to test cases using Google Cloud Speech-to-Text.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the JSON data file.")
    parser.add_argument("--sample_count", type=int, default=None, help="Number of test cases to process (default: all)")
    args = parser.parse_args()

    process_file(args.data_path, args.sample_count)

if __name__ == "__main__":
    main() 