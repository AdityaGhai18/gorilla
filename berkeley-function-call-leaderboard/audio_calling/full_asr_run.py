import argparse
import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
import time
import tempfile
import subprocess

load_dotenv()

client = OpenAI()


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
                with open(use_path, "rb") as audio_file:
                    transcription = client.audio.transcriptions.create(
                        model="gpt-4o-transcribe",
                        file=audio_file
                    )
                return transcription.text
            except Exception as e:
                # Check for rate limit (429) in error message
                if hasattr(e, 'status_code') and e.status_code == 429:
                    print(f"[RATE LIMIT] 429 error for {audio_path}. Pausing for 60 seconds...")
                    time.sleep(60)
                    retries += 1
                    continue
                elif '429' in str(e):
                    print(f"[RATE LIMIT] 429 error for {audio_path}. Pausing for 60 seconds...")
                    time.sleep(60)
                    retries += 1
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
    for idx, case in enumerate(tqdm(data, desc="Processing test cases")):
        case_updated = False
        if idx not in indices_to_process:
            continue
        
        # Process all turn groups in the question
        for turn_group_idx, turn_group in enumerate(case.get("question", [])):
            for turn_idx, turn in enumerate(turn_group):
                audio_path = turn.get("audio_path")
                if audio_path:
                    print(f"[PROCESS] {case.get('id', '')} turn_group_{turn_group_idx} turn_{turn_idx}: Attempting ASR for {audio_path}")
                    asr = transcribe_audio(audio_path)
                    if asr is not None:
                        prev = turn.get("asr_output")
                        turn["asr_output"] = asr
                        updated += 1
                        case_updated = True
                        if prev is not None:
                            print(f"[OVERWRITE] {case.get('id', '')} turn_group_{turn_group_idx} turn_{turn_idx}: asr_output overwritten.")
                        else:
                            print(f"[SUCCESS] {case.get('id', '')} turn_group_{turn_group_idx} turn_{turn_idx}: asr_output added.")
                    else:
                        print(f"[FAIL] {case.get('id', '')} turn_group_{turn_group_idx} turn_{turn_idx}: ASR failed for {audio_path}")
                    time.sleep(1)  # Small sleep to avoid hammering API
                else:
                    print(f"[SKIP] {case.get('id', '')} turn_group_{turn_group_idx} turn_{turn_idx}: No audio_path found.")
        
        # Write after each case if any update was made
        if case_updated:
            with open(data_path, "w") as f:
                json.dump(data, f, indent=2)
    print(f"\nAdded/overwritten ASR output for {updated} audio files.")


def main():
    parser = argparse.ArgumentParser(description="Add ASR output to test cases using OpenAI Whisper.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the JSON data file.")
    parser.add_argument("--sample_count", type=int, default=None, help="Number of test cases to process (default: all)")
    args = parser.parse_args()

    process_file(args.data_path, args.sample_count)

if __name__ == "__main__":
    main() 