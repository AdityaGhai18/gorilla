#!/usr/bin/env python3
"""
Script to update audio_path fields from .mp3 to .wav for Cartesia files in all JSON files.
This script updates audio_path values to reflect that Cartesia files are now .wav format.
Automatically finds all JSON files in the final_results directory.
"""

import argparse
import json
import os
import glob

# Make tqdm optional
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # Simple progress bar replacement
    def tqdm(iterable, desc=""):
        print(f"{desc}: {len(iterable)} items")
        return iterable

def update_audio_paths(data_path):
    """Update audio_path fields from .mp3 to .wav for Cartesia files."""
    print(f"Processing: {data_path}")
    
    # Read the JSON file
    with open(data_path, "r") as f:
        data = json.load(f)
    
    updated_count = 0
    total_cases = len(data)
    
    # Process each test case
    for case in tqdm(data, desc="Processing test cases"):
        case_updated = False
        
        # Handle both single-turn and multi-turn formats
        if "question" in case and isinstance(case["question"], list):
            for turn_group in case["question"]:
                # Handle single-turn format: turn_group is a list with one turn
                if isinstance(turn_group, list):
                    for turn in turn_group:
                        if isinstance(turn, dict) and "audio_path" in turn:
                            audio_path = turn["audio_path"]
                            # Check if this is a Cartesia .mp3 file that needs to be updated to .wav
                            if "cartesia" in audio_path.lower() and audio_path.endswith(".mp3"):
                                new_audio_path = audio_path.replace(".mp3", ".wav")
                                turn["audio_path"] = new_audio_path
                                updated_count += 1
                                case_updated = True
                                print(f"[UPDATE] {case.get('id', '')}: {audio_path} -> {new_audio_path}")
                # Handle direct turn format (if any)
                elif isinstance(turn_group, dict) and "audio_path" in turn_group:
                    audio_path = turn_group["audio_path"]
                    # Check if this is a Cartesia .mp3 file that needs to be updated to .wav
                    if "cartesia" in audio_path.lower() and audio_path.endswith(".mp3"):
                        new_audio_path = audio_path.replace(".mp3", ".wav")
                        turn_group["audio_path"] = new_audio_path
                        updated_count += 1
                        case_updated = True
                        print(f"[UPDATE] {case.get('id', '')}: {audio_path} -> {new_audio_path}")
        
        # Write back to file if any changes were made
        if case_updated:
            with open(data_path, "w") as f:
                json.dump(data, f, indent=2)
    
    print(f"Completed: {data_path}")
    print(f"Total cases processed: {total_cases}")
    print(f"Audio paths updated: {updated_count}")
    print("-" * 50)
    
    return updated_count

def process_all_json_files(base_dir):
    """Find and process all JSON files in the directory."""
    # Find all JSON files in the directory
    json_pattern = os.path.join(base_dir, "*.json")
    json_files = glob.glob(json_pattern)
    
    if not json_files:
        print(f"No JSON files found in: {base_dir}")
        return
    
    print(f"Found {len(json_files)} JSON files to process:")
    for file in json_files:
        print(f"  - {os.path.basename(file)}")
    print()
    
    total_renamed = 0
    
    # Process each JSON file
    for json_file in json_files:
        try:
            updated_count = update_audio_paths(json_file)
            total_renamed += updated_count
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
    
    print("=" * 60)
    print(f"PROCESSING COMPLETED!")
    print(f"Total audio paths updated across all files: {total_renamed}")
    print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description="Update audio_path fields from .mp3 to .wav for Cartesia files in all JSON files")
    parser.add_argument("--data_path", type=str, help="Path to a specific JSON data file (optional)")
    parser.add_argument("--base_dir", type=str, default="audio_calling/clean_to_speech_text/final_results", 
                       help="Base directory containing JSON files (default: final_results)")
    args = parser.parse_args()
    
    if args.data_path:
        # Process single file
        if not os.path.exists(args.data_path):
            print(f"Error: File not found: {args.data_path}")
            return
        
        try:
            updated_count = update_audio_paths(args.data_path)
            print(f"Successfully updated {updated_count} audio paths from .mp3 to .wav")
        except Exception as e:
            print(f"Error processing {args.data_path}: {e}")
    else:
        # Process all JSON files in directory
        if not os.path.exists(args.base_dir):
            print(f"Error: Directory not found: {args.base_dir}")
            return
        
        process_all_json_files(args.base_dir)

if __name__ == "__main__":
    main() 