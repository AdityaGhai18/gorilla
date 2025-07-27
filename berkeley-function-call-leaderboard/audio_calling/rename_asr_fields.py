#!/usr/bin/env python3
"""
Script to safely rename asr_output fields to asr_output_openai in all JSON files.
This script ONLY renames field names, it does NOT change any content.
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

def rename_asr_fields(data_path):
    """Rename asr_output fields to asr_output_openai without changing content."""
    print(f"Processing: {data_path}")
    
    # Read the JSON file
    with open(data_path, "r") as f:
        data = json.load(f)
    
    renamed_count = 0
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
                        if isinstance(turn, dict) and "asr_output" in turn and "asr_output_openai" not in turn:
                            # Rename the field without changing content
                            turn["asr_output_openai"] = turn["asr_output"]
                            del turn["asr_output"]
                            renamed_count += 1
                            case_updated = True
                            print(f"[RENAME] {case.get('id', '')}: asr_output -> asr_output_openai")
                # Handle direct turn format (if any)
                elif isinstance(turn_group, dict) and "asr_output" in turn_group and "asr_output_openai" not in turn_group:
                    # Rename the field without changing content
                    turn_group["asr_output_openai"] = turn_group["asr_output"]
                    del turn_group["asr_output"]
                    renamed_count += 1
                    case_updated = True
                    print(f"[RENAME] {case.get('id', '')}: asr_output -> asr_output_openai")
        
        # Write back to file if any changes were made
        if case_updated:
            with open(data_path, "w") as f:
                json.dump(data, f, indent=2)
    
    print(f"Completed: {data_path}")
    print(f"Total cases processed: {total_cases}")
    print(f"Fields renamed: {renamed_count}")
    print("-" * 50)
    
    return renamed_count

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
            renamed_count = rename_asr_fields(json_file)
            total_renamed += renamed_count
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
    
    print("=" * 60)
    print(f"PROCESSING COMPLETED!")
    print(f"Total fields renamed across all files: {total_renamed}")
    print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description="Safely rename asr_output fields to asr_output_openai in all JSON files")
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
            renamed_count = rename_asr_fields(args.data_path)
            print(f"Successfully renamed {renamed_count} asr_output fields to asr_output_openai")
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