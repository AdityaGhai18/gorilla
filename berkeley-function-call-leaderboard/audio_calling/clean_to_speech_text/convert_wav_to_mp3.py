#!/usr/bin/env python3
"""
WAV to MP3 Conversion Script with JSON Path Updates

This script converts WAV files to MP3 format and updates corresponding JSON files
to change audio_path references from .wav to .mp3. Supports parallel processing
across multiple folders.

Usage:
    python convert_wav_to_mp3.py /path/to/folder --json-dir /path/to/json/files
    python convert_wav_to_mp3.py /path/to/folder --quality 192 --json-dir /path/to/json
    python convert_wav_to_mp3.py /path/to/folder --delete-wav --json-dir /path/to/json
"""

import os
import argparse
import subprocess
import sys
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

def check_ffmpeg_available() -> bool:
    """Check if ffmpeg is available in the system."""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def convert_wav_to_mp3_worker(args_tuple):
    """
    Worker function for parallel processing.
    
    Args:
        args_tuple: Tuple of (wav_path, delete_wav, log_file)
    
    Returns:
        Tuple of (wav_path, success, error_message, log_entry)
    """
    wav_path, delete_wav, log_file = args_tuple
    wav_path = Path(wav_path)
    
    # Create log entry
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "wav_file": str(wav_path),
        "mp3_file": str(wav_path.with_suffix('.mp3')),
        "wav_size_bytes": wav_path.stat().st_size if wav_path.exists() else 0,
        "quality_kbps": 320,  # Best quality
        "delete_wav": delete_wav,
        "success": False,
        "error": None,
        "mp3_size_bytes": 0
    }
    
    try:
        # Create MP3 path
        mp3_path = wav_path.with_suffix('.mp3')
        
        # Skip if MP3 already exists
        if mp3_path.exists():
            log_entry["success"] = True
            log_entry["mp3_size_bytes"] = mp3_path.stat().st_size
            log_entry["error"] = "MP3 already exists"
            if delete_wav and wav_path.exists():
                wav_path.unlink()
                log_entry["wav_deleted"] = True
            return (str(wav_path), True, "MP3 already exists", log_entry)
        
        # Convert WAV to MP3 using ffmpeg (best quality)
        cmd = [
            'ffmpeg',
            '-i', str(wav_path),
            '-codec:a', 'mp3',
            '-b:a', '320k',  # Best quality
            '-y',  # Overwrite output file if it exists
            str(mp3_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            # Verify MP3 file was created and has size > 0
            if mp3_path.exists() and mp3_path.stat().st_size > 0:
                log_entry["success"] = True
                log_entry["mp3_size_bytes"] = mp3_path.stat().st_size
                # Delete original WAV if requested
                if delete_wav:
                    wav_path.unlink()
                    log_entry["wav_deleted"] = True
                return (str(wav_path), True, None, log_entry)
            else:
                log_entry["error"] = "MP3 file not created or empty"
                if mp3_path.exists():
                    mp3_path.unlink()  # Clean up empty file
                return (str(wav_path), False, "MP3 file not created or empty", log_entry)
        else:
            log_entry["error"] = result.stderr
            return (str(wav_path), False, result.stderr, log_entry)
            
    except Exception as e:
        log_entry["error"] = str(e)
        return (str(wav_path), False, str(e), log_entry)

def update_json_audio_paths(json_dir: Path, folder_name: str, log_file: Path) -> tuple[int, List[Dict]]:
    """
    Update audio_path references in JSON files from .wav to .mp3.
    
    Args:
        json_dir: Directory containing JSON files
        folder_name: Name of the audio folder being processed
        log_file: Path to log file for recording changes
    
    Returns:
        Tuple of (number of JSON files updated, list of changes made)
    """
    updated_count = 0
    changes_made = []
    
    # Find all JSON files in the directory
    json_files = list(json_dir.glob('*.json'))
    
    for json_file in json_files:
        try:
            # Read JSON file
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Track if this file needs updating
            file_updated = False
            file_changes = []
            
            # Recursively search and update audio_path fields
            def update_audio_paths(obj, path=""):
                nonlocal file_updated
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        current_path = f"{path}.{key}" if path else key
                        if key == "audio_path" and isinstance(value, str):
                            # Check if this path points to our folder and ends with .wav
                            if folder_name in value and value.endswith('.wav'):
                                new_path = value.replace('.wav', '.mp3')
                                old_path = value
                                obj[key] = new_path
                                file_updated = True
                                
                                # Record the change
                                change_record = {
                                    "timestamp": datetime.now().isoformat(),
                                    "json_file": str(json_file),
                                    "json_path": current_path,
                                    "old_audio_path": old_path,
                                    "new_audio_path": new_path,
                                    "test_case_id": None
                                }
                                
                                # Try to extract test case ID from the path
                                if "id" in obj:
                                    change_record["test_case_id"] = obj["id"]
                                
                                file_changes.append(change_record)
                                print(f"  Updated: {old_path} -> {new_path}")
                        elif isinstance(value, (dict, list)):
                            update_audio_paths(value, current_path)
                elif isinstance(obj, list):
                    for i, item in enumerate(obj):
                        update_audio_paths(item, f"{path}[{i}]")
            
            # Update the data
            update_audio_paths(data)
            
            # Write back to file if changes were made
            if file_updated:
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                updated_count += 1
                changes_made.extend(file_changes)
                print(f"  Updated JSON file: {json_file.name}")
            
        except Exception as e:
            print(f"  Error updating {json_file}: {e}")
    
    return updated_count, changes_made

def process_folder_parallel(folder_path: Path, delete_wav: bool = False, max_workers: int = None, test_mode: bool = False, log_file: Path = None) -> tuple[int, int, List[str], List[Dict]]:
    """
    Process all WAV files in a folder using parallel processing.
    
    Args:
        folder_path: Path to the folder containing WAV files
        quality: MP3 quality (kbps)
        delete_wav: Whether to delete original WAV files after conversion
        max_workers: Maximum number of worker processes (default: CPU count)
    
    Returns:
        Tuple of (successful_conversions, total_files, error_messages)
    """
    if not folder_path.exists():
        print(f"Error: Folder {folder_path} does not exist")
        return 0, 0, [f"Folder {folder_path} does not exist"]
    
    if not folder_path.is_dir():
        print(f"Error: {folder_path} is not a directory")
        return 0, 0, [f"{folder_path} is not a directory"]
    
    # Find all WAV files
    wav_files = list(folder_path.glob('*.wav'))
    
    if not wav_files:
        print(f"No WAV files found in {folder_path}")
        return 0, 0, []
    
    # Apply test mode if enabled
    if test_mode:
        wav_files = wav_files[:3]
        print(f"\nTEST MODE: Processing folder: {folder_path}")
        print(f"Processing first 3 of {len(wav_files)} WAV files")
    else:
        print(f"\nProcessing folder: {folder_path}")
        print(f"Found {len(wav_files)} WAV files")
    
    # Prepare arguments for parallel processing
    args_list = [(str(wav_file), delete_wav, log_file) for wav_file in wav_files]
    
    # Use ProcessPoolExecutor for parallel processing
    successful = 0
    error_messages = []
    all_log_entries = []
    
    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count(), len(wav_files))
    
    print(f"Using {max_workers} worker processes...")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_wav = {executor.submit(convert_wav_to_mp3_worker, args): args[0] for args in args_list}
        
        # Process completed tasks
        for future in as_completed(future_to_wav):
            wav_path, success, error_msg, log_entry = future.result()
            all_log_entries.append(log_entry)
            if success:
                successful += 1
                print(f"  ✓ Converted: {Path(wav_path).name}")
            else:
                error_messages.append(f"{Path(wav_path).name}: {error_msg}")
                print(f"  ✗ Failed: {Path(wav_path).name} - {error_msg}")
    
    print(f"Folder {folder_path}: {successful}/{len(wav_files)} files converted successfully")
    return successful, len(wav_files), error_messages, all_log_entries

def main():
    parser = argparse.ArgumentParser(
        description="Convert WAV files to MP3 format with JSON path updates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python convert_wav_to_mp3.py /path/to/folder --json-dir /path/to/json/files
  python convert_wav_to_mp3.py /path/to/folder --delete-wav --json-dir /path/to/json
  python convert_wav_to_mp3.py /path/to/folder --test-mode --json-dir /path/to/json
        """
    )
    
    parser.add_argument(
        'folder_path',
        type=str,
        help='Path to folder containing WAV files'
    )
    
    parser.add_argument(
        '--json-dir',
        type=str,
        required=True,
        help='Directory containing JSON files to update with new audio paths'
    )
    

    
    parser.add_argument(
        '--delete-wav', '-d',
        action='store_true',
        help='Delete original WAV files after successful conversion'
    )
    
    parser.add_argument(
        '--max-workers',
        type=int,
        help='Maximum number of worker processes (default: CPU count)'
    )
    
    parser.add_argument(
        '--test-mode',
        action='store_true',
        help='Test mode: only process first 3 WAV files'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be converted without actually converting'
    )
    
    args = parser.parse_args()
    
    # Check if ffmpeg is available
    if not check_ffmpeg_available():
        print("Error: ffmpeg is not available. Please install ffmpeg first.")
        print("Installation instructions:")
        print("  macOS: brew install ffmpeg")
        print("  Ubuntu/Debian: sudo apt-get install ffmpeg")
        print("  Windows: Download from https://ffmpeg.org/download.html")
        sys.exit(1)
    
    folder_path = Path(args.folder_path)
    json_dir = Path(args.json_dir)
    
    # Validate paths
    if not folder_path.exists():
        print(f"Error: Folder {folder_path} does not exist")
        sys.exit(1)
    
    if not json_dir.exists():
        print(f"Error: JSON directory {json_dir} does not exist")
        sys.exit(1)
    
    if args.dry_run:
        print(f"DRY RUN: Would process folder: {folder_path}")
        print(f"JSON directory: {json_dir}")
        if folder_path.exists() and folder_path.is_dir():
            wav_files = list(folder_path.glob('*.wav'))
            if args.test_mode:
                wav_files = wav_files[:3]
                print(f"TEST MODE: Would convert first 3 of {len(wav_files)} WAV files")
            else:
                print(f"Would convert {len(wav_files)} WAV files")
            for wav_file in wav_files:
                mp3_path = wav_file.with_suffix('.mp3')
                if mp3_path.exists():
                    print(f"  Skip: {wav_file.name} (MP3 already exists)")
                else:
                    print(f"  Convert: {wav_file.name} -> {mp3_path.name}")
            
            # Show JSON files that would be updated
            json_files = list(json_dir.glob('*.json'))
            print(f"Would update {len(json_files)} JSON files")
        else:
            print(f"Error: Folder {folder_path} does not exist or is not a directory")
        return
    
    # Create logs directory
    logs_dir = Path("mp3_conversion_logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Create log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = folder_path.name
    log_file = logs_dir / f"conversion_{folder_name}_{timestamp}.json"
    
    # Process the folder
    start_time = time.time()
    successful, total, errors, log_entries = process_folder_parallel(folder_path, args.delete_wav, args.max_workers, args.test_mode, log_file)
    
    # Update JSON files if conversion was successful
    json_updated = 0
    json_changes = []
    if successful > 0:
        print(f"\nUpdating JSON files in {json_dir}...")
        json_updated, json_changes = update_json_audio_paths(json_dir, folder_name, log_file)
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Write comprehensive log file
    log_data = {
        "conversion_summary": {
            "folder": str(folder_path),
            "quality_kbps": 320,  # Best quality
            "delete_wav": args.delete_wav,
            "test_mode": args.test_mode,
            "successful_conversions": successful,
            "total_files": total,
            "json_files_updated": json_updated,
            "duration_seconds": duration,
            "timestamp": datetime.now().isoformat()
        },
        "conversion_logs": log_entries,
        "json_changes": json_changes,
        "errors": errors
    }
    
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"CONVERSION SUMMARY")
    print(f"{'='*60}")
    print(f"Folder: {folder_path}")
    print(f"Quality: 320 kbps (best)")
    print(f"Delete WAV: {args.delete_wav}")
    print(f"Test Mode: {args.test_mode}")
    print(f"Successfully converted: {successful}/{total} files")
    print(f"JSON files updated: {json_updated}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Log file: {log_file}")
    
    if errors:
        print(f"\nErrors encountered:")
        for error in errors[:10]:  # Show first 10 errors
            print(f"  - {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
    
    if successful == total:
        print(f"\n✓ All files converted successfully!")
        print(f"📋 Detailed log saved to: {log_file}")
    else:
        print(f"\n⚠ {total - successful} files failed to convert")
        print(f"📋 Error log saved to: {log_file}")
        sys.exit(1)

if __name__ == "__main__":
    main() 