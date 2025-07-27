#!/usr/bin/env python3
"""
Script to convert all .mp3 files to .wav format in the final_results/audio directory.
Converts all MP3 files to high-quality WAV and deletes the original MP3 files.
"""

import os
import subprocess
import glob
from pathlib import Path
import argparse
import logging
from datetime import datetime

def setup_logging(log_dir):
    """Setup logging for the conversion process."""
    os.makedirs(log_dir, exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"conversion_{timestamp}.log")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also print to console
        ]
    )
    
    return logging.getLogger(__name__)

def check_ffmpeg():
    """Check if ffmpeg is available on the system."""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def convert_mp3_to_wav(input_file, output_file, logger):
    """Convert a single .mp3 file to high-quality .wav format using ffmpeg."""
    # High quality WAV settings: 44.1kHz, stereo, 16-bit PCM
    cmd = [
        'ffmpeg',
        '-i', input_file,
        '-y',  # Overwrite output file if it exists
        '-ar', '44100',  # Sample rate: 44.1kHz
        '-ac', '2',      # Audio channels: stereo
        '-sample_fmt', 's16',  # Sample format: 16-bit signed
        output_file
    ]
    
    logger.info(f"Converting: {os.path.basename(input_file)} -> {os.path.basename(output_file)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info(f"SUCCESS: Converted {os.path.basename(input_file)}")
        print(f"Converted: {os.path.basename(input_file)} -> {os.path.basename(output_file)}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"FAILED: Error converting {input_file}: {e.stderr}")
        print(f"Error converting {input_file}: {e.stderr}")
        return False

def find_all_mp3_files(base_dir, logger):
    """Find all .mp3 files in the base directory and all subdirectories."""
    mp3_files = []
    logger.info(f"Searching for MP3 files in: {base_dir}")
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.mp3'):
                full_path = os.path.join(root, file)
                mp3_files.append(full_path)
    
    logger.info(f"Total MP3 files found: {len(mp3_files)}")
    return mp3_files

def convert_files(mp3_files, delete_mp3=True, logger=None):
    """Convert all MP3 files to WAV format and optionally delete original MP3 files."""
    if logger:
        logger.info(f"Starting conversion of {len(mp3_files)} files to high-quality WAV")
        logger.info(f"Delete MP3 after conversion: {delete_mp3}")
    
    print(f"Converting {len(mp3_files)} files to high-quality WAV...")
    
    successful_conversions = 0
    failed_conversions = 0
    skipped_files = 0
    
    for i, mp3_file in enumerate(mp3_files, 1):
        # Create output path by replacing .mp3 with .wav
        wav_file = mp3_file.replace('.mp3', '.wav')
        
        if logger:
            logger.info(f"Processing file {i}/{len(mp3_files)}: {os.path.basename(mp3_file)}")
        
        # Check if output file already exists
        if os.path.exists(wav_file):
            if logger:
                logger.warning(f"WAV file already exists, skipping: {wav_file}")
            print(f"Skipping {os.path.basename(mp3_file)} - .wav file already exists")
            skipped_files += 1
            
            if delete_mp3 and os.path.exists(mp3_file):
                os.remove(mp3_file)
                if logger:
                    logger.info(f"Deleted existing MP3: {mp3_file}")
                print(f"Deleted original MP3: {os.path.basename(mp3_file)}")
            continue
        
        # Convert the file
        if convert_mp3_to_wav(mp3_file, wav_file, logger):
            successful_conversions += 1
            # Delete original MP3 file after successful conversion
            if delete_mp3:
                os.remove(mp3_file)
                if logger:
                    logger.info(f"Deleted original MP3 after successful conversion: {mp3_file}")
                print(f"Deleted original MP3: {os.path.basename(mp3_file)}")
        else:
            failed_conversions += 1
    
    summary = f"Conversion Summary - Successful: {successful_conversions}, Failed: {failed_conversions}, Skipped: {skipped_files}, Total: {len(mp3_files)}"
    
    if logger:
        logger.info(summary)
    
    print(f"\nConversion Summary:")
    print(f"   Successful: {successful_conversions}")
    print(f"   Failed: {failed_conversions}")
    print(f"   Skipped: {skipped_files}")
    print(f"   Total processed: {len(mp3_files)}")
    
    return successful_conversions, failed_conversions

def main():
    parser = argparse.ArgumentParser(description='Convert all MP3 files to high-quality WAV format')
    parser.add_argument('--base-dir', 
                       default='audio_calling/clean_to_speech_text/final_results/audio',
                       help='Base directory containing audio files')
    parser.add_argument('--keep-mp3', action='store_true',
                       help='Keep original MP3 files (default: delete after conversion)')
    parser.add_argument('--confirm', action='store_true',
                       help='Skip confirmation prompt')
    parser.add_argument('--log-dir', 
                       default='audio_calling/clean_to_speech_text/mp3_conversion_logs',
                       help='Directory to store conversion logs')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_dir)
    
    logger.info("=" * 60)
    logger.info("MP3 TO WAV CONVERSION STARTED")
    logger.info(f"Base directory: {args.base_dir}")
    logger.info(f"Log directory: {args.log_dir}")
    logger.info(f"Keep MP3 files: {args.keep_mp3}")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info("=" * 60)
    
    # Check if ffmpeg is available
    if not check_ffmpeg():
        error_msg = "ffmpeg is not installed or not found in PATH"
        logger.error(error_msg)
        print(f"Error: {error_msg}")
        print("Please install ffmpeg first:")
        print("  macOS: brew install ffmpeg")
        print("  Ubuntu/Debian: sudo apt install ffmpeg")
        print("  Windows: Download from https://ffmpeg.org/download.html")
        return
    
    logger.info("ffmpeg check passed")
    
    # Find all .mp3 files
    mp3_files = find_all_mp3_files(args.base_dir, logger)
    
    if not mp3_files:
        logger.warning("No .mp3 files found!")
        print("No .mp3 files found!")
        return
    
    logger.info(f"Found {len(mp3_files)} .mp3 files")
    print(f"Found {len(mp3_files)} .mp3 files")
    
    # Show sample files
    logger.info("Sample .mp3 files found:")
    print("\nSample .mp3 files found:")
    for i, file in enumerate(mp3_files[:5]):
        logger.info(f"  {i+1}. {os.path.basename(file)}")
        print(f"   {i+1}. {os.path.basename(file)}")
    if len(mp3_files) > 5:
        logger.info(f"  ... and {len(mp3_files) - 5} more")
        print(f"   ... and {len(mp3_files) - 5} more")
    
    # Confirmation prompt
    if not args.confirm:
        delete_action = "DELETE" if not args.keep_mp3 else "KEEP"
        response = input(f"\nConvert {len(mp3_files)} files to high-quality WAV and {delete_action} original MP3s? (y/N): ")
        if response.lower() != 'y':
            logger.info("Conversion cancelled by user")
            print("Conversion cancelled")
            return
    
    logger.info("User confirmed conversion")
    
    # Convert files
    successful, failed = convert_files(mp3_files, delete_mp3=not args.keep_mp3, logger=logger)
    
    # Final summary
    final_msg = f"Conversion completed. Successful: {successful}, Failed: {failed}"
    logger.info(final_msg)
    
    if successful > 0:
        logger.info("Conversion completed successfully!")
        print(f"\nConversion completed successfully!")
        if not args.keep_mp3:
            logger.info("Original MP3 files have been deleted")
            print(f"Original MP3 files have been deleted.")
    
    logger.info("=" * 60)
    logger.info("MP3 TO WAV CONVERSION FINISHED")
    logger.info("=" * 60)

if __name__ == "__main__":
    main() 