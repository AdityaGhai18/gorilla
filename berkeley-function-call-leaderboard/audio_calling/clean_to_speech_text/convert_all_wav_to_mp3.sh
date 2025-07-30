#!/bin/bash

# WAV to MP3 Conversion Script for All Audio Folders
# This script converts all WAV files in the 17 audio folders to MP3 format
# and updates the corresponding JSON files with new audio paths.

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/convert_wav_to_mp3.py"
AUDIO_BASE_DIR="$SCRIPT_DIR/final_results/audio"
JSON_DIR="$SCRIPT_DIR/final_results"
DELETE_WAV=false  # Set to true to delete original WAV files after conversion
MAX_WORKERS=4  # Number of parallel processes per folder
TEST_MODE=false  # Set to true to only process 3 files per folder

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -d, --delete-wav         Delete original WAV files after conversion"
    echo "  -w, --max-workers NUM    Maximum workers per folder (default: 4)"
    echo "  -t, --test               Test mode: only process 3 files per folder"
    echo "  --dry-run                Show what would be done without converting"
    echo "  -h, --help               Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Convert with best quality (320kbps)"
    echo "  $0 --delete-wav                       # Best quality, delete WAV files"
    echo "  $0 --test                              # Test mode: 3 files per folder"
    echo "  $0 --dry-run                          # Preview what would be done"
    echo "  $0 --max-workers 8                    # Use 8 workers per folder"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--delete-wav)
            DELETE_WAV=true
            shift
            ;;
        -w|--max-workers)
            MAX_WORKERS="$2"
            shift 2
            ;;
        -t|--test)
            TEST_MODE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate paths
if [[ ! -f "$PYTHON_SCRIPT" ]]; then
    print_error "Python script not found: $PYTHON_SCRIPT"
    exit 1
fi

if [[ ! -d "$AUDIO_BASE_DIR" ]]; then
    print_error "Audio base directory not found: $AUDIO_BASE_DIR"
    exit 1
fi

if [[ ! -d "$JSON_DIR" ]]; then
    print_error "JSON directory not found: $JSON_DIR"
    exit 1
fi

# Get list of audio folders
print_status "Scanning for audio folders in $AUDIO_BASE_DIR..."
AUDIO_FOLDERS=()
while IFS= read -r -d '' folder; do
    if [[ -d "$folder" ]] && [[ -n "$(find "$folder" -name "*.wav" -print -quit)" ]]; then
        AUDIO_FOLDERS+=("$folder")
    fi
done < <(find "$AUDIO_BASE_DIR" -maxdepth 1 -type d -print0)

if [[ ${#AUDIO_FOLDERS[@]} -eq 0 ]]; then
    print_error "No folders with WAV files found in $AUDIO_BASE_DIR"
    exit 1
fi

print_success "Found ${#AUDIO_FOLDERS[@]} folders with WAV files:"
for folder in "${AUDIO_FOLDERS[@]}"; do
    folder_name=$(basename "$folder")
    wav_count=$(find "$folder" -name "*.wav" | wc -l)
    print_status "  $folder_name: $wav_count WAV files"
done

# Build command arguments
CMD_ARGS=()
if [[ "$DELETE_WAV" == true ]]; then
    CMD_ARGS+=("--delete-wav")
fi

if [[ -n "$DRY_RUN" ]]; then
    CMD_ARGS+=("--dry-run")
fi

CMD_ARGS+=("--max-workers" "$MAX_WORKERS")
CMD_ARGS+=("--json-dir" "$JSON_DIR")

# Function to process a single folder
process_folder() {
    local folder="$1"
    local folder_name=$(basename "$folder")
    
    print_status "Processing folder: $folder_name"
    
    # Add test mode argument if enabled
    local python_args=("$folder" "${CMD_ARGS[@]}")
    if [[ "$TEST_MODE" == true ]]; then
        python_args+=("--test-mode")
    fi
    
    if [[ -n "$DRY_RUN" ]]; then
        python3 "$PYTHON_SCRIPT" "${python_args[@]}"
    else
        if python3 "$PYTHON_SCRIPT" "${python_args[@]}"; then
            print_success "Completed: $folder_name"
        else
            print_error "Failed: $folder_name"
            return 1
        fi
    fi
}

# Main execution
print_status "Starting WAV to MP3 conversion..."
print_status "Configuration:"
print_status "  Quality: 320kbps (best)"
print_status "  Delete WAV: $DELETE_WAV"
print_status "  Max workers per folder: $MAX_WORKERS"
print_status "  Test mode: ${TEST_MODE:-false}"
print_status "  Dry run: ${DRY_RUN:-false}"

if [[ -n "$DRY_RUN" ]]; then
    print_warning "DRY RUN MODE - No files will be converted"
fi

if [[ "$TEST_MODE" == true ]]; then
    print_warning "TEST MODE - Only 3 files per folder will be processed"
fi

echo ""

# Process folders
FAILED_FOLDERS=()
SUCCESSFUL_FOLDERS=()

for folder in "${AUDIO_FOLDERS[@]}"; do
    folder_name=$(basename "$folder")
    
    echo "=========================================="
    print_status "Processing $folder_name..."
    echo "=========================================="
    
    if process_folder "$folder"; then
        SUCCESSFUL_FOLDERS+=("$folder_name")
    else
        FAILED_FOLDERS+=("$folder_name")
    fi
    
    echo ""
done

# Summary
echo "=========================================="
print_status "CONVERSION SUMMARY"
echo "=========================================="
print_status "Total folders: ${#AUDIO_FOLDERS[@]}"
print_success "Successful: ${#SUCCESSFUL_FOLDERS[@]}"

if [[ ${#SUCCESSFUL_FOLDERS[@]} -gt 0 ]]; then
    print_status "Successfully processed folders:"
    for folder in "${SUCCESSFUL_FOLDERS[@]}"; do
        print_success "  ✓ $folder"
    done
fi

if [[ ${#FAILED_FOLDERS[@]} -gt 0 ]]; then
    print_error "Failed: ${#FAILED_FOLDERS[@]}"
    print_error "Failed folders:"
    for folder in "${FAILED_FOLDERS[@]}"; do
        print_error "  ✗ $folder"
    done
    exit 1
else
    print_success "All folders processed successfully!"
fi 