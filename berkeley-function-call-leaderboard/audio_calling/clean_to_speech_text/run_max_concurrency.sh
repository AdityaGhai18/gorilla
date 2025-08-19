#!/bin/bash

# Maximum Concurrency Pipeline Runner
# This runs 250 workers simultaneously for maximum speed

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

if [ $# -eq 0 ]; then
    echo "Usage: $0 <file> [--test]"
    echo ""
    echo "Options:"
    echo "  <file>           - JSON file to process"
    echo "  --test           - Test mode (100 workers, 100 cases)"
    echo ""
    echo "Examples:"
    echo "  $0 apigen_mt_5k.json"
    echo "  $0 apigen_mt_5k.json --test"
    exit 1
fi

FILE_PATH=$1
shift

if [ ! -f "$FILE_PATH" ]; then
    echo "Error: File '$FILE_PATH' not found!"
    exit 1
fi

print_info "Starting Maximum Concurrency Pipeline"
print_info "File: $FILE_PATH"

if [[ "$*" == *"--test"* ]]; then
    print_warning "TEST MODE: Processing 100 cases with 100 workers"
    python max_concurrency_pipeline.py --file_path "$FILE_PATH" --test
else
    print_info "PRODUCTION MODE: Processing ALL cases with up to 250 workers"
    print_warning "This will launch 250 parallel processes simultaneously!"
    print_warning "Make sure your system can handle this load."
    echo
    read -p "Continue with maximum concurrency? (y/N): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python max_concurrency_pipeline.py --file_path "$FILE_PATH"
    else
        print_info "Cancelled by user"
        exit 0
    fi
fi

print_success "Pipeline completed!"
print_info "Check the logs in max_concurrency_pipeline.log"
print_info "Results saved with '_processed.json' suffix"
