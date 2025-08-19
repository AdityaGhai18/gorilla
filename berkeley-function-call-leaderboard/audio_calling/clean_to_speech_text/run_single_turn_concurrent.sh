#!/bin/bash

# Concurrent Single-Turn to Multi-Turn Converter Runner
# This runs the conversion with maximum concurrency and robust error handling

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
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

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Load environment variables from .env file if it exists
load_env() {
    local env_file=".env"
    local found_file=""
    
    # Check in current directory
    if [ -f "$env_file" ]; then
        found_file="$env_file"
    else
        # Check in parent directories (up to 3 levels)
        for i in {1..3}; do
            local parent_path=""
            for j in $(seq 1 $i); do
                parent_path="../$parent_path"
            done
            
            if [ -f "${parent_path}$env_file" ]; then
                found_file="${parent_path}$env_file"
                break
            fi
        done
    fi
    
    if [ -n "$found_file" ]; then
        print_info "Loading environment variables from $found_file"
        
        # Read .env file line by line and export variables
        while IFS= read -r line || [ -n "$line" ]; do
            # Skip empty lines and comments
            if [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]]; then
                continue
            fi
            
            # Check if line contains an assignment
            if [[ "$line" =~ ^[[:space:]]*([A-Za-z_][A-Za-z0-9_]*)[[:space:]]*=[[:space:]]*(.*)[[:space:]]*$ ]]; then
                local var_name="${BASH_REMATCH[1]}"
                local var_value="${BASH_REMATCH[2]}"
                
                # Remove surrounding quotes if present
                if [[ "$var_value" =~ ^\"(.*)\"$ ]] || [[ "$var_value" =~ ^\'(.*)\'$ ]]; then
                    var_value="${BASH_REMATCH[1]}"
                fi
                
                # Export the variable
                export "$var_name"="$var_value"
            fi
        done < "$found_file"
        
        return 0
    else
        print_warning "No .env file found in current or parent directories"
        return 1
    fi
}

# Usage function
usage() {
    echo "Usage: $0 <input_file> <output_file> [options]"
    echo ""
    echo "Options:"
    echo "  --test           - Test mode (100 workers, 100 cases)"
    echo "  --workers N      - Number of workers (default: 250)"
    echo "  --help           - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 BFCL_v3_live_simple.json multiturn_simple.json"
    echo "  $0 data.json output.json --test"
    echo "  $0 data.json output.json --workers 100"
    exit 1
}

# Check arguments
if [ $# -lt 2 ]; then
    print_error "Insufficient arguments"
    usage
fi

INPUT_FILE=$1
OUTPUT_FILE=$2
shift 2

# Parse additional options
TEST_MODE=""
WORKERS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --test)
            TEST_MODE="--test"
            shift
            ;;
        --workers)
            WORKERS="--workers $2"
            shift 2
            ;;
        --help)
            usage
            ;;
        *)
            print_error "Unknown option: $1"
            usage
            ;;
    esac
done

# Load environment variables
load_env

# Validate input file
if [ ! -f "$INPUT_FILE" ]; then
    print_error "Input file '$INPUT_FILE' not found!"
    exit 1
fi

# Check if output directory exists, create if not
OUTPUT_DIR=$(dirname "$OUTPUT_FILE")
if [ ! -d "$OUTPUT_DIR" ]; then
    print_info "Creating output directory: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
fi

# Check for required environment variables
if [ -z "$OPENAI_API_KEY" ]; then
    print_error "OPENAI_API_KEY environment variable not set!"
    print_info "Please set your OpenAI API key in a .env file or export it:"
    print_info "echo 'OPENAI_API_KEY=your-api-key-here' > .env"
    print_info "or"
    print_info "export OPENAI_API_KEY='your-api-key-here'"
    exit 1
fi

# Check if Python script exists
SCRIPT_PATH="$(dirname "$0")/single_turn_concurrent.py"
if [ ! -f "$SCRIPT_PATH" ]; then
    print_error "Script not found: $SCRIPT_PATH"
    exit 1
fi

# Display configuration
print_info "=== Single-Turn to Multi-Turn Converter ==="
print_info "Input file: $INPUT_FILE"
print_info "Output file: $OUTPUT_FILE"

if [ -n "$TEST_MODE" ]; then
    print_warning "Running in TEST MODE (100 cases, limited workers)"
fi

if [ -n "$WORKERS" ]; then
    print_info "Custom worker count: $WORKERS"
else
    print_info "Using default workers: 250"
fi

# Check file size and warn if large
FILE_SIZE=$(wc -l < "$INPUT_FILE" 2>/dev/null || echo "0")
if [ "$FILE_SIZE" -gt 10000 ] && [ -z "$TEST_MODE" ]; then
    print_warning "Large input file detected ($FILE_SIZE lines)"
    print_warning "This may take significant time and API costs"
    print_warning "Consider using --test mode first"
    echo -n "Continue? (y/N): "
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        print_info "Aborted by user"
        exit 0
    fi
fi

# Run the conversion
print_info "Starting conversion..."
print_info "Command: python $SCRIPT_PATH --input \"$INPUT_FILE\" --output \"$OUTPUT_FILE\" $TEST_MODE $WORKERS"

# Create log file name
LOG_FILE="${OUTPUT_FILE}.log"
print_info "Logs will be saved to: $LOG_FILE"

# Run the Python script and capture output
if python "$SCRIPT_PATH" --input "$INPUT_FILE" --output "$OUTPUT_FILE" $TEST_MODE $WORKERS 2>&1 | tee "$LOG_FILE"; then
    print_success "Conversion completed successfully!"
    
    # Show summary if output file exists
    if [ -f "$OUTPUT_FILE" ]; then
        OUTPUT_COUNT=$(wc -l < "$OUTPUT_FILE" 2>/dev/null || echo "0")
        print_success "Generated $OUTPUT_COUNT multi-turn conversations"
        print_info "Output saved to: $OUTPUT_FILE"
    fi
    
    # Check for checkpoint file
    CHECKPOINT_FILE="${OUTPUT_FILE}.checkpoint"
    if [ -f "$CHECKPOINT_FILE" ]; then
        print_info "Checkpoint file available: $CHECKPOINT_FILE"
        print_info "You can resume processing if interrupted"
    fi
    
else
    print_error "Conversion failed!"
    print_error "Check the log file for details: $LOG_FILE"
    
    # Check for partial results
    if [ -f "$OUTPUT_FILE" ]; then
        PARTIAL_COUNT=$(wc -l < "$OUTPUT_FILE" 2>/dev/null || echo "0")
        if [ "$PARTIAL_COUNT" -gt 0 ]; then
            print_warning "Partial results available: $PARTIAL_COUNT entries in $OUTPUT_FILE"
        fi
    fi
    
    exit 1
fi

print_success "=== Conversion Complete ==="
