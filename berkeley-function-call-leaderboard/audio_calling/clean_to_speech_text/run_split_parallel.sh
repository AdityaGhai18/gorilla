#!/bin/bash

# Parallel processing script for split_complex_queries_single_turn.py
# Usage: ./run_split_parallel.sh input_file.json output_prefix chunk_size max_workers

set -e

INPUT_FILE="$1"
OUTPUT_PREFIX="$2"
CHUNK_SIZE="${3:-1000}"
MAX_WORKERS="${4:-8}"

if [ -z "$INPUT_FILE" ] || [ -z "$OUTPUT_PREFIX" ]; then
    echo "Usage: $0 <input_file> <output_prefix> [chunk_size] [max_workers]"
    echo "Example: $0 xlam_data.json split_output 1000 8"
    exit 1
fi

if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file $INPUT_FILE not found"
    exit 1
fi

echo "Starting parallel processing..."
echo "Input file: $INPUT_FILE"
echo "Output prefix: $OUTPUT_PREFIX"
echo "Chunk size: $CHUNK_SIZE"
echo "Max workers: $MAX_WORKERS"

# Get total number of cases
TOTAL_CASES=$(python3 -c "
import json
with open('$INPUT_FILE', 'r') as f:
    data = json.load(f)
print(len(data))
")

echo "Total cases to process: $TOTAL_CASES"

# Create output directory
mkdir -p "$(dirname "$OUTPUT_PREFIX")"

# Function to process a chunk
process_chunk() {
    local start=$1
    local chunk_size=$2
    local worker_id=$3
    
    echo "Worker $worker_id: Processing chunk starting at $start"
    
    python3 split_complex_queries_single_turn.py \
        --input "$INPUT_FILE" \
        --output "${OUTPUT_PREFIX}_chunk_${start}.json" \
        --chunk-start "$start" \
        --chunk-size "$chunk_size" \
        --verbose
    
    echo "Worker $worker_id: Completed chunk starting at $start"
}

# Start parallel processing
pids=()
worker_count=0

for ((start=0; start<TOTAL_CASES; start+=CHUNK_SIZE)); do
    # Wait if we've reached max workers
    if [ ${#pids[@]} -ge $MAX_WORKERS ]; then
        wait ${pids[0]}
        pids=("${pids[@]:1}")  # Remove first PID
    fi
    
    # Start new worker
    process_chunk $start $CHUNK_SIZE $worker_count &
    pids+=($!)
    ((worker_count++))
    
    echo "Started worker $worker_count for chunk $start-$((start+CHUNK_SIZE))"
done

# Wait for all remaining workers
echo "Waiting for all workers to complete..."
for pid in "${pids[@]}"; do
    wait $pid
done

echo "All workers completed. Merging results..."

# Merge all chunk files
python3 -c "
import json
import glob
import sys

output_files = sorted(glob.glob('${OUTPUT_PREFIX}_chunk_*.json'), 
                     key=lambda x: int(x.split('_chunk_')[1].split('.')[0]))

if not output_files:
    print('No output files found to merge')
    sys.exit(1)

merged_data = []
total_complex = 0

for file in output_files:
    print(f'Merging {file}...')
    with open(file, 'r') as f:
        chunk_data = json.load(f)
        merged_data.extend(chunk_data)
        
        # Count complex queries in this chunk
        chunk_complex = sum(1 for case in chunk_data if case.get('split_from_single_turn', False))
        total_complex += chunk_complex
        print(f'  - {len(chunk_data)} cases, {chunk_complex} complex')

# Save merged result
output_file = '${OUTPUT_PREFIX}_merged.json'
with open(output_file, 'w') as f:
    json.dump(merged_data, f, indent=2, ensure_ascii=False)

print(f'\\nMerging complete!')
print(f'Total cases: {len(merged_data)}')
print(f'Complex queries split: {total_complex}')
print(f'Output saved to: {output_file}')
"

# Clean up chunk files
echo "Cleaning up chunk files..."
rm -f "${OUTPUT_PREFIX}_chunk_"*.json

echo "Parallel processing complete!"
echo "Final output: ${OUTPUT_PREFIX}_merged.json"
