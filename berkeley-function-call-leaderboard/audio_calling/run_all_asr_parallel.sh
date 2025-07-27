#!/bin/bash

# Script to run ASR processing in parallel across all JSON files
# This will add Deepgram transcriptions to all files

FINAL_RESULTS_DIR="audio_calling/clean_to_speech_text/final_results"
ASR_LOGS_DIR="$FINAL_RESULTS_DIR/asr_logs"

# Create logs directory
mkdir -p "$ASR_LOGS_DIR"

echo "Starting parallel ASR processing for all JSON files..."
echo "Logs will be saved to: $ASR_LOGS_DIR"
echo "=" * 60

# List of all JSON files to process
JSON_FILES=(
    "BFCL_v3_irrelevance.json"
    "BFCL_v3_java.json"
    "BFCL_v3_javascript.json"
    "BFCL_v3_live_irrelevance.json"
    "BFCL_v3_live_multiple.json"
    "BFCL_v3_live_parallel.json"
    "BFCL_v3_live_parallel_multiple.json"
    "BFCL_v3_live_relevance.json"
    "BFCL_v3_live_simple.json"
    "BFCL_v3_multi_turn_base.json"
    "BFCL_v3_multi_turn_long_context.json"
    "BFCL_v3_multi_turn_miss_func.json"
    "BFCL_v3_multi_turn_miss_param.json"
    "BFCL_v3_multiple.json"
    "BFCL_v3_parallel.json"
    "BFCL_v3_parallel_multiple.json"
    "BFCL_v3_simple.json"
)

# Process each JSON file in parallel
for file in "${JSON_FILES[@]}"; do
    file_path="$FINAL_RESULTS_DIR/$file"
    
    if [ -f "$file_path" ]; then
        echo "Starting ASR processing for: $file"
        log_file="$ASR_LOGS_DIR/${file%.json}_asr.log"
        
        # Run the ASR script in background and log output
        # FULL RUN: Process ALL test cases (no test mode)
        python audio_calling/full_asr_run.py --data_path "$file_path" > "$log_file" 2>&1 &
        
        echo "  -> Background process started, log: $log_file"
    else
        echo "WARNING: File not found: $file_path"
    fi
done

echo ""
echo "All ASR processes started in background."
echo "Waiting for all processes to complete..."

# Wait for all background processes to finish
wait

echo ""
echo "=" * 60
echo "All ASR processing completed!"
echo "Check logs in: $ASR_LOGS_DIR"

# Print summary of log files
echo ""
echo "Log files created:"
ls -la "$ASR_LOGS_DIR"/*.log 2>/dev/null || echo "No log files found"

# Print a quick summary of what was processed
echo ""
echo "Files processed:"
for file in "${JSON_FILES[@]}"; do
    if [ -f "$FINAL_RESULTS_DIR/$file" ]; then
        echo "  ✅ $file"
    else
        echo "  ❌ $file (not found)"
    fi
done 