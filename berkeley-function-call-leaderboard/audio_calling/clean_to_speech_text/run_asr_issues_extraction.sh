#!/bin/bash

# Script to run ASR issues extraction on specific test case files
# Processes: live_multiple, live_simple, and multi_turn_long_context

FINAL_RESULTS_DIR="final_results"
OUTPUT_DIR="$FINAL_RESULTS_DIR/clarif_necessary"
LOG_DIR="$FINAL_RESULTS_DIR/extraction_logs"



# Create output and log directories if they don't exist
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# List of specific JSON files to process
JSON_FILES=(
    "BFCL_v3_live_multiple.json"
    "BFCL_v3_live_simple.json"
    "BFCL_v3_multi_turn_long_context.json"
)

# Start all processes in parallel
for file in "${JSON_FILES[@]}"; do
    file_path="$FINAL_RESULTS_DIR/$file"
    
    if [ -f "$file_path" ]; then
        base_name=$(echo "$file" | sed 's/BFCL_v3_//' | sed 's/\.json$//')
        output_file="$OUTPUT_DIR/clarif_necessary_${base_name}.json"
        log_file="$LOG_DIR/${base_name}_extraction.log"
        
        # Run the ASR issues extraction script in parallel
        python3 extract_asr_issues_llm.py "$file_path" "$output_file" > "$log_file" 2>&1 &
        pid=$!
        
        echo "Started processing $file (PID: $pid)"
        
        # Start monitoring this log file in real-time
        tail -f "$log_file" --pid=$pid &
        
    else
        echo "File not found: $file_path"
    fi
done

# Wait for all processes to complete
echo "All processes started. Monitoring logs in real-time..."
echo "Press Ctrl+C to stop monitoring (processes will continue in background)"
wait

echo ""
echo "All processes completed. Results saved to $OUTPUT_DIR"
echo "Logs saved to $LOG_DIR"
