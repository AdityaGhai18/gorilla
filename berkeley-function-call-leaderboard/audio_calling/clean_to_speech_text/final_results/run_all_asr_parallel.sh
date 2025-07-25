#!/bin/bash

FINAL_RESULTS_DIR="audio_calling/clean_to_speech_text/final_results"
ASR_LOGS_DIR="$FINAL_RESULTS_DIR/asr_logs"

mkdir -p "$ASR_LOGS_DIR"

# Explicitly list each JSON file to process
# python audio_calling/full_asr_run.py --data_path "$FINAL_RESULTS_DIR/BFCL_v3_java.json"  > "$ASR_LOGS_DIR/BFCL_v3_java_asr.log" 2>&1 &
# python audio_calling/full_asr_run.py --data_path "$FINAL_RESULTS_DIR/BFCL_v3_irrelevance.json"  > "$ASR_LOGS_DIR/BFCL_v3_irrelevance_asr.log" 2>&1 &
# python audio_calling/full_asr_run.py --data_path "$FINAL_RESULTS_DIR/BFCL_v3_javascript.json"  > "$ASR_LOGS_DIR/BFCL_v3_javascript_asr.log" 2>&1 &
# python audio_calling/full_asr_run.py --data_path "$FINAL_RESULTS_DIR/BFCL_v3_live_parallel_multiple.json"  > "$ASR_LOGS_DIR/BFCL_v3_live_parallel_multiple_asr.log" 2>&1 &
# python audio_calling/full_asr_run.py --data_path "$FINAL_RESULTS_DIR/BFCL_v3_live_parallel.json"  > "$ASR_LOGS_DIR/BFCL_v3_live_parallel_asr.log" 2>&1 &
# python audio_calling/full_asr_run.py --data_path "$FINAL_RESULTS_DIR/BFCL_v3_live_relevance.json"  > "$ASR_LOGS_DIR/BFCL_v3_live_relevance_asr.log" 2>&1 &
# python audio_calling/full_asr_run.py --data_path "$FINAL_RESULTS_DIR/BFCL_v3_live_simple.json"  > "$ASR_LOGS_DIR/BFCL_v3_live_simple_asr.log" 2>&1 &
# python audio_calling/full_asr_run.py --data_path "$FINAL_RESULTS_DIR/BFCL_v3_multiple.json"  > "$ASR_LOGS_DIR/BFCL_v3_multiple_asr.log" 2>&1 &
# python audio_calling/full_asr_run.py --data_path "$FINAL_RESULTS_DIR/BFCL_v3_simple.json"  > "$ASR_LOGS_DIR/BFCL_v3_simple_asr.log" 2>&1 &
# python audio_calling/full_asr_run.py --data_path "$FINAL_RESULTS_DIR/BFCL_v3_parallel_multiple.json"  > "$ASR_LOGS_DIR/BFCL_v3_parallel_multiple_asr.log" 2>&1 &
# python audio_calling/full_asr_run.py --data_path "$FINAL_RESULTS_DIR/BFCL_v3_parallel.json"  > "$ASR_LOGS_DIR/BFCL_v3_parallel_asr.log" 2>&1 &

python audio_calling/full_asr_run.py --data_path "$FINAL_RESULTS_DIR/BFCL_v3_live_irrelevance.json"  > "$ASR_LOGS_DIR/BFCL_v3_live_irrelevance_asr.log" 2>&1 &
python audio_calling/full_asr_run.py --data_path "$FINAL_RESULTS_DIR/BFCL_v3_live_multiple.json"  > "$ASR_LOGS_DIR/BFCL_v3_live_multiple_asr.log" 2>&1 &


wait
echo "All ASR jobs finished."

# Print transcript and asr_output for each test case in each file
for json_file in \
#   "$FINAL_RESULTS_DIR/BFCL_v3_java.json" \
#   "$FINAL_RESULTS_DIR/BFCL_v3_irrelevance.json" \
#   "$FINAL_RESULTS_DIR/BFCL_v3_javascript.json" \
#   "$FINAL_RESULTS_DIR/BFCL_v3_live_parallel_multiple.json" \
#   "$FINAL_RESULTS_DIR/BFCL_v3_live_parallel.json" \
#   "$FINAL_RESULTS_DIR/BFCL_v3_live_relevance.json" \
#   "$FINAL_RESULTS_DIR/BFCL_v3_live_simple.json" \
#   "$FINAL_RESULTS_DIR/BFCL_v3_multiple.json" \
#   "$FINAL_RESULTS_DIR/BFCL_v3_simple.json" \
#   "$FINAL_RESULTS_DIR/BFCL_v3_parallel_multiple.json" \
#   "$FINAL_RESULTS_DIR/BFCL_v3_parallel.json"
   "$FINAL_RESULTS_DIR/BFCL_v3_live_irrelevance.json" \
   "$FINAL_RESULTS_DIR/BFCL_v3_live_multiple.json"
  do
  echo "\n==== $json_file ===="
  python -c '
import json
with open("'$json_file'", "r") as f:
    data = json.load(f)
for case in data:
    for turn in case.get("question", [])[0]:
        transcript = turn.get("transcript")
        asr = turn.get("asr_output")
        if transcript or asr:
            print(f"ID: {case.get('id', '')}\n  transcript: {transcript}\n  asr_output: {asr}\n")
' 
done 