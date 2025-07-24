#!/bin/bash

export PYTHONPATH=$(pwd)

python audio_calling/full_run.py \
  --data_path bfcl_eval/data/BFCL_v3_simple.json \
  --output_file audio_calling/clean_to_speech_text/final_results/BFCL_v3_simple.json \
  --audio_dir audio_calling/clean_to_speech_text/final_results/audio/BFCL_v3_simple \
  --sample_count None > simple.log 2>&1 &

python audio_calling/full_run.py \
  --data_path bfcl_eval/data/BFCL_v3_live_irrelevance.json \
  --output_file audio_calling/clean_to_speech_text/final_results/BFCL_v3_live_irrelevance.json \
  --audio_dir audio_calling/clean_to_speech_text/final_results/audio/BFCL_v3_live_irrelevance \
  --sample_count None > live_irrelevance.log 2>&1 &

python audio_calling/full_run.py \
  --data_path bfcl_eval/data/BFCL_v3_live_multiple.json \
  --output_file audio_calling/clean_to_speech_text/final_results/BFCL_v3_live_multiple.json \
  --audio_dir audio_calling/clean_to_speech_text/final_results/audio/BFCL_v3_live_multiple \
  --sample_count None > live_multiple.log 2>&1 &

wait
echo "All selected jobs finished." 