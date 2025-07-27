#!/bin/bash

export PYTHONPATH=$(pwd)

# Run 4 full_run.py jobs in parallel, each with its own data/output file and log

python audio_calling/full_run.py \
  --data_path bfcl_eval/data/BFCL_v3_parallel.json \
  --output_file audio_calling/clean_to_speech_text/final_results/BFCL_v3_parallel.json \
  --audio_dir audio_calling/clean_to_speech_text/final_results/audio/BFCL_v3_parallel \
  --sample_count None > parallel.log 2>&1 &

# python audio_calling/full_run.py \
#   --data_path bfcl_eval/data/BFCL_v3_simple.json \
#   --output_file audio_calling/clean_to_speech_text/final_results/BFCL_v3_simple.json \
#   --audio_dir audio_calling/clean_to_speech_text/final_results/audio/BFCL_v3_simple \
#   --sample_count -1 > simple.log 2>&1 &

python audio_calling/full_run.py \
  --data_path bfcl_eval/data/BFCL_v3_parallel_multiple.json \
  --output_file audio_calling/clean_to_speech_text/final_results/BFCL_v3_parallel_multiple.json \
  --audio_dir audio_calling/clean_to_speech_text/final_results/audio/BFCL_v3_parallel_multiple \
  --sample_count None > parallel_multiple.log 2>&1 &

python audio_calling/full_run.py \
  --data_path bfcl_eval/data/BFCL_v3_multiple.json \
  --output_file audio_calling/clean_to_speech_text/final_results/BFCL_v3_multiple.json \
  --audio_dir audio_calling/clean_to_speech_text/final_results/audio/BFCL_v3_multiple \
  --sample_count None > multiple.log 2>&1 &

wait
echo "All jobs finished." 