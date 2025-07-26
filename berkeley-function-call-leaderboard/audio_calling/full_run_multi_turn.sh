#!/bin/bash
set -e

export PYTHONPATH=$(pwd)

# Multi-turn base
python3 audio_calling/full_run_multi_turn.py \
  --data_path "bfcl_eval/data/BFCL_v3_multi_turn_base.json" \
  --output_file "audio_calling/clean_to_speech_text/final_results/BFCL_v3_multi_turn_base.json" \
  --sample_count None \
  --audio_dir "audio_calling/clean_to_speech_text/final_results/audio/BFCL_v3_multi_turn_base" \
  > multi_turn_base.log 2>&1 &

# Multi-turn long context
python3 audio_calling/full_run_multi_turn.py \
  --data_path "bfcl_eval/data/BFCL_v3_multi_turn_long_context.json" \
  --output_file "audio_calling/clean_to_speech_text/final_results/BFCL_v3_multi_turn_long_context.json" \
  --sample_count None \
  --audio_dir "audio_calling/clean_to_speech_text/final_results/audio/BFCL_v3_multi_turn_long_context" \
  > multi_turn_long_context.log 2>&1 &

# Multi-turn miss func
python3 audio_calling/full_run_multi_turn.py \
  --data_path "bfcl_eval/data/BFCL_v3_multi_turn_miss_func.json" \
  --output_file "audio_calling/clean_to_speech_text/final_results/BFCL_v3_multi_turn_miss_func.json" \
  --sample_count None \
  --audio_dir "audio_calling/clean_to_speech_text/final_results/audio/BFCL_v3_multi_turn_miss_func" \
  > multi_turn_miss_func.log 2>&1 &

# Multi-turn miss param
python3 audio_calling/full_run_multi_turn.py \
  --data_path "bfcl_eval/data/BFCL_v3_multi_turn_miss_param.json" \
  --output_file "audio_calling/clean_to_speech_text/final_results/BFCL_v3_multi_turn_miss_param.json" \
  --sample_count None \
  --audio_dir "audio_calling/clean_to_speech_text/final_results/audio/BFCL_v3_multi_turn_miss_param" \
  > multi_turn_miss_param.log 2>&1 &

wait
echo "All multi-turn datasets processed in parallel." 