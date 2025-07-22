import json
import os
from audio_calling.clean_to_speech_text.granular_speech_pipeline import GranularSpeechPipeline, PipelineConfig

def load_bfcl_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            return json.load(f)
        else:
            return [json.loads(line) for line in f if line.strip()]

def save_transformed_data(data, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

data_path = "data/BFCL_v3_live_simple.json"  # Set your input file here
output_file = "audio_calling/clean_to_speech_text/new_results/FCL_v3_live_simple.json"  # Set your output file here
sample_count = 1  # Set to an integer for a random sample, or None for all

config = PipelineConfig(
    max_features=6,
    confidence_threshold=0.5,
    temperature=0.8,
    max_retries=3,
    retry_delay=1.0,
    asr=False
)
pipeline = GranularSpeechPipeline(config)

print(f"Loading data from: {data_path}")
data = load_bfcl_data(data_path)
print(f"Loaded {len(data)} test cases.")

# --- Uncomment to filter for a specific test case by ID ---
# specific_id = "live_simple_183-108-0"
# data = [case for case in data if case["id"] == specific_id]
# print(f"Filtered to {len(data)} test case(s) with id {specific_id}.")

# Filter for cases where at least one turn in the first group has role == 'user'
def has_user_role(test_case):
    return any(turn.get("role") == "user" for turn in test_case["question"][0])

user_role_data = [case for case in data if has_user_role(case)]
print(f"Filtered to {len(user_role_data)} cases with at least one 'user' role. Excluded {len(data) - len(user_role_data)} cases.")

# Now filter for English test cases using the pipeline
filtered_data = GranularSpeechPipeline.filter_english_test_cases(pipeline, user_role_data)
print(f"Filtered to {len(filtered_data)} English test cases. Excluded {len(user_role_data) - len(filtered_data)} cases.")

print("Processing with granular speech pipeline...")
processed = pipeline.transform_dataset_for_clean_output(filtered_data, sample_count=sample_count)

print(f"Saving results to: {output_file}")
save_transformed_data(processed, output_file)
print("Done!") 