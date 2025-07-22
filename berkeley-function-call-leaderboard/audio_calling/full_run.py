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

data_path = "bfcl_eval/data/BFCL_v3_live_multiple.json"  
output_file = "audio_calling/clean_to_speech_text/new_results/BFCL_v3_live_multiple.json" 
sample_count = 7  

# === Clean text to speech-like text ===


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

# # --- Uncomment to filter for a specific test case by ID ---
# specific_id = "live_simple_240-125-3"
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

# Post-process and print the transcript of the first test case for inspection
if processed and processed[0]['question'][0][0].get('transcript'):
    text = processed[0]['question'][0][0]['transcript']
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    print("First transcript (post-processed):")
    print(text)

# === TTS with Openai ===
from audio_calling.TTS.scripts.tts_generator_openai import OpenAITTSGenerator
import os

def get_api_key(varname):
    api_key = os.getenv(varname)
    if not api_key:
        raise EnvironmentError(f"{varname} environment variable not set.")
    return api_key

PROVIDER = "openai"
NUM_CASES = sample_count if sample_count is not None else len(processed)
OUTPUT_ROOT = "audio_calling/clean_to_speech_text/new_results/audio/BFCL_v3_live_multiple.json"

voice = os.getenv("OPENAI_TTS_VOICE", "coral")
model = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
instructions = os.getenv("OPENAI_TTS_INSTRUCTIONS", "Speak in a natural, clear tone.")
api_key = get_api_key("OPENAI_API_KEY")
generator = OpenAITTSGenerator(api_key, output_root=OUTPUT_ROOT, model=model, voice=voice, instructions=instructions)
generator.run(output_file, NUM_CASES)
print("TTS audio generation complete!") 