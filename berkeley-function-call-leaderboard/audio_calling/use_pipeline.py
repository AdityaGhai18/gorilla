"""
Simple Usage Example: Integrating LLM Judge with Your Existing Pipeline

This script shows how to integrate the LLM judge with your existing pipeline
code for noise correlation analysis.
"""

import json
from pathlib import Path
from judge import create_llm_judge_function

# Import your existing pipeline code
# Note: Adjust the import based on your actual module structure
# from your_pipeline_module import run_full_experiment


def setup_transcription_function():
    """
    Setup your transcription function.
    Replace this with your actual ASR implementation.
    """
    
    # Option 1: Using Whisper
    try:
        import whisper
        model = whisper.load_model("base")
        
        def transcribe_audio(audio_path: str) -> str:
            """Transcribe audio using Whisper."""
            try:
                result = model.transcribe(audio_path)
                return result["text"].strip()
            except Exception as e:
                return f"[TRANSCRIPTION_ERROR: {str(e)}]"
        
        return transcribe_audio
        
    except ImportError:
        print("Whisper not available. Install with: pip install openai-whisper")
    
    # Option 2: Using Google Speech-to-Text
    # try:
    #     from google.cloud import speech
    #     client = speech.SpeechClient()
    #     
    #     def transcribe_audio(audio_path: str) -> str:
    #         with open(audio_path, "rb") as audio_file:
    #             content = audio_file.read()
    #         
    #         audio = speech.RecognitionAudio(content=content)
    #         config = speech.RecognitionConfig(
    #             encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
    #             sample_rate_hertz=16000,
    #             language_code="en-US",
    #         )
    #         
    #         response = client.recognize(config=config, audio=audio)
    #         return response.results[0].alternatives[0].transcript if response.results else ""
    #     
    #     return transcribe_audio
    # except ImportError:
    #     pass
    
    # Fallback: Mock transcription for testing
    def mock_transcribe_audio(audio_path: str) -> str:
        """Mock transcription function for testing."""
        # This would normally call your actual ASR system
        filename = Path(audio_path).name
        return f"[MOCK_TRANSCRIPTION_OF_{filename}]"
    
    print("Using mock transcription. Replace with your actual ASR function.")
    return mock_transcribe_audio


def setup_llm_client():
    """
    Setup your LLM client for advanced analysis.
    This is optional - the judge works without it too.
    """
    
    # Option 1: OpenAI GPT
    try:
        import openai
        import os
        
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            client = openai.OpenAI(api_key=api_key)
            
            class OpenAILLMClient:
                def generate(self, prompt: str) -> str:
                    response = client.chat.completions.create(
                        model="gpt-4-turbo-preview",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.1,
                        max_tokens=1000
                    )
                    return response.choices[0].message.content
            
            return OpenAILLMClient()
    except ImportError:
        print("OpenAI client not available. Install with: pip install openai")
    except Exception as e:
        print(f"OpenAI setup failed: {e}")
    
    # Option 2: Anthropic Claude
    # try:
    #     import anthropic
    #     import os
    #     
    #     api_key = os.getenv("ANTHROPIC_API_KEY")
    #     if api_key:
    #         client = anthropic.Anthropic(api_key=api_key)
    #         
    #         class AnthropicLLMClient:
    #             def generate(self, prompt: str) -> str:
    #                 response = client.messages.create(
    #                     model="claude-3-sonnet-20240229",
    #                     max_tokens=1000,
    #                     messages=[{"role": "user", "content": prompt}]
    #                 )
    #                 return response.content[0].text
    #         
    #         return AnthropicLLMClient()
    # except ImportError:
    #     pass
    
    print("No LLM client available. Judge will work with basic pattern matching.")
    return None


def create_your_llm_judge():
    """
    Create the LLM judge function that's compatible with your pipeline.
    """
    
    # Setup transcription and LLM
    transcribe_fn = setup_transcription_function()
    llm_client = setup_llm_client()
    
    # Create the judge function
    judge_fn = create_llm_judge_function(
        llm_client=llm_client,
        transcribe_fn=transcribe_fn
    )
    
    return judge_fn


def run_noise_correlation_experiment():
    """
    Example of running your noise correlation experiment.
    """
    
    # 1. Setup your data
    audio_dir = "./original_audio_queries"  # Directory with your original audio files
    
    # 2. Create queries map (basename -> original text)
    queries_map = {
        "weather_query": "What is the weather like today in San Francisco?",
        "call_contact": "Please call Dr. Smith at 555-123-4567",
        "set_reminder": "Set a reminder for my dentist appointment tomorrow at 3 PM",
        "play_music": "Play some classical music by Bach",
        "news_request": "What are the top technology news stories today?",
        "navigate": "Navigate to the nearest coffee shop",
        "translate": "How do you say hello in Spanish?",
        "calculate": "What is 15 percent of 245 dollars?",
        "search": "Search for restaurants near me that serve Italian food",
        "schedule": "What's on my calendar for next Tuesday?"
    }
    
    # 3. Define noise effects to test
    effects_spec = {
        "white_noise_light": {
            "fn": "apply_white_noise",  # This should match functions in your background.py
            "variants": [
                {"noise_level": 0.1},
                {"noise_level": 0.2},
                {"noise_level": 0.3}
            ]
        },
        "white_noise_heavy": {
            "fn": "apply_white_noise",
            "variants": [
                {"noise_level": 0.4},
                {"noise_level": 0.5}
            ]
        },
        "background_cafe": {
            "fn": "apply_background_noise",
            "variants": [
                {"noise_type": "cafe", "noise_level": 0.3},
                {"noise_type": "cafe", "noise_level": 0.5}
            ]
        },
        "reverb": {
            "fn": "apply_reverb",
            "variants": [
                {"room_size": "small", "reverb_amount": 0.4},
                {"room_size": "large", "reverb_amount": 0.6}
            ]
        },
        "compression": {
            "fn": "apply_compression",
            "variants": [
                {"ratio": 4, "threshold": -15},
                {"ratio": 8, "threshold": -10}
            ]
        },
        "volume_reduction": {
            "fn": "apply_volume_change",
            "variants": [
                {"volume_factor": 0.5},
                {"volume_factor": 0.3}
            ]
        }
    }
    
    # 4. Create the LLM judge
    llm_judge_fn = create_your_llm_judge()
    
    # 5. Run the experiment using your existing pipeline
    # Note: Replace this with your actual pipeline function call
    
    # Using your existing run_full_experiment function:
    # summary_path = run_full_experiment(
    #     audio_dir=audio_dir,
    #     queries_map=queries_map,
    #     effects_spec=effects_spec,
    #     llm_judge_fn=llm_judge_fn,
    #     transcribe_fn=setup_transcription_function(),  # If needed separately
    #     output_root="./noise_experiment_outputs",
    #     results_dir="./noise_experiment_results"
    # )
    
    # For demonstration, let's create a mock result
    summary_path = "./noise_experiment_results/experiment_summary.json"
    
    # Create mock results to show the expected output format
    mock_results = {
        "by_effect": {
            "white_noise_light": {
                "total": 30,
                "error_counts": {
                    "background_confusion": 12,
                    "phonetic_confusion": 8,
                    "word_substitution": 5,
                    "no_error": 5
                }
            },
            "reverb": {
                "total": 20,
                "error_counts": {
                    "reverb_distortion": 15,
                    "word_deletion": 3,
                    "no_error": 2
                }
            }
        },
        "by_error_type": {
            "background_confusion": {
                "total": 15,
                "effects": {
                    "white_noise_light": 12,
                    "background_cafe": 3
                }
            },
            "reverb_distortion": {
                "total": 15,
                "effects": {
                    "reverb": 15
                }
            }
        }
    }
    
    # Save mock results
    Path("./noise_experiment_results").mkdir(exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(mock_results, f, indent=2)
    
    print(f"Experiment completed! Results saved to: {summary_path}")
    return summary_path


def analyze_correlation_results(summary_path: str):
    """
    Analyze the correlation results from the experiment.
    """
    
    # Load results
    with open(summary_path, 'r') as f:
        results = json.load(f)
    
    print("\n=== NOISE CORRELATION ANALYSIS ===\n")
    
    # Analyze by effect
    print("NOISE TYPE → ERROR TYPE CORRELATIONS:")
    print("-" * 50)
    
    for effect_name, effect_data in results["by_effect"].items():
        total = effect_data["total"]
        error_counts = effect_data["error_counts"]
        
        print(f"\n{effect_name.upper()}:")
        print(f"  Total samples: {total}")
        
        # Sort errors by frequency
        sorted_errors = sorted(
            [(error, count) for error, count in error_counts.items() if error != "no_error"],
            key=lambda x: x[1],
            reverse=True
        )
        
        for error_type, count in sorted_errors:
            percentage = (count / total) * 100
            print(f"  • {error_type}: {count}/{total} ({percentage:.1f}%)")
    
    # Analyze by error type
    print("\n\nERROR TYPE → NOISE TYPE CORRELATIONS:")
    print("-" * 50)
    
    for error_name, error_data in results["by_error_type"].items():
        total = error_data["total"]
        effects = error_data["effects"]
        
        print(f"\n{error_name.upper()}:")
        print(f"  Total occurrences: {total}")
        
        # Sort noise types by frequency
        sorted_effects = sorted(
            effects.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for effect_type, count in sorted_effects:
            percentage = (count / total) * 100
            print(f"  • {effect_type}: {count}/{total} ({percentage:.1f}%)")
    
    # Generate insights
    print("\n\nKEY INSIGHTS:")
    print("-" * 50)
    
    insights = []
    
    # Find strongest correlations
    for effect_name, effect_data in results["by_effect"].items():
        total = effect_data["total"]
        error_counts = effect_data["error_counts"]
        
        for error_type, count in error_counts.items():
            if error_type != "no_error" and total > 0:
                frequency = count / total
                if frequency > 0.5:  # Strong correlation
                    insights.append(f"• {effect_name} strongly correlates with {error_type} ({frequency:.1%})")
                elif frequency > 0.3:  # Moderate correlation
                    insights.append(f"• {effect_name} moderately correlates with {error_type} ({frequency:.1%})")
    
    for insight in insights[:10]:  # Show top 10 insights
        print(insight)
    
    return results


def test_single_judgment():
    """
    Test the judge on a single audio file to see how it works.
    """
    
    print("\n=== TESTING SINGLE JUDGMENT ===\n")
    
    # Create the judge
    judge_fn = create_your_llm_judge()
    
    # Test with mock data
    result = judge_fn(
        variant_audio_path="/path/to/noisy_audio.wav",  # This would be a real path
        original_query_text="What is the weather like today?",
        metadata={
            "effect_key": "white_noise_light",
            "params": {"noise_level": 0.3},
            "variant_idx": 0
        }
    )
    
    print("JUDGMENT RESULT:")
    print(json.dumps(result, indent=2))
    
    return result


if __name__ == "__main__":
    print("LLM Audio Judge Integration Example")
    print("=" * 50)
    
    # Test single judgment
    test_single_judgment()
    
    # Run full experiment
    print("\n" + "=" * 50)
    print("Running noise correlation experiment...")
    summary_path = run_noise_correlation_experiment()
    
    # Analyze results
    print("\n" + "=" * 50)
    print("Analyzing correlation results...")
    analyze_correlation_results(summary_path)
    
    print("\n" + "=" * 50)
    print("Integration example completed!")
    print("\nNext steps:")
    print("1. Replace mock transcription with your actual ASR function")
    print("2. Set up your LLM client (OpenAI, Anthropic, etc.)")
    print("3. Ensure your background.py has the required effect functions")
    print("4. Run with your actual audio files and queries")