"""
Audio-to-Audio Transcription Comparison with Intent Analysis

This system compares transcriptions from original audio vs noisy audio
to analyze how noise affects transcription quality and intent recognition.
"""

import json
import os
import csv
from pathlib import Path
from typing import Dict, List, Any, Callable, Tuple
import time
from dataclasses import dataclass
import openai
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import concurrent.futures

try:
    import requests
    HAS_DEEPGRAM = True
except ImportError:
    HAS_DEEPGRAM = False

# Import your existing pipeline components
try:
    from generate_variants import generate_variants
    print("✅ Using your existing background.py pipeline")
except ImportError:
    print("⚠️  Could not import generate_variants. Using fallback.")
    generate_variants = None


@dataclass
class AudioComparisonConfig:
    """Configuration for audio-to-audio comparison evaluation."""
    openai_api_key: str
    audio_dir: str
    effects_spec: Dict[str, Dict]
    output_root: str = "audio_comparison_outputs"
    results_dir: str = "audio_comparison_results"
    transcription_model: str = "whisper-1"
    analysis_model: str = "gpt-4-turbo-preview"
    audio_extensions: List[str] = None

    def __post_init__(self):
        if self.audio_extensions is None:
            self.audio_extensions = [".wav", ".mp3", ".flac", ".m4a"]


class IntentAnalyzer:
    """Analyzes intent preservation between original and noisy audio transcriptions."""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
        
    def analyze_intent_preservation(self, original_transcript: str, noisy_transcript: str, 
                                  noise_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze if the intent is preserved between original and noisy transcriptions."""
        
        prompt = f"""
You are an expert at analyzing speech intent recognition. Compare these two transcriptions of the same audio content and analyze how noise affected intent understanding.

ORIGINAL AUDIO TRANSCRIPTION: "{original_transcript}"
NOISY AUDIO TRANSCRIPTION: "{noisy_transcript}"

NOISE APPLIED:
- Type: {noise_metadata.get('effect_key', 'unknown')}
- Parameters: {json.dumps(noise_metadata.get('params', {}), indent=2)}

Please analyze and respond in JSON format:

{{
    "intent_preserved": true/false,
    "intent_confidence": 0.95,
    "semantic_similarity": 0.85,
    "key_information_lost": ["specific info that was lost"],
    "error_types": ["transcription_degradation", "word_substitution", "entity_confusion", etc.],
    "noise_impact_severity": "low/medium/high",
    "intent_category": "command/question/dictation/navigation/etc",
    "critical_words_affected": ["important words that changed"],
    "reasoning": "Detailed explanation of your analysis"
}}

Error types to consider:
- transcription_degradation: Overall quality decrease
- word_substitution: Wrong words used
- word_deletion: Missing words
- word_insertion: Extra words added
- phonetic_confusion: Similar-sounding words confused
- entity_confusion: Names, numbers, places misheard
- command_confusion: Action/intent changed
- semantic_drift: Meaning changed
- background_confusion: Background noise causing confusion
- volume_distortion: Volume-related distortion effects
- frequency_distortion: Frequency filtering effects
- reverb_distortion: Echo/reverb causing issues
- partial_loss: Some content lost
- complete_failure: Completely unintelligible
- no_significant_impact: Noise didn't affect understanding
"""
        
        try:
            response = self.llm_client.generate(prompt)
            # Parse JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {"error": "Could not parse LLM response", "raw_response": response}
        except Exception as e:
            return {"error": f"Intent analysis failed: {str(e)}"}


class AudioComparisonEvaluator:
    """Main class for audio-to-audio transcription comparison."""
    
    def __init__(self, config: AudioComparisonConfig, use_deepgram: bool = False, deepgram_api_key: str = None):
        self.config = config
        self.use_deepgram = use_deepgram
        self.deepgram_api_key = deepgram_api_key
        self.transcription_service = self._setup_transcription_service()
        self.analysis_service = self._setup_analysis_service()
        self.intent_analyzer = IntentAnalyzer(self.analysis_service)

    def _setup_transcription_service(self):
        """Setup transcription service (OpenAI Whisper or Deepgram)."""
        if self.use_deepgram:
            if not HAS_DEEPGRAM:
                raise ImportError("requests package not installed (required for Deepgram).")
            if not self.deepgram_api_key:
                raise ValueError("Deepgram API key required for Deepgram transcription.")
            class DeepgramTranscriptionService:
                def __init__(self, api_key: str):
                    self.api_key = api_key
                    self.url = "https://api.deepgram.com/v1/listen"
                def transcribe_audio(self, audio_path: str) -> str:
                    try:
                        with open(audio_path, "rb") as audio_file:
                            headers = {
                                "Authorization": f"Token {self.api_key}",
                                "Content-Type": "audio/mp3" if audio_path.endswith(".mp3") else "application/octet-stream"
                            }
                            params = {
                                "punctuate": "true",
                                "model": "general"
                            }
                            response = requests.post(
                                self.url,
                                headers=headers,
                                params=params,
                                data=audio_file
                            )
                        if response.status_code != 200:
                            return f"[DEEPGRAM_TRANSCRIPTION_ERROR: HTTP {response.status_code} {response.text}]"
                        dg_json = response.json()
                        return dg_json['results']['channels'][0]['alternatives'][0].get('transcript', '')
                    except Exception as e:
                        return f"[DEEPGRAM_TRANSCRIPTION_ERROR: {str(e)}]"
            return DeepgramTranscriptionService(self.deepgram_api_key)
        else:
            class OpenAITranscriptionService:
                def __init__(self, api_key: str, model: str = "whisper-1"):
                    self.client = openai.OpenAI(api_key=api_key)
                    self.model = model
                
                def transcribe_audio(self, audio_path: str) -> str:
                    try:
                        with open(audio_path, "rb") as audio_file:
                            transcript = self.client.audio.transcriptions.create(
                                model=self.model,
                                file=audio_file,
                                response_format="text"
                            )
                        return transcript.strip()
                    except Exception as e:
                        return f"[TRANSCRIPTION_ERROR: {str(e)}]"
        
        return OpenAITranscriptionService(self.config.openai_api_key, self.config.transcription_model)
    
    def _setup_analysis_service(self):
        """Setup OpenAI analysis service for intent analysis."""
        class OpenAIAnalysisService:
            def __init__(self, api_key: str, model: str = "gpt-4-turbo-preview"):
                self.client = openai.OpenAI(api_key=api_key)
                self.model = model
                
            def generate(self, prompt: str) -> str:
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": "You are an expert at analyzing speech transcription quality and intent preservation. Always respond with valid JSON."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.1,
                        max_tokens=1500
                    )
                    return response.choices[0].message.content
                except Exception as e:
                    return f'{{"error": "LLM analysis failed: {str(e)}"}}'
        
        return OpenAIAnalysisService(self.config.openai_api_key, self.config.analysis_model)
    
    def discover_audio_files(self) -> List[str]:
        """Discover audio files in the specified directory."""
        audio_dir = Path(self.config.audio_dir)
        if not audio_dir.exists():
            raise FileNotFoundError(f"Audio directory not found: {audio_dir}")
        
        audio_files = []
        for ext in self.config.audio_extensions:
            audio_files.extend(audio_dir.rglob(f"*{ext}"))
        
        if not audio_files:
            raise RuntimeError(f"No audio files found in {audio_dir}")
        
        print(f"Found {len(audio_files)} audio files to process")
        return [str(f) for f in audio_files]
    
    def transcribe_original_audio(self, audio_files: List[str]) -> Dict[str, str]:
        """Transcribe all original audio files to get baseline transcriptions (parallelized)."""
        print("🎤 Transcribing original audio files...")

        def transcribe_one(audio_path):
            audio_base = Path(audio_path).stem
            transcript = self.transcription_service.transcribe_audio(audio_path)
            return audio_base, transcript

        original_transcripts = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
            futures = {executor.submit(transcribe_one, audio_path): audio_path for audio_path in audio_files}
            for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
                audio_base, transcript = future.result()
                original_transcripts[audio_base] = transcript
                print(f"   Progress: {i}/{len(audio_files)} - {audio_base}")

        return original_transcripts
    
    def run_audio_comparison_evaluation(self) -> Dict[str, Any]:
        """Run the complete audio-to-audio comparison evaluation."""
        
        print("🎵 Audio-to-Audio Transcription Comparison Evaluation")
        print("=" * 60)
        
        # Step 1: Discover original audio files
        print("📁 Discovering audio files...")
        audio_files = self.discover_audio_files()
        
        # Step 2: Transcribe original audio files
        original_transcripts = self.transcribe_original_audio(audio_files)
        
        # Step 3: Generate noisy variants
        print("🔊 Generating noisy variants...")
        variant_records = self._generate_variants(audio_files)
        print(f"   Generated {len(variant_records)} noisy variants")
        
        # Step 4: Transcribe noisy variants and compare
        print("🤖 Transcribing noisy variants and analyzing...")
        comparison_results = self._compare_transcriptions(variant_records, original_transcripts)
        
        # Step 5: Analyze patterns and correlations
        print("📊 Analyzing noise impact patterns...")
        pattern_analysis = self._analyze_noise_impact_patterns(comparison_results)
        
        # Step 6: Generate visualizations
        print("📈 Generating visualizations...")
        visualization_paths = self._generate_visualizations(comparison_results, pattern_analysis)
        
        # Step 7: Create comprehensive report
        print("📝 Creating comprehensive report...")
        report_paths = self._generate_reports(comparison_results, pattern_analysis, visualization_paths)
        
        # Step 8: Save all results
        print("💾 Saving results...")
        final_results = self._save_comprehensive_results(
            comparison_results, 
            pattern_analysis,
            original_transcripts,
            report_paths,
            visualization_paths
        )
        
        print("✅ Audio comparison evaluation completed!")
        print(f"📋 Report: {final_results['report_path']}")
        print(f"📊 Visualizations: {final_results['visualization_dir']}")
        
        return final_results
    
    def _generate_variants(self, audio_files: List[str]) -> List[Dict[str, Any]]:
        """Generate noisy variants of audio files."""
        if generate_variants is not None:
            try:
                return generate_variants(
                    audio_paths=audio_files,
                    effects_spec=self.config.effects_spec,
                    output_root=self.config.output_root
                )
            except Exception as e:
                print(f"⚠️  Error with your pipeline: {e}")
                print("⚠️  Falling back to simple file copying")
                return self._fallback_generate_variants(audio_files)
        else:
            # Fallback implementation
            return self._fallback_generate_variants(audio_files)
    
    def _fallback_generate_variants(self, audio_files: List[str]) -> List[Dict[str, Any]]:
        """Fallback variant generation for testing."""
        print("⚠️  Using fallback variant generation")
        
        variant_records = []
        output_root = Path(self.config.output_root)
        
        for audio_path in audio_files:
            audio_base = Path(audio_path).stem
            
            for effect_key, spec in self.config.effects_spec.items():
                variants = spec.get("variants", [{}])
                
                for i, params in enumerate(variants):
                    subdir = output_root / audio_base / effect_key
                    subdir.mkdir(parents=True, exist_ok=True)
                    
                    variant_fname = f"{audio_base}_{effect_key}_v{i}.wav"
                    variant_path = subdir / variant_fname
                    
                    # For testing, copy original file
                    import shutil
                    shutil.copy2(audio_path, variant_path)
                    
                    record = {
                        "audio_path": audio_path,
                        "audio_base": audio_base,
                        "effect_key": effect_key,
                        "variant_idx": i,
                        "variant_path": str(variant_path),
                        "params": params
                    }
                    variant_records.append(record)
        
        return variant_records
    
    def _compare_transcriptions(self, variant_records: List[Dict[str, Any]], 
                               original_transcripts: Dict[str, str]) -> List[Dict[str, Any]]:
        """Compare transcriptions between original and noisy audio (parallelized)."""
        results = []
        total_variants = len(variant_records)

        def process_variant(record):
            audio_base = record["audio_base"]
            variant_path = record["variant_path"]
            original_transcript = original_transcripts.get(audio_base, "")
            noisy_transcript = self.transcription_service.transcribe_audio(variant_path)
            similarity_metrics = self._calculate_similarity_metrics(original_transcript, noisy_transcript)
            noise_metadata = {
                "effect_key": record["effect_key"],
                "params": record["params"],
                "variant_idx": record["variant_idx"]
            }
            try:
                intent_analysis = self.intent_analyzer.analyze_intent_preservation(
                    original_transcript, noisy_transcript, noise_metadata
                )
            except Exception as e:
                intent_analysis = {"error": f"Intent analysis failed: {str(e)}"}
            result = {
                **record,
                "original_transcript": original_transcript,
                "noisy_transcript": noisy_transcript,
                "similarity_metrics": similarity_metrics,
                "intent_analysis": intent_analysis
            }
            return result

        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
            futures = {executor.submit(process_variant, record): idx for idx, record in enumerate(variant_records)}
            for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
                result = future.result()
                results.append(result)
                if i % 10 == 0 or i == total_variants:
                    print(f"   Progress: {i}/{total_variants}")

        return results
    
    def _calculate_similarity_metrics(self, original: str, noisy: str) -> Dict[str, float]:
        """Calculate various similarity metrics between transcripts."""
        
        if not original or not noisy:
            return {
                "word_error_rate": 1.0,
                "character_similarity": 0.0,
                "word_overlap": 0.0,
                "length_ratio": 0.0
            }
        
        # Word Error Rate
        wer = self._calculate_wer(original, noisy)
        
        # Character-level similarity
        char_sim = self._calculate_character_similarity(original, noisy)
        
        # Word overlap
        word_overlap = self._calculate_word_overlap(original, noisy)
        
        # Length ratio
        length_ratio = len(noisy) / len(original) if len(original) > 0 else 0.0
        
        return {
            "word_error_rate": wer,
            "character_similarity": char_sim,
            "word_overlap": word_overlap,
            "length_ratio": length_ratio
        }
    
    def _calculate_wer(self, reference: str, hypothesis: str) -> float:
        """Calculate Word Error Rate."""
        import difflib
        
        ref_words = reference.lower().split()
        hyp_words = hypothesis.lower().split()
        
        if not ref_words:
            return 1.0 if hyp_words else 0.0
        
        matcher = difflib.SequenceMatcher(None, ref_words, hyp_words)
        operations = matcher.get_opcodes()
        
        errors = 0
        for op, i1, i2, j1, j2 in operations:
            if op == 'replace':
                errors += max(i2 - i1, j2 - j1)
            elif op == 'delete':
                errors += i2 - i1
            elif op == 'insert':
                errors += j2 - j1
        
        return errors / len(ref_words)
    
    def _calculate_character_similarity(self, text1: str, text2: str) -> float:
        """Calculate character-level similarity."""
        import difflib
        return difflib.SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    
    def _calculate_word_overlap(self, text1: str, text2: str) -> float:
        """Calculate word overlap ratio."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _analyze_noise_impact_patterns(self, comparison_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns in how different noise types impact transcription."""
        
        # Group by effect type
        by_effect = defaultdict(list)
        by_intent_category = defaultdict(list)
        
        for result in comparison_results:
            effect_key = result.get("effect_key")
            intent_analysis = result.get("intent_analysis", {})
            similarity_metrics = result.get("similarity_metrics", {})
            
            by_effect[effect_key].append({
                "wer": similarity_metrics.get("word_error_rate", 1.0),
                "intent_preserved": intent_analysis.get("intent_preserved", False),
                "intent_confidence": intent_analysis.get("intent_confidence", 0.0),
                "noise_impact_severity": intent_analysis.get("noise_impact_severity", "unknown"),
                "error_types": intent_analysis.get("error_types", []),
                "intent_category": intent_analysis.get("intent_category", "unknown")
            })
            
            intent_category = intent_analysis.get("intent_category", "unknown")
            by_intent_category[intent_category].append({
                "effect_key": effect_key,
                "intent_preserved": intent_analysis.get("intent_preserved", False),
                "wer": similarity_metrics.get("word_error_rate", 1.0)
            })
        
        # Calculate aggregate statistics
        effect_stats = {}
        for effect_key, results in by_effect.items():
            wer_values = [r["wer"] for r in results]
            intent_preserved_rate = sum(1 for r in results if r["intent_preserved"]) / len(results)
            avg_confidence = sum(r["intent_confidence"] for r in results) / len(results)
            
            # Count error types
            error_type_counts = Counter()
            for r in results:
                for error_type in r["error_types"]:
                    error_type_counts[error_type] += 1
            
            # Count severity levels
            severity_counts = Counter(r["noise_impact_severity"] for r in results)
            
            effect_stats[effect_key] = {
                "total_samples": len(results),
                "avg_wer": sum(wer_values) / len(wer_values),
                "median_wer": sorted(wer_values)[len(wer_values) // 2],
                "intent_preservation_rate": intent_preserved_rate,
                "avg_intent_confidence": avg_confidence,
                "error_type_distribution": dict(error_type_counts),
                "severity_distribution": dict(severity_counts),
                "wer_distribution": {
                    "low_impact": sum(1 for w in wer_values if w < 0.2),
                    "medium_impact": sum(1 for w in wer_values if 0.2 <= w < 0.5),
                    "high_impact": sum(1 for w in wer_values if w >= 0.5)
                }
            }
        
        # Intent category analysis
        intent_category_stats = {}
        for category, results in by_intent_category.items():
            if category != "unknown":
                intent_preserved_rate = sum(1 for r in results if r["intent_preserved"]) / len(results)
                avg_wer = sum(r["wer"] for r in results) / len(results)
                
                # Most/least robust effects for this intent category
                effect_performance = defaultdict(list)
                for r in results:
                    effect_performance[r["effect_key"]].append(r["intent_preserved"])
                
                intent_category_stats[category] = {
                    "total_samples": len(results),
                    "intent_preservation_rate": intent_preserved_rate,
                    "avg_wer": avg_wer,
                    "effect_performance": {
                        effect: sum(preserved_list) / len(preserved_list)
                        for effect, preserved_list in effect_performance.items()
                    }
                }
        
        return {
            "effect_statistics": effect_stats,
            "intent_category_statistics": intent_category_stats,
            "overall_statistics": {
                "total_comparisons": len(comparison_results),
                "overall_intent_preservation_rate": sum(
                    1 for r in comparison_results 
                    if r.get("intent_analysis", {}).get("intent_preserved", False)
                ) / len(comparison_results),
                "overall_avg_wer": sum(
                    r.get("similarity_metrics", {}).get("word_error_rate", 1.0)
                    for r in comparison_results
                ) / len(comparison_results)
            }
        }
    
    def _generate_visualizations(self, comparison_results: List[Dict[str, Any]], 
                               pattern_analysis: Dict[str, Any]) -> Dict[str, str]:
        """Generate comprehensive visualizations of the results."""
        
        viz_dir = Path(self.config.results_dir) / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        viz_paths = {}
        
        # 1. WER by Effect Type
        viz_paths["wer_by_effect"] = self._plot_wer_by_effect(pattern_analysis, viz_dir)
        
        # 2. Intent Preservation Rates
        viz_paths["intent_preservation"] = self._plot_intent_preservation(pattern_analysis, viz_dir)
        
        # 3. Error Type Heatmap
        viz_paths["error_type_heatmap"] = self._plot_error_type_heatmap(pattern_analysis, viz_dir)
        
        # 4. Noise Impact Severity Distribution
        viz_paths["severity_distribution"] = self._plot_severity_distribution(pattern_analysis, viz_dir)
        
        # 5. Intent Category Performance
        viz_paths["intent_category_performance"] = self._plot_intent_category_performance(pattern_analysis, viz_dir)
        
        # 6. WER Distribution by Effect
        viz_paths["wer_distribution"] = self._plot_wer_distribution(comparison_results, viz_dir)
        
        # 7. Correlation Matrix
        viz_paths["correlation_matrix"] = self._plot_correlation_matrix(comparison_results, viz_dir)
        
        return viz_paths
    
    def _plot_wer_by_effect(self, pattern_analysis: Dict[str, Any], viz_dir: Path) -> str:
        """Plot Word Error Rate by effect type."""
        effect_stats = pattern_analysis["effect_statistics"]
        
        effects = list(effect_stats.keys())
        avg_wers = [effect_stats[effect]["avg_wer"] for effect in effects]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(effects, avg_wers, color=sns.color_palette("husl", len(effects)))
        
        plt.title("Average Word Error Rate by Noise Effect", fontsize=16, fontweight='bold')
        plt.xlabel("Noise Effect Type", fontsize=12)
        plt.ylabel("Average WER", fontsize=12)
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, wer in zip(bars, avg_wers):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{wer:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        path = viz_dir / "wer_by_effect.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(path)
    
    def _plot_intent_preservation(self, pattern_analysis: Dict[str, Any], viz_dir: Path) -> str:
        """Plot intent preservation rates."""
        effect_stats = pattern_analysis["effect_statistics"]
        
        effects = list(effect_stats.keys())
        preservation_rates = [effect_stats[effect]["intent_preservation_rate"] for effect in effects]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(effects, preservation_rates, color=sns.color_palette("RdYlGn", len(effects)))
        
        plt.title("Intent Preservation Rate by Noise Effect", fontsize=16, fontweight='bold')
        plt.xlabel("Noise Effect Type", fontsize=12)
        plt.ylabel("Intent Preservation Rate", fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1)
        
        # Add value labels
        for bar, rate in zip(bars, preservation_rates):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')
        
        # Add horizontal line at 80% threshold
        plt.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='80% Threshold')
        plt.legend()
        
        plt.tight_layout()
        
        path = viz_dir / "intent_preservation_rates.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(path)
    
    def _plot_error_type_heatmap(self, pattern_analysis: Dict[str, Any], viz_dir: Path) -> str:
        """Plot heatmap of error types by effect."""
        effect_stats = pattern_analysis["effect_statistics"]
        
        # Collect all error types
        all_error_types = set()
        for stats in effect_stats.values():
            all_error_types.update(stats["error_type_distribution"].keys())
        
        all_error_types = sorted(list(all_error_types))
        effects = list(effect_stats.keys())
        
        # Create matrix
        matrix = []
        for effect in effects:
            row = []
            for error_type in all_error_types:
                count = effect_stats[effect]["error_type_distribution"].get(error_type, 0)
                total = effect_stats[effect]["total_samples"]
                frequency = count / total if total > 0 else 0
                row.append(frequency)
            matrix.append(row)
        
        plt.figure(figsize=(14, 8))
        sns.heatmap(matrix, 
                   xticklabels=all_error_types,
                   yticklabels=effects,
                   annot=True, 
                   fmt='.2f',
                   cmap='YlOrRd',
                   cbar_kws={'label': 'Error Frequency'})
        
        plt.title("Error Type Frequency by Noise Effect", fontsize=16, fontweight='bold')
        plt.xlabel("Error Type", fontsize=12)
        plt.ylabel("Noise Effect", fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        path = viz_dir / "error_type_heatmap.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(path)
    
    def _plot_severity_distribution(self, pattern_analysis: Dict[str, Any], viz_dir: Path) -> str:
        """Plot distribution of noise impact severity levels."""
        effect_stats = pattern_analysis["effect_statistics"]
        
        severity_levels = ["low", "medium", "high"]
        effects = list(effect_stats.keys())
        
        # Prepare data
        severity_data = {level: [] for level in severity_levels}
        
        for effect in effects:
            severity_dist = effect_stats[effect]["severity_distribution"]
            total = effect_stats[effect]["total_samples"]
            
            for level in severity_levels:
                count = severity_dist.get(level, 0)
                severity_data[level].append(count / total if total > 0 else 0)
        
        # Create stacked bar chart
        plt.figure(figsize=(12, 6))
        
        bottom = np.zeros(len(effects))
        colors = ['green', 'orange', 'red']
        
        for i, level in enumerate(severity_levels):
            plt.bar(effects, severity_data[level], bottom=bottom, 
                   label=f'{level.title()} Impact', color=colors[i], alpha=0.8)
            bottom += severity_data[level]
        
        plt.title("Noise Impact Severity Distribution by Effect", fontsize=16, fontweight='bold')
        plt.xlabel("Noise Effect Type", fontsize=12)
        plt.ylabel("Proportion of Samples", fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        
        plt.tight_layout()
        
        path = viz_dir / "severity_distribution.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(path)
    
    def _plot_intent_category_performance(self, pattern_analysis: Dict[str, Any], viz_dir: Path) -> str:
        """Plot performance by intent category."""
        intent_stats = pattern_analysis["intent_category_statistics"]
        
        if not intent_stats:
            # Create empty plot if no intent categories
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, "No Intent Categories Detected", 
                    ha='center', va='center', fontsize=16)
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.axis('off')
            
            path = viz_dir / "intent_category_performance.png"
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()
            return str(path)
        
        categories = list(intent_stats.keys())
        preservation_rates = [intent_stats[cat]["intent_preservation_rate"] for cat in categories]
        avg_wers = [intent_stats[cat]["avg_wer"] for cat in categories]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Intent preservation by category
        bars1 = ax1.bar(categories, preservation_rates, color=sns.color_palette("Set2", len(categories)))
        ax1.set_title("Intent Preservation by Category", fontweight='bold')
        ax1.set_ylabel("Preservation Rate")
        ax1.set_ylim(0, 1)
        
        for bar, rate in zip(bars1, preservation_rates):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')
        
        # WER by category
        bars2 = ax2.bar(categories, avg_wers, color=sns.color_palette("Set1", len(categories)))
        ax2.set_title("Average WER by Category", fontweight='bold')
        ax2.set_ylabel("Word Error Rate")
        
        for bar, wer in zip(bars2, avg_wers):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{wer:.3f}', ha='center', va='bottom', fontweight='bold')
        
        for ax in [ax1, ax2]:
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        path = viz_dir / "intent_category_performance.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(path)
    
    def _plot_wer_distribution(self, comparison_results: List[Dict[str, Any]], viz_dir: Path) -> str:
        """Plot WER distribution across all effects."""
        # Collect WER values by effect
        effect_wers = defaultdict(list)
        
        for result in comparison_results:
            effect_key = result.get("effect_key")
            wer = result.get("similarity_metrics", {}).get("word_error_rate", 0)
            effect_wers[effect_key].append(wer)
        
        plt.figure(figsize=(14, 8))
        
        # Create box plot
        effects = list(effect_wers.keys())
        wer_data = [effect_wers[effect] for effect in effects]
        
        box_plot = plt.boxplot(wer_data, labels=effects, patch_artist=True)
        
        # Color the boxes
        colors = sns.color_palette("husl", len(effects))
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        plt.title("Word Error Rate Distribution by Noise Effect", fontsize=16, fontweight='bold')
        plt.xlabel("Noise Effect Type", fontsize=12)
        plt.ylabel("Word Error Rate", fontsize=12)
        plt.xticks(rotation=45, ha='right')
        
        # Add grid
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        path = viz_dir / "wer_distribution_boxplot.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(path)
    
    def _plot_correlation_matrix(self, comparison_results: List[Dict[str, Any]], viz_dir: Path) -> str:
        """Plot correlation matrix between different metrics."""
        
        # Extract metrics for correlation analysis
        data = []
        for result in comparison_results:
            similarity_metrics = result.get("similarity_metrics", {})
            intent_analysis = result.get("intent_analysis", {})
            
            row = {
                "WER": similarity_metrics.get("word_error_rate", 0),
                "Character Similarity": similarity_metrics.get("character_similarity", 0),
                "Word Overlap": similarity_metrics.get("word_overlap", 0),
                "Length Ratio": similarity_metrics.get("length_ratio", 0),
                "Intent Confidence": intent_analysis.get("intent_confidence", 0),
                "Semantic Similarity": intent_analysis.get("semantic_similarity", 0)
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        correlation_matrix = df.corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, 
                   annot=True, 
                   cmap='RdBu_r',
                   center=0,
                   square=True,
                   fmt='.3f',
                   cbar_kws={'label': 'Correlation Coefficient'})
        
        plt.title("Correlation Matrix: Transcription Quality Metrics", fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        path = viz_dir / "metrics_correlation_matrix.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(path)
    
    def _generate_reports(self, comparison_results: List[Dict[str, Any]], 
                         pattern_analysis: Dict[str, Any],
                         visualization_paths: Dict[str, str]) -> Dict[str, str]:
        """Generate comprehensive reports."""
        
        results_dir = Path(self.config.results_dir)
        
        # Generate markdown report
        report_content = self._build_audio_comparison_report(pattern_analysis, visualization_paths, comparison_results)
        report_path = results_dir / "audio_comparison_report.md"
        report_path.write_text(report_content)
        
        # Generate detailed CSV
        csv_path = self._generate_comparison_csv(comparison_results, results_dir)
        
        return {
            "markdown_report": str(report_path),
            "detailed_csv": str(csv_path)
        }
    
    def _build_audio_comparison_report(self, pattern_analysis: Dict[str, Any], 
                                     visualization_paths: Dict[str, str],
                                     comparison_results: List[Dict[str, Any]]) -> str:
        """Build comprehensive markdown report for audio comparison."""
        
        overall_stats = pattern_analysis["overall_statistics"]
        effect_stats = pattern_analysis["effect_statistics"]
        
        report = f"""# Audio-to-Audio Transcription Comparison Report

## Executive Summary

**Total Comparisons**: {overall_stats['total_comparisons']}
**Overall Intent Preservation Rate**: {overall_stats['overall_intent_preservation_rate']:.1%}
**Overall Average WER**: {overall_stats['overall_avg_wer']:.3f}

## Key Findings

### Most Robust Effects (Lowest Impact on Transcription)
"""
        
        # Sort effects by WER
        sorted_effects = sorted(effect_stats.items(), key=lambda x: x[1]["avg_wer"])
        
        report += "\n**Best Performing (Lowest WER):**\n"
        for effect, stats in sorted_effects[:3]:
            report += f"- **{effect}**: {stats['avg_wer']:.3f} WER, {stats['intent_preservation_rate']:.1%} intent preservation\n"
        
        report += "\n**Worst Performing (Highest WER):**\n"
        for effect, stats in sorted_effects[-3:]:
            report += f"- **{effect}**: {stats['avg_wer']:.3f} WER, {stats['intent_preservation_rate']:.1%} intent preservation\n"
        
        report += """

### Detailed Effect Analysis

"""
        
        for effect, stats in effect_stats.items():
            report += f"#### {effect.replace('_', ' ').title()}\n\n"
            report += f"- **Samples**: {stats['total_samples']}\n"
            report += f"- **Average WER**: {stats['avg_wer']:.3f}\n"
            report += f"- **Median WER**: {stats['median_wer']:.3f}\n"
            report += f"- **Intent Preservation**: {stats['intent_preservation_rate']:.1%}\n"
            report += f"- **Average Intent Confidence**: {stats['avg_intent_confidence']:.3f}\n"
            
            # WER impact distribution
            wer_dist = stats['wer_distribution']
            report += f"- **Impact Distribution**:\n"
            report += f"  - Low impact (WER < 0.2): {wer_dist['low_impact']} samples\n"
            report += f"  - Medium impact (0.2 ≤ WER < 0.5): {wer_dist['medium_impact']} samples\n"
            report += f"  - High impact (WER ≥ 0.5): {wer_dist['high_impact']} samples\n"
            
            # Top error types
            error_dist = stats['error_type_distribution']
            if error_dist:
                sorted_errors = sorted(error_dist.items(), key=lambda x: x[1], reverse=True)
                report += f"- **Most Common Error Types**:\n"
                for error_type, count in sorted_errors[:3]:
                    freq = count / stats['total_samples']
                    report += f"  - {error_type}: {count} occurrences ({freq:.1%})\n"
            
            report += "\n"
        
        report += """

## Visualizations

"""
        
        # Add visualization references
        for viz_name, viz_path in visualization_paths.items():
            viz_title = viz_name.replace('_', ' ').title()
            report += f"- **{viz_title}**: `{Path(viz_path).name}`\n"
        
        report += f"""

## Methodology

1. **Original Audio Transcription**: Transcribed {len(set(r['audio_base'] for r in comparison_results))} original audio files using OpenAI Whisper
2. **Noise Application**: Applied {len(effect_stats)} different noise effects with multiple parameter variations
3. **Noisy Audio Transcription**: Transcribed all {overall_stats['total_comparisons']} noisy variants
4. **Comparison Analysis**: Compared original vs noisy transcriptions using:
   - Word Error Rate (WER)
   - Character-level similarity
   - Word overlap analysis
   - Intent preservation analysis via GPT-4
5. **Statistical Analysis**: Calculated aggregate statistics and correlation patterns

## Recommendations

Based on the audio-to-audio comparison analysis:

### For Model Training:
1. **Priority Noise Types**: Focus on noise types with highest WER impact
2. **Robustness Training**: Include training data with noise patterns that show high error rates

### For System Design:
1. **Confidence Thresholds**: Use intent confidence scores to trigger fallback mechanisms
2. **Error Detection**: Implement real-time detection for high-impact noise conditions

### For Quality Assurance:
1. **Testing Protocols**: Prioritize testing with noise effects that show >50% intent preservation loss
2. **Monitoring**: Track WER and intent preservation rates in production

## Data Files

- `detailed_comparison_results.csv`: Complete comparison data for further analysis
- `audio_comparison_analysis.json`: Machine-readable analysis results
- `visualizations/`: Directory containing all generated charts and plots

---
*Report generated by Audio-to-Audio Transcription Comparison System*
"""
        
        return report
    
    def _generate_comparison_csv(self, comparison_results: List[Dict[str, Any]], 
                               results_dir: Path) -> str:
        """Generate detailed CSV with all comparison results."""
        
        csv_path = results_dir / "detailed_comparison_results.csv"
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'audio_base', 'effect_key', 'variant_idx', 'variant_path',
                'original_transcript', 'noisy_transcript', 
                'word_error_rate', 'character_similarity', 'word_overlap', 'length_ratio',
                'intent_preserved', 'intent_confidence', 'semantic_similarity',
                'noise_impact_severity', 'intent_category', 'error_types',
                'critical_words_affected', 'effect_params'
            ])
            
            # Data rows
            for result in comparison_results:
                similarity_metrics = result.get('similarity_metrics', {})
                intent_analysis = result.get('intent_analysis', {})
                
                error_types = intent_analysis.get('error_types', [])
                if isinstance(error_types, list):
                    error_types_str = ';'.join(error_types)
                else:
                    error_types_str = str(error_types)
                
                critical_words = intent_analysis.get('critical_words_affected', [])
                if isinstance(critical_words, list):
                    critical_words_str = ';'.join(critical_words)
                else:
                    critical_words_str = str(critical_words)
                
                writer.writerow([
                    result.get('audio_base', ''),
                    result.get('effect_key', ''),
                    result.get('variant_idx', ''),
                    result.get('variant_path', ''),
                    result.get('original_transcript', ''),
                    result.get('noisy_transcript', ''),
                    similarity_metrics.get('word_error_rate', ''),
                    similarity_metrics.get('character_similarity', ''),
                    similarity_metrics.get('word_overlap', ''),
                    similarity_metrics.get('length_ratio', ''),
                    intent_analysis.get('intent_preserved', ''),
                    intent_analysis.get('intent_confidence', ''),
                    intent_analysis.get('semantic_similarity', ''),
                    intent_analysis.get('noise_impact_severity', ''),
                    intent_analysis.get('intent_category', ''),
                    error_types_str,
                    critical_words_str,
                    json.dumps(result.get('params', {}))
                ])
        
        return str(csv_path)
    
    def _save_comprehensive_results(self, comparison_results: List[Dict[str, Any]], 
                                   pattern_analysis: Dict[str, Any],
                                   original_transcripts: Dict[str, str],
                                   report_paths: Dict[str, str],
                                   visualization_paths: Dict[str, str]) -> Dict[str, Any]:
        """Save all comprehensive results."""
        
        results_dir = Path(self.config.results_dir)
        
        # Save raw comparison results
        raw_results_path = results_dir / "raw_comparison_results.json"
        with open(raw_results_path, 'w') as f:
            json.dump(comparison_results, f, indent=2)
        
        # Save pattern analysis
        analysis_path = results_dir / "audio_comparison_analysis.json"
        with open(analysis_path, 'w') as f:
            json.dump(pattern_analysis, f, indent=2)
        
        # Save original transcripts
        transcripts_path = results_dir / "original_transcripts.json"
        with open(transcripts_path, 'w') as f:
            json.dump(original_transcripts, f, indent=2)
        
        # Create summary
        summary = {
            "experiment_type": "audio_to_audio_comparison",
            "total_comparisons": len(comparison_results),
            "effects_tested": len(self.config.effects_spec),
            "original_audio_files": len(original_transcripts),
            "results": {
                "raw_results": str(raw_results_path),
                "pattern_analysis": str(analysis_path),
                "original_transcripts": str(transcripts_path),
                **report_paths
            },
            "visualizations": visualization_paths,
            "summary_stats": pattern_analysis.get("overall_statistics", {})
        }
        
        summary_path = results_dir / "experiment_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return {
            "summary_path": str(summary_path),
            "report_path": report_paths.get("markdown_report"),
            "analysis_path": str(analysis_path),
            "raw_results_path": str(raw_results_path),
            "visualization_dir": str(Path(list(visualization_paths.values())[0]).parent) if visualization_paths else "",
            "all_results": summary["results"]
        }


# Main execution function
def run_audio_comparison_evaluation(
    audio_dir: str,
    effects_spec: Dict[str, Dict],
    openai_api_key: str = None,
    output_root: str = "audio_comparison_outputs",
    results_dir: str = "audio_comparison_results",
    use_deepgram: bool = False,
    deepgram_api_key: str = None
) -> Dict[str, Any]:
    """
    Run complete audio-to-audio transcription comparison evaluation.
    
    Args:
        audio_dir: Directory containing original audio files
        effects_spec: Dictionary defining noise effects to apply
        openai_api_key: OpenAI API key (if None, will load from .env)
        output_root: Directory for noisy audio variants
        results_dir: Directory for results and reports
        
    Returns:
        Dictionary with paths to all generated results
    """
    # Load API key from .env if not provided
    def load_env_file(env_path=".env"):
        """Load environment variables from .env file."""
        env_vars = {}
        try:
            with open(env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        env_vars[key.strip()] = value.strip()
        except FileNotFoundError:
            pass
        return env_vars

    # Always load .env for both OpenAI and Deepgram keys if not provided
    env_vars = load_env_file()
    if openai_api_key is None:
        openai_api_key = env_vars.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in .env file or environment.")

    if use_deepgram:
        if not deepgram_api_key:
            deepgram_api_key = env_vars.get("DEEPGRAM_API_KEY") or os.getenv("DEEPGRAM_API_KEY")
        if not deepgram_api_key:
            raise ValueError("DEEPGRAM_API_KEY not found in .env file, environment, or arguments.")
        output_root = output_root + "_deepgram"
        results_dir = results_dir + "_deepgram"

    config = AudioComparisonConfig(
        openai_api_key=openai_api_key,
        audio_dir=audio_dir,
        effects_spec=effects_spec,
        output_root=output_root,
        results_dir=results_dir
    )
    evaluator = AudioComparisonEvaluator(config, use_deepgram=use_deepgram, deepgram_api_key=deepgram_api_key)
    return evaluator.run_audio_comparison_evaluation()

if __name__ == "__main__":
    # ...existing code...
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_deepgram", action="store_true", help="Use Deepgram for transcription instead of OpenAI Whisper")
    parser.add_argument("--deepgram_api_key", type=str, default=None, help="Deepgram API key (or set DEEPGRAM_API_KEY env var)")
    args = parser.parse_args()

    try:
        from audio_comparison_config import EFFECTS_SPEC, AUDIO_DIRECTORY
    except ImportError:
        print("❌ Could not import EFFECTS_SPEC or AUDIO_DIRECTORY from audio_comparison_config.py")
        EFFECTS_SPEC = None
        AUDIO_DIRECTORY = None

    if EFFECTS_SPEC is None or AUDIO_DIRECTORY is None:
        print("❌ Please define EFFECTS_SPEC and AUDIO_DIRECTORY in audio_comparison_config.py")
    else:
        try:
            results = run_audio_comparison_evaluation(
                audio_dir=AUDIO_DIRECTORY,
                effects_spec=EFFECTS_SPEC,
                use_deepgram=args.use_deepgram,
                deepgram_api_key=args.deepgram_api_key
            )
            print(f"Results: {results}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Please create a .env file with your OpenAI API key:")
            print("OPENAI_API_KEY=your-api-key-here")
        except ValueError as e:
            print(f"Error: {e}")