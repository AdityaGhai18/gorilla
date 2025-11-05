# Audio-to-Audio Transcription Comparison Report

## Executive Summary

**Total Comparisons**: 6480
**Overall Intent Preservation Rate**: 74.0%
**Overall Average WER**: 0.229

## Key Findings

### Most Robust Effects (Lowest Impact on Transcription)

**Best Performing (Lowest WER):**
- **mic_rubbing**: 0.027 WER, 98.8% intent preservation
- **mumbling_effect**: 0.037 WER, 100.0% intent preservation
- **audio_fade**: 0.043 WER, 98.8% intent preservation

**Worst Performing (Highest WER):**
- **network_cuts**: 0.411 WER, 50.6% intent preservation
- **reverb_echo**: 0.458 WER, 48.8% intent preservation
- **competing_speech**: 0.477 WER, 42.4% intent preservation


### Detailed Effect Analysis

#### Background Noise All

- **Samples**: 5346
- **Average WER**: 0.215
- **Median WER**: 0.083
- **Intent Preservation**: 74.6%
- **Average Intent Confidence**: 0.806
- **Impact Distribution**:
  - Low impact (WER < 0.2): 3710 samples
  - Medium impact (0.2 ≤ WER < 0.5): 827 samples
  - High impact (WER ≥ 0.5): 809 samples
- **Most Common Error Types**:
  - no_significant_impact: 2248 occurrences (42.1%)
  - word_substitution: 1887 occurrences (35.3%)
  - entity_confusion: 1044 occurrences (19.5%)

#### Volume Fluctuation Mild

- **Samples**: 162
- **Average WER**: 0.275
- **Median WER**: 0.167
- **Intent Preservation**: 96.3%
- **Average Intent Confidence**: 0.940
- **Impact Distribution**:
  - Low impact (WER < 0.2): 87 samples
  - Medium impact (0.2 ≤ WER < 0.5): 46 samples
  - High impact (WER ≥ 0.5): 29 samples
- **Most Common Error Types**:
  - word_insertion: 118 occurrences (72.8%)
  - transcription_degradation: 40 occurrences (24.7%)
  - no_significant_impact: 34 occurrences (21.0%)

#### Reverb Echo

- **Samples**: 162
- **Average WER**: 0.458
- **Median WER**: 0.391
- **Intent Preservation**: 48.8%
- **Average Intent Confidence**: 0.582
- **Impact Distribution**:
  - Low impact (WER < 0.2): 71 samples
  - Medium impact (0.2 ≤ WER < 0.5): 17 samples
  - High impact (WER ≥ 0.5): 74 samples
- **Most Common Error Types**:
  - reverb_distortion: 64 occurrences (39.5%)
  - entity_confusion: 57 occurrences (35.2%)
  - transcription_degradation: 54 occurrences (33.3%)

#### Reverb Cave

- **Samples**: 81
- **Average WER**: 0.210
- **Median WER**: 0.125
- **Intent Preservation**: 79.0%
- **Average Intent Confidence**: 0.827
- **Impact Distribution**:
  - Low impact (WER < 0.2): 50 samples
  - Medium impact (0.2 ≤ WER < 0.5): 22 samples
  - High impact (WER ≥ 0.5): 9 samples
- **Most Common Error Types**:
  - word_substitution: 50 occurrences (61.7%)
  - entity_confusion: 27 occurrences (33.3%)
  - reverb_distortion: 25 occurrences (30.9%)

#### Network Cuts

- **Samples**: 162
- **Average WER**: 0.411
- **Median WER**: 0.381
- **Intent Preservation**: 50.6%
- **Average Intent Confidence**: 0.641
- **Impact Distribution**:
  - Low impact (WER < 0.2): 40 samples
  - Medium impact (0.2 ≤ WER < 0.5): 65 samples
  - High impact (WER ≥ 0.5): 57 samples
- **Most Common Error Types**:
  - word_deletion: 130 occurrences (80.2%)
  - partial_loss: 102 occurrences (63.0%)
  - transcription_degradation: 86 occurrences (53.1%)

#### Audio Fade

- **Samples**: 162
- **Average WER**: 0.043
- **Median WER**: 0.032
- **Intent Preservation**: 98.8%
- **Average Intent Confidence**: 0.988
- **Impact Distribution**:
  - Low impact (WER < 0.2): 158 samples
  - Medium impact (0.2 ≤ WER < 0.5): 4 samples
  - High impact (WER ≥ 0.5): 0 samples
- **Most Common Error Types**:
  - no_significant_impact: 124 occurrences (76.5%)
  - word_substitution: 22 occurrences (13.6%)
  - entity_confusion: 9 occurrences (5.6%)

#### Mic Rubbing

- **Samples**: 81
- **Average WER**: 0.027
- **Median WER**: 0.000
- **Intent Preservation**: 98.8%
- **Average Intent Confidence**: 0.987
- **Impact Distribution**:
  - Low impact (WER < 0.2): 81 samples
  - Medium impact (0.2 ≤ WER < 0.5): 0 samples
  - High impact (WER ≥ 0.5): 0 samples
- **Most Common Error Types**:
  - no_significant_impact: 64 occurrences (79.0%)
  - word_substitution: 7 occurrences (8.6%)
  - word_insertion: 5 occurrences (6.2%)

#### Mumbling Effect

- **Samples**: 81
- **Average WER**: 0.037
- **Median WER**: 0.000
- **Intent Preservation**: 100.0%
- **Average Intent Confidence**: 0.991
- **Impact Distribution**:
  - Low impact (WER < 0.2): 78 samples
  - Medium impact (0.2 ≤ WER < 0.5): 3 samples
  - High impact (WER ≥ 0.5): 0 samples
- **Most Common Error Types**:
  - no_significant_impact: 65 occurrences (80.2%)
  - word_substitution: 11 occurrences (13.6%)
  - entity_confusion: 4 occurrences (4.9%)

#### Competing Speech

- **Samples**: 243
- **Average WER**: 0.477
- **Median WER**: 0.364
- **Intent Preservation**: 42.4%
- **Average Intent Confidence**: 0.578
- **Impact Distribution**:
  - Low impact (WER < 0.2): 65 samples
  - Medium impact (0.2 ≤ WER < 0.5): 71 samples
  - High impact (WER ≥ 0.5): 107 samples
- **Most Common Error Types**:
  - word_substitution: 158 occurrences (65.0%)
  - entity_confusion: 135 occurrences (55.6%)
  - transcription_degradation: 105 occurrences (43.2%)



## Visualizations

- **Wer By Effect**: `wer_by_effect.png`
- **Intent Preservation**: `intent_preservation_rates.png`
- **Error Type Heatmap**: `error_type_heatmap.png`
- **Severity Distribution**: `severity_distribution.png`
- **Intent Category Performance**: `intent_category_performance.png`
- **Wer Distribution**: `wer_distribution_boxplot.png`
- **Correlation Matrix**: `metrics_correlation_matrix.png`


## Methodology

1. **Original Audio Transcription**: Transcribed 81 original audio files using OpenAI Whisper
2. **Noise Application**: Applied 9 different noise effects with multiple parameter variations
3. **Noisy Audio Transcription**: Transcribed all 6480 noisy variants
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
