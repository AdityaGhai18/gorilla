#!/usr/bin/env python3
"""
Script to filter BFCL test cases with potential ASR issues using LLM.
Takes a single file and creates a new JSON with cases that need clarification.
"""

import json
import sys
import time
from typing import List, Dict, Any
import openai
from dotenv import load_dotenv
import os

# Force unbuffered output for real-time logging
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Load environment variables
load_dotenv()

def llm_analyze_single(test_case: Dict) -> Dict:
    """
    Analyze a single test case with GPT-4o.
    Returns dict with 'should_include' and 'explanation'.
    """
    # Get API key from environment
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("No OpenAI API key found in environment variables")
        return {'should_include': False, 'explanation': 'No API key'}
    
    # Initialize OpenAI client with correct format
    client = openai.OpenAI(api_key=api_key)
    
    # Create a comprehensive prompt for single case
    prompt = """You are analyzing a test case to determine if clarification is ABSOLUTELY NECESSARY to get the function call correct.

IMPORTANT: You are comparing the ORIGINAL TRANSCRIPT (what was actually spoken) vs the ASR OUTPUTS (what the speech recognition systems transcribed).

ONLY flag cases where the ASR actually MISHEARD or LOST critical information that an LLM cannot reasonably infer or convert to derive the correct function call.

CRITICAL: Only flag cases where the ASR error would cause the LLM to:
1. Receive completely wrong parameter values (e.g., "Zhang Wei" → "John Way")
2. Be unable to determine the correct parameter due to ASR transcription errors (e.g., "config.py" → "config.pe")
3. Have multiple valid interpretations due to ASR ambiguity (e.g., "13" vs "30")
4. Miss critical information that cannot be inferred from context

IMPORTANT: The LLM is very capable and can handle:
- Numbers spoken as words ("one" → "1", "twenty four" → "24"), some cases have issues where "twelve thirty four" could be 1234 or 12304 so be careful and if in doubt flag them
- File extensions spoken as "dot" (".csv" → "dot CSV")
- Case differences ("FinalReport.txt" vs "finalreport.txt")
- Common abbreviations ("SQL01" → "SQL zero one")
- Hyphens spoken as "dash" ("feature-branch" → "feature dash branch")

CRITICAL: Don't be overly pedantic about capitalization unless absolutely necessary. Check the function documentation to see if case sensitivity actually matters for the specific function call. Only flag capitalization issues when:
- The function explicitly requires exact case (e.g., database names, API keys, exact filenames)
- The LLM cannot reasonably infer the correct case from context
- Case differences would lead to completely different entities (e.g., "MainDB" vs "main db" for database names)

SPACING AND CAPITALIZATION ASSESSMENT:

DO NOT flag these minor variations (LLM can handle them):
- "Fast Data Server" vs "fast data server" (same meaning, different case)
- "primary_db" vs "primarydb" vs "Primary DB" (same identifier, different formatting)
- "SQL01" vs "SQL zero one" (numbers spoken as words, clear pattern)
- "portfolio-web" vs "portfolio dash web" (hyphen spoken as "dash" is normal)

DO flag these variations (meaning or function affected):
- "my-bot-id" vs "my bot ID" (hyphen lost AND meaning changed)
- "config.py" vs "config.pe" (file extension changed)
- "L_12345" vs "L12345" (underscore lost, format changed)

This helps distinguish between minor formatting differences and actual ASR errors that affect function calls.

IMPORTANT: The LLM has access to the function documentation and can infer required formats. Do NOT flag cases where:
- The function docs show the required format (e.g., if docs show "PIZZA" is required, don't flag "pizza" → "PIZZA")
- The LLM can easily convert between common formats (e.g., "pizza" → "PIZZA" for food orders)
- The function documentation provides clear guidance on expected input format

ENHANCED FUNCTION DOCUMENTATION ANALYSIS: Before flagging any case, check if the function docs show:
- Enum values with specific formats (e.g., ["PIZZA", "BURGER"] means LLM can convert "pizza" → "PIZZA")
- Clear format specifications (e.g., "YYYY-MM-DD" means LLM can handle date format variations)
- Required case sensitivity (only flag if docs explicitly require exact case AND LLM can't infer it)

These improvements help reduce over-flagging while maintaining high precision for genuine ASR issues.

Examples of cases that NEED clarification (Transcript vs ASR comparison):
- Transcript: "Let me see, Im Balkrushn, thats B-A-L-K-R-U-S-H-N, Balkrushn. Id like to, uh, use the main features of this tool." vs ASR: "Let me see, I'm Balkrushnan. That's B-A-L-K-R-U-S-H-N. Balkrushnan. I'd like to, uh, use the main features of this tool." (Name misheard as "Balkrushnan" instead of "Balkrushn", Clarification needed)
- Transcript: "Hey, um, can you add to-do: hit up shopping at nine P M?" vs ASR: "Hey, um, can you add to do hit up shopping at nine p.m.?" (Task misheard as "hit up shopping" instead of "go for shopping", Clarification needed)
- Transcript: "final_report.pdf" vs ASR: "final report dot pdf" (Filename missing underscore, Argument value incorrect, Clarification needed)
- Transcript: "my-bot-id" vs ASR: "my bot ID" (hyphens lost AND meaning changed, argument value incorrect, Clarification needed)
- Transcript: "v2" vs ASR: "v two" (version number ambiguity in terms of spacing, v2 or v 2, could lead to incorrect argument value, Clarification might be needed)
- Transcript: "13" vs ASR: "30" (similar-sounding numbers, argument value incorrect, Clarification needed)
- Transcript: "Zhang Wei" vs ASR: "John Way" (foreign name misheard, argument value incorrect, Clarification needed)
- Transcript: "development-env" vs ASR: "development env" (hyphen lost, meaning changed clarification needed)
- Transcript: "config.py" vs ASR: "config.pe" (File extension changed, clarification needed)
- Transcript: "L_12345" vs ASR: "L12345" (Underscore lost, format changed, clarification needed)


Examples of cases that DON'T need clarification (Transcript vs ASR comparison):
- Transcript: "New York" vs ASR: "New York" (same city name, no ASR issue, no clarification needed)
- Transcript: "pizza" vs ASR: "pizza" (same food item, no ASR issue, no clarification needed)
- Transcript: "Boston" vs ASR: "Boston" (same city name, no ASR issue, no clarification needed)
- Transcript: "greens" vs ASR: "greens" (same food item, no ASR issue, no clarification needed)
- Transcript: "Naples, Florida" vs ASR: "Naples, Florida" (same location, no ASR issue, no clarification needed)
- Transcript: "logistic regression" vs ASR: "logistic regression" (same technical term, no ASR issue, no clarification needed)
- Transcript: "feature-branch" vs ASR: "feature dash branch" (hyphen spoken as "dash" is normal, meaning preserved, no clarification needed)
- Transcript: "DataSet1.csv" vs ASR: "data set one dot CSV" (numbers spoken as words, file extensions spoken as "dot", LLM can convert, no clarification needed)
- Transcript: "2024_backup.txt" vs ASR: "two zero two four backup dot txt" (numbers spoken as words, LLM can convert, no clarification needed)
- Transcript: "654321" vs ASR: "six five four three twenty one" (digits spoken individually, clear pattern, LLM can convert, no clarification needed)
- Transcript: "FinalReport.txt" vs ASR: "finalreport.txt" (case differences, LLM can handle, no clarification needed)
- Transcript: "I want a burger" vs ASR: "I want a burger" (Function docs show enum ["PIZZA", "BURGER"], LLM can convert "burger" → "BURGER", no clarification needed)
- Transcript: "Fast Data Server" vs ASR: "fast data server" (Same meaning, minor case differences, no clarification needed)
- Transcript: "primary_db" vs ASR: "primarydb" (Same identifier, minor formatting differences, no clarification needed)
- Transcript: "info type: Speed" vs ASR: "info type: speed" (Function docs show enum includes "Speed", LLM can convert "speed" → "Speed", no clarification needed)

The key question: "Did the ASR actually MISHEAR or LOSE information from the transcript that an LLM cannot reasonably determine or convert hence producing an incorrect function call?"

For each case, respond with:
YES: [brief explanation of the specific ASR transcription error that would cause wrong parameters]
NO: [brief explanation of why the LLM can handle this case correctly]

"""
    
    transcript = test_case['transcript']
    asr_outputs = test_case['asr_outputs']
    functions = test_case.get('function', [])
    
    prompt += f"\nTest Case:\n"
    prompt += f"Original Transcript (what was spoken): {transcript}\n"
    prompt += f"ASR Outputs (what speech recognition transcribed): {asr_outputs}\n"
    
    # Include function definitions
    if functions:
        prompt += f"Available Functions:\n"
        for j, func in enumerate(functions):
            prompt += f"  Function {j+1}: {json.dumps(func, indent=2)}\n"
    
    prompt += f"\nResponse (YES or NO): "
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1000
        )
        
        # Parse the response
        response_text = response.choices[0].message.content.strip()
        
        # Simple parsing for single case
        if response_text.startswith('YES:'):
            explanation = response_text[4:].strip()
            return {'should_include': True, 'explanation': explanation}
        elif response_text.startswith('NO:'):
            explanation = response_text[3:].strip()
            return {'should_include': False, 'explanation': explanation}
        else:
            # Try to extract YES/NO from anywhere in response
            if 'YES:' in response_text:
                start = response_text.find('YES:') + 4
                explanation = response_text[start:].strip()
                return {'should_include': True, 'explanation': explanation}
            elif 'NO:' in response_text:
                start = response_text.find('NO:') + 3
                explanation = response_text[start:].strip()
                return {'should_include': False, 'explanation': explanation}
            else:
                return {'should_include': False, 'explanation': f'Could not parse response: {response_text}'}
        
    except Exception as e:
        print(f"LLM analysis failed: {e}")
        return {'should_include': False, 'explanation': f'Error: {e}'}
    
def filter_file(input_file: str, output_file: str, max_cases: int = None):
    """
    Filter a single BFCL file for ASR issues using LLM analysis.
    """
    print(f"Processing {input_file}...")
    
    # Load the file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract test cases
    test_cases = []
    for item in data:
        if 'question' in item and isinstance(item['question'], list):
            for question_group in item['question']:
                if isinstance(question_group, list):
                    for question in question_group:
                        if isinstance(question, dict) and question.get('role') == 'user':
                            content = question.get('content', '')
                            asr_outputs = []
                            
                            # Collect ASR outputs
                            for key in question.keys():
                                if key.startswith('asr_output_'):
                                    asr_outputs.append(question[key])
                            
                            if asr_outputs:
                                test_cases.append({
                                    'id': item.get('id', 'unknown'),
                                    'content': content,  # Keep for reference
                                    'transcript': question.get('transcript', ''),  # What was actually spoken
                                    'asr_outputs': asr_outputs,  # What ASR transcribed
                                    'function': item.get('function', []),
                                    'original_item': item
                                })
    
    print(f"Found {len(test_cases)} test cases to analyze")
    
    # Limit cases if specified
    if max_cases and max_cases < len(test_cases):
        test_cases = test_cases[:max_cases]
        print(f"Limited to first {max_cases} cases for testing")
    
    # LLM analysis one case at a time
    filtered_cases = []
    
    print(f"Running LLM analysis on {len(test_cases)} individual cases...")
    
    for i, case in enumerate(test_cases):
        case_num = i + 1
        print(f"Processing case {case_num}/{len(test_cases)}: {case['id']}")
        
        result = llm_analyze_single(case)
        
        if result['should_include']:
            # Create a filtered version with only problematic turns
            original_item = case['original_item'].copy()
            
            # Find which turn this case belongs to and mark it
            # We match by transcript since that's what we're comparing against ASR
            turn_index = None
            for turn_idx, turn in enumerate(original_item['question']):
                for msg_idx, message in enumerate(turn):
                    if (message.get('role') == 'user' and 
                        message.get('transcript') == case['transcript']):
                        turn_index = turn_idx
                        # Add explanation to the specific message
                        message['asr_issue_explanation'] = result['explanation']
                        break
                if turn_index is not None:
                    break
            
            # Only include turns that have ASR issues
            filtered_question = []
            for turn_idx, turn in enumerate(original_item['question']):
                has_issues = False
                for message in turn:
                    if message.get('role') == 'user' and 'asr_issue_explanation' in message:
                        has_issues = True
                        break
                if has_issues:
                    filtered_question.append(turn)
            
            original_item['question'] = filtered_question
            filtered_cases.append(original_item)
            print(f"  -> Case {case_num} flagged for clarification")
        else:
            print(f"  -> Case {case_num} no clarification needed")
        
        # Small delay to avoid rate limiting
        if case_num < len(test_cases):
            time.sleep(0.1)
    
    # Create output - preserve original format
    output_data = {
        'source_file': input_file,
        'total_cases_analyzed': len(test_cases),
        'cases_with_asr_issues': len(filtered_cases),
        'batch_size_used': batch_size,
        'filtered_cases': filtered_cases
    }
    
    # Save output
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nFiltering complete!")
    print(f"Found {len(filtered_cases)} cases with ASR issues out of {len(test_cases)} total cases")
    print(f"Results saved to {output_file}")
    print(f"\nNote: Cases were filtered by comparing TRANSCRIPT (spoken) vs ASR OUTPUTS (transcribed)")
    print(f"to identify where speech recognition actually misheard critical information.")

def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_asr_issues_llm.py <input_file> [output_file] [max_cases] [batch_size]")
        print("Example: python extract_asr_issues_llm.py final_results/BFCL_v3_live_simple.json clarif_necessary.json")
        print("Example (test mode): python extract_asr_issues_llm.py final_results/BFCL_v3_live_simple.json clarif_necessary_test.json 30 10")
        print("Make sure to set OPENAI_API_KEY in your .env file")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "clarif_necessary.json"
    max_cases = int(sys.argv[3]) if len(sys.argv) > 3 else None
    if max_cases:
        print(f"TEST MODE: Processing only first {max_cases} cases")
    
    filter_file(input_file, output_file, max_cases)

if __name__ == "__main__":
    main() 