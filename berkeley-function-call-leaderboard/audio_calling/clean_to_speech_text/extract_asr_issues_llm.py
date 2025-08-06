#!/usr/bin/env python3
"""
Script to filter BFCL test cases with potential ASR issues using LLM.
Takes a single file and creates a new JSON with cases that need clarification.
"""

import json
import sys
from typing import List, Dict, Any
import openai
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

def llm_analyze_batch(test_cases: List[Dict]) -> List[Dict]:
    """
    Analyze a batch of test cases with a single API call using GPT-4o.
    Returns list of dicts with 'should_include' and 'explanation' for each case.
    """
    # Get API key from environment
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("No OpenAI API key found in environment variables")
        return [{'should_include': False, 'explanation': 'No API key'} for _ in test_cases]
    
    # Initialize OpenAI client with correct format
    client = openai.OpenAI(api_key=api_key)
    
    # Create a comprehensive batch prompt
    batch_prompt = """You are analyzing test cases to determine if clarification is ABSOLUTELY NECESSARY to get the function call correct.

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

IMPORTANT: The LLM has access to the function documentation and can infer required formats. Do NOT flag cases where:
- The function docs show the required format (e.g., if docs show "PIZZA" is required, don't flag "pizza" → "PIZZA")
- The LLM can easily convert between common formats (e.g., "pizza" → "PIZZA" for food orders)
- The function documentation provides clear guidance on expected input format

Examples of cases that NEED clarification:
- "final_report.pdf" vs "final report dot pdf" (Filename missing underscore, Argument value incorrect, Clarification needed)
- "my-bot-id" vs "my bot ID" (hyphens lost AND meaning changed, argument value incorrect, Clarification needed)
- "v2" vs "v two" (version number ambiguity in terms of spacing, v2 or v 2, could lead to incorrect argument value, Clarification might be needed)
- "13" vs "30" (similar-sounding numbers, argument value incorrect, Clarification needed)
- "Zhang Wei" vs "John Way" (foreign name misheard, argument value incorrect, Clarification needed)
- Technical codes like "MIIFdTCCBF2gAwIBAgISESG" vs "M I I F D T C C B F 2 G A W I B A G I S E S G" (argument value incorrect due to capitalisation, case sensitivity was not clear, Clarification needed)
- "development-env" vs "development env" (hyphen lost, meaning changed clarification needed)


Examples of cases that DON'T need clarification (these are all literally the same, or simple cases where the LLM could infer and handle it correctly):
- "New York" vs "New York" (same city name, no ASR issue possible, no clarification needed)
- "pizza" vs "pizza" (same food item, no ASR issue possible, no clarification needed)
- "Boston" vs "Boston" (same city name, no ASR issue possible, no clarification needed)
- "greens" vs "greens" (same food item, no ASR issue possible, no clarification needed)
- "Naples, Florida" vs "Naples, Florida" (same location, no ASR issue possible, no clarification needed)
- "logistic regression" vs "logistic regression" (same technical term, no ASR issue possible, no clarification needed)
- "feature-branch" vs "feature dash branch" (hyphen spoken as "dash" is normal, meaning preserved, no clarification needed)
- "DataSet1.csv" vs "data set one dot CSV" (numbers spoken as words, file extensions spoken as "dot", LLM can convert, no clarification needed)
- "2024_backup.txt" vs "two zero two four backup dot txt" (numbers spoken as words, LLM can convert, no clarification needed)
- "654321" vs "six five four three twenty one" (digits spoken individually, clear pattern, LLM can convert, no clarification needed)
- "FinalReport.txt" vs "finalreport.txt" (case differences, LLM can handle, no clarification needed)
- "pizza" vs "PIZZA" (function docs show required format, LLM can convert, no clarification needed)
- "burger" vs "BURGER" (function docs show required format, LLM can convert, no clarification needed)

The key question: "Did the ASR actually MISHEAR or LOSE information that an LLM cannot reasonably determine or convert hence producing an incorrect function call?"

For each case, respond with:
YES: [brief explanation of the specific ASR transcription error that would cause wrong parameters]
NO: [brief explanation of why the LLM can handle this case correctly]

"""
    
    for i, case in enumerate(test_cases):
        content = case['content']
        asr_outputs = case['asr_outputs']
        functions = case.get('function', [])
        
        batch_prompt += f"\nCase {i+1}:\n"
        batch_prompt += f"Original Content: {content}\n"
        batch_prompt += f"ASR Outputs: {asr_outputs}\n"
        
        # Include function definitions
        if functions:
            batch_prompt += f"Available Functions:\n"
            for j, func in enumerate(functions):
                batch_prompt += f"  Function {j+1}: {json.dumps(func, indent=2)}\n"
        
        batch_prompt += f"Response: "
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": batch_prompt}],
            temperature=0.1,
            max_tokens=3000
        )
        
        # Parse the response
        response_text = response.choices[0].message.content
        results = []
        
        # Extract YES/NO responses with explanations
        lines = response_text.split('\n')
        current_case = None
        current_explanation = ""
        
        for line in lines:
            line = line.strip()
            if line.startswith('YES:'):
                current_case = True
                current_explanation = line[4:].strip()
            elif line.startswith('NO:'):
                current_case = False
                current_explanation = line[3:].strip()
            elif current_case is not None:
                # Continue explanation if it spans multiple lines
                if line and not line.startswith('Case'):
                    current_explanation += " " + line
                else:
                    # End of explanation, add result
                    results.append({
                        'should_include': current_case,
                        'explanation': current_explanation.strip()
                    })
                    current_case = None
                    current_explanation = ""
        
        # Add final result if there's one pending
        if current_case is not None:
            results.append({
                'should_include': current_case,
                'explanation': current_explanation.strip()
            })
        
        # Pad with False if we don't have enough results
        while len(results) < len(test_cases):
            results.append({'should_include': False, 'explanation': 'No response parsed'})
        
        return results[:len(test_cases)]
        
    except Exception as e:
        print(f"LLM analysis failed: {e}")
        return [{'should_include': False, 'explanation': f'Error: {e}'} for _ in test_cases]
    
def filter_file(input_file: str, output_file: str, batch_size: int = 15, max_cases: int = None):
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
                                    'content': content,
                                    'transcript': question.get('transcript', ''),
                                    'asr_outputs': asr_outputs,
                                    'function': item.get('function', []),
                                    'original_item': item
                                })
    
    print(f"Found {len(test_cases)} test cases to analyze")
    
    # Limit cases if specified
    if max_cases and max_cases < len(test_cases):
        test_cases = test_cases[:max_cases]
        print(f"Limited to first {max_cases} cases for testing")
    
    # LLM analysis in batches
    filtered_cases = []
    total_batches = (len(test_cases) + batch_size - 1) // batch_size
    
    print(f"Running LLM analysis on {total_batches} batches (batch size: {batch_size})...")
    
    for i in range(0, len(test_cases), batch_size):
        batch = test_cases[i:i+batch_size]
        batch_num = i // batch_size + 1
        
        print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} cases)...")
        
        llm_results = llm_analyze_batch(batch)
        
        for case, result in zip(batch, llm_results):
            if result['should_include']:
                # Create a filtered version with only problematic turns
                original_item = case['original_item'].copy()
                
                # Find which turn this case belongs to and mark it
                turn_index = None
                for turn_idx, turn in enumerate(original_item['question']):
                    for msg_idx, message in enumerate(turn):
                        if (message.get('role') == 'user' and 
                            message.get('content') == case['content']):
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
        
        issues_found = sum(1 for r in llm_results if r['should_include'])
        print(f"Batch {batch_num} complete. Found {issues_found} cases with issues.")
    
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
    batch_size = int(sys.argv[4]) if len(sys.argv) > 4 else 15
    
    if max_cases:
        print(f"TEST MODE: Processing only first {max_cases} cases with batch size {batch_size}")
    
    filter_file(input_file, output_file, batch_size, max_cases)

if __name__ == "__main__":
    main() 