#!/usr/bin/env python3
"""
Convert APIBench format to BFCL format for single-turn to multi-turn testing.

This script converts APIBench datasets into the BFCL format expected by our single-turn to 
multi-turn converter.
"""

import json
import re
import argparse
from pathlib import Path

def extract_instruction(code_field):
    """Extract the instruction/query from the code field."""
    if not code_field:
        return ""
    
    # Look for ###Instruction: pattern
    instruction_match = re.search(r'###Instruction:\s*(.+?)(?:\n###|$)', code_field, re.DOTALL)
    if instruction_match:
        return instruction_match.group(1).strip()
    
    return code_field[:200] + "..." if len(code_field) > 200 else code_field

def create_function_schema(api_data):
    """Create a function schema from API data."""
    if not api_data:
        return {}
    
    function_name = api_data.get('api_name', 'unknown_api')
    api_call = api_data.get('api_call', '')
    description = api_data.get('description', 'API function')
    
    # Extract parameters from api_call if available
    parameters = {
        "type": "dict",
        "required": [],
        "properties": {}
    }
    
    # Try to extract parameters from the API call
    if api_call and '(' in api_call:
        # Simple parameter extraction - this could be enhanced
        param_match = re.search(r'\((.*?)\)', api_call)
        if param_match:
            param_str = param_match.group(1)
            if param_str and param_str != '':
                # Add a generic parameter
                parameters["properties"]["model_name"] = {
                    "type": "string",
                    "description": "The model name or identifier"
                }
                parameters["required"].append("model_name")
    
    return {
        "name": function_name.replace('/', '_').replace('-', '_'),
        "description": description[:500] + "..." if len(description) > 500 else description,
        "parameters": parameters
    }

def convert_apibench_to_bfcl(input_file, output_file, limit=None):
    """Convert APIBench format to BFCL format."""
    
    with open(input_file, 'r', encoding='utf-8') as f:
        apibench_data = json.load(f)
    
    converted_data = []
    
    for i, item in enumerate(apibench_data):
        if limit and i >= limit:
            break
            
        # Extract instruction/query
        instruction = extract_instruction(item.get('code', ''))
        if not instruction:
            continue
        
        # Create function schema
        function_schema = create_function_schema(item.get('api_data'))
        if not function_schema:
            continue
        
        # Create BFCL format entry
        bfcl_entry = {
            "id": f"apibench_{i}",
            "question": [[{"role": "user", "content": instruction}]],
            "function": [function_schema]
        }
        
        converted_data.append(bfcl_entry)
    
    # Save as JSONL format (one JSON object per line)
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in converted_data:
            f.write(json.dumps(entry) + '\n')
    
    print(f"Converted {len(converted_data)} entries from {input_file} to {output_file}")
    return len(converted_data)

def main():
    parser = argparse.ArgumentParser(description="Convert APIBench format to BFCL format")
    parser.add_argument("--input", required=True, help="Input APIBench JSON file")
    parser.add_argument("--output", required=True, help="Output BFCL JSONL file")
    parser.add_argument("--limit", type=int, help="Limit number of entries to convert")
    
    args = parser.parse_args()
    
    if not Path(args.input).exists():
        print(f"Error: Input file {args.input} not found")
        return
    
    convert_apibench_to_bfcl(args.input, args.output, args.limit)

if __name__ == "__main__":
    main()
