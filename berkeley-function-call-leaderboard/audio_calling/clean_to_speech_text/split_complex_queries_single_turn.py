#!/usr/bin/env python3
"""
Split Complex Queries for jsonl format

Usage:
    python split_complex_queries_single_turn.py --input data.json --output output.json
    python split_complex_queries_single_turn.py --input data.json --output output.json --chunk-start 0 --chunk-size 1000
"""

import json
import re
import argparse
import os
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-4o"

class SingleTurnComplexQuerySplitter:
    def __init__(self):
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
    
    def is_complex_query(self, query: str, function_docs: List[Dict[str, Any]]) -> bool:
        """Determine if a query is complex enough to warrant splitting."""
        total_required = 0
        missing_required = 0
        
        for func in function_docs:
            if "parameters" in func and "required" in func["parameters"]:
                required_args = func["parameters"]["required"]
                total_required += len(required_args)
                
                for arg in required_args:
                    if not self._is_parameter_present(query, arg, func):
                        missing_required += 1
        
        return missing_required > 0
    
    def _is_parameter_present(self, query: str, param_name: str, func_info: Dict[str, Any]) -> bool:
        """Check if a parameter is semantically present in the query."""
        query_lower = query.lower()
        param_lower = param_name.lower()
        
        # Get parameter enum values if available
        param_def = func_info.get("parameters", {}).get("properties", {}).get(param_name, {})
        enum_values = param_def.get("enum", [])
        
        # Special handling for "type" parameter with enum values
        if param_lower == "type" and enum_values:
            # Check if any enum value is semantically present
            for enum_val in enum_values:
                if self._check_semantic_match(query_lower, enum_val):
                    return True
            return False
        
        # For "type" parameter without enum values, be more restrictive
        if param_lower == "type":
            # Only consider present if explicitly mentioned
            return "type" in query_lower or "kind" in query_lower or "category" in query_lower
        
        # Special handling for "content" parameter
        if param_lower == "content":
            # Content is present if there's meaningful text beyond action words
            action_words = ["add", "create", "delete", "remove", "update", "edit", "complete", "todo", "task"]
            words = query_lower.split()
            content_words = [w for w in words if w not in action_words and len(w) > 2]
            return len(content_words) > 0
        
        # Check for direct parameter name matches (be more restrictive)
        if param_lower in query_lower:
            return True
        
        # Check for common parameter patterns
        return self._check_common_patterns(query_lower, param_lower)
    
    def _check_semantic_match(self, query: str, enum_value: str) -> bool:
        """Check if an enum value is semantically present in query."""
        enum_lower = enum_value.lower()
        
        # Direct match
        if enum_lower in query:
            return True
        
        # Semantic mappings for common actions
        action_mappings = {
            "add": ["add", "create", "new", "make", "insert"],
            "delete": ["delete", "remove", "del", "erase"],
            "update": ["update", "edit", "modify", "change"],
            "complete": ["complete", "done", "finish", "mark"]
        }
        
        if enum_lower in action_mappings:
            return any(keyword in query for keyword in action_mappings[enum_lower])
        
        return False
    
    def _check_common_patterns(self, query: str, param_name: str) -> bool:
        """Check for common parameter patterns in queries."""
        # Location patterns
        if "location" in param_name or "address" in param_name:
            location_indicators = ["at ", "in ", "from ", "to ", "street", "avenue", "road"]
            return any(indicator in query for indicator in location_indicators)
        
        # Time patterns
        if "time" in param_name or "date" in param_name:
            time_indicators = ["at ", "on ", "pm", "am", "o'clock", ":", "today", "tomorrow"]
            return any(indicator in query for indicator in time_indicators)
        
        # ID patterns
        if "id" in param_name:
            import re
            return bool(re.search(r'\b\d+\b', query))
        
        return False
    
    def extract_entities_and_arguments(self, query: str, function_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract function arguments from query."""
        mentioned_arguments = {}
        missing_arguments = {}
        
        for func in function_docs:
            func_name = func.get("name", "")
            if not func_name:
                continue
                
            mentioned_arguments[func_name] = {}
            missing_arguments[func_name] = []
            
            if "parameters" in func and "properties" in func["parameters"]:
                properties = func["parameters"]["properties"]
                required = func["parameters"].get("required", [])
                
                for param_name in properties.keys():
                    if self._is_parameter_present(query, param_name, func):
                        found_value = self._find_argument_value(query, param_name)
                        if found_value:
                            mentioned_arguments[func_name][param_name] = found_value
                    elif param_name in required:
                        missing_arguments[func_name].append(param_name)
        
        return {
            "mentioned_arguments": mentioned_arguments,
            "missing_arguments": missing_arguments
        }
    
    def _find_argument_value(self, query: str, param_name: str) -> Optional[str]:
        """Find parameter value in query using pattern matching."""
        query_lower = query.lower()
        param_name_lower = param_name.lower()
        
        # Check for exact matches or variations
        if (param_name_lower in query_lower or 
            param_name_lower.replace("_", " ") in query_lower or
            any(word in query_lower for word in param_name_lower.split('_'))):
            
            # Extract numbers for numeric parameters
            if "id" in param_name_lower:
                numbers = re.findall(r'\b\d+\b', query)
                if numbers:
                    return numbers[0]
            
            # Extract locations
            if any(word in param_name_lower for word in ["location", "address", "place"]):
                locations = re.findall(r'\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd)', query, re.IGNORECASE)
                if locations:
                    return locations[0]
            
            # Extract other patterns as needed
            return "found"
        
        return None
    
    def generate_clarification_questions(self, analysis: Dict[str, Any], function_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate clarification questions for missing arguments."""
        missing_args = analysis.get("missing_arguments", {})
        if not missing_args:
            return []
        
        function_docs_text = self._format_function_docs(function_docs)
        
        prompt = f"""
Generate natural clarification questions for missing function arguments.

Function Documentation:
{function_docs_text}

Missing Arguments:
{json.dumps(missing_args, indent=2)}

Return a JSON array with this structure:
[
    {{
        "function_name": "function_name",
        "argument_name": "argument_name", 
        "question": "What [argument] would you like to use?",
        "description": "argument description"
    }}
]
"""
        
        try:
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=800
            )
            
            content = response.choices[0].message.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            
            return json.loads(content.strip())
            
        except Exception as e:
            print(f"Error generating clarifications: {e}")
            return self._generate_fallback_clarifications(analysis, function_docs)
    
    def _generate_fallback_clarifications(self, analysis: Dict[str, Any], function_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate fallback clarifications."""
        clarifications = []
        
        for function_name, missing_args in analysis.get("missing_arguments", {}).items():
            for arg_name in missing_args:
                clarifications.append({
                    "function_name": function_name,
                    "argument_name": arg_name,
                    "question": f"What {arg_name.replace('_', ' ')} would you like to use?",
                    "description": ""
                })
        
        return clarifications
    
    def split_into_conversation_turns(self, query: str, function_docs: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Create multi-turn conversations."""
        analysis = self.extract_entities_and_arguments(query, function_docs)
        clarification_questions = self.generate_clarification_questions(analysis, function_docs)
        
        if not clarification_questions:
            return []
        
        # Limit to 6 clarification rounds
        clarification_questions = clarification_questions[:6]
        
        # Create simplified query
        simplified_query = self._create_simplified_query(query)
        
        conversation_turns = []
        
        # Turn 1: Simplified user query
        conversation_turns.append([{
            "role": "user",
            "content": simplified_query
        }])
        
        # Generate clarification turns
        for clarification in clarification_questions:
            # Assistant asks for clarification
            conversation_turns.append([{
                "role": "assistant",
                "content": clarification["question"]
            }])
            
            # User provides missing information
            conversation_turns.append([{
                "role": "user", 
                "content": self._generate_user_response(clarification, query)
            }])
        
        return conversation_turns
    
    def _create_simplified_query(self, original_query: str) -> str:
        """Create simplified version of query."""
        simplified = original_query
        
        # Remove specific details
        simplified = re.sub(r'\d+\s*(?:seconds?|minutes?|hours?)', '[TIME]', simplified)
        simplified = re.sub(r'\b\d{4,}\b', '[NUMBER]', simplified)
        
        return simplified
    
    def _generate_user_response(self, clarification: Dict[str, Any], original_query: str) -> str:
        """Generate user response from original query."""
        argument_name = clarification["argument_name"]
        
        # Extract specific values based on argument type
        if "id" in argument_name.lower():
            numbers = re.findall(r'\b\d+\b', original_query)
            if numbers:
                return numbers[0]
        
        elif "location" in argument_name.lower() or "address" in argument_name.lower():
            locations = re.findall(r'\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave)', original_query, re.IGNORECASE)
            if locations:
                return locations[0]
        
        elif "time" in argument_name.lower():
            times = re.findall(r'\d+\s*(?:seconds?|minutes?|hours?)', original_query, re.IGNORECASE)
            if times:
                return times[0]
        
        return f"[{argument_name}]"
    
    def _format_function_docs(self, function_docs: List[Dict[str, Any]]) -> str:
        """Format function documentation."""
        formatted = ""
        for func in function_docs:
            formatted += f"Function: {func.get('name', 'Unknown')}\n"
            formatted += f"Description: {func.get('description', 'No description')}\n"
            
            if "parameters" in func and "properties" in func["parameters"]:
                formatted += "Parameters:\n"
                for param_name, param_info in func["parameters"]["properties"].items():
                    required = " (required)" if param_name in func["parameters"].get("required", []) else ""
                    formatted += f"  - {param_name}{required}: {param_info.get('description', '')}\n"
            formatted += "\n"
        
        return formatted
    
    def process_test_case(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single test case."""
        # Debug: Check if test case is empty
        if not test_case or test_case == {}:
            return test_case
            
        if not self._is_single_turn(test_case):
            return test_case
        
        user_query = self._extract_user_query(test_case)
        function_docs = self._get_function_docs(test_case)
        
        if not user_query or not function_docs:
            return test_case
        
        if not self.is_complex_query(user_query, function_docs):
            return test_case
        
        try:
            conversation_turns = self.split_into_conversation_turns(user_query, function_docs)
            
            if conversation_turns:
                test_case["multi_turn_conversations"] = conversation_turns
                test_case["split_from_single_turn"] = True
                print(f"  - Created {len(conversation_turns)} turns for: {user_query[:50]}...")
            
        except Exception as e:
            print(f"Error processing test case {test_case.get('id', 'unknown')}: {e}")
        
        return test_case
    
    def _is_single_turn(self, test_case: Dict[str, Any]) -> bool:
        """Check if test case is single-turn."""
        # XLAM format uses 'messages' field
        if "messages" in test_case:
            messages = test_case["messages"]
            if isinstance(messages, list) and len(messages) >= 1:
                # Check if first message is from user
                first_message = messages[0]
                return first_message.get("role") == "user"
        
        # Legacy format uses 'question' field
        if "question" in test_case:
            question = test_case["question"]
            if isinstance(question, list) and len(question) == 1:
                first_turn = question[0]
                if isinstance(first_turn, list) and len(first_turn) == 1:
                    return first_turn[0].get("role") == "user"
        
        return False
    
    def _extract_user_query(self, test_case: Dict[str, Any]) -> Optional[str]:
        """Extract user query from test case."""
        # XLAM format uses 'messages' field
        if "messages" in test_case and test_case["messages"]:
            messages = test_case["messages"]
            if isinstance(messages, list) and len(messages) > 0:
                first_message = messages[0]
                if first_message.get("role") == "user":
                    return first_message.get("content", "")
        
        # Legacy format uses 'question' field
        if "question" in test_case and test_case["question"]:
            first_turn = test_case["question"][0]
            if isinstance(first_turn, list) and len(first_turn) > 0:
                first_message = first_turn[0]
                return first_message.get("content", "")
        
        return None
    
    def _get_function_docs(self, test_case: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get function documentation from test case."""
        # XLAM format uses 'tools' field with JSON string
        if "tools" in test_case:
            tools = test_case["tools"]
            if isinstance(tools, str):
                try:
                    return json.loads(tools)
                except json.JSONDecodeError:
                    return []
            elif isinstance(tools, list):
                return tools
        
        # Legacy format uses 'function' field
        return test_case.get("function", [])

def main():
    parser = argparse.ArgumentParser(description="Split complex single-turn queries")
    parser.add_argument("--input", required=True, help="Input JSON file")
    parser.add_argument("--output", required=True, help="Output JSON file")
    parser.add_argument("--chunk-start", type=int, default=0, help="Start index for chunk processing")
    parser.add_argument("--chunk-size", type=int, default=None, help="Chunk size for parallel processing")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of cases")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Load data
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            # Handle both JSON array and JSONL formats
            content = f.read().strip()
            if content.startswith('['):
                # JSON array format
                data = json.loads(content)
            else:
                # Multi-line JSON format - parse concatenated JSON objects
                data = []
                valid_count = 0
                error_count = 0
                
                # Split content by "}\n{" pattern to separate JSON objects
                json_objects = content.split('}\n{')
                
                for i, json_str in enumerate(json_objects):
                    # Add back the braces that were removed by split
                    if i > 0:
                        json_str = '{' + json_str
                    if i < len(json_objects) - 1:
                        json_str = json_str + '}'
                    
                    json_str = json_str.strip()
                    if json_str:
                        try:
                            parsed_obj = json.loads(json_str)
                            # Only add if it's a valid non-empty object with required fields
                            if isinstance(parsed_obj, dict) and parsed_obj and "messages" in parsed_obj:
                                data.append(parsed_obj)
                                valid_count += 1
                        except json.JSONDecodeError as e:
                            error_count += 1
                            if args.verbose and error_count <= 10:
                                print(f"Skipping malformed object {i+1}: {str(e)[:50]}...")
                            continue
                
                print(f"Loaded {valid_count} valid cases from multi-line JSON file ({error_count} errors skipped)")
    except Exception as e:
        print(f"Error loading input file: {e}")
        return
    
    if not isinstance(data, list):
        print("Error: Input file should contain a list")
        return
    
    # Handle chunking for parallel processing
    if args.chunk_size:
        end_idx = min(args.chunk_start + args.chunk_size, len(data))
        selected_cases = data[args.chunk_start:end_idx]
        print(f"Processing chunk {args.chunk_start}-{end_idx} ({len(selected_cases)} cases)")
    elif args.limit:
        selected_cases = data[:args.limit]
        print(f"Processing first {len(selected_cases)} cases")
    else:
        selected_cases = data
        print(f"Processing all {len(selected_cases)} cases")
    
    # Process cases
    splitter = SingleTurnComplexQuerySplitter()
    processed_cases = []
    complex_count = 0
    
    for i, test_case in enumerate(selected_cases):
        if args.verbose:
            print(f"Processing {i+1}/{len(selected_cases)}: {test_case.get('id', 'unknown')}")
        
        processed_case = splitter.process_test_case(test_case)
        processed_cases.append(processed_case)
        
        if processed_case.get("split_from_single_turn", False):
            complex_count += 1
    
    # Save results
    try:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(processed_cases, f, indent=2, ensure_ascii=False)
        
        print(f"\nProcessing complete!")
        print(f"Total cases processed: {len(processed_cases)}")
        print(f"Complex queries split: {complex_count}")
        print(f"Output saved to: {args.output}")
        
    except Exception as e:
        print(f"Error saving output: {e}")

if __name__ == "__main__":
    main()
