#!/usr/bin/env python3
"""
Clarification Extraction Module for Voice Assistant Function Calls

This module identifies clarification points for voice assistant queries by:
1. Detecting missing or incomplete arguments
2. Identifying spoken-ambiguity risks (rare names, homophones, etc.)
3. Generating a dictionary of valid clarifications for benchmarking

Supports both single-turn and multi-turn test cases.
"""

import json
import os
import random
import argparse
import time
import re
from typing import Dict, List, Optional, Any
from pathlib import Path
from dotenv import load_dotenv
import openai

# Try to import spaCy for NLP entity extraction
try:
    import spacy
    SPACY_AVAILABLE = True
    # Load English model
    nlp = spacy.load("en_core_web_sm")
except ImportError:
    SPACY_AVAILABLE = False
    print("Warning: spaCy not available. Entity hints will be empty.")
except OSError:
    SPACY_AVAILABLE = False
    print("Warning: spaCy English model not found. Install with: python -m spacy download en_core_web_sm")

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-4o"
MAX_RETRIES = 3
RETRY_DELAY = 1.0

class ClarificationExtractor:
    """Main class for extracting clarifications from voice assistant queries."""
    
    def __init__(self):
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
        
    def get_function_docs_single_turn(self, test_case: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract function documentation from single-turn test case."""
        if "function" in test_case:
            return test_case["function"]
        return []
    
    def get_function_docs_multi_turn(self, test_case: Dict[str, Any], multi_turn_func_doc_dir: str) -> List[Dict[str, Any]]:
        """Extract function documentation for multi-turn test case from external files."""
        if "path" not in test_case:
            return []
        
        function_docs = []
        path = test_case["path"]
        
        # Load all available function documentation files
        func_doc_files = {
            "gorilla_file_system.json": "bfcl_eval/data/multi_turn_func_doc/gorilla_file_system.json",
            "ticket_api.json": "bfcl_eval/data/multi_turn_func_doc/ticket_api.json",
            "trading_bot.json": "bfcl_eval/data/multi_turn_func_doc/trading_bot.json",
            "travel_booking.json": "bfcl_eval/data/multi_turn_func_doc/travel_booking.json",
            "vehicle_control.json": "bfcl_eval/data/multi_turn_func_doc/vehicle_control.json",
            "math_api.json": "bfcl_eval/data/multi_turn_func_doc/math_api.json",
            "message_api.json": "bfcl_eval/data/multi_turn_func_doc/message_api.json",
            "posting_api.json": "bfcl_eval/data/multi_turn_func_doc/posting_api.json"
        }
        
        # Load all function documentation
        all_functions = {}
        for file_path in func_doc_files.values():
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, dict) and "functions" in data:
                        all_functions.update(data["functions"])
                    elif isinstance(data, list):
                        for func in data:
                            if isinstance(func, dict) and "name" in func:
                                all_functions[func["name"]] = func
            except Exception as e:
                print(f"Warning: Could not load {file_path}: {e}")
        
        # Extract functions referenced in the path
        for func_path in path:
            # Handle both "Class.function" and "function" formats
            if "." in func_path:
                class_name, func_name = func_path.split(".", 1)
                full_func_name = f"{class_name}.{func_name}"
            else:
                full_func_name = func_path
            
            if full_func_name in all_functions:
                function_docs.append(all_functions[full_func_name])
            else:
                # Try to find by just the function name
                for func_key, func_doc in all_functions.items():
                    if func_key.endswith(f".{func_path}") or func_key == func_path:
                        function_docs.append(func_doc)
                        break
        
        return function_docs
    
    def extract_user_query(self, test_case: Dict[str, Any]) -> str:
        """Extract the user query from a test case using the transcript field."""
        if "question" not in test_case:
            return ""
        
        questions = test_case["question"]
        if not questions or not questions[0]:
            return ""
        
        # Get the first user message from the first turn
        first_turn = questions[0]
        for message in first_turn:
            if message.get("role") == "user":
                # Use transcript field instead of content field
                return message.get("transcript", message.get("content", ""))
        
        return ""
    
    def extract_entity_hints(self, text: str) -> str:
        """Extract entity hints using NLP methods."""
        if not SPACY_AVAILABLE or not text:
            return "spaCy not available - using basic regex extraction"
        
        try:
            doc = nlp(text)
            entities = []
            
            for ent in doc.ents:
                entity_info = f"{ent.label_}: {ent.text}"
                entities.append(entity_info)
            
            # Also extract potential names, numbers, and locations using regex
            # Names (capitalized words that might be names)
            potential_names = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
            for name in potential_names[:3]:  # Limit to first 3
                if name not in [ent.text for ent in doc.ents]:
                    entities.append(f"PERSON: {name}")
            
            # Numbers
            numbers = re.findall(r'\b\d+(?:\.\d+)?\b', text)
            for num in numbers[:5]:  # Limit to first 5
                entities.append(f"NUMBER: {num}")
            
            # Dates/times
            date_patterns = [
                r'\b(?:today|tomorrow|yesterday|next week|last week)\b',
                r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b',
                r'\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b'
            ]
            for pattern in date_patterns:
                matches = re.findall(pattern, text.lower())
                for match in matches[:3]:
                    entities.append(f"DATE: {match}")
            
            result = "; ".join(entities) if entities else "No entities found"
            print(f"spaCy entities extracted: {result}")
            return result
            
        except Exception as e:
            print(f"Warning: Error extracting entities: {e}")
            return ""
    
    def generate_clarifications(self, user_query: str, function_docs: List[Dict[str, Any]], entity_hints: str = "") -> Dict[str, str]:
        """Generate clarifications using GPT-4o."""
        
        # Extract entity hints if not provided
        if not entity_hints:
            entity_hints = self.extract_entity_hints(user_query)
        
        # Format function documentation for the prompt
        function_docs_text = ""
        for i, func in enumerate(function_docs):
            function_docs_text += f"Function {i+1}:\n"
            function_docs_text += f"Name: {func.get('name', 'Unknown')}\n"
            function_docs_text += f"Description: {func.get('description', 'No description')}\n"
            
            if "parameters" in func:
                params = func["parameters"]
                if "properties" in params:
                    function_docs_text += "Parameters:\n"
                    for param_name, param_info in params["properties"].items():
                        param_type = param_info.get("type", "unknown")
                        param_desc = param_info.get("description", "No description")
                        function_docs_text += f"  - {param_name} ({param_type}): {param_desc}\n"
                
                if "required" in params:
                    function_docs_text += f"Required parameters: {', '.join(params['required'])}\n"
            
            function_docs_text += "\n"
        
        prompt = f"""You are an AI system that identifies clarification points for voice assistant queries.
The goal is to help an AI that will call a function ask valid clarifying questions.

Why Clarifications Are Needed
Voice-based user interactions often contain ambiguities and uncertainties that differ from purely text-based queries:
1. Uncommon or foreign names that are difficult to spell (e.g., "Jukkasjärvi", "Zhang Wei", "Nguyen")
2. Technical terms, filepaths, urls, that sound like different words when spoken or are too long or spelled out weirdly (e.g., "XContentBuilder" sounds like "X content builder")
3. Homophones that could be confused (e.g., "write" vs "right", "to" vs "too" vs "two")
4. Complex product names or brand names that are unusual

IMPORTANT: DO NOT ask for spelling clarification on:
- Common English names (e.g., "John", "Mary", "Smith")
- Common place names (e.g., "New York", "London", "Paris")
- Common words that are clear when spoken (e.g., "football", "World Cup", "twenty twenty-one")
- Simple technical terms that are clear (e.g., "userProfile", "map", "data")
- Any term that would be obvious to a native English speaker

Your Task
1. Inputs:
   - User query (text input)
   - Documentation for the exact function (its arguments and descriptions)
   - Optional entity hints
2. Actions:
   - Map provided arguments from the query to function parameters.
   - Identify ONLY arguments that have genuine spoken-ambiguity risks.
   - If no genuine ambiguities exist, return an empty dictionary {{}}.
   - Be very conservative - only flag things that are truly unclear.
3. Output:
   - One JSON dictionary (no lists).
   - Keys:
     - <argument_name>_spelling for spelling checks
     - <argument_name>_1_spelling, <argument_name>_2_spelling for multiple names
   - Values:
     - The value from the user query that needs clarification.

---

Example 1: Foreign name clarification (VALID)
User Query: "Schedule a meeting with Zhang Wei on Tuesday at 3pm"
Function Documentation:
schedule_meeting(person_name: string, date: string, time: string)
Output:
{{
  "person_name_spelling": "Zhang Wei"
}}

Example 2: Technical term clarification (VALID)
User Query: "Create an XContentBuilder object for the data"
Function Documentation:
create_builder(builder_type: string, data: object)
Output:
{{
  "builder_type_spelling": "XContentBuilder"
}}

Example 3: Common name (NO CLARIFICATION NEEDED)
User Query: "Send a package to John Smith at 123 Main Street"
Function Documentation:
send_package(recipient_name: string, address: string)
Output:
{{}}

Example 4: Common place (NO CLARIFICATION NEEDED)
User Query: "Book a flight to New York on Friday"
Function Documentation:
book_flight(destination: string, date: string)
Output:
{{}}

Example 5: Simple technical term (NO CLARIFICATION NEEDED)
User Query: "Create a userProfile map with name and age"
Function Documentation:
create_map(map_name: string, fields: list)
Output:
{{}}

Example 6: Multiple foreign names (VALID)
User Query: "Schedule a meeting with Priya Sharma and Nguyen Van Minh tomorrow"
Function Documentation:
schedule_meeting(participant_names: list, date: string)
Output:
{{
  "participant_names_1_spelling": "Priya Sharma",
  "participant_names_2_spelling": "Nguyen Van Minh"
}}

---

Task Inputs:
User Query: {user_query}
Function Documentation: {function_docs_text}
Entity Hints: {entity_hints}

Use the entity hints as weak guidance only - they may contain errors and should not override your judgment:
- PERSON entities might need spelling clarification if they're foreign names
- ORG entities might need clarification if they're unusual company names
- GPE (location) entities are usually clear and don't need clarification
- CARDINAL numbers might need clarification if they're complex or spelled out
- Treat entity hints as suggestions only - rely primarily on your own analysis of what's ambiguous

Task Output:
(Return only one JSON dictionary)"""
        
        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1000,
                    temperature=0.0,
                    response_format={"type": "json_object"}
                )
                
                result = response.choices[0].message.content.strip()
                clarifications = json.loads(result)
                
                # Validate that it's a dictionary
                if not isinstance(clarifications, dict):
                    print(f"Warning: Expected dictionary, got {type(clarifications)}")
                    return {}
                
                return clarifications
                
            except json.JSONDecodeError as e:
                print(f"JSON decode error on attempt {attempt + 1}: {e}")
                if attempt == MAX_RETRIES - 1:
                    return {}
                time.sleep(RETRY_DELAY)
                
            except Exception as e:
                print(f"OpenAI API error on attempt {attempt + 1}: {e}")
                if attempt == MAX_RETRIES - 1:
                    return {}
                time.sleep(RETRY_DELAY)
        
        return {}
    
    def is_single_turn(self, test_case: Dict[str, Any]) -> bool:
        """Determine if a test case is single-turn or multi-turn."""
        # Single-turn cases have "function" field, multi-turn have "path" field
        return "function" in test_case and "path" not in test_case
    
    def process_test_case(self, test_case: Dict[str, Any], multi_turn_func_doc_dir: str = "bfcl_eval/data/multi_turn_func_doc") -> Dict[str, Any]:
        """Process a single test case and add clarifications."""
        # Extract user query
        user_query = self.extract_user_query(test_case)
        if not user_query:
            print(f"Warning: No user query found for test case {test_case.get('id', 'unknown')}")
            return test_case
        
        # Get function documentation based on test case type
        if self.is_single_turn(test_case):
            function_docs = self.get_function_docs_single_turn(test_case)
        else:
            function_docs = self.get_function_docs_multi_turn(test_case, multi_turn_func_doc_dir)
        
        if not function_docs:
            print(f"Warning: No function documentation found for test case {test_case.get('id', 'unknown')}")
            return test_case
        
        # Generate clarifications
        clarifications = self.generate_clarifications(user_query, function_docs)
        
        # Add clarifications to the first user message in the first turn
        if "question" in test_case and test_case["question"]:
            first_turn = test_case["question"][0]
            for message in first_turn:
                if message.get("role") == "user":
                    message["clarifications"] = clarifications
                    break
        
        return test_case
    
    def process_file(self, file_path: str, output_path: str = None, test_mode: bool = False) -> None:
        """Process a JSON file and show clarifications preview without making changes."""
        print(f"Processing file: {file_path}")
        
        # Load the JSON file
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            print(f"Error: Expected list of test cases, got {type(data)}")
            return
        
        # Test mode: process only 3 random cases
        if test_mode:
            if len(data) <= 3:
                test_cases = data
            else:
                test_cases = random.sample(data, 3)
            print(f"Test mode: Processing {len(test_cases)} random test cases")
        else:
            test_cases = data
            print(f"Processing all {len(test_cases)} test cases")
        
        # Process each test case (PREVIEW MODE - NO CHANGES)
        processed_count = 0
        for i, test_case in enumerate(test_cases):
            try:
                # Extract user query
                user_query = self.extract_user_query(test_case)
                if not user_query:
                    print(f"  Test case {i+1}: No user query found, skipping")
                    continue
                
                # Get function documentation based on test case type
                if self.is_single_turn(test_case):
                    function_docs = self.get_function_docs_single_turn(test_case)
                else:
                    function_docs = self.get_function_docs_multi_turn(test_case, "bfcl_eval/data/multi_turn_func_doc")
                
                if not function_docs:
                    print(f"  Test case {i+1}: No function documentation found, skipping")
                    continue
                
                # Extract entity hints using spaCy
                entity_hints = self.extract_entity_hints(user_query)
                
                # Generate clarifications
                clarifications = self.generate_clarifications(user_query, function_docs, entity_hints)
                
                # PREVIEW MODE: Show what would be added without making changes
                test_case_id = test_case.get("id", f"test_case_{i+1}")
                
                # Get content field for logging (even though we use transcript)
                content = ""
                if "question" in test_case and test_case["question"]:
                    first_turn = test_case["question"][0]
                    for message in first_turn:
                        if message.get("role") == "user":
                            content = message.get("content", "")
                            break
                
                print(f"\n{'='*80}")
                print(f"TEST CASE {i+1}: {test_case_id}")
                print(f"{'='*80}")
                print(f"CONTENT: {content}")
                print(f"TRANSCRIPT: {user_query}")
                print(f"SPACY ENTITIES: {entity_hints}")
                print(f"FUNCTION DOCS: {json.dumps(function_docs, indent=2)}")
                print(f"CLARIFICATIONS TO ADD: {json.dumps(clarifications, indent=2)}")
                print(f"{'='*80}\n")
                
                processed_count += 1
                    
            except Exception as e:
                print(f"Error processing test case {test_case.get('id', 'unknown')}: {e}")
        
        print(f"PREVIEW MODE: No changes made to {file_path}")
        print(f"  - Total test cases: {len(test_cases)}")
        print(f"  - Successfully processed: {processed_count}")
        print(f"  - Failed: {len(test_cases) - processed_count}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Extract clarifications from voice assistant test cases")
    parser.add_argument("file_path", help="Path to the JSON file to process")
    parser.add_argument("--output", "-o", help="Output file path (defaults to input file)")
    parser.add_argument("--test", "-t", action="store_true", help="Test mode: process only 3 random test cases")
    
    args = parser.parse_args()
    
    # Validate file exists
    if not os.path.exists(args.file_path):
        print(f"Error: File {args.file_path} does not exist")
        return
    
    # Initialize extractor and process file
    extractor = ClarificationExtractor()
    extractor.process_file(args.file_path, args.output, args.test)


if __name__ == "__main__":
    main() 