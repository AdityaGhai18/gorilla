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
        """Extract entity hints using basic regex patterns."""
        if not text:
            return ""
        
        entities = []
        
        # Extract potential names, numbers, and locations using regex
        # Names (capitalized words that might be names)
        potential_names = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        for name in potential_names[:3]:  # Limit to first 3
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
        return result
    
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
Voice-based user interactions often contain ambiguities that differ from text-based queries:
1. Names that could be spelled differently (e.g., "Alex" vs "Alec", "John" vs "Jon", "Zhang Wei", "Nguyen")
2. Technical terms, filepaths, URLs that could be ambiguous when spoken (e.g., "XContentBuilder" vs "X content builder")
3. Homophones that could be confused (e.g., "write" vs "right", "to" vs "too" vs "two")
4. Place names that could have spelling variations (e.g., "New York" vs "NewYork", "Los Angeles" vs "LA")
5. Complex alphanumeric strings, codes, or identifiers (e.g., "MIIFdTCCBF2gAwIBAgISESG", "ABC123XYZ", "UUID-123e4567-e89b-12d3-a456-426614174000")
6. Product names, brand names, or database names (e.g., "Firebird", "MongoDB", "PostgreSQL")
7. Code snippets, SQL queries, or commands with symbols (e.g., "SELECT * FROM table", "x = y + 1", "file.txt")
8. Technical constants, class names, or identifiers (e.g., "VMDeathRequest", "onScriptError.IGNORE")

IMPORTANT: Focus on GENUINE speech ambiguity, DO NOT fixate on just any function parameter. 

DO flag:
- Technical product names, database names, or brand names (e.g., "Elasticsearch", "Firebird", "MongoDB", "PostgreSQL")
- Technical terms, class names, and identifiers (e.g., "mappingParserContext", "compositeScriptFactory", "VMDeathRequest", "PrintStream", "logStream")
- Complex alphanumeric strings, codes, or identifiers
- Complex numbers with suffixes (e.g., "12345L", "54321L", "0xFF")
- Code snippets, SQL queries, or commands with symbols (e.g., "SELECT * FROM table", "x = y + 1")
- Names, place names, and technical terms that could be spelled differently

Do NOT flag:
- Common words that are clear when spoken (e.g., "burglary", "English", "marketing team", "could", "apps")
- Simple concepts that don't have spelling ambiguity (e.g., "new iPhone release")
- Generic terms that are obvious in context
- Numbers or dates that are straightforward
- Common verbs, prepositions, or articles (e.g., "could", "should", "the", "a", "an")

Your Task
1. Inputs:
   - User query (text input) - focus on the actual spoken content
   - Documentation for the exact function (its arguments and descriptions)
   - Optional entity hints
2. Actions:
   - Map provided arguments from the query to function parameters.
   - Identify ONLY arguments that have genuine spoken-ambiguity risks.
   - Focus on the content of what was spoken, not just the function parameter names.
   - Be reasonable - only flag things that could actually be unclear when spoken.
3. Output:
   - One JSON dictionary (no lists).
   - Keys:
     - <argument_name>_spelling for spelling checks
     - <argument_name>_1_spelling, <argument_name>_2_spelling for multiple names
   - Values:
     - The value from the user query that needs clarification.

---

Example 1: Name clarification (VALID)
User Query: "Schedule a meeting with Alex Johnson on Tuesday at 3pm"
Function Documentation:
schedule_meeting(person_name: string, date: string, time: string)
Output:
{{
  "person_name_spelling": "Alex Johnson"
}}

Example 2: Technical term clarification (VALID)
User Query: "Create an XContentBuilder object for the data"
Function Documentation:
create_builder(builder_type: string, data: object)
Output:
{{
  "builder_type_spelling": "XContentBuilder"
}}

Example 3: Place name clarification (VALID)
User Query: "Book a flight to New York on Friday"
Function Documentation:
book_flight(destination: string, date: string)
Output:
{{
  "destination_spelling": "New York"
}}

Example 4: Complex alphanumeric string (VALID)
User Query: "Create a constant named CERTIFICATE with value MIIFdTCCBF2gAwIBAgISESG"
Function Documentation:
create_constant(name: string, value: string)
Output:
{{
  "value_spelling": "MIIFdTCCBF2gAwIBAgISESG"
}}

Example 5: Multiple technical terms (VALID)
User Query: "Set up Elasticsearch with mappingParserContext and compositeScriptFactory, handle errors with onScriptError.IGNORE"
Function Documentation:
setup_elasticsearch(parser_context: string, script_factory: string, error_handling: string)
Output:
{{
  "parser_context_spelling": "mappingParserContext",
  "script_factory_spelling": "compositeScriptFactory",
  "error_handling_spelling": "onScriptError.IGNORE",
  "search_spelling": "Elasticsearch"
}}

Example 6: Database name and SQL query (VALID)
User Query: "Create a Firebird database view with query SELECT * FROM Employee WHERE status = 'active'"
Function Documentation:
create_view(database: string, query: string)
Output:
{{
  "database_spelling": "Firebird",
  "query_spelling": "SELECT * FROM Employee WHERE status = 'active'"
}}

Example 7: Technical class name (VALID)
User Query: "Handle VMDeathRequest in the JVM"
Function Documentation:
handle_request(request_type: string, target: string)
Output:
{{
  "request_type_spelling": "VMDeathRequest"
}}

Example 8: Technical class and complex numbers (VALID)
User Query: "Use PrintStream logStream with values 12345L and 54321L"
Function Documentation:
use_stream(stream_type: string, stream_name: string, old_value: string, new_value: string)
Output:
{{
  "stream_type_spelling": "PrintStream",
  "stream_name_spelling": "logStream",
  "old_value_spelling": "12345L",
  "new_value_spelling": "54321L"
}}

Example 9: Database name in context (VALID)
User Query: "Check Elasticsearch cluster status"
Function Documentation:
check_status(database: string, component: string)
Output:
{{
  "database_spelling": "Elasticsearch"
}}

Example 10: Common words (NO CLARIFICATION NEEDED)
User Query: "Could you help me with the apps database?"
Function Documentation:
help_with_database(database: string, action: string)
Output:
{{}}

Example 11: Common word (NO CLARIFICATION NEEDED)
User Query: "Report a burglary incident"
Function Documentation:
report_incident(crime_type: string, details: string)
Output:
{{}}

Example 6: Simple concept (NO CLARIFICATION NEEDED)
User Query: "Write about the new iPhone release in English"
Function Documentation:
write_article(topic: string, language: string)
Output:
{{}}

Example 7: Multiple names (VALID)
User Query: "Schedule a meeting with John Smith and Mary Johnson tomorrow"
Function Documentation:
schedule_meeting(participant_names: list, date: string)
Output:
{{
  "participant_names_1_spelling": "John Smith",
  "participant_names_2_spelling": "Mary Johnson"
}}

Example 8: No relevant parameters (NO CLARIFICATION NEEDED)
User Query: "What's the weather like today?"
Function Documentation:
get_weather()
Output:
{{}}

---

Task Inputs:
User Query: {user_query}
Function Documentation: {function_docs_text}
Entity Hints: {entity_hints}

IMPORTANT: The entity hints below are often WRONG and should be IGNORED. They frequently misidentify:
- Common words as names (e.g., "could" as a person, "apps" as a database)
- Technical terms as locations
- Generic words as specific entities

DO NOT rely on entity hints. Instead, use your own judgment to identify genuine speech ambiguities based on the user query content.

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
            if len(data) <= 5:
                test_cases = data
            else:
                test_cases = random.sample(data, 5)
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
                print(f"ENTITY HINTS: {entity_hints}")
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