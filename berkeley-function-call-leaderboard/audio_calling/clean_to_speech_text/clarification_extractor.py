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
            "gorilla_file_system.json": "/Users/adityaghai/Desktop/BFCL/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/multi_turn_func_doc/gorilla_file_system.json",
            "ticket_api.json": "/Users/adityaghai/Desktop/BFCL/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/multi_turn_func_doc/ticket_api.json",
            "trading_bot.json": "/Users/adityaghai/Desktop/BFCL/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/multi_turn_func_doc/trading_bot.json",
            "travel_booking.json": "/Users/adityaghai/Desktop/BFCL/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/multi_turn_func_doc/travel_booking.json",
            "vehicle_control.json": "/Users/adityaghai/Desktop/BFCL/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/multi_turn_func_doc/vehicle_control.json",
            "math_api.json": "/Users/adityaghai/Desktop/BFCL/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/multi_turn_func_doc/math_api.json",
            "message_api.json": "/Users/adityaghai/Desktop/BFCL/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/multi_turn_func_doc/message_api.json",
            "posting_api.json": "/Users/adityaghai/Desktop/BFCL/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/multi_turn_func_doc/posting_api.json"
        }
        
        # Load all function documentation robustly (supports JSON, JSONL, or concatenated JSON)
        all_functions = {}
        for file_path in func_doc_files.values():
            try:
                with open(file_path, 'r') as f:
                    content = f.read().strip()
                    # Try JSON array or object
                    try:
                        data = json.loads(content)
                        if isinstance(data, dict) and "functions" in data:
                            all_functions.update(data["functions"])
                        elif isinstance(data, list):
                            for func in data:
                                if isinstance(func, dict) and "name" in func:
                                    all_functions[func["name"]] = func
                    except json.JSONDecodeError:
                        # Try JSONL (one JSON object per line)
                        f.seek(0)
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                func = json.loads(line)
                                if isinstance(func, dict) and "name" in func:
                                    all_functions[func["name"]] = func
                            except Exception:
                                continue
            except Exception as e:
                print(f"Warning: Could not load {file_path}: {e}")
        
        # Extract functions referenced in the path
        for func_path in path:
            # Handle both "Class.function" and "function" formats
            if "." in func_path:
                class_name, func_name = func_path.split(".", 1)
                # Look for the function by just its name (without class prefix)
                if func_name in all_functions:
                    function_docs.append(all_functions[func_name])
                else:
                    # Try to find by full name or partial match
                    for func_key, func_doc in all_functions.items():
                        if func_key == func_name or func_key.endswith(f".{func_name}") or func_key == func_path:
                            function_docs.append(func_doc)
                            break
            else:
                # Direct function name
                if func_path in all_functions:
                    function_docs.append(all_functions[func_path])
                else:
                    # Try to find by partial match
                    for func_key, func_doc in all_functions.items():
                        if func_key == func_path or func_key.endswith(f".{func_path}"):
                            function_docs.append(func_doc)
                            break
        
        return function_docs
    
    def extract_user_query(self, test_case: Dict[str, Any]) -> str:
        """Extract the user query from a test case using the content field."""
        if "question" not in test_case:
            return ""
        
        questions = test_case["question"]
        if not questions or not questions[0]:
            return ""
        
        # Get the first user message from the first turn
        first_turn = questions[0]
        for message in first_turn:
            if message.get("role") == "user":
                # Use content field instead of transcript field
                return message.get("content", message.get("transcript", ""))
        
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
    
    def generate_clarifications(self, user_query: str, function_docs: List[Dict[str, Any]]) -> Dict[str, str]:
        """Generate clarifications using GPT-4o."""
        

        
        # Format function documentation for the prompt
        function_docs_text = ""
        for i, func in enumerate(function_docs):
            try:
                function_docs_text += f"Function {i+1}:\n"
                function_docs_text += f"Name: {func.get('name', 'Unknown')}\n"
                function_docs_text += f"Description: {func.get('description', 'No description')}\n"
                
                if "parameters" in func:
                    params = func["parameters"]
                    if "properties" in params:
                        function_docs_text += "Parameters:\n"
                        for param_name, param_info in params["properties"].items():
                            try:
                                param_type = param_info.get("type", "unknown")
                                param_desc = param_info.get("description", "No description")
                                function_docs_text += f"  - {param_name} ({param_type}): {param_desc}\n"
                                

                            except Exception as e:
                                print(f"Warning: Error processing parameter {param_name}: {e}")
                                continue
                    
                    if "required" in params:
                        function_docs_text += f"Required parameters: {', '.join(params['required'])}\n"
                
                function_docs_text += "\n"
            except Exception as e:
                print(f"Warning: Error processing function {i+1}: {e}")
                continue
        
        prompt = f"""You are an AI system that identifies clarification points for voice assistant queries.
The goal is to help an AI that will call a function ask valid clarifying questions.

Why Clarifications Are Needed
Voice-based user interactions often contain ambiguities that differ from text-based queries:
1. Names that could be spelled differently (e.g., "Zhang Wei", "Nguyen Van Minh", "Shishir")
2. Technical terms, filepaths, URLs that could be ambiguous when spoken (e.g., "XContentBuilder" vs "X content builder")
3. Complex alphanumeric strings, codes, or identifiers (e.g., "MIIFdTCCBF2gAwIBAgISESG", "d0404", "ABC123XYZ")
4. Technical product names, database names, or brand names (e.g., "Elasticsearch", "Firebird", "MongoDB")
5. Code snippets, SQL queries, or commands with symbols (e.g., "SELECT * FROM table", "x = y + 1")
6. Technical constants, class names, or identifiers (e.g., "VMDeathRequest", "onScriptError.IGNORE")
7. Street addresses or specific locations that are critical for function calls
8. Unique identifiers or IDs that are essential for function execution
9. Homophones that could be confused (e.g., "write" vs "right", "to" vs "too" vs "two")

IMPORTANT: Be REASONABLE. Flag items that have genuine spelling ambiguity OR are specific identifiers/names that could be unclear when spoken and are essential for the correctness of the function call.

DO flag:
- Unusual or foreign names (e.g., "Zhang Wei", "Nguyen Van Minh", "Shishir")
- Complex technical terms (e.g., "mappingParserContext", "compositeScriptFactory", "VMDeathRequest")
- Complex alphanumeric strings or codes (e.g., "MIIFdTCCBF2gAwIBAgISESG", "d0404", "ABC123XYZ")
- Technical product names (e.g., "Elasticsearch", "Firebird", "MongoDB")
- Street addresses or specific locations (e.g., "123 Main Street", "456 Oak Avenue")
- Unique identifiers or IDs (e.g., "d0404", "user_123", "config_456")
- Complex numbers with suffixes (e.g., "12345L", "54321L", "0xFF")
- Technical terms, class names, and identifiers (e.g., "mappingParserContext", "compositeScriptFactory", "VMDeathRequest", "PrintStream", "logStream")
- Code snippets, SQL queries, or commands with symbols (e.g., "SELECT * FROM table", "x = y + 1")
- Technical constants, class names, or identifiers (e.g., "VMDeathRequest", "onScriptError.IGNORE")
- Specific event names, task names, or content (e.g., "go for shopping at 9 pm", "budget analysis meeting")
- Technical identifiers or keys (e.g., "site.info", "apikey_info", "config_key")


Do NOT flag:
- Common English words (e.g., "water", "iron", "burglary", "English", "marketing team", "tenant rights")
- Common English names (e.g., "John", "Mary", "Alex", "Smith")
- Common place names or obvious abbreviations (e.g., "New York", "London", "LA" for Los Angeles, "Paris", "Tokyo", "Sydney")
- Simple technical terms (e.g., "userProfile", "config.json")
- Common verbs, prepositions, or articles (e.g., "could", "should", "the", "a", "an")
- Simple concepts or generic terms (e.g., "new iPhone release", "Queue is saturated")
- Numbers or coordinates that are straightforward (e.g., "ten, fifteen", "twenty, twenty-five")
- Simple phrases or expressions (e.g., "Happy Birthday!", "Queue is unsaturated")
- Obvious non-identifiers (e.g., "/q", "quit", "exit", "help")
- Major world cities and countries (e.g., "London", "Paris", "Tokyo", "New York", "USA", "UK", "Canada")

Your Task
1. Inputs:
   - User query (text input) - focus on the actual content
   - Documentation for the exact function (its arguments and descriptions)
2. Actions:
   - Identify ALL potentially ambiguous class names, method names, technical terms, and identifiers in the query.
   - Focus on SUPER IMPORTANT and AMBIGUOUS details that need confirmation for the function call.
   - Flag any critical information that could be unclear when spoken, regardless of which function parameter it maps to.
   - Map these to appropriate function parameters when possible, but don't limit yourself to only function parameters.
   - Focus on the content of what was spoken - any technical term that could be unclear when spoken should be flagged.
   - Be reasonable - flag things that could actually be unclear when spoken, even if they're just context or mentioned in passing.
3. Output:
   - One JSON dictionary (no lists).
   - Keys: Use DESCRIPTIVE, NATURAL LANGUAGE names based on the parameter descriptions:
     - Look at the parameter description to create meaningful names
     - Use your understanding of the context to create appropriate, descriptive names
     - Don't rely on rigid suffix rules - be creative and descriptive
     - Examples: "start_date_spelling", "user_name", "file_path", "database_name", "street_address", "id", "validity_seconds", "encoding_method"
     - NOT: "r_spelling", "e_spelling", "param1_spelling", "validity_spelling", "algorithm_spelling" (avoid lazy "_spelling" suffix for everything)
     - For multiple items: "first_name_spelling", "second_name_spelling"
   - Values:
     - The value from the user query that needs clarification.

---

Example: Foreign name clarification (VALID)
User Query: "Schedule a meeting with Zhang Wei on Tuesday at 3pm"
Function Documentation:
schedule_meeting(person_name: string, date: string, time: string)
Output:
{{
  "person_name_spelling": "Zhang Wei"
}}

Example: Technical term clarification (VALID)
User Query: "Create an XContentBuilder object for the data"
Function Documentation:
create_builder(builder_type: string, data: object)
Output:
{{
  "builder_type_spelling": "XContentBuilder"
}}

Example: Event details clarification (VALID)
User Query: "Who was the U.S. president during the Civil War?"
Function Documentation:
US_President_During_Event(event: string, country: string)
Output:
{{
  "event_details": "Civil War"
}}

Example: Common place name (NO CLARIFICATION NEEDED)
User Query: "Book a flight to New York on Friday"
Function Documentation:
book_flight(destination: string, date: string)
Output:
{{}}

Example: Major city name (NO CLARIFICATION NEEDED)
User Query: "Get me a house to stay for 4 in London"
Function Documentation:
search_house(location: string, guests: integer)
Output:
{{}}

Example: Important ambiguous details (VALID)
User Query: "I'm planning ahead for our big trip and realized our car's fuel tank is running low. It would be great if you could top it up with an additional 30 gallons before I turn on the ignition using the 'START' mode."
Function Documentation:
fillFuelTank(fuelAmount: float), startEngine(ignitionMode: string)
Output:
{{
  "fuel_amount_spelling": "30 gallons"
}}

Example: Descriptive parameter names (VALID)
User Query: "Help me generate an authorization token for a user with username 'johndoe', valid for '3600' seconds, issued by 'myapp.net', with a role of 'admin', and encoded with 'HS256' algorithm?"
Function Documentation:
createAuthToken(username: string, validity: integer, options: issuer: string, role: string, algorithm: string)
Output:
{{
  "username": "johndoe",
  "validity_seconds": "3600",
  "issuer": "myapp.net",
  "user_role": "admin",
  "encoding_method": "HS256"
}}

Example: Complex alphanumeric string (VALID)
User Query: "Create a constant named CERTIFICATE with value MIIFdTCCBF2gAwIBAgISESG"
Function Documentation:
create_constant(name: string, value: string)
Output:
{{
  "constant_name_spelling": "CERTIFICATE",
  "value_id": "MIIFdTCCBF2gAwIBAgISESG"
}}

Example: Multiple technical terms (VALID)
User Query: "Set up Elasticsearch with mappingParserContext and compositeScriptFactory, handle errors with onScriptError.IGNORE"
Function Documentation:
setup_elasticsearch(parser_context: string, script_factory: string, error_handling: string)
Output:
{{
  "search_spelling": "Elasticsearch", 
  "parser_context_spelling": "mappingParserContext",
  "script_factory_spelling": "compositeScriptFactory",
  "error_handling_spelling": "onScriptError.IGNORE"
}}

Example: Database name and SQL query (VALID)
User Query: "Create a Firebird database view with query SELECT * FROM Employee WHERE status = 'active'"
Function Documentation:
create_view(database: string, query: string)
Output:
{{
  "database_name": "Firebird",
  "query": "SELECT * FROM Employee WHERE status = 'active'"
}}

Example: Technical class name (VALID)
User Query: "Handle VMDeathRequest in the JVM"
Function Documentation:
handle_request(request_type: string, target: string)
Output:
{{
  "request_type_constant": "VMDeathRequest"
}}

Example: Technical class and complex numbers (VALID)
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

Example: Database name in context (VALID)
User Query: "Check Elasticsearch cluster status"
Function Documentation:
check_status(database: string, component: string)
Output:
{{
  "database_spelling": "Elasticsearch"
}}

Example: Common words (NO CLARIFICATION NEEDED)
User Query: "Could you help me with the apps database?"
Function Documentation:
help_with_database(database: string, action: string)
Output:
{{}}

Example: Common words (NO CLARIFICATION NEEDED)
User Query: "Report a burglary incident"
Function Documentation:
report_incident(crime_type: string, details: string)
Output:
{{}}

Example: Ridiculous clarifications (NO CLARIFICATION NEEDED)
User Query: "Add water and iron to the mixture"
Function Documentation:
add_ingredients(ingredient1: string, ingredient2: string)
Output:
{{}}

Example: Common names (NO CLARIFICATION NEEDED)
User Query: "Schedule a meeting with John Smith and Mary Johnson tomorrow"
Function Documentation:
schedule_meeting(participant_names: list, date: string)
Output:
{{}}

Example: Simple concept (NO CLARIFICATION NEEDED)
User Query: "Write about the new iPhone release in English"
Function Documentation:
write_article(topic: string, language: string)
Output:
{{}}

Example: Common names (NO CLARIFICATION NEEDED)
User Query: "Schedule a meeting with John Smith and Mary Johnson tomorrow"
Function Documentation:
schedule_meeting(participant_names: list, date: string)
Output:
{{}}

Example: Street address clarification (VALID)
User Query: "Send package to 123 Main Street, New York"
Function Documentation:
send_package(address: string, city: string)
Output:
{{
  "address_address": "123 Main Street"
}}

Example: Unique ID clarification (VALID)
User Query: "Delete the Apdex config for d0404"
Function Documentation:
delete_apdex_configuration(id: string)
Output:
{{
  "id": "d0404"
}}

Example: Foreign name clarification (VALID)
User Query: "Send message to Shishir"
Function Documentation:
send_message(recipient: string, message: string)
Output:
{{
  "recipient_name": "Shishir"
}}

Example: Simple phrases (NO CLARIFICATION NEEDED)
User Query: "Log Queue is saturated message"
Function Documentation:
log_message(message: string)
Output:
{{}}

Example: Simple coordinates (NO CLARIFICATION NEEDED)
User Query: "Calculate coordinates (ten, fifteen) and (twenty, twenty-five)"
Function Documentation:
calculate_coordinates(point1: string, point2: string)
Output:
{{}}

Example: Common words (NO CLARIFICATION NEEDED)
User Query: "Generate tenant rights contract"
Function Documentation:
generate_contract(contract_type: string)
Output:
{{}}

Example: Technical identifiers (VALID)
User Query: "Get DNS resolutions for site.info using apikey_info"
Function Documentation:
get_dns(domain: string, api_key: string)
Output:
{{
  "domain_id": "site.info",
  "api_key_id": "apikey_info"
}}

Example: Specific task content (VALID)
User Query: "Delete todo item 'go for shopping at 9 pm'"
Function Documentation:
delete_todo(content: string)
Output:
{{
  "content_spelling": "go for shopping at 9 pm"
}}

Example: Obvious non-identifier (NO CLARIFICATION NEEDED)
User Query: "Type /q to quit"
Function Documentation:
quit_command(command: string)
Output:
{{}}

Example: Descriptive parameter names (VALID)
User Query: "Calculate difference between dates '2023-04-01' and '2023-04-15'"
Function Documentation:
calculate_date_difference(r: string, e: string, t: string)
# r: "The start date for the calculation"
# e: "The end date for the calculation" 
# t: "The unit of time to calculate the difference in"
Output:
{{
  "start_date_spelling": "2023-04-01",
  "end_date_spelling": "2023-04-15"
}}

Example: Multiple technical terms in context (VALID)
User Query: "I'm trying to fix a compilation error. The class 'StringNumberHandler' extends 'AbstractCellHandler' and overrides methods like 'getCellValue', 'setCellValue', 'getExcelType'. I'm getting an error with 'CellResult' and 'getNumericValue()'. Find relevant classes for 'CellResult'?"
Function Documentation:
get_relevant_classes(search_string: string)
Output:
{{
  "search_string_spelling": "CellResult",
  "context_class1_spelling": "StringNumberHandler",
  "context_class2_spelling": "AbstractCellHandler", 
  "context_method1_spelling": "getCellValue",
  "context_method2_spelling": "setCellValue",
  "context_method3_spelling": "getExcelType",
  "context_method4_spelling": "getNumericValue"
}}

Example: No relevant parameters (NO CLARIFICATION NEEDED)
User Query: "What's the weather like today?"
Function Documentation:
get_weather()
Output:
{{}}

---

Task Inputs:
User Query: {user_query}
Function Documentation: {function_docs_text}

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
        try:
            clarifications = self.generate_clarifications(user_query, function_docs)
        except Exception as e:
            print(f"Error generating clarifications for test case {test_case.get('id', 'unknown')}: {e}")
            import traceback
            traceback.print_exc()
            clarifications = {}
        
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
        
        # Test mode: process only first 3 cases
        if test_mode:
            test_cases = data[:3]  # Take first 3 test cases
            print(f"Test mode: Processing first {len(test_cases)} test cases")
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
                
                # Generate clarifications
                try:
                    clarifications = self.generate_clarifications(user_query, function_docs)
                except Exception as e:
                    print(f"Error generating clarifications for test case {test_case.get('id', 'unknown')}: {e}")
                    import traceback
                    traceback.print_exc()
                    clarifications = {}
                
                # ACTUALLY ADD CLARIFICATIONS TO THE TEST CASE
                if "question" in test_case and test_case["question"]:
                    # For multi-turn: process all turns, for single-turn: just first turn
                    turns_to_process = test_case["question"] if len(test_case["question"]) > 1 else [test_case["question"][0]]
                    
                    for turn in turns_to_process:
                        for message in turn:
                            if message.get("role") == "user":
                                # Generate clarifications for this specific user message
                                user_query = message.get("content", message.get("transcript", ""))
                                if user_query:
                                    try:
                                        clarifications = self.generate_clarifications(user_query, function_docs)
                                        message["clarifications"] = clarifications
                                    except Exception as e:
                                        print(f"Error generating clarifications for message: {e}")
                                        message["clarifications"] = {}
                
                processed_count += 1
                    
            except Exception as e:
                print(f"Error processing test case {test_case.get('id', 'unknown')}: {e}")
        
        # Save the modified data back to the file
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✅ Successfully processed {file_path}")
        print(f"  - Total test cases: {len(test_cases)}")
        print(f"  - Successfully processed: {processed_count}")
        print(f"  - Failed: {len(test_cases) - processed_count}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Extract clarifications from voice assistant test cases")
    parser.add_argument("file_path", help="Path to the JSON file to process")
    parser.add_argument("--output", "-o", help="Output file path (defaults to input file)")
    parser.add_argument("--test", "-t", action="store_true", help="Test mode: process only 1 random test case")
    
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