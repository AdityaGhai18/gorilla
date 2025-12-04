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
import time
import re
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dotenv import load_dotenv
import openai
from huanzhi_utils import *
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import sys
from tqdm import tqdm


# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-5-2025-08-07"
MAX_RETRIES = 3
RETRY_DELAY = 1.0
DEFAULT_MAX_WORKERS = 10  # Number of concurrent threads for processing


class ClarificationExtractor:
    """Main class for extracting clarifications from voice assistant queries."""

    def __init__(self):
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)

    def get_function_docs_single_turn(
        self, test_case: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract function documentation from single-turn test case."""
        if "tools" in test_case:
            return test_case["tools"]
        return []

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
        potential_names = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
        for name in potential_names[:3]:  # Limit to first 3
            entities.append(f"PERSON: {name}")

        # Numbers
        numbers = re.findall(r"\b\d+(?:\.\d+)?\b", text)
        for num in numbers[:5]:  # Limit to first 5
            entities.append(f"NUMBER: {num}")

        # Dates/times
        date_patterns = [
            r"\b(?:today|tomorrow|yesterday|next week|last week)\b",
            r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b",
            r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        ]
        for pattern in date_patterns:
            matches = re.findall(pattern, text.lower())
            for match in matches[:3]:
                entities.append(f"DATE: {match}")

        result = "; ".join(entities) if entities else "No entities found"
        return result

    def generate_clarifications(
        self, user_query: str, function_docs: List[Dict[str, Any]]
    ) -> Dict[str, str]:
        """Generate clarifications using GPT-4o."""

        # Format function documentation for the prompt
        function_docs_text = function_docs

        prompt = f"""You are an AI system that identifies clarification points for voice assistant queries.
The goal is to help an AI that will call a function ask valid clarifying questions.

Why Clarifications Are Needed
Voice-based user interactions often contain ambiguities that differ from text-based queries:
1. Names that could be spelled differently (e.g., "Zhang Wei", "Nguyen Van Minh", "Shishir")
2. Numbers that sound similar when spoken (e.g., "13" vs "30", "14" vs "40", "15" vs "50")
3. Dates and times that are critical for function calls (e.g., "March 15th", "3pm", "Tuesday")
4. Technical terms, filepaths, URLs that could be ambiguous when spoken (e.g., "XContentBuilder" vs "X content builder")
5. Complex alphanumeric strings, codes, or identifiers (e.g., "MIIFdTCCBF2gAwIBAgISESG", "d0404", "ABC123XYZ")
6. Technical product names, database names, or brand names (e.g., "Elasticsearch", "Firebird", "MongoDB")
7. Code snippets, SQL queries, or commands with symbols (e.g., "SELECT * FROM table", "x = y + 1")
8. Technical constants, class names, or identifiers (e.g., "VMDeathRequest", "onScriptError.IGNORE")
9. Street addresses or specific locations that are critical for function calls
10. Unique identifiers or IDs that are essential for function execution
11. Homophones that could be confused (e.g., "write" vs "right", "to" vs "too" vs "two")

IMPORTANT: Be REASONABLE. Flag items that have genuine spelling ambiguity OR are specific identifiers/names that could be unclear when spoken and are essential for the correctness of the function call.

DO flag:
- ALL proper nouns including common names (e.g., "John Smith", "Mary Johnson", "Zhang Wei", "Nguyen Van Minh")
- ALL place names including common cities (e.g., "New York", "London", "Paris", "Tokyo", "Sydney")
- ALL numbers that are essential to function calls (e.g., "13", "30", "100", "2024", "3pm", "9am") - numbers can sound similar when spoken
- MOST dates and times that are essential to function calls (e.g., "Tuesday", "March 15th", "2024-03-15", "next week", "tomorrow")
- Complex technical terms (e.g., "mappingParserContext", "compositeScriptFactory", "VMDeathRequest")
- Complex alphanumeric strings or codes (e.g., "MIIFdTCCBF2gAwIBAgISESG", "d0404", "ABC123XYZ")
- Technical product names (e.g., "Elasticsearch", "Firebird", "MongoDB")
- Street addresses or specific locations (e.g., "123 Main Street", "456 Oak Avenue")
- Unique identifiers or IDs (e.g., "d0404", "user_123", "config_456")
- Complex numbers with suffixes (e.g., "12345L", "54321L", "0xFF")
- Technical terms, class names, and identifiers (e.g., "mappingParserContext", "compositeScriptFactory", "VMDeathRequest", "PrintStream", "logStream")
- Code snippets, SQL queries, or commands with symbols (e.g., "SELECT * FROM table", "x = y + 1")
- Technical constants, class names, or identifiers (e.g., "VMDeathRequest", "onScriptError.IGNORE")
- Specific event names, task names, or content that need exact wording (e.g., "go for shopping at 9 pm", "budget analysis meeting")
- Technical identifiers or keys (e.g., "site.info", "apikey_info", "config_key")
- Quoted content that needs to be exact (e.g., "use the prompt 'qfqwfqf'")
- Essential function arguments that could be ambiguous when spoken


Do NOT flag:
- Common English words (e.g., "water", "iron", "burglary", "English", "marketing team", "tenant rights")
- Simple technical terms (e.g., "user", "config", "file", "data")
- Common verbs, prepositions, or articles (e.g., "could", "should", "the", "a", "an")
- Simple concepts or generic terms (e.g., "new iPhone release", "Queue is saturated")
- Simple phrases or expressions (e.g., "Happy Birthday!", "Queue is unsaturated")
- Obvious non-identifiers (e.g., "/q", "quit", "exit", "help")

Your Task
1. Inputs:
   - User query (text input) - focus on the actual content
   - Documentation for the exact function (its arguments and descriptions)
2. Actions:
   - Identify ALL potentially ambiguous class names, method names, technical terms, and identifiers in the USER QUERY ONLY.
   - Focus on SUPER IMPORTANT and AMBIGUOUS details that need confirmation for the function call.
   - Flag any critical information that could be unclear when spoken, regardless of which function parameter it maps to.
   - Map these to appropriate function parameters when possible, but don't limit yourself to only function parameters.
   - Focus on the content of what was spoken - any technical term that could be unclear when spoken should be flagged.
   - Be reasonable - flag things that could actually be unclear when spoken, even if they're just context or mentioned in passing.
   - CRITICAL: ONLY extract clarifications from the USER QUERY text. DO NOT extract examples, sample values, or text from the function documentation.
   - CRITICAL: If the function documentation contains examples in other languages (like Korean, Chinese, etc.), DO NOT include those as clarifications unless they actually appear in the user query.
   - CRITICAL: The function documentation is for reference only - all clarifications must come from the actual user query content.
3. Output:
   - One JSON dictionary (no lists).
   - Keys: Use DESCRIPTIVE, NATURAL LANGUAGE names based on the parameter descriptions:
     - Look at the parameter description to create meaningful names
     - Use your understanding of the context to create appropriate, descriptive names
     - Don't rely on rigid suffix rules - be creative and descriptive
     - Examples: "start_date", "user_name", "file_path", "database_name", "street_address", "user_id", "validity_seconds", "encoding_method", "task_description", "event_name"
     - NOT: "r_spelling", "e_spelling", "param1_spelling", "content_spelling" (avoid lazy "_spelling" suffix for everything)
     - For multiple items: "first_name", "second_name", "primary_address", "secondary_address"
   - Values:
     - The value from the user query that needs clarification.

---

Example: Foreign name clarification (VALID)
User Query: "Schedule a meeting with Zhang Wei on Tuesday at 3pm"
Function Documentation:
schedule_meeting(person_name: string, date: string, time: string)
Output:
{{
  "person_name": "Zhang Wei"
}}

Example: Technical term clarification (VALID)
User Query: "Create an XContentBuilder object for the data"
Function Documentation:
create_builder(builder_type: string, data: object)
Output:
{{
  "builder_type": "XContentBuilder"
}}

Example: Event details clarification (VALID)
User Query: "Who was the U.S. president during the Civil War?"
Function Documentation:
US_President_During_Event(event: string, country: string)
Output:
{{
  "event_details": "Civil War"
}}

Example: Common place name (VALID)
User Query: "Book a flight to New York on Friday"
Function Documentation:
book_flight(destination: string, date: string)
Output:
{{
  "destination": "New York"
}}

Example: Major city name (VALID)
User Query: "Get me a house to stay for 4 in London"
Function Documentation:
search_house(location: string, guests: integer)
Output:
{{
  "location": "London"
}}

Example: Important ambiguous details (VALID)
User Query: "I'm planning ahead for our big trip and realized our car's fuel tank is running low. It would be great if you could top it up with an additional 30 gallons before I turn on the ignition using the 'START' mode."
Function Documentation:
fillFuelTank(fuelAmount: float), startEngine(ignitionMode: string)
Output:
{{
  "fuel_amount": "30 gallons",
  "ignition_mode": "START"
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
  "constant_name": "CERTIFICATE",
  "constant_value": "MIIFdTCCBF2gAwIBAgISESG"
}}

Example: Multiple technical terms (VALID)
User Query: "Set up Elasticsearch with mappingParserContext and compositeScriptFactory, handle errors with onScriptError.IGNORE"
Function Documentation:
setup_elasticsearch(parser_context: string, script_factory: string, error_handling: string)
Output:
{{
  "database_name": "Elasticsearch", 
  "parser_context": "mappingParserContext",
  "script_factory": "compositeScriptFactory",
  "error_handling": "onScriptError.IGNORE"
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
  "request_type": "VMDeathRequest"
}}

Example: Technical class and complex numbers (VALID)
User Query: "Use PrintStream logStream with values 12345L and 54321L"
Function Documentation:
use_stream(stream_type: string, stream_name: string, old_value: string, new_value: string)
Output:
{{
  "stream_type": "PrintStream",
  "stream_name": "logStream",
  "old_value": "12345L",
  "new_value": "54321L"
}}

Example: Database name in context (VALID)
User Query: "Check Elasticsearch cluster status"
Function Documentation:
check_status(database: string, component: string)
Output:
{{
  "database_name": "Elasticsearch"
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

Example: Common names (VALID)
User Query: "Schedule a meeting with John Smith and Mary Johnson tomorrow"
Function Documentation:
schedule_meeting(participant_names: list, date: string)
Output:
{{
  "participant_name1": "John Smith",
  "participant_name2": "Mary Johnson"
}}

Example: Simple concept (NO CLARIFICATION NEEDED)
User Query: "Write about the new iPhone release in English"
Function Documentation:
write_article(topic: string, language: string)
Output:
{{}}

Example: Street address clarification (VALID)
User Query: "Send package to 123 Main Street, New York"
Function Documentation:
send_package(address: string, city: string)
Output:
{{
  "street_address": "123 Main Street",
  "city": "New York"
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

Example: Number clarification (VALID)
User Query: "Book 13 seats for the conference"
Function Documentation:
book_seats(quantity: integer, event: string)
Output:
{{
  "quantity": "13"
}}

Example: Time clarification (VALID)
User Query: "Schedule the meeting for 3pm tomorrow"
Function Documentation:
schedule_meeting(time: string, date: string)
Output:
{{
  "meeting_time": "3pm, tomorrow"
}}

Example: Date clarification (VALID)
User Query: "Set reminder for March 15th at 9am"
Function Documentation:
set_reminder(date: string, time: string)
Output:
{{
  "reminder_date": "March 15th",
  "reminder_time": "9am"
}}

Example: Similar-sounding numbers (VALID)
User Query: "Transfer 30 dollars to account 12345"
Function Documentation:
transfer_money(amount: float, account: string)
Output:
{{
  "amount": "30",
  "account_number": "12345"
}}

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
  "task_description": "go for shopping at 9 pm"
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
  "start_date": "2023-04-01",
  "end_date": "2023-04-15"
}}

Example: Multiple technical terms in context (VALID)
User Query: "I'm trying to fix a compilation error. The class 'StringNumberHandler' extends 'AbstractCellHandler' and overrides methods like 'getCellValue', 'setCellValue', 'getExcelType'. I'm getting an error with 'CellResult' and 'getNumericValue()'. Find relevant classes for 'CellResult'?"
Function Documentation:
get_relevant_classes(search_string: string)
Output:
{{
  "search_string": "CellResult",
  "context_class1": "StringNumberHandler",
  "context_class2": "AbstractCellHandler", 
  "context_method1": "getCellValue",
  "context_method2": "setCellValue",
  "context_method3": "getExcelType",
  "context_method4": "getNumericValue"
}}

Example: No relevant parameters (NO CLARIFICATION NEEDED)
User Query: "What's the weather like today?"
Function Documentation:
get_weather()
Output:
{{}}

Example: Function documentation examples (DO NOT EXTRACT FROM DOCS)
User Query: "Turn on the air conditioner in the living room"
Function Documentation:
control_appliance(command: string)
# command: The command must be specified as a string in Korean, consisting of the room name, appliance name (or alias), and operation command, separated by commas. Examples: '거실, 에어컨, 실행' for turning on the air conditioner in the living room...
Output:
{{}}
# Note: Even though the function documentation contains Korean examples, we do NOT extract '거실, 에어컨, 실행' because the user query is in English and doesn't contain Korean text.

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
                    # max_tokens=1000,
                    # temperature=0.0,
                    response_format={"type": "json_object"},
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
        # FIXME
        return True
        # Single-turn cases have "function" field, multi-turn have "path" field
        # return "function" in test_case and "path" not in test_case

    def process_single_test_case(
        self, test_case: Dict[str, Any], index: int
    ) -> Tuple[int, Dict[str, Any], bool, str]:
        """
        Process a single test case and return results.

        Returns:
            Tuple of (index, modified_test_case, success, error_message)
        """
        try:
            # Get function documentation based on test case type

            function_docs = self.get_function_docs_single_turn(test_case)

            if not function_docs:
                return (index, test_case, False, "No function documentation found")

            # ACTUALLY ADD CLARIFICATIONS TO THE TEST CASE
            if "messages" in test_case and test_case["messages"]:
                # For multi-turn: process all turns, for single-turn: just first turn
                assert len(test_case["messages"]) > 1
                all_messages = test_case["messages"]

                # for turn_idx, turn in enumerate(turns_to_process):
                for msg_idx, message in enumerate(all_messages):
                    if message.get("role") == "user":
                        # Generate clarifications for this specific user message
                        # FIXME
                        user_query = message.get("content")
                        if user_query:
                            try:
                                clarifications = self.generate_clarifications(
                                    user_query, function_docs
                                )
                                message["clarifications"] = clarifications
                            except Exception as e:
                                message["clarifications"] = {}
                                return (
                                    index,
                                    test_case,
                                    False,
                                    f"Error generating clarifications: {e}",
                                )

            return (index, test_case, True, None)

        except Exception as e:
            return (index, test_case, False, str(e))

    def process_single_test_case_multi_turn(
        self, test_case: Dict[str, Any], index: int
    ) -> Tuple[int, Dict[str, Any], bool, str]:
        """
        Process a single test case and return results.

        Returns:
            Tuple of (index, modified_test_case, success, error_message)
        """
        try:
            # Get function documentation based on test case type
            function_docs = self.get_function_docs_single_turn(test_case)

            if not function_docs:
                return (index, test_case, False, "No function documentation found")

            # ACTUALLY ADD CLARIFICATIONS TO THE TEST CASE
            if "messages" in test_case and test_case["messages"]:
                # For multi-turn: process all turns, for single-turn: just first turn
                assert len(test_case["messages"]) > 1
                all_messages = test_case["messages"]

                # for turn_idx, turn in enumerate(turns_to_process):
                for msg_idx, message in enumerate(all_messages):
                    if message["role"] == "user":
                        try:
                            clarifications = self.generate_clarifications(
                                message["content"], function_docs
                            )
                            message["clarifications"] = clarifications
                        except Exception as e:
                            message["clarifications"] = {}
                            return (
                                index,
                                test_case,
                                False,
                                f"Error generating clarifications: {e}",
                            )

            return (index, test_case, True, None)

        except Exception as e:
            return (index, test_case, False, str(e))

    def process_file(
        self,
        file_path: str,
        output_path: str = None,
        test_mode: bool = False,
        max_workers: int = None,
    ) -> None:
        """
        Process a JSON file with multi-threading support.

        Args:
            file_path: Path to input JSON file
            output_path: Path to output JSON file
            test_mode: If True, process only first 3 cases
            max_workers: Number of concurrent threads (default: DEFAULT_MAX_WORKERS)
        """
        if max_workers is None:
            max_workers = DEFAULT_MAX_WORKERS

        print(f"Processing file: {file_path}")
        print(f"Using {max_workers} worker threads")

        # Load the JSON file
        data = load_file(file_path)

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

        # Prepare for concurrent processing
        processed_results = [None] * len(data)  # Maintain original order
        processed_count = 0
        failed_count = 0
        print_lock = Lock()

        # Create progress bar
        progress_bar = tqdm(
            total=len(test_cases), desc="Processing test cases", unit="case"
        )

        # Process test cases in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_index = {}
            for i, test_case in enumerate(test_cases):
                future = executor.submit(self.process_single_test_case, test_case, i)
                future_to_index[future] = i

            # Collect results as they complete
            for future in as_completed(future_to_index):
                try:
                    index, modified_test_case, success, error_msg = future.result()

                    # Store the modified test case in the correct position
                    processed_results[index] = modified_test_case

                    # Update counters and progress
                    if success:
                        pass
                        # processed_count += 1
                        # with print_lock:
                        #     # Extract some info for logging (optional)
                        #     if "messages" in modified_test_case:
                        #         for msg in modified_test_case["messages"]:
                        #             if msg.get("role") == "user" and "clarifications" in msg:
                        #                 user_query = msg.get("content", msg.get("transcript", ""))
                        #                 clarifications = msg.get("clarifications", {})
                        #                 if clarifications:
                        #                     tqdm.write(f"  ✅ Test {index+1}: {len(clarifications)} clarifications")
                        #                     tqdm.write(f"     User: {user_query}")
                        #                     tqdm.write(f"     Clarifications: {clarifications}")
                        #                 break
                    else:
                        failed_count += 1
                        with print_lock:
                            tqdm.write(f"  ❌ Test {index+1}: {error_msg}")

                    progress_bar.update(1)

                except Exception as e:
                    failed_count += 1
                    with print_lock:
                        tqdm.write(
                            f"  ❌ Test {future_to_index[future]+1}: Thread execution error: {e}"
                        )
                    progress_bar.update(1)

        progress_bar.close()

        # Replace only the processed test cases in the original data
        for i, result in enumerate(processed_results[: len(test_cases)]):
            if result is not None:
                data[i] = result

        # Save the modified data back to the file
        write_list_of_dicts_to_file(output_path, data)

        print(f"\n✅ Successfully processed {file_path}")
        print(f"  - Total test cases: {len(test_cases)}")
        print(f"  - Successfully processed: {processed_count}")
        print(f"  - Failed: {failed_count}")
        print(f"  - Output saved to: {output_path}")
        print()


def main():
    # Hard-coded configuration
    INPUT_PATH = "/Users/hans/repo/API-Gen/hermes_function_calling_v1_filtered_spoken.json"
    OUTPUT_PATH = "/Users/hans/repo/API-Gen/hermes_function_calling_v1_filtered_clarification_spoken.json"
    TEST_MODE = False  # Set to True to process only first 3 cases
    MAX_WORKERS = 300  # Number of concurrent threads

    # Initialize extractor and process file
    extractor = ClarificationExtractor()

    print(f"=" * 60)
    print("Clarification Extraction with Multi-Threading")
    print(f"=" * 60)
    print(f"Input file:  {INPUT_PATH}")
    print(f"Output file: {OUTPUT_PATH}")
    print(f"Test mode:   {TEST_MODE}")
    print(f"Max workers: {MAX_WORKERS}")
    print(f"=" * 60)
    print()

    start_time = time.time()

    extractor.process_file(
        INPUT_PATH, OUTPUT_PATH, test_mode=TEST_MODE, max_workers=MAX_WORKERS
    )

    elapsed_time = time.time() - start_time
    print(f"\n⏱️  Total processing time: {elapsed_time:.2f} seconds")
    print(
        f"   Average time per case: {elapsed_time / (3 if TEST_MODE else 60000):.3f} seconds"
    )


if __name__ == "__main__":
    main()
