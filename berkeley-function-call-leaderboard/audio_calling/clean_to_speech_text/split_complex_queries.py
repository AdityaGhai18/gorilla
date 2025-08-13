#!/usr/bin/env python3
"""
Use:
    python split_complex_queries.py --input data.json --output output.json
    python split_complex_queries.py --input BFCL_v3_live_simple.json --output split_live_simple.json
"""

import json
import argparse
import os
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-4o"

class ComplexQuerySplitter:
    """Main class for splitting complex queries into multiple conversational turns."""
    
    def __init__(self):
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
    
    def is_complex_query(self, query: str, function_docs: List[Dict[str, Any]], test_case: Dict[str, Any] = None) -> bool:
        """Determine if a query is complex enough to warrant splitting into multiple turns."""
        
        # Use LLM to determine if splitting makes conversational sense
        prompt = f"""
Analyze this user query and determine if it should be split into multiple conversational turns.

**QUERY TO ANALYZE:**
"{query}"

**FUNCTION DOCUMENTATION:**
{self._format_function_docs(function_docs)}

**EXISTING CLARIFICATIONS (if any):**
{self._get_existing_clarifications_text(test_case)}

**DEEP ANALYSIS REQUIRED:**

**1. FUNCTION REQUIREMENT ANALYSIS:**
- Check each required parameter in the function documentation
- Identify which parameters are missing from the user query
- Consider if missing parameters would naturally be asked for in conversation

**2. SPEECH AMBIGUITY DETECTION:**
- Look for names, IDs, or terms that might be unclear when spoken
- Check for ambiguous references that need clarification
- Identify potential misheard or unclear information

**3. CONTEXT PRESERVATION ASSESSMENT:**
- Determine what information is obvious and should NOT be removed
- Identify what can be reasonably asked for without breaking conversation flow
- Assess if splitting would feel natural or forced

**RULES FOR SPLITTING:**

**✅ SPLIT WHEN:**
- **Multiple Required Parameters**: Function requires 2+ parameters and only 1 is provided
- **Speech Ambiguity**: Names, IDs, or terms that are unclear when spoken (like complex serial numbers)
- **Complex Requirements**: Query has 3+ specific requirements that could be clarified step-by-step
- **Missing Critical Info**: Function cannot execute without additional information

**❌ DO NOT SPLIT WHEN:**
- Query is simple and complete (like "What's the weather in Toronto?")
- Function has only 1 required parameter and it's provided
- Breaking it down would feel forced or artificial
- All necessary information is already clear and complete
- Query is straightforward with no ambiguity
- Function can execute successfully with current information
- **Single parameter functions** where the parameter is clearly provided or can be inferred
- **Simple requests** that don't have multiple complex requirements

**DETAILED EXAMPLES:**

**Example 1: Weather Request (GOOD SPLIT)**
Query: "What's the weather in Toronto in Celsius?"
Analysis: Location is clear (Toronto), but temperature unit is specified
Split: Yes - asking for unit preference is natural
Reason: Unit clarification is common and doesn't remove obvious location info

**Example 2: Ride Request (GOOD SPLIT)**
Query: "I need a ride to the airport"
Analysis: Destination clear (airport), but missing ride type and pickup location
Split: Yes - these are naturally asked for in ride services
Reason: Missing critical information that would be asked for anyway

**Example 3: Name Ambiguity (GOOD SPLIT)**
Query: "Can you help me with Shishir Patel's account?"
Analysis: Name might be unclear when spoken, could be misheard
Split: Yes - asking for spelling clarification is natural
Reason: Names are commonly clarified in speech interactions

**Example 4: Complex Email Request (GOOD SPLIT)**
Query: "Draft an email to Andy at andy@gorilla.ai with subject 'Sales Forecast Request' and message 'where is the latest sales forecast spreadsheet?'"
Analysis: Multiple pieces of information (recipient, subject, message) that could be asked for naturally
Split: Yes - breaking down email components feels natural
Reason: Email composition is commonly done step-by-step

**Example 5: Complex Sensor Query (GOOD SPLIT)**
Query: "Get today's alerts for sensor Q3CC-CRT3-SZ2G, showing max 10 alerts per page"
Analysis: Complex serial number and pagination details that could be clarified
Split: Yes - asking for sensor ID and pagination separately feels natural
Reason: Technical details are commonly clarified in conversation

**Example 6: Complex Housekeeper Request (GOOD SPLIT)**
Query: "Help find a housekeeper who provides ironing services in Chonburi Province, with review score 4.5+ stars, available 12/03/2024 16:00-18:00, no late history"
Analysis: Multiple specific requirements that could be asked for step-by-step
Split: Yes - breaking down requirements feels natural
Reason: Complex requests are commonly clarified in conversation

**Example 7: Complete Request (NO SPLIT)**
Query: "What's the weather in Toronto?"
Analysis: Location and request are both clear
Split: No - complete request, no missing information
Reason: Splitting would create artificial turns

**Example 8: Simple Service Request (NO SPLIT)**
Query: "Help me find service provider who provide cleaning service"
Analysis: Function requires 'service_id', user clearly wants cleaning service (which maps to service_id=1)
Split: No - function can execute with inferred service_id=1 for cleaning
Reason: All necessary information is provided, splitting would create artificial turns

**Example 9: Single Parameter Request (NO SPLIT)**
Query: "Help me find service provider who provide cleaning service"
Analysis: Function requires only 'service_id', user specifies "cleaning service" which maps to service_id=1
Split: No - function has all required information, no need for clarification
Reason: Single required parameter is provided, splitting creates unnecessary conversation turns

**Example 9: Obvious Context (NO SPLIT)**
Query: "I need a pizza from Pizza Palace"
Analysis: Restaurant and food type are both specified
Split: No - complete request, removing either would feel unnatural
Reason: Both pieces of info are obvious and shouldn't be asked for separately

**RESPONSE FORMAT:**
Return only "SPLIT" or "NO_SPLIT" followed by a detailed explanation.

Example: "SPLIT - Missing ride type (UberX/Comfort) and pickup location, both would naturally be asked for in ride booking"
Example: "NO_SPLIT - Complete request with clear location and food type, no missing information that would be naturally asked for"
"""
        
        try:
            print(f"    - Sending prompt to LLM for complexity check...")
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=100
            )
            
            content = response.choices[0].message.content.strip()
            print(f"    - LLM response: {content}")
            
            # Check if response contains "SPLIT" (handle quotes and extra text)
            if "SPLIT" in content:
                print(f"    - Decision: SPLIT")
                return True
            else:
                print(f"    - Decision: NO_SPLIT")
                return False
                
        except Exception as e:
            print(f"    - Error determining if query should be split: {e}")
            # Fallback: don't split if we can't determine
            return False
    
    def _get_existing_clarifications_text(self, test_case: Dict[str, Any]) -> str:
        """Get existing clarifications as formatted text."""
        if not test_case or "question" not in test_case or not test_case["question"]:
            return "None"
        
        if isinstance(test_case["question"], list) and len(test_case["question"]) > 0:
            first_turn = test_case["question"][0]
            if isinstance(first_turn, list) and len(first_turn) > 0:
                clarifications = first_turn[0].get("clarifications", {})
                if clarifications:
                    return json.dumps(clarifications, indent=2)
        
        return "None"
    
    def _get_transcript_text(self, test_case: Dict[str, Any]) -> str:
        """Get transcript text from test case."""
        if not test_case or "question" not in test_case or not test_case["question"]:
            return "None"
        
        if isinstance(test_case["question"], list) and len(test_case["question"]) > 0:
            first_turn = test_case["question"][0]
            if isinstance(first_turn, list) and len(first_turn) > 0:
                transcript = first_turn[0].get("transcript", "")
                if transcript:
                    return transcript
        
        return "None"
    
    def split_into_conversation_turns(self, query: str, function_docs: List[Dict[str, Any]], test_case: Dict[str, Any] = None) -> List[List[Dict[str, Any]]]:
        """Create new multi-turn conversations that simulate clarification process."""
        
        # Use LLM to create natural conversation flow
        prompt = f"""
Create a natural multi-turn conversation from this single user query. The goal is to simulate how a real conversation would flow when the system needs to clarify information.

**ORIGINAL USER QUERY:**
"{query}"

**ORIGINAL TRANSCRIPT (what was spoken):**
{self._get_transcript_text(test_case)}

**FUNCTION DOCUMENTATION:**
{self._format_function_docs(function_docs)}

**EXISTING CLARIFICATIONS (if any):**
{self._get_existing_clarifications_text(test_case)}

**TWO TYPES OF CONVERSATION SPLITS:**

**TYPE 1: INFORMATION REMOVAL + ADDITION**
- **Remove some information** from Turn 1 to create natural conversation flow
- **Ask for it back** in Turn 2 (system question)
- **User provides it** in Turn 3
- **Goal**: Break down complex requests into natural steps

**TYPE 2: SPEECH AMBIGUITY CLARIFICATION**
- **Keep the information** but it's unclear when spoken
- **Ask for clarification** in Turn 2 (exact spelling, format, etc.)
- **User clarifies** in Turn 3
- **Goal**: Handle speech-to-text issues and unclear references

**CONVERSATION STRATEGY:**

**1. TURN 1 (User Request):**
- Keep the main intent and obvious context
- Remove specific details that could be asked for naturally
- **PRESERVE SPEECH DISFLUENCIES**: Keep "uh", "um", "you know", etc. from transcript
- Make it sound like a natural first request with realistic speech patterns

**2. TURN 2 (System Question):**
- Ask for the removed information OR clarify ambiguous information
- Group related questions together when natural
- Sound conversational, not robotic
- Use natural language that matches the user's speech style

**3. TURN 3 (User Response):**
- User provides the missing information OR clarifies the ambiguous information
- **MAINTAIN SPEECH REALISM**: Include natural disfluencies like "um", "uh", "like"
- Should feel like a natural continuation of the conversation
- Don't make it sound too polished or written

**DETAILED CONVERSATION PATTERNS:**

**TYPE 1: INFORMATION REMOVAL + ADDITION**

**PATTERN 1: Complex Email Request**
**Original**: "Draft an email to Andy at andy@gorilla.ai with subject 'Sales Forecast Request' and message 'where is the latest sales forecast spreadsheet?'"
**Strategy**: Remove recipient, subject, and message details
- Turn 1: "I'd like to draft an email"
- Turn 2: "Who should I send it to, what's the subject, and what message would you like to include?"
- Turn 3: "To Andy at andy@gorilla.ai, subject 'Sales Forecast Request', and message 'where is the latest sales forecast spreadsheet?'"
**Why This Works**: Breaks down complex email into natural conversation steps

**PATTERN 1A: Preserving Speech Disfluencies**
**Original Transcript**: "Uh, can you help me find a cleaning service provider?"
**Strategy**: Keep the "Uh" and natural speech patterns
- Turn 1: "Uh, can you help me find a cleaning service provider?"
- Turn 2: "Sure, what type of cleaning service do you need?"
- Turn 3: "Um, I need regular home cleaning"
**Why This Works**: Maintains realistic speech patterns, doesn't sound too polished

**PATTERN 1B: Speech Realism Examples**
**❌ TOO POLISHED (BAD)**: "I would like to request assistance in locating a cleaning service provider"
**✅ REALISTIC SPEECH (GOOD)**: "Uh, can you help me find a cleaning service provider?"

**❌ TOO POLISHED (BAD)**: "I require a regular home cleaning service"
**✅ REALISTIC SPEECH (GOOD)**: "Um, I need regular home cleaning"

**PATTERN 2: Complex Sensor Query**
**Original**: "Get today's alerts for sensor Q3CC-CRT3-SZ2G, showing max 10 alerts per page"
**Strategy**: Remove specific sensor ID and pagination details
- Turn 1: "I need to get today's sensor alerts"
- Turn 2: "Which sensor would you like alerts for, and how many alerts per page?"
- Turn 3: "Sensor Q3CC-CRT3-SZ2G, and show up to 10 alerts per page"
**Why This Works**: Breaks down technical details into natural questions

**PATTERN 3: Complex Housekeeper Request**
**Original**: "Help find a housekeeper who provides ironing services in Chonburi Province, with review score 4.5+ stars, available 12/03/2024 16:00-18:00, no late history"
**Strategy**: Remove specific requirements to create conversation
- Turn 1: "I need help finding a housekeeper in Chonburi Province"
- Turn 2: "What services do you need, what rating, when are you available, and any other requirements?"
- Turn 3: "Ironing services, 4.5+ stars, available Dec 3rd 4-6 PM, and no history of being late"
**Why This Works**: Breaks down multiple requirements into natural conversation

**TYPE 2: SPEECH AMBIGUITY CLARIFICATION**

**PATTERN 4: Name/ID Ambiguity**
**Original**: "Can you help me with Shishir Patel's account?"
**Strategy**: Keep the name but ask for exact spelling
- Turn 1: "Can you help me with Shishir Patel's account?"
- Turn 2: "Could you please help me spell that name? Is it S-H-I-S-H-I-R P-A-T-E-L, or was there a different spelling?"
- Turn 3: "Yes, that's correct: S-H-I-S-H-I-R P-A-T-E-L"
**Why This Works**: Handles speech ambiguity, provides spelling suggestions

**PATTERN 5: Complex Serial Number**
**Original**: "Get alerts for sensor Q3CC-CRT3-SZ2G"
**Strategy**: Keep the request but ask for exact serial number
- Turn 1: "I need to get sensor alerts"
- Turn 2: "What's the exact serial number? Is it Q-3-C-C dash C-R-T-3 dash S-Z-2-G?"
- Turn 3: "Yes, that's correct: Q3CC-CRT3-SZ2G"
**Why This Works**: Clarifies complex technical identifiers that might be misheard

**CRITICAL RULES - NEVER VIOLATE:**

1. **ONLY USE INFORMATION FROM TRANSCRIPT**: Never introduce new information that wasn't in the original transcript
2. **NO MADE-UP DETAILS**: Don't add locations, names, or details that don't exist in the transcript
3. **STRICT TRANSCRIPT ADHERENCE**: Every piece of information in the conversation must come from the transcript
4. **NO CREATIVE ADDITIONS**: Don't embellish or expand beyond what was actually said
5. **PRESERVE SPEECH DISFLUENCIES**: Keep natural speech features like "uh", "um", "you know", "like", etc. from the transcript
6. **MAINTAIN SPEECH REALISM**: Don't make the conversation sound too polished or written - keep it conversational

**USE TRANSCRIPT, FUNCTION DOCS, AND CLARIFICATIONS TO GUIDE YOUR DECISION:**

1. **Transcript Analysis**: Look at what was actually spoken vs. written - are there speech-to-text issues?
2. **Function Requirements**: What parameters does the function need? Use clarifications to see what's already provided.
3. **Natural Flow**: Would breaking this down feel like a real conversation or forced?

**DECISION PROCESS:**
- **If complex with multiple details** → Use TYPE 1 (remove info, ask for it back)
- **If speech ambiguity exists** → Use TYPE 2 (keep info, ask for clarification)
- **If simple and clear** → Don't split (would feel artificial)

**RESPONSE FORMAT:**
Return a JSON array with this exact structure:
[
    [
        {{
            "role": "user",
            "content": "First user turn (simplified but complete)"
        }}
    ],
    [
        {{
            "role": "assistant", 
            "content": "System asking for missing/ambiguous information"
        }}
    ],
    [
        {{
            "role": "user",
            "content": "User providing the missing information"
        }}
    ]
]

**IMPORTANT:**
- Only create turns if it makes conversational sense
- Use the transcript to identify speech ambiguity
- Use function docs to understand what information is needed
- Use existing clarifications to guide what might be missing
- Make the conversation feel natural, not robotic
- **PRESERVE SPEECH DISFLUENCIES**: Keep "uh", "um", "you know", "like" from transcript
- **MAINTAIN SPEECH REALISM**: Don't make it sound too polished or written

**RESPONSE FORMAT:**
Return a JSON array with this exact structure:
[
    [
        {{
            "role": "user",
            "content": "First user turn (simplified but complete)"
        }}
    ],
    [
        {{
            "role": "assistant", 
            "content": "System asking for missing/ambiguous information"
        }}
    ],
    [
        {{
            "role": "user",
            "content": "User providing the missing information"
        }}
    ]
]

**SPECIFIC SPEECH AMBIGUITY HANDLING:**

**1. NAMES AND IDENTIFIERS:**
- **Difficult Names**: "Could you spell that? Is it S-H-I-S-H-I-R P-A-T-E-L?"
- **User IDs**: "What's your exact username? Is it 'john_doe' or 'johndoe'?"
- **Bot IDs**: "What's the exact bot identifier? Is it 'my-bot-123' or 'mybot123'?"

**2. NUMBERS AND FORMATS:**
- **Time**: "What time exactly? 3 PM, 3:00 PM, or 15:00?"
- **Dates**: "Which date? Tomorrow, March 15th, or 03/15?"
- **Quantities**: "How many? 5, five, or 5.0?"

**3. TECHNICAL TERMS:**
- **API Versions**: "Which version? v1, v2, or version 2?"
- **File Formats**: "What format? JSON, CSV, or text?"
- **Units**: "What unit? Celsius, Fahrenheit, or Kelvin?"

**4. LOCATION AND ADDRESSES:**
- **Street Names**: "What's the exact street name? Is it 'Main Street' or 'Main St'?"
- **City Names**: "Which city? Is it 'San Francisco' or 'SF'?"
- **Postal Codes**: "What's the zip code? 94102 or 94102-1234?"

**IMPORTANT GUIDELINES:**
- Only create turns if information is genuinely missing or ambiguous
- Don't remove obvious information just to create more turns
- Make the system questions sound natural and conversational
- Group related clarifications when possible
- Use existing clarifications to guide what's missing
- **SPEECH AMBIGUITY**: If names, IDs, or specific terms might be unclear in speech, ask for clarification
- **NATURAL FLOW**: The conversation should feel like what a real person would naturally ask for
- **CONTEXT PRESERVATION**: Always preserve the main intent and obvious context
"""
        
        try:
            print(f"    - Sending prompt to LLM for conversation creation...")
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=800
            )
            
            content = response.choices[0].message.content.strip()
            print(f"    - LLM response length: {len(content)} characters")
            print(f"    - LLM response preview: {content[:200]}...")
            
            # Parse JSON response
            try:
                # Remove markdown code blocks if present
                if content.startswith("```json"):
                    content = content[7:]
                if content.endswith("```"):
                    content = content[:-3]
                
                result = json.loads(content.strip())
                if isinstance(result, list) and len(result) >= 2:
                    print(f"    - Successfully parsed {len(result)} conversation turns")
                    return result
                else:
                    print(f"    - Warning: Invalid response format, using fallback")
                    return self._create_fallback_conversation(query, function_docs, test_case)
                    
            except json.JSONDecodeError as e:
                print(f"    - Warning: Invalid JSON response, using fallback: {content[:100]}...")
                return self._create_fallback_conversation(query, function_docs, test_case)
            
        except Exception as e:
            print(f"    - Error creating conversation turns: {e}")
            return self._create_fallback_conversation(query, function_docs, test_case)
    
    def _create_fallback_conversation(self, query: str, function_docs: List[Dict[str, Any]], test_case: Dict[str, Any] = None) -> List[List[Dict[str, Any]]]:
        """Create a simple fallback conversation when LLM fails."""
        # Just return the original query as a single turn
        return [
            [{"role": "user", "content": query}]
        ]
    
    def _format_function_docs(self, function_docs: List[Dict[str, Any]]) -> str:
        """Format function documentation for the prompt."""
        formatted_docs = ""
        
        for i, func in enumerate(function_docs):
            formatted_docs += f"Function {i+1}:\n"
            formatted_docs += f"Name: {func.get('name', 'Unknown')}\n"
            formatted_docs += f"Description: {func.get('description', 'No description')}\n"
            
            if "parameters" in func:
                params = func["parameters"]
                if "properties" in params:
                    formatted_docs += "Parameters:\n"
                    for param_name, param_info in params["properties"].items():
                        param_type = param_info.get("type", "unknown")
                        param_desc = param_info.get("description", "No description")
                        required = " (required)" if param_name in params.get("required", []) else " (optional)"
                        formatted_docs += f"  - {param_name} ({param_type}){required}: {param_desc}\n"
                
                if "required" in params:
                    formatted_docs += f"Required parameters: {', '.join(params['required'])}\n"
            
            formatted_docs += "\n"
        
        return formatted_docs
    
    def process_test_case(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single test case and create multi-turn conversations if complex."""
        
        # Check if it's a single-turn test case
        if not self._is_single_turn(test_case):
            return test_case
        
        # Extract user query and function docs
        user_query = self._extract_user_query(test_case)
        function_docs = self._get_function_docs(test_case)
        
        if not user_query or not function_docs:
            return test_case
        
        # Check if query is complex enough to split
        print(f"  - Checking if query is complex enough to split...")
        print(f"  - Query: {user_query[:100]}...")
        print(f"  - Function docs: {len(function_docs)} functions")
        
        if not self.is_complex_query(user_query, function_docs, test_case):
            print(f"  - Query deemed too simple, keeping as single turn")
            return test_case
        
        print(f"  - Query deemed complex enough, proceeding to split...")
        
        # Create multi-turn conversations
        try:
            print(f"  - Creating multi-turn conversation for: {user_query[:100]}...")
            conversation_turns = self.split_into_conversation_turns(user_query, function_docs, test_case)
            
            if conversation_turns and len(conversation_turns) > 1:
                # Keep the original test case intact, but add the new multi-turn conversations
                test_case["multi_turn_conversations"] = conversation_turns
                test_case["split_from_single_turn"] = True
                print(f"  - Created {len(conversation_turns)} turns")
            else:
                print(f"  - No clarifications needed, keeping as single turn")
            
        except Exception as e:
            print(f"Error processing test case {test_case.get('id', 'unknown')}: {e}")
            # If splitting fails, keep the original test case as-is
            return test_case
        
        return test_case
    
    def _is_single_turn(self, test_case: Dict[str, Any]) -> bool:
        """Check if test case is a single-turn test case."""
        if "question" not in test_case or not test_case["question"]:
            return False
        
        # Single turn structure: question is a list with one turn
        if isinstance(test_case["question"], list) and len(test_case["question"]) == 1:
            first_turn = test_case["question"][0]
            if isinstance(first_turn, list) and len(first_turn) == 1:
                first_message = first_turn[0]
                if isinstance(first_message, dict) and first_message.get("role") == "user":
                    return True
        
        return False
    
    def _extract_user_query(self, test_case: Dict[str, Any]) -> Optional[str]:
        """Extract user query from test case."""
        if "question" in test_case and test_case["question"]:
            if isinstance(test_case["question"], list) and len(test_case["question"]) > 0:
                first_turn = test_case["question"][0]
                if isinstance(first_turn, list) and len(first_turn) > 0:
                    first_message = first_turn[0]
                    if isinstance(first_message, dict) and first_message.get("role") == "user":
                        return first_message.get("content", "")
        
        return None
    
    def _get_function_docs(self, test_case: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get function documentation from test case."""
        if "function" in test_case:
            return test_case["function"]
        return []

def main():
    """Main function to process test cases and create multi-turn conversations."""
    parser = argparse.ArgumentParser(description="Split complex single-turn queries into multi-turn conversations")
    parser.add_argument("--input", required=True, help="Input JSON file path")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of test cases to process")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Find the input file
    input_file = args.input
    if not os.path.exists(input_file):
        # Check in final_results directory
        final_results_path = os.path.join("final_results", input_file)
        if os.path.exists(final_results_path):
            input_file = final_results_path
            print(f"Found file in final_results directory: {input_file}")
        else:
            print(f"Error: Input file {args.input} not found")
            return
    
    # Load the data
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading input file: {e}")
        return
    
    if not isinstance(data, list):
        print("Error: Input file should contain a list of test cases")
        return
    
    # Randomly sample test cases if limit is specified
    if args.limit:
        if args.limit > len(data):
            print(f"Warning: Requested limit ({args.limit}) is greater than available test cases ({len(data)}). Using all available.")
            selected_cases = data
        else:
            # Randomly sample without replacement
            import random
            import time
            random.seed(int(time.time()))  # Use current timestamp for different results each run
            selected_cases = random.sample(data, args.limit)
            print(f"Randomly selected {len(selected_cases)} test cases from {len(data)} available")
    else:
        selected_cases = data
    
    # Initialize the splitter
    splitter = ComplexQuerySplitter()
    
    # Process test cases
    processed_cases = []
    complex_count = 0
    
    for i, test_case in enumerate(selected_cases):
        if args.verbose:
            print(f"Processing test case {i+1}/{len(selected_cases)}: {test_case.get('id', 'unknown')}")
        
        processed_case = splitter.process_test_case(test_case)
        processed_cases.append(processed_case)
        
        if processed_case.get("split_from_single_turn", False):
            complex_count += 1
    
    # Save the results
    try:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(processed_cases, f, indent=2, ensure_ascii=False)
        
        print(f"\nProcessing complete!")
        print(f"Total test cases processed: {len(processed_cases)}")
        print(f"Complex queries split into multi-turn: {complex_count}")
        print(f"Output saved to: {args.output}")
        
    except Exception as e:
        print(f"Error saving output file: {e}")
        return

if __name__ == "__main__":
    main()
