#!/usr/bin/env python3
"""
Use:
    python split_complex_queries.py --input data.json --output output.json
    python split_complex_queries.py --input BFCL_v3_live_simple.json --output split_live_simple.json
"""

import json
import re
import argparse
import os
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dotenv import load_dotenv
import openai

# Try to import spaCy for NER, fallback to regex-based NER
try:
    import spacy
    SPACY_AVAILABLE = True
    # Load English language model
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Warning: spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
        SPACY_AVAILABLE = False
except ImportError:
    print("Warning: spaCy not available. Using regex-based NER fallback.")
    SPACY_AVAILABLE = False

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-4o"

class ComplexQuerySplitter:
    """Main class for splitting complex queries into multiple conversational turns."""
    
    def __init__(self):
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
        
    def extract_named_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract named entities from the query using classical NER techniques."""
        
        if SPACY_AVAILABLE:
            return self._extract_entities_spacy(query)
        else:
            return self._extract_entities_regex(query)
    
    def _extract_entities_spacy(self, query: str) -> Dict[str, List[str]]:
        """Extract named entities using spaCy NER."""
        doc = nlp(query)
        
        entities = {
            "people": [],
            "places": [],
            "organizations": [],
            "dates": [],
            "numbers": [],
            "files": [],
            "other": []
        }
        
        for ent in doc.ents:
            entity_text = ent.text.strip()
            if not entity_text:
                continue
                
            if ent.label_ in ["PERSON"]:
                entities["people"].append(entity_text)
            elif ent.label_ in ["GPE", "LOC", "FAC"]:
                entities["places"].append(entity_text)
            elif ent.label_ in ["ORG"]:
                entities["organizations"].append(entity_text)
            elif ent.label_ in ["DATE", "TIME"]:
                entities["dates"].append(entity_text)
            elif ent.label_ in ["CARDINAL", "QUANTITY", "MONEY", "PERCENT"]:
                entities["numbers"].append(entity_text)
            else:
                entities["other"].append(entity_text)
        
        # Also extract file-like patterns
        file_patterns = re.findall(r'\b[\w\-_]+\.(?:txt|pdf|doc|docx|json|csv|py|js|html|css|md|xml|yaml|yml|sh|sql|log|out|err|tmp|temp|bak|backup)\b', query, re.IGNORECASE)
        entities["files"].extend(file_patterns)
        
        # Extract numbers (IDs, codes, etc.)
        number_patterns = re.findall(r'\b\d{3,}\b', query)  # 3+ digit numbers
        entities["numbers"].extend(number_patterns)
        
        # Remove duplicates
        for key in entities:
            entities[key] = list(set(entities[key]))
        
        return entities
    
    def _extract_entities_regex(self, query: str) -> Dict[str, List[str]]:
        """Extract named entities using regex patterns (fallback when spaCy is not available)."""
        
        entities = {
            "people": [],
            "places": [],
            "organizations": [],
            "dates": [],
            "numbers": [],
            "files": [],
            "other": []
        }
        
        # Extract file names
        file_patterns = re.findall(r'\b[\w\-_]+\.(?:txt|pdf|doc|docx|json|csv|py|js|html|css|md|xml|yaml|yml|sh|sql|log|out|err|tmp|temp|bak|backup)\b', query, re.IGNORECASE)
        entities["files"].extend(file_patterns)
        
        # Extract numbers (IDs, codes, etc.)
        number_patterns = re.findall(r'\b\d{3,}\b', query)  # 3+ digit numbers
        entities["numbers"].extend(number_patterns)
        
        # Extract dates (simple patterns)
        date_patterns = re.findall(r'\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b', query, re.IGNORECASE)
        entities["dates"].extend(date_patterns)
        
        # Extract potential people names (capitalized words)
        people_patterns = re.findall(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', query)
        entities["people"].extend(people_patterns)
        
        # Extract potential organizations (words with common org suffixes)
        org_patterns = re.findall(r'\b[A-Z][a-zA-Z\s&]+(?:Inc|Corp|LLC|Ltd|Company|Organization|University|Institute|School|Hospital)\b', query, re.IGNORECASE)
        entities["organizations"].extend(org_patterns)
        
        # Extract potential places (common place indicators)
        place_patterns = re.findall(r'\b[A-Z][a-zA-Z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|City|Town|State|Country|County)\b', query)
        entities["places"].extend(place_patterns)
        
        # Remove duplicates
        for key in entities:
            entities[key] = list(set(entities[key]))
        
        return entities
    
    def is_complex_query(self, query: str, function_docs: List[Dict[str, Any]], test_case: Dict[str, Any] = None) -> bool:
        """Determine if a query is complex enough to warrant splitting into multiple turns."""
        
        # Get existing clarifications to see what's already provided
        existing_clarifications = {}
        if test_case and "question" in test_case and test_case["question"]:
            if isinstance(test_case["question"], list) and len(test_case["question"]) > 0:
                first_turn = test_case["question"][0]
                if isinstance(first_turn, list) and len(first_turn) > 0:
                    existing_clarifications = first_turn[0].get("clarifications", {})
        
        # Count required arguments and how many are missing
        total_required = 0
        missing_required = 0
        
        for func in function_docs:
            if "parameters" in func and "required" in func["parameters"]:
                required_args = func["parameters"]["required"]
                total_required += len(required_args)
                
                for arg in required_args:
                    # Check if argument is already provided in clarifications
                    if arg in existing_clarifications:
                        continue
                    
                    # Check if argument is mentioned in the query (more flexible matching)
                    arg_lower = arg.lower()
                    query_lower = query.lower()
                    
                    # Check for exact matches or word variations
                    if (arg_lower in query_lower or 
                        arg_lower.replace("_", " ") in query_lower or
                        any(word in query_lower for word in arg_lower.split('_')) or
                        arg_lower.replace("_", "") in query_lower):
                        continue
                    
                    missing_required += 1
        
        # Consider it complex if there are missing required arguments
        if missing_required > 0:
            return True
        
        # Also consider it complex if there are multiple required arguments and less than 70% are provided
        if total_required > 1 and (total_required - missing_required) / total_required < 0.7:
            return True
        
        return False
    
    def extract_entities_and_arguments(self, query: str, function_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract named entities and function arguments from the query using rule-based approach."""
        
        # Extract named entities first
        named_entities = self.extract_named_entities(query)
        
        # Analyze function arguments using rule-based approach
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
                
                for param_name, param_info in properties.items():
                    param_type = param_info.get("type", "")
                    param_desc = param_info.get("description", "")
                    
                    # Check if this parameter is mentioned in the query
                    found_value = self._find_argument_value(query, param_name, param_type, named_entities)
                    
                    if found_value:
                        mentioned_arguments[func_name][param_name] = found_value
                    elif param_name in required:
                        missing_arguments[func_name].append(param_name)
        
        return {
            "named_entities": named_entities,
            "mentioned_arguments": mentioned_arguments,
            "missing_arguments": missing_arguments
        }
    
    def _find_argument_value(self, query: str, param_name: str, param_type: str, named_entities: Dict[str, List[str]]) -> Optional[str]:
        """Find the value of a parameter in the query using rule-based matching."""
        
        query_lower = query.lower()
        param_name_lower = param_name.lower()
        
        # Check for exact parameter name matches
        if param_name_lower in query_lower:
            # Look for values after the parameter name
            pattern = rf'{re.escape(param_name)}\s*[=:]\s*["\']?([^"\'\s,]+)["\']?'
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1)
        
        # Check for parameter name variations
        param_variations = [
            param_name,
            param_name.replace("_", " "),
            param_name.replace("_", "-"),
            param_name.title(),
            param_name.upper()
        ]
        
        for variation in param_variations:
            if variation.lower() in query_lower:
                # Look for values near the parameter name
                pattern = rf'{re.escape(variation)}\s*[=:]\s*["\']?([^"\'\s,]+)["\']?'
                match = re.search(pattern, query, re.IGNORECASE)
                if match:
                    return match.group(1)
        
        # Check named entities for potential matches
        if param_type in ["string", "integer", "float"]:
            # Look for numbers if parameter type is numeric
            if param_type in ["integer", "float"] and named_entities["numbers"]:
                return named_entities["numbers"][0]
            
            # Look for files if parameter name suggests file
            if "file" in param_name.lower() and named_entities["files"]:
                return named_entities["files"][0]
            
            # Look for people if parameter name suggests person
            if any(word in param_name.lower() for word in ["user", "person", "name", "author"]) and named_entities["people"]:
                return named_entities["people"][0]
            
            # Look for places if parameter name suggests location
            if any(word in param_name.lower() for word in ["location", "place", "address", "city", "country"]) and named_entities["places"]:
                return named_entities["places"][0]
        
        return None
    
    def generate_clarification_questions(self, analysis: Dict[str, Any], function_docs: List[Dict[str, Any]], test_case: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Generate clarification questions for missing arguments using LLM."""
        
        missing_args = analysis.get("missing_arguments", {})
        if not missing_args:
            return []
        
        # Get existing clarifications from the test case
        existing_clarifications = {}
        if test_case and "question" in test_case and test_case["question"]:
            if isinstance(test_case["question"], list) and len(test_case["question"]) > 0:
                first_turn = test_case["question"][0]
                if isinstance(first_turn, list) and len(first_turn) > 0:
                    existing_clarifications = first_turn[0].get("clarifications", {})
        
        # Format the analysis for the LLM
        analysis_text = json.dumps(analysis, indent=2)
        function_docs_text = self._format_function_docs(function_docs)
        clarifications_text = json.dumps(existing_clarifications, indent=2)
        
        prompt = f"""
Based on the following analysis of a user query, generate natural language clarification questions for missing required arguments.

**IMPORTANT RULES:**
1. ONLY ask for information that is ACTUALLY missing from the original query
2. DO NOT ask for information that is already provided in the existing clarifications
3. DO NOT add new information that wasn't mentioned in the original query
4. Make questions sound natural and conversational
5. Use existing clarifications as guidance for what information is already available

Analysis:
{analysis_text}

Function Documentation:
{function_docs_text}

Existing Clarifications (already provided):
{clarifications_text}

Here are examples of good multi-turn conversations to follow:

**Example 1: Weather Request (Temperature Unit Clarification)**
Original: "What are the current weather conditions in Tel Aviv, and could you provide that in Fahrenheit, please?"
Multi-turn:
- User: "What's the weather like in Tel Aviv right now?"
- Assistant: "What temperature unit would you prefer for the weather report - Celsius or Fahrenheit?"
- User: "Fahrenheit"

**Example 2: Uber Ride Request (Missing Location)**
Original: "I need a Comfort Uber ride and I can wait up to 600 seconds for it."
Multi-turn:
- User: "I need a Comfort Uber ride and I can wait for 10 minutes"
- Assistant: "What is your pickup location? Please provide it in the format 'Street Address, City, State, Country'."
- User: "2020 Addison Street, Berkeley, CA, USA"

**Example 3: Food Order with Restaurant**
Original: "I want to order five 'burgers' and six 'chicken wings' from uber pitada"
Multi-turn:
- User: "I want to order some food for delivery"
- Assistant: "Which restaurant would you like to order from?"
- User: "uber pitada"
- Assistant: "What food items would you like to include in your order?"
- User: "5 burgers, 6 chicken wings"

**Example 4: Bot Export List Request**
Original: "Could you help me retrieve the list of exports for my bot using the identifier 'my-bot-id' and focusing on version 'v2' sort in ascending? I want max 50 results"
Multi-turn:
- User: "I need to retrieve the list of exports for my bot"
- Assistant: "What is the unique identifier for your bot?"
- User: "my-bot-id"
- Assistant: "What is the specific version of the bot you are interested in listing exports for?"
- User: "v2"

**Example 5: Air Conditioner Control**
Original: "I want to set the airconditioner to dry mode please"
Multi-turn:
- User: "I want to set the airconditioner to dry mode please"
- Assistant: "Sure, could you also please provide the wind strength that you would like?"
- User: "medium wind thank you"

**Example 6: Food Item Change**
Original: "I've changed my mind, can I get a greens instead of what I previously ordered?"
Multi-turn:
- User: "I want to change my food order"
- Assistant: "What food item would you like to change to?"
- User: "salad"

**CRITICAL: Only generate clarifications for information that is ACTUALLY missing and would naturally be asked for in a real conversation.**

Generate clarification questions for each missing argument. Return a JSON array with this structure:
[
    {{
        "function_name": "name of the function",
        "argument_name": "name of the missing argument", 
        "question": "natural language question asking for the missing information",
        "description": "description of what this argument is for"
    }}
]

Make the questions sound natural and conversational. Focus on what specific information is needed. Don't be overly specific about format requirements - keep it natural.
"""
        
        try:
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1000
            )
            
            content = response.choices[0].message.content.strip()
            if not content:
                print(f"Warning: Empty response from OpenAI for clarification generation")
                return self._generate_fallback_clarifications(analysis, function_docs)
            
            # Try to parse JSON, handle common issues
            try:
                # Remove markdown code blocks if present
                if content.startswith("```json"):
                    content = content[7:]
                if content.endswith("```"):
                    content = content[:-3]
                
                result = json.loads(content.strip())
                if isinstance(result, list):
                    return result
                else:
                    print(f"Warning: Expected list but got {type(result)}")
                    return self._generate_fallback_clarifications(analysis, function_docs)
                    
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON response from OpenAI for clarification generation: {content[:100]}...")
                return self._generate_fallback_clarifications(analysis, function_docs)
            
        except Exception as e:
            print(f"Error generating clarification questions: {e}")
            return self._generate_fallback_clarifications(analysis, function_docs)
    
    def _generate_fallback_clarifications(self, analysis: Dict[str, Any], function_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate fallback clarification questions when LLM fails."""
        
        clarification_questions = []
        
        for function_name, missing_args in analysis.get("missing_arguments", {}).items():
            func_doc = next((f for f in function_docs if f.get("name") == function_name), None)
            if not func_doc:
                continue
                
            for arg_name in missing_args:
                arg_desc = ""
                if "parameters" in func_doc and "properties" in func_doc["parameters"]:
                    arg_info = func_doc["parameters"]["properties"].get(arg_name, {})
                    arg_desc = arg_info.get("description", "")
                
                question = self._generate_clarification_question(arg_name, arg_desc)
                clarification_questions.append({
                    "function_name": function_name,
                    "argument_name": arg_name,
                    "question": question,
                    "description": arg_desc
                })
        
        return clarification_questions
    
    def _generate_clarification_question(self, arg_name: str, arg_desc: str) -> str:
        """Generate a natural language clarification question for a missing argument."""
        readable_name = arg_name.replace("_", " ").title()
        
        if arg_desc:
            return f"What {readable_name.lower()} would you like to use? ({arg_desc})"
        else:
            return f"What {readable_name.lower()} would you like to use?"
    
    def split_into_conversation_turns(self, query: str, function_docs: List[Dict[str, Any]], test_case: Dict[str, Any] = None) -> List[List[Dict[str, Any]]]:
        """Create new multi-turn conversations that simulate clarification process."""
        
        analysis = self.extract_entities_and_arguments(query, function_docs)
        clarification_questions = self.generate_clarification_questions(analysis, function_docs, test_case)
        
        # Limit to maximum 6 clarification rounds (7 turns total)
        max_additional_turns = 6
        clarification_questions = clarification_questions[:max_additional_turns]
        
        if not clarification_questions:
            return []
        
        # Create a simplified version of the original query (removing some details)
        simplified_query = self._create_simplified_query(query, analysis)
        
        # Build the multi-turn conversation
        conversation_turns = []
        
        # Turn 1: Simplified user query
        conversation_turns.append([{
            "role": "user",
            "content": simplified_query
        }])
        
        # Generate clarification turns using LLM-generated questions
        for i, clarification in enumerate(clarification_questions):
            # Assistant asks for clarification using LLM-generated question
            assistant_turn = [{
                "role": "assistant",
                "content": clarification["question"]
            }]
            
            # User provides the missing information
            user_turn = [{
                "role": "user", 
                "content": self._generate_user_response(clarification, analysis, test_case)
            }]
            
            conversation_turns.append(assistant_turn)
            conversation_turns.append(user_turn)
        
        return conversation_turns
    
    def _create_simplified_query(self, original_query: str, analysis: Dict[str, Any]) -> str:
        """Create a simplified version of the original query for the multi-turn conversation."""
        
        # Use LLM to create a simplified version
        prompt = f"""
Create a simplified version of this user query that removes some specific details but keeps the main intent:

Original query: "{original_query}"

Analysis of what was found:
{json.dumps(analysis.get('mentioned_arguments', {}), indent=2)}

Create a simplified version that:
1. Keeps the main intent/request
2. Removes some specific details (like exact addresses, times, etc.) but NOT information that would naturally be asked for
3. Sounds natural and conversational
4. Is shorter than the original
5. DO NOT remove information that is essential to the request

**Examples:**
- "I need a Comfort Uber ride from 2020 Addison Street, Berkeley, CA, USA, and I can wait up to 600 seconds for it." → "I need a Comfort Uber ride and I can wait for 10 minutes"
- "What are the current weather conditions in Tel Aviv, and could you provide that in Fahrenheit, please?" → "What's the weather like in Tel Aviv right now?"
- "I want to see the star history of ShishirPatil/gorilla and gorilla-llm/gorilla-cli" → "I want to see the star history of some GitHub repositories"

Return just the simplified query, nothing else.
"""
        
        try:
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200
            )
            
            content = response.choices[0].message.content.strip()
            if content and not content.startswith("```"):
                return content
            else:
                # Fallback: just return a basic version
                return self._create_fallback_simplified_query(original_query)
                
        except Exception as e:
            print(f"Error creating simplified query: {e}")
            return self._create_fallback_simplified_query(original_query)
    
    def _create_fallback_simplified_query(self, original_query: str) -> str:
        """Create a basic simplified version when LLM fails."""
        # Remove specific details like addresses, times, etc.
        simplified = original_query
        
        # Remove specific addresses
        simplified = re.sub(r'\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Place|Pl|Court|Ct|Way|Terrace|Ter|Circle|Cir|Square|Sq|Highway|Hwy|Freeway|Fwy|Interstate|I-\d+)[,\s]*[A-Za-z\s]*[A-Z]{2}\s*\d{5}?', '[LOCATION]', simplified)
        
        # Remove specific times
        simplified = re.sub(r'\d+\s*(?:seconds?|minutes?|hours?|days?|weeks?|months?|years?)', '[TIME]', simplified)
        
        # Remove specific numbers that might be IDs
        simplified = re.sub(r'\b\d{4,}\b', '[NUMBER]', simplified)
        
        return simplified
    
    def _generate_user_response(self, clarification: Dict[str, Any], analysis: Dict[str, Any], test_case: Dict[str, Any] = None) -> str:
        """Generate a user response for the missing argument using existing clarifications or simple extraction."""
        
        argument_name = clarification["argument_name"]
        
        # First, check if there are existing clarifications in the test case
        if test_case and "question" in test_case and test_case["question"]:
            if isinstance(test_case["question"], list) and len(test_case["question"]) > 0:
                first_turn = test_case["question"][0]
                if isinstance(first_turn, list) and len(first_turn) > 0:
                    existing_clarifications = first_turn[0].get("clarifications", {})
                    if existing_clarifications and argument_name in existing_clarifications:
                        return str(existing_clarifications[argument_name])
        
        # Second, try to extract from transcript using simple patterns
        if test_case and "question" in test_case and test_case["question"]:
            if isinstance(test_case["question"], list) and len(test_case["question"]) > 0:
                first_turn = test_case["question"][0]
                if isinstance(first_turn, list) and len(first_turn) > 0:
                    transcript = first_turn[0].get("transcript", "")
                    content = first_turn[0].get("content", "")
                    
                    # Simple pattern matching for common cases
                    if "unit" in argument_name.lower() or "temperature" in argument_name.lower():
                        if "fahrenheit" in content.lower() or "fahr" in content.lower():
                            return "Fahrenheit"
                        elif "celsius" in content.lower():
                            return "Celsius"
                    
                    elif "location" in argument_name.lower() or "address" in argument_name.lower() or "pickup" in argument_name.lower():
                        # Look for location patterns in transcript
                        location_patterns = [
                            r'\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Place|Pl|Court|Ct|Way|Terrace|Ter|Circle|Cir|Square|Sq|Highway|Hwy|Freeway|Fwy|Interstate|I-\d+)',
                            r'[A-Za-z\s]+,\s*[A-Za-z\s]+(?:,\s*[A-Z]{2})?'
                        ]
                        
                        for pattern in location_patterns:
                            matches = re.findall(pattern, transcript, re.IGNORECASE)
                            if matches:
                                return matches[0].strip()
                    
                    elif "time" in argument_name.lower() or "duration" in argument_name.lower() or "wait" in argument_name.lower():
                        # Look for time patterns
                        time_patterns = [
                            r'\d+\s*(?:seconds?|minutes?|hours?|days?|weeks?|months?|years?)',
                            r'\d+\s*(?:sec|min|hr|day|week|month|year)'
                        ]
                        
                        for pattern in time_patterns:
                            matches = re.findall(pattern, transcript, re.IGNORECASE)
                            if matches:
                                return matches[0].strip()
                    
                    elif "type" in argument_name.lower():
                        # Look for ride types
                        if "comfort" in content.lower():
                            return "Comfort"
                        elif "plus" in content.lower():
                            return "Plus"
                        elif "black" in content.lower():
                            return "Black"
                    
                    elif "id" in argument_name.lower():
                        # Look for IDs
                        id_patterns = [
                            r'[A-Z]\d+',
                            r'\d+',
                            r'[a-z-]+-?[a-z-]+'
                        ]
                        
                        for pattern in id_patterns:
                            matches = re.findall(pattern, transcript)
                            if matches:
                                return matches[0].strip()
        
        # Fallback to simple placeholder
        return f"[USER PROVIDES: {argument_name}]"
    
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
        if not self.is_complex_query(user_query, function_docs, test_case):
            return test_case
        
        # Create multi-turn conversations
        try:
            print(f"  - Creating multi-turn conversation for: {user_query[:100]}...")
            conversation_turns = self.split_into_conversation_turns(user_query, function_docs, test_case)
            
            if conversation_turns:
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
            random.seed(42)  # For reproducible results
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
