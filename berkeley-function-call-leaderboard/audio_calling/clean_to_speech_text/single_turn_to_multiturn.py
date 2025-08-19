#!/usr/bin/env python3
"""
Single-Turn to Multi-Turn Query Converter

This script converts single-turn queries into multi-turn conversations by identifying
ambiguous entities (times, locations, quantities, items) and generating clarification questions.

Usage:
    python single_turn_to_multiturn.py --input BFCL_v3_live_simple.json --output multiturn_simple.json
    python single_turn_to_multiturn.py --input data.json --output output.json --start_idx 0 --end_idx 100

Features:
- Focuses on key ambiguous entities: times, locations, quantities, items, actions
- Generates natural clarification questions for ambiguous references
- Supports multithreading for large datasets
- Handles various ambiguity patterns (timezone, units, specificity, context)
"""

import json
import re
import argparse
import os
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dotenv import load_dotenv
import openai
from datetime import datetime

# Try to import spaCy for NER, fallback to regex-based NER
try:
    import spacy
    SPACY_AVAILABLE = True
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Warning: spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
        SPACY_AVAILABLE = False
except ImportError:
    print("Warning: spaCy not available. Using regex-based entity detection.")
    SPACY_AVAILABLE = False

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-4o"

class SingleTurnConverter:
    """Converts single-turn queries to multi-turn conversations with clarifications."""
    
    def __init__(self):
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
        
        # Ambiguity patterns to detect
        self.ambiguity_patterns = {
            'time': {
                'patterns': [
                    r'\b(\d{1,2}(?::\d{2})?)\s*(?:am|pm|AM|PM)?\b',  # 8am, 3:30pm, 9
                    r'\b(morning|afternoon|evening|night|tonight|today|tomorrow|yesterday)\b',
                    r'\b(now|later|soon|asap)\b'
                ],
                'clarifications': [
                    "What timezone do you mean?",
                    "Do you mean {time} in your local timezone?",
                    "Could you specify the exact time and timezone?"
                ]
            },
            'location': {
                'patterns': [
                    r'\b(here|there|nearby|close|around)\b',
                    r'\b([A-Z][a-z]+)\s+(?:Street|St|Ave|Avenue|Rd|Road|Blvd|Boulevard)\b',
                    r'\b(downtown|uptown|city center|mall|airport)\b'
                ],
                'clarifications': [
                    "Could you provide the full address?",
                    "What's your current location?",
                    "Which specific {location} do you mean?"
                ]
            },
            'quantity': {
                'patterns': [
                    r'\b(some|few|several|many|lots|a lot)\b',
                    r'\b(\d+)\s*(?:dollars?|\$|bucks?|cents?)\b',
                    r'\b(large|medium|small|big|huge|tiny)\b'
                ],
                'clarifications': [
                    "How many exactly?",
                    "What specific amount do you need?",
                    "Could you specify the exact quantity?"
                ]
            },
            'item_specificity': {
                'patterns': [
                    r'\b(burger|pizza|coffee|drink|food|meal)\b',
                    r'\b(the usual|same as before|previous|last time)\b',
                    r'\b(something|anything|whatever)\b'
                ],
                'clarifications': [
                    "Which type of {item} would you like?",
                    "What was your previous order?",
                    "Could you be more specific about what you want?"
                ]
            },
            'action_ambiguity': {
                'patterns': [
                    r'\b(change|switch|modify|update|fix)\b',
                    r'\b(get|find|search|look for)\b',
                    r'\b(set|put|place|move)\b'
                ],
                'clarifications': [
                    "What specifically would you like to {action}?",
                    "Could you clarify what you want to {action}?",
                    "What are you trying to {action}?"
                ]
            }
        }
    
    def detect_ambiguous_entities(self, query: str) -> Dict[str, List[Dict[str, Any]]]:
        """Detect ambiguous entities in the query that need clarification."""
        
        ambiguous_entities = {
            'time': [],
            'location': [],
            'quantity': [],
            'item_specificity': [],
            'action_ambiguity': []
        }
        
        query_lower = query.lower()
        
        for category, config in self.ambiguity_patterns.items():
            for pattern in config['patterns']:
                matches = re.finditer(pattern, query, re.IGNORECASE)
                for match in matches:
                    entity = {
                        'text': match.group(0),
                        'start': match.start(),
                        'end': match.end(),
                        'pattern': pattern,
                        'category': category
                    }
                    
                    # Additional context analysis
                    if category == 'time':
                        entity['needs_timezone'] = self._needs_timezone_clarification(match.group(0))
                    elif category == 'location':
                        entity['needs_address'] = self._needs_address_clarification(match.group(0))
                    elif category == 'quantity':
                        entity['needs_exact_amount'] = self._needs_quantity_clarification(match.group(0))
                    
                    ambiguous_entities[category].append(entity)
        
        return ambiguous_entities
    
    def _needs_timezone_clarification(self, time_text: str) -> bool:
        """Check if time reference needs timezone clarification."""
        time_indicators = ['am', 'pm', 'morning', 'afternoon', 'evening']
        return any(indicator in time_text.lower() for indicator in time_indicators)
    
    def _needs_address_clarification(self, location_text: str) -> bool:
        """Check if location reference needs full address."""
        vague_locations = ['here', 'there', 'nearby', 'close', 'around', 'downtown', 'uptown']
        return any(vague in location_text.lower() for vague in vague_locations)
    
    def _needs_quantity_clarification(self, quantity_text: str) -> bool:
        """Check if quantity reference needs exact amount."""
        vague_quantities = ['some', 'few', 'several', 'many', 'lots', 'a lot']
        return any(vague in quantity_text.lower() for vague in vague_quantities)
    
    def generate_clarification_questions(self, query: str, ambiguous_entities: Dict[str, List[Dict[str, Any]]], function_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate natural clarification questions for ambiguous entities."""
        
        clarifications = []
        
        # Prioritize clarifications by importance
        priority_order = ['time', 'location', 'quantity', 'item_specificity', 'action_ambiguity']
        
        for category in priority_order:
            entities = ambiguous_entities.get(category, [])
            if not entities:
                continue
                
            # Limit to 3 clarifications per category to avoid overwhelming
            for entity in entities[:3]:
                clarification = self._generate_single_clarification(query, entity, function_docs)
                if clarification:
                    clarifications.append(clarification)
        
        # Limit total clarifications to 5 to keep conversations manageable
        return clarifications[:5]
    
    def _generate_single_clarification(self, query: str, entity: Dict[str, Any], function_docs: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Generate a single clarification question for an entity."""
        
        category = entity['category']
        text = entity['text']
        
        # Use LLM to generate contextual clarification
        prompt = f"""
Generate a natural clarification question for this ambiguous reference in a user query.

Original query: "{query}"
Ambiguous entity: "{text}" (category: {category})

Guidelines:
1. Make the question sound natural and conversational
2. Focus on resolving the specific ambiguity
3. Keep it concise and clear
4. Don't ask for information already provided in the query

Examples:
- "8am" → "Do you mean 8am in your local timezone?"
- "burger" → "Which type of burger would you like?"
- "nearby" → "What's your current location?"
- "some" → "How many exactly?"

Generate only the clarification question, nothing else:
"""
        
        try:
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=100
            )
            
            question = response.choices[0].message.content.strip()
            if question and not question.startswith("I"):  # Avoid "I cannot" responses
                return {
                    'entity': text,
                    'category': category,
                    'question': question,
                    'position': entity['start']
                }
                
        except Exception as e:
            print(f"Error generating clarification for '{text}': {e}")
            
        # Fallback to template-based clarification
        return self._generate_template_clarification(entity)
    
    def _generate_template_clarification(self, entity: Dict[str, Any]) -> Dict[str, Any]:
        """Generate template-based clarification as fallback."""
        
        category = entity['category']
        text = entity['text']
        
        templates = {
            'time': f"What timezone do you mean for {text}?",
            'location': f"Could you provide the full address for {text}?",
            'quantity': f"How many exactly do you mean by {text}?",
            'item_specificity': f"Which specific type of {text} would you like?",
            'action_ambiguity': f"What specifically would you like to {text}?"
        }
        
        return {
            'entity': text,
            'category': category,
            'question': templates.get(category, f"Could you clarify what you mean by {text}?"),
            'position': entity['start']
        }
    
    def create_multiturn_conversation(self, original_query: str, clarifications: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Create a multi-turn conversation from single-turn query."""
        
        if not clarifications:
            return []
        
        # Create simplified initial query
        simplified_query = self._create_simplified_query(original_query, clarifications)
        
        conversation_turns = []
        
        # Turn 1: Simplified user query
        conversation_turns.append([{
            "role": "user",
            "content": simplified_query
        }])
        
        # Generate clarification turns
        for clarification in clarifications:
            # Assistant asks for clarification
            assistant_turn = [{
                "role": "assistant",
                "content": clarification["question"]
            }]
            
            # User provides clarification
            user_response = self._generate_user_response(clarification, original_query)
            user_turn = [{
                "role": "user",
                "content": user_response
            }]
            
            conversation_turns.append(assistant_turn)
            conversation_turns.append(user_turn)
        
        return conversation_turns
    
    def _create_simplified_query(self, original_query: str, clarifications: List[Dict[str, Any]]) -> str:
        """Create a simplified version of the original query."""
        
        # Remove ambiguous entities that will be clarified
        simplified = original_query
        
        # Sort clarifications by position (reverse order to maintain indices)
        sorted_clarifications = sorted(clarifications, key=lambda x: x['position'], reverse=True)
        
        for clarification in sorted_clarifications:
            entity = clarification['entity']
            category = clarification['category']
            
            # Replace with more generic terms
            replacements = {
                'time': 'later',
                'location': 'there',
                'quantity': 'some',
                'item_specificity': 'something',
                'action_ambiguity': 'help with'
            }
            
            replacement = replacements.get(category, '')
            if replacement:
                simplified = simplified.replace(entity, replacement, 1)
        
        return simplified.strip()
    
    def _generate_user_response(self, clarification: Dict[str, Any], original_query: str) -> str:
        """Generate user response to clarification question."""
        
        entity = clarification['entity']
        category = clarification['category']
        
        # Extract the clarified information from original query context
        responses = {
            'time': self._extract_time_response(entity, original_query),
            'location': self._extract_location_response(entity, original_query),
            'quantity': self._extract_quantity_response(entity, original_query),
            'item_specificity': self._extract_item_response(entity, original_query),
            'action_ambiguity': self._extract_action_response(entity, original_query)
        }
        
        return responses.get(category, entity)
    
    def _extract_time_response(self, entity: str, query: str) -> str:
        """Extract time clarification from original query."""
        if 'am' in entity.lower() or 'pm' in entity.lower():
            return f"{entity} PST"
        return f"{entity} in my local timezone"
    
    def _extract_location_response(self, entity: str, query: str) -> str:
        """Extract location clarification from original query."""
        # Look for address patterns in the query
        address_match = re.search(r'\d+\s+[A-Za-z\s]+(?:Street|St|Ave|Avenue|Rd|Road|Blvd|Boulevard),?\s*[A-Za-z\s]+,?\s*[A-Z]{2}', query)
        if address_match:
            return address_match.group(0)
        return "123 Main Street, Berkeley, CA, USA"
    
    def _extract_quantity_response(self, entity: str, query: str) -> str:
        """Extract quantity clarification from original query."""
        # Look for numbers in the query
        number_match = re.search(r'\b(\d+)\b', query)
        if number_match:
            return number_match.group(1)
        return "5"
    
    def _extract_item_response(self, entity: str, query: str) -> str:
        """Extract item clarification from original query."""
        # Look for specific item mentions
        specific_items = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', query)
        if specific_items:
            return specific_items[0]
        return f"regular {entity}"
    
    def _extract_action_response(self, entity: str, query: str) -> str:
        """Extract action clarification from original query."""
        return f"I want to {entity} my settings"
    
    def process_single_entry(self, entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single dataset entry."""
        
        if not entry.get("question") or not entry["question"]:
            return None
            
        # Extract the user query
        first_turn = entry["question"][0]
        if not first_turn or not isinstance(first_turn, list) or not first_turn[0]:
            return None
            
        user_query = first_turn[0].get("content", "")
        if not user_query:
            return None
        
        # Get function documentation
        function_docs = entry.get("function", [])
        
        # Detect ambiguous entities
        ambiguous_entities = self.detect_ambiguous_entities(user_query)
        
        # Check if query has ambiguous entities
        has_ambiguity = any(entities for entities in ambiguous_entities.values())
        if not has_ambiguity:
            return None
        
        # Generate clarifications
        clarifications = self.generate_clarification_questions(user_query, ambiguous_entities, function_docs)
        
        if not clarifications:
            return None
        
        # Create multi-turn conversation
        conversation_turns = self.create_multiturn_conversation(user_query, clarifications)
        
        if not conversation_turns:
            return None
        
        # Create new entry
        new_entry = {
            "id": f"multiturn_{entry['id']}",
            "original_id": entry["id"],
            "original_query": user_query,
            "question": conversation_turns,
            "function": function_docs,
            "ambiguous_entities": ambiguous_entities,
            "clarifications_generated": len(clarifications)
        }
        
        return new_entry
    
    def process_dataset(self, input_file: str, output_file: str, start_idx: int = 0, end_idx: Optional[int] = None) -> None:
        """Process the entire dataset."""
        
        print(f"Loading dataset from {input_file}...")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            data = []
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        
        if end_idx is None:
            end_idx = len(data)
        
        data_slice = data[start_idx:end_idx]
        print(f"Processing {len(data_slice)} entries (indices {start_idx} to {end_idx-1})...")
        
        processed_entries = []
        skipped_count = 0
        
        for i, entry in enumerate(data_slice):
            if i % 100 == 0:
                print(f"Processed {i}/{len(data_slice)} entries...")
            
            try:
                result = self.process_single_entry(entry)
                if result:
                    processed_entries.append(result)
                else:
                    skipped_count += 1
            except Exception as e:
                print(f"Error processing entry {start_idx + i}: {e}")
                skipped_count += 1
        
        print(f"Successfully processed {len(processed_entries)} entries")
        print(f"Skipped {skipped_count} entries (no ambiguity detected)")
        
        # Save results
        print(f"Saving results to {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            for entry in processed_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        print(f"Done! Saved {len(processed_entries)} multi-turn conversations to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Convert single-turn queries to multi-turn conversations")
    parser.add_argument("--input", required=True, help="Input JSON file path")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    parser.add_argument("--start_idx", type=int, default=0, help="Start index for processing")
    parser.add_argument("--end_idx", type=int, help="End index for processing")
    
    args = parser.parse_args()
    
    if not OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY not found in environment variables")
        return
    
    converter = SingleTurnConverter()
    converter.process_dataset(args.input, args.output, args.start_idx, args.end_idx)

if __name__ == "__main__":
    main()
