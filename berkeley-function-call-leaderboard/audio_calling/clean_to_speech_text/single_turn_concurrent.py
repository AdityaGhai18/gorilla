#!/usr/bin/env python3
"""
Concurrent Single-Turn to Multi-Turn Query Converter

High-performance, bulletproof conversion of single-turn queries to multi-turn conversations
with maximum concurrency, robust error handling, and checkpointing.

Usage:
    python single_turn_concurrent.py --input BFCL_v3_live_simple.json --output multiturn_simple.json
    python single_turn_concurrent.py --input data.json --output output.json --test
    python single_turn_concurrent.py --input data.json --output output.json --workers 100
"""

import json
import sys
import os
import argparse
import time
import pickle
import signal
import atexit
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()

# Maximum concurrency configuration
MAX_WORKERS = 250  # All workers running simultaneously
CASES_PER_WORKER = 1  # Each worker processes exactly 1 case at a time

# BULLETPROOF settings
CHECKPOINT_INTERVAL = 100  # Save progress every 100 cases
MAX_RETRIES = 3  # Retry failed API calls
RETRY_DELAY = 1.0  # Delay between retries
FORCE_SAVE_INTERVAL = 300  # Force save every 5 minutes

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-4o"

# Global variables for checkpointing
processed_cases = {}
failed_cases = []
start_time = None
checkpoint_file = None
output_file = None
data_to_process = None

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('single_turn_concurrent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def signal_handler(signum, frame):
    """Handle signals gracefully and save progress."""
    logger.warning(f"Received signal {signum}, saving progress and exiting gracefully...")
    save_checkpoint()
    sys.exit(0)

def save_checkpoint():
    """Save progress checkpoint."""
    global processed_cases, failed_cases, checkpoint_file
    
    if checkpoint_file and (processed_cases or failed_cases):
        try:
            checkpoint_data = {
                'processed_cases': processed_cases,
                'failed_cases': failed_cases,
                'timestamp': datetime.now().isoformat(),
                'total_processed': len(processed_cases),
                'total_failed': len(failed_cases)
            }
            
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint_data, f)
                
            logger.info(f"Checkpoint saved: {len(processed_cases)} processed, {len(failed_cases)} failed")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

def save_single_case_to_json(case_idx, processed_case):
    """Save a single completed case immediately to the output JSON file."""
    global output_file
    
    if not output_file or not processed_case:
        return
        
    try:
        # Append to output file immediately
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(processed_case, ensure_ascii=False) + '\n')
        
        logger.debug(f"Saved case {case_idx} to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save case {case_idx}: {e}")

def load_checkpoint():
    """Load existing checkpoint if available."""
    global processed_cases, failed_cases, checkpoint_file
    
    if checkpoint_file and os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
                
            processed_cases = checkpoint_data.get('processed_cases', {})
            failed_cases = checkpoint_data.get('failed_cases', [])
            
            logger.info(f"Loaded checkpoint: {len(processed_cases)} processed, {len(failed_cases)} failed")
            return True
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            processed_cases = {}
            failed_cases = []
    
    return False

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
                ]
            },
            'location': {
                'patterns': [
                    r'\b(here|there|nearby|close|around)\b',
                    r'\b([A-Z][a-z]+)\s+(?:Street|St|Ave|Avenue|Rd|Road|Blvd|Boulevard)\b',
                    r'\b(downtown|uptown|city center|mall|airport)\b'
                ]
            },
            'quantity': {
                'patterns': [
                    r'\b(some|few|several|many|lots|a lot)\b',
                    r'\b(\d+)\s*(?:dollars?|\$|bucks?|cents?)\b',
                    r'\b(large|medium|small|big|huge|tiny)\b'
                ]
            },
            'item_specificity': {
                'patterns': [
                    r'\b(burger|pizza|coffee|drink|food|meal)\b',
                    r'\b(the usual|same as before|previous|last time)\b',
                    r'\b(something|anything|whatever)\b'
                ]
            },
            'action_ambiguity': {
                'patterns': [
                    r'\b(change|switch|modify|update|fix)\b',
                    r'\b(get|find|search|look for)\b',
                    r'\b(set|put|place|move)\b'
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
        
        for category, config in self.ambiguity_patterns.items():
            for pattern in config['patterns']:
                matches = re.finditer(pattern, query, re.IGNORECASE)
                for match in matches:
                    entity = {
                        'text': match.group(0),
                        'start': match.start(),
                        'end': match.end(),
                        'category': category
                    }
                    ambiguous_entities[category].append(entity)
        
        return ambiguous_entities
    
    def generate_clarification_with_retry(self, query: str, entity: Dict[str, Any], function_docs: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Generate clarification with retry logic."""
        
        for attempt in range(MAX_RETRIES):
            try:
                return self._generate_single_clarification(query, entity, function_docs)
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for entity '{entity['text']}': {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (2 ** attempt))  # Exponential backoff
                else:
                    logger.error(f"All attempts failed for entity '{entity['text']}'")
                    return self._generate_template_clarification(entity)
        
        return None
    
    def _generate_single_clarification(self, query: str, entity: Dict[str, Any], function_docs: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Generate a single clarification question for an entity."""
        
        category = entity['category']
        text = entity['text']
        
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
        if category == 'time':
            if 'am' in entity.lower() or 'pm' in entity.lower():
                return f"{entity} PST"
            return f"{entity} in my local timezone"
        elif category == 'location':
            # Look for address patterns in the query
            address_match = re.search(r'\d+\s+[A-Za-z\s]+(?:Street|St|Ave|Avenue|Rd|Road|Blvd|Boulevard),?\s*[A-Za-z\s]+,?\s*[A-Z]{2}', original_query)
            if address_match:
                return address_match.group(0)
            return "123 Main Street, Berkeley, CA, USA"
        elif category == 'quantity':
            # Look for numbers in the query
            number_match = re.search(r'\b(\d+)\b', original_query)
            if number_match:
                return number_match.group(1)
            return "5"
        elif category == 'item_specificity':
            # Look for specific item mentions
            specific_items = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', original_query)
            if specific_items:
                return specific_items[0]
            return f"regular {entity}"
        elif category == 'action_ambiguity':
            return f"I want to {entity} my settings"
        
        return entity

def process_single_case_robust(case_data):
    """Process a single case with BULLETPROOF error handling and retries."""
    
    case_idx, entry = case_data
    
    try:
        converter = SingleTurnConverter()
        
        if not entry.get("question") or not entry["question"]:
            return case_idx, None, "No question field"
            
        # Extract the user query
        first_turn = entry["question"][0]
        if not first_turn or not isinstance(first_turn, list) or not first_turn[0]:
            return case_idx, None, "Invalid question structure"
            
        user_query = first_turn[0].get("content", "")
        if not user_query:
            return case_idx, None, "Empty query content"
        
        # Get function documentation
        function_docs = entry.get("function", [])
        
        # Detect ambiguous entities
        ambiguous_entities = converter.detect_ambiguous_entities(user_query)
        
        # Check if query has ambiguous entities
        has_ambiguity = any(entities for entities in ambiguous_entities.values())
        if not has_ambiguity:
            return case_idx, None, "No ambiguity detected"
        
        # Generate clarifications with retry logic
        clarifications = []
        priority_order = ['time', 'location', 'quantity', 'item_specificity', 'action_ambiguity']
        
        for category in priority_order:
            entities = ambiguous_entities.get(category, [])
            if not entities:
                continue
                
            # Limit to 3 clarifications per category
            for entity in entities[:3]:
                clarification = converter.generate_clarification_with_retry(user_query, entity, function_docs)
                if clarification:
                    clarifications.append(clarification)
        
        # Limit total clarifications to 5
        clarifications = clarifications[:5]
        
        if not clarifications:
            return case_idx, None, "No clarifications generated"
        
        # Create multi-turn conversation
        conversation_turns = converter.create_multiturn_conversation(user_query, clarifications)
        
        if not conversation_turns:
            return case_idx, None, "Failed to create conversation"
        
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
        
        return case_idx, new_entry, "Success"
        
    except Exception as e:
        logger.error(f"Error processing case {case_idx}: {e}")
        return case_idx, None, str(e)

def main(input_file, output_file_path, test_mode=False, max_workers=None):
    """Main processing function with bulletproof concurrency."""
    
    global processed_cases, failed_cases, start_time, checkpoint_file, output_file, data_to_process
    
    # Setup
    start_time = datetime.now()
    checkpoint_file = f"{output_file_path}.checkpoint"
    output_file = output_file_path
    
    # Register signal handlers and cleanup
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    atexit.register(save_checkpoint)
    
    # Load checkpoint if exists
    load_checkpoint()
    
    # Load data
    logger.info(f"Loading dataset from {input_file}...")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = []
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    
    data_to_process = data
    
    if test_mode:
        data = data[:100]  # Test mode: only 100 cases
        max_workers = min(max_workers or 100, 100)
        logger.info("TEST MODE: Processing 100 cases with limited workers")
    
    if max_workers is None:
        max_workers = MAX_WORKERS
    
    logger.info(f"Processing {len(data)} entries with {max_workers} workers...")
    
    # Clear output file
    if not os.path.exists(output_file) or len(processed_cases) == 0:
        open(output_file, 'w').close()
    
    # Prepare cases to process (skip already processed)
    cases_to_process = []
    for i, entry in enumerate(data):
        if i not in processed_cases:
            cases_to_process.append((i, entry))
    
    logger.info(f"Processing {len(cases_to_process)} new cases (skipping {len(processed_cases)} already processed)")
    
    # Process with maximum concurrency
    completed_count = len(processed_cases)
    last_checkpoint_time = time.time()
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all cases
        future_to_case = {
            executor.submit(process_single_case_robust, case_data): case_data[0] 
            for case_data in cases_to_process
        }
        
        # Process results as they complete
        for future in as_completed(future_to_case):
            case_idx = future_to_case[future]
            
            try:
                case_idx, result, status = future.result()
                
                if result:
                    processed_cases[case_idx] = result
                    save_single_case_to_json(case_idx, result)
                    completed_count += 1
                    logger.info(f"✓ Case {case_idx}: {status} (Total: {completed_count}/{len(data)})")
                else:
                    failed_cases.append((case_idx, status))
                    logger.warning(f"✗ Case {case_idx}: {status}")
                
                # Periodic checkpoint
                if completed_count % CHECKPOINT_INTERVAL == 0 or time.time() - last_checkpoint_time > FORCE_SAVE_INTERVAL:
                    save_checkpoint()
                    last_checkpoint_time = time.time()
                    
            except Exception as e:
                failed_cases.append((case_idx, str(e)))
                logger.error(f"✗ Case {case_idx}: Execution failed: {e}")
    
    # Final save
    save_checkpoint()
    
    # Summary
    total_time = datetime.now() - start_time
    success_count = len(processed_cases)
    failure_count = len(failed_cases)
    
    logger.info(f"""
=== PROCESSING COMPLETE ===
Total time: {total_time}
Successfully processed: {success_count}
Failed: {failure_count}
Success rate: {success_count/(success_count+failure_count)*100:.1f}%
Output file: {output_file}
""")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Concurrent Single-Turn to Multi-Turn Converter')
    parser.add_argument('--input', required=True, help='Input JSON file path')
    parser.add_argument('--output', required=True, help='Output JSON file path')
    parser.add_argument('--test', action='store_true', help='Test mode: 100 workers, 100 cases')
    parser.add_argument('--workers', type=int, help='Number of workers (default: 250)')
    
    args = parser.parse_args()
    
    if not OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY not found in environment variables")
        sys.exit(1)
    
    main(args.input, args.output, args.test, args.workers)
