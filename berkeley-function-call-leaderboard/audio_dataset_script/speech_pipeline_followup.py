import json
import os
import random
import time
from typing import Dict, List, Optional, Tuple
import openai
import concurrent.futures  # Added for multithreading support
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv
import re
import inspect
from huanzhi_utils import *
from copy import deepcopy
from tqdm import tqdm  # Added for progress bar
import argparse
import contextlib
import sys

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "o3-2025-04-16"

client = openai.OpenAI(api_key=OPENAI_API_KEY)


def _query_openai(messages):
    while True:
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                response_format={"type": "json_object"},
            )

            result = response.choices[0].message.content.strip()
            result = json.loads(result)
            result = result["conversation"]
            # print(result)
            assert isinstance(result, list) 
            assert result[0]["role"] == "user" and result[-1]["role"] == "user"
            assert all(msg["role"] in ["user", "assistant"] for msg in result)
            assert all(len(msg) == 2 for msg in result)
            return result
        except Exception as e:
            print(f"Error: {repr(e)}")
            time.sleep(1)



SYSTEM_INSTRUCTION = """You are a data-augmentation assistant creating training conversations for an audio-based voice agent. To teach the audio agent that it should never blindly trust the ASR module output, you will need to add in extra follow up turns to ask for clarification.

Input:
- user_query: the raw ASR transcript of the user's utterance.
- clarifications: a dictionary of named-entity values detected in the query that the user meant to say, but might have been mis-heard.

Task:
1. Keep the original user query as the first turn: {"role": "user", "content": user_query}.
2. Add follow-up turns so the assistant carefully confirms every entity listed in clarifications. Do not confirm entities that are not listed in clarifications.
   • Turn 2 (assistant): ask for confirmation or spelling of one or more entities.
   • Turn 3 (user): provide the spelling or confirmation using an audio-friendly style (e.g. "J-O-H-N", "0-U-4-N-P-P").
   • Repeat assistant ↔ user turns until ALL entities are confirmed. Feel free to confirm multiple entities in a single assistant turn when it sounds natural.
3. Do NOT introduce new information or omit details from the original query.
4. Always aim for a natural, conversational flow.

Output:
Return a JSON array of dictionaries:
{"conversation": [
    { "role": "user", "content": "<text>" },
    { "role": "assistant", "content": "<text>" },
    ...
]}
Roles must alternate between user and assistant, and the last message MUST come from the user. Do not include any other fields in the dictionary.

Example:
Input:
```json
{
  "user_query": "Hello, I have a request regarding a past flight reservation, 0U4NPP, from PHL to DEN. I was wondering if I could still cancel it due to personal reasons.",
  "clarifications": {
    "reservation_id": "0U4NPP",
    "origin_airport_code": "PHL",
    "destination_airport_code": "DEN"
  }
}
```

Expected Output:
```json
[
  {"role":"user","content":"Hello, I have a request regarding a past flight reservation, 0U4NPP, from PHL to DEN. I was wondering if I could still cancel it due to personal reasons."},
  {"role":"assistant","content":"I just want to make sure I’ve got the reservation ID correct. Could you spell it for me?"},
  {"role":"user","content":"Sure, it's 0-U-4-N-P-P."},
  {"role":"assistant","content":"Great, thanks! And could you please confirm the airport codes? Is that PHL, P-H-L for departure and DEN, D-E-N for arrival?"},
  {"role":"user","content":"Yes, that's correct: P-H-L for departure and D-E-N for arrival."}
]
```
"""

def _format_instruction(content, clarifications):
    return f"""**User Query:**
{content}

**Clarifications:**
{clarifications}
"""

def _process_message(item: dict) -> list[dict]:
    """Takes in a user message, and return a list of messages that have the follow up turns added"""
    content = item["content"]
    clarifications = item["clarifications"]
    if len(clarifications) == 0:
        return [item]
    messages = [
        {"role": "developer", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": _format_instruction(content, clarifications)},
    ]
    result = _query_openai(messages)
    return result




# path = "/Users/hans/repo/API-Gen/apigen_mt_5k_clarification_spoken.json"
path = "/Users/hans/repo/API-Gen/hermes_function_calling_v1_filtered_clarification_spoken.json"
data = load_file(path)


def _process_conversation_item(item: dict) -> dict:
    """Process a single conversation item and return the updated version."""
    messages = item["messages"]
    result: list[dict] = []
    index = 0
    while index < len(messages):
        msg = messages[index]
        result.append(msg)
        index += 1
        if msg["role"] == "user":
            msg_with_followup = _process_message(msg)
            if len(msg_with_followup) > 1:
                msg_with_followup = msg_with_followup[1:]
                for followup_msg in msg_with_followup:
                    followup_msg["eot"] = True
                result.extend(msg_with_followup)
    # Mutate a shallow copy so that original list isn't modified in place
    item["messages"] = result
    return item


# -------------------- Run in parallel -------------------- #


MAX_WORKERS = 500  # Default to number of CPUs

with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    # Use tqdm to get a progress bar that plays nicely with threads
    futures = [executor.submit(_process_conversation_item, itm) for itm in data]
    processed_items: list[dict] = []
    for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing"):
        processed_items.append(f.result())


output_path = "/Users/hans/repo/API-Gen/hermes_function_calling_v1_filtered_clarification_spoken_with_followup.json"
write_list_of_dicts_to_file(output_path, processed_items)

