import os
from huanzhi_utils import *
from openai import OpenAI
import random

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
import json
from string import Template
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


path = "/Users/hans/repo/API-Gen/xlam_function_calling_60k_clarification.json"
data = load_file(path)
# data = random.sample(data, 100)

# write_list_of_dicts_to_file(data[:20], path)

SYSTEM_INSTRUCTION = """Typed user query is often different from what was spoken. People tend to give much shorter queries when speaking. You task is to determine if the given typed query would be natural to get break down into multiple turns if the same idea is expressed in a conversational manner. If it is, then you will need to create a natural multi-turn conversation from this single user query. The goal is to simulate how a real conversation would flow when the system needs to clarify information.

There are two types of conversation splits that you need to perform:

**TYPE 1: Information Breakdown**
Because the spoken query is often shorter, you need to break down a long sentense into multiple short ones. You should hold back some information (those that are the function parameters) from the typed query, and ask for it (and supply back the information) in the next turn. It's up to you to decide what information to remove and what to ask for. You don't have to remove all the information in the first turn, and you don't have to supply all the information in the next user turn; use your judgement to create a natural conversation flow.
- **Remove some information** from Turn 1 to create natural conversation flow
- **Ask for it back** in Turn 2 (system question)
- **User provides it** in Turn 3
- **Goal**: Break down complex requests into natural steps

**EXAMPLES:**
For example, if the function is `get_weather(city, unit)` with both parameters required, and the typed query is "what's the weather in Toronto tomorrow in Celsius?", we could break it down into the following turns:
- Turn 1, User: "What's the weather in my city tomorrow?"    (omit Toronto and Celsius at first, both are critical parameters, so the assistant must ask for them)
- Turn 2, Assistant: "What city are you in?"   (the assistant then asks information that was omitted)
- Turn 3, User: "I'm in Toronto."  (the user only provides the city that assistant asked for)
- Turn 4, Assistant: "Great. What unit would you like the temperature in?"  (the assistant then asks other information that was omitted)
- Turn 5, User: "Celsius."  (the user provides the information that was omitted, either all in once, or across multiple turns)

For the same example, we can also break it down into the following turns:
- Turn 1, User: "What's the weather in my city tomorrow?"  
- Turn 2, Assistant: "What city are you in?"  (same first two turns as above)
- Turn 3, User: "I'm in Toronto. I want it in Celsius."  (the user provides the information that was omitted, either all in once, or across multiple turns)

**Important rule about multi-value parameters**
If a single function parameter requires several values (e.g. multiple cities, multiple people, etc.) treat those values as a bundle. **Keep them together when you hide them** – do **not** reveal one value while postponing the others. Because once the assistant already has every required parameter value it could legitimately perform the function call, leaving no opportunity for the user to supply the remaining values. For example, if the user original query is "what's the weather in Toronto and Vancouver tomorrow?", the split CANNOT be like "what's the weather in Toronto tomorrow?" or "What's the weather in my city tomorrow?" followed by "I'm talking about Vancouver".




**TYPE 2: Spoken Query Spelling Confirmation**
Because the spoken query can often be misheard, or might be ambiguous, the assistant should always be cautious about these named entities, like names, IDs, addresses, etc. The assistant should always double check the information with the user, instead of blindly trusting the asr module output and call the function directly. This split doesn't require you to modify the user query, but you need to add in extra follow up turns to ask for clarification. You should ask for clarification on all named entities that might be misheard in a audio setting.
- **Keep the user query** as is, but add in turns to ask for clarification
- **Ask for clarification** in Turn 2 (exact spelling, format, etc.)
- **User clarifies** in Turn 3
- **Goal**: Handle speech-to-text issues and unclear references


For example, if the function is `send_message(recipient, message_body)` and the typed query is "send a message to my friend John, say that I will be 10 minutes late", notice that the named entity could be misheard as either "John" or "Jon". So the spelling confirmation could look like this:
- Turn 1, User: "send a message to my friend John, say that I will be 10 minutes late"
- Turn 2, Assistant: "Could you please help me spell that name? Is it J-O-H-N, or was there a different spelling?"
- Turn 3, User: "Yes, that's correct: J-O-H-N"




You should combine and execute both types of conversation splits together when possible. The goal is to create natural conversation flow. So only perform the split when it makes sense.
For the same example, the combined conversation flow could look like this:
- Turn 1, User: "send a message to my friend, say that I will be 10 minutes late"
- Turn 2, Assistant: "Who would you like to send the message to?"  (omit information)
- Turn 3, User: "My friend John."
- Turn 4, Assistant: "Got it. Could you please help me spell that name? Is it J-O-H-N, or was there a different spelling?"  (ask for clarification)
- Turn 5, User: "Yes, that's correct: J-O-H-N"


**Important:**
1. Keep in mind that you are only "rephrasing" the user query into a multi-turn style. You should never add in any new details/information, nor remove any information without supplying it back.
2. You should only perform the split when it makes sense. Do not split if it's not necessary; in those cases, just return the original user query.
3. You should try your best to create a natural conversation flow. Avoid unnatural queries like "What's the weather in a specific city tomorrow?" (because you won't never say this in a real conversation).


Notes on input: 
The typed user query is user's real intend. The named entities pairs are some common named entities that we have pre-extracted for you, which should be good starting point for the type 1 split.


Return the final multi-turn conversation in the following JSON format; the output must be a list of dictionaries, each with a "role" and "content" key. The role should alternate between user and assistant, and the last one must be from the user.
{"conversation": [
    {
        "role": "user",
        "content": "..."
    },
    {
        "role": "assistant",
        "content": "..."
    },
    ...
]}
"""


def _query_openai(messages):
    response = client.chat.completions.create(
        model="o3-2025-04-16",
        messages=messages,
        response_format={"type": "json_object"},
    )

    result = response.choices[0].message.content.strip()
    result = json.loads(result)
    return result


def _format_instruction(user_query, function_doc, clarifications):

    template = Template(
        """**User Typed Query:**
$user_query

**Function Documentation:**
$function_doc

**Named Entities Pairs:**
$clarifications
"""
    )

    return template.substitute(
        user_query=user_query,
        function_doc=function_doc,
        clarifications=clarifications,
    )


def _process_item(item):
    """Process a single data item and return the updated version."""

    assert len(item["messages"]) == 2
    user_msg = item["messages"][0]
    user_typed_query = user_msg["content"]
    clarifications = user_msg["clarifications"]
    function_doc = item["tools"]

    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": _format_instruction(user_typed_query, function_doc, clarifications)},
    ]

    query_result = _query_openai(messages)

    # Logging for debugging purposes
    # print("-" * 100)
    # print(user_typed_query)
    # print("*" * 100)
    # print(query_result)
    # print("-" * 100)

    rephrased_conversation = query_result["conversation"]
    item["original_query"] = user_typed_query
    item["messages"] = rephrased_conversation + item["messages"][1:]

    return item


# Use multithreading to speed up the processing of all items.
results = []

# Adjust max_workers according to your system and rate-limit considerations.
MAX_WORKERS = 200

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    future_to_item = {executor.submit(_process_item, itm): itm for itm in data}

    # Initialize a tqdm progress bar
    with tqdm(total=len(data), desc="Processing", unit="item") as progress:
        for future in as_completed(future_to_item):
            try:
                processed_item = future.result()
                results.append(processed_item)
            except Exception as exc:
                print(f"Item processing generated an exception: {exc}")
            finally:
                progress.update(1)


print(len(results))

write_list_of_dicts_to_file(
    results, "/Users/hans/repo/API-Gen/xlam_function_calling_60k_rephrased.json"
)
