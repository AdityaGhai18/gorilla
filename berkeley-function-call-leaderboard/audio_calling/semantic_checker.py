import openai
import numpy as np
import re
import json
import os
from typing import List, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder, SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Gemini integration for google-genai SDK
try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

load_dotenv()

print("OPENAI_API_KEY loaded:", bool(os.getenv("OPENAI_API_KEY")))
print("GOOGLE_API_KEY loaded:", bool(os.getenv("GOOGLE_API_KEY")))

def normalize_text(text: str) -> str:
    text = str(text)
    text = text.lower()
    text = re.sub(r"[\s\-_,./\\*^]+", " ", text)  # Remove/replace common punctuations
    text = re.sub(r"'s\b", "", text)  # Remove possessive 's
    text = re.sub(r"\s+", " ", text)  # Collapse whitespace
    return text.strip()


def get_openai_embedding(text: str, model: str = "text-embedding-ada-002") -> np.ndarray:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    client = OpenAI(api_key=api_key)
    response = client.embeddings.create(input=[text], model=model)
    return np.array(response.data[0].embedding)

def get_function_context(function_name: str, arg_name: str) -> str:
    """
    Get function documentation context for better semantic understanding.
    Returns a string with function description and parameter description.
    """
    # Try to find function doc in various locations - will be dependent on where we test but easier to check all
    doc_paths = [
        "../data/multi_turn_func_doc/",
        "../data/",
        "../data/BFCL_v3_live_simple.json"
    ]
    
    for base_path in doc_paths:
        if os.path.exists(base_path):
            # Look for function docs in the directory
            if os.path.isdir(base_path):
                for filename in os.listdir(base_path):
                    if filename.endswith('.json'):
                        try:
                            with open(os.path.join(base_path, filename), 'r') as f:
                                data = json.load(f)
                                if isinstance(data, list):
                                    for item in data:
                                        if item.get('name') == function_name:
                                            desc = item.get('description', '')
                                            params = item.get('parameters', {}).get('properties', {})
                                            param_desc = params.get(arg_name, {}).get('description', '')
                                            return f"Function: {desc}. Parameter {arg_name}: {param_desc}"
                        except:
                            continue
            else:
                # Single file
                try:
                    with open(base_path, 'r') as f:
                        lines = f.readlines()
                        for line in lines:
                            try:
                                item = json.loads(line)
                                if item.get('function') and isinstance(item['function'], list):
                                    for func in item['function']:
                                        if func.get('name') == function_name:
                                            desc = func.get('description', '')
                                            params = func.get('parameters', {}).get('properties', {})
                                            param_desc = params.get(arg_name, {}).get('description', '')
                                            return f"Function: {desc}. Parameter {arg_name}: {param_desc}"
                            except:
                                continue
                except:
                    continue
    
    # Fallback: return basic context
    return f"Function: {function_name}, Parameter: {arg_name}"

def compute_similarities_and_embeddings(candidate: str, possible_answers: list[str]):
    """
    Normalize and embed candidate and possible answers, compute cosine similarities.
    Returns: (candidate_norm, answers_norm, cand_emb, answers_emb, similarities)
    """
    candidate_norm = normalize_text(candidate)
    answers_norm = [normalize_text(ans) for ans in possible_answers]
    cand_emb = get_openai_embedding(candidate_norm)
    answers_emb = [get_openai_embedding(ans) for ans in answers_norm]
    similarities = [cosine_similarity(cand_emb, ans_emb) for ans_emb in answers_emb]
    return candidate_norm, answers_norm, cand_emb, answers_emb, similarities

def compute_contextual_similarities_and_embeddings(candidate: str, possible_answers: list[str], function_name: str, arg_name: str):
    """
    Contextual version: include function documentation in embeddings. docs + candidate value, docs + answer value comparison
    Returns: (candidate_norm, answers_norm, cand_emb, answers_emb, similarities)
    """
    function_context = get_function_context(function_name, arg_name)
    
    # Create contextual strings
    candidate_contextual = f"{function_context}. Value: {candidate}"
    #candidate + context
    #Function: Finds a suitable Uber ride for customers based on their location, desired ride type, and maximum wait time.. Parameter loc: The starting location for the Uber ride, in the format of 'Street Address, City, State (abbr), Country'.. Value: twenty-twenty Addison St, berkeley
    
    #answer + context
    #"Function: Finds a suitable Uber ride for customers based on their location, desired ride type, and maximum wait time.. Parameter loc: The starting location for the Uber ride, in the format of 'Street Address, City, State (abbr), Country'.. Value: 2020 Addison Street, Berkeley, CA, USA"
    
    answers_contextual = [f"{function_context}. Value: {ans}" for ans in possible_answers]

    # Normalize and embed
    candidate_norm = normalize_text(candidate_contextual)
    answers_norm = [normalize_text(ans) for ans in answers_contextual]
    cand_emb = get_openai_embedding(candidate_norm)
    answers_emb = [get_openai_embedding(ans) for ans in answers_norm]
    similarities = [cosine_similarity(cand_emb, ans_emb) for ans_emb in answers_emb]
    
    return candidate_norm, answers_norm, cand_emb, answers_emb, similarities

def cross_encoder_similarity(candidate: str, possible_answer: str, function_name: str, arg_name: str) -> float:
    """
    the architecture is a transformer model that takes in two strings and outputs a similarity score.
    the score is a number between 0 and 1, where 1 means the two strings are exactly the same, and 0 means they are completely different.
    """

    # Load a cross-encoder model
    model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    # Create the input pair - put context once, then both values
    function_context = get_function_context(function_name, arg_name)
    text1 = f"{function_context}. Value 1: {candidate}"
    text2 = f"{function_context}. Value 2: {possible_answer}"
    
    # Normalize both texts (consistent with other approaches)
    text1_norm = normalize_text(text1)
    text2_norm = normalize_text(text2)
    
    # Get similarity score - pass as tuple
    scores = model.predict([(text1_norm, text2_norm)])
    raw_score = float(scores[0])
    
    # Normalize to 0-1 range using sigmoid (cross-encoders often output unbounded scores)
    normalized_score = 1 / (1 + np.exp(-raw_score))
    return normalized_score

def compute_cross_encoder_similarities(candidate: str, possible_answers: list[str], function_name: str, arg_name: str):
    """
    Compute similarities using cross-encoder approach.
    Returns: similarities list
    """
    similarities = []
    for answer in possible_answers:
        sim = cross_encoder_similarity(candidate, answer, function_name, arg_name)
        similarities.append(sim)
    return similarities

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

def is_semantically_similar_from_similarities(similarities, threshold: float = 0.82) -> bool:
    """
    Returns True if any similarity is above the threshold.
    """
    return any(sim >= threshold for sim in similarities)

def test_semantic_checker_from_json(
    json_path: str,
    test_case_id: str,
    function_arg: str,
    candidate: str,
    threshold: float = 0.82
):
    """
    Loads a possible answer JSON, finds the test case by id, extracts the possible answers for the given argument,
    and compares the candidate string to those answers using the semantic checker.
    """
    # Load JSON file (assume one object per line or a list of objects)
    with open(json_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    # Try to parse as list or as objects per line
    try:
        data = json.loads(''.join(lines))
        if isinstance(data, dict):
            data = [data]
    except Exception:
        data = [json.loads(line) for line in lines if line.strip()]

    # Find the test case
    entry = next((item for item in data if item.get('id') == test_case_id), None)
    if not entry:
        print(f"Test case id '{test_case_id}' not found in {json_path}.")
        return
    gt = entry.get('ground_truth', [])
    if not gt or not isinstance(gt, list) or not isinstance(gt[0], dict):
        print(f"No valid ground_truth for id '{test_case_id}'.")
        return
    func_dict = gt[0]
    if not func_dict:
        print(f"No function call found in ground_truth for id '{test_case_id}'.")
        return
    # Get the first function call
    func_args = list(func_dict.values())[0]
    if function_arg not in func_args:
        print(f"Argument '{function_arg}' not found in function call for id '{test_case_id}'.")
        return
    possible_answers = func_args[function_arg]
    function_name = list(func_dict.keys())[0]
    
    print(f"Candidate: {candidate}")
    print(f"Possible Answers: {possible_answers}")
    print(f"Function: {function_name}")
    print(f"Argument: {function_arg}")
    print(f"Threshold: {threshold}")

    # Test 1: Basic bi-encoder
    print("\n=== BASIC BI-ENCODER ===")
    candidate_norm, answers_norm, cand_emb, answers_emb, similarities = compute_similarities_and_embeddings(candidate, possible_answers)
    for ans, sim in zip(possible_answers, similarities):
        print(f"Similarity to '{ans}': {sim:.4f}")
    result = is_semantically_similar_from_similarities(similarities, threshold)
    print(f"Basic Bi-Encoder Match: {result}")

    # Test 2: Contextual bi-encoder
    print("\n=== CONTEXTUAL BI-ENCODER ===")
    try:
        function_context = get_function_context(function_name, function_arg)
        print(f"Function Context: {function_context}")
        candidate_norm, answers_norm, cand_emb, answers_emb, contextual_similarities = compute_contextual_similarities_and_embeddings(candidate, possible_answers, function_name, function_arg)
        for ans, sim in zip(possible_answers, contextual_similarities):
            print(f"Contextual similarity to '{ans}': {sim:.4f}")
        contextual_result = is_semantically_similar_from_similarities(contextual_similarities, threshold)
        print(f"Contextual Bi-Encoder Match: {contextual_result}")
    except Exception as e:
        print(f"Contextual bi-encoder failed: {e}")

    # Test 3: Cross-encoder
    print("\n=== CROSS-ENCODER ===")
    try:
        cross_similarities = compute_cross_encoder_similarities(candidate, possible_answers, function_name, function_arg)
        for ans, sim in zip(possible_answers, cross_similarities):
            print(f"Cross-encoder similarity to '{ans}': {sim:.4f}")
        cross_result = is_semantically_similar_from_similarities(cross_similarities, threshold)
        print(f"Cross-Encoder Match: {cross_result}")
    except Exception as e:
        print(f"Cross-encoder failed: {e}")

def build_llm_judge_prompt(
    function_name: str,
    function_description: str,
    argument_name: str,
    argument_description: str,
    expected_value: str,
    candidate_value: str
) -> str:
    """
    Build a structured, context-rich prompt for LLM-as-a-judge evaluation.
    """
    prompt = f"""
You are an expert judge of semantic equivalence for function call arguments.

Your task is to decide if the candidate argument value is functionally equivalent to the expected value, given the function and argument descriptions.

- Focus on whether the candidate would work the same as the expected value for the function, not on grammar, spelling, or politeness, unless those are critical for the function to work.
- If the argument is free text, focus on the intended meaning.
- If the argument is structured (like an address), focus on whether all critical components are present and correct.
- If the candidate value is missing critical information, it is NOT equivalent.
- If the candidate value is less formal or contains errors but would still work in the function, it IS equivalent.

**Step-by-step:**
1. List the critical components of the argument, based on the argument description.
2. Compare the expected and candidate values for each component.
3. Decide: Equivalent, Not Equivalent, or Partial (if some but not all critical parts are present).
4. Output in JSON:
{{
  "judgment": "Equivalent | Not Equivalent | Partial",
  "confidence": 0.0-1.0,
  "explanation": "Step-by-step reasoning referencing argument criticality and function requirements."
}}

Function: {function_name}
Function Description: {function_description}
Argument: {argument_name}
Argument Description: {argument_description}
Expected Value: "{expected_value}"
Candidate Value: "{candidate_value}"
"""
    return prompt.strip()

def parse_llm_judge_response(response_text: str):
    """
    Parse the LLM's JSON output robustly. Always try to extract 'explanation'.
    """
    try:
        start = response_text.find('{')
        end = response_text.rfind('}') + 1
        json_str = response_text[start:end]
        parsed = json.loads(json_str)
        if 'explanation' not in parsed:
            print('[Warning] LLM response missing explanation. Raw output:')
            print(response_text)
            parsed['explanation'] = response_text
        return parsed
    except Exception as e:
        print(f'[Warning] Failed to parse LLM output: {e}\nRaw output: {response_text}')
        return {
            "judgment": "Error",
            "confidence": 0.0,
            "explanation": f"Failed to parse LLM output: {e}\nRaw output: {response_text}"
        }

def llm_judge_openai(
    prompt: str,
    model: str = "gpt-4-turbo",
    temperature: float = 0.0,
    max_tokens: int = 512
):
    """
    Query OpenAI LLM as a judge (compatible with openai>=1.0.0).
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return parse_llm_judge_response(response.choices[0].message.content)

def llm_judge_gemini(
    prompt: str,
    model: str = "gemini-1.5-pro",
    temperature: float = 0.0,
    max_tokens: int = 512
):
    """
    Query Google Gemini as a judge (using the google-genai SDK).
    Reference: https://ai.google.dev/gemini-api/docs/get-started/python
    """
    if not GEMINI_AVAILABLE:
        return {
            "judgment": "Error",
            "confidence": 0.0,
            "explanation": "Google Gemini API not available."
        }
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return {
            "judgment": "Error",
            "confidence": 0.0,
            "explanation": "GOOGLE_API_KEY not found in environment variables."
        }
    client = genai.Client(api_key=api_key)
    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
        )
        return parse_llm_judge_response(response.text)
    except Exception as e:
        return {
            "judgment": "Error",
            "confidence": 0.0,
            "explanation": f"Gemini API call failed: {e}"
        }

from collections import Counter

def ensemble_llm_judges(
    prompt: str,
    judge_functions: list,
    weights: list = None
):
    """
    Run multiple LLM judges and aggregate their decisions.
    Majority voting: 'Equivalent' or 'Partial' = True, 'Not Equivalent' = False. Tie = False.
    """
    results = []
    for judge_fn in judge_functions:
        result = judge_fn(prompt)
        results.append(result)

    judgments = [r['judgment'] for r in results]
    confidence_scores = [float(r.get('confidence', 0)) for r in results]
    explanations = [r['explanation'] for r in results]
    majority = Counter(judgments).most_common(1)[0][0]

    # Pass/fail logic
    true_count = sum(j in ('Equivalent', 'Partial') for j in judgments)
    false_count = sum(j == 'Not Equivalent' for j in judgments)
    if true_count > false_count:
        final_bool = True
    elif false_count > true_count:
        final_bool = False
    else:
        final_bool = False  # Tie = fail-safe

    # Weighted confidence (if weights provided)
    if weights:
        weighted_conf = sum(w * c for w, c in zip(weights, confidence_scores)) / sum(weights)
    else:
        weighted_conf = sum(confidence_scores) / len(confidence_scores)

    agg_explanation = "\n---\n".join(explanations)

    return {
        "final_judgment": majority,
        "ensemble_confidence": weighted_conf,
        "all_judgments": judgments,
        "all_confidences": confidence_scores,
        "explanations": explanations,
        "aggregate_explanation": agg_explanation,
        "final_bool": final_bool
    }

# Example usage function for the checker:
def llm_as_judge_semantic_check(
    function_name: str,
    function_description: str,
    argument_name: str,
    argument_description: str,
    expected_value: str,
    candidate_value: str
) -> Dict[str, Any]:
    """
    Use LLM(s) as a judge for function call argument equivalence.
    Returns the ensemble result.
    """
    prompt = build_llm_judge_prompt(
        function_name, function_description, argument_name, argument_description,
        expected_value, candidate_value
    )
    judge_functions = [llm_judge_openai]
    if GEMINI_AVAILABLE:
        judge_functions.append(llm_judge_gemini)
    return ensemble_llm_judges(prompt, judge_functions)

def build_llm_judge_prompt_minimal(
    argument_name: str,
    expected_value: str,
    candidate_value: str
) -> str:
    """
    Build a minimal prompt for LLM-as-a-judge evaluation (no context, just values).
    """
    prompt = f"""
Argument: {argument_name}
Expected Value: "{expected_value}"
Candidate Value: "{candidate_value}"

Task: Are these two values functionally equivalent for their use as an argument in a function call? If not, explain the key differences. Output in JSON:
{{
  "judgment": "Equivalent | Not Equivalent | Partial",
  "confidence": 0.0-1.0,
  "explanation": "Step-by-step reasoning."
}}
"""
    return prompt.strip()

def llm_as_judge_semantic_check_minimal(
    argument_name: str,
    expected_value: str,
    candidate_value: str
) -> Dict[str, Any]:
    """
    Use LLM(s) as a judge for function call argument equivalence (minimal prompt).
    Returns the ensemble result.
    """
    prompt = build_llm_judge_prompt_minimal(
        argument_name,
        expected_value,
        candidate_value
    )
    judge_functions = [llm_judge_openai]
    if GEMINI_AVAILABLE:
        judge_functions.append(llm_judge_gemini)
    return ensemble_llm_judges(prompt, judge_functions)

def build_llm_judge_prompt_general(
    function_name: str,
    function_description: str,
    argument_name: str,
    argument_description: str,
    expected_value: str,
    candidate_value: str
) -> str:
    prompt = f"""
Function: {function_name}
Function Description: {function_description}
Argument: {argument_name}
Argument Description: {argument_description}
Expected Value: "{expected_value}"
Candidate Value: "{candidate_value}"

Task:
1. Based on the argument description, break down the argument value into its logical components (e.g., fields, attributes, or key elements).
2. For each component, state:
   - Whether it matches, is missing, or is different in the candidate compared to the expected value.
   - Whether this component is critical for the function's correct operation, based on the function and argument descriptions.
3. If any critical component is missing or different, return "Not Equivalent".
4. If all critical components match, but there are minor differences in non-critical components, return "Partial".
5. If all components match, return "Equivalent".
6. Provide a confidence score (0.0–1.0) and a step-by-step explanation referencing the function and argument descriptions.

Output (in JSON):
{{
  "components": [
    {{
      "name": "component_name",
      "match": true/false,
      "critical": true/false,
      "notes": "explanation of this component"
    }},
    ...
  ],
  "judgment": "Equivalent | Partial | Not Equivalent",
  "confidence": 0.0–1.0,
  "explanation": "Step-by-step reasoning."
}}
"""
    return prompt.strip()

def llm_as_judge_semantic_check_general(
    function_name: str,
    function_description: str,
    argument_name: str,
    argument_description: str,
    expected_value: str,
    candidate_value: str
) -> dict:
    prompt = build_llm_judge_prompt_general(
        function_name,
        function_description,
        argument_name,
        argument_description,
        expected_value,
        candidate_value
    )
    judge_functions = [llm_judge_openai]
    if GEMINI_AVAILABLE:
        judge_functions.append(llm_judge_gemini)
    return ensemble_llm_judges(prompt, judge_functions)

if __name__ == "__main__":
    print("\n=== Function Call Argument Evaluation: All Approaches ===\n")

    # Load test case from possible answers JSON file
    json_path = "../data/possible_answer/BFCL_v3_live_simple.json"
    test_case_id = "live_simple_2-2-0"
    function_arg = "loc"
    candidate = "2021 Addison St, Berkeley"  # Only this is set here for testing
    threshold = 0.82

    # Load JSON file (assume one object per line or a list of objects)
    with open(json_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    try:
        data = json.loads(''.join(lines))
        if isinstance(data, dict):
            data = [data]
    except Exception:
        data = [json.loads(line) for line in lines if line.strip()]

    # Find the test case
    entry = next((item for item in data if item.get('id') == test_case_id), None)
    if not entry:
        print(f"Test case id '{test_case_id}' not found in {json_path}.")
        exit(1)
    gt = entry.get('ground_truth', [])
    if not gt or not isinstance(gt, list) or not isinstance(gt[0], dict):
        print(f"No valid ground_truth for id '{test_case_id}'.")
        exit(1)
    func_dict = gt[0]
    if not func_dict:
        print(f"No function call found in ground_truth for id '{test_case_id}'.")
        exit(1)
    # Get the first function call
    func_args = list(func_dict.values())[0]
    if function_arg not in func_args:
        print(f"Argument '{function_arg}' not found in function call for id '{test_case_id}'.")
        exit(1)
    possible_answers = func_args[function_arg]
    function_name = list(func_dict.keys())[0]

    # Use get_function_context to robustly fetch function and argument descriptions
    function_context = get_function_context(function_name, function_arg)
    # Parse out function and argument descriptions from the context string
    # Expected format: 'Function: {desc}. Parameter {arg_name}: {arg_desc}'
    if function_context.startswith("Function:"):
        try:
            func_desc_part = function_context.split(". Parameter ")[0]
            function_description = func_desc_part[len("Function: "):].strip()
            if ". Parameter " in function_context:
                arg_desc_part = function_context.split(". Parameter ")[1]
                argument_description = arg_desc_part.strip()
            else:
                argument_description = f"Argument: {function_arg} (description not found)"
        except Exception:
            function_description = function_context
            argument_description = f"Argument: {function_arg} (description not found)"
    else:
        function_description = function_context
        argument_description = f"Argument: {function_arg} (description not found)"

    print(f"Test Case ID: {test_case_id}")
    print(f"Function: {function_name}")
    print(f"Function Description: {function_description}")
    print(f"Argument: {function_arg}")
    print(f"Argument Description: {argument_description}")
    print(f"Possible Answers: {possible_answers}")
    print(f"Candidate: {candidate}")
    print(f"Threshold: {threshold}\n")

    # 1. Basic bi-encoder (no context)
    print("=== BASIC BI-ENCODER (NO CONTEXT) ===")
    candidate_norm, answers_norm, cand_emb, answers_emb, similarities = compute_similarities_and_embeddings(candidate, possible_answers)
    for ans, sim in zip(possible_answers, similarities):
        print(f"Similarity to '{ans}': {sim:.4f}")
    result = is_semantically_similar_from_similarities(similarities, threshold)
    print(f"Basic Bi-Encoder Match: {result}\n")

    # 2. Contextual bi-encoder
    print("=== CONTEXTUAL BI-ENCODER ===")
    try:
        candidate_norm, answers_norm, cand_emb, answers_emb, contextual_similarities = compute_contextual_similarities_and_embeddings(candidate, possible_answers, function_name, function_arg)
        for ans, sim in zip(possible_answers, contextual_similarities):
            print(f"Contextual similarity to '{ans}': {sim:.4f}")
        contextual_result = is_semantically_similar_from_similarities(contextual_similarities, threshold)
        print(f"Contextual Bi-Encoder Match: {contextual_result}\n")
    except Exception as e:
        print(f"Contextual bi-encoder failed: {e}\n")

    # 3. Cross-encoder
    print("=== CROSS-ENCODER ===")
    try:
        cross_similarities = compute_cross_encoder_similarities(candidate, possible_answers, function_name, function_arg)
        for ans, sim in zip(possible_answers, cross_similarities):
            print(f"Cross-encoder similarity to '{ans}': {sim:.4f}")
        cross_result = is_semantically_similar_from_similarities(cross_similarities, threshold)
        print(f"Cross-Encoder Match: {cross_result}\n")
    except Exception as e:
        print(f"Cross-encoder failed: {e}\n")

    # 4. LLM-as-a-Judge (context-rich)
    print("=== LLM-AS-A-JUDGE (CONTEXT-RICH, LLM INFERS CRITICALITY) ===")
    if not GEMINI_AVAILABLE:
        print("[Warning] Google Gemini API not available. Only OpenAI will be used.\n")
    result_context = llm_as_judge_semantic_check(
        function_name=function_name,
        function_description=function_description,
        argument_name=function_arg,
        argument_description=argument_description,
        expected_value=possible_answers[0],  # Use the first possible answer for LLM judge
        candidate_value=candidate
    )
    print(f"Final Judgment: {result_context['final_judgment']}")
    print(f"Ensemble Confidence: {result_context['ensemble_confidence']:.2f}")
    print("\nAll Judgments:")
    for i, (judgment, conf) in enumerate(zip(result_context['all_judgments'], result_context['all_confidences'])):
        print(f"  Model {i+1}: {judgment} (Confidence: {conf:.2f})")
    print("\nExplanations:")
    for i, explanation in enumerate(result_context['explanations']):
        print(f"--- Model {i+1} Explanation ---\n{explanation}\n")

    # 5. LLM-as-a-Judge (minimal prompt)
    print("=== LLM-AS-A-JUDGE (MINIMAL PROMPT, NO CONTEXT) ===")
    result_minimal = llm_as_judge_semantic_check_minimal(
        argument_name=function_arg,
        expected_value=possible_answers[0],
        candidate_value=candidate
    )
    print(f"Final Judgment: {result_minimal['final_judgment']}")
    print(f"Ensemble Confidence: {result_minimal['ensemble_confidence']:.2f}")
    print("\nAll Judgments:")
    for i, (judgment, conf) in enumerate(zip(result_minimal['all_judgments'], result_minimal['all_confidences'])):
        print(f"  Model {i+1}: {judgment} (Confidence: {conf:.2f})")
    print("\nExplanations:")
    for i, explanation in enumerate(result_minimal['explanations']):
        print(f"--- Model {i+1} Explanation ---\n{explanation}\n")

    # 6. LLM-as-a-Judge (generalized, component analysis)
    print("=== LLM-AS-A-JUDGE (GENERALIZED, COMPONENT ANALYSIS) ===")
    result_general = llm_as_judge_semantic_check_general(
        function_name=function_name,
        function_description=function_description,
        argument_name=function_arg,
        argument_description=argument_description,
        expected_value=possible_answers[0],
        candidate_value=candidate
    )
    # Print per-component analysis if present
    components = result_general.get("components")
    if components:
        print("Components:")
        for comp in components:
            print(f"  - name: {comp.get('name')}, match: {comp.get('match')}, critical: {comp.get('critical')}, notes: {comp.get('notes')}")
    print(f"Final Judgment: {result_general.get('final_judgment', result_general.get('judgment'))}")
    print(f"Confidence: {result_general.get('ensemble_confidence', result_general.get('confidence', 0.0))}")
    print(f"Explanation: {result_general.get('explanation', '')}")
    print("=== End of Evaluation ===\n")