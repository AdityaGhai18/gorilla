import openai
import numpy as np
import re
import json
import os
from typing import List, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder

# Gemini integration for google-genai SDK
try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

load_dotenv()

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
        "../data/BFCL_v3_live_simple.json",
        "../data/BFCL_v3_live_multiple.json"
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
    Uses a transformer model to compute similarity between two strings.
    Returns a score between 0 and 1.
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



def build_llm_judge_prompt_context(
    function_name: str,
    function_description: str,
    argument_name: str,
    argument_description: str,
    expected_values: list,
    candidate_value: str
) -> str:
    """
    Build a comprehensive, research-backed prompt for LLM-as-a-judge evaluation with full context.
    Combines structured component analysis with clear functional equivalence criteria.
    """
    prompt = f"""
You are an impartial expert judge evaluating semantic equivalence for function call arguments. Your task is to determine if a candidate argument value would produce the same function behavior as an expected value.

CRITICAL PRINCIPLES:
- Focus on FUNCTIONAL EQUIVALENCE: Would both values cause identical function behavior and output?
- Use ONLY the provided function and argument documentation to decide what details and information are critical and what level of specificity is required in the argument value
- Do NOT assume unspecified behavior or fill in missing information
- DISTINGUISH between format requirements (how the data should be presented) and functional requirements (what the function needs to work correctly)
- If a function can work correctly with either format, they are functionally equivalent
- MOST IMPORTANT: If the function would produce the same result with either value, they are equivalent regardless of formatting differences

Function: {function_name}
Function Description: {function_description}
Argument: {argument_name}
Argument Description: {argument_description}
Expected Values: {expected_values}
Candidate Value: "{candidate_value}"

CONTEXT-AWARE STRICTNESS GUIDANCE:
Based on the function documentation above, determine the appropriate strictness level:

If the function requires a specific format for technical or operational reasons, enforce that format strictly.

Otherwise, focus on whether the candidate and expected value would cause the function to behave identically, regardless of phrasing, abbreviations, or omitted non-essential details.

Allow differences in formatting, abbreviations, or omissions if and only if they do not introduce ambiguity or change the function's output.

When in doubt, prioritize functional equivalence over presentational differences.

WHEN FUNCTION DOCUMENTATION IS LIMITED:
If the function documentation is minimal or not found, default to "medium_strictness" and focus on whether the values would produce the same functional outcome. Consider the function name and argument name for context clues about the expected behavior.

SEARCH FUNCTION GUIDANCE:
For search functions (search, find, query), focus on whether the search terms would return the same type of results. Descriptive variations of the same concept should be considered equivalent if they would lead to the same search results.

IMPORTANT: For search functions, additional descriptive words that don't change the core concept should be considered equivalent. Focus on semantic meaning, not exact keyword matching.

INTENT VS DETAIL GUIDANCE:
- For logging/classification/transfer-to-human functions: Only the intent of the query matters for equivalence. Minor differences in phrasing or grammar, do not affect equivalence.
- For quantities, measurements, and precise data (addresses, dates, IDs, numbers, coordinates): Core components must match exactly, but minor formatting differences (abbreviations, missing optional parts) are acceptable if the core meaning is preserved.
- If both values would lead to the same function output (e.g., same category, same escalation, same location), they are equivalent.

MATCHING LOGIC:
- If there are multiple expected values, the candidate should be considered equivalent if it matches ANY ONE of the expected values
- Do not require the candidate to match all expected values (in the case of multiple expected values), only one needs to match to return equivalent
- Compare the candidate to each expected value individually using the strictness level determined for this function

STEP-BY-STEP ANALYSIS (Chain-of-Thought Reasoning):

Step 1: STRICTNESS ASSESSMENT
- Question: Based on the function documentation, what level of strictness and specificity is appropriate for this function?
- Reasoning: Analyze what the function cares about (intent vs. exact details vs. format). Default to "medium_strictness" unless the function documentation explicitly requires exact format for technical or operational reasons.
- Decision: "high_strictness" / "medium_strictness" / "low_strictness" with reasoning

Step 2: INDIVIDUAL COMPARISON ANALYSIS
For each expected value, compare it to the candidate value INDIVIDUALLY. You must check each expected value separately:

Step 2a: SEMANTIC INTENT ANALYSIS
- Question: Do the candidate and this expected value express the same core meaning and purpose?
- Reasoning: Consider what the function does and whether both values serve the same purpose
- Decision: true/false with specific reasoning

Step 2b: FORMAT VS FUNCTION ANALYSIS
- Question: Are the format differences between the values functionally significant, or are they just presentational differences that don't affect the function's ability to work correctly?
- Reasoning: Analyze whether the function would behave identically with either format, or if the differences would cause different behavior
- Decision: true/false with specific reasoning

Step 2c: CRITICAL DETAILS ANALYSIS  
- Question: Do all function-critical components match at the appropriate strictness and specificity level?
- Reasoning: Check each critical detail against the function's requirements using the determined strictness level. Allow abbreviations and omissions if they do not introduce ambiguity or change the function's output. Focus on whether the differences would cause the function to behave differently.
- Decision: true/false with specific reasoning

Step 2d: SPECIFICITY LEVEL ANALYSIS
- Question: Are both values detailed enough for the function to perform identically at the determined strictness level?
- Reasoning: Assess whether the level of detail is sufficient and consistent
- Decision: true/false with specific reasoning

Step 2e: INDIVIDUAL MATCH RESULT
- If ALL components (intent, format-function, details, specificity) are true for this expected value, this comparison = true
- Otherwise, this comparison = false

Step 3: FINAL JUDGMENT
- Check each individual comparison result from Step 2e
- If ANY expected value comparison resulted in true, the overall judgment is true
- If ALL expected value comparisons resulted in false, the overall judgment is false
- IMPORTANT: You only need ONE match to return true

DECISION CRITERIA:
- true: Both values would cause exactly the same function behavior and output (regardless of formatting differences)
- false: Values would cause different function behavior or output
- REMEMBER: If the function would find the same ride, return the same data, or produce identical results with either value, they are equivalent

IMPORTANT: Output ONLY valid JSON. Do not include any reasoning or explanation outside the JSON block. If you cannot fit all reasoning, prioritize completing the JSON.

Output in JSON:
{{
  "strictness_level": "high_strictness|medium_strictness|low_strictness",
  "should_match": true/false,
  "confidence": 0.0-1.0,
  "explanation": "Step 3 summary: [concise final reasoning based on the strictness level and individual comparisons]"
}}
"""
    return prompt.strip()

def parse_llm_judge_response(response_text: str):
    """
    Parse the LLM's JSON output robustly with fallback for truncated responses.
    """
    try:
        # Try to find and parse complete JSON
        start = response_text.find('{')
        end = response_text.rfind('}') + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON found in response")
        
        json_str = response_text[start:end]
        parsed = json.loads(json_str)
        
        # Ensure required fields exist
        if 'should_match' not in parsed:
            raise ValueError("Missing 'should_match' field")
        if 'confidence' not in parsed:
            parsed['confidence'] = 0.5  # Default confidence
        if 'explanation' not in parsed:
            parsed['explanation'] = response_text
        return parsed
    except Exception as e:
        # Fallback: try to extract should_match and confidence from raw text
        print(f'[Warning] Failed to parse LLM output: {e}')
        print(f'Raw output: {response_text}')
        import re
        def extract_json_field(text, field):
            match = re.search(rf'"{field}"\s*:\s*(true|false|[0-9.]+)', text)
            if match:
                val = match.group(1)
                if val in ['true', 'false']:
                    return val == 'true'
                try:
                    return float(val)
                except:
                    return val
            return None
        should_match = extract_json_field(response_text, 'should_match')
        confidence = extract_json_field(response_text, 'confidence')
        if should_match is None:
            should_match = False
        if confidence is None:
            confidence = 0.5
        return {
            "should_match": should_match,
            "confidence": confidence,
            "explanation": f"Parsed from truncated output. Original error: {e}\nRaw output: {response_text}"
        }

def llm_judge_openai(
    prompt: str,
    model: str = "gpt-4-turbo",
    temperature: float = 0.0,
    max_tokens: int = 1024
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
    max_tokens: int = 1024
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
    Majority voting: true = True, false = False. Tie = False.
    """
    results = []
    provider_names = []
    
    for judge_fn in judge_functions:
        result = judge_fn(prompt)
        results.append(result)
        
        # Determine provider name based on function
        if judge_fn.__name__ == 'llm_judge_openai':
            provider_names.append('OpenAI')
        elif judge_fn.__name__ == 'llm_judge_gemini':
            provider_names.append('Google')
        else:
            provider_names.append('Unknown')

    should_matches = [r.get('should_match', False) for r in results]
    confidence_scores = [float(r.get('confidence', 0)) for r in results]
    explanations = [r.get('explanation', '') for r in results]
    
    # Create provider-specific results
    provider_results = {}
    for i, (provider, should_match, confidence) in enumerate(zip(provider_names, should_matches, confidence_scores)):
        provider_results[provider] = {
            'should_match': should_match,
            'confidence': confidence,
            'explanation': explanations[i]
        }

    # Majority voting logic
    true_count = sum(should_matches)
    false_count = len(should_matches) - true_count
    
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
        "final_should_match": final_bool,
        "ensemble_confidence": weighted_conf,
        "all_should_matches": should_matches,
        "all_confidences": confidence_scores,
        "explanations": explanations,
        "aggregate_explanation": agg_explanation,
        "final_bool": final_bool,
        "provider_results": provider_results
    }

# Example usage function for the checker:
def _get_default_judge_functions():
    """Get default judge functions (OpenAI + Gemini if available)."""
    judge_functions = [llm_judge_openai]
    if GEMINI_AVAILABLE:
        judge_functions.append(llm_judge_gemini)
    return judge_functions

def llm_as_judge_semantic_check_context(
    function_name: str,
    function_description: str,
    argument_name: str,
    argument_description: str,
    expected_values: list,
    candidate_value: str
) -> Dict[str, Any]:
    """
    Use LLM(s) as a judge for function call argument equivalence with full context.
    Returns the ensemble result.
    """
    prompt = build_llm_judge_prompt_context(
        function_name, function_description, argument_name, argument_description,
        expected_values, candidate_value
    )
    return ensemble_llm_judges(prompt, _get_default_judge_functions())

def build_llm_judge_prompt_no_context(
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
  "should_match": true/false,
  "confidence": 0.0-1.0,
  "explanation": "Step-by-step reasoning."
}}
"""
    return prompt.strip()

def llm_as_judge_semantic_check_no_context(
    argument_name: str,
    expected_value: str,
    candidate_value: str
) -> Dict[str, Any]:
    """
    Use LLM(s) as a judge for function call argument equivalence without context.
    Returns the ensemble result.
    """
    prompt = build_llm_judge_prompt_no_context(
        argument_name,
        expected_value,
        candidate_value
    )
    return ensemble_llm_judges(prompt, _get_default_judge_functions())