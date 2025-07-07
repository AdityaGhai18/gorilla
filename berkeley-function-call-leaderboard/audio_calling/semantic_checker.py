import openai
import numpy as np
import re
import json
import os
from typing import List
from openai import OpenAI
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder
import numpy as np

load_dotenv()

def normalize_text(text: str) -> str:
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

if __name__ == "__main__":

    json_path = "../data/possible_answer/BFCL_v3_live_simple.json"
    test_case_id = "live_simple_2-2-0"
    function_arg = "loc"
    candidate = "twenty-twenty Addison St, berkeley"  #edit this to test
    threshold = 0.82  # this could be trained parameter maybe? not sure this would have to be some sort of human input

    print("Semantic Checker Test\n---------------------")
    print(f"JSON Path: {json_path}")
    print(f"Test Case ID: {test_case_id}")
    print(f"Argument Name: {function_arg}")
    print(f"Candidate: {candidate}")
    print(f"Threshold: {threshold}\n")
    test_semantic_checker_from_json(json_path, test_case_id, function_arg, candidate, threshold) 