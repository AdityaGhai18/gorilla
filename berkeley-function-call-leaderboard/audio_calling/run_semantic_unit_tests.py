import os
import json
from semantic_checker import (
    get_function_context,
    compute_similarities_and_embeddings,
    compute_contextual_similarities_and_embeddings,
    compute_cross_encoder_similarities,
    is_semantically_similar_from_similarities,
    llm_as_judge_semantic_check_context,
    llm_as_judge_semantic_check_no_context
)

"""
Semantic argument evaluation approaches:

bi_encoder: Compares candidate and reference using OpenAI embeddings and cosine similarity; context-agnostic.
contextual_bi_encoder: Same as bi-encoder, but includes function and argument descriptions in the embeddings for context.
cross_encoder: Uses a transformer model to jointly encode candidate and reference with context, outputting a similarity score.
llm_context_rich: Uses an LLM (e.g., GPT-4) with full function and argument context to judge functional equivalence, with component analysis.
llm_minimal: Uses an LLM with only the argument name and values, ignoring all context, to judge equivalence.
"""

UNIT_TEST_PATH = os.path.join(os.path.dirname(__file__), 'unit_test_cases.json')
POSSIBLE_ANSWER_PATHS = [
    os.path.join(os.path.dirname(__file__), '../data/possible_answer/BFCL_v3_live_simple.json'),
    os.path.join(os.path.dirname(__file__), '../data/possible_answer/BFCL_v3_live_multiple.json')
]
RESULTS_PATH = os.path.join(os.path.dirname(__file__), 'semantic_unit_test_results.json')

THRESHOLD = 0.82

def load_possible_answers(json_paths):
    """
    Load possible answers from multiple JSON files and merge them.
    """
    all_data = []
    for json_path in json_paths:
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            try:
                data = json.loads(''.join(lines))
                if isinstance(data, dict):
                    data = [data]
            except Exception:
                data = [json.loads(line) for line in lines if line.strip()]
            all_data.extend(data)
            print(f"Loaded {len(data)} entries from {os.path.basename(json_path)}")
        except Exception as e:
            print(f"Warning: Could not load {json_path}: {e}")
    return all_data

def get_context_and_answers(entry, function_arg):
    gt = entry.get('ground_truth', [])
    if not gt or not isinstance(gt, list) or not isinstance(gt[0], dict):
        return None, None, None, None, None
    func_dict = gt[0]
    if not func_dict:
        return None, None, None, None, None
    function_name = list(func_dict.keys())[0]
    func_args = list(func_dict.values())[0]
    possible_answers = func_args.get(function_arg, [])

    function_context = get_function_context(function_name, function_arg)
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
    return function_name, function_description, argument_description, possible_answers, function_context

def main():
    # Flags to control which methods to run
    RUN_BI_ENCODER = False
    RUN_CONTEXTUAL_BI_ENCODER = False
    RUN_CROSS_ENCODER = False
    RUN_LLM_CONTEXT_RICH = True
    RUN_LLM_MINIMAL = False
    
    with open(UNIT_TEST_PATH, 'r') as f:
        data = json.load(f)
        test_cases = data.get('test_cases', data)  # Handle both new structure and old array format

    # Run all test cases
    print(f"Running all {len(test_cases)} test cases.")

    possible_answers_data = load_possible_answers(POSSIBLE_ANSWER_PATHS)
    all_results = []
    for i, test in enumerate(test_cases):
        test_id = test['id']
        function_arg = test['function_arg']
        candidate = test['candidate']
        should_match = test['should_match']
        # Find the entry in possible answers
        entry = next((item for item in possible_answers_data if item.get('id') == test_id), None)
        if not entry:
            print(f"[Test {i+1}] ID: {test_id} -- Not found in possible answers. Skipping.\n")
            continue
        function_name, function_description, argument_description, possible_answers, function_context = get_context_and_answers(entry, function_arg)
        if not possible_answers:
            print(f"[Test {i+1}] ID: {test_id} -- No possible answers for argument '{function_arg}'. Skipping.\n")
            continue
        print(f"\n{'='*60}\n[Test {i+1}] ID: {test_id}")
        print(f"Function: {function_name}")
        print(f"Argument: {function_arg}")
        print(f"Should Match: {should_match}")
        print(f"Candidate: {candidate}")
        print(f"Possible Answers: {possible_answers}")
        print(f"Function Description: {function_description}")
        print(f"Argument Description: {argument_description}")
        print(f"Threshold: {THRESHOLD}")
        case_result = {
            'id': test_id,
            'should_match': should_match,
            'bi_encoder': None,
            'contextual_bi_encoder': None,
            'cross_encoder': None,
            'llm_context_rich': None,
            'llm_minimal': None
        }
        
        # 1. Basic bi-encoder
        if RUN_BI_ENCODER:
            print("\n--- BASIC BI-ENCODER ---")
            _, _, _, _, similarities = compute_similarities_and_embeddings(candidate, possible_answers)
            for ans, sim in zip(possible_answers, similarities):
                print(f"Similarity to '{ans}': {sim:.4f}")
            bi_result = is_semantically_similar_from_similarities(similarities, THRESHOLD)
            print(f"Bi-Encoder Match: {bi_result}")
            if bi_result == should_match:
                print("TEST PASSED ✅ (Bi-Encoder)")
                case_result['bi_encoder'] = 'PASS'
            else:
                print("TEST FAILED ❌ (Bi-Encoder)")
                case_result['bi_encoder'] = 'FAIL'

        # 2. Contextual bi-encoder
        if RUN_CONTEXTUAL_BI_ENCODER:
            print("\n--- CONTEXTUAL BI-ENCODER ---")
            try:
                _, _, _, _, contextual_similarities = compute_contextual_similarities_and_embeddings(candidate, possible_answers, function_name, function_arg)
                for ans, sim in zip(possible_answers, contextual_similarities):
                    print(f"Contextual similarity to '{ans}': {sim:.4f}")
                contextual_result = is_semantically_similar_from_similarities(contextual_similarities, THRESHOLD)
                print(f"Contextual Bi-Encoder Match: {contextual_result}")
                if contextual_result == should_match:
                    print("TEST PASSED ✅ (Contextual Bi-Encoder)")
                    case_result['contextual_bi_encoder'] = 'PASS'
                else:
                    print("TEST FAILED ❌ (Contextual Bi-Encoder)")
                    case_result['contextual_bi_encoder'] = 'FAIL'
            except Exception as e:
                print(f"Contextual bi-encoder failed: {e}")
                case_result['contextual_bi_encoder'] = 'ERROR'

        # 3. Cross-encoder
        if RUN_CROSS_ENCODER:
            print("\n--- CROSS-ENCODER ---")
            try:
                cross_similarities = compute_cross_encoder_similarities(candidate, possible_answers, function_name, function_arg)
                for ans, sim in zip(possible_answers, cross_similarities):
                    print(f"Cross-encoder similarity to '{ans}': {sim:.4f}")
                cross_result = is_semantically_similar_from_similarities(cross_similarities, THRESHOLD)
                print(f"Cross-Encoder Match: {cross_result}")
                if cross_result == should_match:
                    print("TEST PASSED ✅ (Cross-Encoder)")
                    case_result['cross_encoder'] = 'PASS'
                else:
                    print("TEST FAILED ❌ (Cross-Encoder)")
                    case_result['cross_encoder'] = 'FAIL'
            except Exception as e:
                print(f"Cross-encoder failed: {e}")
                case_result['cross_encoder'] = 'ERROR'

        # 4. LLM-as-a-Judge (with context)
        if RUN_LLM_CONTEXT_RICH:
            print("\n--- LLM-AS-A-JUDGE (WITH CONTEXT) ---")
            try:
                result_context = llm_as_judge_semantic_check_context(
                    function_name=function_name,
                    function_description=function_description,
                    argument_name=function_arg,
                    argument_description=argument_description,
                    expected_values=possible_answers,
                    candidate_value=candidate
                )
                
                # Display test case expectation
                print(f"Test Case Expectation: should_match = {should_match}")
                
                # Display provider-specific results
                print(f"Ensemble Result: should_match = {result_context['final_should_match']}")
                print(f"Ensemble Confidence: {result_context['ensemble_confidence']:.2f}")
                
                # Show individual provider results
                for provider, provider_data in result_context['provider_results'].items():
                    print(f"{provider} Output: should_match = {provider_data['should_match']} (Confidence: {provider_data['confidence']:.2f})")
                
                # Show detailed explanations with strictness reasoning
                for provider, provider_data in result_context['provider_results'].items():
                    print(f"\n--- {provider} Explanation ---")
                    # Try to extract strictness info if available
                    try:
                        explanation_text = provider_data['explanation']
                        if '"strictness_level"' in explanation_text:
                            # Extract strictness info from JSON in explanation
                            start = explanation_text.find('{')
                            end = explanation_text.rfind('}') + 1
                            if start != -1 and end != 0:
                                json_str = explanation_text[start:end]
                                parsed = json.loads(json_str)
                                if 'strictness_level' in parsed:
                                    print(f"Strictness Level: {parsed['strictness_level']}")
                                    if 'strictness_reasoning' in parsed:
                                        print(f"Strictness Reasoning: {parsed['strictness_reasoning']}")

                    except:
                        pass
                    print(f"Full Explanation:\n{provider_data['explanation']}\n")
                
                llm_context_result = result_context.get('final_bool', None)
                if llm_context_result is None:
                    print("ERROR: Could not determine should_match value")
                    case_result['llm_context_rich'] = 'ERROR'
                elif llm_context_result == should_match:
                    print("✅ CORRECT: LLM output matches test case expectation")
                    case_result['llm_context_rich'] = 'PASS'
                else:
                    print("❌ INCORRECT: LLM output does not match test case expectation")
                    case_result['llm_context_rich'] = 'FAIL'
            except Exception as e:
                print(f"LLM-as-a-Judge (context-rich) failed: {e}")
                case_result['llm_context_rich'] = 'ERROR'

        # 5. LLM-as-a-Judge (no context)
        if RUN_LLM_MINIMAL:
            print("\n--- LLM-AS-A-JUDGE (NO CONTEXT) ---")
            try:
                result_minimal = llm_as_judge_semantic_check_no_context(
                    argument_name=function_arg,
                    expected_value=possible_answers[0],
                    candidate_value=candidate
                )
                print(f"Final Judgment: {result_minimal['final_judgment']}")
                print(f"Ensemble Confidence: {result_minimal['ensemble_confidence']:.2f}")
                for i, (judgment, conf) in enumerate(zip(result_minimal['all_judgments'], result_minimal['all_confidences'])):
                    print(f"  Model {i+1}: {judgment} (Confidence: {conf:.2f})")
                for i, explanation in enumerate(result_minimal['explanations']):
                    print(f"--- Model {i+1} Explanation ---\n{explanation}\n")
                llm_minimal_result = result_minimal.get('final_bool', None)
                if llm_minimal_result is None:
                    print("TEST RESULT: Could not determine pass/fail (LLM output unclear) (Minimal)")
                    case_result['llm_minimal'] = 'ERROR'
                elif llm_minimal_result == should_match:
                    print("TEST PASSED ✅ (LLM Minimal)")
                    case_result['llm_minimal'] = 'PASS'
                else:
                    print("TEST FAILED ❌ (LLM Minimal)")
                    case_result['llm_minimal'] = 'FAIL'
            except Exception as e:
                print(f"LLM-as-a-Judge (minimal) failed: {e}")
                case_result['llm_minimal'] = 'ERROR'


        print(f"{'='*60}\n")
        all_results.append(case_result)
        
    # Write results to file
    with open(RESULTS_PATH, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary results written to {RESULTS_PATH}\n")

if __name__ == "__main__":
    main() 