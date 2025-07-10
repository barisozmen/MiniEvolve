# End-to-End tests for AlphaEvolve
import asyncio
import pytest
from alphaevolve import AlphaEvolve, Task, MockLLMAPI

# Common evaluation function and test cases for sorting tasks
# (Adapted from example.py)

def evaluate_sorting_test(code_str: str, test_cases: list) -> float:
    """
    Evaluates a sorting function provided as a string for testing.
    The sorting function is expected to be named (e.g., bubble_sort, insertion_sort etc.)
    and should return the sorted array and the number of comparisons.
    """
    try:
        namespace = {}
        exec(code_str, namespace)

        sort_fn = None
        for name, value in namespace.items():
            if callable(value) and not name.startswith("__") and name not in ["evaluate_sorting_test", "pytest_pyfunc_call"]: # Avoid picking up helper/pytest funcs
                sort_fn = value
                break

        if sort_fn is None:
            return float('inf')

        total_comparisons = 0
        for test_case_input in test_cases:
            arr_copy = list(test_case_input)
            sorted_arr, comparisons = sort_fn(arr_copy)

            expected_sorted_arr = sorted(test_case_input)
            if sorted_arr != expected_sorted_arr:
                # print(f"Test failed: Input {test_case_input}, Expected {expected_sorted_arr}, Got {sorted_arr}")
                return float('inf')
            total_comparisons += comparisons
        return float(total_comparisons)
    except Exception:
        # print(f"Error during evaluation in test: {e}")
        return float('inf')

default_test_cases_for_e2e = [
    [5, 2, 8, 1, 9],    # General case (simplified for faster tests)
    [0, 1, 2, 3],       # Already sorted
    [3, 2, 1, 0],       # Reverse sorted
    [2, 2, 1, 1],       # With duplicates
    [],                 # Empty list
    [42],               # Single element list
]

initial_bubble_sort_code_for_e2e = """
def bubble_sort(arr):
    comparisons = 0
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            comparisons += 1
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr, comparisons
"""

# Calculate initial score for reference in tests
initial_score_for_e2e = evaluate_sorting_test(initial_bubble_sort_code_for_e2e, default_test_cases_for_e2e)

@pytest.mark.asyncio
async def test_e2e_simple_diff_improvement():
    """
    Tests a scenario where MockLLMAPI provides a 'simple_diff'
    that should result in a change in code and potentially score.
    The mock diff provided in MockLLMAPI for 'simple_diff' might
    actually make the bubble sort worse or non-functional,
    so the key is to check that a change occurred and was evaluated.
    """
    # The MockLLMAPI's 'simple_diff' changes loop ranges, which will alter behavior.
    # For bubble_sort, this specific mock diff makes it incorrect for some inputs,
    # leading to float('inf') score.
    # If it were a genuinely improving diff, we'd check for score < initial_score_for_e2e.
    # Here, we primarily check that the code was modified and the LLM was called.

    mock_llm = MockLLMAPI(response_type="simple_diff", delay_seconds=0) # Back to simple_diff
    evolver = AlphaEvolve(
        llm_api=mock_llm,
        max_iterations=2, # Allow at least one generation attempt
        population_size=5,
        num_candidates_to_sample=1
    )
    task = Task(
        initial_code=initial_bubble_sort_code_for_e2e,
        evaluate_fn=lambda code_str: evaluate_sorting_test(code_str, default_test_cases_for_e2e),
        language="python",
        metadata={"task_description": "Test simple diff"}
    )

    best_code, best_score = await evolver.optimize(task)

    assert mock_llm.call_count > 0, "MockLLMAPI should have been called."
    assert best_code is not None

    # The 'simple_diff' from MockLLMAPI results in code that evaluates to float('inf').
    # Therefore, AlphaEvolve should correctly identify that the initial code is better.
    assert best_code == initial_bubble_sort_code_for_e2e, \
        "Best code should be the initial code when the diff leads to a worse score (inf)."
    assert best_score == initial_score_for_e2e, \
        "Best score should be the initial score when the diff leads to a worse score (inf)."

    # To be absolutely sure the diff was attempted and evaluated:
    # We can't directly inspect internal states of AlphaEvolve easily here without more logs/callbacks.
    # However, mock_llm.call_count > 0 confirms a generation was attempted.
    # The fact that the initial code (non-inf score) was chosen over a potential inf score variant
    # is implicit proof of the system working to select the actual best.

    # Final check: re-evaluate the returned best_code to ensure score consistency.
    evaluated_best_code_score = evaluate_sorting_test(best_code, default_test_cases_for_e2e)
    assert best_score == evaluated_best_code_score, \
        f"Returned best_score {best_score} does not match re-evaluation of best_code {evaluated_best_code_score}"


@pytest.mark.asyncio
async def test_e2e_no_change_scenario():
    """
    Tests a scenario where MockLLMAPI provides a 'no_change' diff,
    meaning the code should not be altered.
    """
    mock_llm = MockLLMAPI(response_type="no_change", delay_seconds=0)
    evolver = AlphaEvolve(
        llm_api=mock_llm,
        max_iterations=2,
        population_size=3,
        num_candidates_to_sample=1
    )
    task = Task(
        initial_code=initial_bubble_sort_code_for_e2e,
        evaluate_fn=lambda code_str: evaluate_sorting_test(code_str, default_test_cases_for_e2e),
        language="python",
        metadata={"task_description": "Test no_change diff"}
    )

    best_code, best_score = await evolver.optimize(task)

    assert mock_llm.call_count > 0, "MockLLMAPI should have been called."
    assert best_code == initial_bubble_sort_code_for_e2e, "Code should not have changed."
    assert best_score == initial_score_for_e2e, "Score should not have changed."

@pytest.mark.asyncio
async def test_e2e_full_replace_scenario():
    """
    Tests a scenario where MockLLMAPI provides a 'full_replace_example',
    meaning the code should be entirely replaced by the LLM's output.
    The mock returns a valid, correctly sorting function.
    """
    mock_llm = MockLLMAPI(response_type="full_replace_example", delay_seconds=0)
    # Expected code from MockLLMAPI's "full_replace_example"
    expected_replaced_code = """
def totally_new_sort(arr):
    # This is a completely different implementation
    # returned by the mock LLM as a full replacement.
    # For testing, this might be a valid or invalid sort.
    new_comparisons = 0
    # ... some placeholder logic ...
    if len(arr) > 1:
        new_comparisons = len(arr) # dummy value
    return sorted(arr), new_comparisons # Returns a correctly sorted array
"""
    # Calculate the expected score for this replaced code
    expected_score_for_replaced_code = evaluate_sorting_test(expected_replaced_code, default_test_cases_for_e2e)
    # Ensure the mock replacement is actually better or different enough to be chosen
    # For this test, we want it to be better than the initial bubble sort or at least valid.
    assert expected_score_for_replaced_code < initial_score_for_e2e, \
        "The mock 'full_replace_example' code should be better or different enough for the test to be meaningful."


    evolver = AlphaEvolve(
        llm_api=mock_llm,
        max_iterations=2, # Needs at least one iteration to generate and evaluate
        population_size=3,
        num_candidates_to_sample=1
    )
    task = Task(
        initial_code=initial_bubble_sort_code_for_e2e, # Start with bubble sort
        evaluate_fn=lambda code_str: evaluate_sorting_test(code_str, default_test_cases_for_e2e),
        language="python",
        metadata={"task_description": "Test full_replace diff"}
    )

    best_code, best_score = await evolver.optimize(task)

    assert mock_llm.call_count > 0, "MockLLMAPI should have been called."
    # Need to strip() because the LLM output might have leading/trailing whitespace
    # and the generator might also add/remove some.
    # However, the CodeGenerator.generate method with apply_diff currently returns the original code
    # if the llm_response doesn't contain diff markers. This needs to be fixed for this test to pass as intended.
    # For now, let's assume the generator will be fixed to handle full replacement.

    # Post-fix: The CodeGenerator is expected to identify non-diff responses and use them directly.
    assert best_code.strip() == expected_replaced_code.strip(), "Code should have been fully replaced."
    assert best_score == expected_score_for_replaced_code, "Score should match the replaced code."

@pytest.mark.asyncio
async def test_e2e_empty_diff_scenario():
    """
    Tests a scenario where MockLLMAPI provides an 'empty_diff' (empty string),
    meaning the code should not be altered as the diff application should fail or do nothing.
    """
    mock_llm = MockLLMAPI(response_type="empty_diff", delay_seconds=0)
    evolver = AlphaEvolve(
        llm_api=mock_llm,
        max_iterations=2,
        population_size=3,
        num_candidates_to_sample=1
    )
    task = Task(
        initial_code=initial_bubble_sort_code_for_e2e,
        evaluate_fn=lambda code_str: evaluate_sorting_test(code_str, default_test_cases_for_e2e),
        language="python",
        metadata={"task_description": "Test empty_diff response"}
    )

    best_code, best_score = await evolver.optimize(task)

    assert mock_llm.call_count > 0, "MockLLMAPI should have been called."
    # The CodeGenerator.apply_diff method should ideally return original code if diff is empty or invalid.
    assert best_code == initial_bubble_sort_code_for_e2e, "Code should not have changed with an empty diff."
    assert best_score == initial_score_for_e2e, "Score should not have changed with an empty diff."
