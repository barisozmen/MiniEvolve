import asyncio
from alphaevolve import AlphaEvolve, Task, MockLLMAPI, GeminiAPI, LLMMetricsCollector # Correctly import MockLLMAPI

def evaluate_sorting(code_str: str, test_cases: list) -> float:
    """
    Evaluates a sorting function provided as a string.
    The sorting function is expected to be named (e.g., bubble_sort, insertion_sort etc.)
    and should return the sorted array and the number of comparisons.
    """
    try:
        # Create a new namespace for exec to run in
        namespace = {}
        # Execute the code string, defining the sort function in the namespace
        exec(code_str, namespace)

        # Attempt to find the sorting function in the namespace.
        # Assumes the function is the only callable that doesn't start with '__'.
        sort_fn = None
        for name, value in namespace.items():
            if callable(value) and not name.startswith("__") and name != "evaluate_sorting": # Avoid picking up this eval func
                sort_fn = value
                break

        if sort_fn is None:
            # print("No sort function found in the provided code.")
            return float('inf')

        total_comparisons = 0
        for test_case_input in test_cases:
            arr_copy = list(test_case_input) # Use a copy to avoid modifying the original test case

            # Call the sort function
            # It should return (sorted_array, comparisons_count)
            sorted_arr, comparisons = sort_fn(arr_copy)

            # Verify correctness
            if sorted_arr != sorted(test_case_input):
                # print(f"Incorrect sort: expected {sorted(test_case_input)}, got {sorted_arr}")
                return float('inf')

            total_comparisons += comparisons

        return float(total_comparisons)
    except Exception as e:
        # print(f"Error during evaluation: {e}")
        return float('inf')

# Define the initial task for sorting
initial_bubble_sort_code = """
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

# Test cases for the sorting algorithm
# Using a mix of cases: already sorted, reverse sorted, duplicates, empty, single element
default_test_cases = [
    [5, 2, 8, 1, 9, 4, 6, 3, 7, 0], # General case
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], # Already sorted
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0], # Reverse sorted
    [3, 3, 3, 1, 1, 2, 2],          # With duplicates
    [],                             # Empty list
    [42],                           # Single element list
    [5, -2, 8, -1, 0, 9]            # With negative numbers
]


sort_task = Task(
    initial_code=initial_bubble_sort_code,
    evaluate_fn=lambda code_str: evaluate_sorting(code_str, default_test_cases),
    language="python",
    metadata={"task_description": "Optimize a Python sorting function to minimize comparisons."}
)

metrics = LLMMetricsCollector()

async def main():
    print("Starting AlphaEvolve optimization for sorting...")

    # Initialize AlphaEvolve with the MockLLMAPI for testing
    # Using a small number of iterations and population for quick test
    evolver = AlphaEvolve(
        # Use MockLLMAPI for testing, or GeminiAPI for real LLM calls.
        # llm_api=MockLLMAPI(response_type="simple_diff", delay_seconds=0.05),
        llm_api=GeminiAPI(
            metrics_collector=metrics
        ),
        max_iterations=3,       # Reduced for quick example run
        population_size=3,       # Reduced for quick example run
        num_candidates_to_sample=2 # How many codes to mutate each iteration
    )

    # Run the optimization process
    best_code, best_score = await evolver.optimize(sort_task)

    print("\nOptimization Complete.")
    print("-------------------------")
    if best_code:
        print(f"Optimized code ({best_score} comparisons):\n{best_code}")
    else:
        print("No suitable code found.")

    print("\nInitial code for reference:")
    initial_eval = evaluate_sorting(initial_bubble_sort_code, default_test_cases)
    print(f"({initial_eval} comparisons):\n{initial_bubble_sort_code}")
    
    print(f"Average latency: {metrics.get_average_latency():.2f}s")
    print(f"Success rate: {metrics.get_success_rate():.2f}")
    print(f"Token stats: {metrics.get_token_stats()}")
    
    print(f"Metrics: {metrics.get_metrics()}")
    
    pass

if __name__ == "__main__":
    asyncio.run(main())
