import asyncio

class MockLLMAPI:
    """
    A mock LLM API for testing purposes.
    It simulates an API call and returns a predefined diff or a simple modification.
    """
    def __init__(self, response_type: str = "simple_diff", delay_seconds: float = 0.1):
        """
        Args:
            response_type: "simple_diff", "no_change", "full_replace_example", "empty_diff"
            delay_seconds: Simulate network latency.
        """
        self.response_type = response_type
        self.delay_seconds = delay_seconds
        self.call_count = 0

    async def call(self, prompt: str) -> str:
        """
        Simulates an asynchronous API call.

        Args:
            prompt: The input prompt string.

        Returns:
            A string representing the LLM's response (e.g., a diff).
        """
        await asyncio.sleep(self.delay_seconds)
        self.call_count += 1

        if self.response_type == "simple_diff":
            # This is a very basic diff, assuming the input code contains "bubble_sort"
            # and a "comparisons = 0" line.
            # A more robust mock would parse the {code} block from the prompt.
            return """<<<<<< SEARCH
    comparisons = 0
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            comparisons += 1
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
=======
    # Mock LLM change: A slightly different way to count (conceptual)
    comparisons = 0
    n = len(arr)
    # Example: Introduce a small change for testing diff application
    # This is not a functional improvement, just a structural change.
    for i in range(n -1): # Mock change
        for j in range(0, n - i - 2): # Mock change
            comparisons += 1
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    # Note: This mock diff might break the sort logic, for testing evaluation of bad changes.
>>>>>> REPLACE
"""
        elif self.response_type == "no_change":
            # Returns a diff that effectively changes nothing or an empty diff
             return """<<<<<< SEARCH
# NonExistentBlock
=======
# AlsoNonExistent
>>>>>> REPLACE
"""
        elif self.response_type == "full_replace_example":
            # Example of returning a completely new function body, no diff markers
            return """
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
        elif self.response_type == "empty_diff":
            return "" # Simulates LLM returning empty content

        return "" # Default fallback

# Example of how a real API client might look (conceptual)
# class GeminiAPI:
#     def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash"):
#         self.api_key = api_key
#         self.model_name = model_name
#         # Initialize the actual Gemini client here
#         # from google.cloud import aiplatform
#         # from google.generativeai import GenerativeModel

#     async def call(self, prompt: str) -> str:
#         # Actual call to Gemini API
#         # model = GenerativeModel(self.model_name)
#         # response = await model.generate_content_async(prompt)
#         # return response.text
#         pass # Placeholder

# To make this usable, we should also update alphaevolve/__init__.py
# to export MockLLMAPI.
