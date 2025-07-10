import asyncio
import google.generativeai as genai
from tenacity import retry, wait_random_exponential, stop_after_attempt
from typing import Optional

from decouple import config
from .monitoring import LLMMetricsCollector

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
            # This search block MUST be an exact substring of initial_bubble_sort_code_for_e2e
            search_block = """    comparisons = 0
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            comparisons += 1
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]""" # No trailing newline

            replace_block = """    # Mock LLM change: A slightly different way to count (conceptual)
    comparisons = 0
    n = len(arr)
    # Example: Introduce a small change for testing diff application
    # This is not a functional improvement, just a structural change.
    for i in range(n -1): # Mock change
        for j in range(0, n - i - 2): # Mock change
            comparisons += 1
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    # Note: This mock diff might break the sort logic, for testing evaluation of bad changes.""" # No trailing newline

            return f"""<<<<<< SEARCH
{search_block}
=======
{replace_block}
>>>>>> REPLACE
"""
        elif self.response_type == "no_change":
             return """<<<<<< SEARCH
# NonExistentBlock
=======
# AlsoNonExistent
>>>>>> REPLACE
"""
        elif self.response_type == "full_replace_example":
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
            return ""

        return ""

class GeminiAPI:
    """
    Async Gemini LLM API client using google-generativeai.
    Usage:
        metrics = LLMMetricsCollector()
        api = GeminiAPI(api_key="YOUR_API_KEY", metrics_collector=metrics)
        response = await api.call("Your prompt here")
        print(f"Average latency: {metrics.get_average_latency():.2f}s")
    """
    def __init__(
        self, 
        api_key: str = None, 
        model_name: str = "models/gemini-2.0-flash-lite", 
        location: str = "us-central1",
        metrics_collector: Optional[LLMMetricsCollector] = None
    ):
        self.api_key = api_key if api_key else config("GEMINI_API_KEY")
        self.model_name = model_name
        self.location = location
        self.metrics = metrics_collector or LLMMetricsCollector()
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name)

    @retry(wait=wait_random_exponential(multiplier=1, max=60), stop=stop_after_attempt(3))
    async def call(self, prompt: str) -> str:
        """
        Calls Gemini asynchronously and returns the text response.
        Args:
            prompt: The input prompt string.
        Returns:
            The LLM's text response as a string.
        Raises:
            Exception if the API call fails after retries.
        """
        async with self.metrics.track_call(self.model_name) as metrics:
            try:
                response = await self.model.generate_content_async(
                    [prompt],
                    generation_config={
                        "max_output_tokens": 8192,
                        "temperature": 1.0,
                        "top_p": 0.95,
                    },
                    stream=False,
                )
                
                # Update token metrics if available
                if hasattr(response, "usage"):
                    metrics.prompt_tokens = response.usage.prompt_tokens
                    metrics.completion_tokens = response.usage.completion_tokens
                    metrics.total_tokens = response.usage.total_tokens
                
                return response.text
                
            except Exception as e:
                metrics.error = str(e)
                raise

# To make this usable, we should also update alphaevolve/__init__.py
# to export MockLLMAPI.
