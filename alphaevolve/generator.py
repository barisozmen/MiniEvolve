import re
from typing import Any # Placeholder for LLMAPI type

class CodeGenerator:
    """
    Applies LLM-proposed diffs to code.
    Supports diff format or full code replacement.
    Handles errors for invalid diffs.
    """
    def __init__(self, llm_api: Any): # Replace Any with actual LLMAPI type later
        self.llm_api = llm_api

    async def generate(self, prompt: str, original_code: str) -> str:
        """
        Calls the LLM API with the prompt and applies the returned diff.
        """
        # In a real scenario, llm_api.call would be an async method.
        # For now, assuming it's a synchronous mock or a real async implementation.
        if hasattr(self.llm_api, "call") and callable(self.llm_api.call):
            diff = await self.llm_api.call(prompt)
        else:
            # Placeholder if llm_api is not a full object, e.g. a function
            # This branch might be removed once a proper LLMAPI structure is in place
            diff = await self.llm_api(prompt)

        return self.apply_diff(original_code, diff)

    def apply_diff(self, code: str, diff: str) -> str:
        """
        Applies a diff to the given code.
        The diff is expected in the format:
        <<<<<< SEARCH
        # Original code block
        =======
        # New code block
        >>>>>> REPLACE
        If the diff format is not found, or SEARCH block not in code, returns original code.
        """
        try:
            # Using re.DOTALL to make '.' match newlines as well
            match = re.search(r'<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>> REPLACE', diff, re.DOTALL)
            if not match:
                # If the diff format is not found, assume the LLM might have returned full code.
                # This is a heuristic. For stricter control, one might want to return `code` or raise an error.
                # Per RFC: "Full code replacement (for small programs)."
                # We can refine this logic: if diff is substantially different from `code` and doesn't have markers,
                # it might be a full replacement. For now, if no markers, and diff is non-empty, assume it's new code.
                if diff and diff.strip(): # Check if diff is not empty or just whitespace
                    return diff
                return code # Return original code if diff is empty or markers not found

            old_code, new_code = match.groups()

            # Ensure leading/trailing newlines in search/replace blocks are handled consistently.
            # The RFC example implies exact block matching.
            # If old_code is not found, it might be due to subtle LLM changes (e.g. whitespace).
            # Current implementation: exact match required.
            if old_code in code:
                return code.replace(old_code, new_code)
            else:
                # If the specific SEARCH block isn't found, return original code.
                # This prevents accidental corruption if the LLM hallucinates a non-existent block.
                return code
        except Exception:
            # Fallback to original code in case of any error during diff application.
            return code
