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
            # Normalize line endings for robustness, as LLM output or code input might vary.
            code_normalized = code.replace('\r\n', '\n')
            diff_normalized = diff.replace('\r\n', '\n')

            # Using re.DOTALL to make '.' match newlines as well
            match = re.search(r'<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>> REPLACE', diff_normalized, re.DOTALL)
            if not match:
                # If the diff format is not found, assume the LLM might have returned full code.
                # This is a heuristic. For stricter control, one might want to return `code` or raise an error.
                # Per RFC: "Full code replacement (for small programs)."
                # We can refine this logic: if diff is substantially different from `code` and doesn't have markers,
                # it might be a full replacement. For now, if no markers, and diff is non-empty, assume it's new code.
                if diff_normalized and diff_normalized.strip(): # Check if diff is not empty or just whitespace
                    # If it's full replacement, return the normalized diff content
                    return diff_normalized
                return code_normalized # Return original normalized code if diff is empty or markers not found

            old_code_block, new_code_block = match.groups()

            # Captured blocks from regex on normalized diff will also have normalized newlines.
            # No further .replace('\r\n', '\n') needed on old_code_block/new_code_block here.

            # First, try a direct match using normalized code and blocks
            if old_code_block in code_normalized:
                return code_normalized.replace(old_code_block, new_code_block)

            # If direct match fails, try stripping leading/trailing newlines from the captured old_code_block.
            # This can help if the regex captures an extra newline at the start/end of the SEARCH block
            # or if the code string has slightly different whitespace there.
            stripped_old_code_block = old_code_block.strip('\r\n')

            # Check if stripping had an effect and if the stripped version matches
            if stripped_old_code_block != old_code_block and stripped_old_code_block in code_normalized:
                # If using stripped old block, also use stripped new block for consistency in replacement
                stripped_new_code_block = new_code_block.strip('\r\n')
                return code_normalized.replace(stripped_old_code_block, stripped_new_code_block)

            # If neither exact nor stripped old_code is found, return original (normalized) code.
            # This prevents accidental corruption if the LLM hallucinates a non-existent block
            # or if the diff format is subtly mismatched.
            # Optionally, log a warning here: print(f"DEBUG: old_code_block ('{old_code_block[:50]}...') not found in code_normalized.")
            # Ensure to return code_normalized if all attempts fail and it's not the debug case
            return code_normalized # Ensure this is outside the debug block or handled after.
        except Exception: # pylint: disable=broad-except
            # Fallback to original code (or its normalized version if available) in case of any error.
            # Optionally, log the exception here.
            return code.replace('\r\n', '\n') # Ensure returned code is also normalized
