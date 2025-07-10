from typing import Dict, Any

default_template = """
Optimize this {language} code:

```python
{code}
```

Current score: {score}. Goal: minimize score. Suggest a diff:

<<<<<< SEARCH
# Original code block
=======
# New code block
>>>>>> REPLACE

Make small, incremental changes. Iteration: {iteration}.
"""

class PromptSampler:
    """
    Generates LLM prompts with context (code, scores, iteration, metadata).
    """
    def __init__(self, template: str = default_template):
        self.template = template

    def generate(self, code: str, score: float, iteration: int, language: str, metadata: Dict[str, Any] = None) -> str:
        """
        Generates a prompt string based on the template and provided context.
        """
        # Ensure metadata is a dictionary to avoid errors during formatting if None.
        if metadata is None:
            metadata = {}

        return self.template.format(
            code=code,
            score=score,
            iteration=iteration,
            language=language,
            **metadata
        )
