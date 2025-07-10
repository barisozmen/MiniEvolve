from typing import Callable

class Evaluator:
    """
    Runs the user's `evaluate_fn` on generated code.
    Features safe execution (within the context of `evaluate_fn`) and error handling.
    """
    def __init__(self, evaluate_fn: Callable[[str], float]):
        self.evaluate_fn = evaluate_fn

    async def evaluate(self, code: str) -> float:
        """
        Evaluates the given code string using the provided evaluation function.
        Returns the score, or float('inf') if an exception occurs.
        The evaluation function itself is responsible for any sandboxing or safe execution.
        """
        try:
            # The evaluate_fn is expected to handle the execution of the code string.
            # This might involve exec(), importing the code as a module, or other strategies.
            # The RFC example uses exec() within evaluate_sorting.
            score = self.evaluate_fn(code)
            # Ensure the score is a float, as per Callable[[str], float] type hint
            if not isinstance(score, (float, int)): # Allow int, will be float('inf') or actual score
                 # Or raise TypeError("evaluate_fn must return a float or int")
                 return float('inf') # Default to 'inf' if type is wrong
            return float(score)
        except Exception:
            return float('inf')
