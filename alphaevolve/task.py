from dataclasses import dataclass
from typing import Callable, Optional, Dict, Any

@dataclass
class Task:
    """
    Encapsulates the problem: initial code, evaluation function, language, and metadata.
    """
    initial_code: str
    evaluate_fn: Callable[[str], float]
    language: str = "python"
    metadata: Optional[Dict[str, Any]] = None
