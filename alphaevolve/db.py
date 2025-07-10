from collections import deque
from typing import List, Tuple, Optional
import random

class EvolutionDB:
    """
    Stores solutions (code, score pairs) and implements elitism and diverse sampling.
    """
    def __init__(self, max_size: int = 100):
        if max_size <= 0:
            raise ValueError("max_size must be a positive integer")
        # Using a deque for efficient additions and keeping a fixed size (for elitism)
        # However, the RFC implies storing all N best, not just a sliding window.
        # For "Elitism: Keep top N solutions", a simple list sorted by score, then truncated, is better.
        # Let's use a list and manage sorting/truncating manually.
        self.solutions: List[Tuple[str, float]] = []
        self.max_size = max_size

    def add(self, code: str, score: float):
        """
        Adds a new solution (code, score) to the database.
        Maintains the database size according to max_size by keeping the best scores.
        Lower scores are considered better.
        """
        # Avoid adding solutions with infinite scores unless the DB is empty,
        # as they are not useful for improvement.
        if score == float('inf') and any(s_score != float('inf') for _, s_score in self.solutions):
            return

        # Add the new solution
        self.solutions.append((code, score))

        # Sort solutions by score (ascending, lower is better)
        self.solutions.sort(key=lambda x: x[1])

        # Enforce max_size by keeping only the best solutions
        if len(self.solutions) > self.max_size:
            self.solutions = self.solutions[:self.max_size]

    def get_best(self) -> Tuple[Optional[str], float]:
        """
        Returns the best solution (code, score) from the database.
        Returns (None, float('inf')) if the database is empty.
        """
        if not self.solutions:
            return None, float('inf')
        # The list is kept sorted, so the first element is the best
        return self.solutions[0]

    def sample(self, n: int) -> List[Tuple[str, float]]:
        """
        Samples N unique solutions randomly from the database for exploration.
        Returns a list of (code, score) tuples.
        If N is larger than the number of solutions, returns all solutions.
        """
        if n <= 0:
            return []

        num_solutions = len(self.solutions)
        sample_size = min(n, num_solutions)

        if sample_size == 0:
            return []

        return random.sample(self.solutions, sample_size)
