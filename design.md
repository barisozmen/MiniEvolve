# RFC: AlphaEvolve Python Library Design

**Author**: Peter Norvig  
**Date**: July 10, 2025  
**Status**: Draft  
**Version**: 1.0  

---

## Abstract

AlphaEvolve is an evolutionary coding agent that iteratively optimizes algorithms using large language models (LLMs), automated evaluation, and an evolutionary database. This RFC proposes a minimalist, extensible Python library (`alphaevolve`) to democratize algorithm optimization. The design prioritizes simplicity, clarity, and scalability while maintaining a high signal-to-noise ratio. Inspired by Google DeepMind’s AlphaEvolve (arXiv:2506.13131v1), the library enables users to optimize code for diverse tasks—matrix multiplication, sorting, mathematical constructions—via a clean API and modular architecture.

Mental model: *Think of AlphaEvolve as a lean startup for code optimization.* It’s a factory that takes raw code, refines it through iterative feedback, and ships optimized algorithms. Every component is a worker; every line of code is deliberate.

---

## Motivation

Algorithm optimization is critical across science and engineering—faster matrix multiplication powers AI, better scheduling saves compute, novel constructions advance mathematics. Yet, hand-tuning algorithms is slow, and existing tools are either too specialized (e.g., AutoML) or too complex (e.g., genetic programming frameworks). AlphaEvolve bridges this gap by combining LLM-driven code generation with evolutionary search, guided by automated evaluation. A Python library makes this accessible to researchers, engineers, and hobbyists.

**Why now?** LLMs are powerful enough to propose meaningful code changes. Compute is cheap. Automated evaluation is feasible for many tasks. The time is ripe to productize algorithm optimization.

Mental model: *This is a land grab.* First principles: code is malleable, evaluation is truth, evolution is robust. Build a tool that scales human ingenuity.

---

## Goals

1. **Simplicity**: Clean API, minimal dependencies (Python stdlib, LLM API).
2. **Extensibility**: Support diverse tasks (sorting, math, systems) and LLMs.
3. **Scalability**: Async pipeline for parallelism, single-machine default.
4. **Reliability**: Safe code execution, robust error handling.
5. **Reproducibility**: Deterministic where possible, with logging.

Non-goals:
- Support tasks without automated evaluation (e.g., UI design).
- Full distributed computing (focus on single-machine, extensible to clusters).
- Built-in LLM hosting (use external APIs like Gemini).

Mental model: *Build an MVP that nails the core loop.* Like a VC, prioritize product-market fit (task optimization) before chasing scale.

---

## First Principles

1. **What’s optimization?** A search for better code, measured by a user-defined metric (e.g., runtime, operations).
2. **Why evolutionary?** Code space is vast, non-differentiable. Evolution (mutate, evaluate, select) is robust and parallelizable.
3. **Why LLMs?** They generate diverse, context-aware code mutations, acting as a creative engine.
4. **Why automated evaluation?** Objective, fast feedback. No human bottlenecks.
5. **Why Python?** Universal, readable, rapid prototyping. Ecosystem for LLMs and async.

Mental model: *Deconstruct to atoms.* Code, evaluation, mutation, selection. Rebuild with minimal assumptions.

---

## Proposed Design

The `alphaevolve` library is a modular, async pipeline with five core components:
1. **Task Specification**: User provides initial code and evaluation function.
2. **Prompt Sampler**: Generates LLM prompts with context (code, scores).
3. **Code Generator**: Applies LLM-proposed diffs to code.
4. **Evaluator**: Runs code, computes scores, handles errors.
5. **Evolution Database**: Stores solutions, balances exploration/exploitation.

### Architecture

```plaintext
[Task Spec] -> [Prompt Sampler] -> [Code Generator] -> [Evaluator] -> [Evolution DB]
    |                 |                   |                  |                |
    v                 v                   v                  v                v
Initial Code    LLM Context         New Code         Scores/Solutions     Best Code
```

Mental model: *A feedback loop, like A/B testing.* Each component is a specialist: prompt sampler sets strategy, generator executes, evaluator judges, database curates.

### API

```python
from alphaevolve import AlphaEvolve, Task

# Define task
task = Task(
    initial_code="""
def bubble_sort(arr):
    comparisons = 0
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            comparisons += 1
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr, comparisons
""",
    evaluate_fn=lambda code: evaluate_sorting(code, [[5, 2, 8, 1, 9], [3, 3, 3]]),
    language="python"
)

# Initialize with LLM API (e.g., Gemini)
evolver = AlphaEvolve(
    llm_api=GeminiAPI("gemini-2.0-flash"),
    max_iterations=100,
    population_size=50
)

# Run optimization
best_code, best_score = evolver.optimize(task)
print(f"Optimized code:\n{best_code}\nScore: {best_score}")
```

Mental model: *API as contract.* Users provide code and metrics; library handles the rest. Simple, like a REST endpoint.

### Components

#### 1. Task Specification (`Task`)

Encapsulates the problem:
- `initial_code`: String of starting code (e.g., Python function).
- `evaluate_fn`: Function mapping code to a score (lower is better; `inf` for errors).
- `language`: Code language (e.g., "python", "cpp"). Extensible for multi-language support.
- `metadata`: Optional dict for task-specific info (e.g., test cases, constraints).

```python
from dataclasses import dataclass
from typing import Callable, Optional, Dict

@dataclass
class Task:
    initial_code: str
    evaluate_fn: Callable[[str], float]
    language: str = "python"
    metadata: Optional[Dict] = None
```

Mental model: *Task is the spec sheet.* It’s the blueprint for what “better” means.

#### 2. Prompt Sampler (`PromptSampler`)

Generates LLM prompts with:
- Current code.
- Evaluation scores.
- Instructions for diff-based changes.
- Optional context (e.g., problem description, prior solutions).

```python
class PromptSampler:
    def __init__(self, template: str):
        self.template = template
    
    def generate(self, code: str, score: float, iteration: int, metadata: Dict) -> str:
        return self.template.format(
            code=code,
            score=score,
            iteration=iteration,
            **(metadata or {})
        )

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
```

Mental model: *Prompts are the LLM’s marching orders.* Clear, context-rich prompts maximize signal, like a well-crafted pitch deck.

#### 3. Code Generator (`CodeGenerator`)

Applies LLM-proposed diffs to code. Supports:
- Diff format (`<<<<<< SEARCH ... ======= ... >>>>>> REPLACE`).
- Full code replacement (for small programs).
- Error handling for invalid diffs.

```python
import re

class CodeGenerator:
    def __init__(self, llm_api):
        self.llm_api = llm_api
    
    async def generate(self, prompt: str, original_code: str) -> str:
        diff = await self.llm_api.call(prompt)
        return self.apply_diff(original_code, diff)
    
    def apply_diff(self, code: str, diff: str) -> str:
        try:
            match = re.search(r'<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>> REPLACE', diff, re.DOTALL)
            if not match:
                return code
            old_code, new_code = match.groups()
            return code.replace(old_code, new_code) if old_code in code else code
        except Exception:
            return code
```

Mental model: *Generator is the craftsman.* It takes LLM’s raw ideas (diffs) and shapes them into executable code, rejecting garbage.

#### 4. Evaluator (`Evaluator`)

Runs the user’s `evaluate_fn` on generated code. Features:
- Safe execution (isolated namespace).
- Error handling (returns `inf` for crashes).
- Optional cascade (multi-stage testing for efficiency).

```python
class Evaluator:
    def __init__(self, evaluate_fn: Callable[[str], float]):
        self.evaluate_fn = evaluate_fn
    
    async def evaluate(self, code: str) -> float:
        try:
            return self.evaluate_fn(code)
        except Exception:
            return float('inf')
```

Mental model: *Evaluator is the judge.* It’s the objective truth, separating wheat from chaff. No score, no progress.

#### 5. Evolution Database (`EvolutionDB`)

Stores solutions (code, score pairs). Implements:
- Elitism: Keep top N solutions.
- Diversity: Sample randomly for exploration.
- Extensible for multi-objective optimization (e.g., MAP-Elites).

```python
from collections import deque
from typing import List, Tuple
import random

class EvolutionDB:
    def __init__(self, max_size: int = 100):
        self.solutions = deque(maxlen=max_size)
    
    def add(self, code: str, score: float):
        self.solutions.append((code, score))
    
    def get_best(self) -> Tuple[str, float]:
        return min(self.solutions, key=lambda x: x[1]) if self.solutions else (None, float('inf'))
    
    def sample(self, n: int) -> List[Tuple[str, float]]:
        return random.sample(list(self.solutions), min(n, len(self.solutions)))
```

Mental model: *Database is the portfolio.* It curates winners and wildcards, balancing exploitation (best solutions) and exploration (diverse ideas).

#### 6. Core Pipeline (`AlphaEvolve`)

Orchestrates the async pipeline:
- Initializes with LLM API and config.
- Runs iterations: sample, generate, evaluate, store.
- Returns best code and score.

```python
import asyncio
import logging
from typing import Tuple

logging.basicConfig(level=logging.INFO)

class AlphaEvolve:
    def __init__(self, llm_api, max_iterations: int = 100, population_size: int = 50):
        self.llm_api = llm_api
        self.max_iterations = max_iterations
        self.population_size = population_size
        self.sampler = PromptSampler(default_template)
        self.generator = CodeGenerator(llm_api)
        self.db = EvolutionDB(population_size)
    
    async def optimize(self, task: Task) -> Tuple[str, float]:
        self.db.add(task.initial_code, await Evaluator(task.evaluate_fn).evaluate(task.initial_code))
        
        for i in range(self.max_iterations):
            logging.info(f"Iteration {i+1}/{self.max_iterations}")
            
            # Sample candidates
            candidates = [code for code, _ in self.db.sample(3)]
            if not candidates:
                candidates = [task.initial_code]
            
            # Generate and evaluate mutations
            tasks = []
            for code in candidates:
                _, score = self.db.get_best()
                prompt = self.sampler.generate(code, score, i+1, task.metadata)
                tasks.append(self.generator.generate(prompt, code))
            
            new_codes = await asyncio.gather(*tasks)
            evaluator = Evaluator(task.evaluate_fn)
            scores = await asyncio.gather(*(evaluator.evaluate(code) for code in new_codes))
            
            # Update database
            for code, score in zip(new_codes, scores):
                if score < float('inf'):
                    self.db.add(code, score)
            
            # Log progress
            _, best_score = self.db.get_best()
            logging.info(f"Best score: {best_score}")
        
        return self.db.get_best()
```

Mental model: *Pipeline is the CEO.* It delegates, monitors, and iterates, ensuring the factory hums.

---

## Usage Example

Optimize a sorting algorithm to minimize comparisons:

```python
from alphaevolve import AlphaEvolve, Task

def evaluate_sorting(code: str, test_cases: list) -> float:
    try:
        namespace = {}
        exec(code, namespace)
        sort_fn = next(v for k, v in namespace.items() if callable(v))
        total_comparisons = 0
        for test in test_cases:
            arr = test.copy()
            sorted_arr, comparisons = sort_fn(arr)
            if sorted_arr != sorted(test):
                return float('inf')
            total_comparisons += comparisons
        return total_comparisons
    except Exception:
        return float('inf')

task = Task(
    initial_code="""
def bubble_sort(arr):
    comparisons = 0
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            comparisons += 1
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr, comparisons
""",
    evaluate_fn=lambda code: evaluate_sorting(code, [[5, 2, 8, 1, 9], [3, 3, 3]]),
    language="python",
    metadata={"test_cases": [[5, 2, 8, 1, 9], [3, 3, 3]]}
)

evolver = AlphaEvolve(
    llm_api=MockLLMAPI(),  # Replace with GeminiAPI
    max_iterations=100,
    population_size=50
)

best_code, best_score = asyncio.run(evolver.optimize(task))
print(f"Optimized code:\n{best_code}\nScore: {best_score}")
```

Mental model: *User provides the problem; library solves it.* Like hiring a consultant—you set the goal, they deliver.

---

## Implementation Notes

1. **Dependencies**: Python 3.8+, `asyncio`, `re`. Optional: LLM API client (e.g., `google-cloud-aiplatform` for Gemini).
2. **Safety**: Code execution uses isolated namespaces. Future: sandbox with `restrictedpython`.
3. **Logging**: Standard `logging` for progress and debugging.
4. **Testing**: Unit tests for each component (e.g., `PromptSampler.generate`, `CodeGenerator.apply_diff`). Integration tests for pipeline.
5. **Performance**: Async pipeline maximizes throughput. Single-machine default; extensible to clusters via task queues (e.g., Celery).

Mental model: *Build for reliability, not just speed.* Like a startup, ship a stable product before optimizing for scale.

---

## Extensibility

1. **New Tasks**: Define new `Task` with custom `initial_code` and `evaluate_fn`. Example: optimize matrix multiplication by minimizing scalar operations.
2. **LLM Support**: Swap `llm_api` (e.g., Gemini, OpenAI, LLaMA). Interface: `async def call(prompt: str) -> str`.
3. **Evaluation Cascade**: Add multi-stage `evaluate_fn` (e.g., syntax check, then full run).
4. **Database**: Extend `EvolutionDB` for multi-objective optimization or clustering.
5. **Languages**: Support C++, Java, etc., by updating `PromptSampler` and `Evaluator`.

Mental model: *Design for forks.* Like open-source, make it easy for others to hack and extend.

---

## Risks and Mitigations

1. **LLM Hallucination**: Invalid diffs or code. *Mitigation*: Robust diff parsing, fallback to original code.
2. **Evaluation Errors**: Crashes or infinite loops. *Mitigation*: Timeout in `Evaluator`, return `inf`.
3. **Compute Intensity**: Slow evaluations. *Mitigation*: Async parallelism, optional cascade.
4. **Overfitting**: Optimizing for test cases. *Mitigation*: Diverse test cases in `Task.metadata`.
5. **Reproducibility**: LLM non-determinism. *Mitigation*: Seed random sampling, log all inputs/outputs.

Mental model: *Anticipate failure.* Like a VC, stress-test the system and hedge risks.

---

## Future Work

1. **Multi-Objective Optimization**: Support multiple metrics (e.g., speed vs. memory).
2. **Human Feedback**: Integrate manual scoring for non-automatable tasks.
3. **Distributed Scale**: Add cluster support via task queues.
4. **Open-Source LLMs**: Test with LLaMA, Mistral for accessibility.
5. **Visualization**: Add tools to track evolution (e.g., score trends, code diffs).

Mental model: *Plan for growth, but don’t overbuild.* Like a startup, nail the core before chasing features.

---

## Alternatives Considered

1. **Genetic Programming**: Too low-level, requires parsing ASTs. AlphaEvolve’s diff-based approach is simpler.
2. **Reinforcement Learning**: High sample complexity. Evolution is more robust for code.
3. **Manual Code Editing**: Slow, human-dependent. LLMs automate creativity.
4. **Existing Frameworks (e.g., DEAP)**: Too generic, not LLM-integrated. AlphaEvolve is purpose-built.

Mental model: *Choose the simplest tool for the job.* Like a craftsman, pick the right hammer, not the shiniest.

---

## Conclusion

The `alphaevolve` library is a minimalist, powerful tool for algorithm optimization. By combining LLM-driven code generation, automated evaluation, and evolutionary search, it enables users to tackle complex problems—from sorting to matrix multiplication—with a clean API. The design is simple yet extensible, reliable yet scalable, elegant yet practical.

Mental model: *Code is poetry, optimization is truth.* This library is a lever to amplify human ingenuity. Let’s build, iterate, and ship.

---

## Call to Action

- **Reviewers**: Provide feedback on API, extensibility, and risks.
- **Implementers**: Prototype `alphaevolve` using this RFC.
- **Users**: Try it on your optimization tasks and share results on X (@norvig).

*Elegance in code, clarity in thought. Let’s evolve.*

--- 

**Peter Norvig**  
July 10, 2025
