import asyncio
import logging
from typing import Tuple, Any # Any for llm_api placeholder

from .task import Task
from .sampler import PromptSampler, default_template
from .generator import CodeGenerator
from .evaluator import Evaluator
from .db import EvolutionDB

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AlphaEvolve:
    """
    Orchestrates the asynchronous evolutionary optimization pipeline.
    Initializes with LLM API and configuration, then runs iterations:
    sample, generate, evaluate, store. Returns the best code and score.
    """
    def __init__(self, llm_api: Any, max_iterations: int = 100, population_size: int = 50, num_candidates_to_sample: int = 3):
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive.")
        if population_size <= 0:
            raise ValueError("population_size must be positive.")
        if num_candidates_to_sample <= 0:
            raise ValueError("num_candidates_to_sample must be positive.")

        self.llm_api = llm_api
        self.max_iterations = max_iterations
        self.population_size = population_size # This is max_size for EvolutionDB
        self.num_candidates_to_sample = num_candidates_to_sample # Number of codes to mutate per iteration

        # Initialize components
        self.sampler = PromptSampler(default_template) # Uses the default template
        self.generator = CodeGenerator(llm_api)
        self.db = EvolutionDB(max_size=population_size)
        # Evaluator is initialized per task run, as it depends on task.evaluate_fn

    async def optimize(self, task: Task) -> Tuple[str, float]:
        """
        Runs the optimization loop for a given task.
        """
        # Initialize evaluator for this specific task
        evaluator = Evaluator(task.evaluate_fn)

        # Evaluate initial code and add to database
        initial_score = await evaluator.evaluate(task.initial_code)
        self.db.add(task.initial_code, initial_score)

        logging.info(f"Initial code score: {initial_score}")
        if initial_score == float('inf'):
            logging.warning("Initial code is invalid or fails evaluation. Optimization may not be effective.")
            # If initial code is unusable, we might not have a good base for prompts.
            # However, the loop below will try to generate new code from it.

        for i in range(self.max_iterations):
            logging.info(f"Iteration {i+1}/{self.max_iterations}")

            # Sample candidates from the database to mutate
            # The RFC samples 3 candidates. This is configurable.
            candidate_solutions = self.db.sample(self.num_candidates_to_sample)

            if not candidate_solutions:
                # If DB is empty (e.g. initial code was inf and nothing better found yet),
                # or sampling returned nothing, try using the initial code again if available.
                # This ensures the loop can proceed if the DB is temporarily empty.
                if initial_score != float('inf'):
                     candidate_solutions = [(task.initial_code, initial_score)]
                else:
                    # If initial code was also 'inf', and DB is empty, there's nothing to evolve from.
                    logging.error("No valid candidates to evolve from. Stopping optimization.")
                    break

            # Prepare tasks for generating new code variations
            generation_tasks = []
            for base_code, base_score in candidate_solutions:
                # The prompt sampler needs the current best score from the DB for context,
                # or the score of the current candidate if it's better or DB is empty.
                _, current_best_db_score = self.db.get_best()
                prompt_score_context = min(base_score, current_best_db_score) if current_best_db_score != float('inf') else base_score

                prompt = self.sampler.generate(
                    code=base_code,
                    score=prompt_score_context,
                    iteration=i + 1,
                    language=task.language,
                    metadata=task.metadata
                )
                generation_tasks.append(self.generator.generate(prompt, base_code))

            # Execute code generation tasks concurrently
            new_codes = await asyncio.gather(*generation_tasks)

            # Prepare tasks for evaluating new codes
            evaluation_tasks = []
            for new_code in new_codes:
                # Avoid re-evaluating if the code generator returned the original code (e.g. diff failed)
                # This requires checking if new_code is identical to any of the source `base_code`s.
                # For simplicity, we re-evaluate, but this could be an optimization.
                evaluation_tasks.append(evaluator.evaluate(new_code))

            # Execute evaluation tasks concurrently
            new_scores = await asyncio.gather(*evaluation_tasks)

            # Update database with new valid solutions
            num_added = 0
            for new_code, new_score in zip(new_codes, new_scores):
                if new_score < float('inf'):
                    # Check if this new code is an actual improvement or different from existing ones
                    # before adding. The DB's add method handles sorting and size limiting.
                    self.db.add(new_code, new_score)
                    num_added +=1

            logging.info(f"Generated {len(new_codes)} new codes, added {num_added} valid solutions to DB.")

            # Log progress: current best score from the database
            best_code_so_far, best_score_so_far = self.db.get_best()
            if best_code_so_far is not None:
                 logging.info(f"Best score after iteration {i+1}: {best_score_so_far}")
            else:
                 logging.info(f"No valid solutions in DB after iteration {i+1}.")
                 # If DB becomes empty (e.g. all solutions expired or were invalid), stop.
                 break

        final_best_code, final_best_score = self.db.get_best()
        if final_best_code is not None:
            logging.info(f"Optimization finished. Final best score: {final_best_score}")
        else:
            logging.error("Optimization finished, but no valid solution was found.")
            return task.initial_code, initial_score # Fallback to initial if nothing better

        return final_best_code, final_best_score
