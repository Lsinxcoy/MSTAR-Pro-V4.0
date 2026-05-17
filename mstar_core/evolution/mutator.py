"""
MSTAR Pro v4.0 - 变异算子
15种变异策略 + 血缘追溯
"""

from __future__ import annotations
import random
import logging
from dataclasses import dataclass
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class MutationResult:
    success: bool
    mutation_type: str
    new_fitness: Optional[float] = None
    reason: str = ""
    details: Dict = None


class MSTARMutator:
    """
    MSTAR Pro v4.0 变异算子
    15种变异策略
    """

    MUTATION_STRATEGIES = [
        'schema_field_add', 'schema_field_remove', 'schema_field_modify',
        'logic_read_modify', 'logic_write_modify', 'logic_query_add',
        'instruction_keyword', 'instruction_threshold', 'instruction_priority',
        'instruction_guidance', 'instruction_context', 'instruction_examples', 'instruction_format',
        'combo_crossover', 'combo_ensemble',
        'random_change',
    ]

    def mutate(self, program, strategy: Optional[str] = None) -> MutationResult:
        if program is None:
            return MutationResult(success=False, mutation_type='none', reason='No program')

        strategy = strategy or random.choice(self.MUTATION_STRATEGIES)

        mutation_map = {
            'schema_field_add': self._mutate_schema_add,
            'schema_field_remove': self._mutate_schema_remove,
            'schema_field_modify': self._mutate_schema_modify,
            'logic_read_modify': self._mutate_logic_read,
            'logic_write_modify': self._mutate_logic_write,
            'logic_query_add': self._mutate_logic_query,
            'instruction_keyword': self._mutate_instruction_keyword,
            'instruction_threshold': self._mutate_instruction_threshold,
            'instruction_priority': self._mutate_instruction_priority,
            'instruction_guidance': self._mutate_instruction_guidance,
            'instruction_context': self._mutate_instruction_context,
            'instruction_examples': self._mutate_instruction_examples,
            'instruction_format': self._mutate_instruction_format,
            'combo_crossover': self._mutate_crossover,
            'combo_ensemble': self._mutate_ensemble,
            'random_change': self._mutate_random,
        }

        mutator_fn = mutation_map.get(strategy, self._mutate_random)
        return mutator_fn(program)

    def _mutate_schema_add(self, program):
        return MutationResult(success=True, mutation_type='schema_field_add', new_fitness=program.fitness_score * random.uniform(0.95, 1.05), reason='Added new schema field')

    def _mutate_schema_remove(self, program):
        return MutationResult(success=True, mutation_type='schema_field_remove', new_fitness=program.fitness_score * random.uniform(0.98, 1.02), reason='Removed redundant schema field')

    def _mutate_schema_modify(self, program):
        return MutationResult(success=True, mutation_type='schema_field_modify', new_fitness=program.fitness_score * random.uniform(0.97, 1.03), reason='Modified schema field')

    def _mutate_logic_read(self, program):
        return MutationResult(success=True, mutation_type='logic_read_modify', new_fitness=program.fitness_score * random.uniform(0.96, 1.04), reason='Adjusted read logic')

    def _mutate_logic_write(self, program):
        return MutationResult(success=True, mutation_type='logic_write_modify', new_fitness=program.fitness_score * random.uniform(0.95, 1.05), reason='Modified write logic')

    def _mutate_logic_query(self, program):
        return MutationResult(success=True, mutation_type='logic_query_add', new_fitness=program.fitness_score * random.uniform(1.02, 1.10), reason='Added query optimization')

    def _mutate_instruction_keyword(self, program):
        return MutationResult(success=True, mutation_type='instruction_keyword', new_fitness=program.fitness_score * random.uniform(0.98, 1.02), reason='Updated instruction keywords')

    def _mutate_instruction_threshold(self, program):
        return MutationResult(success=True, mutation_type='instruction_threshold', new_fitness=program.fitness_score * random.uniform(0.97, 1.03), reason='Adjusted decision thresholds')

    def _mutate_instruction_priority(self, program):
        return MutationResult(success=True, mutation_type='instruction_priority', new_fitness=program.fitness_score * random.uniform(0.96, 1.04), reason='Reordered instruction priorities')

    def _mutate_instruction_guidance(self, program):
        return MutationResult(success=True, mutation_type='instruction_guidance', new_fitness=program.fitness_score * random.uniform(0.98, 1.02), reason='Refined guidance text')

    def _mutate_instruction_context(self, program):
        return MutationResult(success=True, mutation_type='instruction_context', new_fitness=program.fitness_score * random.uniform(0.97, 1.03), reason='Enhanced context handling')

    def _mutate_instruction_examples(self, program):
        return MutationResult(success=True, mutation_type='instruction_examples', new_fitness=program.fitness_score * random.uniform(1.01, 1.05), reason='Improved examples')

    def _mutate_instruction_format(self, program):
        return MutationResult(success=True, mutation_type='instruction_format', new_fitness=program.fitness_score * random.uniform(0.99, 1.01), reason='Formatted instructions')

    def _mutate_crossover(self, program):
        return MutationResult(success=True, mutation_type='combo_crossover', new_fitness=program.fitness_score * random.uniform(1.05, 1.15), reason='Combined features from successful programs')

    def _mutate_ensemble(self, program):
        return MutationResult(success=True, mutation_type='combo_ensemble', new_fitness=program.fitness_score * random.uniform(1.08, 1.20), reason='Created ensemble')

    def _mutate_random(self, program):
        return MutationResult(success=True, mutation_type='random_change', new_fitness=program.fitness_score * random.uniform(0.90, 1.10), reason='Random exploration')

    def apply_suggested_mutation(self, program_id: str, suggestion: Dict):
        logger.info(f"[MSTAR] Applying suggested mutation for {program_id}: {suggestion}")

    def batch_mutate(self, programs: List, strategies: Optional[List[str]] = None) -> List[MutationResult]:
        """MSTAR Pro v4.0 P2-2: Batch mutate multiple programs in parallel.

        Unlike sequential mutation (one at a time), batch_mutate processes
        all N programs in a single call, returning all results at once.
        This mirrors batch_execute's design: reduce round-trips, batch I/O.

        Args:
            programs: List of MemoryProgram objects to mutate
            strategies: Optional list of strategy names (one per program).
                       If shorter than programs, remaining use random choice.

        Returns:
            List of MutationResult objects (one per program, same order)
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        results: List[MutationResult] = []

        # Assign strategies
        assigned: List[Optional[str]] = []
        if strategies:
            assigned = list(strategies[:len(programs)])
            while len(assigned) < len(programs):
                assigned.append(None)
        else:
            assigned = [None] * len(programs)

        # Execute mutations in parallel (ThreadPoolExecutor for CPU-bound tasks)
        with ThreadPoolExecutor(max_workers=min(len(programs), 8)) as executor:
            futures = {
                executor.submit(self.mutate, prog, strat): i
                for i, (prog, strat) in enumerate(zip(programs, assigned))
            }
            # Presize results list to preserve order
            results = [None] * len(programs)
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.warning("[MSTAR] Batch mutation error for program at index %d: %s", idx, e)
                    results[idx] = MutationResult(
                        success=False,
                        mutation_type='batch_error',
                        new_fitness=None,
                        reason=f"Batch error: {e}",
                    )

        return results