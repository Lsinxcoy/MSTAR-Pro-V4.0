"""
MSTAR Pro v4.0 - 消融实验执行器
"""

from __future__ import annotations
import logging
from typing import Dict, List, Optional, Callable
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AblationResult:
    experiment_id: str
    baseline_score: float
    ablation_scores: Dict[str, float]
    impact: Dict[str, float]
    timestamp: str


class AblationRunner:
    """
    MSTAR Pro v4.0 消融实验执行器
    定义、执行消融实验，计算影响分数
    """

    def __init__(self, fitness_tracker):
        self.fitness_tracker = fitness_tracker
        self._experiments: List[Dict] = []

    def run_ablation(self, target_program_id: str, remove_components: List[str], evaluation_fn: Optional[Callable] = None) -> AblationResult:
        experiment_id = f"ablation_{len(self._experiments)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        baseline_score = self._evaluate_program(target_program_id, evaluation_fn)

        ablation_scores = {}
        for component in remove_components:
            score = self._evaluate_program_with_removal(target_program_id, component, evaluation_fn)
            ablation_scores[component] = score

        impact = {component: baseline_score - score for component, score in ablation_scores.items()}

        result = AblationResult(
            experiment_id=experiment_id,
            baseline_score=baseline_score,
            ablation_scores=ablation_scores,
            impact=impact,
            timestamp=datetime.now().isoformat(),
        )

        self._experiments.append({'experiment_id': experiment_id, 'target': target_program_id, 'result': result})
        logger.info(f"[MSTAR Ablation] {experiment_id}: baseline={baseline_score:.3f}")

        return result

    def _evaluate_program(self, program_id: str, evaluation_fn: Optional[Callable]) -> float:
        if evaluation_fn:
            return evaluation_fn(program_id)
        program = self.fitness_tracker._get_or_create_program(program_id)
        return program.fitness_score if program else 0.5

    def _evaluate_program_with_removal(self, program_id: str, component: str, evaluation_fn: Optional[Callable]) -> float:
        if evaluation_fn:
            return evaluation_fn(program_id, without=component)
        return 0.5

    def get_experiment_history(self) -> List[Dict]:
        return list(self._experiments)