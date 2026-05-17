"""
MSTAR Pro v4.0 - Memory Program
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class MemoryProgram:
    """
    MSTAR记忆程序
    """
    program_id: str
    name: str
    fitness_score: float = 0.5
    lineage_depth: int = 0
    parent_id: Optional[str] = None
    created_at: str = ""
    last_evolution_at: Optional[str] = None
    fitness_history: List[float] = field(default_factory=list)
    explanation_cache: Dict[str, str] = field(default_factory=dict)
    episodes: List[Dict] = field(default_factory=list)
    failure_type: Optional[str] = None
    last_failure_at: Optional[str] = None
    lifecycle_status: str = 'active'

    def update_fitness(self, episode: Dict, dimensions: Optional[Dict[str, float]] = None):
        self.episodes.append(episode)
        if len(self.episodes) > 100:
            self.episodes = self.episodes[-100:]

        if dimensions:
            weights = {
                'success_rate': 0.25, 'quality_score': 0.20, 'latency_p50': 0.15,
                'token_efficiency': 0.15, 'confidence': 0.10, 'error_rate': 0.15,
            }
            self.fitness_score = sum(
                dimensions.get(dim, 0.5) * weights.get(dim, 0.1)
                for dim in dimensions
            )
        else:
            success = episode.get('success', False)
            quality = episode.get('quality', 0.8)
            self.fitness_score = self.fitness_score * 0.9 + (1.0 if success else 0.0) * 0.1 * quality

        self.fitness_history.append(self.fitness_score)

    def add_explanation(self, key: str, explanation: str):
        self.explanation_cache[key] = explanation