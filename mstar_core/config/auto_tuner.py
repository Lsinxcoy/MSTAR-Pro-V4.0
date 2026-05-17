"""
MSTAR Pro v4.0 - 自动调优引擎
"""

from __future__ import annotations
import logging
from typing import Dict, List, Optional
from datetime import datetime
from collections import deque

logger = logging.getLogger(__name__)


class AutoTuner:
    """
    MSTAR Pro v4.0 自动调优引擎
    监控关键指标 + 自动调整参数
    """

    def __init__(self, fitness_tracker, config: Optional[Dict] = None):
        self.fitness_tracker = fitness_tracker
        self.config = config or {}
        self.fitness_threshold_high = self.config.get('fitness_threshold_high', 0.75)
        self.fitness_threshold_low = self.config.get('fitness_threshold_low', 0.35)
        self.evolution_interval_min = self.config.get('evolution_interval_min', 3)
        self.evolution_interval_max = self.config.get('evolution_interval_max', 50)
        self._session_history = deque(maxlen=100)
        self._tuning_decisions: List[Dict] = []
        self._current_interval = 10

    def record_session(self, session_data: Dict):
        self._session_history.append({'timestamp': datetime.now().isoformat(), 'data': session_data})

    def analyze_and_tune(self) -> Dict:
        if len(self._session_history) < 5:
            return {'action': 'wait', 'reason': 'Insufficient data'}

        recent_sessions = list(self._session_history)[-10:]
        avg_fitness = sum(s['data'].get('fitness', 0.5) for s in recent_sessions) / len(recent_sessions)
        trend = self._detect_trend(recent_sessions)

        if avg_fitness < self.fitness_threshold_low:
            if trend < 0:
                action = 'increase_evolution_frequency'
                reason = 'Low fitness with declining trend'
            else:
                action = 'maintain'
                reason = 'Low fitness but stable'
        elif avg_fitness > self.fitness_threshold_high:
            if trend > 0:
                action = 'decrease_evolution_frequency'
                reason = 'High fitness with improving trend'
            else:
                action = 'maintain'
                reason = 'High fitness but plateau'
        else:
            action = 'maintain'
            reason = 'Fitness in acceptable range'

        if action == 'increase_evolution_frequency':
            self._current_interval = max(self.evolution_interval_min, self._current_interval - 3)
        elif action == 'decrease_evolution_frequency':
            self._current_interval = min(self.evolution_interval_max, self._current_interval + 2)

        decision = {
            'action': action, 'reason': reason, 'avg_fitness': avg_fitness,
            'trend': trend, 'new_interval': self._current_interval, 'timestamp': datetime.now().isoformat(),
        }

        self._tuning_decisions.append(decision)
        logger.info(f"[MSTAR AutoTuner] {action}: {reason} (interval={self._current_interval})")

        return decision

    def _detect_trend(self, sessions: List[Dict]) -> float:
        if len(sessions) < 3:
            return 0.0
        fitness_values = [s['data'].get('fitness', 0.5) for s in sessions]
        if len(fitness_values) >= 3:
            slope = (fitness_values[-1] - fitness_values[0]) / len(fitness_values)
            return slope
        return 0.0

    def get_tuning_history(self) -> List[Dict]:
        return list(self._tuning_decisions)

    def get_recommendations(self) -> List[str]:
        recommendations = []
        if self._current_interval < 5:
            recommendations.append("Evolution frequency is very high - aggressive optimization mode")
        elif self._current_interval > 30:
            recommendations.append("Evolution frequency is low - consider more aggressive optimization")
        recent_decisions = self._tuning_decisions[-5:]
        if all(d['action'] == 'maintain' for d in recent_decisions):
            recommendations.append("System appears stable - no major tuning needed")
        return recommendations