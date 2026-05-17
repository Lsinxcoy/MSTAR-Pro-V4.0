"""
MSTAR Pro v4.0 - SelfImprovingBridge
实现 correction -> mutation -> reinforcement 闭环
"""

from __future__ import annotations
import logging
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class CorrectionSignal:
    source: str
    correction_type: str
    target_program_id: str
    data: Dict
    timestamp: str


@dataclass
class ReinforcementSignal:
    signal_type: str
    program_id: str
    metrics: Dict
    timestamp: str


class SelfImprovingBridge:
    """
    MSTAR Pro v4.0 双向通信桥

    self_improver -> MSTAR: correction -> mutation
    MSTAR -> self_improver: evolution -> reinforcement
    Homunculus -> MSTAR: observation -> fitness_adjustment
    """

    def __init__(self, mstar_core, agent_callback: Optional[Callable] = None):
        self.mstar_core = mstar_core
        self.agent_callback = agent_callback
        self._lock = threading.RLock()
        self._pending_corrections: List[CorrectionSignal] = []
        self._evolution_history: List[Dict] = []
        self._reinforcement_queue: List[ReinforcementSignal] = []
        self._corrections_received = 0
        self._reinforcements_sent = 0

    def receive_correction(self, signal: CorrectionSignal):
        with self._lock:
            self._pending_corrections.append(signal)
            self._corrections_received += 1
            logger.info(f"[MSTAR Bridge] Received correction: {signal.correction_type} for {signal.target_program_id}")
            self._process_correction(signal)

    def _process_correction(self, signal: CorrectionSignal):
        if signal.correction_type == 'fitness_adjustment':
            self.mstar_core.fitness_tracker.adjust_fitness(
                program_id=signal.target_program_id,
                adjustment=signal.data.get('adjustment', 0),
                reason=signal.data.get('reason', ''),
            )
        elif signal.correction_type == 'mutation_suggestion':
            self.mstar_core.mutator.apply_suggested_mutation(
                program_id=signal.target_program_id,
                suggestion=signal.data,
            )
        elif signal.correction_type == 'strategy_change':
            self.mstar_core.evolution_engine.update_strategy(
                program_id=signal.target_program_id,
                new_strategy=signal.data.get('strategy'),
            )

    def on_evolution_completed(self, program, result: Dict, session_id: str):
        with self._lock:
            evolution_record = {
                'program_id': program.program_id,
                'mutation_type': result.get('mutation_type'),
                'fitness_delta': result.get('fitness_delta', 0),
                'success': result.get('success', False),
                'timestamp': datetime.now().isoformat(),
                'session_id': session_id,
            }

            self._evolution_history.append(evolution_record)

            if result.get('success'):
                reinforcement = ReinforcementSignal(
                    signal_type='evolution_success',
                    program_id=program.program_id,
                    metrics={'fitness_improvement': result.get('fitness_delta', 0), 'mutation_type': result.get('mutation_type')},
                    timestamp=datetime.now().isoformat(),
                )

                self._reinforcement_queue.append(reinforcement)
                self._reinforcements_sent += 1

                if self.agent_callback:
                    try:
                        self.agent_callback('mstar_evolution_success', {
                            'program_id': program.program_id,
                            'fitness_delta': result.get('fitness_delta', 0),
                            'session_id': session_id,
                        })
                    except Exception as e:
                        logger.warning(f"[MSTAR Bridge] Callback failed: {e}")

    def on_evolution(self, evolution_result: Dict, session_id: str):
        """[MSTAR Pro v4.0] Bridge entry point from run_agent.py.

        Called by the agent loop after each evolution cycle.
        Routes the result through the reinforcement pipeline.
        """
        with self._lock:
            # Extract events from the evolution result dict
            events = evolution_result.get('events', []) if isinstance(evolution_result, dict) else []
            for event in events:
                # Each event represents a program mutation/archive/delete
                if event.get('type') and event.get('program'):
                    program = event['program']
                    self._evolution_history.append({
                        'program_id': program.program_id if hasattr(program, 'program_id') else str(program),
                        'mutation_type': event.get('mutation_type'),
                        'fitness_delta': event.get('fitness_delta', 0),
                        'success': event.get('success', True),
                        'timestamp': event.get('timestamp', datetime.now().isoformat()),
                        'session_id': session_id,
                    })
            logger.info(f"[MSTAR Bridge] on_evolution: {len(events)} events, session={session_id}")
            self._reinforcements_sent += len(events)

    def on_observation(self, observation: Dict):
        if observation.get('type') == 'tool_execution':
            self.mstar_core.record_tool_execution(
                tool_name=observation.get('tool_name'),
                args=observation.get('args', {}),
                result=observation.get('result'),
                evaluation=observation.get('evaluation', {}),
                session_id=observation.get('session_id', ''),
            )
        elif observation.get('type') == 'fitness_feedback':
            signal = CorrectionSignal(
                source='homunculus',
                correction_type='fitness_adjustment',
                target_program_id=observation.get('program_id', ''),
                data={'adjustment': observation.get('adjustment', 0), 'reason': observation.get('reason', 'Homunculus observation')},
                timestamp=datetime.now().isoformat(),
            )
            self.receive_correction(signal)

    def get_bridge_status(self) -> Dict:
        return {
            'pending_corrections': len(self._pending_corrections),
            'evolution_history_size': len(self._evolution_history),
            'reinforcement_queue_size': len(self._reinforcement_queue),
            'corrections_received': self._corrections_received,
            'reinforcements_sent': self._reinforcements_sent,
            'last_evolution': self._evolution_history[-1] if self._evolution_history else None,
        }