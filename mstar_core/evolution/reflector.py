"""
MSTAR Pro v4.0 - 失败模式分析 + 决策解释
8种失败模式识别 + 针对性变异建议
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Dict, List

logger = logging.getLogger(__name__)


@dataclass
class FailureAnalysis:
    failure_type: str
    confidence: float
    evidence: List[str]
    recommended_strategy: str
    explanation: str


class MSTARReflector:
    """
    MSTAR Pro v4.0 失败模式分析 + 决策解释
    识别8种失败模式
    """

    FAILURE_PATTERNS = [
        'schema_mismatch', 'logic_error', 'quality_degradation',
        'token_overflow', 'latency_spike', 'error_accumulation',
        'context_drift', 'strategy_stagnation',
    ]

    STRATEGY_MAP = {
        'schema_mismatch': 'schema_field_modify',
        'logic_error': 'logic_read_modify',
        'quality_degradation': 'instruction_guidance',
        'token_overflow': 'instruction_threshold',
        'latency_spike': 'logic_query_add',
        'error_accumulation': 'instruction_priority',
        'context_drift': 'instruction_context',
        'strategy_stagnation': 'combo_crossover',
    }

    def analyze_failures(self, program) -> Dict:
        if program is None:
            return {
                'failure_type': 'unknown', 'confidence': 0.0,
                'recommended_strategy': 'random', 'explanation': 'No program to analyze',
            }

        episodes = getattr(program, 'episodes', [])

        if not episodes:
            return {
                'failure_type': 'no_data', 'confidence': 0.5,
                'recommended_strategy': 'random', 'explanation': 'No episode history',
            }

        failure_type, confidence = self._detect_failure_pattern(program)
        recommended_strategy = self.STRATEGY_MAP.get(failure_type, 'random_change')
        explanation = self._generate_explanation(failure_type, confidence, recommended_strategy, program)

        return {
            'failure_type': failure_type, 'confidence': confidence,
            'recommended_strategy': recommended_strategy, 'explanation': explanation,
            'evidence': self._collect_evidence(program),
        }

    def _detect_failure_pattern(self, program) -> tuple:
        episodes = getattr(program, 'episodes', [])

        if not episodes:
            return 'no_data', 0.5

        recent_success = sum(1 for e in episodes[-10:] if e.get('success', False))
        if recent_success < 3:
            return 'quality_degradation', 0.8

        latencies = [e.get('latency', 0) for e in episodes[-5:]]
        if latencies and max(latencies) > latencies[0] * 2:
            return 'latency_spike', 0.7

        tokens = [e.get('tokens', 0) for e in episodes[-5:]]
        if tokens and max(tokens) > 10000:
            return 'token_overflow', 0.75

        recent_errors = sum(1 for e in episodes[-10:] if not e.get('success', True))
        if recent_errors > 5:
            return 'error_accumulation', 0.85

        return 'strategy_stagnation', 0.6

    def _collect_evidence(self, program) -> List[str]:
        evidence = []
        if hasattr(program, 'fitness_score'):
            evidence.append(f"Fitness: {program.fitness_score:.3f}")
        episodes = getattr(program, 'episodes', [])
        if episodes:
            recent_success = sum(1 for e in episodes[-10:] if e.get('success', False))
            evidence.append(f"Recent success rate: {recent_success}/10")
        if hasattr(program, 'lineage_depth'):
            evidence.append(f"Lineage depth: {program.lineage_depth}")
        return evidence

    def _generate_explanation(self, failure_type: str, confidence: float, strategy: str, program) -> str:
        explanations = {
            'schema_mismatch': '检测到Schema与实际数据不匹配，建议修改字段结构',
            'logic_error': '检测到逻辑错误，建议调整读写逻辑',
            'quality_degradation': '检测到质量逐步下降，建议增强指导',
            'token_overflow': '检测到Token使用超出预算，建议调整阈值',
            'latency_spike': '检测到延迟突然增加，建议优化查询',
            'error_accumulation': '检测到错误不断累积，建议调整优先级',
            'context_drift': '检测到上下文漂移，建议增强上下文处理',
            'strategy_stagnation': '检测到策略停滞，建议尝试交叉变异',
        }

        base = explanations.get(failure_type, '未知失败模式')
        confidence_desc = '高置信度' if confidence > 0.8 else ('中等置信度' if confidence > 0.6 else '低置信度')

        return f"{base}（{confidence_desc}，建议策略: {strategy}）"

    def get_decision_explanation(self, program, decision: str) -> str:
        return f"MSTAR决策: {decision} | Program: {getattr(program, 'program_id', 'unknown')} | Fitness: {getattr(program, 'fitness_score', 0):.3f}"