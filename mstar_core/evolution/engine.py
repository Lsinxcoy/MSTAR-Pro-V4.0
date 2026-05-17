"""
MSTAR Pro v4.0 - 自适应进化引擎
支持Dashboard Hook + 决策解释 + SelfImprovingBridge集成
+ ToolFitnessPredictor 集成（arxiv 2410.02725 自评估迁移）
"""

from __future__ import annotations
import logging
import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class EvolutionEvent:
    event_id: str
    event_type: str
    program_id: str
    timestamp: str
    details: Dict
    fitness_before: float
    fitness_after: float
    fitness_delta: float
    reason: str
    decision_explanation: str = ""


@dataclass
class EvolutionConfig:
    adaptive_interval_min: int = 3
    adaptive_interval_max: int = 50
    quality_gates: List[str] = field(default_factory=lambda: ['compile', 'runtime', 'logic', 'quality'])
    quality_mode: str = "standard"
    dashboard_hook_enabled: bool = True
    predictor_threshold: float = 0.40  # 预测器触发阈值（论文迁移）


class EvolutionEngine:
    """
    MSTAR Pro v4.0 自适应进化引擎

    核心改进：
    - Adaptive Interval: 根据成功率动态调整进化间隔
    - Dashboard Hook: 所有进化事件记录到ObservabilityDashboard
    - Decision Explanation: 每次决策都有人类可读的解释
    - SelfImprovingBridge: 双向通信集成
    - ToolFitnessPredictor: arxiv 2410.02725 自评估迁移——预测变异是否值得执行
    """

    def __init__(
        self,
        fitness_tracker,
        mutator=None,
        reflector=None,
        config: Optional[EvolutionConfig] = None,
        dashboard=None,
        self_improver=None,
        predictor=None,
    ):
        from mstar_core.evolution.mutator import MSTARMutator
        from mstar_core.evolution.reflector import MSTARReflector
        from mstar_core.evolution.predictor import RuleBasedPredictor

        self.fitness_tracker = fitness_tracker
        self.mutator = mutator or MSTARMutator()
        self.reflector = reflector or MSTARReflector()
        self.config = config or EvolutionConfig()
        self.dashboard = dashboard
        self.self_improver = self_improver

        # ── 预测器（arxiv 2410.02725 自评估迁移）────────────────
        # 默认 RuleBasedPredictor，可替换为 LLMJudge 或 Trained
        if predictor is None:
            predictor = RuleBasedPredictor(fitness_tracker)
        self.predictor = predictor

        self._cycles_run = 0
        self._lock = threading.RLock()
        self._current_interval = self.config.adaptive_interval_min
        self._success_count = 0
        self._failure_count = 0
        self._last_evolution_time = 0

    def should_trigger(self, sessions_processed: int = None) -> bool:
        """
        决定是否触发演化。使用 fitness_tracker 单一来源。
        """
        with self._lock:
            # 使用 DB 单一来源（而非传入参数）
            actual_sessions = self.fitness_tracker.get_sessions_processed()
            if actual_sessions == 0:
                return False
            # Trigger when session count is a multiple of current interval
            return (actual_sessions % self._current_interval == 0)

    def evaluate_session(self, session_id: str, stats: Dict) -> Dict:
        with self._lock:
            self._cycles_run += 1

            # ── Step 1: 候选工具选择（论文迁移：预测哪些值得变异）─────
            all_candidates = self.fitness_tracker.get_low_fitness_programs(
                threshold=0.35, limit=20
            )

            if not all_candidates:
                # 降级：扩展到全部工具（论文的"全覆盖"策略）
                all_candidates = self._get_all_programs()[:10]

            if not all_candidates:
                return {
                    'triggered': False,
                    'reason': 'No candidate programs for evolution',
                    'cycles_run': self._cycles_run,
                    'prediction_used': False,
                }

            # ── Step 2: 预测器决策（arxiv 2410.02725 核心迁移）─────────
            predicted = []
            for prog in all_candidates:
                best_list = self.predictor.predict_best_strategy(prog)
                if not best_list:
                    continue
                best_strategy, prob = best_list[0]
                predicted.append((prog, best_strategy, prob))

            if not predicted:
                return {
                    'triggered': False,
                    'reason': 'No programs pass predictor threshold',
                    'cycles_run': self._cycles_run,
                    'prediction_used': True,
                    'best_prob': 0.0,
                }

            # ── Step 3: 选预测收益最高的 ───────────────────────────
            target, strategy, prob = max(predicted, key=lambda x: x[2])
            threshold = getattr(self.config, 'predictor_threshold', 0.40)

            if prob < threshold:
                return {
                    'triggered': False,
                    'reason': f'Best predicted probability {prob:.3f} < threshold {threshold}',
                    'cycles_run': self._cycles_run,
                    'prediction_used': True,
                    'best_prob': round(prob, 3),
                    'target_program': target.program_id,
                }

            # ── Step 4: 执行变异 ─────────────────────────────────────
            mutation_result = self._execute_mutation(target, strategy=strategy)

            # ── Step 5: Dashboard Hook ──────────────────────────────
            if self.config.dashboard_hook_enabled and self.dashboard:
                evolution_event = EvolutionEvent(
                    event_id=f"evo_{session_id}_{self._cycles_run}",
                    event_type=mutation_result.get('mutation_type', 'mutation'),
                    program_id=target.program_id,
                    timestamp=datetime.now().isoformat(),
                    details=mutation_result,
                    fitness_before=mutation_result.get('fitness_before', 0),
                    fitness_after=mutation_result.get('fitness_after', 0),
                    fitness_delta=mutation_result.get('fitness_delta', 0),
                    reason=mutation_result.get('reason', ''),
                    decision_explanation=self._explain_evolution_decision(target, mutation_result, prob),
                )
                self.dashboard.record_evolution_event(evolution_event)

            # ── Step 6: SelfImprovingBridge 通知 ───────────────────
            if self.self_improver:
                try:
                    self.self_improver.on_evolution_completed(
                        program=target, result=mutation_result, session_id=session_id
                    )
                except Exception as e:
                    logger.warning(f"[MSTAR] SelfImprovingBridge notification failed: {e}")

            # ── Step 7: 反馈记录（用于 TrainedPredictor 后续训练）──
            # Bug-2 fix: predictor.record_outcome() 内部已调用 fitness_tracker.record_evolution_outcome()
            # 删掉 engine 侧多余的第二份写入，避免 evolution_outcomes 重复行
            if hasattr(self.predictor, 'record_outcome'):
                delta = mutation_result.get('fitness_delta', 0)
                self.predictor.record_outcome(
                    program_id=target.program_id,
                    strategy=strategy,
                    predicted_prob=prob,
                    actual_delta=delta,
                )

            # P0: 演化完成，记录到 DB 单一来源
            self.fitness_tracker.record_evolution_complete(
                session_index=self.fitness_tracker.get_sessions_processed()
            )

            self._update_adaptive_interval(mutation_result.get('success', False))

            return {
                'triggered': True,
                'program_id': target.program_id,
                'mutation_type': mutation_result.get('mutation_type'),
                'fitness_delta': mutation_result.get('fitness_delta', 0),
                'decision_explanation': self._explain_evolution_decision(target, mutation_result, prob),
                'cycles_run': self._cycles_run,
                'prediction_used': True,
                'predicted_prob': round(prob, 3),
                'strategy_used': strategy,
                'events': [{
                    'type': mutation_result.get('mutation_type', 'mutation'),
                    'program_id': target.program_id,
                    'fitness_delta': mutation_result.get('fitness_delta', 0),
                    'success': mutation_result.get('success', False),
                    'timestamp': datetime.now().isoformat(),
                }],
            }

    def _get_all_programs(self):
        """获取所有程序（用于无低fitness候选时的降级）"""
        try:
            import sqlite3
            conn = sqlite3.connect(self.fitness_tracker.db_path, timeout=10)
            cur = conn.execute(
                "SELECT program_id, name, fitness_score, lineage_depth FROM programs ORDER BY fitness_score LIMIT 20"
            )
            rows = cur.fetchall()
            conn.close()
            from mstar_core.evolution.fitness_tracker import MemoryProgram
            return [
                MemoryProgram(program_id=r[0], name=r[1] or "", fitness_score=r[2] or 0.5, lineage_depth=r[3] or 0)
                for r in rows
            ]
        except Exception as e:
            logger.warning(f"[EvolutionEngine] _get_all_programs failed: {e}")
            return []

    def _select_mutation_target(self, candidates):
        """旧版选择逻辑（保留作为fallback）"""
        if not candidates:
            return None
        return min(candidates, key=lambda p: (p.fitness_score, getattr(p, 'last_evolution_time', 0) or 0))

    def _execute_mutation(self, program, strategy=None):
        if not program:
            return {'success': False, 'reason': 'No program to mutate'}

        failure_analysis = self.reflector.analyze_failures(program)
        strategy = strategy or failure_analysis.get('recommended_strategy', 'random')

        # ── Lim-3: 从 failure_analysis 提取 failure_type 并写回 program ──
        failure_type_detected = failure_analysis.get('failure_type', 'no_data')
        if hasattr(program, 'failure_type'):
            program.failure_type = failure_type_detected
            from datetime import datetime
            program.last_failure_at = datetime.now().isoformat()

        # 记录变异前状态（Bug-4: 用于 rollback 熔断）
        import copy
        state_before = {
            'fitness_score': program.fitness_score,
            'lineage_depth': getattr(program, 'lineage_depth', 0),
            'parent_id': getattr(program, 'parent_id', None),
        }

        # Bug-4 fix: 在变异前用预测器获取 predicted_prob（Mutator 不设置此字段）
        predicted_prob = 0.5
        if hasattr(self, 'predictor') and self.predictor:
            try:
                predicted_prob = self.predictor.predict_mutation_benefit(program, strategy)
            except Exception:
                predicted_prob = 0.5

        mutation_result = self.mutator.mutate(program, strategy=strategy)

        # 记录变异前fitness（用于Dashboard快照和反馈记录）
        fitness_before = program.fitness_score

        if mutation_result.success and mutation_result.new_fitness is not None:
            program.fitness_score = mutation_result.new_fitness
            # ── BUG FIX: 变异后 lineage_depth 应增加 ─────────────
            program.lineage_depth = (program.lineage_depth or 0) + 1
            # ── BUG FIX 13: 同时更新 parent_id（防止进化后 parent_id 仍为 None）────
            #   如果程序是从 DB 加载的，parent_id 可能为 None；设为自身 ID 表示"无祖先"
            if getattr(program, 'parent_id', None) is None:
                program.parent_id = program.program_id

            fitness_after = mutation_result.new_fitness
            fitness_delta = fitness_after - fitness_before

            # ── Bug-4: 熔断机制 - 预测概率 > 0.5 但 fitness 反而下降则 rollback ──
            # predicted_prob 来自变异前的预测器输出（已在上方计算）
            if predicted_prob > 0.5 and fitness_delta < 0:
                logger.warning(
                    f"[Bug-4 Rollback] prog={program.program_id} "
                    f"predicted={predicted_prob:.3f} actual_delta={fitness_delta:.4f} → REVERTING"
                )
                # 回滚到变异前状态
                program.fitness_score = state_before['fitness_score']
                program.lineage_depth = state_before['lineage_depth']
                program.parent_id = state_before['parent_id']
                mutation_result = copy.deepcopy(mutation_result)
                mutation_result.success = False
                mutation_result.reason = f"[Rollback] predicted={predicted_prob:.3f} but fitness dropped {fitness_delta:.4f}"
                mutation_result.new_fitness = None
                fitness_after = fitness_before
                fitness_delta = 0.0

            # 保存更新后的 lineage_depth 到 DB
            if hasattr(self.fitness_tracker, '_save_program'):
                self.fitness_tracker._save_program(program)
        else:
            fitness_after = mutation_result.new_fitness if mutation_result.new_fitness is not None else fitness_before
            fitness_delta = fitness_after - fitness_before

        return {
            **mutation_result.__dict__,
            'fitness_before': fitness_before,
            'fitness_after': fitness_after,
            'fitness_delta': fitness_delta,
            'predicted_prob': predicted_prob,   # Bug-4 fix: 传递预测概率供 record_outcome 使用
            'rollback': fitness_delta == 0 and predicted_prob > 0.5,
        }

    def _explain_evolution_decision(self, program, result: Dict, predicted_prob: float = None) -> str:
        """生成人类可读的决策解释（含预测概率）"""
        reasons = []
        if program and hasattr(program, 'fitness_score'):
            if program.fitness_score < 0.2:
                reasons.append(f"极低适应度 ({program.fitness_score:.3f})")
            elif program.fitness_score < 0.3:
                reasons.append(f"低适应度 ({program.fitness_score:.3f})")
        if result.get('mutation_type'):
            reasons.append(f"变异类型: {result['mutation_type']}")
        if result.get('fitness_delta', 0) > 0:
            reasons.append(f"预期提升: +{result['fitness_delta']:.3f}")
        if predicted_prob is not None:
            reasons.append(f"预测收益概率: {predicted_prob:.1%}")
        return " | ".join(reasons) if reasons else "常规优化"

    def _update_adaptive_interval(self, success: bool):
        if success:
            self._success_count += 1
            self._failure_count = 0
        else:
            self._failure_count += 1
            self._success_count = 0

        if self._failure_count > 3:
            self._current_interval = min(
                self._current_interval * 1.5,
                self.config.adaptive_interval_max
            )
        elif self._success_count > 3:
            self._current_interval = max(
                self._current_interval * 0.8,
                self.config.adaptive_interval_min
            )