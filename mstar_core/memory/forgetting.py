"""
MSTAR Pro v4.0 - 可解释遗忘机制
每个遗忘决策都附带人类可读的原因
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
from datetime import datetime


@dataclass
class ForgetCandidate:
    program_id: str
    name: str
    fitness_score: float
    age_days: int
    last_used: str
    failure_rate: float
    success_count: int
    total_attempts: int
    lineage_depth: int
    strategy: str


@dataclass
class ForgetDecision:
    strategy: str
    program_id: str
    explanation: str
    reasons: List[str]
    confidence: float
    timestamp: str
    fitness_score: float = None  # Bug-1 fix: 添加缺失字段，正确填充fitness_score


class ForgettingMechanism:
    """
    MSTAR Pro v4.0 智能遗忘 + 可解释性
    4种策略：keep/archive/merge/delete
    """

    def __init__(self, fitness_tracker, config: Optional[Dict] = None):
        self.fitness_tracker = fitness_tracker
        self.config = config or {}
        self.archive_threshold = self.config.get('archive_threshold', 0.25)
        self.merge_threshold = self.config.get('merge_threshold', 0.15)
        self.delete_threshold = self.config.get('delete_threshold', 0.08)
        # MSTAR Pro v4.0 P1-2: per-session metadata from ContextCompressor LLM compressions
        # Maps session_id -> {message_count, compressed_to_turns, compression_ratio, ...}
        self._session_metadata: Dict[str, Dict] = {}

    def decide(self, candidate: ForgetCandidate) -> Tuple[str, str]:
        forget_score = self._calculate_forget_score(candidate)
        strategy = self._determine_strategy(forget_score, candidate)
        explanation = self._generate_explanation(strategy, forget_score, candidate)
        return strategy, explanation

    def decide_detailed(self, candidate: ForgetCandidate) -> ForgetDecision:
        strategy, explanation = self.decide(candidate)
        return ForgetDecision(
            strategy=strategy,
            program_id=candidate.program_id,
            explanation=explanation,
            reasons=self._get_decision_reasons(candidate),
            confidence=self._calculate_confidence(candidate),
            timestamp=datetime.now().isoformat(),
            fitness_score=candidate.fitness_score,  # Bug-1 fix: 正确填充fitness_score
        )

    def _calculate_forget_score(self, candidate: ForgetCandidate) -> float:
        fitness_component = candidate.fitness_score
        # MSTAR Pro v4.0 P1-2: compression_ratio-aware recency
        # Low compression_ratio (many msgs -> few turns) = high information density
        # = LLM decided this session was important enough to PRESERVE in compressed form
        # = protect it longer (reduce effective age)
        # High compression_ratio = easy to summarize away = age faster
        base_recency = 1.0 - min(candidate.age_days / 90.0, 1.0)
        compression_ratio = getattr(candidate, 'compression_ratio', 1.0)
        # compression_ratio 1.0 = no compression, 0.1 = 10 messages compressed to 1
        # Low ratio = more protection = multiply recency by factor < 1
        density_factor = max(0.3, min(1.0, compression_ratio))
        recency_component = base_recency * (0.5 + 0.5 * density_factor)
        quality_component = candidate.success_count / candidate.total_attempts if candidate.total_attempts > 0 else 0.5
        return fitness_component * 0.4 + recency_component * 0.3 + quality_component * 0.3

    def record_session_metadata(self, session_id: str, message_count: int,
                                compressed_to_turns: int, compression_ratio: float,
                                avg_quality: float, total_tokens: int, duration: float):
        """Called by ContextCompressor after LLM summarization.

        Stores metadata so forgetting decisions can account for information density:
        - Low compression_ratio (many msgs -> few) = high-value preserved content
        - High compression_ratio = low-value summarizable content
        """
        self._session_metadata[session_id] = {
            'message_count': message_count,
            'compressed_to_turns': compressed_to_turns,
            'compression_ratio': compression_ratio,
            'avg_quality': avg_quality,
            'total_tokens': total_tokens,
            'duration': duration,
        }

    def get_session_compression_ratio(self, session_id: str) -> float:
        """Return compression ratio for a session, or 1.0 (no compression) if unknown."""
        return self._session_metadata.get(session_id, {}).get('compression_ratio', 1.0)

    def _determine_strategy(self, forget_score: float, candidate: ForgetCandidate) -> str:
        # Bug-6 fix: 刚演化的程序（last_evolution_within_threshold）给予保护期不归档
        import sqlite3
        db_path = getattr(self.fitness_tracker, 'db_path', None)
        if db_path:
            try:
                conn = sqlite3.connect(db_path, timeout=5)
                conn.row_factory = sqlite3.Row
                cur = conn.execute(
                    "SELECT last_evolution_at FROM programs WHERE program_id = ?",
                    (candidate.program_id,)
                )
                row = cur.fetchone()
                if row and row['last_evolution_at']:
                    last_evo = datetime.fromisoformat(row['last_evolution_at'])
                    days_since_evo = (datetime.now() - last_evo).days
                    if days_since_evo < 7:  # 演化后7天内不归档
                        conn.close()
                        return "keep"
                conn.close()
            except Exception:
                pass  # 静默处理，不阻断决策

        if candidate.lineage_depth > 10:
            return "archive"
        if forget_score >= self.archive_threshold:
            return "keep"
        elif forget_score >= self.merge_threshold:
            return "archive"
        elif forget_score >= self.delete_threshold:
            return "merge"
        return "delete"

    def _generate_explanation(self, strategy: str, forget_score: float, candidate: ForgetCandidate) -> str:
        reasons = []
        if strategy == "keep":
            reasons.append(f"高适应度 ({candidate.fitness_score:.3f})")
        elif strategy == "archive":
            reasons.append(f"中等适应度 ({candidate.fitness_score:.3f})")
            reasons.append(f"较久未使用 ({candidate.age_days}天)")
        elif strategy == "merge":
            reasons.append(f"低适应度 ({candidate.fitness_score:.3f})")
            reasons.append(f"高失败率 ({candidate.failure_rate:.1%})")
        elif strategy == "delete":
            reasons.append(f"极低适应度 ({candidate.fitness_score:.3f})")
            reasons.append(f"长期未使用 ({candidate.age_days}天)")
        return f"建议 {strategy}: " + ", ".join(reasons)

    def _get_decision_reasons(self, candidate: ForgetCandidate) -> List[str]:
        reasons = []
        if candidate.fitness_score < 0.2:
            reasons.append(f"极低适应度 ({candidate.fitness_score:.3f})")
        elif candidate.fitness_score < 0.4:
            reasons.append(f"低适应度 ({candidate.fitness_score:.3f})")
        if candidate.age_days > 30:
            reasons.append(f"长期未使用 ({candidate.age_days}天)")
        if candidate.failure_rate > 0.5:
            reasons.append(f"高失败率 ({candidate.failure_rate:.1%})")
        if candidate.lineage_depth > 10:
            reasons.append(f"血缘深度过高 ({candidate.lineage_depth})")
        return reasons

    def _calculate_confidence(self, candidate: ForgetCandidate) -> float:
        confidence = 0.5
        if candidate.total_attempts > 10:
            confidence += 0.2
        if candidate.age_days > 7:
            confidence += 0.1
        if candidate.fitness_score < 0.1 or candidate.fitness_score > 0.7:
            confidence += 0.15
        return min(confidence, 0.95)

    def run(self, all_candidates: List[ForgetCandidate] = None) -> List[ForgetDecision]:
        """
        Evaluate forgetting decisions for all programs (or provided candidates).
        Returns list of ForgetDecision objects for each program.
        """
        import sqlite3
        db_path = self.fitness_tracker.db_path
        conn = sqlite3.connect(db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        # If no candidates provided, build them from the database
        if all_candidates is None:
            cur.execute("""
                SELECT program_id, name, fitness_score, lineage_depth,
                       created_at, last_evolution_at
                FROM programs ORDER BY fitness_score ASC
            """)
            candidates = []
            for row in cur.fetchall():
                pid = row['program_id']
                # Get usage stats (tool_executions may not exist in older DBs)
                total = 0
                successes = 0
                _table_exists = conn.execute(
                    "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='tool_executions'"
                ).fetchone()[0] > 0
                if _table_exists:
                    cur2 = conn.execute(
                        "SELECT COUNT(*) as total, SUM(CASE WHEN success=1 THEN 1 ELSE 0 END) as successes "
                        "FROM tool_executions WHERE program_id = ?",
                        (pid,)
                    )
                    usage = cur2.fetchone()
                    total = usage['total'] if usage else 0
                    successes = usage['successes'] if usage else 0
                created = datetime.fromisoformat(row['created_at']) if row['created_at'] else datetime.now()
                last_evo = datetime.fromisoformat(row['last_evolution_at']) if row['last_evolution_at'] else None
                age_days = (datetime.now() - created).days
                failure_rate = (total - successes) / total if total > 0 else 0.0

                candidates.append(ForgetCandidate(
                    program_id=pid,
                    name=row['name'] or pid,
                    fitness_score=row['fitness_score'] or 0.5,
                    age_days=age_days,
                    last_used=row['last_evolution_at'] or row['created_at'] or "",
                    failure_rate=failure_rate,
                    success_count=successes,
                    total_attempts=total,
                    lineage_depth=row['lineage_depth'] or 0,
                    strategy="",
                ))

        conn.close()

        # Make decisions and apply lifecycle changes
        decisions = []
        changes = {"archived": [], "deleted": []}
        for candidate in candidates:
            decision = self.decide_detailed(candidate)
            decisions.append(decision)

            # Apply lifecycle_status changes to DB
            new_status = None
            if decision.strategy == "archive":
                new_status = "archived"
            elif decision.strategy == "delete":
                new_status = "deleted"

            if new_status:
                conn2 = sqlite3.connect(db_path, timeout=10)
                conn2.execute(
                    "UPDATE programs SET lifecycle_status = ? WHERE program_id = ?",
                    (new_status, decision.program_id)
                )
                conn2.commit()
                conn2.close()
                changes[new_status].append(decision.program_id)

        return decisions


def evaluate_all_forgetting(mc) -> List[Dict]:
    """
    Standalone function to run forgetting evaluation using an MSTARCore instance.
    Returns serializable dict list for API responses.
    """
    decisions = mc.forgetting_mechanism.run()
    return [
        {
            "program_id": d.program_id,
            "strategy": d.strategy,
            "explanation": d.explanation,
            "confidence": round(d.confidence, 4),
            "reasons": d.reasons,
            "timestamp": d.timestamp,
            "fitness_score": d.fitness_score,  # Bug-1 fix: 暴露fitness_score字段
        }
        for d in decisions
    ]