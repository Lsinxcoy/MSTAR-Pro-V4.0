"""
MSTAR Pro v4.0 - 55维适应度追踪器
支持10/20/55维三档模式
"""

from __future__ import annotations
import logging
import time
import sqlite3
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
from statistics import mean

logger = logging.getLogger(__name__)


@dataclass
class FitnessDimensions:
    mode: str = "standard"

    BEGINNER_dims = [
        'success_rate', 'quality_score', 'confidence',
        'latency_p50', 'token_efficiency', 'error_rate',
        'user_satisfaction', 'task_completion', 'context_relevance', 'evolution_stability',
    ]

    STANDARD_dims = BEGINNER_dims + [
        'latency_p90', 'throughput', 'cache_hit_rate',
        'memory_utilization', 'fitness_trend', 'volatility',
        'cross_task_generalization', 'instruction_following', 'creativity_score', 'safety_score',
    ]

    ADVANCED_dims = STANDARD_dims + [
        'reasoning_depth', 'planning_quality', 'self_correction_rate',
        'meta_learning_speed', 'adaptation_flexibility', 'communication_clarity',
        'emotional_intelligence', 'ethical_reasoning', 'common_sense_score', 'world_knowledge',
        'curiosity_index', 'exploration_exploitation', 'long_term_memory', 'short_term_memory',
        'attention_focus', 'multitasking_efficiency', 'learning_from_errors', 'bias_detection',
        'uncertainty_quantification', 'risk_assessment', 'decision_quality', 'action_alignment',
    ]

    @property
    def dimensions(self) -> List[str]:
        if self.mode == "beginner":
            return self.BEGINNER_dims
        elif self.mode == "standard":
            return self.STANDARD_dims
        return self.ADVANCED_dims

    @property
    def num_dimensions(self) -> int:
        return len(self.dimensions)


@dataclass
class MemoryProgram:
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


class FitnessTracker:
    """
    MSTAR Pro v4.0 适应度追踪器
    三档维度模式 (10/20/55)
    SQLite WAL持久化
    """

    def __init__(self, hermes_home: str, mode: str = "standard", db_path: Optional[str] = None):
        import os
        self.hermes_home = hermes_home
        self.mode = mode
        self.dimensions = FitnessDimensions(mode=mode)

        os.makedirs(hermes_home, exist_ok=True)
        self.db_path = db_path or f"{hermes_home}/mstar_fitness.db"
        self._lock = threading.RLock()
        self._init_database()
        self._program_cache: Dict[str, MemoryProgram] = {}

    def _init_database(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS programs (
                program_id TEXT PRIMARY KEY,
                name TEXT,
                lineage_depth INTEGER DEFAULT 0,
                parent_id TEXT,
                created_at TEXT,
                last_evolution_at TEXT,
fitness_score REAL,
                fitness_history TEXT,
                explanation_cache TEXT,
                lifecycle_status TEXT DEFAULT 'active',
                episodes INTEGER DEFAULT 0,
                failure_type TEXT,
                last_failure_at TEXT
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS fitness_snapshots (
                snapshot_id TEXT PRIMARY KEY,
                program_id TEXT NOT NULL,
                program_name TEXT,
                timestamp TEXT NOT NULL,
                fitness_score REAL,
                success_rate REAL,
                quality_score REAL,
                ema_10 REAL,
                ema_50 REAL,
                trend_slope REAL,
                decision_explanation TEXT
            )
        """)

        # ── 单一真实来源表：演化状态计数（修复 sessions vs _cycles 分裂问题）─────────
        conn.execute("""
            CREATE TABLE IF NOT EXISTS evolution_state (
                key TEXT PRIMARY KEY,
                value INTEGER NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        # 初始化关键状态键（如果不存在）
        for key in ('sessions_processed', 'last_evolution_session_index', 'evolutions_completed',
                    'last_retention_session_index'):
            conn.execute("""
                INSERT OR IGNORE INTO evolution_state (key, value, updated_at)
                VALUES (?, 0, ?)
            """, (key, datetime.now().isoformat()))

        # ── 策略效果记录表（用于 TrainedPredictor / 规则预测器学习）───────────────
        conn.execute("""
            CREATE TABLE IF NOT EXISTS evolution_outcomes (
                outcome_id TEXT PRIMARY KEY,
                program_id TEXT NOT NULL,
                strategy TEXT NOT NULL,
                predicted_prob REAL,
                actual_delta REAL,
                improved INTEGER,
                timestamp TEXT NOT NULL
            )
        """)

        # ── 遗忘归档表（memory_archives 表）─────────────────────────────────────
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memory_archives (
                program_id TEXT PRIMARY KEY,
                archived_at TEXT NOT NULL,
                deleted_at TEXT,
                original_lineage_depth INTEGER,
                fitness_at_archive REAL,
                lifecycle_status TEXT DEFAULT 'archived',
                memory_snapshot TEXT
)
        """)

        conn.commit()
        conn.close()

    def _get_or_create_program(self, program_id: str) -> MemoryProgram:
        if program_id in self._program_cache:
            return self._program_cache[program_id]

        conn = sqlite3.connect(self.db_path, timeout=30)
        cursor = conn.execute(
            "SELECT program_id, name, fitness_score, lineage_depth FROM programs WHERE program_id = ?",
            (program_id,)
        )
        row = cursor.fetchone()
        conn.close()

        if row:
            program = MemoryProgram(program_id=row[0], name=row[1] or "", fitness_score=row[2] or 0.5, lineage_depth=row[3] or 0)
        else:
            program = MemoryProgram(program_id=program_id, name=program_id, fitness_score=0.5, lineage_depth=0, created_at=datetime.now().isoformat())
            self._save_program(program)

        self._program_cache[program_id] = program
        return program

    def create_program(self, program_id: str, name: str) -> MemoryProgram:
        program = MemoryProgram(program_id=program_id, name=name, fitness_score=0.5, lineage_depth=0, created_at=datetime.now().isoformat())
        self._save_program(program)
        self._program_cache[program_id] = program
        return program

    def _save_program(self, program: MemoryProgram):
        with self._lock:
            conn = sqlite3.connect(self.db_path, timeout=30)
            conn.execute("""
                INSERT OR REPLACE INTO programs
                (program_id, name, parent_id, lineage_depth, fitness_score, fitness_history,
                 explanation_cache, created_at, last_evolution_at, lifecycle_status,
                 episodes, failure_type, last_failure_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                program.program_id,
                program.name,
                getattr(program, 'parent_id', None),
                program.lineage_depth,
                program.fitness_score,
                str(program.fitness_history),
                str(program.explanation_cache),
                program.created_at,
                getattr(program, 'last_evolution_at', None) or datetime.now().isoformat(),
                getattr(program, 'lifecycle_status', 'active'),
                len(program.episodes),
                getattr(program, 'failure_type', None),
                getattr(program, 'last_failure_at', None),
            ))
            conn.commit()
            conn.close()

    def update(self, program_id: str, success: bool, quality: float, latency: float, tokens_used: int, **metrics) -> Optional[MemoryProgram]:
        with self._lock:
            program = self._get_or_create_program(program_id)

            dimensions = self._calculate_dimensions(success=success, quality=quality, latency=latency, tokens_used=tokens_used, **metrics)

            program.update_fitness(
                episode={'success': success, 'quality': quality, 'latency': latency, 'tokens': tokens_used, 'timestamp': time.time()},
                dimensions=dimensions,
)

            self._save_program(program)
            self._program_cache[program_id] = program

            # ── 快照记录（MSTAR Pro v4.0: 每次 update 都写快照）─────────
            self._write_snapshot(
                program_id=program_id,
                program=program,
                success=success,
                quality=quality,
                latency=latency,
                tokens_used=tokens_used,
            )

            return program

    def _write_snapshot(self, program_id: str, program: MemoryProgram,
                         success: bool, quality: float, latency: float,
                         tokens_used: int) -> None:
        """
        每次 fitness 更新都写入 fitness_snapshots 表。
        论文迁移：类比 2410.02725 每次生成都记录质量信号，
        这里每次工具执行后记录性能快照，为预测器提供历史数据。
        """
        import uuid
        try:
            conn = sqlite3.connect(self.db_path, timeout=30)
            conn.execute("""
                INSERT OR IGNORE INTO fitness_snapshots
                (snapshot_id, program_id, program_name, timestamp,
                 fitness_score, success_rate, quality_score, ema_10, ema_50,
                 trend_slope, decision_explanation)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                f"snap_{uuid.uuid4().hex[:12]}",
                program_id,
                program.name or program_id,
                datetime.now().isoformat(),
                program.fitness_score,
                1.0 if success else 0.0,
                min(1.0, max(0.0, quality / 100.0)),
                program.fitness_score,
                program.fitness_score,
                0.0,
                f"tool_execution: success={success}, quality={quality:.1f}, latency={latency:.3f}s",
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"[FitnessTracker] Failed to write snapshot for {program_id}: {e}")

    def batch_update(self, updates: List[Dict[str, Any]]) -> List[Optional[MemoryProgram]]:
        """Batch-update multiple programs in a SINGLE database transaction.

        MSTAR Pro v4.0 P1 improvement: fitness writes from batch_execute (N sub-tools)
        are merged into one DB commit instead of N separate commits, reducing I/O.

        Args:
            updates: List of dicts with keys: program_id, success, quality, latency, tokens_used

        Returns:
            List of updated MemoryProgram objects (same order as input)
        """
        if not updates:
            return []

        results: List[Optional[MemoryProgram]] = []

        with self._lock:
            # Phase 1: create missing programs (without saving yet)
            for upd in updates:
                pid = upd.get('program_id')
                if pid and pid not in self._program_cache:
                    program = MemoryProgram(
                        program_id=pid, name=pid, fitness_score=0.5,
                        lineage_depth=0, created_at=datetime.now().isoformat(),
                    )
                    self._program_cache[pid] = program

            # Phase 2: compute dimensions for all updates
            for upd in updates:
                pid = upd.get('program_id', f"prog_unknown_{id(upd)}")
                program = self._program_cache.get(pid)
                if not program:
                    program = MemoryProgram(program_id=pid, name=pid)
                    self._program_cache[pid] = program

                dimensions = self._calculate_dimensions(
                    success=upd.get('success', False),
                    quality=upd.get('quality', 80.0),
                    latency=upd.get('latency', 1.0),
                    tokens_used=upd.get('tokens_used', 0),
                )
                program.update_fitness(
                    episode={
                        'success': upd.get('success', False),
                        'quality': upd.get('quality', 80.0) / 100.0,
                        'latency': upd.get('latency', 1.0),
                        'tokens': upd.get('tokens_used', 0),
                        'timestamp': time.time(),
                        'batch': True,
                    },
                    dimensions=dimensions,
                )

            # Phase 3: single DB transaction saves ALL programs at once
            conn = sqlite3.connect(self.db_path, timeout=30)
            conn.execute('BEGIN IMMEDIATE')
            try:
                for upd in updates:
                    pid = upd.get('program_id', '')
                    program = self._program_cache.get(pid)
                    if not program:
                        continue
                    conn.execute('''
                        INSERT OR REPLACE INTO programs
                        (program_id, name, lineage_depth, fitness_score, fitness_history, explanation_cache, created_at, last_evolution_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        program.program_id, program.name, program.lineage_depth,
                        program.fitness_score, str(program.fitness_history),
                        str(program.explanation_cache), program.created_at,
                        program.last_evolution_at,
))
                    results.append(program)

                    # ── BUG FIX: batch_update 也应写快照 ─────────────
                    self._write_snapshot(
                        program_id=program.program_id,
                        program=program,
                        success=upd.get('success', False),
                        quality=upd.get('quality', 80.0),
                        latency=upd.get('latency', 1.0),
                        tokens_used=upd.get('tokens_used', 0),
                    )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

        return results


    def _calculate_dimensions(self, success: bool, quality: float, latency: float, tokens_used: int, **metrics) -> Dict[str, float]:
        dims = {}
        dims['success_rate'] = 1.0 if success else 0.0
        dims['quality_score'] = min(1.0, quality / 100.0)
        dims['confidence'] = metrics.get('confidence', 0.5)
        dims['latency_p50'] = self._normalize_latency(latency, p=50)
        dims['token_efficiency'] = self._calculate_token_efficiency(tokens_used, success)
        dims['error_rate'] = 0.0 if success else 1.0

        if self.dimensions.mode in ('standard', 'advanced'):
            dims['latency_p90'] = self._normalize_latency(latency * 1.5, p=90)
            dims['throughput'] = 1.0 / max(latency, 0.1)
            dims['cache_hit_rate'] = metrics.get('cache_hit', 0.5)
            dims['memory_utilization'] = metrics.get('mem_util', 0.5)

        return dims

    def _normalize_latency(self, latency: float, p: int = 50) -> float:
        if latency <= 0.1: return 1.0
        elif latency <= 1.0: return 0.9
        elif latency <= 5.0: return 0.7
        elif latency <= 30.0: return 0.5
        return 0.2

    def _calculate_token_efficiency(self, tokens: int, success: bool) -> float:
        if tokens == 0: return 0.5
        if success: return min(1.0, 5000 / tokens)
        return min(0.5, 2500 / tokens)

    def get_low_fitness_programs(self, threshold: float = 0.3, limit: int = 5) -> List[MemoryProgram]:
        conn = sqlite3.connect(self.db_path, timeout=30)
        cursor = conn.execute("""
            SELECT program_id, name, fitness_score, lineage_depth FROM programs
            WHERE fitness_score < ?
            ORDER BY fitness_score ASC
            LIMIT ?
        """, (threshold, limit))

        rows = cursor.fetchall()
        conn.close()

        programs = []
        for row in rows:
            pid = row[0]
            if pid in self._program_cache:
                programs.append(self._program_cache[pid])
            else:
                program = MemoryProgram(program_id=pid, name=row[1] or "", fitness_score=row[2] or 0.5, lineage_depth=row[3] or 0)
                programs.append(program)
        return programs

    def get_high_fitness_programs(self, threshold: float = 0.6, limit: int = 3) -> List[MemoryProgram]:
        conn = sqlite3.connect(self.db_path, timeout=30)
        cursor = conn.execute("""
            SELECT program_id, name, fitness_score, lineage_depth FROM programs
            WHERE fitness_score >= ?
            ORDER BY fitness_score DESC
            LIMIT ?
        """, (threshold, limit))

        rows = cursor.fetchall()
        conn.close()

        programs = []
        for row in rows:
            pid = row[0]
            if pid in self._program_cache:
                programs.append(self._program_cache[pid])
            else:
                program = MemoryProgram(program_id=pid, name=row[1] or "", fitness_score=row[2] or 0.5, lineage_depth=row[3] or 0)
                programs.append(program)
        return programs

    def program_exists(self, program_id: str) -> bool:
        if program_id in self._program_cache:
            return True
        conn = sqlite3.connect(self.db_path, timeout=30)
        cursor = conn.execute("SELECT 1 FROM programs WHERE program_id = ?", (program_id,))
        exists = cursor.fetchone() is not None
        conn.close()
        return exists

    def adjust_fitness(self, program_id: str, adjustment: float, reason: str):
        program = self._get_or_create_program(program_id)
        old_score = program.fitness_score
        program.fitness_score = max(0.0, min(1.0, program.fitness_score + adjustment))
        program.explanation_cache[f'adjustment_{datetime.now().isoformat()}'] = reason
        self._save_program(program)
        logger.info(f"[MSTAR] Fitness adjusted for {program_id}: {old_score:.3f} -> {program.fitness_score:.3f}")
    def batch_evaluate(self, program_updates):
        if not program_updates:
            return []
        results = []
        with self._lock:
            programs = {}
            for upd in program_updates:
                pid = upd.get('program_id', '')
                if pid not in programs:
                    programs[pid] = self._get_or_create_program(pid)
            for upd in program_updates:
                pid = upd.get('program_id', '')
                program = programs.get(pid)
                if not program:
                    continue
                mutation = upd.get('mutation_result')
                if mutation is None:
                    results.append({'program_id': pid, 'success': False, 'new_fitness': program.fitness_score, 'fitness_delta': 0.0, 'reason': 'No mutation provided'})
                    continue
                old_fitness = program.fitness_score
                new_fitness = getattr(mutation, 'new_fitness', None)
                if new_fitness is None and isinstance(mutation, dict):
                    new_fitness = mutation.get('new_fitness')
                success = getattr(mutation, 'success', False)
                if isinstance(mutation, dict):
                    success = mutation.get('success', False)
                if new_fitness is not None:
                    program.fitness_score = max(0.0, min(1.0, new_fitness))
                mutation_type = getattr(mutation, 'mutation_type', '') or (mutation.get('mutation_type') if isinstance(mutation, dict) else '')
                program.explanation_cache[f'evolution_{datetime.now().isoformat()}'] = f"{mutation_type}: {upd.get('reason', 'batch evaluation')}"
                fitness_delta = program.fitness_score - old_fitness
                results.append({'program_id': pid, 'success': success, 'new_fitness': program.fitness_score, 'fitness_delta': fitness_delta, 'reason': upd.get('reason', '')})
            conn = sqlite3.connect(self.db_path, timeout=30)
            conn.execute('BEGIN IMMEDIATE')
            try:
                for pid, program in programs.items():
                    conn.execute('''
                        INSERT OR REPLACE INTO programs
                        (program_id, name, lineage_depth, fitness_score, fitness_history, explanation_cache, created_at, last_evolution_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (program.program_id, program.name, program.lineage_depth, program.fitness_score, str(program.fitness_history), str(program.explanation_cache), program.created_at, datetime.now().isoformat()))
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()
        return results


    def get_statistics(self) -> Dict:
        conn = sqlite3.connect(self.db_path, timeout=30)
        cursor = conn.execute("""
            SELECT COUNT(*), AVG(fitness_score), MIN(fitness_score), MAX(fitness_score)
            FROM programs
        """)
        row = cursor.fetchone()
        conn.close()
        return {
            'total_programs': row[0] or 0,
            'avg_fitness': row[1] or 0.5,
            'min_fitness': row[2] or 0.0,
'max_fitness': row[3] or 1.0,
        }

    def get_all_fitness(self, limit=20):
        with self._lock:
            conn = sqlite3.connect(self.db_path, timeout=30)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                'SELECT program_id, name, fitness_score FROM programs ORDER BY fitness_score DESC LIMIT ?',
                (limit,)
            ).fetchall()
            conn.close()
            results = []
            for row in rows:
                p = self._get_or_create_program(row['program_id'])
                p.fitness_score = row['fitness_score']
                p.name = row['name']
                results.append(p)
            return results

    # ═══════════════════════════════════════════════════════════════════════════
    # 单一真实来源方法（P0 核心修复：sessions vs _cycles 分裂问题）
    # 所有 session/evolution 计数都从这些方法访问，不在内存对象中维护
    # ═══════════════════════════════════════════════════════════════════════════

    def increment_sessions(self) -> int:
        """
        原子增加 sessions_processed，返回新的计数值。
        这是 on_session_end 的唯一调用点，全系统唯一真实来源。
        """
        with self._lock:
            conn = sqlite3.connect(self.db_path, timeout=30)
            now = datetime.now().isoformat()
            conn.execute(
                "INSERT INTO evolution_state (key, value, updated_at) VALUES (?, 1, ?)"
                " ON CONFLICT(key) DO UPDATE SET value = value + 1, updated_at = ?",
                ('sessions_processed', now, now)
            )
            conn.commit()
            row = conn.execute(
                "SELECT value FROM evolution_state WHERE key = ?", ('sessions_processed',)
            ).fetchone()
            conn.close()
            return row[0] if row else 1

    def get_sessions_processed(self) -> int:
        """读取当前 sessions_processed（单一来源）"""
        with self._lock:
            conn = sqlite3.connect(self.db_path, timeout=30)
            row = conn.execute(
                "SELECT value FROM evolution_state WHERE key = ?", ('sessions_processed',)
            ).fetchone()
            conn.close()
            return row[0] if row else 0

    def get_sessions_since_last_evolution(self) -> int:
        """全系统唯一的 sessions_since_last_evolution 计算方式"""
        with self._lock:
            conn = sqlite3.connect(self.db_path, timeout=30)
            sp_row = conn.execute(
                "SELECT value FROM evolution_state WHERE key = ?", ('sessions_processed',)
            ).fetchone()
            le_row = conn.execute(
                "SELECT value FROM evolution_state WHERE key = ?", ('last_evolution_session_index',)
            ).fetchone()
            conn.close()
            sessions = sp_row[0] if sp_row else 0
            last_evo = le_row[0] if le_row else 0
            return max(0, sessions - last_evo)

    def record_evolution_complete(self, session_index: int = None):
        """
        演化完成后原子更新状态：
        1. evolutions_completed += 1
        2. last_evolution_session_index = 当前 session_index
        """
        with self._lock:
            if session_index is None:
                session_index = self.get_sessions_processed()
            now = datetime.now().isoformat()
            conn = sqlite3.connect(self.db_path, timeout=30)
            # evolutions_completed += 1
            conn.execute(
                "INSERT INTO evolution_state (key, value, updated_at) VALUES (?, 1, ?)"
                " ON CONFLICT(key) DO UPDATE SET value = value + 1, updated_at = ?",
                ('evolutions_completed', now, now)
            )
            # last_evolution_session_index
            conn.execute(
                "INSERT INTO evolution_state (key, value, updated_at) VALUES (?, ?, ?)"
                " ON CONFLICT(key) DO UPDATE SET value = ?, updated_at = ?",
                ('last_evolution_session_index', session_index, now, session_index, now)
            )
            conn.commit()
            conn.close()

    def get_evolutions_completed(self) -> int:
        """读取已完成的演化次数"""
        with self._lock:
            conn = sqlite3.connect(self.db_path, timeout=30)
            row = conn.execute(
                "SELECT value FROM evolution_state WHERE key = ?", ('evolutions_completed',)
            ).fetchone()
            conn.close()
            return row[0] if row else 0

    def record_evolution_outcome(self, program_id: str, strategy: str,
                                  predicted_prob: float, actual_delta: float):
        """持久化演化结果，供预测器学习"""
        import uuid
        with self._lock:
            conn = sqlite3.connect(self.db_path, timeout=30)
            conn.execute("""
                INSERT OR REPLACE INTO evolution_outcomes
                (outcome_id, program_id, strategy, predicted_prob, actual_delta, improved, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                f"evo_{uuid.uuid4().hex[:12]}",
                program_id,
                strategy,
                predicted_prob,
                actual_delta,
                1 if actual_delta > 0 else 0,
                datetime.now().isoformat(),
            ))
            conn.commit()
            conn.close()

    def archive_program(self, program_id: str, memory_snapshot: str = None) -> bool:
        """将程序标记为 archived 并写入 memory_archives 表"""
        with self._lock:
            conn = sqlite3.connect(self.db_path, timeout=30)
            # 获取程序当前快照数据
            prog_row = conn.execute(
                "SELECT lineage_depth, fitness_score FROM programs WHERE program_id = ?",
                (program_id,)
            ).fetchone()
            if not prog_row:
                conn.close()
                return False

            # 更新 programs.lifecycle_status
            conn.execute(
                "UPDATE programs SET lifecycle_status = 'archived' WHERE program_id = ?",
                (program_id,)
            )
            # 写入归档表
            conn.execute("""
                INSERT OR REPLACE INTO memory_archives
                (program_id, archived_at, original_lineage_depth, fitness_at_archive, lifecycle_status, memory_snapshot)
                VALUES (?, ?, ?, ?, 'archived', ?)
            """, (
                program_id,
                datetime.now().isoformat(),
                prog_row[0],
                prog_row[1],
                memory_snapshot,
            ))
            conn.commit()
            conn.close()
            return True

    def delete_program(self, program_id: str) -> bool:
        """彻底删除程序（从 programs 表物理删除）"""
        with self._lock:
            conn = sqlite3.connect(self.db_path, timeout=30)
            # 先写入 deleted 状态到 archives
            prog_row = conn.execute(
                "SELECT lineage_depth, fitness_score FROM programs WHERE program_id = ?",
                (program_id,)
            ).fetchone()
            if not prog_row:
                conn.close()
                return False
            conn.execute("""
                INSERT OR REPLACE INTO memory_archives
                (program_id, archived_at, deleted_at, original_lineage_depth,
                 fitness_at_archive, lifecycle_status, memory_snapshot)
                VALUES (?, ?, ?, ?, ?, 'deleted', NULL)
            """, (
                program_id,
                datetime.now().isoformat(),
                datetime.now().isoformat(),
                prog_row[0],
                prog_row[1],
            ))
            # 从 programs 表删除
            conn.execute("DELETE FROM programs WHERE program_id = ?", (program_id,))
            # 从 memory_fragments 表删除（如果存在）
            try:
                conn.execute("DELETE FROM memory_fragments WHERE program_id = ?", (program_id,))
            except Exception:
                pass
            conn.commit()
            conn.close()
            # 从缓存中移除
            self._program_cache.pop(program_id, None)
            return True
