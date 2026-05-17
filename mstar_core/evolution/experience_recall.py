"""
MSTAR Pro v4.0 - Phase 6: Experience Recall + Trajectory
参考: Synergy (2603.28428) + LIFE Framework (2605.14892)

核心思想: 在推理时动态召回历史轨迹，形成CoT (Chain of Thoughts)
- Experience Recall: 从历史中检索相似情况下的决策轨迹
- Trajectory Stitching: 将多个相关轨迹拼接成完整的推理链
- Meta-Controller: 决定何时召回、召回什么、如何使用

Synergy关键概念:
- 轨迹相似度: 当前状态与历史状态的embedding相似度
- 动态召回: 不是静态记忆，而是在推理时主动检索
- 轨迹质量: 评估历史轨迹对当前决策的参考价值
"""

from __future__ import annotations
import logging
import time
import sqlite3
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class TrajectoryType(Enum):
    """轨迹类型"""
    SUCCESS = "success"              # 成功轨迹
    FAILURE = "failure"              # 失败轨迹
    PARTIAL_SUCCESS = "partial"      # 部分成功
    ABORTED = "aborted"              # 中止
    EXPLORATION = "exploration"      # 探索性


@dataclass
class TrajectoryStep:
    """轨迹中的单一步骤"""
    step_id: str
    timestamp: str
    state: Dict          # 执行时的状态快照
    action: str          # 采取的动作
    reasoning: str       # 推理过程
    outcome: str         # 结果
    next_state: Dict     # 执行后的状态
    utility: float        # 该步的效用值

    def to_dict(self) -> Dict:
        return {
            'step_id': self.step_id,
            'timestamp': self.timestamp,
            'state': self.state,
            'action': self.action,
            'reasoning': self.reasoning,
            'outcome': self.outcome,
            'next_state': self.next_state,
            'utility': self.utility,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'TrajectoryStep':
        return cls(**d)


@dataclass
class Trajectory:
    """完整的执行轨迹"""
    trajectory_id: str
    program_id: str
    task_type: str
    trajectory_type: TrajectoryType
    steps: List[TrajectoryStep]
    start_time: str
    end_time: str
    overall_success: bool
    final_fitness: float
    task_context: Dict  # 任务上下文信息

    def to_dict(self) -> Dict:
        return {
            'trajectory_id': self.trajectory_id,
            'program_id': self.program_id,
            'task_type': self.task_type,
            'trajectory_type': self.trajectory_type.value if isinstance(self.trajectory_type, TrajectoryType) else self.trajectory_type,
            'steps': [s.to_dict() if isinstance(s, TrajectoryStep) else s for s in self.steps],
            'start_time': self.start_time,
            'end_time': self.end_time,
            'overall_success': self.overall_success,
            'final_fitness': self.final_fitness,
            'task_context': self.task_context,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'Trajectory':
        tt = d.get('trajectory_type', 'success')
        if isinstance(tt, str):
            try:
                tt = TrajectoryType(tt)
            except ValueError:
                tt = TrajectoryType.SUCCESS
        return cls(
            trajectory_id=d['trajectory_id'],
            program_id=d['program_id'],
            task_type=d['task_type'],
            trajectory_type=tt,
            steps=[TrajectoryStep.from_dict(s) if isinstance(s, dict) else s for s in d.get('steps', [])],
            start_time=d['start_time'],
            end_time=d['end_time'],
            overall_success=d['overall_success'],
            final_fitness=d.get('final_fitness', 0.5),
            task_context=d.get('task_context', {}),
        )

    def get_reasoning_chain(self) -> str:
        """提取推理链（所有step的reasoning拼接）"""
        return "\n".join(
            f"[{i+1}] {s.reasoning}" if isinstance(s, TrajectoryStep) else f"[{i+1}] {s.get('reasoning', '')}"
            for i, s in enumerate(self.steps)
        )

    def get_successful_steps(self) -> List[int]:
        """获取成功步骤的索引"""
        return [
            i for i, s in enumerate(self.steps)
            if (isinstance(s, TrajectoryStep) and s.outcome == 'success')
            or (isinstance(s, dict) and s.get('outcome') == 'success')
        ]

    def compute_trajectory_quality(self) -> float:
        """计算轨迹质量分数"""
        if not self.steps:
            return 0.0

        # 基于成功率
        success_steps = self.get_successful_steps()
        success_ratio = len(success_steps) / len(self.steps)

        # 基于最终结果
        final_bonus = 1.0 if self.overall_success else 0.0

        # 基于步数效率（短轨迹更好）
        efficiency = 1.0 / (1.0 + len(self.steps) * 0.1)

        # 综合得分
        quality = success_ratio * 0.5 + final_bonus * 0.3 + efficiency * 0.2
        return min(1.0, max(0.0, quality))


class ExperienceRecall:
    """
    MSTAR Pro v4.0 经验召回系统

    核心功能:
    1. 记录和索引轨迹
    2. 根据当前状态检索相似轨迹
    3. 评估检索到的轨迹的参考价值
    4. 将轨迹知识用于当前决策

    使用场景:
    - AIAgent在推理时调用recall_similar_experience
    - MetaController决定是否需要召回经验
    - 轨迹用于few-shot示例选择
    """

    def __init__(
        self,
        hermes_home: str,
        similarity_threshold: float = 0.6,
        max_recall_results: int = 3,
        trajectory_cache_size: int = 100,
        db_path: Optional[str] = None,
    ):
        import os
        self.hermes_home = hermes_home
        os.makedirs(hermes_home, exist_ok=True)

        self.similarity_threshold = similarity_threshold
        self.max_recall_results = max_recall_results
        self.trajectory_cache_size = trajectory_cache_size

        self.db_path = db_path or f"{hermes_home}/mstar_trajectories.db"
        self._lock = threading.RLock()

        self._init_database()

        # 内存缓存: trajectory_id -> Trajectory
        self._trajectory_cache: Dict[str, Trajectory] = {}
        self._cache_order: List[str] = []  # LRU顺序

        # 统计
        self._total_recalls = 0
        self._successful_recalls = 0

    def _init_database(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS trajectories (
                trajectory_id TEXT PRIMARY KEY,
                program_id TEXT,
                task_type TEXT,
                trajectory_type TEXT,
                steps_json TEXT,
                start_time TEXT,
                end_time TEXT,
                overall_success INTEGER,
                final_fitness REAL,
                task_context_json TEXT,
                trajectory_quality REAL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS trajectory_steps (
                step_id TEXT PRIMARY KEY,
                trajectory_id TEXT,
                step_index INTEGER,
                state_json TEXT,
                action TEXT,
                reasoning TEXT,
                outcome TEXT,
                next_state_json TEXT,
                utility REAL
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_trajectory_program
            ON trajectories(program_id, start_time)
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_trajectory_task
            ON trajectories(task_type, overall_success)
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_steps_trajectory
            ON trajectory_steps(trajectory_id, step_index)
        """)

        conn.commit()
        conn.close()

    def _make_trajectory_id(self) -> str:
        import hashlib
        ts = datetime.now().isoformat()
        raw = f"{ts}_{random.randint(10000, 99999)}"
        return f"traj_{hashlib.md5(raw.encode()).hexdigest()[:12]}"

    def record_trajectory(
        self,
        program_id: str,
        task_type: str,
        trajectory_type: TrajectoryType,
        steps: List[TrajectoryStep],
        start_time: str,
        end_time: str,
        overall_success: bool,
        final_fitness: float,
        task_context: Optional[Dict] = None,
    ) -> Trajectory:
        """
        记录一条新轨迹。

        Args:
            program_id: 程序ID
            task_type: 任务类型
            trajectory_type: 轨迹类型
            steps: 轨迹步骤列表
            start_time/end_time: 开始/结束时间
            overall_success: 是否整体成功
            final_fitness: 最终适应度
            task_context: 任务上下文

        Returns:
            Trajectory: 创建的轨迹对象
        """
        with self._lock:
            trajectory_id = self._make_trajectory_id()

            trajectory = Trajectory(
                trajectory_id=trajectory_id,
                program_id=program_id,
                task_type=task_type,
                trajectory_type=trajectory_type,
                steps=steps,
                start_time=start_time,
                end_time=end_time,
                overall_success=overall_success,
                final_fitness=final_fitness,
                task_context=task_context or {},
            )

            quality = trajectory.compute_trajectory_quality()

            # 保存到数据库
            conn = sqlite3.connect(self.db_path, timeout=30)
            conn.execute("""
                INSERT INTO trajectories
                (trajectory_id, program_id, task_type, trajectory_type, steps_json,
                 start_time, end_time, overall_success, final_fitness,
                 task_context_json, trajectory_quality)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trajectory_id, program_id, task_type, trajectory_type.value,
                json.dumps([s.to_dict() if isinstance(s, TrajectoryStep) else s for s in steps]),
                start_time, end_time, 1 if overall_success else 0, final_fitness,
                json.dumps(task_context or {}), quality,
            ))

            # 保存步骤
            for i, step in enumerate(steps):
                if isinstance(step, TrajectoryStep):
                    s = step
                else:
                    s = TrajectoryStep(**step)

                conn.execute("""
                    INSERT INTO trajectory_steps
                    (step_id, trajectory_id, step_index, state_json, action,
                     reasoning, outcome, next_state_json, utility)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    s.step_id, trajectory_id, i,
                    json.dumps(s.state), s.action,
                    s.reasoning, s.outcome,
                    json.dumps(s.next_state), s.utility,
                ))

            conn.commit()
            conn.close()

            # 更新缓存
            self._add_to_cache(trajectory_id, trajectory)

            logger.info(f"[MSTAR] Recorded trajectory {trajectory_id} ({trajectory_type.value})")

            return trajectory

    def _add_to_cache(self, trajectory_id: str, trajectory: Trajectory):
        """添加到LRU缓存"""
        if trajectory_id in self._trajectory_cache:
            return

        self._trajectory_cache[trajectory_id] = trajectory
        self._cache_order.append(trajectory_id)

        # LRU淘汰
        while len(self._cache_order) > self.trajectory_cache_size:
            oldest = self._cache_order.pop(0)
            if oldest in self._trajectory_cache:
                del self._trajectory_cache[oldest]

    def recall_similar_experience(
        self,
        current_state: Dict,
        task_type: Optional[str] = None,
        program_id: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> List[Tuple[Trajectory, float]]:
        """
        根据当前状态检索相似的历史轨迹。

        使用简单状态匹配（未来可以升级为embedding相似度）:
        1. 提取当前状态的关键特征
        2. 在数据库中搜索匹配的历史轨迹
        3. 计算相似度并排序
        4. 返回top_k个最相似的轨迹

        Args:
            current_state: 当前状态（包含task, context等）
            task_type: 可选，限定任务类型
            program_id: 可选，限定程序ID
            top_k: 返回结果数量（默认self.max_recall_results）

        Returns:
            List[Tuple[Trajectory, float]]: (轨迹, 相似度分数) 列表，按相似度降序
        """
        self._total_recalls += 1

        if top_k is None:
            top_k = self.max_recall_results

        # 提取当前状态的特征
        current_task = current_state.get('task_type', task_type or 'unknown')
        current_action = current_state.get('action', '')
        current_reasoning = current_state.get('reasoning', '')

        # 构建查询
        conditions = []
        params = []

        if task_type:
            conditions.append("task_type = ?")
            params.append(task_type)

        if program_id:
            conditions.append("program_id = ?")
            params.append(program_id)

        # 优先检索成功的轨迹
        conditions.append("overall_success = 1")

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        limit = top_k * 3  # 多取一些，后面过滤

        conn = sqlite3.connect(self.db_path, timeout=30)
        cursor = conn.execute(f"""
            SELECT trajectory_id, program_id, task_type, trajectory_type,
                   start_time, end_time, overall_success, final_fitness,
                   task_context_json, trajectory_quality
            FROM trajectories
            WHERE {where_clause}
            ORDER BY trajectory_quality DESC, start_time DESC
            LIMIT ?
        """, params + [limit])
        rows = cursor.fetchall()
        conn.close()

        results = []
        for row in rows:
            traj_id = row[0]

            # 检查缓存
            if traj_id in self._trajectory_cache:
                trajectory = self._trajectory_cache[traj_id]
            else:
                # 从数据库加载
                trajectory = self._load_trajectory_from_db(traj_id, row)
                if trajectory:
                    self._add_to_cache(traj_id, trajectory)

            if not trajectory:
                continue

            # 计算与当前状态的相似度
            similarity = self._compute_similarity(current_state, trajectory)

            if similarity >= self.similarity_threshold:
                results.append((trajectory, similarity))
                self._successful_recalls += 1

        # 排序并返回top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def _load_trajectory_from_db(self, trajectory_id: str, row: Tuple) -> Optional[Trajectory]:
        """从数据库加载轨迹"""
        try:
            # 获取步骤
            conn = sqlite3.connect(self.db_path, timeout=30)
            cursor = conn.execute("""
                SELECT step_id, state_json, action, reasoning, outcome, next_state_json, utility
                FROM trajectory_steps
                WHERE trajectory_id = ?
                ORDER BY step_index
            """, (trajectory_id,))
            step_rows = cursor.fetchall()
            conn.close()

            steps = []
            for s_row in step_rows:
                steps.append(TrajectoryStep(
                    step_id=s_row[0],
                    timestamp="",
                    state=json.loads(s_row[1]),
                    action=s_row[2],
                    reasoning=s_row[3],
                    outcome=s_row[4],
                    next_state=json.loads(s_row[5]),
                    utility=s_row[6],
                ))

            tt = row[3]
            try:
                tt = TrajectoryType(tt)
            except ValueError:
                tt = TrajectoryType.SUCCESS

            return Trajectory(
                trajectory_id=trajectory_id,
                program_id=row[1],
                task_type=row[2],
                trajectory_type=tt,
                steps=steps,
                start_time=row[4],
                end_time=row[5],
                overall_success=bool(row[6]),
                final_fitness=row[7],
                task_context=json.loads(row[8]) if row[8] else {},
            )
        except Exception as e:
            logger.warning(f"[MSTAR] Failed to load trajectory {trajectory_id}: {e}")
            return None

    def _compute_similarity(self, current_state: Dict, trajectory: Trajectory) -> float:
        """
        计算当前状态与历史轨迹的相似度。

        评分因素:
        - 任务类型匹配度
        - 程序ID匹配度
        - 状态特征重叠度
        - 时间接近度（近期轨迹权重更高）
        """
        score = 0.0

        # 任务类型匹配 (40%)
        current_task = current_state.get('task_type', '')
        if trajectory.task_type == current_task:
            score += 0.4

        # 程序ID匹配 (20%)
        current_program = current_state.get('program_id', '')
        if trajectory.program_id == current_program and current_program:
            score += 0.2

        # 轨迹质量 (20%)
        score += trajectory.compute_trajectory_quality() * 0.2

        # 时间衰减因子 (20%) - 近期轨迹更可靠
        try:
            traj_time = datetime.fromisoformat(trajectory.start_time)
            age_days = (datetime.now() - traj_time).days
            time_factor = max(0, 1.0 - age_days / 30.0)  # 30天内线性衰减
            score += time_factor * 0.2
        except:
            score += 0.1  # 无法解析时间，默认给10%

        return min(1.0, score)

    def get_reasoning_from_similar(
        self,
        current_state: Dict,
        task_type: Optional[str] = None,
        max_chain_length: int = 5,
    ) -> str:
        """
        从相似轨迹中提取推理链。

        用于在当前决策时提供历史参考:
        - 找到相似的成功轨迹
        - 提取其推理过程
        - 返回格式化的参考文本

        Returns:
            str: 格式化的推理链参考
        """
        similar = self.recall_similar_experience(
            current_state=current_state,
            task_type=task_type,
            top_k=1,  # 只取最相似的
        )

        if not similar:
            return ""

        trajectory, similarity = similar[0]

        # 提取推理链
        reasoning_chain = trajectory.get_reasoning_chain()

        # 格式化为参考文本
        ref_text = f"""
[参考历史轨迹] (相似度: {similarity:.2f})
任务类型: {trajectory.task_type}
程序: {trajectory.program_id}
结果: {'成功' if trajectory.overall_success else '失败'}
推理链:
{reasoning_chain}
""".strip()

        return ref_text

    def build_few_shot_examples(
        self,
        task_type: str,
        n_examples: int = 3,
        prioritize_success: bool = True,
    ) -> List[Trajectory]:
        """
        为特定任务类型构建few-shot示例。

        用于:
        - 为LLM提供少样本提示
        - 训练数据增强

        Returns:
            List[Trajectory]: 选定的示例轨迹
        """
        conn = sqlite3.connect(self.db_path, timeout=30)

        success_clause = "AND overall_success = 1" if prioritize_success else ""

        cursor = conn.execute(f"""
            SELECT trajectory_id
            FROM trajectories
            WHERE task_type = ?
            {success_clause}
            ORDER BY trajectory_quality DESC
            LIMIT ?
        """, (task_type, n_examples))

        rows = cursor.fetchall()
        conn.close()

        examples = []
        for row in rows:
            traj_id = row[0]
            if traj_id in self._trajectory_cache:
                examples.append(self._trajectory_cache[traj_id])
            else:
                # 从数据库加载
                conn = sqlite3.connect(self.db_path, timeout=30)
                cursor = conn.execute("""
                    SELECT program_id, task_type, trajectory_type, steps_json,
                           start_time, end_time, overall_success, final_fitness,
                           task_context_json
                    FROM trajectories WHERE trajectory_id = ?
                """, (traj_id,))
                db_row = cursor.fetchone()
                conn.close()

                if db_row:
                    trajectory = self._load_trajectory_from_db(traj_id, db_row)
                    if trajectory:
                        self._add_to_cache(traj_id, trajectory)
                        examples.append(trajectory)

        return examples

    def analyze_failure_patterns(self, program_id: Optional[str] = None) -> Dict:
        """
        分析失败模式。

        用于:
        - 识别常见的失败原因
        - 为Mutation提供失败模式信息
        """
        conn = sqlite3.connect(self.db_path, timeout=30)

        if program_id:
            cursor = conn.execute("""
                SELECT trajectory_type, COUNT(*) as cnt
                FROM trajectories
                WHERE program_id = ? AND overall_success = 0
                GROUP BY trajectory_type
            """, (program_id,))
        else:
            cursor = conn.execute("""
                SELECT trajectory_type, COUNT(*) as cnt
                FROM trajectories
                WHERE overall_success = 0
                GROUP BY trajectory_type
            """)

        rows = cursor.fetchall()
        conn.close()

        patterns = {}
        for tt, cnt in rows:
            patterns[tt] = cnt

        return {
            'program_id': program_id,
            'failure_types': patterns,
            'total_failures': sum(patterns.values()),
        }

    def get_statistics(self) -> Dict:
        """获取轨迹统计"""
        conn = sqlite3.connect(self.db_path, timeout=30)

        total = conn.execute("SELECT COUNT(*) FROM trajectories").fetchone()[0]
        successful = conn.execute("SELECT COUNT(*) FROM trajectories WHERE overall_success = 1").fetchone()[0]

        cursor = conn.execute("SELECT task_type, COUNT(*) FROM trajectories GROUP BY task_type")
        by_task = dict(cursor.fetchall())

        cursor = conn.execute("SELECT AVG(trajectory_quality) FROM trajectories")
        avg_quality = cursor.fetchone()[0] or 0.0

        conn.close()

        return {
            'total_trajectories': total,
            'successful_trajectories': successful,
            'failure_rate': (total - successful) / total if total > 0 else 0.0,
            'by_task_type': by_task,
            'avg_trajectory_quality': avg_quality,
            'total_recalls': self._total_recalls,
            'successful_recalls': self._successful_recalls,
            'recall_success_rate': self._successful_recalls / self._total_recalls if self._total_recalls > 0 else 0.0,
            'cache_size': len(self._trajectory_cache),
        }


import json
import random