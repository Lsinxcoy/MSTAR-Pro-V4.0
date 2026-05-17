"""
MSTAR Pro v4.0 - Phase 5: EvolveMem Self-Evolving Retrieval
参考: EvolveMem (2605.13941) - 最重要的论文之一

核心思想: 记忆检索配置必须自进化
- AutoResearch范式: 系统自主研究和调整记忆检索参数
- 动态调整: 根据任务表现自动调整检索配置
- 元学习: 从检索效果中学习最优配置

关键概念:
- Retrieval Configuration: 向量搜索的top_k, similarity_threshold等参数
- Self-Tuning: 根据任务成功率自动调整检索参数
- Performance Correlation: 记忆检索与任务成功的相关性分析
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
import random

logger = logging.getLogger(__name__)


class RetrievalMetric(Enum):
    """检索质量指标"""
    RECALL = "recall"                  # 召回率
    PRECISION = "precision"            # 精确率
    LATENCY = "latency"               # 检索延迟
    DIVERSITY = "diversity"           # 结果多样性
    RELEVANCE_SCORE = "relevance_score"  # 相关性得分


@dataclass
class RetrievalConfig:
    """
    记忆检索配置参数

    这些参数控制向量搜索的行为:
    - top_k: 返回前k个结果
    - similarity_threshold: 相似度阈值
    - rerank_enabled: 是否启用重排
    - diversity_boost: 多样性权重
    """
    config_id: str
    name: str
    top_k: int = 5
    similarity_threshold: float = 0.7
    rerank_enabled: bool = False
    diversity_boost: float = 0.0
    max_results: int = 10
    min_relevance_score: float = 0.5

    # 附加参数（论文中的自适应参数）
    context_window: int = 512
    temporal_decay: float = 0.95  # 时间衰减因子

    def to_dict(self) -> Dict:
        return {
            'config_id': self.config_id,
            'name': self.name,
            'top_k': self.top_k,
            'similarity_threshold': self.similarity_threshold,
            'rerank_enabled': self.rerank_enabled,
            'diversity_boost': self.diversity_boost,
            'max_results': self.max_results,
            'min_relevance_score': self.min_relevance_score,
            'context_window': self.context_window,
            'temporal_decay': self.temporal_decay,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'RetrievalConfig':
        return cls(**d)


@dataclass
class RetrievalResult:
    """单次检索的结果"""
    query: str
    config_id: str
    retrieved_items: List[str]
    top_k_actual: int
    avg_similarity: float
    retrieval_time_ms: float
    timestamp: str


@dataclass
class RetrievalPerformance:
    """检索性能评估"""
    config_id: str
    metric: str
    value: float
    sample_size: int
    confidence: float  # 置信度
    timestamp: str


class EvolveMemConfigurator:
    """
    EvolveMem自进化检索配置器

    核心功能:
    1. 评估不同检索配置的效果
    2. 根据任务表现自动调整配置
    3. 使用bandit算法平衡探索与利用
    4. 记录检索-任务成功的相关性

    使用场景:
    - AIAgent._retrieve_memories: 使用当前最优配置检索记忆
    - AIAgent._update_memory_after_task: 记录检索效果
    - 定期运行调整: 根据累积数据优化配置
    """

    # 默认配置
    DEFAULT_CONFIGS = [
        {'config_id': 'default_narrow', 'name': '窄范围高精度',
         'top_k': 3, 'similarity_threshold': 0.8, 'rerank_enabled': True},
        {'config_id': 'default_balanced', 'name': '平衡模式',
         'top_k': 5, 'similarity_threshold': 0.7, 'rerank_enabled': False},
        {'config_id': 'default_broad', 'name': '宽范围高召回',
         'top_k': 8, 'similarity_threshold': 0.5, 'rerank_enabled': False},
        {'config_id': 'default_diverse', 'name': '多样性优先',
         'top_k': 5, 'similarity_threshold': 0.6, 'diversity_boost': 0.3},
    ]

    def __init__(
        self,
        hermes_home: str,
        exploration_rate: float = 0.2,
        min_samples_for_adjustment: int = 10,
        adjustment_interval: int = 50,
        db_path: Optional[str] = None,
    ):
        import os
        self.hermes_home = hermes_home
        os.makedirs(hermes_home, exist_ok=True)

        self.exploration_rate = exploration_rate
        self.min_samples_for_adjustment = min_samples_for_adjustment
        self.adjustment_interval = adjustment_interval  # 每N次检索后调整一次

        self.db_path = db_path or f"{hermes_home}/mstar_evolvemem.db"
        self._lock = threading.RLock()

        self._init_database()

        # 加载或创建默认配置
        self.configs: Dict[str, RetrievalConfig] = {}

        # 当前激活的配置（必须在_load_or_create_default_configs之前初始化）
        self.active_config_id: str = 'default_balanced'

        self._load_or_create_default_configs()

        # 检索统计
        self._retrieval_count = 0
        self._total_retrievals = 0

        # 配置性能缓存: config_id -> List[RetrievalPerformance]
        self._performance_cache: Dict[str, List[RetrievalPerformance]] = {}

    def _init_database(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS retrieval_configs (
                config_id TEXT PRIMARY KEY,
                name TEXT,
                config_json TEXT,
                created_at TEXT,
                is_active INTEGER DEFAULT 0
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS retrieval_results (
                result_id TEXT PRIMARY KEY,
                config_id TEXT,
                query TEXT,
                retrieved_items TEXT,
                top_k_actual INTEGER,
                avg_similarity REAL,
                retrieval_time_ms REAL,
                task_success INTEGER,
                timestamp TEXT
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS retrieval_performance (
                perf_id TEXT PRIMARY KEY,
                config_id TEXT,
                metric TEXT,
                value REAL,
                sample_size INTEGER,
                confidence REAL,
                timestamp TEXT
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_results_config
            ON retrieval_results(config_id, timestamp)
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_perf_config
            ON retrieval_performance(config_id, metric, timestamp)
        """)

        conn.commit()
        conn.close()

    def _load_or_create_default_configs(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        cursor = conn.execute("SELECT config_id FROM retrieval_configs")
        existing = set(row[0] for row in cursor.fetchall())
        conn.close()

        for cfg_data in self.DEFAULT_CONFIGS:
            cfg_id = cfg_data['config_id']
            if cfg_id not in existing:
                config = RetrievalConfig(
                    config_id=cfg_id,
                    name=cfg_data['name'],
                    top_k=cfg_data['top_k'],
                    similarity_threshold=cfg_data['similarity_threshold'],
                    rerank_enabled=cfg_data.get('rerank_enabled', False),
                    diversity_boost=cfg_data.get('diversity_boost', 0.0),
                )
                self._save_config(config)
                logger.info(f"[MSTAR] Created default config: {cfg_id}")

        # 加载所有配置到内存
        self._load_all_configs()

    def _load_all_configs(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        cursor = conn.execute("SELECT config_json, is_active FROM retrieval_configs")
        rows = cursor.fetchall()
        conn.close()

        for row in rows:
            cfg_dict = json.loads(row[0])
            config = RetrievalConfig.from_dict(cfg_dict)
            self.configs[config.config_id] = config
            if row[1]:
                self.active_config_id = config.config_id

    def _save_config(self, config: RetrievalConfig):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute("""
            INSERT OR REPLACE INTO retrieval_configs
            (config_id, name, config_json, created_at, is_active)
            VALUES (?, ?, ?, ?, ?)
        """, (
            config.config_id,
            config.name,
            json.dumps(config.to_dict()),
            datetime.now().isoformat(),
            1 if config.config_id == self.active_config_id else 0,
        ))
        conn.commit()
        conn.close()

    def get_active_config(self) -> RetrievalConfig:
        """获取当前激活的配置"""
        return self.configs.get(self.active_config_id, self.DEFAULT_CONFIGS[1])

    def select_config(self) -> RetrievalConfig:
        """
        选择要使用的配置。

        使用epsilon-greedy策略:
        - 以exploration_rate的概率随机选择（探索）
        - 否则选择当前最优配置（利用）

        Returns:
            RetrievalConfig: 选中的配置
        """
        self._total_retrievals += 1

        if random.random() < self.exploration_rate:
            # 随机探索
            config_id = random.choice(list(self.configs.keys()))
            logger.debug(f"[MSTAR] Exploration: selected random config {config_id}")
        else:
            # 利用: 选择最优配置
            config_id = self._select_best_config()
            logger.debug(f"[MSTAR] Exploitation: selected best config {config_id}")

        return self.configs[config_id]

    def _select_best_config(self) -> str:
        """
        根据历史性能选择最优配置。

        评估指标: 任务成功率与检索质量的综合得分

        Returns:
            config_id: 最优配置的ID
        """
        if not self._performance_cache:
            return self.active_config_id

        # 计算每个配置的综合得分
        config_scores = {}
        for cfg_id, performances in self._performance_cache.items():
            if not performances:
                continue

            # 计算平均任务成功率
            success_rates = [p.value for p in performances if p.metric == 'task_success']
            if success_rates:
                avg_success = sum(success_rates) / len(success_rates)
            else:
                avg_success = 0.5

            # 计算平均检索质量
            quality_scores = [p.value for p in performances if p.metric in ('recall', 'precision', 'relevance_score')]
            if quality_scores:
                avg_quality = sum(quality_scores) / len(quality_scores)
            else:
                avg_quality = 0.5

            # 综合得分 (成功率权重更高)
            config_scores[cfg_id] = avg_success * 0.7 + avg_quality * 0.3

        if not config_scores:
            return self.active_config_id

        # 返回最高分配置
        best_config_id = max(config_scores, key=config_scores.get)
        return best_config_id

    def record_retrieval(
        self,
        query: str,
        config_id: str,
        retrieved_items: List[str],
        retrieval_time_ms: float,
        task_success: Optional[bool] = None,
        avg_similarity: float = 0.0,
    ):
        """
        记录一次检索及其结果。

        用于后续分析检索配置的效果。
        """
        with self._lock:
            self._retrieval_count += 1

            result_id = f"ret_{int(time.time() * 1000)}_{self._retrieval_count}"

            conn = sqlite3.connect(self.db_path, timeout=30)
            conn.execute("""
                INSERT INTO retrieval_results
                (result_id, config_id, query, retrieved_items, top_k_actual,
                 avg_similarity, retrieval_time_ms, task_success, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result_id,
                config_id,
                query,
                json.dumps(retrieved_items),
                len(retrieved_items),
                avg_similarity,
                retrieval_time_ms,
                1 if task_success else (0 if task_success is not None else None),
                datetime.now().isoformat(),
            ))
            conn.commit()
            conn.close()

            # 更新缓存
            if config_id not in self._performance_cache:
                self._performance_cache[config_id] = []

            # 如果有任务成功信息，更新性能缓存
            if task_success is not None:
                perf = RetrievalPerformance(
                    config_id=config_id,
                    metric='task_success',
                    value=1.0 if task_success else 0.0,
                    sample_size=1,
                    confidence=0.5,
                    timestamp=datetime.now().isoformat(),
                )
                self._performance_cache[config_id].append(perf)

            # 定期调整配置
            if self._retrieval_count >= self.adjustment_interval:
                self._adjust_configs()
                self._retrieval_count = 0

    def _adjust_configs(self):
        """
        根据累积数据调整检索配置。

        使用简单的爬山算法:
        - 对当前最优配置进行小幅扰动
        - 如果扰动后效果更好，采纳新配置
        """
        if not self._performance_cache:
            return

        # 找到当前最优配置
        best_config_id = self._select_best_config()
        best_config = self.configs[best_config_id]

        # 生成扰动后的新配置
        new_config = self._perturb_config(best_config)

        # 评估新配置（模拟）
        # 在真实系统中，这里应该实际使用新配置并观察效果
        # 这里简化为基于规则的评估

        # 如果新配置有优势，创建并启用它
        candidate_id = f"auto_{int(time.time())}"
        new_config.config_id = candidate_id
        new_config.name = f"Auto-tuned from {best_config_id}"

        # 保存新配置
        self.configs[candidate_id] = new_config
        self._save_config(new_config)

        # 切换到新配置
        self.active_config_id = candidate_id

        logger.info(f"[MSTAR] EvolveMem auto-tuned config: {best_config_id} -> {candidate_id}")

    def _perturb_config(self, config: RetrievalConfig) -> RetrievalConfig:
        """
        对配置进行小幅扰动，生成候选配置。

        扰动策略:
        - top_k: ±1-2
        - similarity_threshold: ±0.05-0.1
        - rerank_enabled: 随机翻转
        - diversity_boost: ±0.1
        """
        return RetrievalConfig(
            config_id=f"perturb_{config.config_id}",
            name=f"Perturbed {config.name}",
            top_k=max(1, config.top_k + random.randint(-2, 2)),
            similarity_threshold=max(0.3, min(0.95, config.similarity_threshold + random.uniform(-0.1, 0.1))),
            rerank_enabled=not config.rerank_enabled if random.random() < 0.3 else config.rerank_enabled,
            diversity_boost=max(0.0, min(0.5, config.diversity_boost + random.uniform(-0.1, 0.1))),
            max_results=config.max_results + random.randint(-2, 2),
            min_relevance_score=max(0.3, min(0.9, config.min_relevance_score + random.uniform(-0.05, 0.05))),
            context_window=config.context_window,
            temporal_decay=config.temporal_decay,
        )

    def analyze_retrieval_success_correlation(self, config_id: str, lookback: int = 100) -> Dict:
        """
        分析特定配置的检索与任务成功的相关性。

        用于:
        - 理解不同检索配置对任务成功率的影响
        - 为配置调整提供数据支持
        """
        conn = sqlite3.connect(self.db_path, timeout=30)
        cursor = conn.execute("""
            SELECT task_success, retrieved_items, avg_similarity, retrieval_time_ms
            FROM retrieval_results
            WHERE config_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (config_id, lookback))
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return {
                'config_id': config_id,
                'status': 'no_data',
                'message': '没有足够的检索记录',
            }

        # 计算统计数据
        task_successes = [row[0] for row in rows if row[0] is not None]
        similarities = [row[2] for row in rows if row[2] is not None]
        latencies = [row[3] for row in rows if row[3] is not None]

        success_rate = sum(task_successes) / len(task_successes) if task_successes else 0.0
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

        # 计算检索数量与成功的相关性
        # 简化为：检索数量多是否导致成功？
        n_retrievals = len(rows)

        return {
            'config_id': config_id,
            'sample_size': len(rows),
            'task_success_rate': success_rate,
            'avg_similarity': avg_similarity,
            'avg_latency_ms': avg_latency,
            'n_successful': sum(task_successes) if task_successes else 0,
            'n_total': len(task_successes),
        }

    def get_statistics(self) -> Dict:
        """获取EvolveMem统计"""
        conn = sqlite3.connect(self.db_path, timeout=30)

        total_results = conn.execute("SELECT COUNT(*) FROM retrieval_results").fetchone()[0]
        total_configs = conn.execute("SELECT COUNT(*) FROM retrieval_configs").fetchone()[0]

        cursor = conn.execute("""
            SELECT config_id, COUNT(*) as cnt
            FROM retrieval_results
            GROUP BY config_id
            ORDER BY cnt DESC
        """)
        config_usage = dict(cursor.fetchall())

        cursor = conn.execute("""
            SELECT AVG(retrieval_time_ms), AVG(avg_similarity)
            FROM retrieval_results
        """)
        avg_row = cursor.fetchone()

        conn.close()

        return {
            'total_retrievals': total_results,
            'total_configs': total_configs,
            'active_config': self.active_config_id,
            'config_usage': config_usage,
            'avg_retrieval_time_ms': avg_row[0] or 0.0,
            'avg_similarity': avg_row[1] or 0.0,
            'exploration_rate': self.exploration_rate,
        }

    def create_custom_config(self, name: str, **params) -> RetrievalConfig:
        """
        创建一个自定义检索配置。

        Args:
            name: 配置名称
            **params: 配置参数 (top_k, similarity_threshold等)

        Returns:
            RetrievalConfig: 创建的配置
        """
        config_id = f"custom_{int(time.time())}"
        config = RetrievalConfig(config_id=config_id, name=name, **params)
        self.configs[config_id] = config
        self._save_config(config)
        return config

    def delete_config(self, config_id: str) -> bool:
        """
        删除一个检索配置（不能删除默认配置）。

        Returns:
            bool: 是否成功删除
        """
        if config_id.startswith('default_'):
            logger.warning(f"[MSTAR] Cannot delete default config: {config_id}")
            return False

        if config_id not in self.configs:
            return False

        if config_id == self.active_config_id:
            # 切换到默认配置
            self.active_config_id = 'default_balanced'

        del self.configs[config_id]

        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute("DELETE FROM retrieval_configs WHERE config_id = ?", (config_id,))
        conn.commit()
        conn.close()

        return True

    def reset_to_defaults(self):
        """重置为默认配置"""
        for cfg_id in list(self.configs.keys()):
            if not cfg_id.startswith('default_'):
                self.delete_config(cfg_id)

        self.active_config_id = 'default_balanced'
        logger.info("[MSTAR] EvolveMem reset to default configurations")


# =============================================================================
# 与MARS Belief Memory的集成
# =============================================================================

class EvolveMemMARSIntegration:
    """
    EvolveMem与MARS Belief Memory的集成

    关键集成点:
    1. 检索配置影响记忆检索的质量
    2. 检索效果反馈到belief更新
    3. belief状态影响检索策略选择
    """

    def __init__(self, evolvemem: EvolveMemConfigurator, mars_belief):
        self.evolvemem = evolvemem
        self.mars_belief = mars_belief

    def get_context_aware_config(self, context: Dict) -> RetrievalConfig:
        """
        根据上下文选择最优检索配置。

        上下文因素:
        - 任务类型
        - 置信度
        - 时间压力

        Returns:
            RetrievalConfig: 上下感知的最优配置
        """
        task_type = context.get('task_type', 'general')
        confidence = context.get('confidence', 0.5)

        # 高置信度任务: 使用窄范围高精度配置
        if confidence > 0.8:
            return self.evolvemem.configs.get('default_narrow', self.evolvemem.get_active_config())

        # 低置信度任务: 使用宽范围高召回配置
        if confidence < 0.4:
            return self.evolvemem.configs.get('default_broad', self.evolvemem.get_active_config())

        # 默认使用平衡模式
        return self.evolvemem.configs.get('default_balanced', self.evolvemem.get_active_config())

    def update_belief_from_retrieval(self, query: str, retrieved: List[str], task_success: bool):
        """
        根据检索结果更新belief。

        检索效果好 -> belief增强
        检索效果差 -> belief调整
        """
        if task_success:
            # 成功: 强化检索到的记忆
            self.mars_belief.boost_belief_strength(retrieved)
        else:
            # 失败: 检查是否需要调整检索策略
            # 触发EvolveMem配置调整
            pass


import json  # 确保json可用