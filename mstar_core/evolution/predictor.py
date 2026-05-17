"""
MSTAR Pro v4.0 - 工具适应度变异预测器
迁移自论文 arxiv 2410.02725 的自评估思想

论文核心迁移对照：
  论文: "Is restarting likely to produce a better response? YES/NO"
        → LLM 输出 P(YES) 判断是否值得继续生成
  MSTAR: "Is mutating this tool likely to improve its fitness? YES/NO"
        → 预测器输出 P(提升) 判断是否值得执行变异

三层实现路径：
  Path 3: RuleBasedPredictor（零成本，立即可用）
  Path 1: LLMJudgePredictor（利用现有 LLM 的内部知识）
  Path 2: TrainedPredictor（积累数据后升级，需 fitness_snapshots 有历史）
"""

from __future__ import annotations
import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from mstar_core.evolution.fitness_tracker import FitnessTracker
    from mstar_core.evolution.fitness_tracker import MemoryProgram

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────────────────────

@dataclass
class PredictionResult:
    """单次预测结果"""
    strategy: str
    benefit_prob: float       # P(fitness提升), [0.0, 1.0]
    reasoning: str = ""


# ─────────────────────────────────────────────────────────────
# 抽象基类
# ─────────────────────────────────────────────────────────────

class ToolFitnessPredictor(ABC):
    """
    工具变异收益预测器抽象基类

    论文迁移：类似 2410.02725 的 Capability-Aware Self-Evaluation
    判断"该工具变异后是否值得"，而非仅仅判断"该工具是否需要变异"
    """

    @abstractmethod
    def predict_best_strategy(self, program) -> List[Tuple[str, float]]:
        """
        返回所有策略的收益概率排序
        Returns: [(strategy_name, benefit_prob), ...] 按概率降序
        """
        raise NotImplementedError

    @abstractmethod
    def predict_mutation_benefit(self, program, strategy: str) -> float:
        """
        返回 P(fitness提升), [0.0, 1.0]
        论文迁移：LLM 判断"继续生成是否值得" → 这里判断"变异是否值得"
        """
        raise NotImplementedError

    def should_mutate(self, program, threshold: float = 0.40) -> bool:
        """
        快速决策：是否值得对该工具执行变异
        类似论文的"是否值得生成更多样本"

        Args:
            program: 目标工具
            threshold: 触发阈值（默认 0.40）
        Returns:
            True: 预测变异会带来收益，值得执行
            False: 预测变异可能无效或降低，跳过
        """
        best = self.predict_best_strategy(program)
        if not best:
            return False
        top_strategy, top_prob = best[0]
        return top_prob >= threshold


# ─────────────────────────────────────────────────────────────
# Path 3: 规则推理预测器（零成本，立即可用）
# ─────────────────────────────────────────────────────────────

class RuleBasedPredictor(ToolFitnessPredictor):
    """
    纯规则预测器

    论文核心迁移：
      论文用 LLM 的内部知识判断"是否值得重启"
      这里用人工设计的规则判断"是否值得变异"

    规则设计原则：
      - 高 fitness 工具 → 变异风险大（已接近最优）
      - 低 fitness 工具 + 负趋势 → 强烈建议变异
      - 高 lineage_depth → 边际收益递减，应保守
      - 高波动性 → 变异效果不确定，应保守
      - 某些策略本身更有效 → 策略固有加成
    """

    # 策略历史效果加成（经验预设，非训练得出）
    STRATEGY_BONUS = {
        'combo_crossover':       +0.10,   # 组合交叉效果好
        'combo_ensemble':         +0.10,
        'instruction_examples':   +0.06,
        'instruction_context':    +0.05,
        'logic_query_add':        +0.05,
        'instruction_guidance':   +0.04,
        'schema_field_add':       +0.03,
        'schema_field_modify':    +0.02,
        'instruction_threshold':  +0.02,
        'logic_read_modify':      +0.01,
        'instruction_priority':    0.00,
        'instruction_keyword':     0.00,
        'instruction_format':      0.00,
        'schema_field_remove':    -0.01,   # 删除有风险
        'logic_write_modify':     -0.02,
        'random_change':          -0.08,   # 随机策略效果差
    }

    THRESHOLDS = {
        'fitness_very_low':  0.35,   # < 0.35: 强烈建议变异
        'fitness_low':       0.50,   # < 0.50: 建议变异
        'fitness_high':      0.75,   # > 0.75: 变异风险大
        'lineage_max':       4,      # > 4: 边际收益递减
        'volatility_high':   0.25,   # 波动性过高则保守
    }

    def __init__(self, fitness_tracker: Optional["FitnessTracker"] = None):
        self.fitness_tracker = fitness_tracker
        # 策略历史效果记录（真实学习数据）
        self._strategy_history: List[Dict] = []
        self._max_history = 200  # 最多保留200条

    def record_outcome(self, program_id: str, strategy: str,
                       predicted_prob: float, actual_delta: float):
        """
        记录变异结果并更新策略效果统计。
        RuleBasedPredictor 用这些数据动态调整 STRATEGY_BONUS。
        真实学习闭环：每次演化结果都反馈到预测精度。
        """
        entry = {
            'program_id': program_id,
            'strategy': strategy,
            'predicted_prob': predicted_prob,
            'actual_delta': actual_delta,
            # 实际提升为正则算"成功"
            'improved': actual_delta > 0,
            'timestamp': datetime.now().isoformat(),
        }
        self._strategy_history.append(entry)
        if len(self._strategy_history) > self._max_history:
            self._strategy_history = self._strategy_history[-self._max_history:]

        # 同时持久化到 DB（FitnessTracker 的 evolution_outcomes 表）
        if self.fitness_tracker:
            try:
                self.fitness_tracker.record_evolution_outcome(
                    program_id=program_id,
                    strategy=strategy,
                    predicted_prob=predicted_prob,
                    actual_delta=actual_delta,
                )
            except AttributeError:
                # FitnessTracker 可能还没初始化 evolution_outcomes 表，降级到内存
                pass

    def get_strategy_success_rate(self, strategy: str) -> float:
        """计算某策略历史成功率（用于动态调整预测）"""
        entries = [e for e in self._strategy_history if e['strategy'] == strategy]
        if not entries:
            return 0.5  # 无数据返回中性
        improved = sum(1 for e in entries if e['improved'])
        return improved / len(entries)

    def get_strategy_avg_delta(self, strategy: str) -> float:
        """计算某策略历史平均适应度变化"""
        entries = [e for e in self._strategy_history if e['strategy'] == strategy]
        if not entries:
            return 0.0
        return sum(e['actual_delta'] for e in entries) / len(entries)

    def predict_best_strategy(self, program) -> List[Tuple[str, float]]:
        """对所有策略按预测收益排序"""
        from mstar_core.evolution.mutator import MSTARMutator
        strategies = MSTARMutator.MUTATION_STRATEGIES
        results = []
        for s in strategies:
            prob = self.predict_mutation_benefit(program, s)
            results.append((s, prob))
        return sorted(results, key=lambda x: -x[1])

    def predict_mutation_benefit(self, program, strategy: str) -> float:
        score = 0.50  # 中性起点

        # ── 规则1: Fitness 水平判断 ──────────────────────────
        f = getattr(program, 'fitness_score', None) or 0.5
        if f < self.THRESHOLDS['fitness_very_low']:
            score += 0.25      # 极低 fitness → 强烈建议变异
        elif f < self.THRESHOLDS['fitness_low']:
            score += 0.12      # 低 fitness → 建议变异
        elif f > self.THRESHOLDS['fitness_high']:
            score -= 0.22      # 高 fitness → 变异风险大

        # ── 规则2: 血缘深度（边际收益递减）──────────────────
        depth = getattr(program, 'lineage_depth', None) or 0
        if depth >= self.THRESHOLDS['lineage_max']:
            score -= 0.15
        elif depth >= 2:
            score -= 0.05 * (depth - 1)

        # ── 规则3: 趋势判断（从快照数据获取）────────────────
        if self.fitness_tracker:
            program_id = getattr(program, 'program_id', None) or (program if isinstance(program, str) else None)
            if program_id:
                trend = self._get_trend(program_id)
                if trend < -0.02:
                    score += 0.12   # 下降趋势 → 强化变异需求
                elif trend > 0.05:
                    score -= 0.08   # 上升趋势 → 保守

        # ── 规则4: 波动性（高波动则保守）────────────────────
        if self.fitness_tracker:
            program_id = getattr(program, 'program_id', None) or (program if isinstance(program, str) else None)
            if program_id:
                vol = self._get_volatility(program_id)
                if vol > self.THRESHOLDS['volatility_high']:
                    score -= 0.10

        # ── 规则5: 策略固有加成 ─────────────────────────────
        # Bug-5 fix: 用真实历史数据替代硬编码 STRATEGY_BONUS
        # Bayesian smoothing: 综合先验（STRATEGY_BONUS_BASE）和观测数据
        success_rate = self.get_strategy_success_rate(strategy)
        avg_delta = self.get_strategy_avg_delta(strategy)

        STRATEGY_BONUS_BASE = {
            'combo_crossover':       0.10,
            'combo_ensemble':        0.10,
            'instruction_examples':  0.06,
            'instruction_context':   0.05,
            'logic_query_add':       0.05,
            'instruction_guidance':  0.04,
            'schema_field_add':      0.03,
            'schema_field_modify':   0.02,
            'instruction_threshold':0.02,
            'logic_read_modify':     0.01,
            'instruction_priority':  0.00,
            'instruction_keyword':   0.00,
            'instruction_format':    0.00,
            'schema_field_remove':  -0.01,
            'logic_write_modify':   -0.02,
            'random_change':        -0.08,
        }
        prior = STRATEGY_BONUS_BASE.get(strategy, 0.0)

        # 从历史数据计算学习型加成
        history_bonus = (success_rate - 0.5) * 0.20  # 成功率偏离0.5的比例 * 系数
        delta_bonus = avg_delta * 0.30                 # 平均 delta 的权重

        # Bayesian blend: 数据少时偏先验，数据多时偏观测
        n = len([e for e in self._strategy_history if e['strategy'] == strategy])
        alpha = min(n / 10.0, 1.0)  # 10条以上数据基本只看观测
        dynamic_bonus = (1 - alpha) * prior + alpha * (history_bonus + delta_bonus)

        score += dynamic_bonus

        # ── 规则6: 探索噪声（给"探索精神"留空间）──────────
        score += random.uniform(-0.03, 0.03)

        return max(0.05, min(0.95, score))

    def _get_trend(self, program_id: str) -> float:
        """从快照获取趋势斜率（简化：取最近2个快照的fitness差值）"""
        if not self.fitness_tracker:
            return 0.0
        try:
            import sqlite3
            conn = sqlite3.connect(self.fitness_tracker.db_path, timeout=10)
            cur = conn.execute("""
                SELECT fitness_score FROM fitness_snapshots
                WHERE program_id = ? AND fitness_score IS NOT NULL
                ORDER BY timestamp DESC LIMIT 2
            """, (program_id,))
            rows = cur.fetchall()
            conn.close()
            if len(rows) >= 2:
                return rows[0][0] - rows[1][0]  # delta
            return 0.0
        except Exception:
            return 0.0

    def _get_volatility(self, program_id: str) -> float:
        """计算 fitness 历史标准差"""
        if not self.fitness_tracker:
            return 0.0
        try:
            import sqlite3, statistics
            conn = sqlite3.connect(self.fitness_tracker.db_path, timeout=10)
            cur = conn.execute("""
                SELECT fitness_score FROM fitness_snapshots
                WHERE program_id = ? AND fitness_score IS NOT NULL
                ORDER BY timestamp DESC LIMIT 10
            """, (program_id,))
            scores = [r[0] for r in cur.fetchall()]
            conn.close()
            if len(scores) < 2:
                return 0.0
            return statistics.stdev(scores)
        except Exception:
            return 0.0


# ─────────────────────────────────────────────────────────────
# Path 1: LLM-as-Judge 预测器
# ─────────────────────────────────────────────────────────────

class LLMJudgePredictor(ToolFitnessPredictor):
    """
    使用现有 LLM 作为 Judge

    论文核心迁移：
      论文："Is restarting likely to produce a better response? YES/NO"
            → LLM 输出 P(YES) 判断是否值得继续生成
      MSTAR："Is mutating this tool likely to improve its fitness? YES/NO"
            → LLM 输出 P(YES) 判断是否值得变异

    优点：零训练成本，利用现有 LLM 的内部知识
    缺点：每次调用需要一次 LLM API，有延迟和成本
    """

    PROMPT_TEMPLATE = """作为AI工具进化评估专家，评估以下工具变异的预期收益。

工具信息：
- 名称: {tool_name}
- 当前 fitness: {fitness:.3f} (0=完全失败, 1=完美)
- 进化深度: {lineage_depth}
- 已变异次数: {evolutions}
- 错误率: {error_rate:.1%}

候选变异策略: {strategy}

请评估：该工具执行此变异后，fitness 提升的概率是多少？

分析：{reasoning}

答案是：P(提升) = {prob}（填入0.0~1.0之间的小数，仅输出数字）"""

    def __init__(self, llm_client, model: str = "MiniMax-M2.7"):
        """
        Args:
            llm_client: LLM 客户端，需支持 chat.completions.create 接口
            model: 模型名称，默认 MiniMax-M2.7（可覆盖）
        """
        self.llm = llm_client
        self.model = model

    def predict_mutation_benefit(self, program, strategy: str) -> float:
        """调用 LLM 判断变异收益概率"""
        tool_name = getattr(program, 'name', None) or getattr(program, 'program_id', 'unknown')
        fitness = getattr(program, 'fitness_score', None) or 0.5
        lineage_depth = getattr(program, 'lineage_depth', None) or 0

        # 推理过程（模拟）
        reasoning = self._generate_reasoning(program, strategy)

        prompt = self.PROMPT_TEMPLATE.format(
            tool_name=tool_name,
            fitness=fitness,
            lineage_depth=lineage_depth,
            evolutions=lineage_depth,  # lineage_depth ≈ 变异次数
            error_rate=0.05,           # TODO: 接入真实 error_rate
            strategy=strategy,
            reasoning=reasoning,
            prob="?"
        )

        try:
            response = self.llm.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=20,
                temperature=0.0,  # 确定性输出
            )
            prob_text = response.choices[0].message.content.strip()
            # 提取数字
            import re
            match = re.search(r'0?\.\d+|1\.0+', prob_text)
            if match:
                prob = float(match.group())
                return max(0.0, min(1.0, prob))
            # 尝试解析百分比
            match = re.search(r'(\d+(?:\.\d+)?)\s*%', prob_text)
            if match:
                return max(0.0, min(1.0, float(match.group(1)) / 100.0))
            logger.warning(f"[LLMJudge] Could not parse probability from: {prob_text}")
            return 0.5
        except Exception as e:
            logger.warning(f"[LLMJudge] predict failed: {e}")
            return 0.5

    def _generate_reasoning(self, program, strategy: str) -> str:
        """生成简短的推理过程（用于 prompt）"""
        f = getattr(program, 'fitness_score', None) or 0.5
        depth = getattr(program, 'lineage_depth', None) or 0

        if f < 0.35:
            return "fitness极低，变异很可能带来改善。"
        elif f > 0.75:
            return "fitness已较高，变异风险较大。"
        elif depth > 3:
            return "已多次变异，边际收益递减。"
        else:
            return "处于中等区间，变异有一定收益可能性。"

    def predict_best_strategy(self, program) -> List[Tuple[str, float]]:
        """对所有策略并行查询 LLM 并排序"""
        from mstar_core.evolution.mutator import MSTARMutator
        strategies = MSTARMutator.MUTATION_STRATEGIES
        results = []
        for s in strategies:
            prob = self.predict_mutation_benefit(program, s)
            results.append((s, prob))
        return sorted(results, key=lambda x: -x[1])


# ─────────────────────────────────────────────────────────────
# Path 2: 训练型预测器（待数据积累后启用）
# ─────────────────────────────────────────────────────────────

class TrainedPredictor(ToolFitnessPredictor):
    """
    基于历史变异数据的回归预测器

    论文迁移：类似 2410.02727（如果存在）基于真实反馈训练预测模型

    训练数据来源：
      每次真实变异后调用 record_outcome() 记录：
        - 特征: (fitness, lineage_depth, strategy_type, trend, volatility)
        - 标签: delta > 0 ? 1 : 0

    模型选择：
      - Logistic Regression（可解释性强）
      - XGBoost（效果更好）
    """

    def __init__(self, fitness_tracker: Optional["FitnessTracker"] = None):
        self.fitness_tracker = fitness_tracker
        self.model = None
        self._strategy_encoding = self._build_strategy_encoding()
        self._training_records: List[dict] = []

    def _build_strategy_encoding(self) -> dict:
        from mstar_core.evolution.mutator import MSTARMutator
        strategies = MSTARMutator.MUTATION_STRATEGIES
        return {s: i / len(strategies) for i, s in enumerate(strategies)}

    def predict_best_strategy(self, program) -> List[Tuple[str, float]]:
        """无模型时回退到 RuleBased"""
        if self.model is None:
            fallback = RuleBasedPredictor(self.fitness_tracker)
            return fallback.predict_best_strategy(program)

        from mstar_core.evolution.mutator import MSTARMutator
        strategies = MSTARMutator.MUTATION_STRATEGIES
        results = []
        for s in strategies:
            prob = self.predict_mutation_benefit(program, s)
            results.append((s, prob))
        return sorted(results, key=lambda x: -x[1])

    def predict_mutation_benefit(self, program, strategy: str) -> float:
        """有模型时用模型预测，无模型时回退"""
        if self.model is None:
            fallback = RuleBasedPredictor(self.fitness_tracker)
            return fallback.predict_mutation_benefit(program, strategy)

        import numpy as np
        program_id = getattr(program, 'program_id', None)
        features = np.array([
            getattr(program, 'fitness_score', None) or 0.5,
            (getattr(program, 'lineage_depth', None) or 0) / 10.0,
            self._strategy_encoding.get(strategy, 0.5),
            self._get_trend(program_id) if program_id and self.fitness_tracker else 0.0,
            self._get_volatility(program_id) if program_id and self.fitness_tracker else 0.0,
        ])
        try:
            prob = self.model.predict_proba([features])[0][1]
            return float(prob)
        except Exception as e:
            logger.warning(f"[TrainedPredictor] prediction failed: {e}")
            fallback = RuleBasedPredictor(self.fitness_tracker)
            return fallback.predict_mutation_benefit(program, strategy)

    def _get_trend(self, program_id: str) -> float:
        if not program_id or not self.fitness_tracker:
            return 0.0
        try:
            import sqlite3
            conn = sqlite3.connect(self.fitness_tracker.db_path, timeout=10)
            cur = conn.execute("""
                SELECT fitness_score FROM fitness_snapshots
                WHERE program_id = ? AND fitness_score IS NOT NULL
                ORDER BY timestamp DESC LIMIT 2
            """, (program_id,))
            rows = cur.fetchall()
            conn.close()
            if len(rows) >= 2:
                return rows[0][0] - rows[1][0]
            return 0.0
        except Exception:
            return 0.0

    def _get_volatility(self, program_id: str) -> float:
        if not program_id or not self.fitness_tracker:
            return 0.0
        try:
            import sqlite3, statistics
            conn = sqlite3.connect(self.fitness_tracker.db_path, timeout=10)
            cur = conn.execute("""
                SELECT fitness_score FROM fitness_snapshots
                WHERE program_id = ? AND fitness_score IS NOT NULL
                ORDER BY timestamp DESC LIMIT 10
            """, (program_id,))
            scores = [r[0] for r in cur.fetchall()]
            conn.close()
            if len(scores) < 2:
                return 0.0
            return statistics.stdev(scores)
        except Exception:
            return 0.0

    def record_outcome(self, program_id: str, strategy: str,
                       predicted_prob: float, actual_delta: float):
        """
        记录变异结果，用于后续训练或分析。

        调用时机：EvolutionEngine 执行完变异后

        Args:
            program_id: 工具ID
            strategy: 使用的策略
            predicted_prob: 预测概率（predict_mutation_benefit 的返回值）
            actual_delta: 实际 fitness 变化量 (fitness_after - fitness_before)
        """
        record = {
            'program_id': program_id,
            'strategy': strategy,
            'predicted_prob': predicted_prob,
            'actual_delta': actual_delta,
            'improved': actual_delta > 0,
            'timestamp': None,  # 待填充
        }
        self._training_records.append(record)
        logger.info(
            f"[TrainedPredictor] recorded: prog={program_id} "
            f"strategy={strategy} pred={predicted_prob:.3f} "
            f"actual_delta={actual_delta:+.4f} improved={actual_delta > 0}"
        )

    def train_model(self):
        """
        基于已有记录训练 LogisticRegression 模型。

        特征向量: [fitness_score, lineage_depth/N, strategy_encoding, trend, volatility]
        标签:     improved (delta > 0 ? 1 : 0)

        需要 >= 20 条记录才训练。训练后 self.model 可用，
        predict_mutation_benefit 自动切换到模型预测。
        """
        if len(self._training_records) < 20:
            logger.info(f"[TrainedPredictor] Not enough records to train: {len(self._training_records)} < 20")
            return

        try:
            import numpy as np
            from sklearn.linear_model import LogisticRegression
        except ImportError:
            logger.warning("[TrainedPredictor] sklearn not available, cannot train model")
            return

        # ── 构建特征矩阵 ────────────────────────────────────
        X_rows = []
        y_rows = []
        for rec in self._training_records:
            pid = rec.get('program_id')
            strategy = rec.get('strategy', 'random')

            # 从 DB 查询程序当前状态（fitness, lineage_depth）
            fitness = 0.5
            depth = 0
            if pid and self.fitness_tracker:
                try:
                    import sqlite3
                    conn = sqlite3.connect(self.fitness_tracker.db_path, timeout=10)
                    row = conn.execute(
                        "SELECT fitness_score, lineage_depth FROM programs WHERE program_id = ?",
                        (pid,)
                    ).fetchone()
                    conn.close()
                    if row:
                        fitness = row[0] or 0.5
                        depth = row[1] or 0
                except Exception:
                    pass

            trend = self._get_trend(pid) if pid else 0.0
            volatility = self._get_volatility(pid) if pid else 0.0

            features = np.array([
                fitness,
                depth / 10.0,
                self._strategy_encoding.get(strategy, 0.5),
                trend,
                volatility,
            ], dtype=np.float32)
            X_rows.append(features)
            y_rows.append(1 if rec.get('improved') else 0)

        X = np.array(X_rows, dtype=np.float32)
        y = np.array(y_rows, dtype=np.int32)

        # ── 训练 ───────────────────────────────────────────
        model = LogisticRegression(
            max_iter=200,
            class_weight='balanced',  # 处理类别不平衡
            random_state=42,
        )
        model.fit(X, y)

        self.model = model
        logger.info(
            f"[TrainedPredictor] Model trained on {len(self._training_records)} records. "
            f"Accuracy: {model.score(X, y):.3f}"
        )


# ─────────────────────────────────────────────────────────────
# 工厂函数
# ─────────────────────────────────────────────────────────────

def create_predictor(predictor_type: str = "rule",
                     fitness_tracker=None,
                     llm_client=None) -> ToolFitnessPredictor:
    """
    预测器工厂函数

    Args:
        predictor_type: "rule" | "llm" | "trained"
        fitness_tracker: FitnessTracker 实例（用于 RuleBased 和 Trained）
        llm_client: LLM 客户端（用于 LLMJudge）
    """
    if predictor_type == "rule":
        return RuleBasedPredictor(fitness_tracker)
    elif predictor_type == "llm":
        if llm_client is None:
            raise ValueError("LLMJudge predictor requires llm_client")
        return LLMJudgePredictor(llm_client)
    elif predictor_type == "trained":
        return TrainedPredictor(fitness_tracker)
    else:
        raise ValueError(f"Unknown predictor type: {predictor_type}")