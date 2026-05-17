"""
MSTAR Pro v4.0 - Phase 1c: Perturbation-Robust Fitness
参考: Learning Perturbations (2605.13284)

核心思想: Fitness分数必须对环境扰动（测量噪声、模型非确定性、上下文变化）具有鲁棒性。
论文关键可移植点:
- Bootstrap Confidence Intervals: 不使用点估计，使用置信区间
- Perturbation Resistance: 在受控扰动下测试 fitness 稳定性
- Cross-Validation: 分 fold 验证 fitness 可靠性

新增 FitnessDimensions:
- fitness_volatility: 多次测量的方差
- perturbation_resistance: 扰动下的稳定性得分
- confidence_bandwidth: 置信区间宽度
- measurement_noise_floor: 测量噪声底
"""

from __future__ import annotations
import logging
import random
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class PerturbationResult:
    """单次扰动测试结果"""
    program_id: str
    baseline_fitness: float
    perturbed_fitness: float
    noise_level: float
    noise_type: str
    timestamp: str
    stable: bool  # perturbed_fitness 在置信区间内


@dataclass
class BootstrapCI:
    """Bootstrap 置信区间"""
    mean: float
    lower: float
    upper: float
    width: float  # upper - lower
    n_samples: int
    confidence_level: float


class PerturbationRobustFitness:
    """
    MSTAR Pro v4.0 Phase 1c: 扰动鲁棒适应度分析器

    功能:
    1. 对每个Program进行多次测量，计算统计显著性
    2. 使用Bootstrap方法计算fitness的置信区间
    3. 注入受控噪声测试扰动抵抗能力
    4. 将鲁棒性指标加入FitnessDimensions

    使用场景:
    - EvolutionEngine.select_mutation_target: 只选择置信区间窄（稳定）的program
    - FitnessTracker.update: 计算perturbation_resistance
    - Ablation study: 对比不同扰动级别下的系统表现
    """

    NOISE_TYPES = ['gaussian', 'uniform', 'dropout', 'context_shift', 'token_drift']

    # 不同扰动级别对应的噪声强度
    NOISE_LEVELS = {
        'off': 0.0,
        'low': 0.05,    # 5% 噪声
        'medium': 0.10, # 10% 噪声
        'high': 0.20,   # 20% 噪声
    }

    def __init__(
        self,
        n_bootstrap: int = 100,
        confidence_level: float = 0.95,
        default_noise_level: str = 'medium',
    ):
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.default_noise_level = default_noise_level

        # 扰动结果缓存: program_id -> List[PerturbationResult]
        self._perturbation_cache: Dict[str, List[PerturbationResult]] = {}

        # 统计
        self._total_perturbations = 0
        self._stable_count = 0

    def compute_bootstrap_ci(self, fitness_history: List[float]) -> BootstrapCI:
        """
        使用Bootstrap方法计算fitness的置信区间。

        Args:
            fitness_history: 历史fitness分数列表

        Returns:
            BootstrapCI: 包含均值、下界、上界、宽度
        """
        if not fitness_history:
            return BootstrapCI(
                mean=0.5, lower=0.5, upper=0.5,
                width=0.0, n_samples=0, confidence_level=self.confidence_level,
            )

        history = np.array(fitness_history)
        n = len(history)

        if n < 3:
            # 数据太少，使用简化的置信区间
            mean = np.mean(history)
            std = np.std(history) if n > 1 else 0.1
            return BootstrapCI(
                mean=mean,
                lower=max(0.0, mean - 2 * std),
                upper=min(1.0, mean + 2 * std),
                width=4 * std,
                n_samples=n,
                confidence_level=self.confidence_level,
            )

        # Bootstrap重采样
        bootstrap_means = []
        for _ in range(self.n_bootstrap):
            sample = np.random.choice(history, size=n, replace=True)
            bootstrap_means.append(np.mean(sample))

        bootstrap_means = np.array(bootstrap_means)
        mean = np.mean(bootstrap_means)

        # 计算置信区间
        alpha = 1 - self.confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        lower = np.percentile(bootstrap_means, lower_percentile)
        upper = np.percentile(bootstrap_means, upper_percentile)

        return BootstrapCI(
            mean=float(mean),
            lower=float(max(0.0, lower)),
            upper=float(min(1.0, upper)),
            width=float(upper - lower),
            n_samples=n,
            confidence_level=self.confidence_level,
        )

    def inject_noise(self, value: float, noise_level: float, noise_type: str = 'gaussian') -> float:
        """
        向值注入噪声，模拟环境扰动。

        Args:
            value: 原始值 (0-1范围)
            noise_level: 噪声强度 (0-1)
            noise_type: 'gaussian', 'uniform', 'dropout', 'context_shift', 'token_drift'

        Returns:
            扰动后的值 (clamped到0-1)
        """
        if noise_level <= 0:
            return value

        if noise_type == 'gaussian':
            noise = np.random.normal(0, noise_level * 0.3)
        elif noise_type == 'uniform':
            noise = np.random.uniform(-noise_level, noise_level)
        elif noise_type == 'dropout':
            if random.random() < noise_level:
                return 0.0  # 完全失败
            noise = 0.0
        elif noise_type == 'context_shift':
            # 上下文偏移：偏向中间值（回归均值）
            noise = (0.5 - value) * noise_level * 0.5
        elif noise_type == 'token_drift':
            # Token漂移：成比例偏移
            drift = random.uniform(-1, 1) * noise_level
            noise = value * drift
        else:
            noise = 0.0

        return max(0.0, min(1.0, value + noise))

    def run_perturbation_test(
        self,
        program_id: str,
        baseline_fitness: float,
        fitness_history: List[float],
        noise_level: float = 0.1,
        noise_type: str = 'gaussian',
        n_trials: int = 20,
    ) -> Dict:
        """
        运行完整的扰动测试。

        1. 计算baseline的Bootstrap CI
        2. 进行n_trials次扰动测量
        3. 计算扰动抵抗能力
        4. 判断每次扰动是否stable（在CI内）

        Returns:
            {
                'ci': BootstrapCI,
                'perturbation_results': List[PerturbationResult],
                'perturbation_resistance': float,  # stable比率
                'mean_perturbed': float,
                'max_deviation': float,
            }
        """
        # 计算baseline CI
        ci = self.compute_bootstrap_ci(fitness_history)

        # 运行扰动试验
        perturbation_results: List[PerturbationResult] = []
        stable_count = 0

        for i in range(n_trials):
            perturbed = self.inject_noise(baseline_fitness, noise_level, noise_type)

            # 判断是否stable（在CI内）
            stable = ci.lower <= perturbed <= ci.upper

            if stable:
                stable_count += 1

            result = PerturbationResult(
                program_id=program_id,
                baseline_fitness=baseline_fitness,
                perturbed_fitness=perturbed,
                noise_level=noise_level,
                noise_type=noise_type,
                timestamp=datetime.now().isoformat(),
                stable=stable,
            )
            perturbation_results.append(result)

        # 计算指标
        perturbation_resistance = stable_count / n_trials
        perturbed_values = [r.perturbed_fitness for r in perturbation_results]
        mean_perturbed = np.mean(perturbed_values)
        max_deviation = max(abs(v - baseline_fitness) for v in perturbed_values)

        # 缓存
        if program_id not in self._perturbation_cache:
            self._perturbation_cache[program_id] = []
        self._perturbation_cache[program_id].extend(perturbation_results)

        # 更新统计
        self._total_perturbations += n_trials
        self._stable_count += stable_count

        return {
            'ci': ci,
            'perturbation_results': perturbation_results,
            'perturbation_resistance': float(perturbation_resistance),
            'mean_perturbed': float(mean_perturbed),
            'max_deviation': float(max_deviation),
            'noise_level': noise_level,
            'noise_type': noise_type,
            'n_trials': n_trials,
        }

    def get_robustness_report(self, program_id: str) -> Dict:
        """
        获取某个program的完整鲁棒性报告。
        """
        results = self._perturbation_cache.get(program_id, [])

        if not results:
            return {
                'program_id': program_id,
                'status': 'no_perturbation_data',
                'message': '尚未进行扰动测试',
            }

        # 按noise_level分组
        by_level: Dict[str, List[PerturbationResult]] = {}
        for r in results:
            key = f"{r.noise_type}_{r.noise_level}"
            if key not in by_level:
                by_level[key] = []
            by_level[key].append(r)

        report = {
            'program_id': program_id,
            'total_tests': len(results),
            'breakdown': {},
        }

        for key, items in by_level.items():
            stable = sum(1 for r in items if r.stable)
            report['breakdown'][key] = {
                'n': len(items),
                'stable_ratio': stable / len(items),
            }

        return report

    def evaluate_population_robustness(self, programs: List, noise_level: float = 0.1) -> Dict:
        """
        评估整个群体的鲁棒性。

        用于：
        - 选择最鲁棒的program进行下一步进化
        - 识别需要加固的脆弱program

        Returns:
            {
                'avg_perturbation_resistance': float,
                'robust_programs': List[program_id],
                'fragile_programs': List[program_id],
                'population_stability': float,
            }
        """
        resistances = []

        for program in programs:
            fitness_history = getattr(program, 'fitness_history', [])
            baseline = getattr(program, 'fitness_score', 0.5)

            result = self.run_perturbation_test(
                program_id=program.program_id,
                baseline_fitness=baseline,
                fitness_history=fitness_history,
                noise_level=noise_level,
                n_trials=10,
            )
            resistances.append((program.program_id, result['perturbation_resistance']))

        # 按抵抗能力排序
        resistances.sort(key=lambda x: x[1], reverse=True)

        avg_resistance = sum(r for _, r in resistances) / len(resistances) if resistances else 0.0

        threshold = avg_resistance * 0.8  # 低于平均值80%的视为脆弱

        robust = [pid for pid, r in resistances if r >= threshold]
        fragile = [pid for pid, r in resistances if r < threshold]

        return {
            'avg_perturbation_resistance': avg_resistance,
            'robust_programs': robust,
            'fragile_programs': fragile,
            'population_stability': avg_resistance,
            'all_resistances': dict(resistances),
        }

    def get_statistics(self) -> Dict:
        """获取扰动分析统计"""
        overall_stability = self._stable_count / self._total_perturbations if self._total_perturbations > 0 else 0.0
        return {
            'total_perturbations': self._total_perturbations,
            'stable_count': self._stable_count,
            'overall_stability_ratio': overall_stability,
            'programs_tested': len(self._perturbation_cache),
            'n_bootstrap': self.n_bootstrap,
            'confidence_level': self.confidence_level,
        }


# =============================================================================
# 集成到FitnessDimensions的鲁棒性指标
# =============================================================================

# 在FitnessDimensions.ADVANCED_dims末尾添加:
ROBUSTNESS_dims = [
    'fitness_volatility',      # 方差（低=稳定）
    'perturbation_resistance', # 扰动抵抗能力（高=鲁棒）
    'confidence_bandwidth',    # 置信区间宽度（窄=精确）
    'measurement_noise_floor', # 测量噪声底（低=精确）
    'cross_validation_score',  # 交叉验证得分（高=可靠）
]