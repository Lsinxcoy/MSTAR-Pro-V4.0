"""
CANTANTE Contrastive Credit Attribution - Layer 3 of MSTAR Pro V4.0

Implements contrastive credit attribution for multi-agent/multi-tool systems.
Reference: CANTANTE (arXiv:2605.15155)

Key idea: Compare rollouts of different joint configurations on the same query
to decompose system-level rewards into per-agent update signals.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict
import numpy as np


@dataclass
class RolloutResult:
    """Result of a single rollout (one configuration on one query)."""
    rollout_id: str
    query: str
    config: Dict[str, Any]  # Configuration of agents/tools
    success: bool
    reward: float  # System-level reward
    tokens_consumed: int = 0
    latency_ms: float = 0.0
    error: Optional[str] = None


@dataclass
class AgentConfig:
    """Configuration for a single agent/tool."""
    agent_id: str
    prompt: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True


@dataclass
class CreditScore:
    """Credit score for a single agent."""
    agent_id: str
    credit: float  # 0.0 to 1.0, higher = more responsible
    positive_evidence: List[str] = field(default_factory=list)
    negative_evidence: List[str] = field(default_factory=list)
    confidence: float = 1.0


@dataclass
class AttributionResult:
    """Result of credit attribution."""
    query: str
    credits: List[CreditScore]
    root_cause_agent: Optional[str] = None
    explanation: str = ""
    method: str = "cantante"


class ContrastiveAnalyzer:
    """
    Analyzes contrastive rollouts to attribute credit.
    
    Takes multiple rollouts with different configurations on the same query
    and identifies which agent configurations led to better/worse outcomes.
    """
    
    def __init__(self, top_k: int = 3):
        self.top_k = top_k  # Top-k configurations to compare
    
    def analyze(self, rollouts: List[RolloutResult]) -> List[CreditScore]:
        """
        Analyze rollouts to compute per-agent credit scores.
        
        Method:
        1. Separate successful and failed rollouts
        2. Identify which agent configs differ between success and failure
        3. Attribute credit based on config differences
        """
        if len(rollouts) < 2:
            return self._uniform_credit(rollouts)
        
        successful = [r for r in rollouts if r.success]
        failed = [r for r in rollouts if not r.success]
        
        if not successful or not failed:
            return self._uniform_credit(rollouts)
        
        # Collect all agent IDs
        agent_ids = set()
        for r in rollouts:
            agent_ids.update(r.config.keys())
        
        credits = []
        
        for agent_id in agent_ids:
            credit = self._compute_credit(agent_id, successful, failed)
            credits.append(credit)
        
        # Normalize credits to sum to 1.0
        total = sum(c.credit for c in credits)
        if total > 0:
            for c in credits:
                c.credit /= total
        
        return credits
    
    def _compute_credit(self, agent_id: str, 
                       successful: List[RolloutResult],
                       failed: List[RolloutResult]) -> CreditScore:
        """Compute credit for a single agent based on config differences."""
        positive_evidence = []
        negative_evidence = []
        
        # Analyze successful rollouts
        success_configs = [r.config.get(agent_id) for r in successful]
        
        # Analyze failed rollouts
        fail_configs = [r.config.get(agent_id) for r in failed]
        
        # Compute credit based on correlation with success/failure
        # If agent config is consistently different between success and failure,
        # it likely caused the difference
        
        # Simple heuristic: count how often this agent's config differs
        # between success and failure
        same_in_success = self._count_agreement(success_configs)
        same_in_failure = self._count_agreement(fail_configs)
        diff_between = abs(same_in_success - same_in_failure)
        
        # Higher credit if config consistently differs between success/failure
        credit = min(diff_between / max(len(successful), len(failed)), 1.0)
        
        # Add evidence
        if same_in_success > same_in_failure:
            positive_evidence.append(f"Consistent config in success: {same_in_success}/{len(successful)}")
        elif same_in_failure > same_in_success:
            negative_evidence.append(f"Consistent config in failure: {same_in_failure}/{len(failed)}")
        
        return CreditScore(
            agent_id=agent_id,
            credit=credit,
            positive_evidence=positive_evidence,
            negative_evidence=negative_evidence,
            confidence=min(credit + 0.1, 1.0)
        )
    
    def _count_agreement(self, configs: List) -> int:
        """Count how many configs are the same."""
        if not configs:
            return 0
        first = configs[0]
        return sum(1 for c in configs if c == first)
    
    def _uniform_credit(self, rollouts: List[RolloutResult]) -> List[CreditScore]:
        """Return uniform credit if not enough data."""
        agent_ids = set()
        for r in rollouts:
            agent_ids.update(r.config.keys())
        
        n = len(agent_ids)
        if n == 0:
            return []
        
        uniform = 1.0 / n
        return [
            CreditScore(
                agent_id=aid,
                credit=uniform,
                positive_evidence=["Uniform credit due to insufficient data"],
                negative_evidence=[],
                confidence=0.5
            )
            for aid in agent_ids
        ]


class CANTANTEAttributor:
    """
    CANTANTE: Contrastive Credit Attribution for Multi-Agent Systems
    
    Main entry point for Phase 1b credit attribution.
    
    Usage:
        attrib = CANTANTEAttributor()
        result = attrib.attribute(query, agent_configs, system_reward)
        # result.credits tells which tools to evolve
    """
    
    def __init__(self, min_rollouts: int = 3):
        self.min_rollouts = min_rollouts
        self.analyzer = ContrastiveAnalyzer()
        self._rollout_history: List[RolloutResult] = []
        self._attribution_cache: Dict[str, List[CreditScore]] = {}
    
    def add_rollout(self, rollout: RolloutResult):
        """Add a rollout result for future attribution."""
        self._rollout_history.append(rollout)
        # Keep only last 1000 rollouts
        if len(self._rollout_history) > 1000:
            self._rollout_history = self._rollout_history[-1000:]
        # Invalidate cache
        self._attribution_cache.clear()
    
    def add_rollouts(self, rollouts: List[RolloutResult]):
        """Add multiple rollout results."""
        for r in rollouts:
            self.add_rollout(r)
    
    def attribute(self, query: str, 
                  agent_configs: List[Dict[str, Any]],
                  rewards: List[float]) -> AttributionResult:
        """
        Attribute credit based on rollouts with different configs.
        
        Args:
            query: The query/task being evaluated
            agent_configs: List of agent configurations tried
            rewards: Corresponding reward for each config
            
        Returns:
            AttributionResult with per-agent credit scores
        """
        # Create rollouts
        rollouts = []
        for i, (config, reward) in enumerate(zip(agent_configs, rewards)):
            rollout = RolloutResult(
                rollout_id=f"rollout_{i}",
                query=query,
                config=config,
                success=reward > 0.5,  # Threshold for success
                reward=reward
            )
            rollouts.append(rollout)
        
        # Add to history
        self.add_rollouts(rollouts)
        
        # Analyze
        credits = self.analyzer.analyze(rollouts)
        
        # Find root cause (highest credit agent)
        root_cause = None
        if credits:
            max_credit = max(c.credit for c in credits)
            root_causes = [c for c in credits if c.credit >= max_credit * 0.9]
            if root_causes:
                root_cause = root_causes[0].agent_id
        
        return AttributionResult(
            query=query,
            credits=credits,
            root_cause_agent=root_cause,
            explanation=self._generate_explanation(credits, root_cause),
            method="cantante"
        )
    
    def attribute_from_history(self, query: str) -> AttributionResult:
        """
        Attribute credit using historical rollouts for the same query.
        
        Args:
            query: The query to find rollouts for
            
        Returns:
            AttributionResult from cached rollouts
        """
        # Check cache
        if query in self._attribution_cache:
            return AttributionResult(
                query=query,
                credits=self._attribution_cache[query],
                root_cause_agent=self._attribution_cache[query][0].agent_id if self._attribution_cache[query] else None,
                explanation="From cache",
                method="cantante_cached"
            )
        
        # Find matching rollouts
        matching = [r for r in self._rollout_history if r.query == query]
        
        if len(matching) < self.min_rollouts:
            return AttributionResult(
                query=query,
                credits=[],
                root_cause_agent=None,
                explanation=f"Insufficient rollouts: {len(matching)} < {self.min_rollouts}",
                method="cantante_insufficient"
            )
        
        credits = self.analyzer.analyze(matching)
        self._attribution_cache[query] = credits
        
        root_cause = credits[0].agent_id if credits else None
        
        return AttributionResult(
            query=query,
            credits=credits,
            root_cause_agent=root_cause,
            explanation=self._generate_explanation(credits, root_cause),
            method="cantante_history"
        )
    
    def _generate_explanation(self, credits: List[CreditScore], 
                              root_cause: Optional[str]) -> str:
        """Generate human-readable explanation."""
        if not credits:
            return "No credit data available."
        
        lines = ["CANTANTE Credit Attribution:"]
        for c in sorted(credits, key=lambda x: x.credit, reverse=True):
            marker = " [ROOT CAUSE]" if c.agent_id == root_cause else ""
            lines.append(f"  {c.agent_id}: {c.credit:.3f}{marker}")
            for ev in c.positive_evidence:
                lines.append(f"    + {ev}")
            for ev in c.negative_evidence:
                lines.append(f"    - {ev}")
        
        return "\n".join(lines)
    
    def should_evolve(self, agent_id: str, threshold: float = 0.5) -> bool:
        """
        Determine if an agent should evolve based on credit.
        
        Args:
            agent_id: The agent to check
            threshold: Minimum credit to trigger evolution
            
        Returns:
            True if agent should evolve
        """
        # Find credit for this agent from recent attribution
        for rollout in self._rollout_history[-10:]:
            if agent_id in rollout.config:
                # Compute credit from this rollout
                credits = self.analyzer.analyze([rollout])
                for c in credits:
                    if c.agent_id == agent_id and c.credit >= threshold:
                        return True
        return False
    
    def get_top_candidates(self, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Get top-k agents that should be evolved.
        
        Returns:
            List of (agent_id, credit) tuples
        """
        # Aggregate credits from recent rollouts
        agent_credits = defaultdict(float)
        agent_counts = defaultdict(int)
        
        for rollout in self._rollout_history[-50:]:
            credits = self.analyzer.analyze([rollout])
            for c in credits:
                agent_credits[c.agent_id] += c.credit
                agent_counts[c.agent_id] += 1
        
        # Average credits
        avg_credits = [
            (aid, agent_credits[aid] / max(agent_counts[aid], 1))
            for aid in agent_credits
        ]
        
        # Sort and return top-k
        avg_credits.sort(key=lambda x: x[1], reverse=True)
        return avg_credits[:top_k]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get attribution statistics."""
        return {
            'total_rollouts': len(self._rollout_history),
            'cached_queries': len(self._attribution_cache),
            'top_candidates': self.get_top_candidates(5),
        }


class ContrastiveCreditAttributor:
    """
    Wrapper class for CANTANTE attribution in MSTAR Core.
    
    This is the main interface for the Evolution Engine to use
    contrastive credit attribution.
    """
    
    def __init__(self):
        self._cantante = CANTANTEAttributor()
    
    def record_session(self, query: str, agent_configs: Dict[str, Any], 
                       reward: float, success: bool):
        """Record a session for future attribution."""
        rollout = RolloutResult(
            rollout_id=f"session_{hash(query)}",
            query=query,
            config=agent_configs,
            success=success,
            reward=reward
        )
        self._cantante.add_rollout(rollout)
    
    def get_credit(self, agent_id: str) -> float:
        """Get current credit score for an agent."""
        candidates = self._cantante.get_top_candidates(10)
        for aid, credit in candidates:
            if aid == agent_id:
                return credit
        return 0.0
    
    def get_evolution_candidates(self, threshold: float = 0.3) -> List[str]:
        """Get agents that should evolve based on credit scores."""
        candidates = self._cantante.get_top_candidates(10)
        return [aid for aid, credit in candidates if credit >= threshold]
    
    def attribute_failure(self, query: str, 
                         agent_configs: List[Dict[str, Any]],
                         rewards: List[float]) -> AttributionResult:
        """Attribute failure to specific agents."""
        return self._cantante.attribute(query, agent_configs, rewards)
    
    def explain(self, query: str) -> str:
        """Get explanation for a query."""
        result = self._cantante.attribute_from_history(query)
        return result.explanation
    
    def get_stats(self) -> Dict[str, Any]:
        """Get attribution statistics."""
        return self._cantante.get_statistics()