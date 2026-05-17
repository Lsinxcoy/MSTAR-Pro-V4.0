"""
LIFE Failure Attribution Layer - Layer 2 of MSTAR Pro V4.0

Implements failure attribution for multi-agent/multi-tool systems.
Reference: LIFE Framework (arXiv:2605.14892)

Three attribution methods:
1. Data-Driven Attribution: Statistical contribution analysis
2. Constraint-Guided Attribution: Check against pre-defined constraints
3. Causal Inference Attribution: Build causal graphs, identify key paths
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from collections import defaultdict
from enum import Enum
import math


class AttributionMethod(Enum):
    DATA_DRIVEN = "data_driven"
    CONSTRAINT_GUIDED = "constraint_guided"
    CAUSAL_INFERENCE = "causal_inference"
    ALL = "all"


@dataclass
class AgentContribution:
    """Represents an agent/tool's contribution to a failure."""
    agent_id: str
    contribution_score: float  # 0.0 to 1.0
    contribution_type: str     # 'data', 'constraint', 'causal'
    evidence: List[str] = field(default_factory=list)
    root_cause: bool = False


@dataclass
class FailureTrace:
    """Represents a failure trace for attribution analysis."""
    trace_id: str
    error_message: str
    agents_involved: List[str] = field(default_factory=list)
    operations: List[Dict] = field(default_factory=list)  # List of {agent, operation, success, duration}
    context: Dict = field(default_factory=dict)  # Additional context
    timestamp: float = 0.0


@dataclass
class Constraint:
    """Represents a constraint for constraint-guided attribution."""
    name: str
    description: str
    constraint_type: str  # 'temporal', 'permission', 'resource'
    violated: bool = False
    violated_by: Optional[str] = None


@dataclass
class CausalEdge:
    """Represents a causal relationship between agents."""
    source: str
    target: str
    weight: float  # Causal strength
    relationship: str  # 'enables', 'depends_on', 'blocks'


class DataDrivenAttributor:
    """
    Data-Driven Attribution using statistical analysis.
    
    Analyzes how each agent contributed to the final failure
    by examining operation patterns and outcomes.
    """
    
    def __init__(self):
        self._historical_traces: List[FailureTrace] = []
    
    def attribute(self, trace: FailureTrace) -> List[AgentContribution]:
        """
        Attribute failure to agents using statistical analysis.
        
        Method:
        1. Calculate base failure rate per agent
        2. Weight by recency (recent failures matter more)
        3. Normalize to contribution scores
        """
        if not trace.operations:
            return []
        
        # Calculate per-agent statistics
        agent_stats = defaultdict(lambda: {'failures': 0, 'total': 0, 'total_latency': 0.0})
        
        for op in trace.operations:
            agent_id = op.get('agent', 'unknown')
            success = op.get('success', True)
            latency = op.get('duration', 0.0)
            
            agent_stats[agent_id]['total'] += 1
            agent_stats[agent_id]['total_latency'] += latency
            if not success:
                agent_stats[agent_id]['failures'] += 1
        
        # Calculate contribution scores
        contributions = []
        total_failures = sum(s['failures'] for s in agent_stats.values())
        
        if total_failures == 0:
            # No failures - distribute based on latency contribution
            total_latency = sum(s['total_latency'] for s in agent_stats.values())
            for agent_id, stats in agent_stats.items():
                score = stats['total_latency'] / max(total_latency, 1e-9)
                contributions.append(AgentContribution(
                    agent_id=agent_id,
                    contribution_score=score,
                    contribution_type='data_driven',
                    evidence=[f'Latency contribution: {score:.3f}']
                ))
        else:
            for agent_id, stats in agent_stats.items():
                # Failure rate weighted by recency
                failure_rate = stats['failures'] / max(stats['total'], 1)
                
                # Weighted contribution
                score = (stats['failures'] / total_failures) * failure_rate
                
                contributions.append(AgentContribution(
                    agent_id=agent_id,
                    contribution_score=score,
                    contribution_type='data_driven',
                    evidence=[
                        f'Failures: {stats["failures"]}/{stats["total"]}',
                        f'Failure rate: {failure_rate:.2%}',
                        f'Avg latency: {stats["total_latency"]/max(stats["total"],1):.2f}ms'
                    ]
                ))
        
        # Sort by contribution score descending
        contributions.sort(key=lambda c: c.contribution_score, reverse=True)
        
        # Mark root cause (highest contributor)
        if contributions:
            contributions[0].root_cause = True
        
        return contributions
    
    def add_historical_trace(self, trace: FailureTrace):
        """Add to historical traces for better attribution."""
        self._historical_traces.append(trace)
        # Keep only last 1000 traces
        if len(self._historical_traces) > 1000:
            self._historical_traces = self._historical_traces[-1000:]


class ConstraintGuidedAttributor:
    """
    Constraint-Guided Attribution using pre-defined constraints.
    
    Checks if failures were caused by constraint violations.
    """
    
    def __init__(self):
        self._constraints: List[Constraint] = []
    
    def add_constraint(self, constraint: Constraint):
        """Add a constraint to check against."""
        self._constraints.append(constraint)
    
    def attribute(self, trace: FailureTrace, constraints: List[Constraint]) -> List[AgentContribution]:
        """
        Attribute failure using constraint checking.
        
        For each constraint:
        1. Check if violated
        2. Identify which agent violated it
        """
        contributions = []
        
        for constraint in constraints:
            if self._is_violated(constraint, trace):
                # Find which agent likely violated
                violator = self._identify_violator(constraint, trace)
                contributions.append(AgentContribution(
                    agent_id=violator,
                    contribution_score=1.0,  # Certain violation
                    contribution_type='constraint_guided',
                    evidence=[f'Constraint violated: {constraint.name}'],
                    root_cause=True
                ))
        
        return contributions
    
    def _is_violated(self, constraint: Constraint, trace: FailureTrace) -> bool:
        """Check if a constraint is violated in this trace."""
        if constraint.constraint_type == 'temporal':
            # Check temporal ordering
            return self._check_temporal_constraint(constraint, trace)
        elif constraint.constraint_type == 'permission':
            # Check permission constraints
            return self._check_permission_constraint(constraint, trace)
        elif constraint.constraint_type == 'resource':
            # Check resource constraints
            return self._check_resource_constraint(constraint, trace)
        return False
    
    def _check_temporal_constraint(self, constraint: Constraint, trace: FailureTrace) -> bool:
        """Check temporal constraint violation."""
        # Example: operation B must happen after operation A
        # This is a simplified placeholder
        return False
    
    def _check_permission_constraint(self, constraint: Constraint, trace: FailureTrace) -> bool:
        """Check permission constraint violation."""
        # Example: agent X cannot perform operation Y
        return False
    
    def _check_resource_constraint(self, constraint: Constraint, trace: FailureTrace) -> bool:
        """Check resource constraint violation."""
        # Example: total memory usage must not exceed limit
        return False
    
    def _identify_violator(self, constraint: Constraint, trace: FailureTrace) -> str:
        """Identify which agent violated the constraint."""
        # Simplified: return first agent that did something unusual
        for op in trace.operations:
            if not op.get('success', True):
                return op.get('agent', 'unknown')
        return 'unknown'


class CausalInferenceAttributor:
    """
    Causal Inference Attribution using causal graphs.
    
    Builds causal graphs of agent interactions and identifies
    key paths leading to failure.
    """
    
    def __init__(self):
        self._causal_graph: Dict[str, List[CausalEdge]] = defaultdict(list)
    
    def build_causal_graph(self, traces: List[FailureTrace]) -> Dict[str, List[CausalEdge]]:
        """
        Build causal graph from historical traces.
        
        Edges represent causal relationships:
        - 'enables': source enables target
        - 'depends_on': source is required for target
        - 'blocks': source blocks target
        """
        self._causal_graph.clear()
        
        for trace in traces:
            for i, op in enumerate(trace.operations):
                source = op.get('agent', f'op_{i}')
                
                # Create edges to subsequent operations
                if i < len(trace.operations) - 1:
                    next_op = trace.operations[i + 1]
                    target = next_op.get('agent', f'op_{i+1}')
                    
                    # Determine relationship based on outcomes
                    if op.get('success', True) and next_op.get('success', True):
                        relationship = 'enables'
                        weight = 0.8
                    elif not op.get('success', True):
                        relationship = 'blocks'
                        weight = 1.0
                    else:
                        relationship = 'depends_on'
                        weight = 0.5
                    
                    self._causal_graph[source].append(CausalEdge(
                        source=source,
                        target=target,
                        weight=weight,
                        relationship=relationship
                    ))
        
        return self._causal_graph
    
    def attribute(self, trace: FailureTrace) -> List[AgentContribution]:
        """
        Attribute failure using causal inference.
        
        Method:
        1. Identify failed operations
        2. Trace back through causal graph
        3. Calculate causal contribution scores
        """
        contributions = defaultdict(float)
        
        # Find failed operations
        failed_ops = [op for op in trace.operations if not op.get('success', True)]
        
        for failed_op in failed_ops:
            agent_id = failed_op.get('agent', 'unknown')
            
            # Trace back through causal graph
            causal_score = self._trace_causal_path(agent_id, trace)
            contributions[agent_id] += causal_score
        
        # Normalize and convert to contributions
        total = sum(contributions.values())
        result = []
        
        for agent_id, score in contributions.items():
            normalized_score = score / max(total, 1e-9)
            result.append(AgentContribution(
                agent_id=agent_id,
                contribution_score=normalized_score,
                contribution_type='causal_inference',
                evidence=[f'Causal path score: {normalized_score:.3f}'],
                root_cause=(normalized_score == max(c.contribution_score for c in result) if result else False)
            ))
        
        result.sort(key=lambda c: c.contribution_score, reverse=True)
        return result
    
    def _trace_causal_path(self, target_agent: str, trace: FailureTrace) -> float:
        """Trace causal path back from target agent."""
        score = 1.0
        
        # Find the target in operations
        target_idx = -1
        for i, op in enumerate(trace.operations):
            if op.get('agent') == target_agent:
                target_idx = i
                break
        
        if target_idx < 0:
            return 0.0
        
        # Look at predecessors
        for i in range(target_idx - 1, -1, -1):
            prev_op = trace.operations[i]
            prev_agent = prev_op.get('agent', f'op_{i}')
            
            # Check if this agent enabled the failure
            if prev_op.get('success', True):
                score *= 0.8  # Partial contribution
            else:
                score *= 1.0  # Full contribution to failure propagation
        
        return score


class FailureAttributor:
    """
    Unified Failure Attribution combining all three methods.
    
    Reference: LIFE Framework (arXiv:2605.14892)
    """
    
    def __init__(self):
        self.data_driven = DataDrivenAttributor()
        self.constraint_guided = ConstraintGuidedAttributor()
        self.causal_inference = CausalInferenceAttributor()
        self._traces: List[FailureTrace] = []
    
    def add_trace(self, trace: FailureTrace):
        """Add a failure trace for analysis."""
        self._traces.append(trace)
        self.data_driven.add_historical_trace(trace)
        
        # Rebuild causal graph periodically
        if len(self._traces) % 10 == 0:
            self.causal_inference.build_causal_graph(self._traces[-100:])
    
    def attribute_failure(self, trace: FailureTrace,
                         method: AttributionMethod = AttributionMethod.ALL,
                         constraints: List[Constraint] = None) -> List[AgentContribution]:
        """
        Attribute a failure to agents using specified method(s).
        
        Args:
            trace: Failure trace to analyze
            method: Attribution method to use
            constraints: Optional constraints for constraint-guided attribution
            
        Returns:
            List of AgentContributions sorted by contribution score
        """
        all_contributions = []
        
        if method in (AttributionMethod.DATA_DRIVEN, AttributionMethod.ALL):
            all_contributions.extend(self.data_driven.attribute(trace))
        
        if method in (AttributionMethod.CONSTRAINT_GUIDED, AttributionMethod.ALL):
            if constraints:
                all_contributions.extend(self.constraint_guided.attribute(trace, constraints))
        
        if method in (AttributionMethod.CAUSAL_INFERENCE, AttributionMethod.ALL):
            all_contributions.extend(self.causal_inference.attribute(trace))
        
        # Merge contributions for same agent
        merged = self._merge_contributions(all_contributions)
        
        # Sort by contribution score
        merged.sort(key=lambda c: c.contribution_score, reverse=True)
        
        # Mark root cause
        if merged:
            max_score = merged[0].contribution_score
            for c in merged:
                c.root_cause = (c.contribution_score >= max_score * 0.9)
        
        return merged
    
    def _merge_contributions(self, contributions: List[AgentContribution]) -> List[AgentContribution]:
        """Merge contributions for same agent from different methods."""
        merged_dict = {}
        
        for c in contributions:
            if c.agent_id in merged_dict:
                existing = merged_dict[c.agent_id]
                # Weighted average of scores
                existing.contribution_score = (
                    existing.contribution_score * 0.5 + c.contribution_score * 0.5
                )
                existing.evidence.extend(c.evidence)
            else:
                merged_dict[c.agent_id] = c
        
        return list(merged_dict.values())
    
    def get_root_cause(self, trace: FailureTrace,
                       method: AttributionMethod = AttributionMethod.ALL) -> Optional[AgentContribution]:
        """Get the root cause agent for a failure."""
        attributions = self.attribute_failure(trace, method)
        root_causes = [c for c in attributions if c.root_cause]
        
        if root_causes:
            return root_causes[0]
        return None
    
    def explain_failure(self, trace: FailureTrace,
                        method: AttributionMethod = AttributionMethod.ALL) -> str:
        """Generate human-readable explanation of failure attribution."""
        attributions = self.attribute_failure(trace, method)
        
        if not attributions:
            return "No attribution available"
        
        lines = [f"Failure Analysis for trace: {trace.trace_id}"]
        lines.append(f"Error: {trace.error_message}")
        lines.append("")
        lines.append("Agent Contributions:")
        
        for c in attributions:
            root_marker = " [ROOT CAUSE]" if c.root_cause else ""
            lines.append(f"  - {c.agent_id}: {c.contribution_score:.3f}{root_marker}")
            for evidence in c.evidence:
                lines.append(f"    Evidence: {evidence}")
        
        return "\n".join(lines)


class LIFEAttributor:
    """
    LIFE (Lifecycle, Inference, Failure) Attribution wrapper.
    
    This is the main entry point for Layer 2 failure attribution.
    """
    
    def __init__(self):
        self.attributor = FailureAttributor()
        self._statistics = {
            'total_traces': 0,
            'total_attributions': 0,
            'method_usage': defaultdict(int),
        }
    
    def analyze(self, error_trace: Dict, context: Dict = None) -> Dict:
        """
        Analyze a failure and return attribution results.
        
        Args:
            error_trace: Dict with 'trace_id', 'error_message', 'operations'
            context: Optional additional context
            
        Returns:
            Dict with 'root_cause', 'contributions', 'explanation'
        """
        trace = FailureTrace(
            trace_id=error_trace.get('trace_id', 'unknown'),
            error_message=error_trace.get('error_message', 'Unknown error'),
            agents_involved=error_trace.get('agents', []),
            operations=error_trace.get('operations', []),
            context=context or {}
        )
        
        self.attributor.add_trace(trace)
        self._statistics['total_traces'] += 1
        
        # Perform attribution
        attributions = self.attributor.attribute_failure(trace)
        root_cause = self.attributor.get_root_cause(trace)
        
        self._statistics['total_attributions'] += 1
        
        return {
            'trace_id': trace.trace_id,
            'root_cause': root_cause.agent_id if root_cause else None,
            'contributions': [
                {'agent': c.agent_id, 'score': c.contribution_score, 'root_cause': c.root_cause}
                for c in attributions
            ],
            'explanation': self.attributor.explain_failure(trace),
            'statistics': self._statistics
        }
    
    def get_statistics(self) -> Dict:
        """Get attribution statistics."""
        return dict(self._statistics)