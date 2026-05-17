"""
MSTAR Pro v4.0 Core
Integrating: DDTree + Security + CANTANTE + MARS + EvolveMem + etc.
"""

from __future__ import annotations
from datetime import datetime
from typing import Dict, List, Optional

from mstar_core.evolution.engine import EvolutionEngine, EvolutionConfig, EvolutionEvent
from mstar_core.evolution.fitness_tracker import FitnessTracker, FitnessDimensions, MemoryProgram
from mstar_core.evolution.mutator import MSTARMutator
from mstar_core.evolution.reflector import MSTARReflector
from mstar_core.memory.forgetting import ForgettingMechanism, ForgetCandidate, ForgetDecision
from mstar_core.memory.router import MemoryRouter
from mstar_core.bridge.self_improving import SelfImprovingBridge, CorrectionSignal, ReinforcementSignal
from mstar_core.observability.dashboard import ObservabilityDashboard, get_dashboard
from mstar_core.config.simplified import SimplifiedConfig, get_config
from mstar_core.config.auto_tuner import AutoTuner
from mstar_core.research.ablation import AblationRunner
from mstar_core.acceleration.dd_tree import DDTreeAccelerator, DDTreeConfig, DDTreeIntegration
from mstar_core.security.sanitizer import SecurityLayer
from mstar_core.attribution.failure_attributor import LIFEAttributor
from mstar_core.evolution.robustness import PerturbationRobustFitness, BootstrapCI
from mstar_core.evolution.version_control import VersionControl, FGGMVerifier, MutationVersion, FGGMContract
from mstar_core.evolution.evolvemem import EvolveMemConfigurator, RetrievalConfig, EvolveMemMARSIntegration
from mstar_core.evolution.experience_recall import ExperienceRecall, Trajectory, TrajectoryStep, TrajectoryType
from mstar_core.evolution.protocol import RSPLHandler, SEPLHandler, SkillRegistry, SkillMetadata, RSPLMessage, SEPLMessage, EvolutionProposal

__version__ = "4.0.0"


class MSTARCore:
    def __init__(self, hermes_home, mode='balanced', fitness_dimensions=20,
                 dashboard_enabled=True, dashboard_port=18792):
        self.hermes_home = hermes_home
        self.mode = mode
        self.fitness_dimensions = fitness_dimensions
        self.dashboard_enabled = dashboard_enabled

        self.fitness_tracker = FitnessTracker(hermes_home, mode=self._get_fitness_mode())
        self.evolution_engine = EvolutionEngine(
            self.fitness_tracker,
            MSTARMutator(),
            MSTARReflector(),
            config=EvolutionConfig(adaptive_interval_min=max(3, self.fitness_dimensions // 2)),
            dashboard=get_dashboard() if dashboard_enabled else None,
            self_improver=getattr(self, '_self_improver', None),
        )
        self.forgetting_mechanism = ForgettingMechanism(self.fitness_tracker)
        self.memory_router = MemoryRouter()
        self.memory_router.set_mstar_core(self)
        self._sessions_processed = 0
        self._evolutions_triggered = 0
        
        # Phase 0: DDTree Acceleration Layer
        self.dd_tree_config = DDTreeConfig(
            block_size=16,
            node_budget=256,
            verification_threshold=0.01,
            max_draft_length=128,
        )
        self.dd_tree_integration = DDTreeIntegration(None, self.dd_tree_config)
        self.dd_tree_enabled = False  # Enable once model is set
        
        # Phase 1a: Security Layer
        self.security_layer = SecurityLayer()
        
        # Phase 2: LIFE Failure Attribution
        self.life_attributor = LIFEAttributor()

        # Phase 1c: Perturbation-Robust Fitness
        self.robustness_analyzer = PerturbationRobustFitness(
            n_bootstrap=100,
            confidence_level=0.95,
        )

        # Phase 3: Version Control + FGGM Contracts
        self.version_control = VersionControl(hermes_home)

        # Phase 5: EvolveMem Self-Evolving Retrieval
        self.evolvemem = EvolveMemConfigurator(hermes_home)

        # Phase 6: Experience Recall
        self.experience_recall = ExperienceRecall(hermes_home)

        # Phase 7: RSPL/SEPL Protocol Layer
        self.skill_registry = SkillRegistry()
        self.rspl_handler = RSPLHandler(self.skill_registry)
        self.sepl_handler = SEPLHandler(
            version_control=self.version_control,
            fggm_verifier=self.version_control.fggm_verifier,
        )

    def _get_fitness_mode(self):
        if self.fitness_dimensions <= 10:
            return 'beginner'
        elif self.fitness_dimensions <= 20:
            return 'standard'
        return 'advanced'

    def record_tool_execution(self, tool_name, args, result, evaluation, session_id):
        program_id = f"prog_{tool_name}"
        if not self.fitness_tracker.program_exists(program_id):
            self.fitness_tracker.create_program(program_id, tool_name)
        self.fitness_tracker.update(
            program_id=program_id,
            success=evaluation.get('success', True),
            quality=evaluation.get('quality', 0.8) * 100,
            latency=evaluation.get('latency', 1.0),
            tokens_used=evaluation.get('tokens', 0),
        )

    def should_trigger_evolution(self):
        # P0-5: 直接使用 fitness_tracker 单一来源，不再依赖内存计数
        return self.evolution_engine.should_trigger()

    def run_evolution_cycle(self):
        # P0-5: 使用 fitness_tracker 中的 session index
        session_idx = self.fitness_tracker.get_sessions_processed()
        result = self.evolution_engine.evaluate_session(
            session_id=f"session_{session_idx}", stats={}
        )
        if result.get('triggered'):
            self._evolutions_triggered += 1
        return result

    def trigger_evolution(self):
        """Manually trigger an evolution cycle. Alias for run_evolution_cycle()."""
        return self.run_evolution_cycle()

    def get_all_fitness(self, limit: int = 20):
        """Return all tracked programs ordered by fitness score descending."""
        return self.fitness_tracker.get_all_fitness(limit=limit)

    def on_session_end(self, messages, session_id):
        # P0-5: 使用 fitness_tracker 单一来源
        self.fitness_tracker.increment_sessions()
        if self.should_trigger_evolution():
            self.run_evolution_cycle()

    def get_fitness_aware_context(self, query, session_id):
        high = self.fitness_tracker.get_high_fitness_programs(threshold=0.6, limit=3)
        if not high:
            return ''
        parts = ['[MSTAR Fitness Context]', 'Top Programs:']
        for p in high:
            parts.append(f'- {p.name}: fitness={p.fitness_score:.3f}')
        return '\n'.join(parts)

    def get_statistics(self):
        return {
            # P0-5: 从 DB 单一来源读取 sessions_processed
            'sessions_processed': self.fitness_tracker.get_sessions_processed(),
            'evolutions_triggered': self.fitness_tracker.get_evolutions_completed(),
            'fitness_stats': self.fitness_tracker.get_statistics(),
        }

    def record_batch_session(self, session_id: str, message_count: int,
                             quality_scores: List[float], total_tokens: int,
                             duration: float):
        """Record a session that was compressed via LLM summarization.

        Stores per-session metadata so ForgettingMechanism can evaluate
        information density (actual compressed turns vs raw turn count)
        rather than relying solely on wall-clock age.
        """
        if not hasattr(self, '_compressed_sessions'):
            self._compressed_sessions: Dict[str, Dict] = {}

        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.5
        compression_ratio = len(quality_scores) / message_count if message_count > 0 else 1.0

        self._compressed_sessions[session_id] = {
            'message_count': message_count,
            'compressed_to_turns': len(quality_scores),
            'compression_ratio': compression_ratio,
            'avg_quality': avg_quality,
            'total_tokens': total_tokens,
            'duration': duration,
            'timestamp': datetime.now().isoformat(),
        }
        self._sessions_processed += 1

    def get_session_info(self, session_id: str) -> Optional[Dict]:
        """Return compressed session metadata, or None if not found."""
        return getattr(self, '_compressed_sessions', {}).get(session_id)

    # === Phase 0: DDTree Acceleration ===
    
    def enable_dd_tree(self, model):
        """Enable DDTree acceleration with the given model."""
        self.dd_tree_integration.initialize(model)
        self.dd_tree_integration.enable()
        self.dd_tree_enabled = True
    
    def disable_dd_tree(self):
        """Disable DDTree acceleration."""
        self.dd_tree_integration.disable()
        self.dd_tree_enabled = False
    
    def get_dd_tree_stats(self) -> Dict:
        """Get DDTree acceleration statistics."""
        return self.dd_tree_integration.get_stats()
    
    def wrap_llm_for_dd_tree(self, call_fn):
        """
        Wrap an LLM call function for DDTree acceleration.
        
        Usage:
            wrapped = mstar.wrap_llm_for_dd_tree(original_call)
            result = wrapped(prompt, max_tokens)
        """
        return self.dd_tree_integration.wrap_llm_call(call_fn)
    
    # === Phase 1a: Security Layer ===
    
    def sanitize_memory_write(self, content: str, source: str):
        """Sanitize a memory write to prevent injection attacks."""
        return self.security_layer.sanitize_memory_write(content, source)
    
    def validate_input(self, content: str, content_type: str = 'generic'):
        """Validate external input."""
        return self.security_layer.validate_input(content, content_type)
    
    def check_privilege(self, session_id: str, operation: str):
        """Check if session has privilege for operation."""
        return self.security_layer.check_privilege(session_id, operation)
    
    def get_security_status(self) -> Dict:
        """Get current security status."""
        return self.security_layer.get_security_status()
    
    # === Phase 2: LIFE Failure Attribution ===
    
    def analyze_failure(self, error_trace: Dict, context: Dict = None) -> Dict:
        """
        Analyze a failure and return attribution results.
        
        Args:
            error_trace: Dict with 'trace_id', 'error_message', 'operations'
            context: Optional additional context
            
        Returns:
            Dict with 'root_cause', 'contributions', 'explanation'
        """
        return self.life_attributor.analyze(error_trace, context)
    
    def get_attribution_stats(self) -> Dict:
        """Get failure attribution statistics."""
        return self.life_attributor.get_statistics()

    # === Phase 1c: Perturbation-Robust Fitness ===

    def compute_fitness_confidence_interval(self, program_id: str) -> BootstrapCI:
        """
        Compute Bootstrap CI for a program's fitness score.

        Args:
            program_id: Program ID

        Returns:
            BootstrapCI with mean, lower, upper, width
        """
        from mstar_core.evolution.fitness_tracker import FitnessTracker

        # Get program's fitness history
        ft = self.fitness_tracker
        if hasattr(ft, '_get_or_create_program'):
            program = ft._get_or_create_program(program_id)
            history = getattr(program, 'fitness_history', [])
        else:
            history = []

        return self.robustness_analyzer.compute_bootstrap_ci(history)

    def run_perturbation_test(self, program_id: str, noise_level: float = 0.1) -> Dict:
        """
        Run perturbation test on a program.

        Args:
            program_id: Program ID
            noise_level: Noise level (0-1)

        Returns:
            Dict with ci, perturbation_resistance, etc.
        """
        ft = self.fitness_tracker
        if hasattr(ft, '_get_or_create_program'):
            program = ft._get_or_create_program(program_id)
            baseline = getattr(program, 'fitness_score', 0.5)
            history = getattr(program, 'fitness_history', [])
        else:
            baseline = 0.5
            history = []

        return self.robustness_analyzer.run_perturbation_test(
            program_id=program_id,
            baseline_fitness=baseline,
            fitness_history=history,
            noise_level=noise_level,
        )

    def get_robustness_stats(self) -> Dict:
        """Get perturbation robustness statistics."""
        return self.robustness_analyzer.get_statistics()

    # === Phase 3: Version Control ===

    def create_version(self, program_id: str, mutation_type: str, mutation_details: Dict) -> MutationVersion:
        """
        Create a new mutation version with FGGM verification.

        Args:
            program_id: Program ID
            mutation_type: Type of mutation
            mutation_details: Mutation details dict

        Returns:
            MutationVersion object
        """
        return self.version_control.create_version(
            program_id=program_id,
            mutation_type=mutation_type,
            mutation_details=mutation_details,
        )

    def rollback_version(self, program_id: str, version_id: str, reason: str = "") -> Optional[MutationVersion]:
        """Rollback program to a specific version."""
        return self.version_control.rollback_to(program_id, version_id, reason)

    def get_program_versions(self, program_id: str, limit: int = 50) -> List[MutationVersion]:
        """Get version history for a program."""
        return self.version_control.get_program_versions(program_id, limit)

    def get_version_control_stats(self) -> Dict:
        """Get version control statistics."""
        return self.version_control.get_statistics()

    # === Phase 5: EvolveMem ===

    def get_retrieval_config(self) -> RetrievalConfig:
        """Get current active retrieval configuration."""
        return self.evolvemem.get_active_config()

    def select_retrieval_config(self) -> RetrievalConfig:
        """Select a retrieval config using epsilon-greedy strategy."""
        return self.evolvemem.select_config()

    def record_retrieval(self, query: str, config_id: str, retrieved_items: List[str],
                         retrieval_time_ms: float, task_success: Optional[bool] = None):
        """Record a retrieval event for EvolveMem analysis."""
        self.evolvemem.record_retrieval(query, config_id, retrieved_items, retrieval_time_ms, task_success)

    def get_evolvemem_stats(self) -> Dict:
        """Get EvolveMem statistics."""
        return self.evolvemem.get_statistics()

    # === Phase 6: Experience Recall ===

    def recall_similar_experience(self, current_state: Dict, task_type: Optional[str] = None) -> List:
        """
        Recall similar experiences from history.

        Args:
            current_state: Current state dict
            task_type: Optional task type filter

        Returns:
            List of (Trajectory, similarity_score) tuples
        """
        return self.experience_recall.recall_similar_experience(current_state, task_type)

    def record_trajectory(self, program_id: str, task_type: str, trajectory_type: TrajectoryType,
                         steps: List[TrajectoryStep], start_time: str, end_time: str,
                         overall_success: bool, final_fitness: float,
                         task_context: Optional[Dict] = None) -> Trajectory:
        """Record a new execution trajectory."""
        return self.experience_recall.record_trajectory(
            program_id=program_id, task_type=task_type,
            trajectory_type=trajectory_type, steps=steps,
            start_time=start_time, end_time=end_time,
            overall_success=overall_success, final_fitness=final_fitness,
            task_context=task_context,
        )

    def get_reasoning_from_history(self, current_state: Dict, task_type: Optional[str] = None) -> str:
        """Get formatted reasoning chain from similar historical experiences."""
        return self.experience_recall.get_reasoning_from_similar(current_state, task_type)

    def get_experience_recall_stats(self) -> Dict:
        """Get experience recall statistics."""
        return self.experience_recall.get_statistics()

    # === Phase 7: RSPL/SEPL Protocol ===

    def register_skill(self, skill_metadata: SkillMetadata) -> RSPLMessage:
        """Register a new skill via RSPL."""
        return self.rspl_handler.handle_register(skill_metadata)

    def list_skills(self, category: Optional[str] = None) -> RSPLMessage:
        """List skills via RSPL."""
        return self.rspl_handler.handle_list(category)

    def propose_evolution(self, program_id: str, mutation_type: str, mutation_details: Dict) -> SEPLMessage:
        """Propose an evolution via SEPL."""
        from mstar_core.evolution.protocol import EvolutionProposal
        import hashlib

        proposal_id = f"prop_{hashlib.md5(f'{program_id}{mutation_type}'.encode()).hexdigest()[:12]}"
        proposal = EvolutionProposal(
            proposal_id=proposal_id,
            program_id=program_id,
            mutation_type=mutation_type,
            mutation_details=mutation_details,
            proposed_at=datetime.now().isoformat(),
        )
        return self.sepl_handler.handle_propose(proposal)

    def verify_evolution(self, proposal_id: str) -> SEPLMessage:
        """Verify an evolution proposal via SEPL."""
        return self.sepl_handler.handle_verify(proposal_id)

    def apply_evolution(self, proposal_id: str) -> SEPLMessage:
        """Apply an evolution proposal via SEPL."""
        return self.sepl_handler.handle_apply(proposal_id)

    def rollback_evolution(self, proposal_id: str, reason: str = "") -> SEPLMessage:
        """Rollback an evolution via SEPL."""
        return self.sepl_handler.handle_rollback(proposal_id, reason)

    def get_protocol_stats(self) -> Dict:
        """Get RSPL/SEPL protocol statistics."""
        return {
            'skill_registry_size': len(self.skill_registry._skills),
            'active_proposals': len([p for p in self.sepl_handler._proposals.values() if p.status in ('proposed', 'verified', 'applied')]),
            'version_control': self.version_control.get_statistics(),
        }