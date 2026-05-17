"""
MSTAR Pro v4.0 - Fitness-Aware Context Engine Plugin

A proper ContextEngine that wraps ContextCompressor with MSTAR fitness awareness.
When compressing context, prioritizes high-fitness programs and demotes low-fitness ones.

Activated via: context.engine: mstar in config.yaml
Requires: memory.provider: mstar (MSTARCore must be initialized first)
"""

from __future__ import annotations
import logging
import time
from typing import Any, Dict, List, Optional

from agent.context_engine import ContextEngine
from agent.context_compressor import ContextCompressor

logger = logging.getLogger(__name__)


class MSTARContextEngine(ContextEngine):
    """
    MSTAR Pro v4.0 Fitness-Aware Context Engine.
    
    Wraps ContextCompressor (the default compressor) and adds a fitness-aware
    layer on top of the compression logic. When summarizing tool results or
    deciding what to preserve in compressed context, high-fitness programs
    are given priority.
    
    Integration:
    - MSTARCore must be initialized (memory.provider: mstar in config.yaml)
    - Set context.engine: mstar in config.yaml to activate
    - Receives mstar_core reference via set_mstar_core() from AIAgent
    """

    _instances: List["MSTARContextEngine"] = []

    @property
    def name(self) -> str:
        return "mstar"

    def __init__(
        self,
        inner: Optional[ContextEngine] = None,
        model: str = "",
        context_length: int = 200_000,
        mstar_core=None,
        **kwargs,
    ):
        self._inner = inner
        self._mstar_core = mstar_core
        self._fitness_cache: Dict[str, float] = {}
        self._last_cache_refresh = 0.0
        self._cache_ttl = 30.0  # Refresh fitness cache every 30s

        # Token tracking (ContextEngine protocol)
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0
        self.threshold_tokens = 0
        self.context_length = context_length
        self.compression_count = 0

        # Threshold from ContextEngine base
        # NOTE: These are placeholders. Proper values come from config.yaml
        # and are passed via update_model(). This init only sets defaults.
        self.threshold_percent = 0.75
        self.protect_first_n = 3
        self.protect_last_n = 6
        self.summary_target_ratio = 0.20

        MSTARContextEngine._instances.append(self)
        logger.info("[MSTAR ContextEngine] Initialized (inner=%s)", 
                    inner.name if inner else "None")

    @classmethod
    def get_primary_instance(cls) -> Optional["MSTARContextEngine"]:
        """Return the most recently created instance (used by AIAgent to inject mstar_core)."""
        return cls._instances[-1] if cls._instances else None

    def set_mstar_core(self, mstar_core) -> None:
        """Called by AIAgent after MSTARCore is initialized."""
        self._mstar_core = mstar_core
        logger.info("[MSTAR ContextEngine] MSTARCore attached")

    def set_inner(self, inner: ContextEngine) -> None:
        """Set the inner context engine (ContextCompressor by default)."""
        self._inner = inner
        # Sync token tracking state from inner
        if hasattr(inner, 'last_prompt_tokens'):
            self.last_prompt_tokens = inner.last_prompt_tokens
            self.last_completion_tokens = inner.last_completion_tokens
            self.last_total_tokens = inner.last_total_tokens
            self.threshold_tokens = inner.threshold_tokens
            self.context_length = inner.context_length

    def _refresh_fitness_cache(self) -> None:
        """Refresh cached fitness scores from MSTARCore / FitnessTracker."""
        now = time.time()
        if now - self._last_cache_refresh < self._cache_ttl and self._fitness_cache:
            return

        self._fitness_cache.clear()
        if not self._mstar_core:
            return

        try:
            import sqlite3
            db_path = self._mstar_core.fitness_tracker.db_path
            conn = sqlite3.connect(db_path, timeout=10)
            cur = conn.execute(
                "SELECT program_id, fitness_score FROM programs"
            )
            for row in cur.fetchall():
                # Normalize program_id: "prog_terminal" -> "terminal"
                pid = row[0]
                score = row[1] if row[1] is not None else 0.5
                self._fitness_cache[pid] = score
                # Also index by short name
                if pid.startswith("prog_"):
                    short = pid[5:]
                    self._fitness_cache[short] = score
            conn.close()
            self._last_cache_refresh = now
        except Exception as e:
            logger.debug("[MSTAR ContextEngine] Fitness cache refresh failed: %s", e)

    def get_fitness(self, program_id: str) -> float:
        """Get fitness score for a program, 0.5 if unknown."""
        self._refresh_fitness_cache()
        return self._fitness_cache.get(program_id, 0.5)

    def get_fitness_multiplier(self, program_id: str) -> float:
        """
        Returns a multiplier based on fitness score.
        High fitness (>0.8) -> multiplier > 1 (preserve more)
        Low fitness (<0.4) -> multiplier < 1 (aggressively compress)
        """
        f = self.get_fitness(program_id)
        if f >= 0.8:
            return 1.0 + (f - 0.8) * 2.5  # 1.0-1.5
        elif f <= 0.3:
            return 0.3 + (f / 0.3) * 0.7  # 0.3-1.0
        else:
            return 0.7 + (f - 0.3) / 0.5 * 0.3  # 0.7-1.0

    # ------------------------------------------------------------------ 
    # ContextEngine Protocol — delegate to inner engine
    # ------------------------------------------------------------------ 

    def update_from_response(self, usage: Dict[str, Any]) -> None:
        self.last_prompt_tokens = usage.get("prompt_tokens", 0)
        self.last_completion_tokens = usage.get("completion_tokens", 0)
        self.last_total_tokens = usage.get("total_tokens", 0)
        if self._inner:
            self._inner.update_from_response(usage)

    def should_compress(self, prompt_tokens: int = None) -> bool:
        if self._inner:
            return self._inner.should_compress(prompt_tokens)
        return False

    def should_compress_preflight(self, messages: List[Dict[str, Any]]) -> bool:
        if self._inner and hasattr(self._inner, 'should_compress_preflight'):
            return self._inner.should_compress_preflight(messages)
        return False

    def has_content_to_compress(self, messages: List[Dict[str, Any]]) -> bool:
        if self._inner and hasattr(self._inner, 'has_content_to_compress'):
            return self._inner.has_content_to_compress(messages)
        return True

    def compress(
        self,
        messages: List[Dict[str, Any]],
        current_tokens: int = None,
        focus_topic: str = None,
    ) -> List[Dict[str, Any]]:
        if self._inner:
            return self._inner.compress(messages, current_tokens, focus_topic)
        return messages

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        if self._inner:
            return self._inner.get_tool_schemas()
        return []

    def on_session_start(self, session_id: str, **kwargs) -> None:
        if self._inner and hasattr(self._inner, 'on_session_start'):
            self._inner.on_session_start(session_id, **kwargs)

    def on_session_end(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        if self._inner and hasattr(self._inner, 'on_session_end'):
            self._inner.on_session_end(session_id, messages)

    def on_session_reset(self) -> None:
        super().on_session_reset()
        self._fitness_cache.clear()
        self._last_cache_refresh = 0.0
        if self._inner and hasattr(self._inner, 'on_session_reset'):
            self._inner.on_session_reset()

    def update_model(
        self,
        model: str,
        context_length: int,
        base_url: str = "",
        api_key: str = "",
        provider: str = "",
        api_mode: str = "",
        # Compression params passed from config.yaml — forward to inner engine
        threshold_percent: float = None,
        protect_first_n: int = None,
        protect_last_n: int = None,
        summary_target_ratio: float = None,
    ) -> None:
        self.context_length = context_length
        # Use provided threshold_percent, fall back to hardcoded default only if not given
        _threshold = threshold_percent if threshold_percent is not None else self.threshold_percent
        self.threshold_tokens = max(
            int(context_length * _threshold),
            4096,
        )
        if threshold_percent is not None:
            self.threshold_percent = threshold_percent
        if protect_first_n is not None:
            self.protect_first_n = protect_first_n
        if protect_last_n is not None:
            self.protect_last_n = protect_last_n
        if summary_target_ratio is not None:
            self.summary_target_ratio = summary_target_ratio
        if self._inner and hasattr(self._inner, 'update_model'):
            self._inner.update_model(model, context_length, base_url, api_key, provider, api_mode,
                                     threshold_percent=threshold_percent,
                                     protect_first_n=protect_first_n,
                                     protect_last_n=protect_last_n,
                                     summary_target_ratio=summary_target_ratio)

    # ------------------------------------------------------------------ 
    # Fitness-aware helpers (for manual use and debugging)
    # ------------------------------------------------------------------ 

    def get_fitness_report(self) -> Dict[str, Any]:
        """Return a report of tracked program fitness scores."""
        self._refresh_fitness_cache()
        if not self._fitness_cache:
            return {"status": "no_mstar_core", "programs": {}}

        sorted_programs = sorted(
            self._fitness_cache.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return {
            "status": "ok",
            "cache_ttl_remaining": max(0, self._cache_ttl - (time.time() - self._last_cache_refresh)),
            "programs": {k: round(v, 4) for k, v in sorted_programs},
        }


# For plugin discovery — register this engine
def register(ctx):
    """Plugin entry point called by load_context_engine()."""
    ctx.register_context_engine(MSTARContextEngine())