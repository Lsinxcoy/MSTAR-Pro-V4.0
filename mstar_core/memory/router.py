"""
MSTAR Pro v4.0 - Memory Router
5种内存类型路由
"""

from __future__ import annotations
from typing import Dict, Optional


class MemoryRouter:
    """
    MSTAR Pro v4.0 5路内存类型路由

    VECTOR -> SimpleVectorStore
    TEMPORAL -> HybridSeed
    SEMANTIC -> RelationalStore
    STRUCTURED -> SQLKB
    PROCEDURAL -> SkillSystem
    MSTAR_FITNESS -> FitnessAware
    """

    MEMORY_TYPES = {
        'VECTOR': 'SimpleVectorStore',
        'TEMPORAL': 'HybridSeed',
        'SEMANTIC': 'RelationalStore',
        'STRUCTURED': 'SQLKB',
        'PROCEDURAL': 'SkillSystem',
        'MSTAR_FITNESS': 'FitnessAware',
    }

    def __init__(self):
        self._handlers = {}
        self._mstar_core = None

    def register_handler(self, memory_type: str, handler):
        self._handlers[memory_type] = handler

    def set_mstar_core(self, mstar_core):
        self._mstar_core = mstar_core

    def route(self, query: str, memory_type: Optional[str] = None) -> str:
        if memory_type == 'MSTAR_FITNESS' and self._mstar_core:
            return self._mstar_core.get_fitness_aware_context(query, session_id='')

        handler_name = self.MEMORY_TYPES.get(memory_type, 'DefaultHandler')
        handler = self._handlers.get(handler_name)

        if handler:
            return handler(query)

        return ""