"""
MSTAR Pro v4.0 - 简化配置系统
"""

from __future__ import annotations
import os
import yaml
from typing import Dict, Optional, Any


class SimplifiedConfig:
    """
    MSTAR Pro v4.0 简化配置
    5种模式：beginner/standard/balanced/production/research
    """

    MODES = ['beginner', 'standard', 'balanced', 'production', 'research']

    MODE_CONFIGS = {
        'beginner': {'fitness_dimensions': 10, 'evolution_interval': 20, 'dashboard_enabled': True, 'self_improving_bridge': False, 'observability_level': 'basic'},
        'standard': {'fitness_dimensions': 20, 'evolution_interval': 15, 'dashboard_enabled': True, 'self_improving_bridge': True, 'observability_level': 'standard'},
        'balanced': {'fitness_dimensions': 20, 'evolution_interval': 10, 'dashboard_enabled': True, 'self_improving_bridge': True, 'observability_level': 'standard'},
        'production': {'fitness_dimensions': 55, 'evolution_interval': 5, 'dashboard_enabled': True, 'self_improving_bridge': True, 'observability_level': 'full'},
        'research': {'fitness_dimensions': 55, 'evolution_interval': 3, 'dashboard_enabled': True, 'self_improving_bridge': True, 'observability_level': 'full', 'ablation_enabled': True},
    }

    def __init__(self, mode: str = 'balanced', config_path: Optional[str] = None):
        self.mode = mode
        self.config_path = config_path
        self._config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        base_config = self.MODE_CONFIGS.get(self.mode, self.MODE_CONFIGS['balanced'])
        if self.config_path and os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                yaml_config = yaml.safe_load(f) or {}
            base_config.update(yaml_config.get('mstar', {}))
        return base_config

    def get(self, key: str, default: Any = None) -> Any:
        return self._config.get(key, default)

    def set(self, key: str, value: Any):
        self._config[key] = value

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._config)

    def apply_overrides(self, overrides: Dict[str, Any]):
        self._config.update(overrides)


_global_config = None


def get_config(mode: str = 'balanced', config_path: Optional[str] = None) -> SimplifiedConfig:
    global _global_config
    if _global_config is None:
        _global_config = SimplifiedConfig(mode, config_path)
    return _global_config