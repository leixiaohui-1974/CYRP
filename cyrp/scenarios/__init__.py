"""
Scenario Management System for CYRP.
穿黄工程全场景管理系统

实现32种细分场景的生成、管理和调度
"""

from cyrp.scenarios.scenario_manager import ScenarioManager
from cyrp.scenarios.scenario_generator import ScenarioGenerator
from cyrp.scenarios.scenario_definitions import (
    Scenario,
    ScenarioType,
    ScenarioDomain,
    ScenarioFamily,
    SCENARIO_REGISTRY
)

__all__ = [
    "ScenarioManager",
    "ScenarioGenerator",
    "Scenario",
    "ScenarioType",
    "ScenarioDomain",
    "ScenarioFamily",
    "SCENARIO_REGISTRY",
]
