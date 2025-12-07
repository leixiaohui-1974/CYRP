"""
Scenario Management System for CYRP.
穿黄工程全场景管理系统

实现全场景的生成、管理和调度，包括：
- 基础场景定义（32种）
- 大规模参数化场景生成（万级）
- 组合场景生成
- 计划和预测信息集成
"""

from cyrp.scenarios.scenario_manager import ScenarioManager
from cyrp.scenarios.scenario_generator import ScenarioGenerator, TestScenario
from cyrp.scenarios.scenario_definitions import (
    Scenario,
    ScenarioType,
    ScenarioDomain,
    ScenarioFamily,
    SCENARIO_REGISTRY
)
from cyrp.scenarios.enhanced_scenario_generator import (
    EnhancedScenarioGenerator,
    SignalGenerator,
    BoundaryCondition,
    ScenarioComplexity,
    TemporalPattern,
)
from cyrp.scenarios.full_scenario_matrix import (
    ScenarioMatrixGenerator,
    ScenarioParameters,
    ScenarioCoverageAnalyzer,
    TemporalScenarioGenerator,
    create_full_scenario_matrix,
    FlowLevel,
    PressureLevel,
    FaultType,
    ExternalEvent,
)
from cyrp.scenarios.massive_scenario_generator import (
    MassiveScenarioGenerator,
    ParameterSpace,
    generate_massive_scenarios,
)
from cyrp.scenarios.planning_forecast_integration import (
    PlanManager,
    ForecastManager,
    PlanningForecastScenarioIntegrator,
    DispatchPlan,
    MaintenancePlan,
    WeatherForecast,
    FloodForecast,
    DemandForecast,
    EquipmentHealthForecast,
    create_integrated_scenario_system,
)

__all__ = [
    # 基础场景
    "ScenarioManager",
    "ScenarioGenerator",
    "TestScenario",
    "Scenario",
    "ScenarioType",
    "ScenarioDomain",
    "ScenarioFamily",
    "SCENARIO_REGISTRY",
    # 增强场景生成
    "EnhancedScenarioGenerator",
    "SignalGenerator",
    "BoundaryCondition",
    "ScenarioComplexity",
    "TemporalPattern",
    # 全场景矩阵
    "ScenarioMatrixGenerator",
    "ScenarioParameters",
    "ScenarioCoverageAnalyzer",
    "TemporalScenarioGenerator",
    "create_full_scenario_matrix",
    "FlowLevel",
    "PressureLevel",
    "FaultType",
    "ExternalEvent",
    # 大规模场景生成
    "MassiveScenarioGenerator",
    "ParameterSpace",
    "generate_massive_scenarios",
    # 计划预测集成
    "PlanManager",
    "ForecastManager",
    "PlanningForecastScenarioIntegrator",
    "DispatchPlan",
    "MaintenancePlan",
    "WeatherForecast",
    "FloodForecast",
    "DemandForecast",
    "EquipmentHealthForecast",
    "create_integrated_scenario_system",
]
