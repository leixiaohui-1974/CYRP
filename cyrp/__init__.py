"""
CYRP - Crossing Yellow River Project
南水北调中线穿黄工程全场景自主运行在环测试与多智能体系统平台

版本: 1.1.0
"""

__version__ = "1.1.0"
__author__ = "CYRP Development Team"

# 核心模块
from cyrp.core import PhysicalSystem, TunnelParameters, HydraulicState
from cyrp.scenarios import ScenarioManager, Scenario, ScenarioType
from cyrp.perception import PerceptionSystem
from cyrp.control import HDMPCController
from cyrp.agents import MultiAgentSystem
from cyrp.hil import HILTestFramework
from cyrp.digital_twin import DigitalTwin

# 新增模块 - 传感器/执行器仿真
from cyrp.simulation import (
    SensorSimulationManager,
    ActuatorSimulationManager,
    VirtualSensor,
    VirtualSensorNetwork,
    VirtualValve,
    VirtualPump,
    VirtualActuatorNetwork,
)

# 新增模块 - 数据治理
from cyrp.governance import (
    DataGovernanceManager,
    DataQualityEngine,
    DataLineageTracker,
    DataValidator,
)

# 新增模块 - 数据同化
from cyrp.assimilation import (
    DataAssimilationManager,
    KalmanFilter,
    ExtendedKalmanFilter,
    UnscentedKalmanFilter,
    EnsembleKalmanFilter,
    ParticleFilter,
)

# 新增模块 - IDZ参数更新
from cyrp.idz import (
    IDZParameterUpdater,
    RecursiveLeastSquares,
    SystemIdentification,
)

# 新增模块 - 状态评价
from cyrp.evaluation import (
    StateEvaluator,
    ControlPerformanceEvaluator,
    SafetyEvaluator,
    DeviationAnalyzer,
)

# 新增模块 - 状态预测
from cyrp.prediction import (
    StatePredictorManager,
    ARIMAPredictor,
    ExponentialSmoothingPredictor,
    PhysicsBasedPredictor,
    EnsemblePredictor,
)

__all__ = [
    # 核心模块
    "PhysicalSystem",
    "TunnelParameters",
    "HydraulicState",
    "ScenarioManager",
    "Scenario",
    "ScenarioType",
    "PerceptionSystem",
    "HDMPCController",
    "MultiAgentSystem",
    "HILTestFramework",
    "DigitalTwin",
    # 传感器/执行器仿真
    "SensorSimulationManager",
    "ActuatorSimulationManager",
    "VirtualSensor",
    "VirtualSensorNetwork",
    "VirtualValve",
    "VirtualPump",
    "VirtualActuatorNetwork",
    # 数据治理
    "DataGovernanceManager",
    "DataQualityEngine",
    "DataLineageTracker",
    "DataValidator",
    # 数据同化
    "DataAssimilationManager",
    "KalmanFilter",
    "ExtendedKalmanFilter",
    "UnscentedKalmanFilter",
    "EnsembleKalmanFilter",
    "ParticleFilter",
    # IDZ参数更新
    "IDZParameterUpdater",
    "RecursiveLeastSquares",
    "SystemIdentification",
    # 状态评价
    "StateEvaluator",
    "ControlPerformanceEvaluator",
    "SafetyEvaluator",
    "DeviationAnalyzer",
    # 状态预测
    "StatePredictorManager",
    "ARIMAPredictor",
    "ExponentialSmoothingPredictor",
    "PhysicsBasedPredictor",
    "EnsemblePredictor",
]
