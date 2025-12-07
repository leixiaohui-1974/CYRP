"""
Hardware-in-the-Loop Testing Framework for CYRP.
穿黄工程在环测试框架

实现全场景自主运行的在环测试与验证
"""

from cyrp.hil.hil_framework import HILTestFramework
from cyrp.hil.test_runner import TestRunner, TestCase, TestResult
from cyrp.hil.virtual_plc import VirtualPLC
from cyrp.hil.simulation_engine import SimulationEngine
from cyrp.hil.sensor_models import (
    SensorModel,
    PressureSensorModel,
    FlowMeterModel,
    TemperatureSensorModel,
    DASensorModel,
    DTSensorModel,
    MEMSSensorModel,
    SensorArray,
    InstrumentationSystem,
    SensorStatus,
    FailureMode,
)
from cyrp.hil.actuator_models import (
    ActuatorModel,
    GateValveModel,
    ButterflyValveModel,
    PumpModel,
    MotorModel,
    ActuatorSystem,
    ActuatorStatus,
    ActuatorFailureMode,
)
from cyrp.hil.full_hil_system import (
    FullHILSystem,
    ClosedLoopController,
    HILMode,
    HILTestCase,
    HILTestResult,
    PerformanceMetrics,
)

__all__ = [
    "HILTestFramework",
    "TestRunner",
    "TestCase",
    "TestResult",
    "VirtualPLC",
    "SimulationEngine",
    "SensorModel",
    "PressureSensorModel",
    "FlowMeterModel",
    "TemperatureSensorModel",
    "DASensorModel",
    "DTSensorModel",
    "MEMSSensorModel",
    "SensorArray",
    "InstrumentationSystem",
    "SensorStatus",
    "FailureMode",
    "ActuatorModel",
    "GateValveModel",
    "ButterflyValveModel",
    "PumpModel",
    "MotorModel",
    "ActuatorSystem",
    "ActuatorStatus",
    "ActuatorFailureMode",
    "FullHILSystem",
    "ClosedLoopController",
    "HILMode",
    "HILTestCase",
    "HILTestResult",
    "PerformanceMetrics",
]
