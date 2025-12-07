"""
仿真模块 - Simulation Module

包含传感器仿真和执行器仿真的完整实现
Contains complete implementation of sensor and actuator simulation
"""

from .sensor_simulation import (
    SensorSimulationManager,
    VirtualSensorNetwork,
    SensorDataGenerator,
    NoiseModel,
    DriftModel,
    FailureInjector,
)

from .actuator_simulation import (
    ActuatorSimulationManager,
    VirtualActuatorNetwork,
    ActuatorDynamicsEngine,
    FailureSimulator,
)

__all__ = [
    # Sensor simulation
    'SensorSimulationManager',
    'VirtualSensorNetwork',
    'SensorDataGenerator',
    'NoiseModel',
    'DriftModel',
    'FailureInjector',
    # Actuator simulation
    'ActuatorSimulationManager',
    'VirtualActuatorNetwork',
    'ActuatorDynamicsEngine',
    'FailureSimulator',
]
