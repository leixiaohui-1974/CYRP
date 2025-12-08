"""
仿真模块 - Simulation Module

包含传感器仿真和执行器仿真的完整实现
Contains complete implementation of sensor and actuator simulation
"""

from .sensor_simulation import (
    SensorSimulationManager,
    VirtualSensor,
    VirtualSensorNetwork,
    SensorDataGenerator,
    SensorCharacteristics,
    NoiseModel,
    NoiseType,
    DriftModel,
    DriftType,
    FailureInjector,
)

from .actuator_simulation import (
    ActuatorSimulationManager,
    VirtualValve,
    VirtualPump,
    VirtualMotor,
    VirtualActuatorNetwork,
    ActuatorDynamicsEngine,
    FailureSimulator,
)

__all__ = [
    # Sensor simulation
    'SensorSimulationManager',
    'VirtualSensor',
    'VirtualSensorNetwork',
    'SensorDataGenerator',
    'SensorCharacteristics',
    'NoiseModel',
    'NoiseType',
    'DriftModel',
    'DriftType',
    'FailureInjector',
    # Actuator simulation
    'ActuatorSimulationManager',
    'VirtualValve',
    'VirtualPump',
    'VirtualMotor',
    'VirtualActuatorNetwork',
    'ActuatorDynamicsEngine',
    'FailureSimulator',
]
