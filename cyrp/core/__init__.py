"""
Core physical models and system definitions for CYRP.
核心物理模型与系统定义
"""

from cyrp.core.physical_system import PhysicalSystem
from cyrp.core.parameters import TunnelParameters, EnvironmentParameters
from cyrp.core.hydraulic_model import HydraulicModel, HydraulicState
from cyrp.core.structural_model import StructuralModel, StructuralState
from cyrp.core.high_fidelity_model import (
    CoupledPhysicalModel,
    HighFidelityHydraulicModel,
    ThermalModel,
    StructuralModel as HighFidelityStructuralModel,
    TunnelGeometry,
    MaterialProperties,
    SimulationState,
)

__all__ = [
    "PhysicalSystem",
    "TunnelParameters",
    "EnvironmentParameters",
    "HydraulicModel",
    "HydraulicState",
    "StructuralModel",
    "StructuralState",
    "CoupledPhysicalModel",
    "HighFidelityHydraulicModel",
    "ThermalModel",
    "HighFidelityStructuralModel",
    "TunnelGeometry",
    "MaterialProperties",
    "SimulationState",
]
