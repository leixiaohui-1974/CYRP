"""
Hardware-in-the-Loop Testing Framework for CYRP.
穿黄工程在环测试框架

实现全场景自主运行的在环测试与验证
"""

from cyrp.hil.hil_framework import HILTestFramework
from cyrp.hil.test_runner import TestRunner, TestCase, TestResult
from cyrp.hil.virtual_plc import VirtualPLC
from cyrp.hil.simulation_engine import SimulationEngine

__all__ = [
    "HILTestFramework",
    "TestRunner",
    "TestCase",
    "TestResult",
    "VirtualPLC",
    "SimulationEngine",
]
