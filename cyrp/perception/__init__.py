"""
Multi-Modal Perception System for CYRP.
穿黄工程多模态感知系统

包含:
- DAS (分布式光纤声波传感)
- DTS (分布式光纤测温)
- MEMS (微机电系统传感器阵列)
- CV (计算机视觉)
- 数据融合引擎
- 场景分类器
"""

from cyrp.perception.sensors import (
    DASensor,
    DTSensor,
    MEMSSensor,
    PressureSensor,
    FlowMeter,
    WaterQualitySensor
)
from cyrp.perception.fusion import DataFusionEngine
from cyrp.perception.classifier import ScenarioClassifier
from cyrp.perception.perception_system import PerceptionSystem
from cyrp.perception.advanced_classifier import (
    AdvancedScenarioClassifier,
    FeatureExtractor,
    PatternRecognizer,
    AnomalyDetector,
    BayesianClassifier,
    IsolationForest,
)

__all__ = [
    "DASensor",
    "DTSensor",
    "MEMSSensor",
    "PressureSensor",
    "FlowMeter",
    "WaterQualitySensor",
    "DataFusionEngine",
    "ScenarioClassifier",
    "PerceptionSystem",
    "AdvancedScenarioClassifier",
    "FeatureExtractor",
    "PatternRecognizer",
    "AnomalyDetector",
    "BayesianClassifier",
    "IsolationForest",
]
