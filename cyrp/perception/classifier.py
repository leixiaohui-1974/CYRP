"""
Scenario Classifier for CYRP Perception System.
穿黄工程场景分类器

采用分层状态机 + 随机森林的混合架构
实现32种细分工况的识别
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
from enum import Enum, auto
import numpy as np
from collections import deque


class ScenarioDomain(Enum):
    """场景域"""
    NOMINAL = "nominal"  # 常态运行域
    TRANSITION = "transition"  # 过渡运维域
    EMERGENCY = "emergency"  # 应急灾害域


class ScenarioFamily(Enum):
    """场景族"""
    # 常态域
    S1_BALANCED = "S1_balanced"  # 平衡输水
    S2_MAINTENANCE = "S2_maintenance"  # 维护性输水

    # 过渡域
    S3_TWO_PHASE = "S3_two_phase"  # 气液两相转换
    S4_TOPOLOGY = "S4_topology"  # 拓扑切换

    # 应急域
    S5_STRUCTURAL = "S5_structural"  # 结构失效
    S6_GEOLOGICAL = "S6_geological"  # 极端地质
    S7_SYSTEM = "S7_system"  # 系统性故障


class ScenarioType(Enum):
    """具体场景类型 (32种)"""
    # S1 平衡输水场景族
    S1_A_DUAL_BALANCED = "S1-A"  # 双洞均分模式
    S1_B_DYNAMIC_PEAK = "S1-B"  # 动态调峰模式

    # S2 维护性输水场景族
    S2_A_SEDIMENT_FLUSH = "S2-A"  # 单洞排沙冲淤
    S2_B_MUSSEL_CONTROL = "S2-B"  # 贝类消杀运行

    # S3 气液两相转换场景族
    S3_A_FILLING = "S3-A"  # 充水排气
    S3_B_DRAINING = "S3-B"  # 停水排空

    # S4 拓扑切换场景族
    S4_A_SWITCH_TUNNEL = "S4-A"  # 不停水倒洞
    S4_B_ISOLATION = "S4-B"  # 检修隔离

    # S5 结构失效场景族
    S5_A_INNER_LEAK = "S5-A"  # 内衬渗漏
    S5_B_OUTER_INTRUSION = "S5-B"  # 外衬入侵
    S5_C_JOINT_OFFSET = "S5-C"  # 接头错位

    # S6 极端地质场景族
    S6_A_LIQUEFACTION = "S6-A"  # 地震液化上浮
    S6_B_INTAKE_VORTEX = "S6-B"  # 进口吸气漩涡

    # S7 系统性故障场景族
    S7_A_PIPE_BURST = "S7-A"  # 爆管/断流
    S7_B_GATE_ASYNC = "S7-B"  # 闸门非同步故障

    # 默认
    UNKNOWN = "UNKNOWN"


@dataclass
class ScenarioClassification:
    """场景分类结果"""
    timestamp: float = 0.0
    domain: ScenarioDomain = ScenarioDomain.NOMINAL
    family: ScenarioFamily = ScenarioFamily.S1_BALANCED
    scenario_type: ScenarioType = ScenarioType.S1_A_DUAL_BALANCED
    confidence: float = 0.95
    evidence: Dict[str, float] = field(default_factory=dict)
    transition_from: Optional[ScenarioType] = None
    duration_in_scenario: float = 0.0


class RuleBasedClassifier:
    """
    基于规则的场景分类器

    实现报告中的判据逻辑
    """

    def __init__(self):
        # 阈值参数
        self.thresholds = {
            'flow_imbalance': 0.01,  # 流量不平衡阈值
            'cavity_level_rate': 0.02,  # 空腔水位上升速率 (m/h)
            'temp_diff_inner': 2.0,  # 内衬渗漏温差
            'temp_diff_outer': 2.0,  # 外衬入侵温差
            'vibration': 0.1,  # 振动阈值 (g)
            'tilt': 0.005,  # 倾斜阈值 (rad)
            'pressure_rate': 0.5,  # 压力变化率 (m/min)
            'velocity_sediment': 3.0,  # 冲淤流速 (m/s)
            'pga_earthquake': 0.1,  # 地震阈值 (g)
        }

    def classify_s5a_inner_leak(self, features: Dict[str, float]) -> Tuple[bool, float]:
        """
        判定 S5-A 内衬渗漏

        条件:
        - 空腔水位上升速率 > δ1
        - DTS温差 ≈ 0 (接近丹江口水温)
        - 水质浊度 < 阈值 (清水)
        """
        conditions = []

        # 空腔水位上升
        if features.get('cavity_level_rate', 0) > self.thresholds['cavity_level_rate']:
            conditions.append(0.4)

        # 温差接近丹江口水温
        temp_diff = abs(features.get('leak_temp', 15) - 8)  # 丹江口水温8°C
        if temp_diff < self.thresholds['temp_diff_inner']:
            conditions.append(0.3)

        # 水质清澈
        if features.get('turbidity', 100) < 20:
            conditions.append(0.3)

        if len(conditions) >= 2:
            confidence = sum(conditions)
            return True, confidence
        return False, 0.0

    def classify_s5b_outer_intrusion(self, features: Dict[str, float]) -> Tuple[bool, float]:
        """
        判定 S5-B 外衬入侵

        条件:
        - 空腔水位上升
        - DTS温差接近黄河水温
        - 水质浑浊
        """
        conditions = []

        if features.get('cavity_level_rate', 0) > self.thresholds['cavity_level_rate']:
            conditions.append(0.4)

        temp_diff = abs(features.get('leak_temp', 15) - 12)  # 黄河水温12°C
        if temp_diff < self.thresholds['temp_diff_outer']:
            conditions.append(0.3)

        if features.get('turbidity', 0) > 50:
            conditions.append(0.3)

        if len(conditions) >= 2:
            return True, sum(conditions)
        return False, 0.0

    def classify_s6a_liquefaction(self, features: Dict[str, float]) -> Tuple[bool, float]:
        """
        判定 S6-A 地震液化

        条件:
        - MEMS震动 > 0.1g
        - 土压力盒读数骤降
        - MEMS倾角异常
        """
        conditions = []

        if features.get('vibration', 0) > self.thresholds['vibration']:
            conditions.append(0.4)

        if features.get('soil_pressure_drop', 0) > 0.3:
            conditions.append(0.3)

        if features.get('tilt', 0) > self.thresholds['tilt']:
            conditions.append(0.3)

        if len(conditions) >= 2:
            return True, min(0.99, sum(conditions))
        return False, 0.0

    def classify_s6b_intake_vortex(self, features: Dict[str, float]) -> Tuple[bool, float]:
        """
        判定 S6-B 进口吸气漩涡

        条件:
        - 进口水位 < f(Q)
        - CV检测到漩涡纹理
        - 洞内声呐含气泡特征
        """
        conditions = []

        inlet_level = features.get('inlet_level', 106)
        flow = features.get('total_flow', 265)
        critical_level = 105 + flow / 500  # 简化的临界水位公式

        if inlet_level < critical_level:
            conditions.append(0.4)

        if features.get('vortex_detected', False):
            conditions.append(0.3)

        if features.get('bubble_noise', False):
            conditions.append(0.3)

        if len(conditions) >= 2:
            return True, sum(conditions)
        return False, 0.0

    def classify(self, features: Dict[str, float]) -> Tuple[ScenarioType, float]:
        """
        执行分类

        Args:
            features: 特征字典

        Returns:
            场景类型, 置信度
        """
        # 按优先级检查各场景

        # 首先检查紧急场景
        detected, conf = self.classify_s6a_liquefaction(features)
        if detected:
            return ScenarioType.S6_A_LIQUEFACTION, conf

        detected, conf = self.classify_s5a_inner_leak(features)
        if detected:
            return ScenarioType.S5_A_INNER_LEAK, conf

        detected, conf = self.classify_s5b_outer_intrusion(features)
        if detected:
            return ScenarioType.S5_B_OUTER_INTRUSION, conf

        detected, conf = self.classify_s6b_intake_vortex(features)
        if detected:
            return ScenarioType.S6_B_INTAKE_VORTEX, conf

        # 检查常态场景
        flow_imbalance = features.get('flow_imbalance', 0)
        if flow_imbalance < self.thresholds['flow_imbalance']:
            return ScenarioType.S1_A_DUAL_BALANCED, 0.9

        # 默认
        return ScenarioType.S1_B_DYNAMIC_PEAK, 0.7


class ScenarioClassifier:
    """
    场景分类器

    采用分层状态机 + 规则推理的混合架构
    """

    def __init__(self):
        """初始化分类器"""
        self.rule_classifier = RuleBasedClassifier()

        # 当前状态
        self.current_scenario = ScenarioType.S1_A_DUAL_BALANCED
        self.current_domain = ScenarioDomain.NOMINAL
        self.scenario_start_time = 0.0

        # 历史缓冲
        self.history_buffer: deque = deque(maxlen=100)

        # 转换矩阵 (定义有效的场景转换)
        self.valid_transitions = self._init_transition_matrix()

        # 滤波参数 (防止频繁切换)
        self.min_duration = 5.0  # 最小场景持续时间 (s)
        self.confirm_count = 3  # 确认计数

        self.pending_scenario = None
        self.pending_count = 0

    def _init_transition_matrix(self) -> Dict[ScenarioType, List[ScenarioType]]:
        """初始化有效转换矩阵"""
        return {
            ScenarioType.S1_A_DUAL_BALANCED: [
                ScenarioType.S1_B_DYNAMIC_PEAK,
                ScenarioType.S2_A_SEDIMENT_FLUSH,
                ScenarioType.S4_A_SWITCH_TUNNEL,
                ScenarioType.S5_A_INNER_LEAK,
                ScenarioType.S5_B_OUTER_INTRUSION,
                ScenarioType.S6_A_LIQUEFACTION,
                ScenarioType.S7_A_PIPE_BURST,
            ],
            ScenarioType.S1_B_DYNAMIC_PEAK: [
                ScenarioType.S1_A_DUAL_BALANCED,
                ScenarioType.S2_A_SEDIMENT_FLUSH,
                ScenarioType.S5_A_INNER_LEAK,
                ScenarioType.S6_A_LIQUEFACTION,
            ],
            ScenarioType.S4_A_SWITCH_TUNNEL: [
                ScenarioType.S1_A_DUAL_BALANCED,
                ScenarioType.S3_A_FILLING,
                ScenarioType.S3_B_DRAINING,
            ],
            ScenarioType.S3_A_FILLING: [
                ScenarioType.S1_A_DUAL_BALANCED,
                ScenarioType.S4_A_SWITCH_TUNNEL,
            ],
            # ... 其他转换规则
        }

    def _extract_features(self, fused_state: Any) -> Dict[str, float]:
        """从融合状态提取特征"""
        features = {
            'flow_1': getattr(fused_state, 'Q1', 132.5),
            'flow_2': getattr(fused_state, 'Q2', 132.5),
            'total_flow': getattr(fused_state, 'Q1', 132.5) + getattr(fused_state, 'Q2', 132.5),
            'flow_imbalance': abs(getattr(fused_state, 'Q1', 132.5) - getattr(fused_state, 'Q2', 132.5)) /
                             max(getattr(fused_state, 'Q1', 132.5) + getattr(fused_state, 'Q2', 132.5), 1),
            'pressure_inlet': getattr(fused_state, 'P_inlet', 6e5),
            'pressure_outlet': getattr(fused_state, 'P_outlet', 5e5),
            'cavity_level': getattr(fused_state, 'cavity_level', 0),
            'cavity_level_rate': 0,  # 需要计算
            'leak_detected': 1.0 if getattr(fused_state, 'leak_detected', False) else 0.0,
            'leak_position': getattr(fused_state, 'leak_position', 0),
            'leak_rate': getattr(fused_state, 'leak_rate', 0),
            'settlement': getattr(fused_state, 'settlement', 0),
            'tilt': getattr(fused_state, 'tilt', 0),
            'vibration': getattr(fused_state, 'vibration_level', 0),
            'turbidity': 10,  # 默认值
        }

        # 计算变化率
        if len(self.history_buffer) > 0:
            prev = self.history_buffer[-1]
            dt = features.get('timestamp', 0) - prev.get('timestamp', 0)
            if dt > 0:
                features['cavity_level_rate'] = (
                    features['cavity_level'] - prev.get('cavity_level', 0)
                ) / dt * 3600  # m/h

        return features

    def _get_domain(self, scenario: ScenarioType) -> ScenarioDomain:
        """获取场景所属域"""
        if scenario in [ScenarioType.S1_A_DUAL_BALANCED, ScenarioType.S1_B_DYNAMIC_PEAK,
                        ScenarioType.S2_A_SEDIMENT_FLUSH, ScenarioType.S2_B_MUSSEL_CONTROL]:
            return ScenarioDomain.NOMINAL
        elif scenario in [ScenarioType.S3_A_FILLING, ScenarioType.S3_B_DRAINING,
                          ScenarioType.S4_A_SWITCH_TUNNEL, ScenarioType.S4_B_ISOLATION]:
            return ScenarioDomain.TRANSITION
        else:
            return ScenarioDomain.EMERGENCY

    def _get_family(self, scenario: ScenarioType) -> ScenarioFamily:
        """获取场景所属族"""
        family_map = {
            ScenarioType.S1_A_DUAL_BALANCED: ScenarioFamily.S1_BALANCED,
            ScenarioType.S1_B_DYNAMIC_PEAK: ScenarioFamily.S1_BALANCED,
            ScenarioType.S2_A_SEDIMENT_FLUSH: ScenarioFamily.S2_MAINTENANCE,
            ScenarioType.S2_B_MUSSEL_CONTROL: ScenarioFamily.S2_MAINTENANCE,
            ScenarioType.S3_A_FILLING: ScenarioFamily.S3_TWO_PHASE,
            ScenarioType.S3_B_DRAINING: ScenarioFamily.S3_TWO_PHASE,
            ScenarioType.S4_A_SWITCH_TUNNEL: ScenarioFamily.S4_TOPOLOGY,
            ScenarioType.S4_B_ISOLATION: ScenarioFamily.S4_TOPOLOGY,
            ScenarioType.S5_A_INNER_LEAK: ScenarioFamily.S5_STRUCTURAL,
            ScenarioType.S5_B_OUTER_INTRUSION: ScenarioFamily.S5_STRUCTURAL,
            ScenarioType.S5_C_JOINT_OFFSET: ScenarioFamily.S5_STRUCTURAL,
            ScenarioType.S6_A_LIQUEFACTION: ScenarioFamily.S6_GEOLOGICAL,
            ScenarioType.S6_B_INTAKE_VORTEX: ScenarioFamily.S6_GEOLOGICAL,
            ScenarioType.S7_A_PIPE_BURST: ScenarioFamily.S7_SYSTEM,
            ScenarioType.S7_B_GATE_ASYNC: ScenarioFamily.S7_SYSTEM,
        }
        return family_map.get(scenario, ScenarioFamily.S1_BALANCED)

    def classify(
        self,
        fused_state: Any,
        timestamp: float
    ) -> ScenarioClassification:
        """
        执行场景分类

        Args:
            fused_state: 融合后状态
            timestamp: 时间戳

        Returns:
            场景分类结果
        """
        # 提取特征
        features = self._extract_features(fused_state)
        features['timestamp'] = timestamp

        # 记录历史
        self.history_buffer.append(features)

        # 规则分类
        detected_scenario, confidence = self.rule_classifier.classify(features)

        # 场景转换滤波
        if detected_scenario != self.current_scenario:
            # 检查是否是有效转换
            valid_next = self.valid_transitions.get(self.current_scenario, [])

            # 紧急场景可以直接切换
            if self._get_domain(detected_scenario) == ScenarioDomain.EMERGENCY:
                self.current_scenario = detected_scenario
                self.scenario_start_time = timestamp
            elif detected_scenario in valid_next:
                # 累积确认
                if detected_scenario == self.pending_scenario:
                    self.pending_count += 1
                else:
                    self.pending_scenario = detected_scenario
                    self.pending_count = 1

                # 达到确认阈值
                if self.pending_count >= self.confirm_count:
                    self.current_scenario = detected_scenario
                    self.scenario_start_time = timestamp
                    self.pending_scenario = None
                    self.pending_count = 0
        else:
            self.pending_scenario = None
            self.pending_count = 0

        # 构建分类结果
        result = ScenarioClassification(
            timestamp=timestamp,
            domain=self._get_domain(self.current_scenario),
            family=self._get_family(self.current_scenario),
            scenario_type=self.current_scenario,
            confidence=confidence,
            evidence=features,
            duration_in_scenario=timestamp - self.scenario_start_time
        )

        return result
