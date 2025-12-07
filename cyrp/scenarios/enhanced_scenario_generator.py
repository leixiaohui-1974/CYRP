"""
增强型场景生成器 - Enhanced Scenario Generator

实现边界条件测试、组合场景、时序场景、蒙特卡洛场景生成
Implements boundary condition testing, combined scenarios, temporal sequences, Monte Carlo generation
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import itertools
import random
from abc import ABC, abstractmethod


class ScenarioComplexity(Enum):
    """场景复杂度等级"""
    SIMPLE = 1          # 单一故障/工况
    MODERATE = 2        # 双重故障/工况叠加
    COMPLEX = 3         # 多重故障/极端工况
    EXTREME = 4         # 边界条件/最恶劣工况


class TemporalPattern(Enum):
    """时序模式"""
    STEP = "step"               # 阶跃变化
    RAMP = "ramp"               # 斜坡变化
    SINUSOIDAL = "sinusoidal"   # 正弦变化
    PULSE = "pulse"             # 脉冲
    RANDOM_WALK = "random_walk" # 随机游走
    CHIRP = "chirp"             # 扫频
    EXPONENTIAL = "exponential" # 指数变化


@dataclass
class BoundaryCondition:
    """边界条件定义"""
    parameter: str              # 参数名
    nominal_value: float        # 标称值
    min_value: float           # 最小值
    max_value: float           # 最大值
    critical_low: float        # 临界低值
    critical_high: float       # 临界高值
    unit: str                  # 单位

    def get_test_values(self) -> List[float]:
        """获取边界测试值"""
        return [
            self.min_value,
            self.min_value + 0.01 * (self.max_value - self.min_value),
            self.critical_low,
            self.nominal_value,
            self.critical_high,
            self.max_value - 0.01 * (self.max_value - self.min_value),
            self.max_value
        ]


@dataclass
class TemporalEvent:
    """时序事件"""
    time: float                 # 发生时间 (s)
    event_type: str            # 事件类型
    parameters: Dict[str, Any] # 事件参数
    duration: float = 0.0      # 持续时间 (s)
    pattern: TemporalPattern = TemporalPattern.STEP


@dataclass
class ScenarioSequence:
    """场景序列"""
    name: str
    description: str
    events: List[TemporalEvent]
    total_duration: float
    complexity: ScenarioComplexity
    success_criteria: Dict[str, Any]


@dataclass
class CombinedScenario:
    """组合场景"""
    name: str
    base_scenarios: List[str]   # 基础场景ID列表
    interaction_mode: str       # 交互模式: simultaneous, sequential, cascading
    parameters: Dict[str, Any]
    probability: float          # 发生概率
    severity: float            # 严重程度 0-1


class SignalGenerator:
    """信号生成器 - 用于生成各种测试信号"""

    @staticmethod
    def step(t: np.ndarray, t_step: float, amplitude: float, baseline: float = 0) -> np.ndarray:
        """阶跃信号"""
        return np.where(t >= t_step, baseline + amplitude, baseline)

    @staticmethod
    def ramp(t: np.ndarray, t_start: float, t_end: float,
             start_value: float, end_value: float) -> np.ndarray:
        """斜坡信号"""
        result = np.zeros_like(t)
        for i, ti in enumerate(t):
            if ti < t_start:
                result[i] = start_value
            elif ti > t_end:
                result[i] = end_value
            else:
                result[i] = start_value + (end_value - start_value) * (ti - t_start) / (t_end - t_start)
        return result

    @staticmethod
    def sinusoidal(t: np.ndarray, amplitude: float, frequency: float,
                  phase: float = 0, offset: float = 0) -> np.ndarray:
        """正弦信号"""
        return offset + amplitude * np.sin(2 * np.pi * frequency * t + phase)

    @staticmethod
    def pulse(t: np.ndarray, t_start: float, duration: float,
              amplitude: float, baseline: float = 0) -> np.ndarray:
        """脉冲信号"""
        return np.where((t >= t_start) & (t < t_start + duration),
                       baseline + amplitude, baseline)

    @staticmethod
    def random_walk(t: np.ndarray, volatility: float, initial: float = 0) -> np.ndarray:
        """随机游走"""
        dt = t[1] - t[0] if len(t) > 1 else 0.1
        n = len(t)
        increments = np.random.normal(0, volatility * np.sqrt(dt), n)
        return initial + np.cumsum(increments)

    @staticmethod
    def chirp(t: np.ndarray, f0: float, f1: float, amplitude: float) -> np.ndarray:
        """扫频信号"""
        T = t[-1] - t[0]
        k = (f1 - f0) / T
        phase = 2 * np.pi * (f0 * t + 0.5 * k * t**2)
        return amplitude * np.sin(phase)

    @staticmethod
    def exponential(t: np.ndarray, t_start: float, tau: float,
                   initial: float, final: float) -> np.ndarray:
        """指数信号"""
        result = np.zeros_like(t)
        for i, ti in enumerate(t):
            if ti < t_start:
                result[i] = initial
            else:
                result[i] = final + (initial - final) * np.exp(-(ti - t_start) / tau)
        return result


class EnhancedScenarioGenerator:
    """增强型场景生成器"""

    def __init__(self):
        self.signal_gen = SignalGenerator()
        self._init_boundary_conditions()
        self._init_combined_scenarios()
        self._init_temporal_sequences()

    def _init_boundary_conditions(self):
        """初始化边界条件"""
        self.boundary_conditions = {
            # 水力学边界
            'flow_rate': BoundaryCondition(
                parameter='flow_rate',
                nominal_value=265.0,
                min_value=0.0,
                max_value=320.0,
                critical_low=50.0,
                critical_high=300.0,
                unit='m³/s'
            ),
            'pressure': BoundaryCondition(
                parameter='pressure',
                nominal_value=0.5,
                min_value=-0.05,  # 轻微负压
                max_value=1.0,
                critical_low=0.02,
                critical_high=0.9,
                unit='MPa'
            ),
            'velocity': BoundaryCondition(
                parameter='velocity',
                nominal_value=3.44,
                min_value=0.0,
                max_value=5.0,
                critical_low=0.5,
                critical_high=4.5,
                unit='m/s'
            ),
            'water_level': BoundaryCondition(
                parameter='water_level',
                nominal_value=6.5,
                min_value=0.5,
                max_value=7.0,
                critical_low=2.0,
                critical_high=6.8,
                unit='m'
            ),
            # 结构边界
            'temperature': BoundaryCondition(
                parameter='temperature',
                nominal_value=15.0,
                min_value=2.0,
                max_value=35.0,
                critical_low=5.0,
                critical_high=30.0,
                unit='°C'
            ),
            'external_pressure': BoundaryCondition(
                parameter='external_pressure',
                nominal_value=0.7,
                min_value=0.0,
                max_value=1.5,
                critical_low=0.1,
                critical_high=1.2,
                unit='MPa'
            ),
            # 环境边界
            'seismic_intensity': BoundaryCondition(
                parameter='seismic_intensity',
                nominal_value=0.0,
                min_value=0.0,
                max_value=0.4,  # VIII度烈度对应加速度
                critical_low=0.0,
                critical_high=0.3,
                unit='g'
            ),
            'groundwater_level': BoundaryCondition(
                parameter='groundwater_level',
                nominal_value=10.0,
                min_value=0.0,
                max_value=30.0,
                critical_low=2.0,
                critical_high=25.0,
                unit='m'
            ),
        }

    def _init_combined_scenarios(self):
        """初始化组合场景"""
        self.combined_scenarios = [
            # 双重故障场景
            CombinedScenario(
                name="leakage_during_switch",
                base_scenarios=["S5-B", "S3-B"],
                interaction_mode="simultaneous",
                parameters={
                    'leakage_rate': 0.02,
                    'switch_urgency': 'emergency'
                },
                probability=0.001,
                severity=0.8
            ),
            CombinedScenario(
                name="seismic_with_leakage",
                base_scenarios=["S6-B", "S5-C"],
                interaction_mode="cascading",
                parameters={
                    'seismic_intensity': 0.2,
                    'induced_leakage_delay': 30.0
                },
                probability=0.0005,
                severity=0.95
            ),
            CombinedScenario(
                name="high_flow_with_blockage",
                base_scenarios=["S1-A", "blockage"],
                interaction_mode="simultaneous",
                parameters={
                    'flow_rate': 300.0,
                    'blockage_ratio': 0.15
                },
                probability=0.002,
                severity=0.7
            ),
            CombinedScenario(
                name="maintenance_emergency",
                base_scenarios=["S4-B", "S5-A"],
                interaction_mode="sequential",
                parameters={
                    'maintenance_phase': 'inspection',
                    'leakage_onset_time': 600.0
                },
                probability=0.003,
                severity=0.6
            ),
            # 三重故障场景
            CombinedScenario(
                name="seismic_leakage_power_loss",
                base_scenarios=["S6-B", "S5-B", "power_failure"],
                interaction_mode="cascading",
                parameters={
                    'seismic_intensity': 0.15,
                    'power_restoration_time': 300.0
                },
                probability=0.0001,
                severity=0.99
            ),
            CombinedScenario(
                name="dual_tunnel_asymmetric_failure",
                base_scenarios=["S2-A", "asymmetric_blockage", "sensor_failure"],
                interaction_mode="simultaneous",
                parameters={
                    'blockage_tunnel': 'north',
                    'blockage_ratio': 0.3,
                    'failed_sensors': ['pressure_north_2', 'pressure_north_3']
                },
                probability=0.0002,
                severity=0.85
            ),
        ]

    def _init_temporal_sequences(self):
        """初始化时序场景"""
        self.temporal_sequences = [
            # 日常运行序列
            ScenarioSequence(
                name="daily_operation_cycle",
                description="24小时日常运行周期",
                events=[
                    TemporalEvent(0, "flow_change", {"target": 200.0}, pattern=TemporalPattern.RAMP),
                    TemporalEvent(6*3600, "flow_change", {"target": 265.0}, pattern=TemporalPattern.RAMP),
                    TemporalEvent(12*3600, "flow_change", {"target": 280.0}, pattern=TemporalPattern.RAMP),
                    TemporalEvent(18*3600, "flow_change", {"target": 250.0}, pattern=TemporalPattern.RAMP),
                    TemporalEvent(22*3600, "flow_change", {"target": 220.0}, pattern=TemporalPattern.RAMP),
                ],
                total_duration=24*3600,
                complexity=ScenarioComplexity.SIMPLE,
                success_criteria={"flow_tracking_error": 5.0, "pressure_stability": 0.05}
            ),
            # 计划检修序列
            ScenarioSequence(
                name="planned_maintenance_sequence",
                description="计划检修完整流程",
                events=[
                    TemporalEvent(0, "notification", {"type": "maintenance_start"}),
                    TemporalEvent(300, "flow_reduction", {"target": 150.0}, duration=600, pattern=TemporalPattern.RAMP),
                    TemporalEvent(900, "tunnel_switch", {"from": "north", "to": "south"}),
                    TemporalEvent(1200, "flow_stabilization", {"target": 265.0}, pattern=TemporalPattern.RAMP),
                    TemporalEvent(1800, "maintenance_start", {"tunnel": "north"}),
                    TemporalEvent(7200, "maintenance_end", {"tunnel": "north"}),
                    TemporalEvent(7500, "tunnel_switch", {"from": "south", "to": "dual"}),
                    TemporalEvent(8100, "flow_restoration", {"target": 265.0}, pattern=TemporalPattern.RAMP),
                ],
                total_duration=9000,
                complexity=ScenarioComplexity.MODERATE,
                success_criteria={"switch_time": 120, "flow_deviation": 10.0}
            ),
            # 紧急响应序列
            ScenarioSequence(
                name="emergency_response_sequence",
                description="紧急情况响应流程",
                events=[
                    TemporalEvent(0, "leakage_detection", {"location": 2125, "rate": 0.01}),
                    TemporalEvent(5, "alarm_activation", {"level": "warning"}),
                    TemporalEvent(10, "flow_reduction", {"target": 200.0}, pattern=TemporalPattern.STEP),
                    TemporalEvent(30, "leakage_confirmation", {"rate": 0.015}),
                    TemporalEvent(35, "alarm_escalation", {"level": "critical"}),
                    TemporalEvent(40, "emergency_switch", {"to": "south"}),
                    TemporalEvent(120, "north_tunnel_isolation", {}),
                    TemporalEvent(180, "flow_restoration", {"target": 265.0}, pattern=TemporalPattern.RAMP),
                ],
                total_duration=600,
                complexity=ScenarioComplexity.COMPLEX,
                success_criteria={"response_time": 60, "isolation_time": 180}
            ),
            # 地震响应序列
            ScenarioSequence(
                name="seismic_response_sequence",
                description="地震事件响应流程",
                events=[
                    TemporalEvent(0, "seismic_detection", {"intensity": 0.15, "duration": 30}),
                    TemporalEvent(0.5, "emergency_mode_activation", {}),
                    TemporalEvent(1, "flow_reduction", {"target": 100.0}, pattern=TemporalPattern.STEP),
                    TemporalEvent(30, "seismic_end", {}),
                    TemporalEvent(35, "structural_assessment", {}),
                    TemporalEvent(60, "damage_evaluation", {"result": "minor"}),
                    TemporalEvent(120, "gradual_restoration", {"target": 200.0}, pattern=TemporalPattern.RAMP),
                    TemporalEvent(600, "full_restoration", {"target": 265.0}, pattern=TemporalPattern.RAMP),
                ],
                total_duration=1200,
                complexity=ScenarioComplexity.EXTREME,
                success_criteria={"response_time": 2, "structural_safety": True}
            ),
        ]

    def generate_boundary_test_scenarios(self) -> List[Dict[str, Any]]:
        """生成边界条件测试场景"""
        scenarios = []

        for param_name, bc in self.boundary_conditions.items():
            test_values = bc.get_test_values()

            for i, value in enumerate(test_values):
                scenario = {
                    'id': f"BC_{param_name}_{i}",
                    'name': f"Boundary test: {param_name} = {value:.3f} {bc.unit}",
                    'type': 'boundary_condition',
                    'parameter': param_name,
                    'value': value,
                    'nominal': bc.nominal_value,
                    'is_critical': value <= bc.critical_low or value >= bc.critical_high,
                    'is_extreme': value == bc.min_value or value == bc.max_value,
                    'duration': 300.0,
                    'expected_behavior': self._get_expected_behavior(param_name, value, bc)
                }
                scenarios.append(scenario)

        return scenarios

    def _get_expected_behavior(self, param: str, value: float, bc: BoundaryCondition) -> str:
        """获取预期行为描述"""
        if value <= bc.critical_low:
            return f"Critical low {param}: expect protective action"
        elif value >= bc.critical_high:
            return f"Critical high {param}: expect limiting action"
        elif value == bc.min_value:
            return f"Minimum {param}: expect shutdown protection"
        elif value == bc.max_value:
            return f"Maximum {param}: expect emergency response"
        else:
            return f"Normal operation range for {param}"

    def generate_combinatorial_scenarios(self, max_combinations: int = 100) -> List[Dict[str, Any]]:
        """生成组合场景 - 使用正交试验设计"""
        scenarios = []

        # 关键参数及其水平
        factors = {
            'flow_rate': [100.0, 200.0, 265.0, 300.0],
            'pressure_mode': ['normal', 'high', 'low', 'fluctuating'],
            'tunnel_config': ['north_only', 'south_only', 'dual', 'switching'],
            'environment': ['normal', 'high_temp', 'low_temp', 'seismic'],
            'sensor_status': ['all_ok', 'partial_failure', 'redundant_mode', 'degraded'],
        }

        # 生成正交表的子集
        all_combinations = list(itertools.product(*factors.values()))
        selected = random.sample(all_combinations, min(max_combinations, len(all_combinations)))

        for i, combo in enumerate(selected):
            scenario = {
                'id': f"COMB_{i:03d}",
                'name': f"Combinatorial scenario {i}",
                'type': 'combinatorial',
                'factors': dict(zip(factors.keys(), combo)),
                'complexity': self._assess_complexity(combo),
                'duration': 600.0,
                'test_objectives': self._get_test_objectives(combo)
            }
            scenarios.append(scenario)

        return scenarios

    def _assess_complexity(self, combo: tuple) -> ScenarioComplexity:
        """评估场景复杂度"""
        complexity_score = 0

        # 高流量
        if combo[0] >= 280:
            complexity_score += 1

        # 异常压力模式
        if combo[1] in ['high', 'fluctuating']:
            complexity_score += 1

        # 切换模式
        if combo[2] == 'switching':
            complexity_score += 1

        # 异常环境
        if combo[3] in ['seismic', 'high_temp']:
            complexity_score += 2

        # 传感器故障
        if combo[4] in ['partial_failure', 'degraded']:
            complexity_score += 1

        if complexity_score <= 1:
            return ScenarioComplexity.SIMPLE
        elif complexity_score <= 3:
            return ScenarioComplexity.MODERATE
        elif complexity_score <= 5:
            return ScenarioComplexity.COMPLEX
        else:
            return ScenarioComplexity.EXTREME

    def _get_test_objectives(self, combo: tuple) -> List[str]:
        """获取测试目标"""
        objectives = ["basic_control_stability"]

        if combo[0] >= 280:
            objectives.append("high_flow_capacity")
        if combo[1] == 'fluctuating':
            objectives.append("disturbance_rejection")
        if combo[2] == 'switching':
            objectives.append("bumpless_transfer")
        if combo[3] == 'seismic':
            objectives.append("emergency_response")
        if combo[4] != 'all_ok':
            objectives.append("fault_tolerance")

        return objectives

    def generate_monte_carlo_scenarios(self, n_samples: int = 1000) -> List[Dict[str, Any]]:
        """蒙特卡洛场景生成"""
        scenarios = []

        for i in range(n_samples):
            # 随机采样参数
            flow_rate = np.random.triangular(50, 265, 320)
            pressure_var = np.random.exponential(0.02)
            temp = np.random.normal(15, 5)

            # 随机故障
            has_leakage = np.random.random() < 0.05
            leakage_rate = np.random.exponential(0.005) if has_leakage else 0

            has_sensor_fault = np.random.random() < 0.02
            n_failed_sensors = np.random.poisson(1) if has_sensor_fault else 0

            has_seismic = np.random.random() < 0.001
            seismic_intensity = np.random.exponential(0.1) if has_seismic else 0

            scenario = {
                'id': f"MC_{i:04d}",
                'name': f"Monte Carlo sample {i}",
                'type': 'monte_carlo',
                'parameters': {
                    'flow_rate': float(flow_rate),
                    'pressure_variance': float(pressure_var),
                    'temperature': float(np.clip(temp, 2, 35)),
                    'leakage_rate': float(leakage_rate),
                    'n_failed_sensors': int(n_failed_sensors),
                    'seismic_intensity': float(min(seismic_intensity, 0.4)),
                },
                'faults': {
                    'leakage': has_leakage,
                    'sensor_fault': has_sensor_fault,
                    'seismic': has_seismic,
                },
                'risk_score': self._calculate_risk_score(
                    flow_rate, leakage_rate, n_failed_sensors, seismic_intensity
                ),
                'duration': 300.0,
            }
            scenarios.append(scenario)

        return scenarios

    def _calculate_risk_score(self, flow: float, leakage: float,
                             n_sensors: int, seismic: float) -> float:
        """计算风险分数"""
        risk = 0.0

        # 流量风险
        if flow > 280:
            risk += 0.2 * (flow - 280) / 40
        elif flow < 100:
            risk += 0.1 * (100 - flow) / 100

        # 泄漏风险
        risk += min(leakage / 0.05, 1.0) * 0.4

        # 传感器故障风险
        risk += min(n_sensors / 5, 1.0) * 0.2

        # 地震风险
        risk += min(seismic / 0.3, 1.0) * 0.3

        return min(risk, 1.0)

    def generate_stress_test_scenarios(self) -> List[Dict[str, Any]]:
        """生成压力测试场景"""
        scenarios = []

        # 最大流量持续运行
        scenarios.append({
            'id': 'STRESS_001',
            'name': 'Maximum flow sustained operation',
            'type': 'stress_test',
            'description': '最大流量320m³/s持续运行测试',
            'parameters': {
                'flow_rate': 320.0,
                'duration': 24 * 3600,
            },
            'success_criteria': {
                'pressure_within_limits': True,
                'velocity_within_limits': True,
                'no_cavitation': True,
            }
        })

        # 快速流量变化
        scenarios.append({
            'id': 'STRESS_002',
            'name': 'Rapid flow changes',
            'type': 'stress_test',
            'description': '快速流量变化抗扰测试',
            'parameters': {
                'flow_changes': [
                    {'time': 0, 'target': 265},
                    {'time': 60, 'target': 100},
                    {'time': 120, 'target': 300},
                    {'time': 180, 'target': 150},
                    {'time': 240, 'target': 280},
                ],
                'change_rate': 10.0,  # m³/s per second
            },
            'success_criteria': {
                'overshoot': 0.1,
                'settling_time': 30,
            }
        })

        # 多重故障叠加
        scenarios.append({
            'id': 'STRESS_003',
            'name': 'Multiple simultaneous faults',
            'type': 'stress_test',
            'description': '多重故障同时发生测试',
            'parameters': {
                'faults': [
                    {'type': 'sensor_failure', 'sensors': ['P1', 'P2', 'F1']},
                    {'type': 'leakage', 'rate': 0.02, 'location': 1500},
                    {'type': 'actuator_degradation', 'valve': 'north_inlet', 'factor': 0.7},
                ],
            },
            'success_criteria': {
                'safe_shutdown_achieved': True,
                'no_structural_damage': True,
            }
        })

        # 极端环境条件
        scenarios.append({
            'id': 'STRESS_004',
            'name': 'Extreme environmental conditions',
            'type': 'stress_test',
            'description': '极端环境条件运行测试',
            'parameters': {
                'temperature': 35.0,
                'external_pressure': 1.2,
                'groundwater_level': 25.0,
                'duration': 8 * 3600,
            },
            'success_criteria': {
                'structural_integrity': True,
                'thermal_stability': True,
            }
        })

        return scenarios

    def generate_temporal_scenario(self, sequence: ScenarioSequence,
                                   dt: float = 0.1) -> Dict[str, Any]:
        """根据时序序列生成具体的时间序列数据"""
        t = np.arange(0, sequence.total_duration, dt)
        n_steps = len(t)

        # 初始化输出信号
        signals = {
            'time': t,
            'flow_setpoint': np.ones(n_steps) * 265.0,
            'events': [],
            'alarms': [],
        }

        for event in sequence.events:
            event_idx = int(event.time / dt)

            if event.event_type == 'flow_change' or event.event_type == 'flow_reduction':
                target = event.parameters.get('target', 265.0)

                if event.pattern == TemporalPattern.STEP:
                    signals['flow_setpoint'][event_idx:] = target
                elif event.pattern == TemporalPattern.RAMP:
                    ramp_duration = event.duration if event.duration > 0 else 300
                    ramp_end_idx = min(event_idx + int(ramp_duration / dt), n_steps)
                    current_value = signals['flow_setpoint'][event_idx - 1] if event_idx > 0 else 265.0

                    for i in range(event_idx, ramp_end_idx):
                        progress = (i - event_idx) / (ramp_end_idx - event_idx)
                        signals['flow_setpoint'][i] = current_value + progress * (target - current_value)
                    signals['flow_setpoint'][ramp_end_idx:] = target

            # 记录事件
            signals['events'].append({
                'time': event.time,
                'type': event.event_type,
                'parameters': event.parameters
            })

        return {
            'sequence_name': sequence.name,
            'description': sequence.description,
            'complexity': sequence.complexity.name,
            'signals': signals,
            'success_criteria': sequence.success_criteria
        }

    def generate_full_test_suite(self) -> Dict[str, List[Dict[str, Any]]]:
        """生成完整测试套件"""
        return {
            'boundary_tests': self.generate_boundary_test_scenarios(),
            'combinatorial_tests': self.generate_combinatorial_scenarios(50),
            'monte_carlo_tests': self.generate_monte_carlo_scenarios(100),
            'stress_tests': self.generate_stress_test_scenarios(),
            'temporal_sequences': [
                self.generate_temporal_scenario(seq)
                for seq in self.temporal_sequences
            ],
            'combined_scenarios': [
                {
                    'id': cs.name,
                    'base_scenarios': cs.base_scenarios,
                    'interaction_mode': cs.interaction_mode,
                    'parameters': cs.parameters,
                    'probability': cs.probability,
                    'severity': cs.severity
                }
                for cs in self.combined_scenarios
            ]
        }

    def get_scenario_coverage_report(self) -> Dict[str, Any]:
        """获取场景覆盖报告"""
        suite = self.generate_full_test_suite()

        return {
            'total_scenarios': sum(len(v) for v in suite.values()),
            'by_category': {k: len(v) for k, v in suite.items()},
            'boundary_parameters_covered': list(self.boundary_conditions.keys()),
            'complexity_distribution': {
                'simple': sum(1 for s in suite['combinatorial_tests']
                            if s['complexity'] == ScenarioComplexity.SIMPLE),
                'moderate': sum(1 for s in suite['combinatorial_tests']
                              if s['complexity'] == ScenarioComplexity.MODERATE),
                'complex': sum(1 for s in suite['combinatorial_tests']
                             if s['complexity'] == ScenarioComplexity.COMPLEX),
                'extreme': sum(1 for s in suite['combinatorial_tests']
                             if s['complexity'] == ScenarioComplexity.EXTREME),
            },
            'risk_distribution': {
                'low': sum(1 for s in suite['monte_carlo_tests'] if s['risk_score'] < 0.3),
                'medium': sum(1 for s in suite['monte_carlo_tests']
                            if 0.3 <= s['risk_score'] < 0.6),
                'high': sum(1 for s in suite['monte_carlo_tests'] if s['risk_score'] >= 0.6),
            }
        }
