"""
Scenario Generator for CYRP.
穿黄工程场景生成器

自动生成测试场景和扰动序列
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
from enum import Enum

from cyrp.scenarios.scenario_definitions import (
    Scenario, ScenarioType, ScenarioDomain, ScenarioFamily,
    ScenarioConstraints, ScenarioObjective,
    SCENARIO_REGISTRY
)


class DisturbanceType(Enum):
    """扰动类型"""
    STEP = "step"  # 阶跃扰动
    RAMP = "ramp"  # 斜坡扰动
    SINE = "sine"  # 正弦扰动
    NOISE = "noise"  # 噪声扰动
    PULSE = "pulse"  # 脉冲扰动
    EARTHQUAKE = "earthquake"  # 地震
    FAULT = "fault"  # 故障


@dataclass
class Disturbance:
    """扰动定义"""
    disturbance_type: DisturbanceType
    target_variable: str  # 扰动目标变量
    start_time: float
    duration: float
    magnitude: float
    parameters: Dict[str, Any] = field(default_factory=dict)

    def get_value(self, t: float) -> float:
        """获取时刻t的扰动值"""
        if t < self.start_time or t > self.start_time + self.duration:
            return 0.0

        t_rel = t - self.start_time

        if self.disturbance_type == DisturbanceType.STEP:
            return self.magnitude

        elif self.disturbance_type == DisturbanceType.RAMP:
            return self.magnitude * min(t_rel / self.duration, 1.0)

        elif self.disturbance_type == DisturbanceType.SINE:
            freq = self.parameters.get('frequency', 0.1)
            return self.magnitude * np.sin(2 * np.pi * freq * t_rel)

        elif self.disturbance_type == DisturbanceType.NOISE:
            return self.magnitude * np.random.normal()

        elif self.disturbance_type == DisturbanceType.PULSE:
            pulse_width = self.parameters.get('width', 1.0)
            if t_rel <= pulse_width:
                return self.magnitude
            return 0.0

        elif self.disturbance_type == DisturbanceType.EARTHQUAKE:
            # 地震波模拟 (简化)
            pga = self.magnitude
            freq = self.parameters.get('frequency', 2.0)
            decay = np.exp(-t_rel / (self.duration / 3))
            return pga * decay * np.sin(2 * np.pi * freq * t_rel)

        return 0.0


@dataclass
class FaultInjection:
    """故障注入定义"""
    fault_type: str  # leakage, gate_stuck, sensor_fault
    start_time: float
    duration: float
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestScenario:
    """测试场景"""
    name: str
    description: str
    base_scenario: ScenarioType
    duration: float
    disturbances: List[Disturbance] = field(default_factory=list)
    faults: List[FaultInjection] = field(default_factory=list)
    expected_outcomes: Dict[str, Any] = field(default_factory=dict)


class ScenarioGenerator:
    """
    场景生成器

    功能:
    1. 生成标准测试场景
    2. 生成随机扰动序列
    3. 故障注入
    4. 边界条件测试
    """

    def __init__(self, seed: int = 42):
        """初始化生成器"""
        self.rng = np.random.default_rng(seed)

    def generate_nominal_test(
        self,
        duration: float = 3600.0,
        flow_variation: float = 20.0
    ) -> TestScenario:
        """
        生成常态运行测试场景

        Args:
            duration: 测试时长 (s)
            flow_variation: 流量变化幅度 (m³/s)

        Returns:
            测试场景
        """
        disturbances = []

        # 添加周期性流量需求变化
        disturbances.append(Disturbance(
            disturbance_type=DisturbanceType.SINE,
            target_variable='Q_demand',
            start_time=0,
            duration=duration,
            magnitude=flow_variation,
            parameters={'frequency': 1 / 1800}  # 30分钟周期
        ))

        # 添加上游水位波动
        disturbances.append(Disturbance(
            disturbance_type=DisturbanceType.NOISE,
            target_variable='H_inlet',
            start_time=0,
            duration=duration,
            magnitude=0.1
        ))

        return TestScenario(
            name="常态运行测试",
            description="测试系统在正常工况下的流量跟踪和平衡控制能力",
            base_scenario=ScenarioType.S1_A_DUAL_BALANCED,
            duration=duration,
            disturbances=disturbances,
            expected_outcomes={
                'flow_imbalance_max': 0.02,
                'pressure_oscillation_max': 5000
            }
        )

    def generate_tunnel_switch_test(
        self,
        switch_time: float = 300.0,
        preparation_time: float = 60.0
    ) -> TestScenario:
        """
        生成倒洞切换测试场景 (S4-A)

        Args:
            switch_time: 切换时间 (s)
            preparation_time: 准备时间 (s)

        Returns:
            测试场景
        """
        disturbances = []

        # 目标: 将2#洞流量降至0
        disturbances.append(Disturbance(
            disturbance_type=DisturbanceType.RAMP,
            target_variable='Q2_target',
            start_time=preparation_time,
            duration=switch_time,
            magnitude=-132.5  # 从132.5降到0
        ))

        # 1#洞补偿
        disturbances.append(Disturbance(
            disturbance_type=DisturbanceType.RAMP,
            target_variable='Q1_target',
            start_time=preparation_time,
            duration=switch_time,
            magnitude=132.5  # 从132.5升到265
        ))

        return TestScenario(
            name="不停水倒洞测试",
            description="测试双洞向单洞平滑切换的控制能力",
            base_scenario=ScenarioType.S4_A_SWITCH_TUNNEL,
            duration=preparation_time + switch_time + 300,
            disturbances=disturbances,
            expected_outcomes={
                'total_flow_deviation_max': 0.05,  # 总流量波动 < 5%
                'pressure_spike_max': 50000  # 压力尖峰 < 0.5bar
            }
        )

    def generate_filling_test(
        self,
        target_level: float = 7.0,
        max_rate: float = 0.5
    ) -> TestScenario:
        """
        生成充水排气测试场景 (S3-A)

        Args:
            target_level: 目标水位 (m)
            max_rate: 最大水位上升速率 (m/min)

        Returns:
            测试场景
        """
        # 估算充水时间
        fill_time = target_level / (max_rate / 60) * 1.5  # 安全系数1.5

        disturbances = []

        # 充水阀开启
        disturbances.append(Disturbance(
            disturbance_type=DisturbanceType.STEP,
            target_variable='valve_fill',
            start_time=0,
            duration=fill_time,
            magnitude=0.5  # 50%开度
        ))

        return TestScenario(
            name="充水排气测试",
            description="测试空管充水过程的安全控制",
            base_scenario=ScenarioType.S3_A_FILLING,
            duration=fill_time + 300,
            disturbances=disturbances,
            expected_outcomes={
                'water_level_rate_max': max_rate,
                'air_pressure_max': 150000,  # 1.5atm
                'no_air_explosion': True
            }
        )

    def generate_leakage_test(
        self,
        leak_position: float = 2150.0,
        leak_rate_initial: float = 0.05,
        detection_time_max: float = 60.0
    ) -> TestScenario:
        """
        生成渗漏检测测试场景 (S5-A)

        Args:
            leak_position: 渗漏位置 (m)
            leak_rate_initial: 初始渗漏率 (m³/s)
            detection_time_max: 最大检测时间 (s)

        Returns:
            测试场景
        """
        faults = [
            FaultInjection(
                fault_type='leakage',
                start_time=60.0,  # 1分钟后开始渗漏
                duration=600.0,
                parameters={
                    'position': leak_position,
                    'initial_rate': leak_rate_initial,
                    'growth_rate': 0.001,
                    'type': 'inner'
                }
            )
        ]

        return TestScenario(
            name="内衬渗漏检测测试",
            description="测试DAS/DTS系统的渗漏检测和定位能力",
            base_scenario=ScenarioType.S5_A_INNER_LEAK,
            duration=900,
            faults=faults,
            expected_outcomes={
                'detection_time': detection_time_max,
                'position_accuracy': 10.0,  # ±10m
                'type_correct': True
            }
        )

    def generate_earthquake_test(
        self,
        magnitude: float = 6.5,
        pga: float = 0.15,
        duration: float = 30.0
    ) -> TestScenario:
        """
        生成地震测试场景 (S6-A)

        Args:
            magnitude: 震级
            pga: 峰值地面加速度 (g)
            duration: 地震持续时间 (s)

        Returns:
            测试场景
        """
        disturbances = [
            Disturbance(
                disturbance_type=DisturbanceType.EARTHQUAKE,
                target_variable='ground_acceleration',
                start_time=60.0,  # 1分钟后发生地震
                duration=duration,
                magnitude=pga * 9.81,
                parameters={
                    'frequency': 2.0,
                    'magnitude': magnitude
                }
            )
        ]

        return TestScenario(
            name="地震液化测试",
            description="测试系统在地震工况下的抗浮和结构保护能力",
            base_scenario=ScenarioType.S6_A_LIQUEFACTION,
            duration=7200,  # 2小时 (含震后恢复)
            disturbances=disturbances,
            expected_outcomes={
                'no_flotation': True,
                'structural_integrity': True,
                'max_pressure': 1.5e6
            }
        )

    def generate_gate_failure_test(
        self,
        failed_gate: int = 2,
        stuck_position: float = 0.3
    ) -> TestScenario:
        """
        生成闸门故障测试场景 (S7-B)

        Args:
            failed_gate: 故障闸门编号 (1或2)
            stuck_position: 卡死位置 (0-1)

        Returns:
            测试场景
        """
        faults = [
            FaultInjection(
                fault_type='gate_stuck',
                start_time=120.0,
                duration=600.0,
                parameters={
                    'gate': failed_gate,
                    'position': stuck_position
                }
            )
        ]

        return TestScenario(
            name="闸门故障测试",
            description="测试闸门卡死时的应急处理能力",
            base_scenario=ScenarioType.S7_B_GATE_ASYNC,
            duration=900,
            faults=faults,
            expected_outcomes={
                'total_flow_maintained': True,
                'safe_shutdown': True
            }
        )

    def generate_random_scenario(
        self,
        domain: Optional[ScenarioDomain] = None,
        num_disturbances: int = 3,
        duration: float = 1800.0
    ) -> TestScenario:
        """
        生成随机测试场景

        Args:
            domain: 场景域 (None表示随机选择)
            num_disturbances: 扰动数量
            duration: 测试时长 (s)

        Returns:
            随机测试场景
        """
        # 随机选择基础场景
        if domain is None:
            domain = self.rng.choice(list(ScenarioDomain))

        scenarios_in_domain = [
            s for s in SCENARIO_REGISTRY.values()
            if s.domain == domain
        ]

        if not scenarios_in_domain:
            scenarios_in_domain = list(SCENARIO_REGISTRY.values())

        base = self.rng.choice(scenarios_in_domain)

        # 生成随机扰动
        disturbances = []
        for i in range(num_disturbances):
            dist_type = self.rng.choice([
                DisturbanceType.STEP,
                DisturbanceType.RAMP,
                DisturbanceType.SINE,
                DisturbanceType.NOISE
            ])

            target_var = self.rng.choice([
                'H_inlet', 'H_outlet', 'Q_demand', 'P_external'
            ])

            start = self.rng.uniform(0, duration * 0.7)
            dur = self.rng.uniform(duration * 0.1, duration * 0.3)
            mag = self.rng.uniform(0.5, 2.0)

            disturbances.append(Disturbance(
                disturbance_type=dist_type,
                target_variable=target_var,
                start_time=start,
                duration=dur,
                magnitude=mag
            ))

        return TestScenario(
            name=f"随机测试_{self.rng.integers(10000)}",
            description="随机生成的测试场景",
            base_scenario=base.scenario_type,
            duration=duration,
            disturbances=disturbances
        )

    def generate_full_coverage_suite(self) -> List[TestScenario]:
        """
        生成全场景覆盖测试套件

        Returns:
            测试场景列表
        """
        suite = []

        # 常态测试
        suite.append(self.generate_nominal_test())

        # 过渡态测试
        suite.append(self.generate_tunnel_switch_test())
        suite.append(self.generate_filling_test())

        # 应急测试
        suite.append(self.generate_leakage_test())
        suite.append(self.generate_earthquake_test())
        suite.append(self.generate_gate_failure_test())

        # 随机测试
        for _ in range(5):
            suite.append(self.generate_random_scenario())

        return suite

    def get_disturbance_schedule(
        self,
        test_scenario: TestScenario
    ) -> Dict[float, Dict[str, float]]:
        """
        将测试场景转换为扰动时间表

        Args:
            test_scenario: 测试场景

        Returns:
            时间 -> 扰动值的映射
        """
        dt = 0.1  # 采样间隔
        times = np.arange(0, test_scenario.duration, dt)

        schedule = {}
        for t in times:
            disturbance_values = {}
            for dist in test_scenario.disturbances:
                value = dist.get_value(t)
                if abs(value) > 1e-10:
                    disturbance_values[dist.target_variable] = value

            if disturbance_values:
                schedule[t] = disturbance_values

        return schedule
