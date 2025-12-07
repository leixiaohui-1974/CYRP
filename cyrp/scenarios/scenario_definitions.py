"""
Scenario Definitions for CYRP.
穿黄工程场景定义

定义32种细分工况的完整参数
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
from enum import Enum
import numpy as np


class ScenarioDomain(Enum):
    """场景域 (3大类)"""
    NOMINAL = "nominal"  # 常态运行域 (95%运行时间)
    TRANSITION = "transition"  # 过渡运维域
    EMERGENCY = "emergency"  # 应急灾害域


class ScenarioFamily(Enum):
    """场景族 (8个)"""
    S1_BALANCED = "S1_balanced"  # 平衡输水
    S2_MAINTENANCE = "S2_maintenance"  # 维护性输水
    S3_TWO_PHASE = "S3_two_phase"  # 气液两相转换
    S4_TOPOLOGY = "S4_topology"  # 拓扑切换
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

    # S3 气液两相转换场景族 (最危险)
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


@dataclass
class ScenarioConstraints:
    """场景约束条件"""
    # 流量约束
    Q_min: float = 0.0  # 最小流量 (m³/s)
    Q_max: float = 305.0  # 最大流量 (m³/s)
    Q_imbalance_max: float = 0.1  # 最大流量不平衡度

    # 压力约束
    P_min: float = -5e4  # 最小压力 (Pa)
    P_max: float = 1e6  # 最大压力 (Pa)
    dP_dt_max: float = 5e4  # 最大压力变化率 (Pa/s)

    # 水位约束
    H_rate_max: float = 0.5  # 最大水位变化率 (m/min)

    # 闸门约束
    gate_rate_max: float = 0.01  # 最大闸门动作速率 (1/s)

    # 结构约束
    P_internal_min: float = 0.0  # 最小内压 (相对于外压)


@dataclass
class ScenarioObjective:
    """场景控制目标"""
    # 目标类型
    objective_type: str = "tracking"  # tracking, regulation, emergency

    # 跟踪目标
    Q_target: float = 265.0  # 目标流量
    P_target: float = 5e5  # 目标压力
    balance_weight: float = 1.0  # 平衡权重

    # 权重
    weights: Dict[str, float] = field(default_factory=lambda: {
        'flow_tracking': 1.0,
        'flow_balance': 1.0,
        'pressure_smooth': 0.1,
        'actuator_effort': 0.01
    })


@dataclass
class Scenario:
    """
    场景完整定义

    包含场景的所有参数和约束
    """
    # 基本信息
    scenario_type: ScenarioType
    domain: ScenarioDomain
    family: ScenarioFamily
    name: str
    description: str

    # 初始条件
    initial_conditions: Dict[str, float] = field(default_factory=dict)

    # 约束条件
    constraints: ScenarioConstraints = field(default_factory=ScenarioConstraints)

    # 控制目标
    objective: ScenarioObjective = field(default_factory=ScenarioObjective)

    # 扰动定义
    disturbances: Dict[str, Any] = field(default_factory=dict)

    # 故障注入
    faults: Dict[str, Any] = field(default_factory=dict)

    # 持续时间 (s)
    duration: float = 3600.0

    # 风险等级 (1-5)
    risk_level: int = 1

    # 优先级
    priority: int = 1

    # MPC模型类型
    mpc_model_type: str = "LTV"  # LTV, NMPC, Robust

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'type': self.scenario_type.value,
            'domain': self.domain.value,
            'family': self.family.value,
            'name': self.name,
            'description': self.description,
            'risk_level': self.risk_level,
            'mpc_model_type': self.mpc_model_type
        }


# 场景注册表
SCENARIO_REGISTRY: Dict[ScenarioType, Scenario] = {}


def register_scenario(scenario: Scenario):
    """注册场景"""
    SCENARIO_REGISTRY[scenario.scenario_type] = scenario


# ============================================================
# 常态运行域场景定义
# ============================================================

# S1-A 双洞均分模式
register_scenario(Scenario(
    scenario_type=ScenarioType.S1_A_DUAL_BALANCED,
    domain=ScenarioDomain.NOMINAL,
    family=ScenarioFamily.S1_BALANCED,
    name="双洞均分模式",
    description="Q1 ≈ Q2, ΔQ ≤ 1%，最优水力工况",
    initial_conditions={
        'Q1': 132.5, 'Q2': 132.5,
        'H_inlet': 106.05, 'H_outlet': 104.79,
        'gate_1': 1.0, 'gate_2': 1.0
    },
    constraints=ScenarioConstraints(
        Q_min=100, Q_max=305,
        Q_imbalance_max=0.01
    ),
    objective=ScenarioObjective(
        objective_type="tracking",
        Q_target=265.0,
        weights={'flow_tracking': 1.0, 'flow_balance': 10.0, 'actuator_effort': 0.01}
    ),
    duration=86400,  # 24小时
    risk_level=1,
    mpc_model_type="LTV"
))

# S1-B 动态调峰模式
register_scenario(Scenario(
    scenario_type=ScenarioType.S1_B_DYNAMIC_PEAK,
    domain=ScenarioDomain.NOMINAL,
    family=ScenarioFamily.S1_BALANCED,
    name="动态调峰模式",
    description="流量在150~265 m³/s之间平滑波动",
    initial_conditions={
        'Q1': 100, 'Q2': 100,
        'H_inlet': 106.05, 'H_outlet': 104.79
    },
    constraints=ScenarioConstraints(
        Q_min=150, Q_max=265,
        dP_dt_max=3e4
    ),
    objective=ScenarioObjective(
        objective_type="tracking",
        Q_target=200.0,  # 动态变化
        weights={'flow_tracking': 1.0, 'pressure_smooth': 0.5}
    ),
    duration=7200,
    risk_level=1,
    mpc_model_type="LTV"
))

# S2-A 单洞排沙冲淤
register_scenario(Scenario(
    scenario_type=ScenarioType.S2_A_SEDIMENT_FLUSH,
    domain=ScenarioDomain.NOMINAL,
    family=ScenarioFamily.S2_MAINTENANCE,
    name="单洞排沙冲淤",
    description="流量集中于单洞，v > 3.0 m/s，清除淤积",
    initial_conditions={
        'Q1': 200, 'Q2': 50,
        'H_inlet': 107.0
    },
    constraints=ScenarioConstraints(
        Q_min=0, Q_max=200,
        Q_imbalance_max=0.8  # 允许较大不平衡
    ),
    objective=ScenarioObjective(
        objective_type="regulation",
        Q_target=200.0,
        weights={'flow_tracking': 1.0, 'flow_balance': 0.0}  # 不要求平衡
    ),
    duration=3600,
    risk_level=2,
    mpc_model_type="LTV"
))

# ============================================================
# 过渡运维域场景定义
# ============================================================

# S3-A 充水排气
register_scenario(Scenario(
    scenario_type=ScenarioType.S3_A_FILLING,
    domain=ScenarioDomain.TRANSITION,
    family=ScenarioFamily.S3_TWO_PHASE,
    name="充水排气",
    description="空管充水过程，控制气囊风险",
    initial_conditions={
        'Q1': 0, 'Q2': 132.5,
        'water_level': 0,
        'air_volume': 38.48 * 4250  # 空管体积
    },
    constraints=ScenarioConstraints(
        H_rate_max=0.5,  # 关键: 水位上升速率 ≤ 0.5 m/min
        P_min=0,
        dP_dt_max=1e4
    ),
    objective=ScenarioObjective(
        objective_type="regulation",
        weights={'pressure_rate': 10.0, 'air_pressure': 5.0}
    ),
    disturbances={
        'air_pocket': True
    },
    duration=7200,
    risk_level=4,  # 高风险
    mpc_model_type="NMPC"  # 需要非线性模型
))

# S3-B 停水排空
register_scenario(Scenario(
    scenario_type=ScenarioType.S3_B_DRAINING,
    domain=ScenarioDomain.TRANSITION,
    family=ScenarioFamily.S3_TWO_PHASE,
    name="停水排空",
    description="检修前排空，防止外压压溃",
    initial_conditions={
        'Q1': 132.5, 'Q2': 0,
        'water_level': 7.0
    },
    constraints=ScenarioConstraints(
        P_internal_min=1e4,  # 关键: 内压必须大于外压
        H_rate_max=0.3
    ),
    objective=ScenarioObjective(
        objective_type="emergency",
        weights={'pressure_margin': 10.0}
    ),
    duration=10800,
    risk_level=4,
    mpc_model_type="NMPC"
))

# S4-A 不停水倒洞
register_scenario(Scenario(
    scenario_type=ScenarioType.S4_A_SWITCH_TUNNEL,
    domain=ScenarioDomain.TRANSITION,
    family=ScenarioFamily.S4_TOPOLOGY,
    name="不停水倒洞",
    description="双洞切换为单洞，总流量波动 < ±5%",
    initial_conditions={
        'Q1': 132.5, 'Q2': 132.5,
        'gate_1': 1.0, 'gate_2': 1.0
    },
    constraints=ScenarioConstraints(
        Q_imbalance_max=0.5,
        dP_dt_max=2e4,
        gate_rate_max=0.005
    ),
    objective=ScenarioObjective(
        objective_type="tracking",
        Q_target=265.0,
        weights={'flow_tracking': 5.0, 'pressure_smooth': 2.0, 'actuator_effort': 0.1}
    ),
    duration=1800,
    risk_level=3,
    mpc_model_type="LTV"
))

# ============================================================
# 应急灾害域场景定义
# ============================================================

# S5-A 内衬渗漏
register_scenario(Scenario(
    scenario_type=ScenarioType.S5_A_INNER_LEAK,
    domain=ScenarioDomain.EMERGENCY,
    family=ScenarioFamily.S5_STRUCTURAL,
    name="内衬渗漏",
    description="管内清水漏入环形空腔",
    initial_conditions={
        'Q1': 132.5, 'Q2': 132.5,
        'leak_rate': 0.1,
        'leak_position': 2150
    },
    faults={
        'leakage': {
            'type': 'inner',
            'position': 2150,
            'rate': 0.1,
            'growth_rate': 0.001
        }
    },
    constraints=ScenarioConstraints(
        P_internal_min=5e4  # 建立压差封堵
    ),
    objective=ScenarioObjective(
        objective_type="emergency",
        weights={'leak_mitigation': 10.0, 'pressure_control': 5.0}
    ),
    duration=300,
    risk_level=5,
    mpc_model_type="Robust"
))

# S5-B 外衬入侵
register_scenario(Scenario(
    scenario_type=ScenarioType.S5_B_OUTER_INTRUSION,
    domain=ScenarioDomain.EMERGENCY,
    family=ScenarioFamily.S5_STRUCTURAL,
    name="外衬入侵",
    description="黄河浑水漏入环形空腔",
    initial_conditions={
        'Q1': 132.5, 'Q2': 132.5,
        'leak_rate': 0.05,
        'cavity_turbidity': 100
    },
    faults={
        'leakage': {
            'type': 'outer',
            'position': 2000
        }
    },
    duration=300,
    risk_level=5,
    mpc_model_type="Robust"
))

# S6-A 地震液化上浮
register_scenario(Scenario(
    scenario_type=ScenarioType.S6_A_LIQUEFACTION,
    domain=ScenarioDomain.EMERGENCY,
    family=ScenarioFamily.S6_GEOLOGICAL,
    name="地震液化上浮",
    description="地震导致土壤液化，隧道受浮力上抬",
    initial_conditions={
        'Q1': 132.5, 'Q2': 132.5,
        'liquefaction_index': 0
    },
    disturbances={
        'earthquake': {
            'pga': 0.2,  # 0.2g
            'magnitude': 6.5,
            'duration': 30
        }
    },
    constraints=ScenarioConstraints(
        P_max=1.5e6  # 允许超压压重
    ),
    objective=ScenarioObjective(
        objective_type="emergency",
        weights={'anti_float': 10.0, 'structural_safety': 10.0}
    ),
    duration=7200,
    risk_level=5,
    mpc_model_type="Robust"
))

# S6-B 进口吸气漩涡
register_scenario(Scenario(
    scenario_type=ScenarioType.S6_B_INTAKE_VORTEX,
    domain=ScenarioDomain.EMERGENCY,
    family=ScenarioFamily.S6_GEOLOGICAL,
    name="进口吸气漩涡",
    description="低水位大流量导致吸气漩涡",
    initial_conditions={
        'Q1': 150, 'Q2': 150,
        'H_inlet': 104.0  # 低水位
    },
    constraints=ScenarioConstraints(
        Q_max=250  # 限制流量
    ),
    duration=600,
    risk_level=4,
    mpc_model_type="NMPC"
))

# S7-A 爆管/断流
register_scenario(Scenario(
    scenario_type=ScenarioType.S7_A_PIPE_BURST,
    domain=ScenarioDomain.EMERGENCY,
    family=ScenarioFamily.S7_SYSTEM,
    name="爆管断流",
    description="隧道破裂，流量严重失衡",
    initial_conditions={
        'Q1': 50, 'Q2': 132.5
    },
    faults={
        'burst': {
            'tunnel': 1,
            'position': 2000,
            'severity': 0.8
        }
    },
    duration=120,
    risk_level=5,
    mpc_model_type="Robust"
))

# S7-B 闸门非同步故障
register_scenario(Scenario(
    scenario_type=ScenarioType.S7_B_GATE_ASYNC,
    domain=ScenarioDomain.EMERGENCY,
    family=ScenarioFamily.S7_SYSTEM,
    name="闸门非同步故障",
    description="双洞闸门开度不一致导致严重偏流",
    initial_conditions={
        'gate_1': 0.8, 'gate_2': 0.3
    },
    faults={
        'gate_stuck': {
            'gate': 2,
            'position': 0.3
        }
    },
    constraints=ScenarioConstraints(
        Q_imbalance_max=0.3
    ),
    duration=300,
    risk_level=4,
    mpc_model_type="LTV"
))
