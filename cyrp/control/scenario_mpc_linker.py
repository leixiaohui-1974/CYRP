"""
场景-MPC自动联动系统 - Scenario-MPC Auto-Linking System

实现场景识别后自动更新MPC目标函数、约束和参数
Implements automatic MPC objective/constraint update based on scenario detection
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum

from cyrp.core.real_parameters import (
    REAL_TUNNEL_PARAMS,
    REAL_SCENARIO_THRESHOLDS,
    REAL_CONTROL_PARAMS
)


@dataclass
class MPCObjective:
    """MPC目标函数配置"""
    # 状态权重矩阵对角元素
    Q_flow: float = 100.0           # 流量跟踪权重
    Q_pressure: float = 50.0        # 压力跟踪权重
    Q_velocity: float = 20.0        # 流速权重
    Q_asymmetric: float = 200.0     # 双洞不对称惩罚

    # 控制权重
    R_valve: float = 1.0            # 阀门控制权重
    R_pump: float = 10.0            # 泵控制权重

    # 变化率权重
    R_delta_valve: float = 10.0     # 阀门变化率权重

    # 终端权重倍数
    Qf_multiplier: float = 10.0

    # 松弛变量惩罚
    slack_penalty: float = 1000.0


@dataclass
class MPCConstraints:
    """MPC约束配置"""
    # 状态约束
    flow_min: float = 0.0
    flow_max: float = 320.0
    pressure_min: float = -0.05e6
    pressure_max: float = 1.0e6
    velocity_max: float = 5.0

    # 输入约束
    valve_min: float = 0.0
    valve_max: float = 1.0
    valve_rate_max: float = 0.01    # 1%/s

    # 软约束边界
    flow_soft_margin: float = 10.0
    pressure_soft_margin: float = 0.05e6

    # 安全约束
    asymmetric_limit: float = 0.1   # 10%不对称
    min_pressure_margin: float = 0.02e6  # 最小正压裕度


@dataclass
class MPCSetpoints:
    """MPC设定值"""
    flow_setpoint: float = 265.0
    pressure_setpoint: float = 0.5e6
    north_flow_ratio: float = 0.5    # 北洞流量占比
    south_flow_ratio: float = 0.5    # 南洞流量占比


@dataclass
class ScenarioMPCConfig:
    """场景对应的MPC完整配置"""
    scenario_id: str
    objective: MPCObjective
    constraints: MPCConstraints
    setpoints: MPCSetpoints
    prediction_horizon: int = 30
    control_horizon: int = 10
    sampling_time: float = 1.0
    priority: str = "normal"         # normal, high, critical
    description: str = ""


class ScenarioMPCLinker:
    """
    场景-MPC自动联动器

    功能:
    1. 根据场景ID自动生成MPC配置
    2. 实现配置的平滑切换
    3. 动态调整目标函数和约束
    4. 提供配置验证和监控
    """

    def __init__(self):
        # 场景配置库
        self.scenario_configs: Dict[str, ScenarioMPCConfig] = {}
        self._init_scenario_configs()

        # 当前配置
        self.current_config: Optional[ScenarioMPCConfig] = None
        self.current_scenario: str = "S2-A"

        # 切换参数
        self.transition_rate: float = 0.1  # 10%/step
        self.transition_in_progress: bool = False
        self.target_config: Optional[ScenarioMPCConfig] = None

        # 配置历史
        self.config_history: List[Dict[str, Any]] = []

        # 回调函数
        self.on_config_change: Optional[Callable] = None

    def _init_scenario_configs(self):
        """初始化所有场景的MPC配置"""

        # ===== 域1: 常规运行 =====
        # S1-A: 单洞高流量
        self.scenario_configs['S1-A'] = ScenarioMPCConfig(
            scenario_id='S1-A',
            objective=MPCObjective(
                Q_flow=150.0,      # 高流量需要更精确跟踪
                Q_pressure=60.0,
                Q_velocity=30.0,
                Q_asymmetric=0.0,   # 单洞运行无需不对称惩罚
                R_valve=1.0,
                R_delta_valve=15.0  # 限制变化率
            ),
            constraints=MPCConstraints(
                flow_max=320.0,
                velocity_max=4.5,
                valve_rate_max=0.008  # 更严格的变化率
            ),
            setpoints=MPCSetpoints(
                flow_setpoint=280.0,
                north_flow_ratio=1.0,
                south_flow_ratio=0.0
            ),
            prediction_horizon=30,
            priority="normal",
            description="单洞高流量运行"
        )

        # S1-B: 单洞中流量
        self.scenario_configs['S1-B'] = ScenarioMPCConfig(
            scenario_id='S1-B',
            objective=MPCObjective(
                Q_flow=100.0,
                Q_pressure=50.0,
                Q_velocity=20.0,
                R_valve=1.0
            ),
            constraints=MPCConstraints(
                flow_max=250.0,
                velocity_max=4.0
            ),
            setpoints=MPCSetpoints(
                flow_setpoint=200.0,
                north_flow_ratio=1.0,
                south_flow_ratio=0.0
            ),
            prediction_horizon=25,
            priority="normal",
            description="单洞中流量运行"
        )

        # S2-A: 双洞均衡运行
        self.scenario_configs['S2-A'] = ScenarioMPCConfig(
            scenario_id='S2-A',
            objective=MPCObjective(
                Q_flow=100.0,
                Q_pressure=50.0,
                Q_velocity=20.0,
                Q_asymmetric=200.0,  # 强调双洞均衡
                R_valve=1.0,
                R_delta_valve=10.0
            ),
            constraints=MPCConstraints(
                asymmetric_limit=0.1
            ),
            setpoints=MPCSetpoints(
                flow_setpoint=265.0,
                north_flow_ratio=0.5,
                south_flow_ratio=0.5
            ),
            prediction_horizon=30,
            priority="normal",
            description="双洞均衡运行"
        )

        # ===== 域2: 过渡运行 =====
        # S3-A: 计划切换
        self.scenario_configs['S3-A'] = ScenarioMPCConfig(
            scenario_id='S3-A',
            objective=MPCObjective(
                Q_flow=80.0,         # 降低流量跟踪权重
                Q_pressure=100.0,    # 增加压力控制权重
                Q_velocity=50.0,     # 关注流速变化
                Q_asymmetric=50.0,   # 允许适度不对称
                R_valve=0.5,         # 放宽控制
                R_delta_valve=5.0    # 允许较快变化
            ),
            constraints=MPCConstraints(
                valve_rate_max=0.02,  # 允许更快阀门动作
                asymmetric_limit=0.3   # 切换时允许不对称
            ),
            setpoints=MPCSetpoints(
                flow_setpoint=200.0,
                north_flow_ratio=0.3,
                south_flow_ratio=0.7
            ),
            prediction_horizon=20,
            priority="high",
            description="计划隧道切换"
        )

        # S3-B: 紧急切换
        self.scenario_configs['S3-B'] = ScenarioMPCConfig(
            scenario_id='S3-B',
            objective=MPCObjective(
                Q_flow=50.0,
                Q_pressure=200.0,    # 最高优先级
                Q_velocity=100.0,
                Q_asymmetric=20.0,   # 不对称是预期的
                R_valve=0.2,         # 最小化控制限制
                R_delta_valve=1.0,
                slack_penalty=500.0  # 降低软约束惩罚
            ),
            constraints=MPCConstraints(
                valve_rate_max=0.05,  # 紧急情况快速动作
                asymmetric_limit=0.5,
                flow_soft_margin=30.0
            ),
            setpoints=MPCSetpoints(
                flow_setpoint=150.0
            ),
            prediction_horizon=10,
            control_horizon=5,
            priority="critical",
            description="紧急隧道切换"
        )

        # S4-A: 检修模式
        self.scenario_configs['S4-A'] = ScenarioMPCConfig(
            scenario_id='S4-A',
            objective=MPCObjective(
                Q_flow=50.0,
                Q_pressure=30.0,
                R_valve=2.0,          # 保守控制
                R_delta_valve=20.0    # 缓慢变化
            ),
            constraints=MPCConstraints(
                flow_max=150.0,
                valve_rate_max=0.005
            ),
            setpoints=MPCSetpoints(
                flow_setpoint=100.0
            ),
            prediction_horizon=40,
            priority="normal",
            description="检修模式运行"
        )

        # ===== 域3: 应急运行 =====
        # S5-A: 轻微渗漏
        self.scenario_configs['S5-A'] = ScenarioMPCConfig(
            scenario_id='S5-A',
            objective=MPCObjective(
                Q_flow=120.0,
                Q_pressure=80.0,     # 增加压力关注
                Q_velocity=40.0,
                R_valve=0.8
            ),
            constraints=MPCConstraints(
                pressure_max=0.9e6   # 降低最大压力
            ),
            setpoints=MPCSetpoints(
                flow_setpoint=250.0
            ),
            prediction_horizon=25,
            priority="high",
            description="轻微渗漏响应"
        )

        # S5-B: 中度渗漏
        self.scenario_configs['S5-B'] = ScenarioMPCConfig(
            scenario_id='S5-B',
            objective=MPCObjective(
                Q_flow=80.0,
                Q_pressure=150.0,
                Q_velocity=60.0,
                R_valve=0.5,
                R_delta_valve=3.0
            ),
            constraints=MPCConstraints(
                pressure_max=0.8e6,
                valve_rate_max=0.03
            ),
            setpoints=MPCSetpoints(
                flow_setpoint=200.0
            ),
            prediction_horizon=15,
            priority="critical",
            description="中度渗漏响应"
        )

        # S5-C: 严重渗漏
        self.scenario_configs['S5-C'] = ScenarioMPCConfig(
            scenario_id='S5-C',
            objective=MPCObjective(
                Q_flow=30.0,         # 流量跟踪次要
                Q_pressure=300.0,    # 压力控制最重要
                Q_velocity=100.0,
                R_valve=0.1,         # 快速响应
                R_delta_valve=0.5,
                slack_penalty=200.0
            ),
            constraints=MPCConstraints(
                pressure_max=0.6e6,
                valve_rate_max=0.05
            ),
            setpoints=MPCSetpoints(
                flow_setpoint=150.0
            ),
            prediction_horizon=10,
            control_horizon=5,
            priority="critical",
            description="严重渗漏响应"
        )

        # S6-A: 地震VI度
        self.scenario_configs['S6-A'] = ScenarioMPCConfig(
            scenario_id='S6-A',
            objective=MPCObjective(
                Q_flow=100.0,
                Q_pressure=80.0,
                Q_velocity=60.0,
                R_valve=0.8
            ),
            constraints=MPCConstraints(
                velocity_max=4.0
            ),
            setpoints=MPCSetpoints(
                flow_setpoint=220.0
            ),
            prediction_horizon=20,
            priority="high",
            description="地震VI度响应"
        )

        # S6-B: 地震VII度
        self.scenario_configs['S6-B'] = ScenarioMPCConfig(
            scenario_id='S6-B',
            objective=MPCObjective(
                Q_flow=50.0,
                Q_pressure=150.0,
                Q_velocity=100.0,
                R_valve=0.3,
                R_delta_valve=1.0
            ),
            constraints=MPCConstraints(
                velocity_max=3.5,
                valve_rate_max=0.04
            ),
            setpoints=MPCSetpoints(
                flow_setpoint=150.0
            ),
            prediction_horizon=10,
            priority="critical",
            description="地震VII度响应"
        )

        # S6-C: 地震VIII度
        self.scenario_configs['S6-C'] = ScenarioMPCConfig(
            scenario_id='S6-C',
            objective=MPCObjective(
                Q_flow=20.0,
                Q_pressure=200.0,
                Q_velocity=150.0,
                R_valve=0.1,
                R_delta_valve=0.2,
                slack_penalty=100.0
            ),
            constraints=MPCConstraints(
                flow_max=150.0,
                velocity_max=3.0,
                valve_rate_max=0.06
            ),
            setpoints=MPCSetpoints(
                flow_setpoint=100.0
            ),
            prediction_horizon=5,
            control_horizon=3,
            priority="critical",
            description="地震VIII度响应"
        )

        # S7: 综合应急
        self.scenario_configs['S7'] = ScenarioMPCConfig(
            scenario_id='S7',
            objective=MPCObjective(
                Q_flow=10.0,
                Q_pressure=300.0,
                Q_velocity=200.0,
                R_valve=0.05,
                R_delta_valve=0.1,
                slack_penalty=50.0
            ),
            constraints=MPCConstraints(
                flow_max=100.0,
                pressure_max=0.5e6,
                velocity_max=2.5,
                valve_rate_max=0.1
            ),
            setpoints=MPCSetpoints(
                flow_setpoint=50.0
            ),
            prediction_horizon=5,
            control_horizon=2,
            priority="critical",
            description="综合应急响应"
        )

        # 设置默认配置
        self.current_config = self.scenario_configs['S2-A']

    def update_scenario(self, scenario_id: str, smooth: bool = True) -> ScenarioMPCConfig:
        """
        更新场景配置

        Args:
            scenario_id: 新场景ID
            smooth: 是否平滑过渡

        Returns:
            新的MPC配置
        """
        if scenario_id not in self.scenario_configs:
            # 默认回退到S2-A
            scenario_id = 'S2-A'

        new_config = self.scenario_configs[scenario_id]

        if smooth and self.current_config is not None:
            self.target_config = new_config
            self.transition_in_progress = True
        else:
            self.current_config = new_config
            self.transition_in_progress = False

        self.current_scenario = scenario_id

        # 记录历史
        self.config_history.append({
            'scenario': scenario_id,
            'config': new_config,
            'smooth': smooth
        })

        # 回调通知
        if self.on_config_change:
            self.on_config_change(scenario_id, new_config)

        return new_config

    def step_transition(self) -> ScenarioMPCConfig:
        """
        执行一步平滑过渡

        Returns:
            当前（可能是过渡中的）配置
        """
        if not self.transition_in_progress or self.target_config is None:
            return self.current_config

        # 插值各参数
        rate = self.transition_rate

        # 目标函数插值
        curr_obj = self.current_config.objective
        tgt_obj = self.target_config.objective

        new_obj = MPCObjective(
            Q_flow=curr_obj.Q_flow + rate * (tgt_obj.Q_flow - curr_obj.Q_flow),
            Q_pressure=curr_obj.Q_pressure + rate * (tgt_obj.Q_pressure - curr_obj.Q_pressure),
            Q_velocity=curr_obj.Q_velocity + rate * (tgt_obj.Q_velocity - curr_obj.Q_velocity),
            Q_asymmetric=curr_obj.Q_asymmetric + rate * (tgt_obj.Q_asymmetric - curr_obj.Q_asymmetric),
            R_valve=curr_obj.R_valve + rate * (tgt_obj.R_valve - curr_obj.R_valve),
            R_delta_valve=curr_obj.R_delta_valve + rate * (tgt_obj.R_delta_valve - curr_obj.R_delta_valve),
        )

        # 设定值插值
        curr_sp = self.current_config.setpoints
        tgt_sp = self.target_config.setpoints

        new_sp = MPCSetpoints(
            flow_setpoint=curr_sp.flow_setpoint + rate * (tgt_sp.flow_setpoint - curr_sp.flow_setpoint),
            north_flow_ratio=curr_sp.north_flow_ratio + rate * (tgt_sp.north_flow_ratio - curr_sp.north_flow_ratio),
            south_flow_ratio=curr_sp.south_flow_ratio + rate * (tgt_sp.south_flow_ratio - curr_sp.south_flow_ratio),
        )

        # 预测时域插值
        new_horizon = int(self.current_config.prediction_horizon +
                         rate * (self.target_config.prediction_horizon - self.current_config.prediction_horizon))

        # 创建过渡配置
        transition_config = ScenarioMPCConfig(
            scenario_id=f"{self.current_scenario}->",
            objective=new_obj,
            constraints=self.target_config.constraints,  # 约束直接切换
            setpoints=new_sp,
            prediction_horizon=new_horizon,
            priority=self.target_config.priority,
            description="过渡中"
        )

        # 检查是否完成过渡
        if abs(new_obj.Q_flow - tgt_obj.Q_flow) < 1.0:
            self.current_config = self.target_config
            self.transition_in_progress = False
            self.target_config = None
            return self.current_config

        self.current_config = transition_config
        return transition_config

    def get_mpc_matrices(self) -> Dict[str, np.ndarray]:
        """
        获取MPC矩阵形式的配置

        Returns:
            包含Q, R, Qf等矩阵的字典
        """
        config = self.current_config
        obj = config.objective

        # 状态权重矩阵 (4x4: flow, pressure, velocity, asymmetric)
        Q = np.diag([obj.Q_flow, obj.Q_pressure, obj.Q_velocity, obj.Q_asymmetric])

        # 控制权重矩阵 (2x2: valve1, valve2)
        R = np.diag([obj.R_valve, obj.R_valve])

        # 控制增量权重
        R_delta = np.diag([obj.R_delta_valve, obj.R_delta_valve])

        # 终端权重
        Qf = Q * obj.Qf_multiplier

        return {
            'Q': Q,
            'R': R,
            'R_delta': R_delta,
            'Qf': Qf,
            'slack_penalty': obj.slack_penalty
        }

    def get_constraint_vectors(self) -> Dict[str, np.ndarray]:
        """
        获取约束向量

        Returns:
            约束上下界向量
        """
        cons = self.current_config.constraints

        return {
            'x_min': np.array([cons.flow_min, cons.pressure_min, 0, -cons.asymmetric_limit]),
            'x_max': np.array([cons.flow_max, cons.pressure_max, cons.velocity_max, cons.asymmetric_limit]),
            'u_min': np.array([cons.valve_min, cons.valve_min]),
            'u_max': np.array([cons.valve_max, cons.valve_max]),
            'du_max': np.array([cons.valve_rate_max, cons.valve_rate_max]),
        }

    def get_setpoint_vector(self) -> np.ndarray:
        """获取设定值向量"""
        sp = self.current_config.setpoints
        return np.array([
            sp.flow_setpoint,
            sp.pressure_setpoint,
            0,  # 目标流速 (由流量决定)
            0   # 目标不对称度
        ])

    def get_config_summary(self) -> Dict[str, Any]:
        """获取当前配置摘要"""
        config = self.current_config

        return {
            'scenario_id': config.scenario_id,
            'priority': config.priority,
            'description': config.description,
            'prediction_horizon': config.prediction_horizon,
            'control_horizon': config.control_horizon,
            'flow_setpoint': config.setpoints.flow_setpoint,
            'Q_flow': config.objective.Q_flow,
            'Q_pressure': config.objective.Q_pressure,
            'R_valve': config.objective.R_valve,
            'transition_in_progress': self.transition_in_progress
        }

    def validate_config(self, config: ScenarioMPCConfig) -> Tuple[bool, List[str]]:
        """验证配置有效性"""
        errors = []

        # 检查权重非负
        if config.objective.Q_flow < 0:
            errors.append("Q_flow must be non-negative")
        if config.objective.R_valve <= 0:
            errors.append("R_valve must be positive")

        # 检查约束一致性
        if config.constraints.flow_min > config.constraints.flow_max:
            errors.append("flow_min > flow_max")
        if config.constraints.valve_min > config.constraints.valve_max:
            errors.append("valve_min > valve_max")

        # 检查设定值在约束内
        if not (config.constraints.flow_min <= config.setpoints.flow_setpoint <= config.constraints.flow_max):
            errors.append("flow_setpoint out of bounds")

        # 检查预测时域
        if config.prediction_horizon < config.control_horizon:
            errors.append("prediction_horizon < control_horizon")

        return len(errors) == 0, errors
