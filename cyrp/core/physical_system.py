"""
Integrated Physical System for Yellow River Crossing Project.
穿黄工程物理系统集成模型

整合水力学模型与结构动力学模型，实现多物理场耦合仿真
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
from enum import Enum
import numpy as np
from datetime import datetime

from cyrp.core.hydraulic_model import HydraulicModel, HydraulicState, FlowRegime
from cyrp.core.structural_model import StructuralModel, StructuralState, StructuralCondition, FailureMode
from cyrp.core.parameters import TunnelParameters, EnvironmentParameters, SystemLimits, GateParameters


class SystemMode(Enum):
    """系统运行模式"""
    NORMAL = "normal"  # 常态运行
    TRANSITION = "transition"  # 过渡态
    MAINTENANCE = "maintenance"  # 维护模式
    EMERGENCY = "emergency"  # 应急模式
    SHUTDOWN = "shutdown"  # 停机


@dataclass
class ActuatorState:
    """执行机构状态"""
    # 进口闸门
    gate_inlet_1: float = 1.0  # 1#洞进口闸门开度 (0-1)
    gate_inlet_2: float = 1.0  # 2#洞进口闸门开度 (0-1)

    # 出口闸门
    gate_outlet_1: float = 1.0  # 1#洞出口闸门开度 (0-1)
    gate_outlet_2: float = 1.0  # 2#洞出口闸门开度 (0-1)

    # 阀门
    valve_fill_1: float = 0.0  # 充水阀 (0-1)
    valve_fill_2: float = 0.0
    valve_drain_1: float = 0.0  # 排水阀 (0-1)
    valve_drain_2: float = 0.0
    valve_vent_1: float = 0.0  # 排气阀 (0-1)
    valve_vent_2: float = 0.0
    valve_vacuum_break: float = 0.0  # 真空破坏阀

    # 泵站
    pump_drain_speed: float = 0.0  # 排水泵转速 (0-1)

    def to_control_vector(self) -> np.ndarray:
        """转换为控制向量"""
        return np.array([
            self.gate_inlet_1, self.gate_inlet_2,
            self.gate_outlet_1, self.gate_outlet_2,
            self.valve_fill_1, self.valve_fill_2,
            self.valve_drain_1, self.valve_drain_2,
            self.valve_vent_1, self.valve_vent_2,
            self.valve_vacuum_break,
            self.pump_drain_speed
        ])


@dataclass
class ControlCommand:
    """控制指令"""
    # 目标开度
    gate_inlet_1_target: float = 1.0
    gate_inlet_2_target: float = 1.0
    gate_outlet_1_target: float = 1.0
    gate_outlet_2_target: float = 1.0

    # 阀门指令
    valve_fill_1_cmd: float = 0.0
    valve_fill_2_cmd: float = 0.0
    valve_vent_cmd: float = 0.0

    # 泵站指令
    pump_drain_cmd: float = 0.0

    # 执行速率
    gate_rate: float = 0.002  # 闸门动作速率 (1/s)

    # 时间戳
    timestamp: float = 0.0


@dataclass
class SystemState:
    """
    系统完整状态

    整合水力状态、结构状态、执行机构状态
    """
    time: float = 0.0
    timestamp: str = ""

    # 子系统状态
    hydraulic: HydraulicState = field(default_factory=HydraulicState)
    structural: StructuralState = field(default_factory=StructuralState)
    actuators: ActuatorState = field(default_factory=ActuatorState)

    # 系统模式
    mode: SystemMode = SystemMode.NORMAL

    # 报警标志
    alarms: Dict[str, bool] = field(default_factory=dict)

    # 累计运行参数
    total_water_transferred: float = 0.0  # 累计输水量 (m³)
    operation_hours: float = 0.0  # 运行小时数

    def to_observation(self) -> np.ndarray:
        """转换为观测向量 (用于RL/MPC)"""
        return np.concatenate([
            self.hydraulic.to_vector(),
            self.structural.to_vector(),
            self.actuators.to_control_vector()
        ])


class PhysicalSystem:
    """
    穿黄工程物理系统

    实现:
    1. 多物理场耦合仿真
    2. 执行机构动力学
    3. 故障注入
    4. 状态监测
    """

    def __init__(
        self,
        tunnel_params: Optional[TunnelParameters] = None,
        env_params: Optional[EnvironmentParameters] = None,
        limits: Optional[SystemLimits] = None
    ):
        """
        初始化物理系统

        Args:
            tunnel_params: 隧洞参数
            env_params: 环境参数
            limits: 系统限值
        """
        self.tunnel_params = tunnel_params or TunnelParameters()
        self.env_params = env_params or EnvironmentParameters()
        self.limits = limits or SystemLimits()

        # 初始化子模型
        self.hydraulic_model = HydraulicModel(
            length=self.tunnel_params.length,
            diameter=self.tunnel_params.inner_diameter,
            manning_n=self.tunnel_params.manning_coefficient
        )

        self.structural_model = StructuralModel(
            inner_diameter=self.tunnel_params.inner_diameter,
            outer_diameter=self.tunnel_params.outer_diameter,
            inner_thickness=self.tunnel_params.wall_thickness_inner,
            outer_thickness=self.tunnel_params.wall_thickness_outer,
            length=self.tunnel_params.length,
            burial_depth=self.env_params.burial_depth
        )

        # 初始化状态
        self.state = SystemState(
            timestamp=datetime.now().isoformat()
        )

        # 故障注入器
        self.fault_injector = FaultInjector()

        # 仿真参数
        self.dt = 0.1  # 默认时间步长 (s)

        # 历史记录
        self.history: List[SystemState] = []

    def reset(
        self,
        initial_flow: float = 265.0,
        H_inlet: float = 106.05,
        H_outlet: float = 104.79
    ) -> SystemState:
        """
        重置系统到初始状态

        Args:
            initial_flow: 初始流量 (m³/s)
            H_inlet: 进口水位 (m)
            H_outlet: 出口水位 (m)

        Returns:
            初始系统状态
        """
        # 计算稳态
        hydraulic_state = self.hydraulic_model.get_steady_state(H_inlet, H_outlet)
        hydraulic_state.Q1 = initial_flow / 2
        hydraulic_state.Q2 = initial_flow / 2

        # 初始结构状态
        P_internal = self.hydraulic_model.compute_pressure(H_inlet, 85.0)
        P_external = self.env_params.get_external_water_pressure(85.0)

        structural_state = StructuralState()
        structural_state.stress_hoop_inner = self.structural_model.compute_hoop_stress(
            P_internal, P_external, "inner"
        )

        # 初始执行机构状态
        actuator_state = ActuatorState()

        # 组装系统状态
        self.state = SystemState(
            time=0.0,
            timestamp=datetime.now().isoformat(),
            hydraulic=hydraulic_state,
            structural=structural_state,
            actuators=actuator_state,
            mode=SystemMode.NORMAL
        )

        self.history = [self.state]
        return self.state

    def _update_actuators(
        self,
        command: ControlCommand,
        dt: float
    ) -> ActuatorState:
        """
        更新执行机构状态 (含动力学)

        Args:
            command: 控制指令
            dt: 时间步长

        Returns:
            新执行机构状态
        """
        current = self.state.actuators
        rate = command.gate_rate

        # 闸门一阶惯性模型
        def actuator_dynamics(current_val, target_val, rate, dt):
            delta = target_val - current_val
            max_delta = rate * dt
            if abs(delta) <= max_delta:
                return target_val
            return current_val + np.sign(delta) * max_delta

        new_state = ActuatorState(
            gate_inlet_1=actuator_dynamics(
                current.gate_inlet_1, command.gate_inlet_1_target, rate, dt
            ),
            gate_inlet_2=actuator_dynamics(
                current.gate_inlet_2, command.gate_inlet_2_target, rate, dt
            ),
            gate_outlet_1=actuator_dynamics(
                current.gate_outlet_1, command.gate_outlet_1_target, rate, dt
            ),
            gate_outlet_2=actuator_dynamics(
                current.gate_outlet_2, command.gate_outlet_2_target, rate, dt
            ),
            valve_fill_1=command.valve_fill_1_cmd,
            valve_fill_2=command.valve_fill_2_cmd,
            valve_vent_1=command.valve_vent_cmd,
            valve_vent_2=command.valve_vent_cmd,
            pump_drain_speed=command.pump_drain_cmd
        )

        return new_state

    def _check_alarms(self, state: SystemState) -> Dict[str, bool]:
        """检查报警条件"""
        alarms = {}

        # 真空报警
        alarms['vacuum'] = state.hydraulic.P_min < self.limits.min_internal_pressure

        # 超压报警
        alarms['overpressure'] = state.hydraulic.P_max > self.limits.max_internal_pressure

        # 流量不平衡报警
        alarms['flow_imbalance'] = state.hydraulic.flow_imbalance > self.limits.max_flow_imbalance

        # 水锤报警
        alarms['water_hammer'] = state.hydraulic.P_hammer > self.limits.max_water_hammer_pressure

        # 结构报警
        alarms['structural_warning'] = state.structural.safety_factor < 1.5
        alarms['structural_critical'] = state.structural.safety_factor < 1.0

        # 渗漏报警
        alarms['leakage'] = state.structural.leakage_rate > 0.01

        # 液化报警
        alarms['liquefaction'] = state.structural.liquefaction_index > 0.5

        return alarms

    def step(
        self,
        command: ControlCommand,
        dt: Optional[float] = None,
        disturbance: Optional[Dict[str, float]] = None
    ) -> SystemState:
        """
        系统单步仿真

        Args:
            command: 控制指令
            dt: 时间步长 (s)
            disturbance: 外部扰动 (地震、水位变化等)

        Returns:
            新系统状态
        """
        dt = dt or self.dt
        disturbance = disturbance or {}

        # 1. 更新执行机构
        new_actuators = self._update_actuators(command, dt)

        # 2. 故障注入
        faults = self.fault_injector.inject(self.state.time)

        # 3. 水力学仿真
        control = np.array([
            new_actuators.gate_inlet_1,
            new_actuators.gate_inlet_2
        ])

        new_hydraulic = self.hydraulic_model.step(
            self.state.hydraulic,
            control,
            dt
        )

        # 应用扰动
        if 'H_inlet_delta' in disturbance:
            new_hydraulic.H_inlet += disturbance['H_inlet_delta']
        if 'H_outlet_delta' in disturbance:
            new_hydraulic.H_outlet += disturbance['H_outlet_delta']

        # 应用故障
        if 'leakage' in faults:
            leak_rate = faults['leakage']['rate']
            new_hydraulic.Q1 -= leak_rate / 2
            new_hydraulic.Q2 -= leak_rate / 2

        # 4. 结构仿真
        P_internal = self.hydraulic_model.compute_pressure(
            new_hydraulic.H_inlet, 85.0
        )
        P_external = self.env_params.get_external_water_pressure(85.0)

        ground_motion = disturbance.get('ground_acceleration', 0.0)
        if 'earthquake' in faults:
            ground_motion = faults['earthquake']['pga'] * 9.81

        new_structural = self.structural_model.step(
            self.state.structural,
            P_internal,
            P_external,
            ground_motion,
            dt
        )

        # 5. 组装新状态
        new_time = self.state.time + dt
        new_state = SystemState(
            time=new_time,
            timestamp=datetime.now().isoformat(),
            hydraulic=new_hydraulic,
            structural=new_structural,
            actuators=new_actuators,
            mode=self.state.mode,
            total_water_transferred=self.state.total_water_transferred + \
                                    new_hydraulic.total_flow * dt,
            operation_hours=self.state.operation_hours + dt / 3600
        )

        # 6. 检查报警
        new_state.alarms = self._check_alarms(new_state)

        # 7. 更新模式
        if any([new_state.alarms.get('structural_critical', False),
                new_state.alarms.get('leakage', False),
                new_state.alarms.get('liquefaction', False)]):
            new_state.mode = SystemMode.EMERGENCY

        # 更新状态并记录
        self.state = new_state
        self.history.append(new_state)

        return new_state

    def simulate(
        self,
        commands: List[ControlCommand],
        duration: float,
        dt: float = 0.1,
        disturbance_schedule: Optional[Dict[float, Dict[str, float]]] = None
    ) -> List[SystemState]:
        """
        运行仿真

        Args:
            commands: 控制指令序列
            duration: 仿真时长 (s)
            dt: 时间步长 (s)
            disturbance_schedule: 扰动时间表

        Returns:
            状态历史
        """
        disturbance_schedule = disturbance_schedule or {}
        num_steps = int(duration / dt)

        for i in range(num_steps):
            t = i * dt
            cmd = commands[min(i, len(commands) - 1)]

            # 查找当前扰动
            disturbance = {}
            for t_disturb, disturb_data in disturbance_schedule.items():
                if t >= t_disturb:
                    disturbance = disturb_data

            self.step(cmd, dt, disturbance)

        return self.history

    def get_observation(self) -> np.ndarray:
        """获取当前观测向量"""
        return self.state.to_observation()

    def get_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        return {
            'time': self.state.time,
            'mode': self.state.mode.value,
            'alarms': self.state.alarms,
            'total_flow': self.state.hydraulic.total_flow,
            'safety_factor': self.state.structural.safety_factor,
            'water_transferred': self.state.total_water_transferred
        }


class FaultInjector:
    """
    故障注入器

    用于在仿真中注入各类故障，测试系统鲁棒性
    """

    def __init__(self):
        self.fault_schedule: Dict[str, Dict] = {}
        self.active_faults: Dict[str, Dict] = {}

    def schedule_fault(
        self,
        fault_type: str,
        start_time: float,
        duration: float,
        parameters: Dict[str, float]
    ):
        """
        计划故障注入

        Args:
            fault_type: 故障类型 ('leakage', 'earthquake', 'gate_stuck', ...)
            start_time: 开始时间 (s)
            duration: 持续时间 (s)
            parameters: 故障参数
        """
        self.fault_schedule[fault_type] = {
            'start': start_time,
            'end': start_time + duration,
            'params': parameters
        }

    def inject(self, current_time: float) -> Dict[str, Dict]:
        """
        注入故障

        Args:
            current_time: 当前时间 (s)

        Returns:
            当前活动的故障
        """
        active = {}
        for fault_type, schedule in self.fault_schedule.items():
            if schedule['start'] <= current_time <= schedule['end']:
                active[fault_type] = schedule['params']

        self.active_faults = active
        return active

    def clear_all(self):
        """清除所有故障"""
        self.fault_schedule = {}
        self.active_faults = {}
