"""
执行器仿真模型 - Actuator Simulation Models

实现阀门动力学、泵曲线、电机模型
Implements valve dynamics, pump curves, motor models
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod


class ActuatorStatus(Enum):
    """执行器状态"""
    READY = "ready"
    OPERATING = "operating"
    FAULT = "fault"
    MAINTENANCE = "maintenance"
    EMERGENCY_STOP = "emergency_stop"


class ActuatorFailureMode(Enum):
    """执行器故障模式"""
    NONE = "none"
    STUCK_OPEN = "stuck_open"
    STUCK_CLOSED = "stuck_closed"
    STUCK_POSITION = "stuck_position"
    SLOW_RESPONSE = "slow_response"
    HYSTERESIS = "hysteresis"
    LEAKAGE = "leakage"
    POWER_LOSS = "power_loss"
    MECHANICAL_WEAR = "mechanical_wear"


@dataclass
class ValveSpec:
    """阀门规格"""
    name: str
    nominal_diameter: float           # 公称直径 (m)
    cv: float                         # 流量系数
    stroke_time: float               # 全行程时间 (s)
    max_dp: float                    # 最大压差 (Pa)
    leakage_class: str               # 泄漏等级 (IV, V, VI)
    characteristic: str              # 特性曲线 (linear, equal_percentage, quick_opening)
    fail_position: str               # 故障位置 (open, closed, as_is)


@dataclass
class PumpSpec:
    """泵规格"""
    name: str
    rated_flow: float                # 额定流量 (m³/s)
    rated_head: float                # 额定扬程 (m)
    rated_speed: float               # 额定转速 (rpm)
    rated_power: float               # 额定功率 (kW)
    efficiency: float                # 额定效率
    npsh_required: float             # 必需汽蚀余量 (m)
    inertia: float                   # 转动惯量 (kg·m²)


class ActuatorModel(ABC):
    """执行器基类"""

    def __init__(self, name: str):
        self.name = name
        self.status = ActuatorStatus.READY
        self.failure_mode = ActuatorFailureMode.NONE

        # 位置/状态
        self.position = 0.0           # 0-1
        self.command = 0.0
        self.velocity = 0.0

        # 时间
        self.time = 0.0

        # 故障参数
        self.failure_params: Dict[str, Any] = {}

    @abstractmethod
    def step(self, command: float, dt: float) -> float:
        """执行一步"""
        pass

    @abstractmethod
    def get_output(self) -> Dict[str, float]:
        """获取输出"""
        pass

    def inject_failure(self, mode: ActuatorFailureMode,
                      params: Optional[Dict[str, Any]] = None):
        """注入故障"""
        self.failure_mode = mode
        self.failure_params = params or {}
        if mode != ActuatorFailureMode.NONE:
            self.status = ActuatorStatus.FAULT

    def clear_failure(self):
        """清除故障"""
        self.failure_mode = ActuatorFailureMode.NONE
        self.failure_params = {}
        self.status = ActuatorStatus.READY

    def emergency_stop(self):
        """紧急停止"""
        self.status = ActuatorStatus.EMERGENCY_STOP
        self.command = 0.0


class GateValveModel(ActuatorModel):
    """闸阀模型"""

    def __init__(self, spec: ValveSpec):
        super().__init__(spec.name)
        self.spec = spec

        # 动态参数
        self.position = 1.0  # 初始全开
        self.velocity_limit = 1.0 / spec.stroke_time

        # 摩擦和惯性
        self.friction = 0.05
        self.inertia_factor = 0.1

        # 流量特性
        self.characteristic = spec.characteristic

    def step(self, command: float, dt: float) -> float:
        """
        阀门动态响应

        Args:
            command: 开度指令 (0-1)
            dt: 时间步长

        Returns:
            实际开度
        """
        self.time += dt
        self.command = np.clip(command, 0, 1)

        # 故障处理
        if self.failure_mode == ActuatorFailureMode.STUCK_OPEN:
            self.position = 1.0
            return self.position
        elif self.failure_mode == ActuatorFailureMode.STUCK_CLOSED:
            self.position = 0.0
            return self.position
        elif self.failure_mode == ActuatorFailureMode.STUCK_POSITION:
            return self.position
        elif self.failure_mode == ActuatorFailureMode.POWER_LOSS:
            # 失电后返回安全位置
            if self.spec.fail_position == 'open':
                target = 1.0
            elif self.spec.fail_position == 'closed':
                target = 0.0
            else:
                return self.position
            self.command = target

        # 位置误差
        error = self.command - self.position

        # 速度限制
        max_velocity = self.velocity_limit
        if self.failure_mode == ActuatorFailureMode.SLOW_RESPONSE:
            max_velocity *= self.failure_params.get('factor', 0.5)

        # 计算速度
        if abs(error) > 0.001:
            desired_velocity = np.sign(error) * max_velocity
        else:
            desired_velocity = 0

        # 加速度限制 (惯性)
        max_accel = max_velocity / self.inertia_factor
        accel = np.clip(
            desired_velocity - self.velocity,
            -max_accel * dt,
            max_accel * dt
        )
        self.velocity += accel

        # 摩擦 (死区)
        if abs(error) < self.friction and abs(self.velocity) < 0.01:
            self.velocity = 0

        # 滞环
        if self.failure_mode == ActuatorFailureMode.HYSTERESIS:
            hysteresis = self.failure_params.get('width', 0.05)
            if abs(error) < hysteresis:
                self.velocity = 0

        # 更新位置
        self.position += self.velocity * dt
        self.position = np.clip(self.position, 0, 1)

        self.status = ActuatorStatus.OPERATING if self.velocity != 0 else ActuatorStatus.READY

        return self.position

    def get_flow_coefficient(self, opening: float) -> float:
        """获取流量系数"""
        if self.characteristic == 'linear':
            return self.spec.cv * opening
        elif self.characteristic == 'equal_percentage':
            R = 50  # 可调比
            return self.spec.cv * R ** (opening - 1)
        elif self.characteristic == 'quick_opening':
            return self.spec.cv * np.sqrt(opening)
        return self.spec.cv * opening

    def compute_flow(self, dp: float, rho: float = 1000.0) -> float:
        """
        计算流量

        Q = Cv * sqrt(dp / rho)

        Args:
            dp: 压差 (Pa)
            rho: 密度 (kg/m³)

        Returns:
            流量 (m³/s)
        """
        cv = self.get_flow_coefficient(self.position)

        # 泄漏
        leakage = 0
        if self.failure_mode == ActuatorFailureMode.LEAKAGE:
            leakage = self.failure_params.get('rate', 0.01) * self.spec.cv

        if self.position < 0.01:
            cv = leakage  # 关闭时仍有泄漏

        flow = cv * np.sqrt(abs(dp) / rho) * np.sign(dp)
        return flow

    def get_output(self) -> Dict[str, float]:
        """获取输出"""
        return {
            'position': self.position,
            'velocity': self.velocity,
            'command': self.command,
            'cv': self.get_flow_coefficient(self.position)
        }


class ButterflyValveModel(GateValveModel):
    """蝶阀模型"""

    def __init__(self, spec: ValveSpec):
        super().__init__(spec)
        # 蝶阀特有特性
        self.torque_curve = self._init_torque_curve()

    def _init_torque_curve(self) -> np.ndarray:
        """初始化力矩曲线"""
        # 典型蝶阀力矩曲线 (角度 vs 力矩系数)
        angles = np.linspace(0, 90, 91)  # 0-90度
        torque = np.zeros_like(angles)

        for i, angle in enumerate(angles):
            # 动力矩 + 摩擦力矩
            if angle < 10:
                torque[i] = 0.8 + angle * 0.02
            elif angle < 70:
                torque[i] = 1.0 + 0.5 * np.sin(np.radians(angle * 2))
            else:
                torque[i] = 1.5 - (90 - angle) * 0.02

        return torque

    def get_torque_requirement(self, dp: float) -> float:
        """获取力矩需求"""
        angle = self.position * 90  # 开度转角度
        idx = int(angle)
        torque_coef = self.torque_curve[min(idx, 90)]

        # 力矩 = 系数 * 压差 * 阀盘面积
        area = np.pi * (self.spec.nominal_diameter / 2) ** 2
        return torque_coef * dp * area * self.spec.nominal_diameter / 4


class PumpModel(ActuatorModel):
    """离心泵模型"""

    def __init__(self, spec: PumpSpec):
        super().__init__(spec.name)
        self.spec = spec

        # 运行状态
        self.speed = 0.0              # 当前转速 (rpm)
        self.flow = 0.0               # 当前流量 (m³/s)
        self.head = 0.0               # 当前扬程 (m)
        self.power = 0.0              # 当前功率 (kW)
        self.is_running = False

        # 泵曲线系数 (H = H0 - a*Q²)
        self.H0 = spec.rated_head * 1.2
        self.a = (self.H0 - spec.rated_head) / (spec.rated_flow ** 2)

        # 效率曲线系数
        self.eta_max = spec.efficiency
        self.Q_bep = spec.rated_flow  # 最佳效率点流量

        # 动态参数
        self.startup_time = 10.0      # 启动时间 (s)
        self.shutdown_time = 30.0     # 惯性停车时间 (s)

    def step(self, command: float, dt: float) -> float:
        """
        泵动态响应

        Args:
            command: 转速指令 (0-1, 相对于额定转速)
            dt: 时间步长

        Returns:
            实际转速比
        """
        self.time += dt
        self.command = np.clip(command, 0, 1)

        target_speed = self.command * self.spec.rated_speed

        # 故障处理
        if self.failure_mode == ActuatorFailureMode.POWER_LOSS:
            target_speed = 0

        # 启停动态
        if target_speed > self.speed:
            # 启动
            tau = self.startup_time / 3
            self.speed += (target_speed - self.speed) * (1 - np.exp(-dt / tau))
            self.is_running = True
        elif target_speed < self.speed:
            # 停机 (惯性)
            tau = self.shutdown_time / 3
            self.speed += (target_speed - self.speed) * (1 - np.exp(-dt / tau))
            if self.speed < 0.01 * self.spec.rated_speed:
                self.speed = 0
                self.is_running = False

        # 机械磨损影响
        if self.failure_mode == ActuatorFailureMode.MECHANICAL_WEAR:
            wear_factor = self.failure_params.get('factor', 0.9)
            self.speed *= wear_factor

        self.status = ActuatorStatus.OPERATING if self.is_running else ActuatorStatus.READY

        return self.speed / self.spec.rated_speed

    def compute_operating_point(self, system_head: float) -> Tuple[float, float, float]:
        """
        计算工作点

        Args:
            system_head: 系统扬程需求 (m)

        Returns:
            (流量, 扬程, 效率)
        """
        if not self.is_running or self.speed < 1:
            self.flow = 0
            self.head = 0
            self.power = 0
            return 0, 0, 0

        # 相似定律
        n_ratio = self.speed / self.spec.rated_speed
        H0_actual = self.H0 * n_ratio ** 2
        a_actual = self.a

        # 求解: H_pump = H_system
        # H0 - a*Q² = system_head
        Q_squared = (H0_actual - system_head) / a_actual
        if Q_squared < 0:
            # 扬程不足
            self.flow = 0
            self.head = H0_actual
        else:
            self.flow = np.sqrt(Q_squared) * n_ratio
            self.head = system_head

        # 效率计算
        eta = self._compute_efficiency(self.flow)

        # 功率计算
        if eta > 0:
            self.power = 9.81 * 1000 * self.flow * self.head / (1000 * eta)
        else:
            self.power = 0

        return self.flow, self.head, eta

    def _compute_efficiency(self, flow: float) -> float:
        """计算效率"""
        if flow <= 0:
            return 0

        # 效率曲线 (抛物线近似)
        q_ratio = flow / self.Q_bep
        eta = self.eta_max * (2 * q_ratio - q_ratio ** 2)

        # 磨损影响
        if self.failure_mode == ActuatorFailureMode.MECHANICAL_WEAR:
            eta *= self.failure_params.get('efficiency_factor', 0.9)

        return max(0, min(eta, 1))

    def check_cavitation(self, npsh_available: float) -> bool:
        """检查汽蚀"""
        return npsh_available < self.spec.npsh_required

    def get_output(self) -> Dict[str, float]:
        """获取输出"""
        return {
            'speed': self.speed,
            'speed_ratio': self.speed / self.spec.rated_speed if self.spec.rated_speed > 0 else 0,
            'flow': self.flow,
            'head': self.head,
            'power': self.power,
            'is_running': float(self.is_running)
        }


class MotorModel(ActuatorModel):
    """电机模型"""

    def __init__(self, name: str, rated_power: float = 100.0, rated_speed: float = 1500.0):
        super().__init__(name)

        # 电机参数
        self.rated_power = rated_power     # kW
        self.rated_speed = rated_speed     # rpm
        self.rated_torque = rated_power * 1000 * 60 / (2 * np.pi * rated_speed)

        # 状态
        self.speed = 0.0
        self.torque = 0.0
        self.current = 0.0
        self.temperature = 25.0

        # 动态参数
        self.inertia = 10.0               # kg·m²
        self.damping = 0.1
        self.thermal_constant = 600.0     # 热时间常数 (s)

        # 保护参数
        self.max_temperature = 120.0      # °C
        self.overcurrent_limit = 6.0      # 倍额定电流
        self.is_protected = False

    def step(self, command: float, load_torque: float, dt: float) -> float:
        """
        电机动态响应

        Args:
            command: 速度指令 (0-1)
            load_torque: 负载力矩 (N·m)
            dt: 时间步长

        Returns:
            实际转速比
        """
        self.time += dt
        self.command = np.clip(command, 0, 1)

        target_speed = self.command * self.rated_speed

        # 故障和保护检查
        if self.failure_mode == ActuatorFailureMode.POWER_LOSS or self.is_protected:
            target_speed = 0

        # 计算电磁力矩
        speed_error = target_speed - self.speed
        self.torque = self._compute_electromagnetic_torque(speed_error)

        # 运动方程
        alpha = (self.torque - load_torque - self.damping * self.speed) / self.inertia
        self.speed += alpha * dt

        if self.speed < 0:
            self.speed = 0

        # 电流计算
        self.current = abs(self.torque / self.rated_torque) * 1.0  # 简化

        # 温度计算
        power_loss = self.current ** 2 * 0.05  # 简化的损耗模型
        cooling = (self.temperature - 25.0) / self.thermal_constant
        self.temperature += (power_loss - cooling) * dt

        # 保护动作
        if self.temperature > self.max_temperature:
            self.is_protected = True
            self.status = ActuatorStatus.FAULT

        if self.current > self.overcurrent_limit:
            self.is_protected = True
            self.status = ActuatorStatus.FAULT

        return self.speed / self.rated_speed

    def _compute_electromagnetic_torque(self, speed_error: float) -> float:
        """计算电磁力矩"""
        # 简化的PI调速
        Kp = self.rated_torque / (0.1 * self.rated_speed)
        torque = Kp * speed_error

        # 力矩限制
        torque = np.clip(torque, -2 * self.rated_torque, 2 * self.rated_torque)

        return torque

    def reset_protection(self):
        """复位保护"""
        if self.temperature < self.max_temperature * 0.8:
            self.is_protected = False
            self.status = ActuatorStatus.READY

    def get_output(self) -> Dict[str, float]:
        """获取输出"""
        return {
            'speed': self.speed,
            'speed_ratio': self.speed / self.rated_speed,
            'torque': self.torque,
            'current': self.current,
            'temperature': self.temperature,
            'is_protected': float(self.is_protected)
        }


class ActuatorSystem:
    """执行器系统"""

    def __init__(self):
        self.actuators: Dict[str, ActuatorModel] = {}

        # 创建阀门
        self._create_valves()

        # 创建泵
        self._create_pumps()

    def _create_valves(self):
        """创建阀门"""
        # 北洞进口闸阀
        self.actuators['north_inlet'] = GateValveModel(ValveSpec(
            name="North_Inlet_Gate",
            nominal_diameter=7.0,
            cv=50000,
            stroke_time=120.0,
            max_dp=1.0e6,
            leakage_class="IV",
            characteristic="linear",
            fail_position="as_is"
        ))

        # 北洞出口闸阀
        self.actuators['north_outlet'] = GateValveModel(ValveSpec(
            name="North_Outlet_Gate",
            nominal_diameter=7.0,
            cv=50000,
            stroke_time=120.0,
            max_dp=1.0e6,
            leakage_class="IV",
            characteristic="linear",
            fail_position="as_is"
        ))

        # 南洞进口闸阀
        self.actuators['south_inlet'] = GateValveModel(ValveSpec(
            name="South_Inlet_Gate",
            nominal_diameter=7.0,
            cv=50000,
            stroke_time=120.0,
            max_dp=1.0e6,
            leakage_class="IV",
            characteristic="linear",
            fail_position="as_is"
        ))

        # 南洞出口闸阀
        self.actuators['south_outlet'] = GateValveModel(ValveSpec(
            name="South_Outlet_Gate",
            nominal_diameter=7.0,
            cv=50000,
            stroke_time=120.0,
            max_dp=1.0e6,
            leakage_class="IV",
            characteristic="linear",
            fail_position="as_is"
        ))

        # 调节蝶阀
        self.actuators['control_valve'] = ButterflyValveModel(ValveSpec(
            name="Control_Butterfly",
            nominal_diameter=3.0,
            cv=10000,
            stroke_time=30.0,
            max_dp=0.5e6,
            leakage_class="V",
            characteristic="equal_percentage",
            fail_position="as_is"
        ))

        # 紧急切断阀
        self.actuators['emergency_shutoff'] = GateValveModel(ValveSpec(
            name="Emergency_Shutoff",
            nominal_diameter=7.0,
            cv=50000,
            stroke_time=30.0,  # 快速关闭
            max_dp=1.5e6,
            leakage_class="VI",
            characteristic="linear",
            fail_position="closed"
        ))

    def _create_pumps(self):
        """创建泵"""
        # 排水泵
        self.actuators['drain_pump_1'] = PumpModel(PumpSpec(
            name="Drain_Pump_1",
            rated_flow=0.5,
            rated_head=50.0,
            rated_speed=1480,
            rated_power=45,
            efficiency=0.8,
            npsh_required=5.0,
            inertia=5.0
        ))

        self.actuators['drain_pump_2'] = PumpModel(PumpSpec(
            name="Drain_Pump_2",
            rated_flow=0.5,
            rated_head=50.0,
            rated_speed=1480,
            rated_power=45,
            efficiency=0.8,
            npsh_required=5.0,
            inertia=5.0
        ))

    def step_all(self, commands: Dict[str, float], dt: float,
                external_conditions: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, float]]:
        """
        步进所有执行器

        Args:
            commands: 各执行器指令
            dt: 时间步长
            external_conditions: 外部条件 (压差、负载等)

        Returns:
            所有执行器输出
        """
        outputs = {}
        external_conditions = external_conditions or {}

        for act_id, actuator in self.actuators.items():
            command = commands.get(act_id, 0.0)

            if isinstance(actuator, PumpModel):
                # 泵需要系统扬程
                actuator.step(command, dt)
                system_head = external_conditions.get(f'{act_id}_head', 30.0)
                actuator.compute_operating_point(system_head)
            elif isinstance(actuator, MotorModel):
                # 电机需要负载力矩
                load_torque = external_conditions.get(f'{act_id}_load', 0.0)
                actuator.step(command, load_torque, dt)
            else:
                actuator.step(command, dt)

            outputs[act_id] = actuator.get_output()

        return outputs

    def get_valve_positions(self) -> Dict[str, float]:
        """获取所有阀门位置"""
        return {
            act_id: actuator.position
            for act_id, actuator in self.actuators.items()
            if isinstance(actuator, GateValveModel)
        }

    def emergency_shutdown(self):
        """紧急停车"""
        for actuator in self.actuators.values():
            actuator.emergency_stop()

        # 紧急切断阀强制关闭
        if 'emergency_shutoff' in self.actuators:
            self.actuators['emergency_shutoff'].inject_failure(
                ActuatorFailureMode.NONE
            )
            self.actuators['emergency_shutoff'].command = 0.0

    def get_status_report(self) -> Dict[str, Dict[str, Any]]:
        """获取状态报告"""
        return {
            act_id: {
                'status': actuator.status.value,
                'failure_mode': actuator.failure_mode.value,
                **actuator.get_output()
            }
            for act_id, actuator in self.actuators.items()
        }
