"""
执行器仿真系统 - Actuator Simulation System

实现完整的执行器仿真功能，包括：
- 多类型执行器动力学仿真
- 阀门、泵、电机等设备模型
- 故障模式仿真
- 执行器响应和性能分析

Implements complete actuator simulation including:
- Multi-type actuator dynamics simulation
- Valve, pump, motor device models
- Failure mode simulation
- Actuator response and performance analysis
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from collections import deque
import threading
from concurrent.futures import ThreadPoolExecutor


class ActuatorType(Enum):
    """执行器类型"""
    GATE_VALVE = "gate_valve"             # 闸阀
    BUTTERFLY_VALVE = "butterfly_valve"   # 蝶阀
    BALL_VALVE = "ball_valve"             # 球阀
    CONTROL_VALVE = "control_valve"       # 调节阀
    CHECK_VALVE = "check_valve"           # 止回阀
    CENTRIFUGAL_PUMP = "centrifugal_pump" # 离心泵
    AXIAL_PUMP = "axial_pump"             # 轴流泵
    MOTOR = "motor"                       # 电机
    HYDRAULIC_CYLINDER = "hydraulic_cylinder"  # 液压缸


class ActuatorFailureType(Enum):
    """执行器故障类型"""
    NONE = "none"
    STUCK_OPEN = "stuck_open"             # 卡开
    STUCK_CLOSED = "stuck_closed"         # 卡关
    STUCK_POSITION = "stuck_position"     # 卡在某位置
    SLOW_RESPONSE = "slow_response"       # 响应缓慢
    OSCILLATION = "oscillation"           # 振荡
    DEAD_BAND = "dead_band"               # 死区
    HYSTERESIS = "hysteresis"             # 滞环
    LEAKAGE = "leakage"                   # 泄漏
    POWER_LOSS = "power_loss"             # 失电
    MECHANICAL_WEAR = "mechanical_wear"   # 机械磨损
    SEAL_FAILURE = "seal_failure"         # 密封失效
    BEARING_FAILURE = "bearing_failure"   # 轴承故障
    CAVITATION = "cavitation"             # 气蚀


class ControlMode(Enum):
    """控制模式"""
    MANUAL = "manual"
    AUTO = "auto"
    CASCADE = "cascade"
    REMOTE = "remote"
    LOCAL = "local"


@dataclass
class ActuatorDynamics:
    """执行器动力学参数"""
    # 运动参数
    max_velocity: float = 0.01          # 最大速度 (1/s, 开度变化率)
    acceleration: float = 0.1           # 加速度 (1/s²)
    inertia: float = 1.0               # 惯性系数

    # 摩擦参数
    static_friction: float = 0.05      # 静摩擦 (开度)
    dynamic_friction: float = 0.02     # 动摩擦系数
    viscous_damping: float = 0.1       # 粘性阻尼

    # 弹性参数
    stiffness: float = 1000.0          # 刚度 (N/m)
    backlash: float = 0.005            # 间隙 (开度)

    # 限位参数
    min_position: float = 0.0          # 最小位置
    max_position: float = 1.0          # 最大位置
    soft_limit_margin: float = 0.02    # 软限位余量


@dataclass
class ValveCharacteristics:
    """阀门特性参数"""
    valve_type: ActuatorType = ActuatorType.GATE_VALVE
    nominal_diameter: float = 1.0       # 公称直径 (m)
    cv: float = 1000.0                  # 流量系数
    stroke_time: float = 60.0           # 全行程时间 (s)
    max_dp: float = 1.0e6               # 最大压差 (Pa)
    leakage_class: str = "IV"           # 泄漏等级

    # 流量特性
    characteristic: str = "linear"      # linear, equal_percentage, quick_opening
    rangeability: float = 50.0          # 可调比

    # 故障安全
    fail_position: str = "as_is"        # open, closed, as_is


@dataclass
class PumpCharacteristics:
    """泵特性参数"""
    pump_type: ActuatorType = ActuatorType.CENTRIFUGAL_PUMP
    rated_flow: float = 1.0             # 额定流量 (m³/s)
    rated_head: float = 50.0            # 额定扬程 (m)
    rated_speed: float = 1480.0         # 额定转速 (rpm)
    rated_power: float = 100.0          # 额定功率 (kW)
    rated_efficiency: float = 0.85      # 额定效率
    npsh_required: float = 5.0          # 必需汽蚀余量 (m)

    # 泵曲线系数 (H = H0 - a*Q² - b*Q)
    H0: float = 60.0                    # 关死点扬程
    curve_coef_a: float = 10.0          # 二次项系数
    curve_coef_b: float = 1.0           # 一次项系数

    # 动态参数
    rotor_inertia: float = 10.0         # 转子惯量 (kg·m²)
    startup_time: float = 10.0          # 启动时间 (s)
    rundown_time: float = 30.0          # 惯性停车时间 (s)


@dataclass
class MotorCharacteristics:
    """电机特性参数"""
    rated_power: float = 100.0          # 额定功率 (kW)
    rated_speed: float = 1480.0         # 额定转速 (rpm)
    rated_voltage: float = 380.0        # 额定电压 (V)
    rated_current: float = 180.0        # 额定电流 (A)
    power_factor: float = 0.85          # 功率因数
    efficiency: float = 0.92            # 效率

    # 启动参数
    starting_current_ratio: float = 6.0  # 启动电流倍数
    starting_torque_ratio: float = 1.5   # 启动转矩倍数
    max_torque_ratio: float = 2.5        # 最大转矩倍数

    # 热参数
    thermal_time_constant: float = 600.0  # 热时间常数 (s)
    max_temperature: float = 120.0        # 最大温度 (°C)


class VirtualActuator(ABC):
    """虚拟执行器基类"""

    def __init__(self, actuator_id: str, dynamics: ActuatorDynamics):
        self.actuator_id = actuator_id
        self.dynamics = dynamics

        # 状态
        self.position = 0.0             # 当前位置 (0-1)
        self.velocity = 0.0             # 当前速度
        self.command = 0.0              # 指令位置
        self.is_active = True
        self.control_mode = ControlMode.AUTO

        # 故障状态
        self.failure_type = ActuatorFailureType.NONE
        self.failure_params: Dict[str, Any] = {}

        # 时间和统计
        self._time = 0.0
        self._cycles = 0
        self._total_travel = 0.0

    @abstractmethod
    def step(self, command: float, dt: float, external_load: float = 0.0) -> float:
        """执行一步仿真"""
        pass

    @abstractmethod
    def get_output(self) -> Dict[str, float]:
        """获取输出"""
        pass

    def inject_failure(self, failure_type: ActuatorFailureType,
                       params: Optional[Dict[str, Any]] = None):
        """注入故障"""
        self.failure_type = failure_type
        self.failure_params = params or {}

    def clear_failure(self):
        """清除故障"""
        self.failure_type = ActuatorFailureType.NONE
        self.failure_params = {}

    def emergency_stop(self):
        """紧急停止"""
        self.is_active = False
        self.command = self.position  # 保持当前位置

    def reset(self):
        """重置"""
        self.position = 0.0
        self.velocity = 0.0
        self.command = 0.0
        self._time = 0.0
        self._cycles = 0
        self._total_travel = 0.0
        self.clear_failure()
        self.is_active = True

    def get_health_status(self) -> Dict[str, Any]:
        """获取健康状态"""
        return {
            'actuator_id': self.actuator_id,
            'is_active': self.is_active,
            'failure_type': self.failure_type.value,
            'position': self.position,
            'velocity': self.velocity,
            'command': self.command,
            'cycles': self._cycles,
            'total_travel': self._total_travel
        }


class VirtualValve(VirtualActuator):
    """虚拟阀门"""

    def __init__(self, valve_id: str, characteristics: ValveCharacteristics,
                 dynamics: Optional[ActuatorDynamics] = None):
        dynamics = dynamics or ActuatorDynamics(
            max_velocity=1.0 / characteristics.stroke_time
        )
        super().__init__(valve_id, dynamics)
        self.characteristics = characteristics
        self.position = 1.0  # 默认全开

        # 阀门特定状态
        self._torque = 0.0
        self._flow_rate = 0.0
        self._last_direction = 0

    def step(self, command: float, dt: float, external_load: float = 0.0) -> float:
        """
        阀门动态仿真

        Args:
            command: 开度指令 (0-1)
            dt: 时间步长
            external_load: 外部负载 (压差产生的力矩)

        Returns:
            实际开度
        """
        self._time += dt
        command = np.clip(command, 0, 1)
        old_position = self.position

        # 故障处理
        if self.failure_type != ActuatorFailureType.NONE:
            return self._apply_failure(command, dt)

        if not self.is_active:
            return self.position

        self.command = command
        error = command - self.position

        # 死区检测
        if abs(error) < self.dynamics.static_friction:
            self.velocity = 0
            return self.position

        # 速度限制
        max_v = self.dynamics.max_velocity
        if self.failure_type == ActuatorFailureType.SLOW_RESPONSE:
            max_v *= self.failure_params.get('factor', 0.5)

        # 加速度限制
        desired_velocity = np.sign(error) * min(abs(error) / dt, max_v)
        max_accel = self.dynamics.acceleration
        accel = np.clip(
            desired_velocity - self.velocity,
            -max_accel * dt,
            max_accel * dt
        )

        # 摩擦力
        friction_force = (self.dynamics.dynamic_friction *
                         np.sign(self.velocity) if abs(self.velocity) > 0.001 else 0)

        # 更新速度
        self.velocity += accel - friction_force * dt

        # 粘性阻尼
        self.velocity *= (1 - self.dynamics.viscous_damping * dt)

        # 更新位置
        self.position += self.velocity * dt
        self.position = np.clip(self.position,
                               self.dynamics.min_position,
                               self.dynamics.max_position)

        # 统计
        travel = abs(self.position - old_position)
        self._total_travel += travel
        if np.sign(self.velocity) != self._last_direction and self._last_direction != 0:
            self._cycles += 0.5
        self._last_direction = np.sign(self.velocity) if abs(self.velocity) > 0.001 else 0

        return self.position

    def _apply_failure(self, command: float, dt: float) -> float:
        """应用故障模式"""
        if self.failure_type == ActuatorFailureType.STUCK_OPEN:
            self.position = 1.0
            self.velocity = 0
            return self.position

        elif self.failure_type == ActuatorFailureType.STUCK_CLOSED:
            self.position = 0.0
            self.velocity = 0
            return self.position

        elif self.failure_type == ActuatorFailureType.STUCK_POSITION:
            stuck_pos = self.failure_params.get('position', self.position)
            self.position = stuck_pos
            self.velocity = 0
            return self.position

        elif self.failure_type == ActuatorFailureType.OSCILLATION:
            freq = self.failure_params.get('frequency', 1.0)
            amplitude = self.failure_params.get('amplitude', 0.05)
            osc = amplitude * np.sin(2 * np.pi * freq * self._time)
            return np.clip(command + osc, 0, 1)

        elif self.failure_type == ActuatorFailureType.HYSTERESIS:
            width = self.failure_params.get('width', 0.05)
            error = command - self.position
            if abs(error) < width:
                return self.position
            self.position += np.sign(error) * (abs(error) - width) * 0.1
            return self.position

        elif self.failure_type == ActuatorFailureType.POWER_LOSS:
            # 返回故障安全位置
            if self.characteristics.fail_position == 'open':
                target = 1.0
            elif self.characteristics.fail_position == 'closed':
                target = 0.0
            else:
                return self.position

            # 缓慢移动到故障安全位置
            if abs(self.position - target) > 0.01:
                self.position += np.sign(target - self.position) * dt * 0.1
            return self.position

        return self.step(command, dt)

    def compute_flow(self, dp: float, rho: float = 1000.0) -> float:
        """
        计算通过阀门的流量

        Args:
            dp: 压差 (Pa)
            rho: 流体密度 (kg/m³)

        Returns:
            流量 (m³/s)
        """
        cv = self._get_flow_coefficient()

        # 泄漏
        if self.failure_type == ActuatorFailureType.LEAKAGE:
            leakage_rate = self.failure_params.get('rate', 0.01)
            if self.position < 0.01:
                cv = leakage_rate * self.characteristics.cv

        self._flow_rate = cv * np.sqrt(abs(dp) / rho) * np.sign(dp)
        return self._flow_rate

    def _get_flow_coefficient(self) -> float:
        """获取流量系数"""
        opening = self.position
        cv_max = self.characteristics.cv
        char = self.characteristics.characteristic

        if char == "linear":
            return cv_max * opening

        elif char == "equal_percentage":
            R = self.characteristics.rangeability
            if opening < 0.01:
                return 0
            return cv_max * R ** (opening - 1)

        elif char == "quick_opening":
            return cv_max * np.sqrt(opening)

        return cv_max * opening

    def get_output(self) -> Dict[str, float]:
        """获取输出"""
        return {
            'position': self.position,
            'velocity': self.velocity,
            'command': self.command,
            'cv': self._get_flow_coefficient(),
            'flow_rate': self._flow_rate,
            'torque': self._torque
        }


class VirtualPump(VirtualActuator):
    """虚拟泵"""

    def __init__(self, pump_id: str, characteristics: PumpCharacteristics,
                 dynamics: Optional[ActuatorDynamics] = None):
        dynamics = dynamics or ActuatorDynamics(
            max_velocity=characteristics.rated_speed / (60 * characteristics.startup_time),
            inertia=characteristics.rotor_inertia
        )
        super().__init__(pump_id, dynamics)
        self.characteristics = characteristics

        # 泵特定状态
        self.speed = 0.0                # 转速 (rpm)
        self.flow = 0.0                 # 流量 (m³/s)
        self.head = 0.0                 # 扬程 (m)
        self.power = 0.0                # 功率 (kW)
        self.efficiency = 0.0           # 效率
        self.is_running = False
        self.npsh_available = 10.0      # 可用汽蚀余量

        # 启停状态
        self._startup_timer = 0.0
        self._rundown_speed = 0.0

    def step(self, command: float, dt: float, external_load: float = 0.0) -> float:
        """
        泵动态仿真

        Args:
            command: 转速指令 (0-1, 相对于额定转速)
            dt: 时间步长
            external_load: 系统扬程需求 (m)

        Returns:
            实际转速比
        """
        self._time += dt
        self.command = np.clip(command, 0, 1)

        # 故障处理
        if self.failure_type != ActuatorFailureType.NONE:
            return self._apply_failure(command, dt, external_load)

        if not self.is_active:
            self._coast_down(dt)
            return self.speed / self.characteristics.rated_speed

        target_speed = self.command * self.characteristics.rated_speed

        # 启动/停止逻辑
        if target_speed > 0 and not self.is_running:
            self._startup_timer += dt
            if self._startup_timer > 1.0:  # 1秒启动延迟
                self.is_running = True
        elif target_speed == 0:
            self.is_running = False
            self._startup_timer = 0.0

        # 转速动态
        if self.is_running:
            tau = self.characteristics.startup_time / 3
            self.speed += (target_speed - self.speed) * (1 - np.exp(-dt / tau))
        else:
            self._coast_down(dt)

        # 计算工作点
        self._compute_operating_point(external_load)

        self.position = self.speed / self.characteristics.rated_speed
        return self.position

    def _coast_down(self, dt: float):
        """惯性停车"""
        if self.speed > 0:
            tau = self.characteristics.rundown_time / 3
            self.speed *= np.exp(-dt / tau)
            if self.speed < 1.0:
                self.speed = 0.0

    def _compute_operating_point(self, system_head: float):
        """计算工作点"""
        if self.speed < 1.0:
            self.flow = 0.0
            self.head = 0.0
            self.power = 0.0
            self.efficiency = 0.0
            return

        n_ratio = self.speed / self.characteristics.rated_speed
        char = self.characteristics

        # 相似定律
        H0_actual = char.H0 * n_ratio ** 2
        a_actual = char.curve_coef_a
        b_actual = char.curve_coef_b * n_ratio

        # 求解工作点 H_pump = H_system
        # H0 - a*Q² - b*Q = system_head
        # a*Q² + b*Q + (system_head - H0) = 0
        a = a_actual
        b = b_actual
        c = system_head - H0_actual

        discriminant = b**2 - 4*a*c
        if discriminant < 0:
            # 无解，泵扬程不足
            self.flow = 0.0
            self.head = H0_actual
        else:
            Q = (-b + np.sqrt(discriminant)) / (2*a)
            self.flow = max(0, Q * n_ratio)
            self.head = system_head

        # 效率计算
        self.efficiency = self._compute_efficiency()

        # 功率计算
        if self.efficiency > 0:
            self.power = 9.81 * 1000 * self.flow * self.head / (1000 * self.efficiency)
        else:
            self.power = 0.0

    def _compute_efficiency(self) -> float:
        """计算效率"""
        if self.flow <= 0:
            return 0.0

        Q_bep = self.characteristics.rated_flow
        eta_max = self.characteristics.rated_efficiency

        q_ratio = self.flow / Q_bep
        # 效率曲线 (抛物线近似)
        eta = eta_max * (2 * q_ratio - q_ratio ** 2)

        if self.failure_type == ActuatorFailureType.MECHANICAL_WEAR:
            eta *= self.failure_params.get('efficiency_factor', 0.9)

        return max(0, min(eta, 1))

    def _apply_failure(self, command: float, dt: float, external_load: float) -> float:
        """应用故障模式"""
        if self.failure_type == ActuatorFailureType.POWER_LOSS:
            self.is_running = False
            self._coast_down(dt)
            self._compute_operating_point(external_load)
            return self.speed / self.characteristics.rated_speed

        elif self.failure_type == ActuatorFailureType.MECHANICAL_WEAR:
            # 效率下降在_compute_efficiency中处理
            return self.step(command, dt, external_load)

        elif self.failure_type == ActuatorFailureType.CAVITATION:
            if self.npsh_available < self.characteristics.npsh_required:
                # 气蚀导致性能下降
                cavitation_factor = self.npsh_available / self.characteristics.npsh_required
                self.flow *= cavitation_factor
                self.head *= cavitation_factor ** 2
            return self.step(command, dt, external_load)

        elif self.failure_type == ActuatorFailureType.BEARING_FAILURE:
            # 轴承故障导致振动和效率下降
            vibration = np.random.normal(0, self.failure_params.get('vibration', 0.1))
            self.speed *= (1 + vibration * 0.01)
            return self.speed / self.characteristics.rated_speed

        return self.step(command, dt, external_load)

    def check_cavitation(self) -> bool:
        """检查是否发生气蚀"""
        return self.npsh_available < self.characteristics.npsh_required

    def get_output(self) -> Dict[str, float]:
        """获取输出"""
        return {
            'position': self.position,
            'speed': self.speed,
            'speed_ratio': self.position,
            'flow': self.flow,
            'head': self.head,
            'power': self.power,
            'efficiency': self.efficiency,
            'is_running': float(self.is_running),
            'npsh_margin': self.npsh_available - self.characteristics.npsh_required
        }


class VirtualMotor(VirtualActuator):
    """虚拟电机"""

    def __init__(self, motor_id: str, characteristics: MotorCharacteristics,
                 dynamics: Optional[ActuatorDynamics] = None):
        dynamics = dynamics or ActuatorDynamics(max_velocity=0.1)
        super().__init__(motor_id, dynamics)
        self.characteristics = characteristics

        # 电机特定状态
        self.speed = 0.0                # 转速 (rpm)
        self.torque = 0.0               # 输出转矩 (N·m)
        self.current = 0.0              # 电流 (A)
        self.temperature = 25.0         # 温度 (°C)
        self.power_input = 0.0          # 输入功率 (kW)
        self.power_output = 0.0         # 输出功率 (kW)

        # 保护状态
        self.is_protected = False
        self.protection_reason = ""

        # 计算额定转矩
        self.rated_torque = (characteristics.rated_power * 1000 * 60) / (
            2 * np.pi * characteristics.rated_speed
        )

    def step(self, command: float, dt: float, external_load: float = 0.0) -> float:
        """
        电机动态仿真

        Args:
            command: 速度指令 (0-1)
            dt: 时间步长
            external_load: 负载转矩 (N·m)

        Returns:
            实际转速比
        """
        self._time += dt
        self.command = np.clip(command, 0, 1)

        # 保护检查
        if self.is_protected:
            self._coast_down(dt)
            return self.speed / self.characteristics.rated_speed

        # 故障处理
        if self.failure_type != ActuatorFailureType.NONE:
            return self._apply_failure(command, dt, external_load)

        if not self.is_active:
            self._coast_down(dt)
            return self.speed / self.characteristics.rated_speed

        target_speed = self.command * self.characteristics.rated_speed

        # 计算电磁转矩
        speed_error = target_speed - self.speed
        self.torque = self._compute_electromagnetic_torque(speed_error)

        # 运动方程
        J = self.dynamics.inertia
        B = self.dynamics.viscous_damping * self.rated_torque / self.characteristics.rated_speed

        alpha = (self.torque - external_load - B * self.speed) / J
        self.speed = max(0, self.speed + alpha * dt)

        # 电气计算
        self._compute_electrical()

        # 热计算
        self._compute_thermal(dt)

        # 保护动作
        self._check_protection()

        self.position = self.speed / self.characteristics.rated_speed
        return self.position

    def _coast_down(self, dt: float):
        """惯性停车"""
        if self.speed > 0:
            B = self.dynamics.viscous_damping * self.rated_torque / self.characteristics.rated_speed
            J = self.dynamics.inertia
            alpha = -B * self.speed / J
            self.speed = max(0, self.speed + alpha * dt)

    def _compute_electromagnetic_torque(self, speed_error: float) -> float:
        """计算电磁转矩"""
        # 简化的PI调速
        Kp = self.rated_torque / (0.1 * self.characteristics.rated_speed)
        torque = Kp * speed_error

        # 转矩限制
        max_torque = self.characteristics.max_torque_ratio * self.rated_torque
        torque = np.clip(torque, -max_torque, max_torque)

        return torque

    def _compute_electrical(self):
        """计算电气参数"""
        # 简化模型
        torque_ratio = abs(self.torque) / self.rated_torque
        speed_ratio = self.speed / self.characteristics.rated_speed

        # 电流 (与转矩近似正比)
        self.current = self.characteristics.rated_current * torque_ratio

        # 功率
        self.power_output = self.torque * self.speed * 2 * np.pi / 60 / 1000  # kW
        if self.power_output > 0:
            self.power_input = self.power_output / self.characteristics.efficiency
        else:
            self.power_input = 0

    def _compute_thermal(self, dt: float):
        """计算热参数"""
        # 损耗功率产生热量
        power_loss = self.power_input - self.power_output
        tau = self.characteristics.thermal_time_constant

        # 温度上升
        ambient = 25.0
        equilibrium_temp = ambient + power_loss * 0.5  # 简化热阻模型
        self.temperature += (equilibrium_temp - self.temperature) * dt / tau

    def _check_protection(self):
        """检查保护"""
        char = self.characteristics

        # 过温保护
        if self.temperature > char.max_temperature:
            self.is_protected = True
            self.protection_reason = "overtemperature"
            return

        # 过流保护
        if self.current > char.starting_current_ratio * char.rated_current:
            self.is_protected = True
            self.protection_reason = "overcurrent"

    def _apply_failure(self, command: float, dt: float, external_load: float) -> float:
        """应用故障模式"""
        if self.failure_type == ActuatorFailureType.POWER_LOSS:
            self._coast_down(dt)
            self.current = 0
            self.power_input = 0
            self.power_output = 0
            return self.speed / self.characteristics.rated_speed

        elif self.failure_type == ActuatorFailureType.BEARING_FAILURE:
            # 增加摩擦损耗
            extra_friction = self.failure_params.get('friction', 0.2)
            friction_torque = extra_friction * self.rated_torque
            return self.step(command, dt, external_load + friction_torque)

        return self.step(command, dt, external_load)

    def reset_protection(self):
        """复位保护"""
        if self.temperature < self.characteristics.max_temperature * 0.8:
            self.is_protected = False
            self.protection_reason = ""

    def get_output(self) -> Dict[str, float]:
        """获取输出"""
        return {
            'position': self.position,
            'speed': self.speed,
            'speed_ratio': self.position,
            'torque': self.torque,
            'current': self.current,
            'temperature': self.temperature,
            'power_input': self.power_input,
            'power_output': self.power_output,
            'is_protected': float(self.is_protected)
        }


class VirtualActuatorNetwork:
    """虚拟执行器网络"""

    def __init__(self, name: str = "default_network"):
        self.name = name
        self.actuators: Dict[str, VirtualActuator] = {}
        self._lock = threading.Lock()

    def add_actuator(self, actuator: VirtualActuator):
        """添加执行器"""
        with self._lock:
            self.actuators[actuator.actuator_id] = actuator

    def remove_actuator(self, actuator_id: str):
        """移除执行器"""
        with self._lock:
            if actuator_id in self.actuators:
                del self.actuators[actuator_id]

    def step_all(self, commands: Dict[str, float], dt: float,
                 external_loads: Optional[Dict[str, float]] = None) -> Dict[str, Dict[str, float]]:
        """
        步进所有执行器

        Args:
            commands: 各执行器指令
            dt: 时间步长
            external_loads: 外部负载

        Returns:
            所有执行器输出
        """
        outputs = {}
        external_loads = external_loads or {}

        for act_id, actuator in self.actuators.items():
            command = commands.get(act_id, 0.0)
            load = external_loads.get(act_id, 0.0)
            actuator.step(command, dt, load)
            outputs[act_id] = actuator.get_output()

        return outputs

    def get_positions(self) -> Dict[str, float]:
        """获取所有位置"""
        return {act_id: act.position for act_id, act in self.actuators.items()}

    def emergency_shutdown(self):
        """紧急停车"""
        for actuator in self.actuators.values():
            actuator.emergency_stop()

    def get_network_status(self) -> Dict[str, Any]:
        """获取网络状态"""
        return {
            'name': self.name,
            'total_actuators': len(self.actuators),
            'actuators': {
                aid: act.get_health_status()
                for aid, act in self.actuators.items()
            }
        }

    def reset_all(self):
        """重置所有执行器"""
        for actuator in self.actuators.values():
            actuator.reset()


class ActuatorDynamicsEngine:
    """执行器动力学引擎"""

    def __init__(self):
        self._physics_callbacks: List[Callable] = []

    def register_physics_callback(self, callback: Callable):
        """注册物理回调"""
        self._physics_callbacks.append(callback)

    def compute_valve_torque(self, valve: VirtualValve, dp: float) -> float:
        """计算阀门力矩"""
        char = valve.characteristics
        area = np.pi * (char.nominal_diameter / 2) ** 2

        # 动力矩 (与压差和开度相关)
        dynamic_torque = dp * area * char.nominal_diameter * 0.1 * valve.position

        # 摩擦力矩
        friction_torque = 0.1 * char.nominal_diameter ** 3 * 1000  # 简化

        return dynamic_torque + friction_torque

    def compute_pump_coupling(self, pump: VirtualPump, system_curve: Callable) -> Dict[str, float]:
        """计算泵-系统耦合"""
        # 系统曲线 H = f(Q)
        Q = pump.flow
        H_system = system_curve(Q)

        # 更新工作点
        pump._compute_operating_point(H_system)

        return {
            'flow': pump.flow,
            'head': pump.head,
            'operating_point': (Q, pump.head)
        }


class FailureSimulator:
    """故障仿真器"""

    def __init__(self, network: VirtualActuatorNetwork):
        self.network = network
        self._scheduled_failures: List[Dict[str, Any]] = []
        self._active_failures: Dict[str, Dict[str, Any]] = {}

    def schedule_failure(self, actuator_id: str, failure_type: ActuatorFailureType,
                         start_time: float, duration: float = -1,
                         params: Optional[Dict[str, Any]] = None):
        """调度故障"""
        self._scheduled_failures.append({
            'actuator_id': actuator_id,
            'failure_type': failure_type,
            'start_time': start_time,
            'duration': duration,
            'params': params or {},
            'activated': False
        })

    def update(self, current_time: float):
        """更新故障状态"""
        # 激活计划的故障
        for failure in self._scheduled_failures:
            if not failure['activated'] and current_time >= failure['start_time']:
                act_id = failure['actuator_id']
                if act_id in self.network.actuators:
                    self.network.actuators[act_id].inject_failure(
                        failure['failure_type'],
                        failure['params']
                    )
                    self._active_failures[act_id] = {
                        'start_time': failure['start_time'],
                        'duration': failure['duration'],
                        'failure_type': failure['failure_type']
                    }
                    failure['activated'] = True

        # 移除过期的故障
        for act_id in list(self._active_failures.keys()):
            info = self._active_failures[act_id]
            if info['duration'] > 0:
                if current_time >= info['start_time'] + info['duration']:
                    if act_id in self.network.actuators:
                        self.network.actuators[act_id].clear_failure()
                    del self._active_failures[act_id]

    def inject_immediate(self, actuator_id: str, failure_type: ActuatorFailureType,
                         params: Optional[Dict[str, Any]] = None):
        """立即注入故障"""
        if actuator_id in self.network.actuators:
            self.network.actuators[actuator_id].inject_failure(failure_type, params)

    def clear_all(self):
        """清除所有故障"""
        for actuator in self.network.actuators.values():
            actuator.clear_failure()
        self._scheduled_failures.clear()
        self._active_failures.clear()


class ActuatorSimulationManager:
    """执行器仿真管理器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.networks: Dict[str, VirtualActuatorNetwork] = {}
        self.engines: Dict[str, ActuatorDynamicsEngine] = {}
        self.simulators: Dict[str, FailureSimulator] = {}

        self._time = 0.0
        self._is_running = False

    def create_network(self, network_name: str) -> VirtualActuatorNetwork:
        """创建执行器网络"""
        network = VirtualActuatorNetwork(network_name)
        self.networks[network_name] = network
        self.engines[network_name] = ActuatorDynamicsEngine()
        self.simulators[network_name] = FailureSimulator(network)
        return network

    def create_standard_tunnel_network(self, network_name: str = "tunnel") -> VirtualActuatorNetwork:
        """
        创建标准隧道执行器网络

        包含进出口闸阀、调节阀、排水泵等
        """
        network = self.create_network(network_name)

        # 北洞进口闸阀
        north_inlet_char = ValveCharacteristics(
            valve_type=ActuatorType.GATE_VALVE,
            nominal_diameter=7.0,
            cv=50000,
            stroke_time=120.0,
            max_dp=1.0e6,
            leakage_class="IV",
            characteristic="linear",
            fail_position="as_is"
        )
        network.add_actuator(VirtualValve("north_inlet", north_inlet_char))

        # 北洞出口闸阀
        north_outlet_char = ValveCharacteristics(
            valve_type=ActuatorType.GATE_VALVE,
            nominal_diameter=7.0,
            cv=50000,
            stroke_time=120.0,
            max_dp=1.0e6,
            characteristic="linear",
            fail_position="as_is"
        )
        network.add_actuator(VirtualValve("north_outlet", north_outlet_char))

        # 南洞进口闸阀
        south_inlet_char = ValveCharacteristics(
            valve_type=ActuatorType.GATE_VALVE,
            nominal_diameter=7.0,
            cv=50000,
            stroke_time=120.0,
            max_dp=1.0e6,
            characteristic="linear",
            fail_position="as_is"
        )
        network.add_actuator(VirtualValve("south_inlet", south_inlet_char))

        # 南洞出口闸阀
        south_outlet_char = ValveCharacteristics(
            valve_type=ActuatorType.GATE_VALVE,
            nominal_diameter=7.0,
            cv=50000,
            stroke_time=120.0,
            max_dp=1.0e6,
            characteristic="linear",
            fail_position="as_is"
        )
        network.add_actuator(VirtualValve("south_outlet", south_outlet_char))

        # 调节蝶阀
        control_char = ValveCharacteristics(
            valve_type=ActuatorType.BUTTERFLY_VALVE,
            nominal_diameter=3.0,
            cv=10000,
            stroke_time=30.0,
            max_dp=0.5e6,
            characteristic="equal_percentage",
            rangeability=50.0,
            fail_position="as_is"
        )
        network.add_actuator(VirtualValve("control_valve", control_char))

        # 排水泵
        drain_pump_char = PumpCharacteristics(
            pump_type=ActuatorType.CENTRIFUGAL_PUMP,
            rated_flow=0.5,
            rated_head=50.0,
            rated_speed=1480,
            rated_power=45,
            rated_efficiency=0.8,
            npsh_required=5.0,
            H0=60.0,
            curve_coef_a=40.0,
            curve_coef_b=5.0,
            startup_time=10.0,
            rundown_time=30.0
        )
        network.add_actuator(VirtualPump("drain_pump_1", drain_pump_char))
        network.add_actuator(VirtualPump("drain_pump_2", drain_pump_char))

        return network

    def step(self, commands: Dict[str, Dict[str, float]], dt: float,
             external_loads: Optional[Dict[str, Dict[str, float]]] = None
             ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        执行一步仿真

        Args:
            commands: 各网络各执行器指令
            dt: 时间步长
            external_loads: 外部负载

        Returns:
            各网络执行器输出
        """
        self._time += dt
        results = {}
        external_loads = external_loads or {}

        for network_name, network in self.networks.items():
            # 更新故障状态
            if network_name in self.simulators:
                self.simulators[network_name].update(self._time)

            # 步进执行器
            cmds = commands.get(network_name, {})
            loads = external_loads.get(network_name, {})
            results[network_name] = network.step_all(cmds, dt, loads)

        return results

    def get_status(self) -> Dict[str, Any]:
        """获取仿真状态"""
        return {
            'time': self._time,
            'is_running': self._is_running,
            'networks': {
                name: network.get_network_status()
                for name, network in self.networks.items()
            }
        }

    def reset(self):
        """重置仿真"""
        self._time = 0.0
        for network in self.networks.values():
            network.reset_all()
        for simulator in self.simulators.values():
            simulator.clear_all()
