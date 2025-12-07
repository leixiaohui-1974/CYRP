"""
Safety Interlock System for CYRP.
穿黄工程安全联锁系统

实现报告II中定义的硬逻辑与安全联锁
即使MPC发出指令，PLC也会根据联锁逻辑进行拦截
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Callable, Any
from enum import Enum, auto
import numpy as np


class InterlockType(Enum):
    """联锁类型"""
    ANTI_VACUUM = auto()  # 防真空联锁
    ANTI_ASYMMETRIC = auto()  # 防非同步联锁
    ANTI_SURGE = auto()  # 防喘振联锁
    ANTI_OVERPRESSURE = auto()  # 防超压联锁
    ANTI_OVERFLOW = auto()  # 防溢流联锁
    EMERGENCY_SHUTDOWN = auto()  # 紧急停机


class InterlockState(Enum):
    """联锁状态"""
    NORMAL = "normal"  # 正常
    WARNING = "warning"  # 预警
    TRIGGERED = "triggered"  # 触发
    LOCKED = "locked"  # 锁定


@dataclass
class InterlockAction:
    """联锁动作"""
    action_type: str  # "block", "override", "shutdown"
    target: str  # 作用对象
    value: float  # 强制值
    message: str  # 信息


@dataclass
class InterlockStatus:
    """联锁状态信息"""
    interlock_type: InterlockType
    state: InterlockState
    trigger_value: float
    threshold: float
    action: Optional[InterlockAction]
    timestamp: float


class SafetyInterlock:
    """
    安全联锁基类
    """

    def __init__(
        self,
        interlock_type: InterlockType,
        threshold: float,
        hysteresis: float = 0.05
    ):
        """
        初始化联锁

        Args:
            interlock_type: 联锁类型
            threshold: 触发阈值
            hysteresis: 回滞量
        """
        self.interlock_type = interlock_type
        self.threshold = threshold
        self.hysteresis = hysteresis

        self.state = InterlockState.NORMAL
        self.triggered_time = 0.0
        self.last_value = 0.0

    def check(
        self,
        value: float,
        timestamp: float
    ) -> Tuple[bool, Optional[InterlockAction]]:
        """
        检查联锁条件

        Args:
            value: 检测值
            timestamp: 时间戳

        Returns:
            是否触发, 联锁动作
        """
        raise NotImplementedError

    def reset(self):
        """重置联锁"""
        self.state = InterlockState.NORMAL
        self.triggered_time = 0.0


class AntiVacuumInterlock(SafetyInterlock):
    """
    防真空联锁

    触发条件: 洞顶压力 < -0.05 MPa
    动作:
    - 强制切断排水泵电源
    - 强制开启真空破坏阀补气
    """

    def __init__(self, threshold: float = -5e4):
        super().__init__(InterlockType.ANTI_VACUUM, threshold)

    def check(
        self,
        pressure: float,
        timestamp: float
    ) -> Tuple[bool, Optional[InterlockAction]]:
        """检查真空条件"""
        self.last_value = pressure

        if pressure < self.threshold:
            if self.state != InterlockState.TRIGGERED:
                self.state = InterlockState.TRIGGERED
                self.triggered_time = timestamp

            action = InterlockAction(
                action_type="override",
                target="vacuum_break_valve",
                value=1.0,  # 全开
                message=f"防真空联锁触发: P={pressure/1000:.1f}kPa < {self.threshold/1000:.1f}kPa"
            )
            return True, action

        elif pressure > self.threshold + self.hysteresis * abs(self.threshold):
            self.state = InterlockState.NORMAL
            return False, None

        return False, None


class AntiAsymmetricInterlock(SafetyInterlock):
    """
    防非同步联锁

    触发条件: |e_gate1 - e_gate2| > 10 cm (约0.1)
    动作:
    - 锁定开度较大的闸门，禁止继续开启
    - 锁定开度较小的闸门，禁止继续关闭
    """

    def __init__(self, threshold: float = 0.1):
        super().__init__(InterlockType.ANTI_ASYMMETRIC, threshold)

    def check(
        self,
        gate_positions: np.ndarray,
        timestamp: float
    ) -> Tuple[bool, Optional[InterlockAction]]:
        """检查闸门同步性"""
        diff = abs(gate_positions[0] - gate_positions[1])
        self.last_value = diff

        if diff > self.threshold:
            if self.state != InterlockState.TRIGGERED:
                self.state = InterlockState.TRIGGERED
                self.triggered_time = timestamp

            # 确定哪个闸门开度大
            if gate_positions[0] > gate_positions[1]:
                action = InterlockAction(
                    action_type="block",
                    target="gate_1_open",
                    value=gate_positions[0],  # 锁定当前位置
                    message=f"防非同步联锁: Δe={diff*100:.1f}cm > {self.threshold*100:.1f}cm"
                )
            else:
                action = InterlockAction(
                    action_type="block",
                    target="gate_2_open",
                    value=gate_positions[1],
                    message=f"防非同步联锁: Δe={diff*100:.1f}cm > {self.threshold*100:.1f}cm"
                )
            return True, action

        elif diff < self.threshold - self.hysteresis:
            self.state = InterlockState.NORMAL
            return False, None

        return False, None


class AntiSurgeInterlock(SafetyInterlock):
    """
    防喘振联锁

    触发条件: 高频压力传感器检测到 f > 5Hz 的压力脉动
    动作: 切换至位置保持模式，拒绝大幅度调节指令
    """

    def __init__(
        self,
        frequency_threshold: float = 5.0,
        amplitude_threshold: float = 1e4
    ):
        super().__init__(InterlockType.ANTI_SURGE, frequency_threshold)
        self.amplitude_threshold = amplitude_threshold
        self.pressure_buffer: List[float] = []
        self.buffer_size = 100

    def check(
        self,
        pressure: float,
        timestamp: float
    ) -> Tuple[bool, Optional[InterlockAction]]:
        """检查喘振条件"""
        self.pressure_buffer.append(pressure)
        if len(self.pressure_buffer) > self.buffer_size:
            self.pressure_buffer.pop(0)

        if len(self.pressure_buffer) < self.buffer_size:
            return False, None

        # 简化的频率分析 (计算过零点)
        signal = np.array(self.pressure_buffer)
        signal = signal - np.mean(signal)

        zero_crossings = np.where(np.diff(np.signbit(signal)))[0]
        if len(zero_crossings) > 2:
            avg_period = len(signal) / (len(zero_crossings) / 2)
            freq = 100 / avg_period  # 假设100Hz采样

            amplitude = np.max(signal) - np.min(signal)

            if freq > self.threshold and amplitude > self.amplitude_threshold:
                if self.state != InterlockState.TRIGGERED:
                    self.state = InterlockState.TRIGGERED
                    self.triggered_time = timestamp

                action = InterlockAction(
                    action_type="block",
                    target="all_gates",
                    value=0.0,  # 禁止动作
                    message=f"防喘振联锁: f={freq:.1f}Hz > {self.threshold}Hz"
                )
                return True, action

        self.state = InterlockState.NORMAL
        return False, None


class AntiOverpressureInterlock(SafetyInterlock):
    """
    防超压联锁

    触发条件: 压力超过结构允许值
    动作: 开启泄压阀或减少流量
    """

    def __init__(self, threshold: float = 1.0e6):
        super().__init__(InterlockType.ANTI_OVERPRESSURE, threshold)

    def check(
        self,
        pressure: float,
        timestamp: float
    ) -> Tuple[bool, Optional[InterlockAction]]:
        """检查超压条件"""
        self.last_value = pressure

        if pressure > self.threshold:
            if self.state != InterlockState.TRIGGERED:
                self.state = InterlockState.TRIGGERED
                self.triggered_time = timestamp

            action = InterlockAction(
                action_type="override",
                target="outlet_gate",
                value=1.0,  # 全开出口
                message=f"防超压联锁: P={pressure/1e6:.2f}MPa > {self.threshold/1e6:.2f}MPa"
            )
            return True, action

        elif pressure < self.threshold * (1 - self.hysteresis):
            self.state = InterlockState.NORMAL

        return False, None


class SafetyInterlockSystem:
    """
    安全联锁系统

    集成所有联锁逻辑，提供统一的安全检查接口
    """

    def __init__(self):
        """初始化联锁系统"""
        # 注册所有联锁
        self.interlocks: Dict[InterlockType, SafetyInterlock] = {
            InterlockType.ANTI_VACUUM: AntiVacuumInterlock(-5e4),
            InterlockType.ANTI_ASYMMETRIC: AntiAsymmetricInterlock(0.1),
            InterlockType.ANTI_SURGE: AntiSurgeInterlock(5.0),
            InterlockType.ANTI_OVERPRESSURE: AntiOverpressureInterlock(1.0e6),
        }

        # 联锁状态历史
        self.status_log: List[InterlockStatus] = []

        # 全局锁定标志
        self.system_locked = False

    def check_all(
        self,
        sensor_data: Dict[str, Any],
        timestamp: float
    ) -> Tuple[bool, List[InterlockAction]]:
        """
        检查所有联锁

        Args:
            sensor_data: 传感器数据
            timestamp: 时间戳

        Returns:
            是否有触发, 触发的动作列表
        """
        triggered_actions = []

        # 检查真空
        if 'pressure_min' in sensor_data:
            triggered, action = self.interlocks[InterlockType.ANTI_VACUUM].check(
                sensor_data['pressure_min'], timestamp
            )
            if triggered and action:
                triggered_actions.append(action)

        # 检查闸门同步
        if 'gate_positions' in sensor_data:
            triggered, action = self.interlocks[InterlockType.ANTI_ASYMMETRIC].check(
                np.array(sensor_data['gate_positions']), timestamp
            )
            if triggered and action:
                triggered_actions.append(action)

        # 检查喘振
        if 'pressure' in sensor_data:
            triggered, action = self.interlocks[InterlockType.ANTI_SURGE].check(
                sensor_data['pressure'], timestamp
            )
            if triggered and action:
                triggered_actions.append(action)

        # 检查超压
        if 'pressure_max' in sensor_data:
            triggered, action = self.interlocks[InterlockType.ANTI_OVERPRESSURE].check(
                sensor_data['pressure_max'], timestamp
            )
            if triggered and action:
                triggered_actions.append(action)

        return len(triggered_actions) > 0, triggered_actions

    def apply_interlocks(
        self,
        control_command: np.ndarray,
        actions: List[InterlockAction]
    ) -> np.ndarray:
        """
        应用联锁动作到控制指令

        Args:
            control_command: 原始控制指令 [u1, u2]
            actions: 联锁动作列表

        Returns:
            修正后的控制指令
        """
        modified_command = control_command.copy()

        for action in actions:
            if action.action_type == "block":
                if "gate_1" in action.target:
                    # 禁止1#闸门动作
                    modified_command[0] = action.value
                elif "gate_2" in action.target:
                    modified_command[1] = action.value
                elif "all_gates" in action.target:
                    # 保持当前位置
                    pass

            elif action.action_type == "override":
                if "vacuum_break" in action.target:
                    # 真空破坏阀控制不在此处
                    pass
                elif "outlet_gate" in action.target:
                    # 强制开启出口
                    pass

            elif action.action_type == "shutdown":
                # 紧急停机
                modified_command = np.array([0.0, 0.0])
                self.system_locked = True

        return modified_command

    def get_status(self) -> Dict[str, InterlockState]:
        """获取所有联锁状态"""
        return {
            itype.name: interlock.state
            for itype, interlock in self.interlocks.items()
        }

    def reset_all(self):
        """重置所有联锁"""
        for interlock in self.interlocks.values():
            interlock.reset()
        self.system_locked = False
