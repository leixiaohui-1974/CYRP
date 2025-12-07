"""
PID Controllers for CYRP Local Execution Layer.
穿黄工程局部执行层PID控制器

MPC-Guided PID架构:
u(t) = u_MPC(t) + Kp*e(t) + Ki*∫e(t)dt + Kd*de(t)/dt
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
import numpy as np
from enum import Enum


class PIDMode(Enum):
    """PID运行模式"""
    AUTO = "auto"  # 自动模式
    MANUAL = "manual"  # 手动模式
    CASCADE = "cascade"  # 串级模式
    TRACKING = "tracking"  # 跟踪模式 (无扰切换)


@dataclass
class PIDConfig:
    """PID配置参数"""
    Kp: float = 1.0  # 比例增益
    Ki: float = 0.1  # 积分增益
    Kd: float = 0.01  # 微分增益

    # 输出限幅
    output_min: float = 0.0
    output_max: float = 1.0

    # 积分抗饱和
    integral_min: float = -10.0
    integral_max: float = 10.0

    # 微分滤波
    derivative_filter: float = 0.1  # 微分滤波系数

    # 死区
    deadband: float = 0.001

    # 采样时间
    sample_time: float = 0.01  # 10ms


@dataclass
class PIDState:
    """PID状态"""
    setpoint: float = 0.0
    process_value: float = 0.0
    error: float = 0.0
    integral: float = 0.0
    derivative: float = 0.0
    output: float = 0.0
    feedforward: float = 0.0
    mode: PIDMode = PIDMode.AUTO


class PIDController:
    """
    PID控制器

    功能:
    1. 标准PID计算
    2. 积分抗饱和
    3. 微分滤波
    4. 无扰切换
    5. MPC前馈
    """

    def __init__(
        self,
        config: Optional[PIDConfig] = None,
        name: str = "PID"
    ):
        """
        初始化PID控制器

        Args:
            config: PID配置
            name: 控制器名称
        """
        self.config = config or PIDConfig()
        self.name = name

        # 状态
        self.state = PIDState()
        self.mode = PIDMode.AUTO

        # 历史值
        self.prev_error = 0.0
        self.prev_derivative = 0.0
        self.prev_pv = 0.0

        # 积分器
        self.integral = 0.0

        # 上一输出 (用于无扰切换)
        self.last_output = 0.0

    def reset(self):
        """重置控制器"""
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_derivative = 0.0
        self.last_output = 0.0

    def set_mode(self, mode: PIDMode, bumpless: bool = True):
        """
        设置运行模式

        Args:
            mode: 目标模式
            bumpless: 是否进行无扰切换
        """
        if bumpless and mode == PIDMode.AUTO and self.mode != PIDMode.AUTO:
            # 从手动切换到自动，初始化积分器
            self.integral = self.last_output - self.config.Kp * self.prev_error

        self.mode = mode
        self.state.mode = mode

    def compute(
        self,
        setpoint: float,
        process_value: float,
        feedforward: float = 0.0,
        dt: Optional[float] = None
    ) -> float:
        """
        计算PID输出

        Args:
            setpoint: 设定值
            process_value: 过程值
            feedforward: 前馈量 (来自MPC)
            dt: 采样时间

        Returns:
            控制输出
        """
        dt = dt or self.config.sample_time

        # 计算误差
        error = setpoint - process_value

        # 死区处理
        if abs(error) < self.config.deadband:
            error = 0.0

        # 比例项
        P = self.config.Kp * error

        # 积分项 (带抗饱和)
        self.integral += error * dt
        self.integral = np.clip(
            self.integral,
            self.config.integral_min,
            self.config.integral_max
        )
        I = self.config.Ki * self.integral

        # 微分项 (对PV微分，避免设定值突变)
        derivative_raw = -(process_value - self.prev_pv) / dt
        # 一阶滤波
        alpha = self.config.derivative_filter
        derivative = alpha * derivative_raw + (1 - alpha) * self.prev_derivative
        D = self.config.Kd * derivative

        # 总输出
        output = feedforward + P + I + D

        # 输出限幅
        output = np.clip(output, self.config.output_min, self.config.output_max)

        # 积分抗饱和 (back-calculation)
        if output >= self.config.output_max or output <= self.config.output_min:
            # 饱和时不再积分
            self.integral -= error * dt

        # 更新状态
        self.prev_error = error
        self.prev_derivative = derivative
        self.prev_pv = process_value
        self.last_output = output

        # 记录状态
        self.state = PIDState(
            setpoint=setpoint,
            process_value=process_value,
            error=error,
            integral=self.integral,
            derivative=derivative,
            output=output,
            feedforward=feedforward,
            mode=self.mode
        )

        return output

    def get_state(self) -> PIDState:
        """获取当前状态"""
        return self.state

    def set_gains(self, Kp: float, Ki: float, Kd: float):
        """在线调整增益"""
        self.config.Kp = Kp
        self.config.Ki = Ki
        self.config.Kd = Kd


class CascadePID:
    """
    串级PID控制器

    外环: 流量控制
    内环: 闸门位置控制
    """

    def __init__(
        self,
        outer_config: Optional[PIDConfig] = None,
        inner_config: Optional[PIDConfig] = None,
        name: str = "CascadePID"
    ):
        """
        初始化串级控制器

        Args:
            outer_config: 外环配置
            inner_config: 内环配置
            name: 控制器名称
        """
        self.name = name

        # 外环 (慢环)
        outer_cfg = outer_config or PIDConfig(Kp=0.5, Ki=0.05, Kd=0.01)
        self.outer = PIDController(outer_cfg, f"{name}_outer")

        # 内环 (快环)
        inner_cfg = inner_config or PIDConfig(Kp=2.0, Ki=0.2, Kd=0.05)
        self.inner = PIDController(inner_cfg, f"{name}_inner")

    def compute(
        self,
        flow_setpoint: float,
        flow_actual: float,
        gate_position: float,
        feedforward: float = 0.0,
        dt: float = 0.01
    ) -> float:
        """
        计算串级控制输出

        Args:
            flow_setpoint: 流量设定值
            flow_actual: 实际流量
            gate_position: 闸门位置
            feedforward: MPC前馈
            dt: 采样时间

        Returns:
            闸门控制量
        """
        # 外环计算 (流量环)
        gate_setpoint = self.outer.compute(
            flow_setpoint,
            flow_actual,
            feedforward,
            dt * 10  # 外环慢10倍
        )

        # 内环计算 (位置环)
        output = self.inner.compute(
            gate_setpoint,
            gate_position,
            0.0,
            dt
        )

        return output

    def reset(self):
        """重置两个环"""
        self.outer.reset()
        self.inner.reset()


class DualTunnelPIDController:
    """
    双洞协同PID控制器

    同时控制两条隧洞的流量平衡
    """

    def __init__(self):
        """初始化双洞控制器"""
        # 总流量控制器
        self.total_flow_pid = PIDController(
            PIDConfig(Kp=0.3, Ki=0.02, Kd=0.01),
            "total_flow"
        )

        # 平衡控制器
        self.balance_pid = PIDController(
            PIDConfig(Kp=1.0, Ki=0.1, Kd=0.02,
                      output_min=-0.1, output_max=0.1),
            "balance"
        )

        # 各洞串级控制器
        self.tunnel1_pid = CascadePID(name="tunnel1")
        self.tunnel2_pid = CascadePID(name="tunnel2")

    def compute(
        self,
        Q_total_sp: float,
        Q1: float,
        Q2: float,
        gate1: float,
        gate2: float,
        mpc_ff: np.ndarray = None,
        dt: float = 0.01
    ) -> Tuple[float, float]:
        """
        计算双洞控制量

        Args:
            Q_total_sp: 总流量设定值
            Q1, Q2: 各洞实际流量
            gate1, gate2: 闸门位置
            mpc_ff: MPC前馈 [ff1, ff2]
            dt: 采样时间

        Returns:
            (控制量1, 控制量2)
        """
        Q_total = Q1 + Q2
        mpc_ff = mpc_ff if mpc_ff is not None else np.array([0.0, 0.0])

        # 总流量控制
        avg_ff = (mpc_ff[0] + mpc_ff[1]) / 2
        base_sp = self.total_flow_pid.compute(
            Q_total_sp,
            Q_total,
            avg_ff,
            dt
        )

        # 平衡控制 (目标: Q1 = Q2)
        imbalance = Q1 - Q2
        balance_adj = self.balance_pid.compute(0.0, imbalance, 0.0, dt)

        # 各洞设定值
        Q1_sp = base_sp / 2 + balance_adj / 2
        Q2_sp = base_sp / 2 - balance_adj / 2

        # 串级控制
        u1 = self.tunnel1_pid.compute(Q1_sp, Q1, gate1, mpc_ff[0], dt)
        u2 = self.tunnel2_pid.compute(Q2_sp, Q2, gate2, mpc_ff[1], dt)

        return u1, u2

    def reset(self):
        """重置所有控制器"""
        self.total_flow_pid.reset()
        self.balance_pid.reset()
        self.tunnel1_pid.reset()
        self.tunnel2_pid.reset()
