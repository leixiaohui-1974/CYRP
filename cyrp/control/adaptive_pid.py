"""
自适应PID控制器 - Adaptive PID Controller

实现多种自适应PID算法:自整定、模糊PID、神经网络PID、IMC-PID
Implements various adaptive PID algorithms: auto-tuning, fuzzy PID, neural network PID, IMC-PID
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque


@dataclass
class PIDParameters:
    """PID参数"""
    Kp: float = 1.0     # 比例增益
    Ki: float = 0.1     # 积分增益
    Kd: float = 0.01    # 微分增益
    Tf: float = 0.1     # 微分滤波时间常数
    N: float = 10.0     # 微分滤波系数


@dataclass
class TuningResult:
    """整定结果"""
    Kp: float
    Ki: float
    Kd: float
    method: str
    performance_index: float
    settling_time: float
    overshoot: float


class TuningMethod(Enum):
    """整定方法"""
    ZIEGLER_NICHOLS = "ziegler_nichols"
    COHEN_COON = "cohen_coon"
    IMC = "imc"
    RELAY_FEEDBACK = "relay_feedback"
    PATTERN_SEARCH = "pattern_search"


class RelayFeedbackTuner:
    """继电反馈自整定"""

    def __init__(self, relay_amplitude: float = 1.0, hysteresis: float = 0.05):
        self.d = relay_amplitude  # 继电器幅值
        self.h = hysteresis       # 滞环宽度

        self.relay_state = 1      # 继电器状态
        self.oscillation_data: List[Tuple[float, float]] = []  # (time, value)
        self.zero_crossings: List[float] = []

        self.tuning_complete = False
        self.Ku = 0.0  # 临界增益
        self.Tu = 0.0  # 临界周期

    def update(self, t: float, error: float) -> float:
        """
        继电反馈更新

        Args:
            t: 当前时间
            error: 误差信号

        Returns:
            继电器输出
        """
        # 带滞环的继电器
        if error > self.h:
            self.relay_state = 1
        elif error < -self.h:
            self.relay_state = -1

        output = self.d * self.relay_state

        # 记录数据
        self.oscillation_data.append((t, error))

        # 检测过零点
        if len(self.oscillation_data) >= 2:
            prev_error = self.oscillation_data[-2][1]
            if prev_error * error < 0:  # 过零
                self.zero_crossings.append(t)

        # 检查是否可以计算临界参数
        if len(self.zero_crossings) >= 6:
            self._calculate_critical_params()

        return output

    def _calculate_critical_params(self):
        """计算临界参数"""
        # 计算周期 (使用后几个周期的平均)
        periods = []
        for i in range(2, len(self.zero_crossings) - 1, 2):
            period = self.zero_crossings[i] - self.zero_crossings[i - 2]
            periods.append(period)

        if len(periods) >= 2:
            self.Tu = np.mean(periods)

            # 计算振幅
            recent_data = [d for d in self.oscillation_data if d[0] > self.zero_crossings[-3]]
            if recent_data:
                amplitudes = [abs(d[1]) for d in recent_data]
                a = np.mean(amplitudes)

                # 临界增益 Ku = 4d / (π * a)
                if a > 0:
                    self.Ku = 4 * self.d / (np.pi * a)
                    self.tuning_complete = True

    def get_pid_params(self, method: str = 'ziegler_nichols') -> PIDParameters:
        """获取PID参数"""
        if not self.tuning_complete:
            return PIDParameters()

        if method == 'ziegler_nichols':
            # Ziegler-Nichols 公式
            Kp = 0.6 * self.Ku
            Ki = Kp / (0.5 * self.Tu)
            Kd = Kp * 0.125 * self.Tu
        elif method == 'tyreus_luyben':
            # Tyreus-Luyben (更保守)
            Kp = 0.45 * self.Ku
            Ki = Kp / (2.2 * self.Tu)
            Kd = Kp * self.Tu / 6.3
        elif method == 'some_overshoot':
            # 一些超调
            Kp = 0.33 * self.Ku
            Ki = Kp / (0.5 * self.Tu)
            Kd = Kp * 0.33 * self.Tu
        else:  # no_overshoot
            Kp = 0.2 * self.Ku
            Ki = Kp / (0.5 * self.Tu)
            Kd = Kp * 0.33 * self.Tu

        return PIDParameters(Kp=Kp, Ki=Ki, Kd=Kd)


class IMCTuner:
    """IMC (Internal Model Control) 整定"""

    def __init__(self, closed_loop_time_constant: float = 1.0):
        self.lambda_c = closed_loop_time_constant

    def tune(self, K: float, tau: float, theta: float) -> PIDParameters:
        """
        IMC-PID整定

        Args:
            K: 过程增益
            tau: 时间常数
            theta: 纯滞后

        Returns:
            PID参数
        """
        if K == 0 or tau == 0:
            return PIDParameters()

        # FOPDT模型的IMC-PID公式
        # λ_c 建议取 max(0.25*tau, 0.8*theta)
        lambda_c = max(self.lambda_c, max(0.25 * tau, 0.8 * theta))

        # PI控制器 (theta较小时)
        if theta < 0.1 * tau:
            Kp = tau / (K * lambda_c)
            Ki = Kp / tau
            Kd = 0
        else:
            # PID控制器
            Kp = (tau + 0.5 * theta) / (K * (lambda_c + 0.5 * theta))
            Ki = Kp / (tau + 0.5 * theta)
            Kd = Kp * tau * theta / (2 * tau + theta)

        return PIDParameters(Kp=Kp, Ki=Ki, Kd=Kd)


class FuzzyPIDController:
    """模糊PID控制器"""

    def __init__(self, base_params: PIDParameters):
        self.base = base_params

        # 当前参数
        self.Kp = base_params.Kp
        self.Ki = base_params.Ki
        self.Kd = base_params.Kd

        # 模糊变量范围
        self.e_range = (-100, 100)    # 误差范围
        self.de_range = (-50, 50)     # 误差变化率范围

        # 模糊规则表 (7x7)
        # 行: e (NB, NM, NS, ZO, PS, PM, PB)
        # 列: de (NB, NM, NS, ZO, PS, PM, PB)
        # 值: (dKp, dKi, dKd) 的模糊输出

        # Kp规则表
        self.Kp_rules = np.array([
            [6, 5, 4, 3, 2, 1, 0],
            [5, 4, 3, 2, 1, 0, -1],
            [4, 3, 2, 1, 0, -1, -2],
            [3, 2, 1, 0, -1, -2, -3],
            [2, 1, 0, -1, -2, -3, -4],
            [1, 0, -1, -2, -3, -4, -5],
            [0, -1, -2, -3, -4, -5, -6],
        ]) * 0.1  # 缩放因子

        # Ki规则表
        self.Ki_rules = np.array([
            [-6, -5, -4, -3, -2, -1, 0],
            [-5, -4, -3, -2, -1, 0, 1],
            [-4, -3, -2, -1, 0, 1, 2],
            [-3, -2, -1, 0, 1, 2, 3],
            [-2, -1, 0, 1, 2, 3, 4],
            [-1, 0, 1, 2, 3, 4, 5],
            [0, 1, 2, 3, 4, 5, 6],
        ]) * 0.05

        # Kd规则表
        self.Kd_rules = np.array([
            [2, 2, 1, 1, 0, 0, 0],
            [2, 1, 1, 0, 0, -1, -1],
            [1, 1, 0, 0, -1, -1, -2],
            [1, 0, 0, 0, 0, 0, -1],
            [-1, -1, 0, 0, 0, 1, 1],
            [-1, -1, -1, 0, 1, 1, 2],
            [-2, -1, -1, 0, 1, 2, 2],
        ]) * 0.02

        # 状态
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_derivative = 0.0

    def _fuzzify(self, value: float, range_: Tuple[float, float]) -> np.ndarray:
        """模糊化 - 返回7个隶属度"""
        min_val, max_val = range_
        # 归一化到 [-1, 1]
        normalized = 2 * (value - min_val) / (max_val - min_val) - 1
        normalized = np.clip(normalized, -1, 1)

        # 7个三角形隶属函数
        centers = np.linspace(-1, 1, 7)
        width = 2 / 6  # 重叠宽度

        memberships = np.zeros(7)
        for i, center in enumerate(centers):
            if normalized >= center - width and normalized <= center + width:
                memberships[i] = 1 - abs(normalized - center) / width

        # 归一化
        if np.sum(memberships) > 0:
            memberships /= np.sum(memberships)

        return memberships

    def _defuzzify(self, rule_output: np.ndarray, memberships_e: np.ndarray,
                   memberships_de: np.ndarray) -> float:
        """解模糊 - 重心法"""
        weighted_sum = 0.0
        weight_total = 0.0

        for i in range(7):
            for j in range(7):
                weight = memberships_e[i] * memberships_de[j]
                weighted_sum += weight * rule_output[i, j]
                weight_total += weight

        if weight_total > 0:
            return weighted_sum / weight_total
        return 0.0

    def compute(self, setpoint: float, measurement: float, dt: float) -> float:
        """
        计算模糊PID输出

        Args:
            setpoint: 设定值
            measurement: 测量值
            dt: 采样时间

        Returns:
            控制输出
        """
        error = setpoint - measurement

        # 误差变化率
        if dt > 0:
            derivative = (error - self.prev_error) / dt
        else:
            derivative = 0

        # 滤波微分
        alpha = 0.1
        filtered_derivative = alpha * derivative + (1 - alpha) * self.prev_derivative

        # 模糊化
        e_memberships = self._fuzzify(error, self.e_range)
        de_memberships = self._fuzzify(filtered_derivative, self.de_range)

        # 模糊推理
        dKp = self._defuzzify(self.Kp_rules, e_memberships, de_memberships)
        dKi = self._defuzzify(self.Ki_rules, e_memberships, de_memberships)
        dKd = self._defuzzify(self.Kd_rules, e_memberships, de_memberships)

        # 更新参数
        self.Kp = self.base.Kp * (1 + dKp)
        self.Ki = self.base.Ki * (1 + dKi)
        self.Kd = self.base.Kd * (1 + dKd)

        # 参数限制
        self.Kp = np.clip(self.Kp, 0.1 * self.base.Kp, 5 * self.base.Kp)
        self.Ki = np.clip(self.Ki, 0.1 * self.base.Ki, 5 * self.base.Ki)
        self.Kd = np.clip(self.Kd, 0, 5 * self.base.Kd)

        # 积分项 (带抗积分饱和)
        self.integral += error * dt

        # PID输出
        output = (self.Kp * error +
                 self.Ki * self.integral +
                 self.Kd * filtered_derivative)

        self.prev_error = error
        self.prev_derivative = filtered_derivative

        return output

    def get_current_params(self) -> PIDParameters:
        """获取当前参数"""
        return PIDParameters(Kp=self.Kp, Ki=self.Ki, Kd=self.Kd)

    def reset(self):
        """重置控制器"""
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_derivative = 0.0
        self.Kp = self.base.Kp
        self.Ki = self.base.Ki
        self.Kd = self.base.Kd


class NeuralNetworkPID:
    """神经网络PID控制器"""

    def __init__(self, n_hidden: int = 10, learning_rate: float = 0.01):
        """
        单隐层神经网络PID

        网络结构: 3输入(e, de, ie) -> 隐层 -> 3输出(Kp, Ki, Kd)
        """
        self.n_input = 3
        self.n_hidden = n_hidden
        self.n_output = 3
        self.lr = learning_rate

        # 权重初始化 (Xavier)
        self.W1 = np.random.randn(self.n_input, self.n_hidden) * np.sqrt(2.0 / self.n_input)
        self.b1 = np.zeros(self.n_hidden)
        self.W2 = np.random.randn(self.n_hidden, self.n_output) * np.sqrt(2.0 / self.n_hidden)
        self.b2 = np.array([1.0, 0.1, 0.01])  # 初始化为合理的PID参数

        # 状态
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_output = 0.0

        # 归一化参数
        self.e_scale = 100.0
        self.de_scale = 50.0
        self.ie_scale = 1000.0

        # 输出缩放
        self.Kp_range = (0.1, 10.0)
        self.Ki_range = (0.01, 1.0)
        self.Kd_range = (0.0, 1.0)

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid激活函数"""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def _sigmoid_derivative(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid导数"""
        s = self._sigmoid(x)
        return s * (1 - s)

    def _forward(self, inputs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """前向传播"""
        # 隐层
        z1 = inputs @ self.W1 + self.b1
        a1 = self._sigmoid(z1)

        # 输出层
        z2 = a1 @ self.W2 + self.b2
        a2 = self._sigmoid(z2)

        return a2, a1, z1

    def compute(self, setpoint: float, measurement: float, dt: float) -> float:
        """
        计算神经网络PID输出

        Args:
            setpoint: 设定值
            measurement: 测量值
            dt: 采样时间

        Returns:
            控制输出
        """
        error = setpoint - measurement

        # 误差变化率
        if dt > 0:
            derivative = (error - self.prev_error) / dt
        else:
            derivative = 0

        # 积分
        self.integral += error * dt

        # 网络输入 (归一化)
        inputs = np.array([
            error / self.e_scale,
            derivative / self.de_scale,
            self.integral / self.ie_scale
        ])

        # 前向传播
        nn_output, self.hidden_activation, self.hidden_z = self._forward(inputs)

        # 反归一化得到PID参数
        Kp = self.Kp_range[0] + nn_output[0] * (self.Kp_range[1] - self.Kp_range[0])
        Ki = self.Ki_range[0] + nn_output[1] * (self.Ki_range[1] - self.Ki_range[0])
        Kd = self.Kd_range[0] + nn_output[2] * (self.Kd_range[1] - self.Kd_range[0])

        # PID输出
        output = Kp * error + Ki * self.integral + Kd * derivative

        # 保存状态供反向传播
        self.last_inputs = inputs
        self.last_nn_output = nn_output
        self.last_Kp = Kp
        self.last_Ki = Ki
        self.last_Kd = Kd
        self.last_error = error
        self.last_derivative = derivative

        self.prev_error = error
        self.prev_output = output

        return output

    def learn(self, new_measurement: float, dt: float):
        """
        在线学习 - 基于性能误差的反向传播

        Args:
            new_measurement: 新的测量值 (用于评估控制效果)
            dt: 采样时间
        """
        if not hasattr(self, 'last_inputs'):
            return

        # 计算性能误差 (使用输出变化方向作为梯度估计)
        # 目标: 最小化控制误差
        error_gradient = -self.last_error  # 误差减小的方向

        # 简化的梯度估计
        # ∂J/∂Kp ≈ -error * (∂u/∂Kp) = -error * e
        # ∂J/∂Ki ≈ -error * (∂u/∂Ki) = -error * integral
        # ∂J/∂Kd ≈ -error * (∂u/∂Kd) = -error * de

        dJ_dKp = -self.last_error * self.last_error
        dJ_dKi = -self.last_error * self.integral
        dJ_dKd = -self.last_error * self.last_derivative

        # 链式法则: ∂J/∂W = ∂J/∂output * ∂output/∂W
        dJ_d_nn = np.array([
            dJ_dKp * (self.Kp_range[1] - self.Kp_range[0]),
            dJ_dKi * (self.Ki_range[1] - self.Ki_range[0]),
            dJ_dKd * (self.Kd_range[1] - self.Kd_range[0])
        ])

        # 输出层梯度
        d_sigmoid_output = self.last_nn_output * (1 - self.last_nn_output)
        delta2 = dJ_d_nn * d_sigmoid_output

        # 隐层梯度
        delta1 = (delta2 @ self.W2.T) * self._sigmoid_derivative(self.hidden_z)

        # 权重更新
        self.W2 -= self.lr * np.outer(self.hidden_activation, delta2)
        self.b2 -= self.lr * delta2
        self.W1 -= self.lr * np.outer(self.last_inputs, delta1)
        self.b1 -= self.lr * delta1

    def get_current_params(self) -> PIDParameters:
        """获取当前参数"""
        if hasattr(self, 'last_Kp'):
            return PIDParameters(Kp=self.last_Kp, Ki=self.last_Ki, Kd=self.last_Kd)
        return PIDParameters()

    def reset(self):
        """重置状态"""
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_output = 0.0


class AdaptivePIDController:
    """综合自适应PID控制器"""

    def __init__(self, initial_params: Optional[PIDParameters] = None):
        if initial_params is None:
            initial_params = PIDParameters(Kp=1.0, Ki=0.1, Kd=0.05)

        self.params = initial_params

        # 控制器状态
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_derivative = 0.0
        self.prev_output = 0.0

        # 自适应组件
        self.relay_tuner = RelayFeedbackTuner()
        self.imc_tuner = IMCTuner()
        self.fuzzy_pid = FuzzyPIDController(initial_params)
        self.nn_pid = NeuralNetworkPID()

        # 模式
        self.mode = 'standard'  # 'standard', 'tuning', 'fuzzy', 'neural'

        # 性能监测
        self.performance_buffer = deque(maxlen=100)
        self.iae = 0.0  # 积分绝对误差
        self.ise = 0.0  # 积分平方误差

        # 输出限制
        self.output_min = -np.inf
        self.output_max = np.inf

        # 抗积分饱和
        self.anti_windup_enabled = True
        self.output_saturated = False

    def compute(self, setpoint: float, measurement: float, dt: float) -> float:
        """
        计算控制输出

        Args:
            setpoint: 设定值
            measurement: 测量值
            dt: 采样时间

        Returns:
            控制输出
        """
        if self.mode == 'tuning':
            # 继电反馈整定模式
            error = setpoint - measurement
            return self.relay_tuner.update(dt, error)

        elif self.mode == 'fuzzy':
            # 模糊PID模式
            output = self.fuzzy_pid.compute(setpoint, measurement, dt)
            self.params = self.fuzzy_pid.get_current_params()

        elif self.mode == 'neural':
            # 神经网络PID模式
            output = self.nn_pid.compute(setpoint, measurement, dt)
            self.params = self.nn_pid.get_current_params()

        else:
            # 标准PID
            output = self._standard_pid(setpoint, measurement, dt)

        # 输出限制
        output_unlimited = output
        output = np.clip(output, self.output_min, self.output_max)
        self.output_saturated = (output != output_unlimited)

        # 性能监测
        error = setpoint - measurement
        self.iae += abs(error) * dt
        self.ise += error ** 2 * dt
        self.performance_buffer.append(abs(error))

        self.prev_output = output
        return output

    def _standard_pid(self, setpoint: float, measurement: float, dt: float) -> float:
        """标准PID计算"""
        error = setpoint - measurement

        # 比例项
        P = self.params.Kp * error

        # 积分项 (带条件积分)
        if self.anti_windup_enabled and self.output_saturated:
            # 积分饱和时停止积分
            pass
        else:
            self.integral += error * dt

        I = self.params.Ki * self.integral

        # 微分项 (带滤波)
        if dt > 0:
            derivative_raw = (error - self.prev_error) / dt
        else:
            derivative_raw = 0

        # 一阶滤波
        alpha = dt / (self.params.Tf + dt)
        derivative = alpha * derivative_raw + (1 - alpha) * self.prev_derivative

        D = self.params.Kd * derivative

        self.prev_error = error
        self.prev_derivative = derivative

        return P + I + D

    def start_auto_tuning(self, relay_amplitude: float = 1.0):
        """开始自动整定"""
        self.relay_tuner = RelayFeedbackTuner(relay_amplitude=relay_amplitude)
        self.mode = 'tuning'

    def finish_auto_tuning(self, method: str = 'ziegler_nichols') -> PIDParameters:
        """完成自动整定"""
        if self.relay_tuner.tuning_complete:
            self.params = self.relay_tuner.get_pid_params(method)
            self.fuzzy_pid.base = self.params
        self.mode = 'standard'
        return self.params

    def tune_with_imc(self, K: float, tau: float, theta: float,
                     lambda_c: Optional[float] = None):
        """IMC整定"""
        if lambda_c is not None:
            self.imc_tuner = IMCTuner(lambda_c)

        self.params = self.imc_tuner.tune(K, tau, theta)
        self.fuzzy_pid.base = self.params

    def set_mode(self, mode: str):
        """设置控制模式"""
        if mode in ['standard', 'tuning', 'fuzzy', 'neural']:
            self.mode = mode
            if mode == 'standard':
                self.reset()

    def set_parameters(self, Kp: float, Ki: float, Kd: float):
        """设置PID参数"""
        self.params = PIDParameters(Kp=Kp, Ki=Ki, Kd=Kd)
        self.fuzzy_pid.base = self.params

    def set_output_limits(self, min_val: float, max_val: float):
        """设置输出限制"""
        self.output_min = min_val
        self.output_max = max_val

    def get_performance(self) -> Dict[str, float]:
        """获取性能指标"""
        return {
            'IAE': self.iae,
            'ISE': self.ise,
            'mean_error': np.mean(self.performance_buffer) if self.performance_buffer else 0,
            'max_error': max(self.performance_buffer) if self.performance_buffer else 0,
        }

    def reset(self):
        """重置控制器"""
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_derivative = 0.0
        self.prev_output = 0.0
        self.iae = 0.0
        self.ise = 0.0
        self.performance_buffer.clear()
        self.output_saturated = False
        self.fuzzy_pid.reset()
        self.nn_pid.reset()


class DualTunnelAdaptivePID:
    """双洞自适应PID控制器"""

    def __init__(self):
        # 两条隧道的控制器
        self.north_controller = AdaptivePIDController(
            PIDParameters(Kp=2.0, Ki=0.2, Kd=0.1)
        )
        self.south_controller = AdaptivePIDController(
            PIDParameters(Kp=2.0, Ki=0.2, Kd=0.1)
        )

        # 协调控制器
        self.balance_controller = AdaptivePIDController(
            PIDParameters(Kp=1.0, Ki=0.1, Kd=0.05)
        )

        # 模式
        self.mode = 'dual'  # 'north_only', 'south_only', 'dual'

        # 平衡系数
        self.balance_factor = 0.5  # 0.5 = 均衡, <0.5 偏向北洞, >0.5 偏向南洞

    def compute(self, setpoint: float,
               north_measurement: float,
               south_measurement: float,
               dt: float) -> Tuple[float, float]:
        """
        计算双洞控制输出

        Args:
            setpoint: 总流量设定值
            north_measurement: 北洞流量测量
            south_measurement: 南洞流量测量
            dt: 采样时间

        Returns:
            (北洞控制, 南洞控制)
        """
        total_flow = north_measurement + south_measurement

        if self.mode == 'north_only':
            north_output = self.north_controller.compute(setpoint, north_measurement, dt)
            return north_output, 0.0

        elif self.mode == 'south_only':
            south_output = self.south_controller.compute(setpoint, south_measurement, dt)
            return 0.0, south_output

        else:  # dual
            # 分配设定值
            north_setpoint = setpoint * self.balance_factor
            south_setpoint = setpoint * (1 - self.balance_factor)

            # 各洞控制
            north_output = self.north_controller.compute(
                north_setpoint, north_measurement, dt
            )
            south_output = self.south_controller.compute(
                south_setpoint, south_measurement, dt
            )

            # 平衡调节
            balance_error = north_measurement - south_measurement
            balance_correction = self.balance_controller.compute(0, balance_error, dt)

            # 应用平衡修正
            north_output -= balance_correction * 0.1
            south_output += balance_correction * 0.1

            return north_output, south_output

    def set_mode(self, mode: str, balance_factor: float = 0.5):
        """设置运行模式"""
        self.mode = mode
        self.balance_factor = balance_factor

    def set_all_modes(self, controller_mode: str):
        """设置所有控制器的模式"""
        self.north_controller.set_mode(controller_mode)
        self.south_controller.set_mode(controller_mode)
        self.balance_controller.set_mode(controller_mode)

    def reset(self):
        """重置所有控制器"""
        self.north_controller.reset()
        self.south_controller.reset()
        self.balance_controller.reset()
