"""
自适应MPC控制器 - Adaptive MPC Controller

实现在线模型辨识、增益调度、自整定MPC
Implements online model identification, gain scheduling, self-tuning MPC
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque


@dataclass
class SystemIdentificationResult:
    """系统辨识结果"""
    A: np.ndarray           # 状态矩阵
    B: np.ndarray           # 输入矩阵
    C: np.ndarray           # 输出矩阵
    D: np.ndarray           # 直接传递矩阵
    time_constant: float    # 时间常数
    gain: float             # 稳态增益
    delay: int              # 纯滞后步数
    fit_score: float        # 拟合度
    covariance: np.ndarray  # 参数协方差


class IdentificationMethod(Enum):
    """辨识方法"""
    RLS = "recursive_least_squares"      # 递推最小二乘
    ARMAX = "armax"                       # ARMAX模型
    SUBSPACE = "subspace"                 # 子空间辨识
    NEURAL = "neural_network"             # 神经网络辨识


class RecursiveLeastSquares:
    """递推最小二乘辨识器"""

    def __init__(self, n_params: int, forgetting_factor: float = 0.98):
        self.n_params = n_params
        self.lambda_ = forgetting_factor

        # 参数估计
        self.theta = np.zeros(n_params)
        # 协方差矩阵
        self.P = np.eye(n_params) * 1000

        # 历史数据
        self.initialized = False

    def update(self, phi: np.ndarray, y: float) -> np.ndarray:
        """
        递推更新参数估计

        Args:
            phi: 回归向量 [n_params]
            y: 输出测量值

        Returns:
            更新后的参数估计
        """
        phi = phi.reshape(-1, 1)

        # 预测误差
        y_pred = float(phi.T @ self.theta)
        e = y - y_pred

        # 增益计算
        denominator = self.lambda_ + float(phi.T @ self.P @ phi)
        K = (self.P @ phi) / denominator

        # 参数更新
        self.theta = self.theta + (K * e).flatten()

        # 协方差更新
        self.P = (self.P - K @ phi.T @ self.P) / self.lambda_

        self.initialized = True

        return self.theta

    def get_parameters(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取参数估计和协方差"""
        return self.theta, self.P

    def reset(self):
        """重置辨识器"""
        self.theta = np.zeros(self.n_params)
        self.P = np.eye(self.n_params) * 1000
        self.initialized = False


class ARMAXIdentifier:
    """ARMAX模型辨识器"""

    def __init__(self, na: int = 2, nb: int = 2, nc: int = 1, nk: int = 1):
        """
        ARMAX: A(q)y(t) = B(q)u(t-nk) + C(q)e(t)

        Args:
            na: 自回归阶数
            nb: 输入阶数
            nc: 移动平均阶数
            nk: 纯滞后
        """
        self.na = na
        self.nb = nb
        self.nc = nc
        self.nk = nk

        self.n_params = na + nb + nc
        self.rls = RecursiveLeastSquares(self.n_params, forgetting_factor=0.98)

        # 历史数据
        self.y_history = deque(maxlen=max(na, nc) + 1)
        self.u_history = deque(maxlen=nb + nk)
        self.e_history = deque(maxlen=nc + 1)

        # 填充初始值
        for _ in range(max(na, nc) + 1):
            self.y_history.append(0)
            self.e_history.append(0)
        for _ in range(nb + nk):
            self.u_history.append(0)

    def update(self, y: float, u: float) -> SystemIdentificationResult:
        """更新辨识"""
        # 构建回归向量
        phi = self._build_regressor()

        # RLS更新
        theta = self.rls.update(phi, y)

        # 更新历史
        y_pred = float(phi @ theta)
        e = y - y_pred

        self.y_history.append(y)
        self.u_history.append(u)
        self.e_history.append(e)

        # 解析参数
        return self._parse_parameters(theta)

    def _build_regressor(self) -> np.ndarray:
        """构建回归向量"""
        phi = []

        # -y(t-1), ..., -y(t-na)
        for i in range(1, self.na + 1):
            idx = -(i)
            if abs(idx) <= len(self.y_history):
                phi.append(-self.y_history[idx])
            else:
                phi.append(0)

        # u(t-nk), ..., u(t-nk-nb+1)
        for i in range(self.nb):
            idx = -(self.nk + i)
            if abs(idx) <= len(self.u_history):
                phi.append(self.u_history[idx])
            else:
                phi.append(0)

        # e(t-1), ..., e(t-nc)
        for i in range(1, self.nc + 1):
            idx = -(i)
            if abs(idx) <= len(self.e_history):
                phi.append(self.e_history[idx])
            else:
                phi.append(0)

        return np.array(phi)

    def _parse_parameters(self, theta: np.ndarray) -> SystemIdentificationResult:
        """解析ARMAX参数为状态空间模型"""
        a = theta[:self.na]  # AR系数
        b = theta[self.na:self.na + self.nb]  # 输入系数
        c = theta[self.na + self.nb:]  # MA系数

        # 转换为状态空间 (可观标准型)
        n = max(self.na, self.nb)

        A = np.zeros((n, n))
        if n > 1:
            A[1:, :-1] = np.eye(n - 1)
        A[0, :self.na] = -a

        B = np.zeros((n, 1))
        B[:self.nb, 0] = b

        C = np.zeros((1, n))
        C[0, 0] = 1

        D = np.zeros((1, 1))

        # 计算时间常数和增益
        if abs(1 + np.sum(a)) > 1e-6:
            gain = np.sum(b) / (1 + np.sum(a))
        else:
            gain = 0

        # 近似时间常数
        poles = np.roots(np.concatenate([[1], a]))
        real_poles = poles[np.isreal(poles)].real
        if len(real_poles) > 0 and np.max(np.abs(real_poles)) < 1:
            dominant_pole = real_poles[np.argmax(np.abs(real_poles))]
            if dominant_pole != 0:
                time_constant = -1 / np.log(abs(dominant_pole))
            else:
                time_constant = 0.1
        else:
            time_constant = 1.0

        # 拟合度评估
        theta_params, P = self.rls.get_parameters()
        fit_score = 1.0 / (1.0 + np.trace(P))

        return SystemIdentificationResult(
            A=A, B=B, C=C, D=D,
            time_constant=time_constant,
            gain=gain,
            delay=self.nk,
            fit_score=fit_score,
            covariance=P
        )


class GainScheduler:
    """增益调度器"""

    def __init__(self):
        # 操作点和对应的控制参数
        self.operating_points: List[Dict[str, float]] = []
        self.gain_sets: List[Dict[str, float]] = []

        self._init_default_schedule()

    def _init_default_schedule(self):
        """初始化默认调度表"""
        # 低流量工况
        self.operating_points.append({'flow_rate': 100, 'pressure': 0.3})
        self.gain_sets.append({
            'Q_weight': 100, 'R_weight': 1, 'horizon': 20,
            'terminal_weight': 10, 'constraint_softening': 0.1
        })

        # 中流量工况
        self.operating_points.append({'flow_rate': 200, 'pressure': 0.45})
        self.gain_sets.append({
            'Q_weight': 150, 'R_weight': 1, 'horizon': 25,
            'terminal_weight': 15, 'constraint_softening': 0.05
        })

        # 高流量工况
        self.operating_points.append({'flow_rate': 280, 'pressure': 0.6})
        self.gain_sets.append({
            'Q_weight': 200, 'R_weight': 2, 'horizon': 30,
            'terminal_weight': 20, 'constraint_softening': 0.02
        })

        # 紧急工况
        self.operating_points.append({'flow_rate': 265, 'pressure': 0.8})
        self.gain_sets.append({
            'Q_weight': 300, 'R_weight': 0.5, 'horizon': 15,
            'terminal_weight': 50, 'constraint_softening': 0.01
        })

    def schedule(self, operating_point: Dict[str, float]) -> Dict[str, float]:
        """根据操作点调度增益"""
        if len(self.operating_points) == 0:
            return self.gain_sets[0] if self.gain_sets else {}

        # 计算与各操作点的距离
        distances = []
        for op in self.operating_points:
            dist = 0
            for key in operating_point:
                if key in op:
                    # 归一化距离
                    dist += ((operating_point[key] - op[key]) / (op[key] + 1e-6)) ** 2
            distances.append(np.sqrt(dist))

        # 加权插值
        distances = np.array(distances)
        weights = 1.0 / (distances + 1e-6)
        weights /= np.sum(weights)

        # 插值计算增益
        result = {}
        for key in self.gain_sets[0]:
            result[key] = sum(w * gs[key] for w, gs in zip(weights, self.gain_sets))

        return result

    def add_operating_point(self, op: Dict[str, float], gains: Dict[str, float]):
        """添加操作点"""
        self.operating_points.append(op)
        self.gain_sets.append(gains)


class AdaptiveMPCController:
    """自适应MPC控制器"""

    def __init__(self, n_states: int = 2, n_inputs: int = 1, n_outputs: int = 1):
        self.n_states = n_states
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        # 系统模型 (初始模型)
        self.A = np.eye(n_states) * 0.9
        self.B = np.ones((n_states, n_inputs)) * 0.1
        self.C = np.eye(n_outputs, n_states)
        self.D = np.zeros((n_outputs, n_inputs))

        # MPC参数
        self.horizon = 20
        self.Q = np.eye(n_outputs) * 100      # 输出权重
        self.R = np.eye(n_inputs) * 1         # 输入权重
        self.Qf = np.eye(n_outputs) * 50      # 终端权重

        # 约束
        self.u_min = np.array([-10.0])
        self.u_max = np.array([10.0])
        self.du_max = np.array([1.0])
        self.y_min = np.array([-np.inf])
        self.y_max = np.array([np.inf])

        # 自适应组件
        self.identifier = ARMAXIdentifier(na=2, nb=2, nc=1, nk=1)
        self.gain_scheduler = GainScheduler()

        # 状态
        self.x = np.zeros(n_states)
        self.u_prev = np.zeros(n_inputs)

        # 适应参数
        self.model_update_interval = 10
        self.step_count = 0
        self.adaptation_enabled = True

        # 历史
        self.model_history: List[SystemIdentificationResult] = []
        self.performance_history: List[float] = []

    def compute_control(self, y_ref: np.ndarray, y_meas: np.ndarray,
                       disturbance: Optional[np.ndarray] = None) -> np.ndarray:
        """
        计算控制输入

        Args:
            y_ref: 参考输出
            y_meas: 测量输出
            disturbance: 可测扰动

        Returns:
            控制输入
        """
        # 状态估计
        self._update_state(y_meas)

        # 增益调度
        operating_point = {'flow_rate': y_meas[0], 'pressure': 0.5}
        gains = self.gain_scheduler.schedule(operating_point)
        self._apply_gains(gains)

        # 构建QP问题
        H, f, A_ineq, b_ineq = self._build_qp(y_ref, disturbance)

        # 求解QP
        u_opt = self._solve_qp(H, f, A_ineq, b_ineq)

        # 应用第一步控制
        u = u_opt[:self.n_inputs]

        # 约束处理
        u = np.clip(u, self.u_min, self.u_max)

        # 速率约束
        du = u - self.u_prev
        du = np.clip(du, -self.du_max, self.du_max)
        u = self.u_prev + du

        self.u_prev = u.copy()
        self.step_count += 1

        return u

    def adapt(self, y_meas: float, u_applied: float):
        """
        在线适应

        Args:
            y_meas: 测量输出
            u_applied: 应用的控制输入
        """
        if not self.adaptation_enabled:
            return

        # 系统辨识
        id_result = self.identifier.update(y_meas, u_applied)
        self.model_history.append(id_result)

        # 定期更新模型
        if self.step_count % self.model_update_interval == 0:
            if id_result.fit_score > 0.5:
                self._update_model(id_result)

        # 性能监测
        self._monitor_performance(y_meas)

    def _update_state(self, y_meas: np.ndarray):
        """状态估计更新"""
        # 简单的输出反馈
        y_pred = self.C @ self.x
        error = y_meas - y_pred

        # 观测器增益
        L = np.array([[0.5], [0.3]])
        self.x = self.A @ self.x + L @ error.reshape(-1, 1)
        self.x = self.x.flatten()

    def _apply_gains(self, gains: Dict[str, float]):
        """应用调度增益"""
        if 'Q_weight' in gains:
            self.Q = np.eye(self.n_outputs) * gains['Q_weight']
        if 'R_weight' in gains:
            self.R = np.eye(self.n_inputs) * gains['R_weight']
        if 'horizon' in gains:
            self.horizon = int(gains['horizon'])
        if 'terminal_weight' in gains:
            self.Qf = np.eye(self.n_outputs) * gains['terminal_weight']

    def _build_qp(self, y_ref: np.ndarray,
                  disturbance: Optional[np.ndarray]) -> Tuple[np.ndarray, ...]:
        """构建二次规划问题"""
        N = self.horizon
        nu = self.n_inputs
        ny = self.n_outputs
        nx = self.n_states

        # 预测矩阵
        Psi = np.zeros((N * ny, nx))
        Theta = np.zeros((N * ny, N * nu))

        # 构建预测矩阵
        Ai = np.eye(nx)
        for i in range(N):
            Ai = Ai @ self.A
            Psi[i * ny:(i + 1) * ny, :] = self.C @ Ai

            for j in range(i + 1):
                Aj = np.linalg.matrix_power(self.A, i - j)
                Theta[i * ny:(i + 1) * ny, j * nu:(j + 1) * nu] = self.C @ Aj @ self.B

        # 构建代价矩阵
        Q_bar = np.kron(np.eye(N), self.Q)
        Q_bar[-ny:, -ny:] = self.Qf  # 终端权重
        R_bar = np.kron(np.eye(N), self.R)

        # Hessian: H = Theta'*Q*Theta + R
        H = Theta.T @ Q_bar @ Theta + R_bar

        # 参考轨迹
        Y_ref = np.tile(y_ref, N)

        # 线性项: f = -Theta'*Q*(Y_ref - Psi*x)
        f = -Theta.T @ Q_bar @ (Y_ref - Psi @ self.x)

        # 不等式约束
        n_constraints = 2 * N * nu  # 输入上下界
        A_ineq = np.zeros((n_constraints, N * nu))
        b_ineq = np.zeros(n_constraints)

        for i in range(N):
            # u <= u_max
            A_ineq[i * nu:(i + 1) * nu, i * nu:(i + 1) * nu] = np.eye(nu)
            b_ineq[i * nu:(i + 1) * nu] = self.u_max

            # -u <= -u_min
            idx = N * nu + i * nu
            A_ineq[idx:idx + nu, i * nu:(i + 1) * nu] = -np.eye(nu)
            b_ineq[idx:idx + nu] = -self.u_min

        return H, f, A_ineq, b_ineq

    def _solve_qp(self, H: np.ndarray, f: np.ndarray,
                  A_ineq: np.ndarray, b_ineq: np.ndarray) -> np.ndarray:
        """求解QP问题 (简化的梯度投影法)"""
        n = len(f)
        u = np.zeros(n)

        # 正则化
        H = H + np.eye(n) * 1e-6

        # 梯度下降迭代
        max_iter = 100
        alpha = 0.01

        for _ in range(max_iter):
            grad = H @ u + f

            # 梯度步
            u_new = u - alpha * grad

            # 投影到约束集
            for i in range(len(u_new) // self.n_inputs):
                idx = i * self.n_inputs
                u_new[idx:idx + self.n_inputs] = np.clip(
                    u_new[idx:idx + self.n_inputs],
                    self.u_min, self.u_max
                )

            u = u_new

        return u

    def _update_model(self, id_result: SystemIdentificationResult):
        """更新系统模型"""
        # 渐进更新 (避免突变)
        alpha = 0.3  # 学习率

        n = min(self.n_states, id_result.A.shape[0])

        self.A[:n, :n] = (1 - alpha) * self.A[:n, :n] + alpha * id_result.A[:n, :n]
        self.B[:n, :1] = (1 - alpha) * self.B[:n, :1] + alpha * id_result.B[:n, :1]

    def _monitor_performance(self, y_meas: float):
        """性能监测"""
        if len(self.model_history) > 10:
            # 计算预测误差
            recent_fits = [m.fit_score for m in self.model_history[-10:]]
            avg_fit = np.mean(recent_fits)

            self.performance_history.append(avg_fit)

            # 如果性能下降,增加辨识激励
            if len(self.performance_history) > 20:
                recent_perf = np.mean(self.performance_history[-10:])
                past_perf = np.mean(self.performance_history[-20:-10])

                if recent_perf < past_perf * 0.9:
                    # 性能下降,可能需要重新辨识
                    self.identifier.rls.reset()

    def set_constraints(self, u_min: np.ndarray, u_max: np.ndarray,
                       du_max: np.ndarray,
                       y_min: Optional[np.ndarray] = None,
                       y_max: Optional[np.ndarray] = None):
        """设置约束"""
        self.u_min = u_min
        self.u_max = u_max
        self.du_max = du_max
        if y_min is not None:
            self.y_min = y_min
        if y_max is not None:
            self.y_max = y_max

    def enable_adaptation(self, enable: bool = True):
        """启用/禁用自适应"""
        self.adaptation_enabled = enable

    def get_model(self) -> Dict[str, np.ndarray]:
        """获取当前模型"""
        return {
            'A': self.A.copy(),
            'B': self.B.copy(),
            'C': self.C.copy(),
            'D': self.D.copy()
        }


class RobustAdaptiveMPC(AdaptiveMPCController):
    """鲁棒自适应MPC"""

    def __init__(self, n_states: int = 2, n_inputs: int = 1, n_outputs: int = 1):
        super().__init__(n_states, n_inputs, n_outputs)

        # 模型不确定性边界
        self.delta_A = np.ones_like(self.A) * 0.1
        self.delta_B = np.ones_like(self.B) * 0.1

        # 鲁棒性参数
        self.robustness_margin = 0.1
        self.constraint_tightening = 0.05

    def compute_control(self, y_ref: np.ndarray, y_meas: np.ndarray,
                       disturbance: Optional[np.ndarray] = None) -> np.ndarray:
        """鲁棒MPC控制"""
        # 约束收紧
        original_u_max = self.u_max.copy()
        original_u_min = self.u_min.copy()

        self.u_max = self.u_max * (1 - self.constraint_tightening)
        self.u_min = self.u_min * (1 - self.constraint_tightening)

        # 调用基类控制
        u = super().compute_control(y_ref, y_meas, disturbance)

        # 恢复约束
        self.u_max = original_u_max
        self.u_min = original_u_min

        return u

    def update_uncertainty(self, prediction_error: float):
        """更新不确定性估计"""
        # 根据预测误差调整不确定性边界
        error_norm = abs(prediction_error)

        # 自适应调整
        alpha = 0.1
        self.delta_A = (1 - alpha) * self.delta_A + alpha * error_norm * np.ones_like(self.A)
        self.delta_B = (1 - alpha) * self.delta_B + alpha * error_norm * np.ones_like(self.B)

        # 更新约束收紧因子
        self.constraint_tightening = min(0.2, self.constraint_tightening + 0.01 * error_norm)


class ScenarioAdaptiveMPC(AdaptiveMPCController):
    """场景自适应MPC - 根据场景切换控制策略"""

    def __init__(self):
        super().__init__(n_states=2, n_inputs=1, n_outputs=1)

        # 场景特定的控制配置
        self.scenario_configs = {
            # 常规运行
            'S1-A': {'Q': 100, 'R': 1, 'horizon': 25, 'constraint_mode': 'normal'},
            'S1-B': {'Q': 100, 'R': 1, 'horizon': 25, 'constraint_mode': 'normal'},
            'S2-A': {'Q': 120, 'R': 1, 'horizon': 25, 'constraint_mode': 'normal'},

            # 切换过程
            'S3-A': {'Q': 150, 'R': 0.5, 'horizon': 15, 'constraint_mode': 'relaxed'},
            'S3-B': {'Q': 200, 'R': 0.3, 'horizon': 10, 'constraint_mode': 'emergency'},

            # 检修模式
            'S4-A': {'Q': 80, 'R': 2, 'horizon': 30, 'constraint_mode': 'conservative'},

            # 渗漏场景
            'S5-A': {'Q': 150, 'R': 0.8, 'horizon': 20, 'constraint_mode': 'normal'},
            'S5-B': {'Q': 200, 'R': 0.5, 'horizon': 15, 'constraint_mode': 'alert'},
            'S5-C': {'Q': 300, 'R': 0.3, 'horizon': 10, 'constraint_mode': 'emergency'},

            # 地震场景
            'S6-A': {'Q': 200, 'R': 0.5, 'horizon': 15, 'constraint_mode': 'alert'},
            'S6-B': {'Q': 300, 'R': 0.2, 'horizon': 10, 'constraint_mode': 'emergency'},
            'S6-C': {'Q': 500, 'R': 0.1, 'horizon': 5, 'constraint_mode': 'critical'},
        }

        self.current_scenario = 'S1-A'
        self.transition_smoothing = 0.3

    def set_scenario(self, scenario_id: str):
        """设置当前场景"""
        if scenario_id in self.scenario_configs:
            config = self.scenario_configs[scenario_id]

            # 平滑过渡
            target_Q = config['Q']
            target_R = config['R']

            current_Q = self.Q[0, 0]
            current_R = self.R[0, 0]

            self.Q = np.eye(self.n_outputs) * (
                (1 - self.transition_smoothing) * current_Q +
                self.transition_smoothing * target_Q
            )
            self.R = np.eye(self.n_inputs) * (
                (1 - self.transition_smoothing) * current_R +
                self.transition_smoothing * target_R
            )
            self.horizon = config['horizon']

            # 约束模式
            self._apply_constraint_mode(config['constraint_mode'])

            self.current_scenario = scenario_id

    def _apply_constraint_mode(self, mode: str):
        """应用约束模式"""
        if mode == 'normal':
            self.du_max = np.array([1.0])
            self.constraint_tightening = 0.0
        elif mode == 'relaxed':
            self.du_max = np.array([2.0])
            self.constraint_tightening = -0.1  # 放宽
        elif mode == 'conservative':
            self.du_max = np.array([0.5])
            self.constraint_tightening = 0.1
        elif mode == 'alert':
            self.du_max = np.array([1.5])
            self.constraint_tightening = 0.05
        elif mode == 'emergency':
            self.du_max = np.array([3.0])
            self.constraint_tightening = -0.15
        elif mode == 'critical':
            self.du_max = np.array([5.0])
            self.constraint_tightening = -0.2
