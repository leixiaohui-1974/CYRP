"""
Model Predictive Controllers for CYRP.
穿黄工程模型预测控制器

包含:
- LTV MPC: 线性时变MPC (常态工况)
- NMPC: 非线性MPC (气液两相)
- Robust MPC: 鲁棒MPC (应急工况)
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from abc import ABC, abstractmethod
import numpy as np
from scipy.linalg import solve, block_diag
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint


@dataclass
class MPCConfig:
    """MPC配置参数"""
    # 预测与控制时域
    prediction_horizon: int = 20  # 预测步数 Np
    control_horizon: int = 10  # 控制步数 Nc

    # 采样时间
    sample_time: float = 1.0  # 秒

    # 权重矩阵
    Q: np.ndarray = field(default_factory=lambda: np.eye(4))  # 状态权重
    R: np.ndarray = field(default_factory=lambda: np.eye(2) * 0.01)  # 控制权重
    Rd: np.ndarray = field(default_factory=lambda: np.eye(2) * 0.1)  # 控制增量权重

    # 终端权重
    Qf: Optional[np.ndarray] = None

    # 约束
    u_min: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    u_max: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0]))
    du_max: np.ndarray = field(default_factory=lambda: np.array([0.01, 0.01]))

    # 软约束松弛
    slack_weight: float = 1000.0


@dataclass
class MPCResult:
    """MPC求解结果"""
    success: bool = False
    u_optimal: np.ndarray = field(default_factory=lambda: np.zeros(2))
    u_sequence: np.ndarray = field(default_factory=lambda: np.zeros((10, 2)))
    x_predicted: np.ndarray = field(default_factory=lambda: np.zeros((20, 4)))
    cost: float = 0.0
    solve_time: float = 0.0
    iterations: int = 0
    message: str = ""


class MPCController(ABC):
    """
    MPC控制器基类
    """

    def __init__(self, config: Optional[MPCConfig] = None):
        """
        初始化MPC控制器

        Args:
            config: MPC配置
        """
        self.config = config or MPCConfig()
        self.Np = self.config.prediction_horizon
        self.Nc = self.config.control_horizon
        self.Ts = self.config.sample_time

        # 状态维度 (将在子类中设置)
        self.nx = 4  # [Q1, Q2, H_in, H_out]
        self.nu = 2  # [e1, e2] 闸门开度

        # 上一时刻控制量
        self.u_prev = np.array([1.0, 1.0])

        # 参考轨迹
        self.x_ref = np.zeros((self.Np, self.nx))
        self.u_ref = np.zeros((self.Np, self.nu))

    @abstractmethod
    def solve(
        self,
        x0: np.ndarray,
        x_ref: np.ndarray,
        u_ref: Optional[np.ndarray] = None
    ) -> MPCResult:
        """
        求解MPC优化问题

        Args:
            x0: 当前状态
            x_ref: 参考轨迹
            u_ref: 参考控制

        Returns:
            MPC求解结果
        """
        pass

    def update_reference(
        self,
        x_ref: np.ndarray,
        u_ref: Optional[np.ndarray] = None
    ):
        """更新参考轨迹"""
        self.x_ref = x_ref
        if u_ref is not None:
            self.u_ref = u_ref

    def get_control(
        self,
        x0: np.ndarray,
        x_ref: np.ndarray
    ) -> np.ndarray:
        """
        获取控制量

        Args:
            x0: 当前状态
            x_ref: 参考状态

        Returns:
            控制量
        """
        result = self.solve(x0, x_ref)
        if result.success:
            self.u_prev = result.u_optimal
            return result.u_optimal
        else:
            return self.u_prev


class LTVMPCController(MPCController):
    """
    线性时变MPC控制器

    适用场景: S1/S2 常态运行
    模型: 基于圣维南方程离散化的状态空间模型

    x_{k+1} = A_k * x_k + B_k * u_k + d_k

    状态: x = [Q1, Q2, H_in, H_out]
    控制: u = [e1, e2] (闸门开度变化率)
    """

    def __init__(
        self,
        config: Optional[MPCConfig] = None,
        tunnel_length: float = 4250.0,
        tunnel_area: float = 38.48
    ):
        super().__init__(config)

        self.L = tunnel_length
        self.A_tunnel = tunnel_area
        self.g = 9.81

        # 线性化点
        self.x_op = np.array([132.5, 132.5, 106.05, 104.79])
        self.u_op = np.array([1.0, 1.0])

        # 系统矩阵 (将在线性化时更新)
        self.A = np.eye(self.nx)
        self.B = np.zeros((self.nx, self.nu))
        self.d = np.zeros(self.nx)

    def linearize(self, x_op: np.ndarray, u_op: np.ndarray):
        """
        在工作点处线性化

        Args:
            x_op: 工作点状态
            u_op: 工作点控制
        """
        Q1, Q2, H_in, H_out = x_op
        e1, e2 = u_op

        # 曼宁糙率系数
        n = 0.014
        R = self.A_tunnel / (np.pi * 7.0)  # 水力半径

        # 摩阻系数
        k_f = n ** 2 * self.L / (self.A_tunnel ** 2 * R ** (4 / 3))

        # 状态矩阵 A
        # dQ/dt = (g*A/L) * (H_in - H_out - h_f)
        # 对Q线性化
        dh_f_dQ = 2 * k_f * abs(Q1)  # 沿程损失对流量的偏导

        a11 = -self.g * self.A_tunnel / self.L * dh_f_dQ
        a22 = -self.g * self.A_tunnel / self.L * dh_f_dQ
        a13 = self.g * self.A_tunnel / self.L
        a14 = -self.g * self.A_tunnel / self.L
        a23 = self.g * self.A_tunnel / self.L
        a24 = -self.g * self.A_tunnel / self.L

        self.A = np.array([
            [1 + a11 * self.Ts, 0, a13 * self.Ts, a14 * self.Ts],
            [0, 1 + a22 * self.Ts, a23 * self.Ts, a24 * self.Ts],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # 输入矩阵 B
        # 闸门开度影响有效面积
        b11 = self.g * self.A_tunnel / self.L * (H_in - H_out) * self.Ts
        b22 = self.g * self.A_tunnel / self.L * (H_in - H_out) * self.Ts

        self.B = np.array([
            [b11, 0],
            [0, b22],
            [0, 0],
            [0, 0]
        ])

        # 常数项 d
        self.d = np.zeros(self.nx)

    def _build_qp_matrices(
        self,
        x0: np.ndarray,
        x_ref: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        构建QP问题矩阵

        min 0.5 * u^T * H * u + f^T * u
        s.t. A_ineq * u <= b_ineq

        Returns:
            H, f, A_ineq, b_ineq
        """
        Q = self.config.Q
        R = self.config.R
        Rd = self.config.Rd

        # 预测矩阵
        Phi = np.zeros((self.Np * self.nx, self.nx))  # 状态传播
        Gamma = np.zeros((self.Np * self.nx, self.Nc * self.nu))  # 控制影响

        A_power = np.eye(self.nx)
        for i in range(self.Np):
            A_power = A_power @ self.A
            Phi[i * self.nx:(i + 1) * self.nx, :] = A_power

            for j in range(min(i + 1, self.Nc)):
                A_shift = np.linalg.matrix_power(self.A, i - j)
                Gamma[i * self.nx:(i + 1) * self.nx, j * self.nu:(j + 1) * self.nu] = A_shift @ self.B

        # 构建目标函数
        Q_bar = block_diag(*[Q] * self.Np)
        R_bar = block_diag(*[R] * self.Nc)
        Rd_bar = block_diag(*[Rd] * self.Nc)

        # 控制增量矩阵
        Delta = np.eye(self.Nc * self.nu)
        for i in range(1, self.Nc):
            Delta[i * self.nu:(i + 1) * self.nu, (i - 1) * self.nu:i * self.nu] = -np.eye(self.nu)

        # Hessian矩阵
        H = Gamma.T @ Q_bar @ Gamma + R_bar + Delta.T @ Rd_bar @ Delta

        # 梯度向量
        x_free = Phi @ x0  # 零输入响应
        x_ref_flat = x_ref.flatten()
        f = Gamma.T @ Q_bar @ (x_free - x_ref_flat) - Delta.T @ Rd_bar @ np.tile(self.u_prev, self.Nc)

        # 约束矩阵
        # 控制约束: u_min <= u <= u_max
        u_min = np.tile(self.config.u_min, self.Nc)
        u_max = np.tile(self.config.u_max, self.Nc)

        # 控制增量约束: -du_max <= du <= du_max
        du_max = np.tile(self.config.du_max, self.Nc)

        A_ineq = np.vstack([np.eye(self.Nc * self.nu), -np.eye(self.Nc * self.nu),
                            Delta, -Delta])
        b_ineq = np.hstack([u_max, -u_min, du_max, du_max])

        return H, f, A_ineq, b_ineq

    def solve(
        self,
        x0: np.ndarray,
        x_ref: np.ndarray,
        u_ref: Optional[np.ndarray] = None
    ) -> MPCResult:
        """求解LTV MPC"""
        import time
        start_time = time.time()

        # 更新线性化
        self.linearize(x0, self.u_prev)

        # 扩展参考轨迹
        if x_ref.ndim == 1:
            x_ref = np.tile(x_ref, (self.Np, 1))

        # 构建QP
        H, f, A_ineq, b_ineq = self._build_qp_matrices(x0, x_ref)

        # 求解QP (使用scipy)
        def objective(u):
            return 0.5 * u @ H @ u + f @ u

        def gradient(u):
            return H @ u + f

        # 初始猜测
        u0 = np.tile(self.u_prev, self.Nc)

        # 构造约束
        constraints = LinearConstraint(A_ineq, -np.inf, b_ineq)

        # 求解
        result = minimize(
            objective,
            u0,
            method='SLSQP',
            jac=gradient,
            constraints=constraints,
            options={'maxiter': 100}
        )

        solve_time = time.time() - start_time

        if result.success:
            u_sequence = result.x.reshape(self.Nc, self.nu)
            u_optimal = u_sequence[0]

            # 预测状态轨迹
            x_pred = np.zeros((self.Np, self.nx))
            x = x0.copy()
            for i in range(self.Np):
                u = u_sequence[min(i, self.Nc - 1)]
                x = self.A @ x + self.B @ u + self.d
                x_pred[i] = x

            return MPCResult(
                success=True,
                u_optimal=u_optimal,
                u_sequence=u_sequence,
                x_predicted=x_pred,
                cost=result.fun,
                solve_time=solve_time,
                iterations=result.nit,
                message="Optimal solution found"
            )
        else:
            return MPCResult(
                success=False,
                u_optimal=self.u_prev,
                message=result.message,
                solve_time=solve_time
            )


class NMPCController(MPCController):
    """
    非线性MPC控制器

    适用场景: S3 气液两相转换
    使用Preissmann狭缝模型处理气液界面
    """

    def __init__(
        self,
        config: Optional[MPCConfig] = None,
        tunnel_length: float = 4250.0,
        tunnel_diameter: float = 7.0
    ):
        super().__init__(config)

        self.L = tunnel_length
        self.D = tunnel_diameter
        self.A_full = np.pi * (tunnel_diameter / 2) ** 2
        self.g = 9.81

        # 波速
        self.wave_speed = 1000.0

        # Preissmann狭缝宽度
        self.slot_width = self.g * self.A_full / self.wave_speed ** 2

    def _nonlinear_dynamics(
        self,
        x: np.ndarray,
        u: np.ndarray
    ) -> np.ndarray:
        """
        非线性动力学模型

        包含气液两相流特性
        """
        Q1, Q2, h_water, P_air = x
        e1, e2 = u

        # 过水面积
        if h_water >= self.D:
            A_wet = self.A_full
        else:
            theta = 2 * np.arccos(max(-1, min(1, 1 - 2 * h_water / self.D)))
            A_wet = self.D ** 2 / 8 * (theta - np.sin(theta))
            A_wet = max(0.1, A_wet)

        # 摩擦损失
        n = 0.014
        R = A_wet / (np.pi * self.D)
        h_f = n ** 2 * (Q1 + Q2) * abs(Q1 + Q2) * self.L / (A_wet ** 2 * R ** (4 / 3))

        # 气压影响
        P_atm = 101325.0
        h_air = (P_air - P_atm) / (1000 * self.g)

        # 流量动力学
        dQ1_dt = self.g * A_wet * e1 / self.L * (106.0 - 104.0 - h_f - h_air)
        dQ2_dt = self.g * A_wet * e2 / self.L * (106.0 - 104.0 - h_f - h_air)

        # 水位变化
        if h_water < self.D:
            dh_dt = (Q1 + Q2) / (self.L * self.slot_width)
        else:
            dh_dt = 0.0

        # 气压变化 (理想气体)
        V_air = self.A_full * self.L * (1 - h_water / self.D)
        if V_air > 1:
            dP_dt = -P_air * (Q1 + Q2) / V_air
        else:
            dP_dt = 0.0

        return np.array([dQ1_dt, dQ2_dt, dh_dt, dP_dt])

    def solve(
        self,
        x0: np.ndarray,
        x_ref: np.ndarray,
        u_ref: Optional[np.ndarray] = None
    ) -> MPCResult:
        """求解NMPC"""
        import time
        start_time = time.time()

        # 扩展参考轨迹
        if x_ref.ndim == 1:
            x_ref = np.tile(x_ref, (self.Np, 1))

        Q = self.config.Q
        R = self.config.R

        def objective(u_flat):
            """目标函数"""
            u_seq = u_flat.reshape(self.Nc, self.nu)
            x = x0.copy()
            cost = 0.0

            for i in range(self.Np):
                u = u_seq[min(i, self.Nc - 1)]
                # 积分动力学
                dx = self._nonlinear_dynamics(x, u)
                x = x + self.Ts * dx

                # 累积代价
                x_err = x - x_ref[i]
                cost += x_err @ Q @ x_err + u @ R @ u

            return cost

        # 初始猜测
        u0 = np.tile(self.u_prev, self.Nc)

        # 边界约束
        bounds = [(self.config.u_min[i % self.nu], self.config.u_max[i % self.nu])
                  for i in range(self.Nc * self.nu)]

        # 求解
        result = minimize(
            objective,
            u0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 50}
        )

        solve_time = time.time() - start_time

        if result.success:
            u_sequence = result.x.reshape(self.Nc, self.nu)
            u_optimal = u_sequence[0]

            return MPCResult(
                success=True,
                u_optimal=u_optimal,
                u_sequence=u_sequence,
                cost=result.fun,
                solve_time=solve_time,
                message="NMPC solution found"
            )
        else:
            return MPCResult(
                success=False,
                u_optimal=self.u_prev,
                message=result.message,
                solve_time=solve_time
            )


class RobustMPCController(MPCController):
    """
    鲁棒MPC控制器

    适用场景: S5/S6 应急灾害
    处理参数不确定性和模型偏差
    """

    def __init__(
        self,
        config: Optional[MPCConfig] = None,
        uncertainty_bound: float = 0.1
    ):
        super().__init__(config)
        self.uncertainty = uncertainty_bound

        # 紧急模式标志
        self.emergency_mode = False
        self.emergency_type = None

    def set_emergency_mode(
        self,
        mode: bool,
        emergency_type: Optional[str] = None
    ):
        """设置应急模式"""
        self.emergency_mode = mode
        self.emergency_type = emergency_type

        if mode:
            # 应急模式下调整权重
            self.config.Q = np.diag([0.1, 0.1, 10.0, 10.0])  # 更关注压力
            self.config.R = np.eye(2) * 0.001  # 允许大幅度控制

    def _robust_dynamics(
        self,
        x: np.ndarray,
        u: np.ndarray,
        worst_case: bool = True
    ) -> np.ndarray:
        """
        考虑不确定性的动力学

        Args:
            x: 状态
            u: 控制
            worst_case: 是否使用最坏情况

        Returns:
            状态变化率
        """
        # 标称动力学
        Q1, Q2, H_in, H_out = x
        e1, e2 = u

        L = 4250.0
        A = 38.48
        g = 9.81

        # 不确定性因子
        if worst_case:
            delta = self.uncertainty
        else:
            delta = -self.uncertainty

        # 考虑渗漏的质量损失
        leak_rate = 0.0
        if self.emergency_mode and self.emergency_type == 'leakage':
            leak_rate = 0.1  # 假设渗漏率

        # 动力学
        dH = H_in - H_out
        dQ1_dt = g * A * e1 / L * dH * (1 + delta) - leak_rate / 2
        dQ2_dt = g * A * e2 / L * dH * (1 + delta) - leak_rate / 2

        return np.array([dQ1_dt, dQ2_dt, 0, 0])

    def solve(
        self,
        x0: np.ndarray,
        x_ref: np.ndarray,
        u_ref: Optional[np.ndarray] = None
    ) -> MPCResult:
        """求解鲁棒MPC (min-max 方法)"""
        import time
        start_time = time.time()

        if x_ref.ndim == 1:
            x_ref = np.tile(x_ref, (self.Np, 1))

        Q = self.config.Q
        R = self.config.R

        def robust_objective(u_flat):
            """最坏情况目标函数"""
            u_seq = u_flat.reshape(self.Nc, self.nu)

            max_cost = 0.0
            # 评估多个不确定性实现
            for worst_case in [True, False]:
                x = x0.copy()
                cost = 0.0

                for i in range(self.Np):
                    u = u_seq[min(i, self.Nc - 1)]
                    dx = self._robust_dynamics(x, u, worst_case)
                    x = x + self.Ts * dx

                    x_err = x - x_ref[i]
                    cost += x_err @ Q @ x_err + u @ R @ u

                max_cost = max(max_cost, cost)

            return max_cost

        u0 = np.tile(self.u_prev, self.Nc)
        bounds = [(self.config.u_min[i % self.nu], self.config.u_max[i % self.nu])
                  for i in range(self.Nc * self.nu)]

        result = minimize(
            robust_objective,
            u0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 30}
        )

        solve_time = time.time() - start_time

        if result.success:
            u_sequence = result.x.reshape(self.Nc, self.nu)
            return MPCResult(
                success=True,
                u_optimal=u_sequence[0],
                u_sequence=u_sequence,
                cost=result.fun,
                solve_time=solve_time,
                message="Robust MPC solution found"
            )
        else:
            return MPCResult(
                success=False,
                u_optimal=self.u_prev,
                message=result.message,
                solve_time=solve_time
            )
