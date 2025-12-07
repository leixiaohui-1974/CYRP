"""
数据同化系统 - Data Assimilation System

实现完整的数据同化功能，包括：
- 卡尔曼滤波 (KF, EKF, UKF)
- 集合卡尔曼滤波 (EnKF)
- 粒子滤波
- 变分同化 (3D-Var, 4D-Var)
- 混合同化方法

Implements complete data assimilation including:
- Kalman Filters (KF, EKF, UKF)
- Ensemble Kalman Filter (EnKF)
- Particle Filter
- Variational Assimilation (3D-Var, 4D-Var)
- Hybrid Assimilation Methods
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from scipy.linalg import cholesky, solve, inv
from scipy.optimize import minimize
import threading


class AssimilationMethod(Enum):
    """同化方法"""
    KALMAN = "kalman"
    EXTENDED_KALMAN = "extended_kalman"
    UNSCENTED_KALMAN = "unscented_kalman"
    ENSEMBLE_KALMAN = "ensemble_kalman"
    PARTICLE = "particle"
    VAR_3D = "3d_var"
    VAR_4D = "4d_var"
    HYBRID = "hybrid"


@dataclass
class AssimilationResult:
    """同化结果"""
    timestamp: float
    state_estimate: np.ndarray              # 状态估计
    state_covariance: np.ndarray            # 状态协方差
    innovation: np.ndarray                   # 新息 (观测 - 预测)
    kalman_gain: Optional[np.ndarray] = None  # 卡尔曼增益
    analysis_increment: Optional[np.ndarray] = None  # 分析增量

    # 诊断信息
    rmse: float = 0.0                       # 均方根误差
    spread: float = 0.0                     # 集合离散度
    consistency: float = 0.0                # 一致性指标
    chi_squared: float = 0.0                # 卡方统计量

    def compute_diagnostics(self, observation: np.ndarray):
        """计算诊断指标"""
        self.rmse = np.sqrt(np.mean(self.innovation ** 2))
        if self.state_covariance is not None:
            self.spread = np.sqrt(np.mean(np.diag(self.state_covariance)))

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'timestamp': self.timestamp,
            'state_estimate': self.state_estimate.tolist(),
            'rmse': self.rmse,
            'spread': self.spread,
            'consistency': self.consistency
        }


class BaseAssimilator(ABC):
    """同化器基类"""

    def __init__(self, state_dim: int, obs_dim: int):
        self.state_dim = state_dim
        self.obs_dim = obs_dim

        # 状态
        self.x = np.zeros(state_dim)        # 状态估计
        self.P = np.eye(state_dim)          # 状态协方差

        # 模型和观测误差
        self.Q = np.eye(state_dim) * 0.01   # 过程噪声协方差
        self.R = np.eye(obs_dim) * 0.1      # 观测噪声协方差

        self._time = 0.0
        self._results_history: List[AssimilationResult] = []

    @abstractmethod
    def predict(self, dt: float, control_input: Optional[np.ndarray] = None) -> np.ndarray:
        """预测步"""
        pass

    @abstractmethod
    def update(self, observation: np.ndarray) -> AssimilationResult:
        """更新步"""
        pass

    def assimilate(self, observation: np.ndarray, dt: float,
                   control_input: Optional[np.ndarray] = None) -> AssimilationResult:
        """执行一步同化"""
        self._time += dt
        self.predict(dt, control_input)
        result = self.update(observation)
        self._results_history.append(result)
        return result

    def set_initial_state(self, x0: np.ndarray, P0: Optional[np.ndarray] = None):
        """设置初始状态"""
        self.x = x0.copy()
        if P0 is not None:
            self.P = P0.copy()

    def set_noise_covariances(self, Q: np.ndarray, R: np.ndarray):
        """设置噪声协方差"""
        self.Q = Q.copy()
        self.R = R.copy()

    def get_history(self, n_samples: Optional[int] = None) -> List[AssimilationResult]:
        """获取历史"""
        if n_samples is None:
            return self._results_history.copy()
        return self._results_history[-n_samples:]

    def reset(self):
        """重置"""
        self.x = np.zeros(self.state_dim)
        self.P = np.eye(self.state_dim)
        self._time = 0.0
        self._results_history.clear()


class KalmanFilter(BaseAssimilator):
    """标准卡尔曼滤波"""

    def __init__(self, state_dim: int, obs_dim: int,
                 A: Optional[np.ndarray] = None,
                 B: Optional[np.ndarray] = None,
                 H: Optional[np.ndarray] = None):
        super().__init__(state_dim, obs_dim)

        # 系统矩阵
        self.A = A if A is not None else np.eye(state_dim)  # 状态转移矩阵
        self.B = B if B is not None else np.zeros((state_dim, 1))  # 控制输入矩阵
        self.H = H if H is not None else np.eye(obs_dim, state_dim)  # 观测矩阵

    def predict(self, dt: float, control_input: Optional[np.ndarray] = None) -> np.ndarray:
        """
        预测步

        x_k|k-1 = A * x_k-1|k-1 + B * u_k
        P_k|k-1 = A * P_k-1|k-1 * A^T + Q
        """
        # 状态预测
        self.x = self.A @ self.x
        if control_input is not None:
            self.x += self.B @ control_input

        # 协方差预测
        self.P = self.A @ self.P @ self.A.T + self.Q

        return self.x

    def update(self, observation: np.ndarray) -> AssimilationResult:
        """
        更新步

        K = P_k|k-1 * H^T * (H * P_k|k-1 * H^T + R)^-1
        x_k|k = x_k|k-1 + K * (z_k - H * x_k|k-1)
        P_k|k = (I - K * H) * P_k|k-1
        """
        # 新息
        y = observation - self.H @ self.x

        # 新息协方差
        S = self.H @ self.P @ self.H.T + self.R

        # 卡尔曼增益
        K = self.P @ self.H.T @ inv(S)

        # 状态更新
        x_prior = self.x.copy()
        self.x = self.x + K @ y

        # 协方差更新 (Joseph形式，更稳定)
        I_KH = np.eye(self.state_dim) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        # 构建结果
        result = AssimilationResult(
            timestamp=self._time,
            state_estimate=self.x.copy(),
            state_covariance=self.P.copy(),
            innovation=y,
            kalman_gain=K,
            analysis_increment=self.x - x_prior
        )
        result.compute_diagnostics(observation)

        return result


class ExtendedKalmanFilter(BaseAssimilator):
    """扩展卡尔曼滤波"""

    def __init__(self, state_dim: int, obs_dim: int,
                 state_transition: Optional[Callable] = None,
                 observation_model: Optional[Callable] = None,
                 state_jacobian: Optional[Callable] = None,
                 observation_jacobian: Optional[Callable] = None):
        super().__init__(state_dim, obs_dim)

        # 非线性函数
        self.f = state_transition or self._default_state_transition
        self.h = observation_model or self._default_observation_model

        # 雅可比矩阵函数
        self.F = state_jacobian or self._numerical_state_jacobian
        self.H_func = observation_jacobian or self._numerical_observation_jacobian

    def _default_state_transition(self, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        """默认状态转移（线性）"""
        return x

    def _default_observation_model(self, x: np.ndarray) -> np.ndarray:
        """默认观测模型（线性）"""
        return x[:self.obs_dim]

    def _numerical_state_jacobian(self, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        """数值计算状态雅可比矩阵"""
        eps = 1e-7
        F = np.zeros((self.state_dim, self.state_dim))

        for i in range(self.state_dim):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += eps
            x_minus[i] -= eps

            f_plus = self.f(x_plus, u, dt)
            f_minus = self.f(x_minus, u, dt)

            F[:, i] = (f_plus - f_minus) / (2 * eps)

        return F

    def _numerical_observation_jacobian(self, x: np.ndarray) -> np.ndarray:
        """数值计算观测雅可比矩阵"""
        eps = 1e-7
        H = np.zeros((self.obs_dim, self.state_dim))

        for i in range(self.state_dim):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += eps
            x_minus[i] -= eps

            h_plus = self.h(x_plus)
            h_minus = self.h(x_minus)

            H[:, i] = (h_plus - h_minus) / (2 * eps)

        return H

    def predict(self, dt: float, control_input: Optional[np.ndarray] = None) -> np.ndarray:
        """EKF预测步"""
        u = control_input if control_input is not None else np.zeros(1)

        # 状态预测 (非线性)
        self.x = self.f(self.x, u, dt)

        # 雅可比矩阵
        F = self.F(self.x, u, dt)

        # 协方差预测
        self.P = F @ self.P @ F.T + self.Q

        return self.x

    def update(self, observation: np.ndarray) -> AssimilationResult:
        """EKF更新步"""
        # 预测观测
        h_x = self.h(self.x)

        # 新息
        y = observation - h_x

        # 观测雅可比矩阵
        H = self.H_func(self.x)

        # 新息协方差
        S = H @ self.P @ H.T + self.R

        # 卡尔曼增益
        K = self.P @ H.T @ inv(S)

        # 状态更新
        x_prior = self.x.copy()
        self.x = self.x + K @ y

        # 协方差更新
        I_KH = np.eye(self.state_dim) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        # 构建结果
        result = AssimilationResult(
            timestamp=self._time,
            state_estimate=self.x.copy(),
            state_covariance=self.P.copy(),
            innovation=y,
            kalman_gain=K,
            analysis_increment=self.x - x_prior
        )
        result.compute_diagnostics(observation)

        return result


class UnscentedKalmanFilter(BaseAssimilator):
    """无迹卡尔曼滤波"""

    def __init__(self, state_dim: int, obs_dim: int,
                 state_transition: Optional[Callable] = None,
                 observation_model: Optional[Callable] = None,
                 alpha: float = 1e-3,
                 beta: float = 2.0,
                 kappa: float = 0.0):
        super().__init__(state_dim, obs_dim)

        # 非线性函数
        self.f = state_transition or (lambda x, u, dt: x)
        self.h = observation_model or (lambda x: x[:obs_dim])

        # UKF参数
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

        # 计算权重
        self._compute_weights()

    def _compute_weights(self):
        """计算sigma点权重"""
        n = self.state_dim
        lam = self.alpha ** 2 * (n + self.kappa) - n

        # 均值权重
        self.Wm = np.zeros(2 * n + 1)
        self.Wm[0] = lam / (n + lam)
        self.Wm[1:] = 1 / (2 * (n + lam))

        # 协方差权重
        self.Wc = np.zeros(2 * n + 1)
        self.Wc[0] = lam / (n + lam) + (1 - self.alpha ** 2 + self.beta)
        self.Wc[1:] = 1 / (2 * (n + lam))

        self.gamma = np.sqrt(n + lam)

    def _generate_sigma_points(self, x: np.ndarray, P: np.ndarray) -> np.ndarray:
        """生成sigma点"""
        n = len(x)
        sigma_points = np.zeros((2 * n + 1, n))

        # 矩阵平方根
        try:
            sqrt_P = cholesky(P, lower=True)
        except np.linalg.LinAlgError:
            # 如果不正定，使用特征值分解
            eigvals, eigvecs = np.linalg.eigh(P)
            eigvals = np.maximum(eigvals, 1e-10)
            sqrt_P = eigvecs @ np.diag(np.sqrt(eigvals))

        # 中心点
        sigma_points[0] = x

        # 其余点
        for i in range(n):
            sigma_points[i + 1] = x + self.gamma * sqrt_P[:, i]
            sigma_points[n + i + 1] = x - self.gamma * sqrt_P[:, i]

        return sigma_points

    def predict(self, dt: float, control_input: Optional[np.ndarray] = None) -> np.ndarray:
        """UKF预测步"""
        u = control_input if control_input is not None else np.zeros(1)

        # 生成sigma点
        sigma_points = self._generate_sigma_points(self.x, self.P)

        # 传播sigma点
        n = self.state_dim
        sigma_points_pred = np.zeros_like(sigma_points)
        for i in range(2 * n + 1):
            sigma_points_pred[i] = self.f(sigma_points[i], u, dt)

        # 计算预测均值
        self.x = np.sum(self.Wm[:, np.newaxis] * sigma_points_pred, axis=0)

        # 计算预测协方差
        self.P = np.zeros((n, n))
        for i in range(2 * n + 1):
            diff = sigma_points_pred[i] - self.x
            self.P += self.Wc[i] * np.outer(diff, diff)
        self.P += self.Q

        self._sigma_points_pred = sigma_points_pred

        return self.x

    def update(self, observation: np.ndarray) -> AssimilationResult:
        """UKF更新步"""
        n = self.state_dim
        m = self.obs_dim

        # 传播sigma点通过观测模型
        sigma_points_obs = np.zeros((2 * n + 1, m))
        for i in range(2 * n + 1):
            sigma_points_obs[i] = self.h(self._sigma_points_pred[i])

        # 计算预测观测均值
        z_pred = np.sum(self.Wm[:, np.newaxis] * sigma_points_obs, axis=0)

        # 新息
        y = observation - z_pred

        # 计算协方差
        Pzz = np.zeros((m, m))  # 观测协方差
        Pxz = np.zeros((n, m))  # 交叉协方差

        for i in range(2 * n + 1):
            diff_z = sigma_points_obs[i] - z_pred
            diff_x = self._sigma_points_pred[i] - self.x
            Pzz += self.Wc[i] * np.outer(diff_z, diff_z)
            Pxz += self.Wc[i] * np.outer(diff_x, diff_z)

        Pzz += self.R

        # 卡尔曼增益
        K = Pxz @ inv(Pzz)

        # 状态更新
        x_prior = self.x.copy()
        self.x = self.x + K @ y

        # 协方差更新
        self.P = self.P - K @ Pzz @ K.T

        # 构建结果
        result = AssimilationResult(
            timestamp=self._time,
            state_estimate=self.x.copy(),
            state_covariance=self.P.copy(),
            innovation=y,
            kalman_gain=K,
            analysis_increment=self.x - x_prior
        )
        result.compute_diagnostics(observation)

        return result


class EnsembleKalmanFilter(BaseAssimilator):
    """集合卡尔曼滤波"""

    def __init__(self, state_dim: int, obs_dim: int,
                 ensemble_size: int = 50,
                 state_transition: Optional[Callable] = None,
                 observation_model: Optional[Callable] = None,
                 inflation_factor: float = 1.02,
                 localization_radius: Optional[float] = None):
        super().__init__(state_dim, obs_dim)

        self.N = ensemble_size
        self.f = state_transition or (lambda x, u, dt: x)
        self.h = observation_model or (lambda x: x[:obs_dim])

        # 同化参数
        self.inflation_factor = inflation_factor
        self.localization_radius = localization_radius

        # 集合
        self.ensemble = np.zeros((ensemble_size, state_dim))
        self._init_ensemble()

    def _init_ensemble(self):
        """初始化集合"""
        for i in range(self.N):
            self.ensemble[i] = self.x + np.random.multivariate_normal(
                np.zeros(self.state_dim), self.P
            )

    def _compute_ensemble_mean_cov(self) -> Tuple[np.ndarray, np.ndarray]:
        """计算集合均值和协方差"""
        mean = np.mean(self.ensemble, axis=0)

        # 扰动矩阵
        A = (self.ensemble - mean) / np.sqrt(self.N - 1)

        # 协方差
        cov = A.T @ A

        return mean, cov

    def _apply_inflation(self):
        """应用协方差膨胀"""
        mean = np.mean(self.ensemble, axis=0)
        for i in range(self.N):
            self.ensemble[i] = mean + self.inflation_factor * (self.ensemble[i] - mean)

    def _compute_localization_matrix(self) -> np.ndarray:
        """计算局地化矩阵"""
        if self.localization_radius is None:
            return np.ones((self.state_dim, self.obs_dim))

        # 简化的距离基局地化
        L = np.ones((self.state_dim, self.obs_dim))
        # 实际应用中需要根据物理距离计算
        return L

    def predict(self, dt: float, control_input: Optional[np.ndarray] = None) -> np.ndarray:
        """EnKF预测步"""
        u = control_input if control_input is not None else np.zeros(1)

        # 传播每个集合成员
        for i in range(self.N):
            # 状态传播
            self.ensemble[i] = self.f(self.ensemble[i], u, dt)

            # 添加过程噪声
            self.ensemble[i] += np.random.multivariate_normal(
                np.zeros(self.state_dim), self.Q
            )

        # 更新均值和协方差
        self.x, self.P = self._compute_ensemble_mean_cov()

        return self.x

    def update(self, observation: np.ndarray) -> AssimilationResult:
        """EnKF更新步"""
        # 应用协方差膨胀
        self._apply_inflation()

        # 计算预测观测集合
        H_ensemble = np.zeros((self.N, self.obs_dim))
        for i in range(self.N):
            H_ensemble[i] = self.h(self.ensemble[i])

        # 观测均值
        H_mean = np.mean(H_ensemble, axis=0)

        # 新息
        y = observation - H_mean

        # 扰动矩阵
        A = (self.ensemble - self.x) / np.sqrt(self.N - 1)
        HA = (H_ensemble - H_mean) / np.sqrt(self.N - 1)

        # 卡尔曼增益
        PHT = A.T @ HA
        HPHT = HA.T @ HA + self.R
        K = PHT @ inv(HPHT)

        # 局地化
        L = self._compute_localization_matrix()
        K = K * L

        # 生成扰动观测
        perturbed_obs = np.zeros((self.N, self.obs_dim))
        for i in range(self.N):
            perturbed_obs[i] = observation + np.random.multivariate_normal(
                np.zeros(self.obs_dim), self.R
            )

        # 更新集合
        x_prior = self.x.copy()
        for i in range(self.N):
            innovation = perturbed_obs[i] - H_ensemble[i]
            self.ensemble[i] = self.ensemble[i] + K @ innovation

        # 更新均值和协方差
        self.x, self.P = self._compute_ensemble_mean_cov()

        # 计算集合离散度
        spread = np.sqrt(np.mean(np.var(self.ensemble, axis=0)))

        # 构建结果
        result = AssimilationResult(
            timestamp=self._time,
            state_estimate=self.x.copy(),
            state_covariance=self.P.copy(),
            innovation=y,
            kalman_gain=K,
            analysis_increment=self.x - x_prior,
            spread=spread
        )
        result.compute_diagnostics(observation)

        return result

    def reset(self):
        """重置"""
        super().reset()
        self._init_ensemble()


class ParticleFilter(BaseAssimilator):
    """粒子滤波"""

    def __init__(self, state_dim: int, obs_dim: int,
                 n_particles: int = 100,
                 state_transition: Optional[Callable] = None,
                 likelihood_model: Optional[Callable] = None,
                 resampling_threshold: float = 0.5):
        super().__init__(state_dim, obs_dim)

        self.n_particles = n_particles
        self.f = state_transition or (lambda x, u, dt: x + np.random.randn(state_dim) * 0.1)
        self.likelihood = likelihood_model or self._default_likelihood

        self.resampling_threshold = resampling_threshold

        # 粒子和权重
        self.particles = np.zeros((n_particles, state_dim))
        self.weights = np.ones(n_particles) / n_particles
        self._init_particles()

    def _init_particles(self):
        """初始化粒子"""
        for i in range(self.n_particles):
            self.particles[i] = self.x + np.random.multivariate_normal(
                np.zeros(self.state_dim), self.P
            )

    def _default_likelihood(self, observation: np.ndarray, predicted: np.ndarray) -> float:
        """默认似然函数（高斯）"""
        diff = observation - predicted[:self.obs_dim]
        exponent = -0.5 * diff @ inv(self.R) @ diff
        return np.exp(exponent)

    def _effective_sample_size(self) -> float:
        """计算有效样本大小"""
        return 1.0 / np.sum(self.weights ** 2)

    def _resample(self):
        """重采样"""
        indices = np.random.choice(
            self.n_particles,
            size=self.n_particles,
            replace=True,
            p=self.weights
        )
        self.particles = self.particles[indices]
        self.weights = np.ones(self.n_particles) / self.n_particles

    def predict(self, dt: float, control_input: Optional[np.ndarray] = None) -> np.ndarray:
        """粒子滤波预测步"""
        u = control_input if control_input is not None else np.zeros(1)

        # 传播粒子
        for i in range(self.n_particles):
            self.particles[i] = self.f(self.particles[i], u, dt)

        # 计算均值
        self.x = np.average(self.particles, weights=self.weights, axis=0)

        return self.x

    def update(self, observation: np.ndarray) -> AssimilationResult:
        """粒子滤波更新步"""
        x_prior = self.x.copy()

        # 计算每个粒子的似然
        likelihoods = np.zeros(self.n_particles)
        for i in range(self.n_particles):
            likelihoods[i] = self.likelihood(observation, self.particles[i])

        # 更新权重
        self.weights *= likelihoods
        self.weights /= np.sum(self.weights)  # 归一化

        # 重采样检查
        neff = self._effective_sample_size()
        if neff < self.resampling_threshold * self.n_particles:
            self._resample()

        # 计算后验估计
        self.x = np.average(self.particles, weights=self.weights, axis=0)

        # 计算协方差
        self.P = np.cov(self.particles.T, aweights=self.weights)

        # 新息
        y = observation - self.x[:self.obs_dim]

        # 构建结果
        result = AssimilationResult(
            timestamp=self._time,
            state_estimate=self.x.copy(),
            state_covariance=self.P.copy(),
            innovation=y,
            analysis_increment=self.x - x_prior,
            spread=np.sqrt(np.mean(np.var(self.particles, axis=0)))
        )
        result.compute_diagnostics(observation)

        return result

    def reset(self):
        """重置"""
        super().reset()
        self._init_particles()


class VariationalAssimilation(BaseAssimilator):
    """变分同化 (3D-Var / 4D-Var)"""

    def __init__(self, state_dim: int, obs_dim: int,
                 observation_model: Optional[Callable] = None,
                 state_transition: Optional[Callable] = None,
                 assimilation_window: int = 1,
                 max_iterations: int = 100,
                 tolerance: float = 1e-6):
        super().__init__(state_dim, obs_dim)

        self.h = observation_model or (lambda x: x[:obs_dim])
        self.f = state_transition or (lambda x, u, dt: x)

        self.assimilation_window = assimilation_window  # 1 for 3D-Var, >1 for 4D-Var
        self.max_iterations = max_iterations
        self.tolerance = tolerance

        # 背景状态
        self.xb = np.zeros(state_dim)
        self.B = np.eye(state_dim)  # 背景误差协方差

        # 观测窗口数据
        self._obs_window: List[Tuple[float, np.ndarray]] = []

    def _cost_function_3dvar(self, x: np.ndarray, observation: np.ndarray) -> float:
        """3D-Var代价函数"""
        # 背景项
        diff_b = x - self.xb
        Jb = 0.5 * diff_b @ inv(self.B) @ diff_b

        # 观测项
        diff_o = observation - self.h(x)
        Jo = 0.5 * diff_o @ inv(self.R) @ diff_o

        return Jb + Jo

    def _gradient_3dvar(self, x: np.ndarray, observation: np.ndarray) -> np.ndarray:
        """3D-Var梯度"""
        # 数值梯度
        eps = 1e-7
        grad = np.zeros(self.state_dim)

        for i in range(self.state_dim):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += eps
            x_minus[i] -= eps

            grad[i] = (self._cost_function_3dvar(x_plus, observation) -
                       self._cost_function_3dvar(x_minus, observation)) / (2 * eps)

        return grad

    def _cost_function_4dvar(self, x0: np.ndarray) -> float:
        """4D-Var代价函数"""
        # 背景项
        diff_b = x0 - self.xb
        Jb = 0.5 * diff_b @ inv(self.B) @ diff_b

        # 观测项
        Jo = 0.0
        x = x0.copy()

        for t, obs in self._obs_window:
            # 传播到观测时间
            x = self.f(x, np.zeros(1), t)
            diff_o = obs - self.h(x)
            Jo += 0.5 * diff_o @ inv(self.R) @ diff_o

        return Jb + Jo

    def predict(self, dt: float, control_input: Optional[np.ndarray] = None) -> np.ndarray:
        """预测步（用于背景状态）"""
        u = control_input if control_input is not None else np.zeros(1)
        self.xb = self.f(self.x, u, dt)
        return self.xb

    def update(self, observation: np.ndarray) -> AssimilationResult:
        """变分同化更新步"""
        x_prior = self.x.copy()

        if self.assimilation_window == 1:
            # 3D-Var
            result = minimize(
                self._cost_function_3dvar,
                self.xb,
                args=(observation,),
                method='L-BFGS-B',
                jac=lambda x: self._gradient_3dvar(x, observation),
                options={'maxiter': self.max_iterations, 'gtol': self.tolerance}
            )
            self.x = result.x
        else:
            # 4D-Var
            self._obs_window.append((self._time, observation))
            if len(self._obs_window) >= self.assimilation_window:
                result = minimize(
                    self._cost_function_4dvar,
                    self.xb,
                    method='L-BFGS-B',
                    options={'maxiter': self.max_iterations, 'gtol': self.tolerance}
                )
                self.x = result.x
                self._obs_window.clear()

        # 更新协方差（近似）
        # 使用Hessian的逆作为分析误差协方差的近似
        self.P = self.B.copy()  # 简化处理

        # 新息
        y = observation - self.h(self.x)

        # 构建结果
        result = AssimilationResult(
            timestamp=self._time,
            state_estimate=self.x.copy(),
            state_covariance=self.P.copy(),
            innovation=y,
            analysis_increment=self.x - x_prior
        )
        result.compute_diagnostics(observation)

        return result

    def set_background(self, xb: np.ndarray, B: Optional[np.ndarray] = None):
        """设置背景状态"""
        self.xb = xb.copy()
        if B is not None:
            self.B = B.copy()


class HybridAssimilation(BaseAssimilator):
    """混合同化方法"""

    def __init__(self, state_dim: int, obs_dim: int,
                 ensemble_size: int = 30,
                 state_transition: Optional[Callable] = None,
                 observation_model: Optional[Callable] = None,
                 ensemble_weight: float = 0.5):
        super().__init__(state_dim, obs_dim)

        # 子方法
        self.enkf = EnsembleKalmanFilter(
            state_dim, obs_dim, ensemble_size,
            state_transition, observation_model
        )
        self.var = VariationalAssimilation(
            state_dim, obs_dim,
            observation_model, state_transition
        )

        # 混合权重
        self.ensemble_weight = ensemble_weight
        self.static_weight = 1.0 - ensemble_weight

    def predict(self, dt: float, control_input: Optional[np.ndarray] = None) -> np.ndarray:
        """混合预测步"""
        self.enkf.predict(dt, control_input)
        self.var.predict(dt, control_input)

        # 混合背景协方差
        self.P = (self.ensemble_weight * self.enkf.P +
                  self.static_weight * self.var.B)

        self.x = self.enkf.x.copy()
        return self.x

    def update(self, observation: np.ndarray) -> AssimilationResult:
        """混合更新步"""
        x_prior = self.x.copy()

        # EnKF更新
        enkf_result = self.enkf.update(observation)

        # 变分更新
        self.var.xb = self.x.copy()
        self.var.B = self.P.copy()
        var_result = self.var.update(observation)

        # 混合分析
        self.x = (self.ensemble_weight * enkf_result.state_estimate +
                  self.static_weight * var_result.state_estimate)

        # 混合协方差
        self.P = (self.ensemble_weight * enkf_result.state_covariance +
                  self.static_weight * var_result.state_covariance)

        # 新息
        y = observation - self.enkf.h(self.x)

        # 构建结果
        result = AssimilationResult(
            timestamp=self._time,
            state_estimate=self.x.copy(),
            state_covariance=self.P.copy(),
            innovation=y,
            analysis_increment=self.x - x_prior,
            spread=(self.ensemble_weight * enkf_result.spread)
        )
        result.compute_diagnostics(observation)

        return result


class DataAssimilationManager:
    """数据同化管理器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.assimilators: Dict[str, BaseAssimilator] = {}
        self._time = 0.0
        self._lock = threading.Lock()

    def create_assimilator(self, name: str, method: AssimilationMethod,
                          state_dim: int, obs_dim: int,
                          **kwargs) -> BaseAssimilator:
        """
        创建同化器

        Args:
            name: 同化器名称
            method: 同化方法
            state_dim: 状态维度
            obs_dim: 观测维度
            **kwargs: 其他参数

        Returns:
            同化器实例
        """
        if method == AssimilationMethod.KALMAN:
            assimilator = KalmanFilter(state_dim, obs_dim, **kwargs)
        elif method == AssimilationMethod.EXTENDED_KALMAN:
            assimilator = ExtendedKalmanFilter(state_dim, obs_dim, **kwargs)
        elif method == AssimilationMethod.UNSCENTED_KALMAN:
            assimilator = UnscentedKalmanFilter(state_dim, obs_dim, **kwargs)
        elif method == AssimilationMethod.ENSEMBLE_KALMAN:
            assimilator = EnsembleKalmanFilter(state_dim, obs_dim, **kwargs)
        elif method == AssimilationMethod.PARTICLE:
            assimilator = ParticleFilter(state_dim, obs_dim, **kwargs)
        elif method == AssimilationMethod.VAR_3D:
            assimilator = VariationalAssimilation(state_dim, obs_dim, assimilation_window=1, **kwargs)
        elif method == AssimilationMethod.VAR_4D:
            window = kwargs.pop('window', 10)
            assimilator = VariationalAssimilation(state_dim, obs_dim, assimilation_window=window, **kwargs)
        elif method == AssimilationMethod.HYBRID:
            assimilator = HybridAssimilation(state_dim, obs_dim, **kwargs)
        else:
            raise ValueError(f"Unknown assimilation method: {method}")

        with self._lock:
            self.assimilators[name] = assimilator

        return assimilator

    def assimilate(self, name: str, observation: np.ndarray, dt: float,
                  control_input: Optional[np.ndarray] = None) -> AssimilationResult:
        """执行同化"""
        if name not in self.assimilators:
            raise ValueError(f"Assimilator not found: {name}")

        self._time += dt
        return self.assimilators[name].assimilate(observation, dt, control_input)

    def assimilate_all(self, observations: Dict[str, np.ndarray], dt: float,
                      control_inputs: Optional[Dict[str, np.ndarray]] = None
                      ) -> Dict[str, AssimilationResult]:
        """对所有同化器执行同化"""
        results = {}
        control_inputs = control_inputs or {}

        for name in self.assimilators:
            if name in observations:
                u = control_inputs.get(name)
                results[name] = self.assimilate(name, observations[name], dt, u)

        return results

    def get_state_estimates(self) -> Dict[str, np.ndarray]:
        """获取所有状态估计"""
        return {name: ass.x.copy() for name, ass in self.assimilators.items()}

    def get_diagnostics(self) -> Dict[str, Dict[str, Any]]:
        """获取诊断信息"""
        diagnostics = {}

        for name, assimilator in self.assimilators.items():
            history = assimilator.get_history(100)
            if history:
                latest = history[-1]
                rmse_history = [r.rmse for r in history]
                diagnostics[name] = {
                    'current_rmse': latest.rmse,
                    'mean_rmse': np.mean(rmse_history),
                    'spread': latest.spread,
                    'state_dim': assimilator.state_dim,
                    'obs_dim': assimilator.obs_dim
                }

        return diagnostics

    def reset(self):
        """重置所有同化器"""
        self._time = 0.0
        for assimilator in self.assimilators.values():
            assimilator.reset()
