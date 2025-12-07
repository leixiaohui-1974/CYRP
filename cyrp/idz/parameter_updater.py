"""
IDZ模型参数动态更新系统 - IDZ Model Parameter Dynamic Update System

基于高保真仿真模型当前状态进行IDZ（Intelligent Digital Zone）模型参数的动态更新
实现功能包括：
- 递推最小二乘参数辨识
- 扩展最小二乘法
- 系统辨识
- 模型校准
- 参数约束处理

Dynamically updates IDZ model parameters based on high-fidelity simulation states
Implements:
- Recursive Least Squares parameter identification
- Extended Least Squares
- System Identification
- Model Calibration
- Parameter constraint handling
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from collections import deque
from scipy.optimize import minimize, least_squares
from scipy.linalg import inv, pinv
import threading


class IdentificationMethod(Enum):
    """辨识方法"""
    RLS = "recursive_least_squares"       # 递推最小二乘
    ELS = "extended_least_squares"        # 扩展最小二乘
    IV = "instrumental_variable"          # 工具变量法
    ML = "maximum_likelihood"             # 最大似然
    SUBSPACE = "subspace"                 # 子空间辨识
    PEM = "prediction_error_method"       # 预测误差法


class ModelStructure(Enum):
    """模型结构"""
    ARX = "arx"           # Auto-Regressive with eXogenous input
    ARMAX = "armax"       # ARX with Moving Average
    OE = "oe"             # Output Error
    BJ = "bj"             # Box-Jenkins
    STATE_SPACE = "state_space"  # 状态空间模型


@dataclass
class ParameterConstraints:
    """参数约束"""
    min_values: Optional[np.ndarray] = None    # 最小值约束
    max_values: Optional[np.ndarray] = None    # 最大值约束
    rate_limits: Optional[np.ndarray] = None   # 变化率约束
    equality_constraints: Optional[List[Callable]] = None  # 等式约束
    inequality_constraints: Optional[List[Callable]] = None  # 不等式约束

    def apply(self, parameters: np.ndarray,
              prev_parameters: Optional[np.ndarray] = None,
              dt: float = 1.0) -> np.ndarray:
        """应用约束"""
        result = parameters.copy()

        # 范围约束
        if self.min_values is not None:
            result = np.maximum(result, self.min_values)
        if self.max_values is not None:
            result = np.minimum(result, self.max_values)

        # 变化率约束
        if self.rate_limits is not None and prev_parameters is not None:
            delta = result - prev_parameters
            max_delta = self.rate_limits * dt
            delta = np.clip(delta, -max_delta, max_delta)
            result = prev_parameters + delta

        return result


@dataclass
class UpdateResult:
    """更新结果"""
    timestamp: float
    parameters: np.ndarray                    # 更新后的参数
    parameter_covariance: np.ndarray          # 参数协方差
    prediction_error: float                   # 预测误差
    model_fit: float                          # 模型拟合度

    # 诊断信息
    condition_number: float = 0.0             # 条件数
    convergence: bool = True                  # 收敛标志
    iterations: int = 1                       # 迭代次数

    # 参数变化
    parameter_change: Optional[np.ndarray] = None
    relative_change: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'timestamp': self.timestamp,
            'parameters': self.parameters.tolist(),
            'prediction_error': self.prediction_error,
            'model_fit': self.model_fit,
            'condition_number': self.condition_number,
            'convergence': self.convergence,
            'relative_change': self.relative_change
        }


class RecursiveLeastSquares:
    """递推最小二乘"""

    def __init__(self, n_params: int, forgetting_factor: float = 0.99,
                 initial_covariance: float = 1000.0):
        """
        初始化RLS

        Args:
            n_params: 参数数量
            forgetting_factor: 遗忘因子 (0.95-0.999)
            initial_covariance: 初始协方差
        """
        self.n_params = n_params
        self.lambda_ = forgetting_factor

        # 参数估计
        self.theta = np.zeros(n_params)

        # 协方差矩阵
        self.P = np.eye(n_params) * initial_covariance

        # 统计
        self._update_count = 0
        self._total_error = 0.0

    def update(self, phi: np.ndarray, y: float) -> Tuple[np.ndarray, float]:
        """
        更新参数估计

        Args:
            phi: 回归向量 (n_params,)
            y: 输出观测值

        Returns:
            (更新后的参数, 预测误差)
        """
        # 预测
        y_pred = phi @ self.theta
        error = y - y_pred

        # 增益计算
        Pphi = self.P @ phi
        denominator = self.lambda_ + phi @ Pphi
        K = Pphi / denominator

        # 参数更新
        self.theta = self.theta + K * error

        # 协方差更新
        self.P = (self.P - np.outer(K, phi @ self.P)) / self.lambda_

        # 统计更新
        self._update_count += 1
        self._total_error += error ** 2

        return self.theta.copy(), error

    def reset(self, initial_covariance: float = 1000.0):
        """重置"""
        self.theta = np.zeros(self.n_params)
        self.P = np.eye(self.n_params) * initial_covariance
        self._update_count = 0
        self._total_error = 0.0

    def get_parameter_uncertainty(self) -> np.ndarray:
        """获取参数不确定性（标准差）"""
        return np.sqrt(np.diag(self.P))

    @property
    def mean_squared_error(self) -> float:
        """均方误差"""
        if self._update_count > 0:
            return self._total_error / self._update_count
        return 0.0


class ExtendedLeastSquares:
    """扩展最小二乘（含噪声模型）"""

    def __init__(self, n_params: int, n_noise_params: int = 2,
                 forgetting_factor: float = 0.99):
        """
        初始化ELS

        Args:
            n_params: 系统参数数量
            n_noise_params: 噪声模型参数数量
            forgetting_factor: 遗忘因子
        """
        self.n_params = n_params
        self.n_noise_params = n_noise_params
        self.total_params = n_params + n_noise_params
        self.lambda_ = forgetting_factor

        # 参数估计
        self.theta = np.zeros(self.total_params)
        self.P = np.eye(self.total_params) * 1000.0

        # 噪声历史
        self._noise_history = deque(maxlen=n_noise_params)
        for _ in range(n_noise_params):
            self._noise_history.append(0.0)

    def update(self, phi_sys: np.ndarray, y: float) -> Tuple[np.ndarray, float]:
        """
        更新参数估计

        Args:
            phi_sys: 系统回归向量
            y: 输出观测值

        Returns:
            (更新后的参数, 预测误差)
        """
        # 构建扩展回归向量
        noise_vec = np.array(list(self._noise_history))
        phi = np.concatenate([phi_sys, -noise_vec])

        # 预测
        y_pred = phi @ self.theta
        error = y - y_pred

        # 更新噪声历史
        self._noise_history.append(error)

        # RLS更新
        Pphi = self.P @ phi
        denominator = self.lambda_ + phi @ Pphi
        K = Pphi / denominator

        self.theta = self.theta + K * error
        self.P = (self.P - np.outer(K, phi @ self.P)) / self.lambda_

        return self.theta[:self.n_params].copy(), error

    def get_noise_parameters(self) -> np.ndarray:
        """获取噪声模型参数"""
        return self.theta[self.n_params:].copy()


class SystemIdentification:
    """系统辨识"""

    def __init__(self, model_structure: ModelStructure = ModelStructure.ARX,
                 na: int = 2, nb: int = 2, nk: int = 1):
        """
        初始化系统辨识

        Args:
            model_structure: 模型结构
            na: 输出阶次
            nb: 输入阶次
            nk: 纯延迟
        """
        self.model_structure = model_structure
        self.na = na  # y的阶次
        self.nb = nb  # u的阶次
        self.nk = nk  # 延迟

        # 计算参数数量
        self.n_params = na + nb

        # 辨识器
        if model_structure == ModelStructure.ARX:
            self.identifier = RecursiveLeastSquares(self.n_params)
        elif model_structure == ModelStructure.ARMAX:
            nc = 2  # 噪声模型阶次
            self.identifier = ExtendedLeastSquares(self.n_params, nc)
        else:
            self.identifier = RecursiveLeastSquares(self.n_params)

        # 数据历史
        self._y_history = deque(maxlen=max(na, nb + nk))
        self._u_history = deque(maxlen=max(na, nb + nk))

        for _ in range(max(na, nb + nk)):
            self._y_history.append(0.0)
            self._u_history.append(0.0)

    def update(self, y: float, u: float) -> UpdateResult:
        """
        更新辨识

        Args:
            y: 输出
            u: 输入

        Returns:
            更新结果
        """
        # 构建回归向量
        phi = self._build_regressor()

        # 更新参数
        theta, error = self.identifier.update(phi, y)

        # 更新历史
        self._y_history.append(y)
        self._u_history.append(u)

        # 计算拟合度
        y_pred = phi @ theta
        fit = 1.0 - abs(error) / (abs(y) + 1e-10)

        return UpdateResult(
            timestamp=0.0,  # 外部设置
            parameters=theta,
            parameter_covariance=self.identifier.P.copy(),
            prediction_error=error,
            model_fit=max(0, min(1, fit))
        )

    def _build_regressor(self) -> np.ndarray:
        """构建回归向量"""
        phi = np.zeros(self.n_params)

        # 输出项 -y(k-1), -y(k-2), ...
        y_list = list(self._y_history)
        for i in range(self.na):
            if len(y_list) > i:
                phi[i] = -y_list[-(i + 1)]

        # 输入项 u(k-nk), u(k-nk-1), ...
        u_list = list(self._u_history)
        for i in range(self.nb):
            idx = i + self.nk
            if len(u_list) > idx:
                phi[self.na + i] = u_list[-(idx + 1)]

        return phi

    def predict(self, horizon: int = 1) -> np.ndarray:
        """多步预测"""
        predictions = np.zeros(horizon)
        theta = self.identifier.theta

        # 临时历史
        y_temp = list(self._y_history)
        u_temp = list(self._u_history)

        for k in range(horizon):
            # 构建回归向量
            phi = np.zeros(self.n_params)
            for i in range(self.na):
                if len(y_temp) > i:
                    phi[i] = -y_temp[-(i + 1)]
            for i in range(self.nb):
                idx = i + self.nk
                if len(u_temp) > idx:
                    phi[self.na + i] = u_temp[-(idx + 1)]

            # 预测
            predictions[k] = phi @ theta

            # 更新临时历史
            y_temp.append(predictions[k])

        return predictions

    def get_transfer_function(self) -> Dict[str, np.ndarray]:
        """获取传递函数系数"""
        theta = self.identifier.theta

        # A(q) = 1 + a1*q^-1 + a2*q^-2 + ...
        A = np.concatenate([[1], theta[:self.na]])

        # B(q) = b0*q^-nk + b1*q^(-nk-1) + ...
        B = theta[self.na:self.na + self.nb]

        return {'A': A, 'B': B, 'nk': self.nk}


class ParameterIdentifier(ABC):
    """参数辨识器基类"""

    def __init__(self, n_params: int):
        self.n_params = n_params
        self.parameters = np.zeros(n_params)
        self._time = 0.0

    @abstractmethod
    def identify(self, inputs: np.ndarray, outputs: np.ndarray,
                 dt: float) -> UpdateResult:
        """执行参数辨识"""
        pass

    @abstractmethod
    def reset(self):
        """重置"""
        pass


class HydraulicParameterIdentifier(ParameterIdentifier):
    """水力参数辨识器"""

    def __init__(self):
        # 参数: [曼宁系数, 波速, 局部损失系数, 泄漏系数]
        super().__init__(4)

        # 默认参数值
        self.parameters = np.array([0.015, 1000.0, 0.5, 0.0])

        # 参数约束
        self.constraints = ParameterConstraints(
            min_values=np.array([0.008, 500.0, 0.0, 0.0]),
            max_values=np.array([0.030, 1500.0, 5.0, 0.1]),
            rate_limits=np.array([0.001, 50.0, 0.1, 0.01])
        )

        # RLS辨识器
        self.rls = RecursiveLeastSquares(4, forgetting_factor=0.98)
        self.rls.theta = self.parameters.copy()

        # 数据缓存
        self._data_buffer: List[Dict] = []
        self._buffer_size = 100

    def identify(self, inputs: np.ndarray, outputs: np.ndarray,
                 dt: float) -> UpdateResult:
        """
        辨识水力参数

        Args:
            inputs: 输入 [Q1, Q2, gate_positions...]
            outputs: 输出 [H1, H2, P_max, P_min]
            dt: 时间步长

        Returns:
            更新结果
        """
        self._time += dt
        prev_params = self.parameters.copy()

        # 构建回归向量
        Q_total = inputs[0] + inputs[1] if len(inputs) > 1 else inputs[0]
        phi = np.array([
            Q_total ** 2,           # 曼宁摩擦项
            1.0,                    # 波速相关项
            Q_total ** 2,           # 局部损失项
            1.0                     # 泄漏项
        ])

        # 目标值（从高保真模型）
        if len(outputs) >= 2:
            y = outputs[0] - outputs[1]  # 水头差
        else:
            y = outputs[0]

        # RLS更新
        theta, error = self.rls.update(phi, y)

        # 应用约束
        self.parameters = self.constraints.apply(theta, prev_params, dt)
        self.rls.theta = self.parameters.copy()

        # 计算变化
        change = self.parameters - prev_params
        rel_change = np.linalg.norm(change) / (np.linalg.norm(prev_params) + 1e-10)

        return UpdateResult(
            timestamp=self._time,
            parameters=self.parameters.copy(),
            parameter_covariance=self.rls.P.copy(),
            prediction_error=error,
            model_fit=1.0 - abs(error) / (abs(y) + 1e-10),
            parameter_change=change,
            relative_change=rel_change,
            condition_number=np.linalg.cond(self.rls.P)
        )

    def reset(self):
        """重置"""
        self.parameters = np.array([0.015, 1000.0, 0.5, 0.0])
        self.rls.reset()
        self.rls.theta = self.parameters.copy()
        self._time = 0.0
        self._data_buffer.clear()

    def get_manning_coefficient(self) -> float:
        """获取曼宁系数"""
        return self.parameters[0]

    def get_wave_speed(self) -> float:
        """获取波速"""
        return self.parameters[1]


class ModelCalibrator:
    """模型校准器"""

    def __init__(self, parameter_names: List[str],
                 initial_values: np.ndarray,
                 bounds: Optional[List[Tuple[float, float]]] = None):
        """
        初始化校准器

        Args:
            parameter_names: 参数名称列表
            initial_values: 初始参数值
            bounds: 参数边界
        """
        self.parameter_names = parameter_names
        self.n_params = len(parameter_names)
        self.parameters = initial_values.copy()
        self.bounds = bounds

        # 校准历史
        self._calibration_history: List[Dict] = []

        # 目标函数定义
        self._objective_func: Optional[Callable] = None
        self._reference_data: Optional[Dict] = None

    def set_objective_function(self, func: Callable):
        """设置目标函数"""
        self._objective_func = func

    def set_reference_data(self, data: Dict[str, np.ndarray]):
        """设置参考数据（来自高保真模型）"""
        self._reference_data = data

    def calibrate(self, method: str = 'L-BFGS-B',
                 max_iterations: int = 100,
                 tolerance: float = 1e-6) -> UpdateResult:
        """
        执行校准

        Args:
            method: 优化方法
            max_iterations: 最大迭代次数
            tolerance: 收敛容差

        Returns:
            校准结果
        """
        if self._objective_func is None:
            raise ValueError("Objective function not set")

        prev_params = self.parameters.copy()

        # 优化
        result = minimize(
            self._objective_func,
            self.parameters,
            method=method,
            bounds=self.bounds,
            options={'maxiter': max_iterations, 'gtol': tolerance}
        )

        self.parameters = result.x.copy()

        # 计算变化
        change = self.parameters - prev_params
        rel_change = np.linalg.norm(change) / (np.linalg.norm(prev_params) + 1e-10)

        # 记录历史
        self._calibration_history.append({
            'parameters': self.parameters.copy(),
            'objective': result.fun,
            'success': result.success,
            'iterations': result.nit
        })

        return UpdateResult(
            timestamp=0.0,
            parameters=self.parameters.copy(),
            parameter_covariance=np.eye(self.n_params),  # 近似
            prediction_error=result.fun,
            model_fit=1.0 / (1.0 + result.fun),  # 简单转换
            parameter_change=change,
            relative_change=rel_change,
            convergence=result.success,
            iterations=result.nit
        )

    def calibrate_online(self, model_output: np.ndarray,
                        reference_output: np.ndarray,
                        learning_rate: float = 0.01) -> np.ndarray:
        """
        在线校准（梯度下降）

        Args:
            model_output: 模型输出
            reference_output: 参考输出
            learning_rate: 学习率

        Returns:
            更新后的参数
        """
        # 计算误差
        error = model_output - reference_output

        # 数值梯度
        eps = 1e-7
        gradient = np.zeros(self.n_params)

        for i in range(self.n_params):
            params_plus = self.parameters.copy()
            params_minus = self.parameters.copy()
            params_plus[i] += eps
            params_minus[i] -= eps

            # 需要模型评估函数
            if self._objective_func:
                grad_i = (self._objective_func(params_plus) -
                         self._objective_func(params_minus)) / (2 * eps)
                gradient[i] = grad_i

        # 更新参数
        self.parameters -= learning_rate * gradient

        # 应用边界约束
        if self.bounds:
            for i, (lb, ub) in enumerate(self.bounds):
                self.parameters[i] = np.clip(self.parameters[i], lb, ub)

        return self.parameters.copy()

    def get_parameter_dict(self) -> Dict[str, float]:
        """获取参数字典"""
        return dict(zip(self.parameter_names, self.parameters))


class IDZParameterUpdater:
    """IDZ模型参数动态更新器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化参数更新器

        Args:
            config: 配置字典
        """
        self.config = config or {}

        # 参数辨识器
        self.identifiers: Dict[str, ParameterIdentifier] = {}

        # 系统辨识
        self.system_id: Dict[str, SystemIdentification] = {}

        # 模型校准器
        self.calibrators: Dict[str, ModelCalibrator] = {}

        # 全局参数
        self.global_parameters: Dict[str, np.ndarray] = {}

        # 更新历史
        self._update_history: List[UpdateResult] = []
        self._max_history = 1000

        # 时间
        self._time = 0.0
        self._lock = threading.Lock()

        # 初始化默认组件
        self._init_defaults()

    def _init_defaults(self):
        """初始化默认组件"""
        # 水力参数辨识器
        self.identifiers['hydraulic'] = HydraulicParameterIdentifier()

        # 系统辨识（流量-压力关系）
        self.system_id['flow_pressure'] = SystemIdentification(
            model_structure=ModelStructure.ARX,
            na=2, nb=2, nk=1
        )

        # 模型校准器
        self.calibrators['main'] = ModelCalibrator(
            parameter_names=['manning_n', 'wave_speed', 'loss_coef', 'leakage'],
            initial_values=np.array([0.015, 1000.0, 0.5, 0.0]),
            bounds=[(0.008, 0.030), (500, 1500), (0, 5), (0, 0.1)]
        )

    def update_from_hifi_model(self, hifi_state: Dict[str, np.ndarray],
                               idz_state: Dict[str, np.ndarray],
                               control_input: np.ndarray,
                               dt: float) -> Dict[str, UpdateResult]:
        """
        基于高保真模型状态更新IDZ模型参数

        Args:
            hifi_state: 高保真模型状态
            idz_state: IDZ模型状态
            control_input: 控制输入
            dt: 时间步长

        Returns:
            各参数组的更新结果
        """
        self._time += dt
        results = {}

        with self._lock:
            # 1. 水力参数辨识
            if 'flow_rate' in hifi_state and 'pressure' in hifi_state:
                Q = hifi_state['flow_rate']
                P = hifi_state['pressure']

                inputs = np.array([np.mean(Q), np.max(Q) - np.min(Q)])
                outputs = np.array([np.max(P), np.min(P)])

                result = self.identifiers['hydraulic'].identify(inputs, outputs, dt)
                result.timestamp = self._time
                results['hydraulic'] = result

                # 更新全局参数
                self.global_parameters['hydraulic'] = result.parameters

            # 2. 系统辨识更新
            if 'flow_rate' in hifi_state and 'water_level' in hifi_state:
                Q_mean = np.mean(hifi_state['flow_rate'])
                H_mean = np.mean(hifi_state['water_level'])

                u = control_input[0] if len(control_input) > 0 else Q_mean
                y = H_mean

                sys_result = self.system_id['flow_pressure'].update(y, u)
                sys_result.timestamp = self._time
                results['system_id'] = sys_result

                self.global_parameters['system_tf'] = self.system_id['flow_pressure'].get_transfer_function()

            # 3. 模型误差校准
            if idz_state and 'pressure' in idz_state and 'pressure' in hifi_state:
                model_error = np.mean(idz_state['pressure']) - np.mean(hifi_state['pressure'])

                if abs(model_error) > 1000:  # 误差阈值
                    # 触发在线校准
                    calibrator = self.calibrators['main']
                    calibrator.set_objective_function(
                        lambda p: abs(model_error) * np.sum((p - calibrator.parameters) ** 2)
                    )
                    calibrator.calibrate_online(
                        np.array([model_error]),
                        np.zeros(1),
                        learning_rate=0.001
                    )
                    self.global_parameters['calibrated'] = calibrator.parameters.copy()

        # 保存历史
        for result in results.values():
            self._update_history.append(result)
            if len(self._update_history) > self._max_history:
                self._update_history.pop(0)

        return results

    def get_idz_parameters(self) -> Dict[str, Any]:
        """获取当前IDZ模型参数"""
        params = {}

        # 水力参数
        if 'hydraulic' in self.global_parameters:
            hp = self.global_parameters['hydraulic']
            params['manning_coefficient'] = hp[0]
            params['wave_speed'] = hp[1]
            params['local_loss_coefficient'] = hp[2]
            params['leakage_coefficient'] = hp[3]

        # 传递函数
        if 'system_tf' in self.global_parameters:
            params['transfer_function'] = self.global_parameters['system_tf']

        # 校准参数
        if 'calibrated' in self.global_parameters:
            params['calibrated_params'] = self.global_parameters['calibrated'].tolist()

        return params

    def get_parameter_statistics(self) -> Dict[str, Any]:
        """获取参数统计信息"""
        if not self._update_history:
            return {}

        recent = self._update_history[-100:]

        return {
            'mean_prediction_error': np.mean([r.prediction_error for r in recent]),
            'mean_model_fit': np.mean([r.model_fit for r in recent]),
            'mean_relative_change': np.mean([r.relative_change for r in recent]),
            'total_updates': len(self._update_history),
            'convergence_rate': sum(1 for r in recent if r.convergence) / len(recent)
        }

    def predict_parameters(self, horizon: int = 10) -> np.ndarray:
        """预测未来参数值"""
        if 'hydraulic' not in self.global_parameters:
            return np.array([])

        # 使用系统辨识进行预测
        if 'flow_pressure' in self.system_id:
            return self.system_id['flow_pressure'].predict(horizon)

        return self.global_parameters['hydraulic']

    def get_update_history(self, n_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """获取更新历史"""
        history = self._update_history[-n_samples:] if n_samples else self._update_history
        return [r.to_dict() for r in history]

    def reset(self):
        """重置"""
        self._time = 0.0
        for identifier in self.identifiers.values():
            identifier.reset()
        self.global_parameters.clear()
        self._update_history.clear()

    def save_parameters(self) -> Dict[str, Any]:
        """保存当前参数（用于持久化）"""
        return {
            'time': self._time,
            'global_parameters': {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in self.global_parameters.items()
            },
            'identifiers': {
                name: {
                    'parameters': id_.parameters.tolist()
                }
                for name, id_ in self.identifiers.items()
            }
        }

    def load_parameters(self, data: Dict[str, Any]):
        """加载参数"""
        self._time = data.get('time', 0.0)

        if 'global_parameters' in data:
            for k, v in data['global_parameters'].items():
                if isinstance(v, list):
                    self.global_parameters[k] = np.array(v)
                else:
                    self.global_parameters[k] = v

        if 'identifiers' in data:
            for name, id_data in data['identifiers'].items():
                if name in self.identifiers:
                    self.identifiers[name].parameters = np.array(id_data['parameters'])
