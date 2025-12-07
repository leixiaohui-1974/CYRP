"""
系统状态实时预测系统 - System State Real-time Prediction System

实现系统状态的实时预测，包括：
- 时间序列预测 (ARIMA, 指数平滑)
- 机器学习预测 (LSTM, 神经网络)
- 基于物理模型的预测
- 集成预测方法
- 预测不确定性量化

Implements real-time prediction of system state including:
- Time series prediction (ARIMA, Exponential Smoothing)
- Machine learning prediction (LSTM, Neural Networks)
- Physics-based prediction
- Ensemble prediction methods
- Prediction uncertainty quantification
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from collections import deque
from scipy.optimize import minimize
from scipy.linalg import inv
import threading


class PredictionMethod(Enum):
    """预测方法"""
    ARIMA = "arima"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"
    LSTM = "lstm"
    PHYSICS_BASED = "physics_based"
    ENSEMBLE = "ensemble"
    KALMAN = "kalman"


@dataclass
class PredictionInterval:
    """预测区间"""
    lower: np.ndarray
    upper: np.ndarray
    confidence_level: float = 0.95

    def contains(self, actual: np.ndarray) -> np.ndarray:
        """检查实际值是否在区间内"""
        return (actual >= self.lower) & (actual <= self.upper)


@dataclass
class PredictionResult:
    """预测结果"""
    timestamp: float
    horizon: int                          # 预测步数
    predictions: np.ndarray               # 预测值
    confidence_intervals: Optional[PredictionInterval] = None
    uncertainty: Optional[np.ndarray] = None  # 预测不确定性

    # 元信息
    method: str = ""
    computation_time: float = 0.0

    # 验证指标（事后填充）
    actual_values: Optional[np.ndarray] = None
    prediction_errors: Optional[np.ndarray] = None
    rmse: float = 0.0
    mae: float = 0.0
    mape: float = 0.0

    def compute_accuracy(self, actual: np.ndarray):
        """计算预测精度"""
        self.actual_values = actual
        self.prediction_errors = actual - self.predictions[:len(actual)]

        n = len(self.prediction_errors)
        if n > 0:
            self.rmse = np.sqrt(np.mean(self.prediction_errors ** 2))
            self.mae = np.mean(np.abs(self.prediction_errors))

            # MAPE (避免除零)
            mask = np.abs(actual) > 1e-10
            if np.any(mask):
                self.mape = np.mean(np.abs(self.prediction_errors[mask] / actual[mask])) * 100

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {
            'timestamp': self.timestamp,
            'horizon': self.horizon,
            'predictions': self.predictions.tolist(),
            'method': self.method,
            'rmse': self.rmse,
            'mae': self.mae,
            'mape': self.mape
        }

        if self.uncertainty is not None:
            result['uncertainty'] = self.uncertainty.tolist()

        return result


class BasePredictor(ABC):
    """预测器基类"""

    def __init__(self, name: str):
        self.name = name
        self._is_fitted = False
        self._history: deque = deque(maxlen=10000)
        self._time = 0.0

    @abstractmethod
    def fit(self, data: np.ndarray):
        """训练/拟合"""
        pass

    @abstractmethod
    def predict(self, horizon: int) -> PredictionResult:
        """预测"""
        pass

    def update(self, value: float, time: float):
        """更新数据"""
        self._history.append(value)
        self._time = time

    def reset(self):
        """重置"""
        self._history.clear()
        self._is_fitted = False
        self._time = 0.0


class ARIMAPredictor(BasePredictor):
    """ARIMA预测器"""

    def __init__(self, p: int = 2, d: int = 1, q: int = 2, name: str = "arima"):
        """
        初始化ARIMA预测器

        Args:
            p: AR阶数
            d: 差分阶数
            q: MA阶数
            name: 预测器名称
        """
        super().__init__(name)
        self.p = p
        self.d = d
        self.q = q

        # 模型参数
        self.ar_params = np.zeros(p)
        self.ma_params = np.zeros(q)
        self.constant = 0.0

        # 残差历史
        self._residuals = deque(maxlen=max(q, 10))
        for _ in range(max(q, 10)):
            self._residuals.append(0.0)

    def _difference(self, data: np.ndarray, order: int = 1) -> np.ndarray:
        """差分"""
        result = data.copy()
        for _ in range(order):
            result = np.diff(result)
        return result

    def _inverse_difference(self, diff_data: np.ndarray, last_values: np.ndarray) -> np.ndarray:
        """逆差分"""
        result = diff_data.copy()
        for i in range(self.d):
            result = np.cumsum(np.concatenate([[last_values[-(i + 1)]], result]))
        return result[1:]

    def fit(self, data: np.ndarray):
        """
        拟合ARIMA模型

        使用最小二乘法估计参数
        """
        if len(data) < self.p + self.d + self.q + 10:
            return

        # 差分
        diff_data = self._difference(data, self.d)

        # 构建回归矩阵
        n = len(diff_data)
        n_obs = n - max(self.p, self.q)

        if n_obs < 10:
            return

        # AR部分
        X = np.zeros((n_obs, self.p + self.q + 1))
        y = diff_data[max(self.p, self.q):]

        for i in range(n_obs):
            idx = max(self.p, self.q) + i

            # AR项
            for j in range(self.p):
                X[i, j] = diff_data[idx - j - 1]

            # MA项 (使用残差近似)
            for j in range(self.q):
                if idx - j - 1 >= 0:
                    # 初始残差用0
                    X[i, self.p + j] = 0

            # 常数项
            X[i, -1] = 1.0

        # 最小二乘估计
        try:
            params, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            self.ar_params = params[:self.p]
            self.ma_params = params[self.p:self.p + self.q]
            self.constant = params[-1]
            self._is_fitted = True
        except np.linalg.LinAlgError:
            pass

    def predict(self, horizon: int) -> PredictionResult:
        """预测"""
        import time
        start_time = time.time()

        if not self._is_fitted or len(self._history) < self.p + self.d:
            return PredictionResult(
                timestamp=self._time,
                horizon=horizon,
                predictions=np.full(horizon, np.mean(list(self._history)) if self._history else 0),
                method=self.name
            )

        data = np.array(self._history)

        # 差分后的最后几个值
        diff_data = self._difference(data, self.d)
        last_diff_values = diff_data[-(self.p + 1):].tolist()

        # 残差
        residuals = list(self._residuals)

        # 预测差分值
        predictions_diff = []

        for h in range(horizon):
            # AR项
            ar_sum = 0.0
            for j in range(self.p):
                if len(last_diff_values) > j:
                    ar_sum += self.ar_params[j] * last_diff_values[-(j + 1)]

            # MA项
            ma_sum = 0.0
            for j in range(self.q):
                if len(residuals) > j:
                    ma_sum += self.ma_params[j] * residuals[-(j + 1)]

            pred_diff = self.constant + ar_sum + ma_sum
            predictions_diff.append(pred_diff)

            # 更新历史
            last_diff_values.append(pred_diff)
            residuals.append(0)  # 未来残差未知，假设为0

        # 逆差分
        last_values = data[-self.d:] if self.d > 0 else data[-1:]
        predictions = self._inverse_difference(np.array(predictions_diff), last_values)

        # 计算预测区间
        sigma = np.std(list(self._residuals)) if self._residuals else 0.1
        z_score = 1.96  # 95%置信区间

        ci_width = np.array([sigma * np.sqrt(h + 1) * z_score for h in range(horizon)])

        ci = PredictionInterval(
            lower=predictions - ci_width,
            upper=predictions + ci_width,
            confidence_level=0.95
        )

        computation_time = time.time() - start_time

        return PredictionResult(
            timestamp=self._time,
            horizon=horizon,
            predictions=predictions,
            confidence_intervals=ci,
            uncertainty=ci_width,
            method=self.name,
            computation_time=computation_time
        )


class ExponentialSmoothingPredictor(BasePredictor):
    """指数平滑预测器"""

    def __init__(self, alpha: float = 0.3, beta: float = 0.1, gamma: float = 0.1,
                 seasonal_period: int = 0, name: str = "exp_smoothing"):
        """
        初始化指数平滑预测器

        Args:
            alpha: 水平平滑系数
            beta: 趋势平滑系数
            gamma: 季节平滑系数
            seasonal_period: 季节周期 (0表示无季节性)
            name: 预测器名称
        """
        super().__init__(name)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.seasonal_period = seasonal_period

        # 状态
        self.level = 0.0
        self.trend = 0.0
        self.seasonal = np.zeros(max(seasonal_period, 1))

    def fit(self, data: np.ndarray):
        """拟合"""
        if len(data) < 3:
            return

        # 初始化
        self.level = data[0]
        self.trend = data[1] - data[0] if len(data) > 1 else 0

        if self.seasonal_period > 0 and len(data) >= self.seasonal_period:
            self.seasonal = data[:self.seasonal_period] - np.mean(data[:self.seasonal_period])
        else:
            self.seasonal = np.zeros(max(self.seasonal_period, 1))

        # 更新状态
        for i, value in enumerate(data):
            self._update_state(value, i)

        self._is_fitted = True

    def _update_state(self, value: float, t: int):
        """更新状态"""
        if self.seasonal_period > 0:
            season_idx = t % self.seasonal_period

            # 去季节化
            value_deseasonalized = value - self.seasonal[season_idx]

            # 水平更新
            new_level = self.alpha * value_deseasonalized + (1 - self.alpha) * (self.level + self.trend)

            # 趋势更新
            new_trend = self.beta * (new_level - self.level) + (1 - self.beta) * self.trend

            # 季节更新
            self.seasonal[season_idx] = self.gamma * (value - new_level) + (1 - self.gamma) * self.seasonal[season_idx]

            self.level = new_level
            self.trend = new_trend
        else:
            # Holt线性趋势
            new_level = self.alpha * value + (1 - self.alpha) * (self.level + self.trend)
            new_trend = self.beta * (new_level - self.level) + (1 - self.beta) * self.trend

            self.level = new_level
            self.trend = new_trend

    def predict(self, horizon: int) -> PredictionResult:
        """预测"""
        predictions = np.zeros(horizon)

        for h in range(horizon):
            pred = self.level + (h + 1) * self.trend

            if self.seasonal_period > 0:
                season_idx = (len(self._history) + h) % self.seasonal_period
                pred += self.seasonal[season_idx]

            predictions[h] = pred

        # 估计预测区间
        if len(self._history) > 10:
            data = np.array(self._history)
            residuals = np.zeros(len(data) - 1)
            for i in range(1, len(data)):
                expected = self.level + i * self.trend
                residuals[i - 1] = data[i] - expected

            sigma = np.std(residuals)
        else:
            sigma = 0.1

        ci_width = np.array([sigma * np.sqrt(h + 1) * 1.96 for h in range(horizon)])

        return PredictionResult(
            timestamp=self._time,
            horizon=horizon,
            predictions=predictions,
            confidence_intervals=PredictionInterval(
                lower=predictions - ci_width,
                upper=predictions + ci_width
            ),
            uncertainty=ci_width,
            method=self.name
        )

    def update(self, value: float, time: float):
        """更新"""
        super().update(value, time)
        self._update_state(value, len(self._history))


class LSTMPredictor(BasePredictor):
    """LSTM预测器 (简化实现)"""

    def __init__(self, hidden_size: int = 32, lookback: int = 20, name: str = "lstm"):
        """
        初始化LSTM预测器

        Args:
            hidden_size: 隐藏层大小
            lookback: 回看窗口大小
            name: 预测器名称
        """
        super().__init__(name)
        self.hidden_size = hidden_size
        self.lookback = lookback

        # 权重 (简化的单层LSTM)
        self.Wf = np.random.randn(hidden_size, lookback + hidden_size) * 0.1
        self.Wi = np.random.randn(hidden_size, lookback + hidden_size) * 0.1
        self.Wc = np.random.randn(hidden_size, lookback + hidden_size) * 0.1
        self.Wo = np.random.randn(hidden_size, lookback + hidden_size) * 0.1
        self.Wy = np.random.randn(1, hidden_size) * 0.1

        self.bf = np.zeros(hidden_size)
        self.bi = np.zeros(hidden_size)
        self.bc = np.zeros(hidden_size)
        self.bo = np.zeros(hidden_size)
        self.by = np.zeros(1)

        # 状态
        self.h = np.zeros(hidden_size)
        self.c = np.zeros(hidden_size)

        # 归一化参数
        self.mean = 0.0
        self.std = 1.0

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid激活"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def _tanh(self, x: np.ndarray) -> np.ndarray:
        """Tanh激活"""
        return np.tanh(np.clip(x, -500, 500))

    def _lstm_cell(self, x: np.ndarray) -> float:
        """LSTM单元前向传播"""
        # 连接输入和隐藏状态
        combined = np.concatenate([x, self.h])

        # 门计算
        f = self._sigmoid(self.Wf @ combined + self.bf)
        i = self._sigmoid(self.Wi @ combined + self.bi)
        c_tilde = self._tanh(self.Wc @ combined + self.bc)
        o = self._sigmoid(self.Wo @ combined + self.bo)

        # 状态更新
        self.c = f * self.c + i * c_tilde
        self.h = o * self._tanh(self.c)

        # 输出
        y = self.Wy @ self.h + self.by
        return y[0]

    def fit(self, data: np.ndarray):
        """
        拟合LSTM

        使用简化的在线学习
        """
        if len(data) < self.lookback + 10:
            return

        # 归一化
        self.mean = np.mean(data)
        self.std = np.std(data) + 1e-10
        data_norm = (data - self.mean) / self.std

        # 简化训练：使用随机初始化权重
        # 实际应用中应使用反向传播训练
        self._is_fitted = True

        # 初始化状态
        for i in range(self.lookback, len(data)):
            x = data_norm[i - self.lookback:i]
            self._lstm_cell(x)

    def predict(self, horizon: int) -> PredictionResult:
        """预测"""
        if not self._is_fitted or len(self._history) < self.lookback:
            # 退化到简单预测
            mean_val = np.mean(list(self._history)) if self._history else 0
            return PredictionResult(
                timestamp=self._time,
                horizon=horizon,
                predictions=np.full(horizon, mean_val),
                method=self.name
            )

        predictions = np.zeros(horizon)
        data = np.array(list(self._history)[-self.lookback:])
        data_norm = (data - self.mean) / self.std

        # 保存状态
        h_save = self.h.copy()
        c_save = self.c.copy()

        for h in range(horizon):
            x = data_norm[-self.lookback:]
            pred_norm = self._lstm_cell(x)
            pred = pred_norm * self.std + self.mean
            predictions[h] = pred

            # 更新输入序列
            data_norm = np.append(data_norm[1:], pred_norm)

        # 恢复状态
        self.h = h_save
        self.c = c_save

        # 估计不确定性
        uncertainty = np.array([self.std * np.sqrt(h + 1) for h in range(horizon)])

        return PredictionResult(
            timestamp=self._time,
            horizon=horizon,
            predictions=predictions,
            uncertainty=uncertainty,
            confidence_intervals=PredictionInterval(
                lower=predictions - 1.96 * uncertainty,
                upper=predictions + 1.96 * uncertainty
            ),
            method=self.name
        )


class PhysicsBasedPredictor(BasePredictor):
    """基于物理模型的预测器"""

    def __init__(self, model_func: Optional[Callable] = None, name: str = "physics"):
        """
        初始化物理预测器

        Args:
            model_func: 物理模型函数 (state, params, dt) -> next_state
            name: 预测器名称
        """
        super().__init__(name)
        self.model_func = model_func or self._default_model

        # 模型参数
        self.parameters = {
            'wave_speed': 1000.0,
            'friction_factor': 0.02,
            'area': 38.48,  # pi * 3.5^2
            'length': 4250.0
        }

        # 状态
        self.state = np.zeros(4)  # [Q1, Q2, H1, H2]

    def _default_model(self, state: np.ndarray, params: Dict, dt: float) -> np.ndarray:
        """
        默认物理模型（简化的水力学方程）

        dQ/dt = -g*A*(dH/dx) - f*Q*|Q|/(2*D*A)
        dH/dt = -a²/(g*A)*(dQ/dx)
        """
        Q1, Q2, H1, H2 = state
        a = params.get('wave_speed', 1000.0)
        f = params.get('friction_factor', 0.02)
        A = params.get('area', 38.48)
        L = params.get('length', 4250.0)
        g = 9.81
        D = np.sqrt(4 * A / np.pi)

        # 简化的差分近似
        dQ_dx = (Q2 - Q1) / L
        dH_dx = (H2 - H1) / L

        # 动量方程
        Q_mean = (Q1 + Q2) / 2
        friction = f * Q_mean * abs(Q_mean) / (2 * D * A)

        dQ1_dt = -g * A * dH_dx - friction
        dQ2_dt = dQ1_dt  # 简化假设

        # 连续方程
        dH1_dt = -a**2 / (g * A) * dQ_dx
        dH2_dt = dH1_dt

        # 欧拉积分
        next_state = np.array([
            Q1 + dQ1_dt * dt,
            Q2 + dQ2_dt * dt,
            H1 + dH1_dt * dt,
            H2 + dH2_dt * dt
        ])

        return next_state

    def set_state(self, state: np.ndarray):
        """设置当前状态"""
        self.state = state.copy()

    def set_parameters(self, params: Dict[str, float]):
        """设置模型参数"""
        self.parameters.update(params)

    def fit(self, data: np.ndarray):
        """拟合（对于物理模型，主要是参数校准）"""
        if len(data) < 10:
            return

        # 简单校准：基于数据估计摩擦系数
        # 实际应用中应使用优化方法
        self._is_fitted = True

    def predict(self, horizon: int, dt: float = 1.0) -> PredictionResult:
        """
        预测

        Args:
            horizon: 预测步数
            dt: 时间步长
        """
        predictions = np.zeros((horizon, len(self.state)))
        state = self.state.copy()

        for h in range(horizon):
            state = self.model_func(state, self.parameters, dt)
            predictions[h] = state

        # 主要输出（假设第一个是流量）
        main_predictions = predictions[:, 0]

        # 不确定性估计（基于模型参数不确定性）
        param_uncertainty = 0.05  # 5%参数不确定性
        uncertainty = np.abs(main_predictions) * param_uncertainty * np.sqrt(np.arange(1, horizon + 1))

        return PredictionResult(
            timestamp=self._time,
            horizon=horizon,
            predictions=main_predictions,
            uncertainty=uncertainty,
            confidence_intervals=PredictionInterval(
                lower=main_predictions - 1.96 * uncertainty,
                upper=main_predictions + 1.96 * uncertainty
            ),
            method=self.name
        )


class EnsemblePredictor(BasePredictor):
    """集成预测器"""

    def __init__(self, predictors: Optional[List[BasePredictor]] = None,
                 weights: Optional[np.ndarray] = None,
                 name: str = "ensemble"):
        """
        初始化集成预测器

        Args:
            predictors: 基预测器列表
            weights: 权重
            name: 预测器名称
        """
        super().__init__(name)

        self.predictors = predictors or []
        self.weights = weights if weights is not None else np.ones(len(self.predictors))

        # 归一化权重
        if len(self.weights) > 0:
            self.weights = self.weights / np.sum(self.weights)

        # 自适应权重
        self._prediction_errors: Dict[str, List[float]] = {
            p.name: [] for p in self.predictors
        }

    def add_predictor(self, predictor: BasePredictor, weight: float = 1.0):
        """添加预测器"""
        self.predictors.append(predictor)
        self.weights = np.append(self.weights, weight)
        self.weights = self.weights / np.sum(self.weights)
        self._prediction_errors[predictor.name] = []

    def fit(self, data: np.ndarray):
        """拟合所有基预测器"""
        for predictor in self.predictors:
            predictor.fit(data)
        self._is_fitted = True

    def update(self, value: float, time: float):
        """更新所有预测器"""
        super().update(value, time)
        for predictor in self.predictors:
            predictor.update(value, time)

    def predict(self, horizon: int) -> PredictionResult:
        """集成预测"""
        if not self.predictors:
            return PredictionResult(
                timestamp=self._time,
                horizon=horizon,
                predictions=np.zeros(horizon),
                method=self.name
            )

        # 收集所有预测
        all_predictions = []
        all_uncertainties = []

        for predictor in self.predictors:
            result = predictor.predict(horizon)
            all_predictions.append(result.predictions)
            if result.uncertainty is not None:
                all_uncertainties.append(result.uncertainty)
            else:
                all_uncertainties.append(np.ones(horizon) * 0.1)

        all_predictions = np.array(all_predictions)
        all_uncertainties = np.array(all_uncertainties)

        # 加权平均
        ensemble_prediction = np.sum(
            self.weights[:, np.newaxis] * all_predictions, axis=0
        )

        # 不确定性合并（考虑预测器之间的差异）
        prediction_spread = np.std(all_predictions, axis=0)
        weighted_uncertainty = np.sum(
            self.weights[:, np.newaxis] * all_uncertainties, axis=0
        )
        total_uncertainty = np.sqrt(weighted_uncertainty ** 2 + prediction_spread ** 2)

        return PredictionResult(
            timestamp=self._time,
            horizon=horizon,
            predictions=ensemble_prediction,
            uncertainty=total_uncertainty,
            confidence_intervals=PredictionInterval(
                lower=ensemble_prediction - 1.96 * total_uncertainty,
                upper=ensemble_prediction + 1.96 * total_uncertainty
            ),
            method=self.name
        )

    def update_weights_adaptive(self, actual_value: float):
        """自适应更新权重"""
        if len(self._history) < 2:
            return

        # 计算每个预测器的误差
        for i, predictor in enumerate(self.predictors):
            # 获取上一步的1步预测
            result = predictor.predict(1)
            if len(result.predictions) > 0:
                error = abs(actual_value - result.predictions[0])
                self._prediction_errors[predictor.name].append(error)

                # 限制历史长度
                if len(self._prediction_errors[predictor.name]) > 100:
                    self._prediction_errors[predictor.name].pop(0)

        # 基于误差更新权重
        mean_errors = []
        for predictor in self.predictors:
            errors = self._prediction_errors[predictor.name]
            if errors:
                mean_errors.append(np.mean(errors))
            else:
                mean_errors.append(1.0)

        mean_errors = np.array(mean_errors)

        # 逆误差加权
        if np.min(mean_errors) > 0:
            inv_errors = 1.0 / (mean_errors + 1e-10)
            self.weights = inv_errors / np.sum(inv_errors)


class StatePredictorManager:
    """状态预测管理器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.predictors: Dict[str, BasePredictor] = {}
        self.ensemble_predictors: Dict[str, EnsemblePredictor] = {}

        # 预测历史
        self._prediction_history: List[PredictionResult] = []
        self._max_history = 1000

        # 时间
        self._time = 0.0
        self._lock = threading.Lock()

        # 初始化默认预测器
        self._init_defaults()

    def _init_defaults(self):
        """初始化默认预测器"""
        # ARIMA预测器
        self.predictors['arima'] = ARIMAPredictor(p=2, d=1, q=2)

        # 指数平滑预测器
        self.predictors['exp_smoothing'] = ExponentialSmoothingPredictor(
            alpha=0.3, beta=0.1
        )

        # LSTM预测器
        self.predictors['lstm'] = LSTMPredictor(hidden_size=32, lookback=20)

        # 物理预测器
        self.predictors['physics'] = PhysicsBasedPredictor()

        # 集成预测器
        ensemble = EnsemblePredictor(name='main_ensemble')
        ensemble.add_predictor(self.predictors['arima'], weight=0.3)
        ensemble.add_predictor(self.predictors['exp_smoothing'], weight=0.3)
        ensemble.add_predictor(self.predictors['physics'], weight=0.4)
        self.ensemble_predictors['main'] = ensemble

    def add_predictor(self, name: str, predictor: BasePredictor):
        """添加预测器"""
        with self._lock:
            self.predictors[name] = predictor

    def update(self, values: Dict[str, float], time: float):
        """更新所有预测器"""
        self._time = time

        with self._lock:
            for name, predictor in self.predictors.items():
                if name in values:
                    predictor.update(values[name], time)

            for name, ensemble in self.ensemble_predictors.items():
                if name in values:
                    ensemble.update(values[name], time)
                    ensemble.update_weights_adaptive(values[name])

    def predict(self, variable: str, horizon: int,
                method: Optional[str] = None) -> PredictionResult:
        """
        执行预测

        Args:
            variable: 变量名
            horizon: 预测步数
            method: 预测方法（None表示使用集成）

        Returns:
            预测结果
        """
        with self._lock:
            if method and method in self.predictors:
                result = self.predictors[method].predict(horizon)
            elif 'main' in self.ensemble_predictors:
                result = self.ensemble_predictors['main'].predict(horizon)
            elif self.predictors:
                # 使用第一个可用的预测器
                result = list(self.predictors.values())[0].predict(horizon)
            else:
                result = PredictionResult(
                    timestamp=self._time,
                    horizon=horizon,
                    predictions=np.zeros(horizon),
                    method="none"
                )

            result.timestamp = self._time

            # 保存历史
            self._prediction_history.append(result)
            if len(self._prediction_history) > self._max_history:
                self._prediction_history.pop(0)

            return result

    def predict_multi_variable(self, variables: List[str], horizon: int
                               ) -> Dict[str, PredictionResult]:
        """多变量预测"""
        results = {}
        for var in variables:
            results[var] = self.predict(var, horizon)
        return results

    def fit_all(self, data: Dict[str, np.ndarray]):
        """拟合所有预测器"""
        with self._lock:
            for name, predictor in self.predictors.items():
                if name in data:
                    predictor.fit(data[name])

            for name, ensemble in self.ensemble_predictors.items():
                if name in data:
                    ensemble.fit(data[name])

    def get_prediction_accuracy(self, n_recent: int = 100) -> Dict[str, Any]:
        """获取预测精度统计"""
        recent = self._prediction_history[-n_recent:]

        if not recent:
            return {}

        rmses = [r.rmse for r in recent if r.rmse > 0]
        maes = [r.mae for r in recent if r.mae > 0]
        mapes = [r.mape for r in recent if r.mape > 0]

        return {
            'mean_rmse': np.mean(rmses) if rmses else 0,
            'mean_mae': np.mean(maes) if maes else 0,
            'mean_mape': np.mean(mapes) if mapes else 0,
            'n_predictions': len(recent)
        }

    def get_prediction_history(self, n_samples: Optional[int] = None
                               ) -> List[Dict[str, Any]]:
        """获取预测历史"""
        history = self._prediction_history[-n_samples:] if n_samples else self._prediction_history
        return [r.to_dict() for r in history]

    def reset(self):
        """重置"""
        self._time = 0.0
        for predictor in self.predictors.values():
            predictor.reset()
        for ensemble in self.ensemble_predictors.values():
            ensemble.reset()
        self._prediction_history.clear()
