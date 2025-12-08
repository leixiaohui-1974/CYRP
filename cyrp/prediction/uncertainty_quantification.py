"""
预测不确定性量化系统 - Prediction Uncertainty Quantification System

实现贝叶斯不确定性量化，包括：
- 校准的预测区间
- 模型不确定性估计
- 集成预测的不确定性融合
- 在线不确定性学习

Implements Bayesian uncertainty quantification including:
- Calibrated prediction intervals
- Model uncertainty estimation
- Ensemble prediction uncertainty fusion
- Online uncertainty learning
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
from scipy import stats
from scipy.special import logsumexp


@dataclass
class UncertaintyEstimate:
    """不确定性估计结果"""
    mean: np.ndarray                    # 预测均值
    variance: np.ndarray                # 预测方差
    aleatoric: np.ndarray               # 随机不确定性 (数据噪声)
    epistemic: np.ndarray               # 认知不确定性 (模型不确定性)
    confidence_intervals: Dict[float, Tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)

    @property
    def total_uncertainty(self) -> np.ndarray:
        """总不确定性"""
        return np.sqrt(self.aleatoric + self.epistemic)

    def get_interval(self, confidence: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """获取指定置信度的预测区间"""
        if confidence in self.confidence_intervals:
            return self.confidence_intervals[confidence]

        z = stats.norm.ppf((1 + confidence) / 2)
        std = self.total_uncertainty
        return (self.mean - z * std, self.mean + z * std)


class BayesianUncertaintyQuantifier:
    """贝叶斯不确定性量化器"""

    def __init__(
        self,
        prior_variance: float = 1.0,
        observation_noise: float = 0.1,
        n_samples: int = 100
    ):
        """
        初始化贝叶斯不确定性量化器

        Args:
            prior_variance: 先验方差
            observation_noise: 观测噪声
            n_samples: 采样数量
        """
        self.prior_variance = prior_variance
        self.observation_noise = observation_noise
        self.n_samples = n_samples

        # 在线学习参数
        self._posterior_mean = 0.0
        self._posterior_variance = prior_variance
        self._n_observations = 0

        # 历史数据
        self._prediction_history: deque = deque(maxlen=1000)
        self._error_history: deque = deque(maxlen=1000)

    def update_posterior(self, predicted: float, actual: float):
        """
        更新后验分布 (贝叶斯在线学习)

        Args:
            predicted: 预测值
            actual: 实际值
        """
        error = actual - predicted
        self._prediction_history.append(predicted)
        self._error_history.append(error)
        self._n_observations += 1

        # Kalman-like 更新
        # 后验方差
        posterior_precision = 1.0 / self._posterior_variance + 1.0 / self.observation_noise
        new_posterior_variance = 1.0 / posterior_precision

        # 后验均值 (误差的均值)
        k = self._posterior_variance / (self._posterior_variance + self.observation_noise)
        self._posterior_mean = self._posterior_mean + k * (error - self._posterior_mean)
        self._posterior_variance = new_posterior_variance

    def quantify(
        self,
        predictions: np.ndarray,
        model_variances: Optional[np.ndarray] = None
    ) -> UncertaintyEstimate:
        """
        量化预测的不确定性

        Args:
            predictions: 预测值数组
            model_variances: 模型预测的方差 (可选)

        Returns:
            UncertaintyEstimate: 不确定性估计结果
        """
        n = len(predictions)

        # 随机不确定性 (数据噪声)
        aleatoric = np.full(n, self.observation_noise)

        # 认知不确定性 (模型不确定性)
        if model_variances is not None:
            epistemic = model_variances
        else:
            # 基于历史误差估计
            if len(self._error_history) > 10:
                epistemic = np.full(n, np.var(list(self._error_history)))
            else:
                epistemic = np.full(n, self._posterior_variance)

        # 计算多个置信水平的区间
        confidence_levels = [0.50, 0.80, 0.90, 0.95, 0.99]
        confidence_intervals = {}

        total_std = np.sqrt(aleatoric + epistemic)
        for conf in confidence_levels:
            z = stats.norm.ppf((1 + conf) / 2)
            lower = predictions - z * total_std
            upper = predictions + z * total_std
            confidence_intervals[conf] = (lower, upper)

        return UncertaintyEstimate(
            mean=predictions,
            variance=aleatoric + epistemic,
            aleatoric=aleatoric,
            epistemic=epistemic,
            confidence_intervals=confidence_intervals
        )

    def calibrate(
        self,
        historical_predictions: np.ndarray,
        historical_actuals: np.ndarray,
        confidence_levels: List[float] = None
    ) -> Dict[float, float]:
        """
        校准预测区间

        检验预测区间的覆盖率是否与名义置信度匹配

        Args:
            historical_predictions: 历史预测值
            historical_actuals: 历史实际值
            confidence_levels: 要检验的置信水平列表

        Returns:
            Dict[float, float]: 各置信水平的实际覆盖率
        """
        if confidence_levels is None:
            confidence_levels = [0.50, 0.80, 0.90, 0.95]

        errors = historical_actuals - historical_predictions
        error_std = np.std(errors)

        coverage = {}
        for conf in confidence_levels:
            z = stats.norm.ppf((1 + conf) / 2)
            lower = historical_predictions - z * error_std
            upper = historical_predictions + z * error_std

            in_interval = (historical_actuals >= lower) & (historical_actuals <= upper)
            coverage[conf] = np.mean(in_interval)

        return coverage

    def get_calibration_factor(self) -> float:
        """
        获取校准因子

        如果预测区间过窄或过宽，返回调整因子
        """
        if len(self._error_history) < 30:
            return 1.0

        errors = np.array(list(self._error_history))
        empirical_std = np.std(errors)
        model_std = np.sqrt(self._posterior_variance + self.observation_noise)

        return empirical_std / model_std if model_std > 0 else 1.0


class EnsembleUncertaintyFusion:
    """集成预测不确定性融合"""

    def __init__(self, n_models: int):
        """
        初始化集成不确定性融合

        Args:
            n_models: 模型数量
        """
        self.n_models = n_models
        self.model_weights = np.ones(n_models) / n_models
        self._performance_history: List[List[float]] = [[] for _ in range(n_models)]

    def fuse(
        self,
        predictions: List[np.ndarray],
        variances: List[np.ndarray]
    ) -> UncertaintyEstimate:
        """
        融合多个模型的预测和不确定性

        Args:
            predictions: 各模型的预测值列表
            variances: 各模型的预测方差列表

        Returns:
            UncertaintyEstimate: 融合后的不确定性估计
        """
        n_models = len(predictions)
        horizon = len(predictions[0])

        # 堆叠预测
        pred_stack = np.stack(predictions)  # (n_models, horizon)
        var_stack = np.stack(variances)

        # 加权平均预测
        mean_pred = np.average(pred_stack, axis=0, weights=self.model_weights)

        # 随机不确定性: 加权平均方差
        aleatoric = np.average(var_stack, axis=0, weights=self.model_weights)

        # 认知不确定性: 模型间分歧
        epistemic = np.average((pred_stack - mean_pred) ** 2, axis=0, weights=self.model_weights)

        # 计算置信区间
        total_std = np.sqrt(aleatoric + epistemic)
        confidence_intervals = {}
        for conf in [0.50, 0.80, 0.90, 0.95, 0.99]:
            z = stats.norm.ppf((1 + conf) / 2)
            confidence_intervals[conf] = (
                mean_pred - z * total_std,
                mean_pred + z * total_std
            )

        return UncertaintyEstimate(
            mean=mean_pred,
            variance=aleatoric + epistemic,
            aleatoric=aleatoric,
            epistemic=epistemic,
            confidence_intervals=confidence_intervals
        )

    def update_weights(self, model_errors: List[float]):
        """
        根据模型性能更新权重

        Args:
            model_errors: 各模型的预测误差
        """
        for i, error in enumerate(model_errors):
            self._performance_history[i].append(abs(error))

        # 基于指数加权性能计算权重
        if len(self._performance_history[0]) >= 10:
            recent_mae = []
            for history in self._performance_history:
                recent = history[-50:]  # 最近50个样本
                recent_mae.append(np.mean(recent))

            # 反比权重
            inv_mae = 1.0 / (np.array(recent_mae) + 1e-6)
            self.model_weights = inv_mae / np.sum(inv_mae)


class ConformalPredictor:
    """保型预测区间 (Conformal Prediction)"""

    def __init__(self, coverage: float = 0.95, max_history: int = 500):
        """
        初始化保型预测器

        Args:
            coverage: 目标覆盖率
            max_history: 最大历史长度
        """
        self.coverage = coverage
        self.max_history = max_history
        self._nonconformity_scores: deque = deque(maxlen=max_history)

    def update(self, predicted: float, actual: float):
        """
        更新非一致性分数

        Args:
            predicted: 预测值
            actual: 实际值
        """
        score = abs(actual - predicted)
        self._nonconformity_scores.append(score)

    def predict_interval(self, prediction: float) -> Tuple[float, float]:
        """
        计算保型预测区间

        Args:
            prediction: 点预测值

        Returns:
            (lower, upper): 预测区间
        """
        if len(self._nonconformity_scores) < 10:
            # 默认区间
            return (prediction - 10.0, prediction + 10.0)

        scores = np.array(list(self._nonconformity_scores))

        # 计算分位数
        n = len(scores)
        k = int(np.ceil((n + 1) * self.coverage))
        k = min(k, n)

        sorted_scores = np.sort(scores)
        width = sorted_scores[k - 1]

        return (prediction - width, prediction + width)

    def predict_intervals_batch(
        self,
        predictions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        批量计算保型预测区间

        Args:
            predictions: 预测值数组

        Returns:
            (lower, upper): 预测区间数组
        """
        if len(self._nonconformity_scores) < 10:
            width = 10.0
        else:
            scores = np.array(list(self._nonconformity_scores))
            n = len(scores)
            k = int(np.ceil((n + 1) * self.coverage))
            k = min(k, n)
            width = np.sort(scores)[k - 1]

        # 随预测步长增加不确定性
        horizons = np.arange(1, len(predictions) + 1)
        widths = width * np.sqrt(horizons)

        return (predictions - widths, predictions + widths)

    def get_empirical_coverage(self) -> float:
        """
        获取经验覆盖率
        """
        if len(self._nonconformity_scores) < 10:
            return self.coverage

        scores = np.array(list(self._nonconformity_scores))
        n = len(scores)
        k = int(np.ceil((n + 1) * self.coverage))
        k = min(k, n)
        threshold = np.sort(scores)[k - 1]

        return np.mean(scores <= threshold)


class UncertaintyMonitor:
    """不确定性监控器"""

    def __init__(
        self,
        warning_threshold: float = 2.0,
        critical_threshold: float = 3.0
    ):
        """
        初始化不确定性监控器

        Args:
            warning_threshold: 预警阈值 (相对于基线的倍数)
            critical_threshold: 严重阈值
        """
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold

        self._baseline_uncertainty: Optional[float] = None
        self._uncertainty_history: deque = deque(maxlen=500)

    def update(self, uncertainty: float):
        """更新不确定性"""
        self._uncertainty_history.append(uncertainty)

        # 更新基线
        if len(self._uncertainty_history) >= 100:
            self._baseline_uncertainty = np.percentile(list(self._uncertainty_history), 25)

    def check_status(self, current_uncertainty: float) -> str:
        """
        检查当前不确定性状态

        Args:
            current_uncertainty: 当前不确定性

        Returns:
            str: "normal", "warning", 或 "critical"
        """
        if self._baseline_uncertainty is None:
            return "normal"

        ratio = current_uncertainty / self._baseline_uncertainty

        if ratio >= self.critical_threshold:
            return "critical"
        elif ratio >= self.warning_threshold:
            return "warning"
        return "normal"

    def get_statistics(self) -> Dict[str, float]:
        """获取统计信息"""
        if not self._uncertainty_history:
            return {}

        uncertainties = np.array(list(self._uncertainty_history))
        return {
            'mean': float(np.mean(uncertainties)),
            'std': float(np.std(uncertainties)),
            'min': float(np.min(uncertainties)),
            'max': float(np.max(uncertainties)),
            'baseline': float(self._baseline_uncertainty) if self._baseline_uncertainty else 0.0,
            'current_ratio': float(uncertainties[-1] / self._baseline_uncertainty)
                            if self._baseline_uncertainty else 0.0
        }
