"""
系统状态实时评价系统 - System State Real-time Evaluation System

实现系统状态与控制目标偏差的实时评价，包括：
- 状态偏差计算
- 控制性能评价
- 安全边界评估
- 多目标综合评价
- 趋势分析和预警

Implements real-time evaluation of system state including:
- State deviation calculation
- Control performance evaluation
- Safety boundary assessment
- Multi-objective comprehensive evaluation
- Trend analysis and early warning
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime
import threading


class EvaluationLevel(Enum):
    """评价等级"""
    EXCELLENT = "excellent"     # 优秀 (>95%)
    GOOD = "good"              # 良好 (80-95%)
    ACCEPTABLE = "acceptable"   # 可接受 (60-80%)
    MARGINAL = "marginal"      # 边缘 (40-60%)
    POOR = "poor"              # 差 (<40%)
    CRITICAL = "critical"      # 危险


class ObjectiveType(Enum):
    """控制目标类型"""
    SETPOINT = "setpoint"           # 定值控制
    TRACKING = "tracking"           # 跟踪控制
    REGULATION = "regulation"       # 调节控制
    OPTIMIZATION = "optimization"   # 优化控制
    CONSTRAINT = "constraint"       # 约束满足


class SafetyStatus(Enum):
    """安全状态"""
    SAFE = "safe"
    WARNING = "warning"
    ALARM = "alarm"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class ControlObjective:
    """控制目标"""
    name: str
    objective_type: ObjectiveType
    target_value: float
    tolerance: float = 0.05          # 容差 (相对于目标值)
    weight: float = 1.0              # 权重
    priority: int = 1                # 优先级 (1最高)

    # 约束
    min_value: Optional[float] = None
    max_value: Optional[float] = None

    # 动态目标
    trajectory: Optional[Callable[[float], float]] = None  # 时间相关目标

    def get_target(self, time: float = 0.0) -> float:
        """获取目标值"""
        if self.trajectory is not None:
            return self.trajectory(time)
        return self.target_value

    def compute_deviation(self, actual_value: float, time: float = 0.0) -> float:
        """计算偏差"""
        target = self.get_target(time)
        return actual_value - target

    def compute_relative_deviation(self, actual_value: float, time: float = 0.0) -> float:
        """计算相对偏差"""
        target = self.get_target(time)
        if abs(target) > 1e-10:
            return (actual_value - target) / target
        return actual_value - target

    def is_within_tolerance(self, actual_value: float, time: float = 0.0) -> bool:
        """检查是否在容差内"""
        rel_dev = abs(self.compute_relative_deviation(actual_value, time))
        return rel_dev <= self.tolerance


@dataclass
class PerformanceMetrics:
    """性能指标"""
    timestamp: float

    # 偏差指标
    error: float = 0.0               # 当前误差
    abs_error: float = 0.0           # 绝对误差
    relative_error: float = 0.0      # 相对误差

    # 积分指标
    iae: float = 0.0                 # 积分绝对误差
    ise: float = 0.0                 # 积分平方误差
    itae: float = 0.0                # 时间加权积分绝对误差

    # 统计指标
    rmse: float = 0.0                # 均方根误差
    mae: float = 0.0                 # 平均绝对误差
    max_error: float = 0.0           # 最大误差

    # 动态指标
    settling_time: float = 0.0       # 调节时间
    rise_time: float = 0.0           # 上升时间
    overshoot: float = 0.0           # 超调量
    undershoot: float = 0.0          # 下冲量

    # 综合评价
    overall_score: float = 1.0       # 综合得分 (0-1)
    evaluation_level: EvaluationLevel = EvaluationLevel.GOOD

    def compute_score(self) -> float:
        """计算综合得分"""
        # 基于相对误差的得分
        error_score = max(0, 1.0 - abs(self.relative_error))

        # 基于超调量的得分
        overshoot_score = max(0, 1.0 - self.overshoot / 0.2)

        # 基于调节时间的得分（假设参考时间为60秒）
        settling_score = max(0, 1.0 - self.settling_time / 120.0)

        # 综合得分
        self.overall_score = 0.5 * error_score + 0.3 * overshoot_score + 0.2 * settling_score

        # 确定等级
        if self.overall_score >= 0.95:
            self.evaluation_level = EvaluationLevel.EXCELLENT
        elif self.overall_score >= 0.80:
            self.evaluation_level = EvaluationLevel.GOOD
        elif self.overall_score >= 0.60:
            self.evaluation_level = EvaluationLevel.ACCEPTABLE
        elif self.overall_score >= 0.40:
            self.evaluation_level = EvaluationLevel.MARGINAL
        else:
            self.evaluation_level = EvaluationLevel.POOR

        return self.overall_score

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'timestamp': self.timestamp,
            'error': self.error,
            'relative_error': self.relative_error,
            'rmse': self.rmse,
            'mae': self.mae,
            'max_error': self.max_error,
            'settling_time': self.settling_time,
            'overshoot': self.overshoot,
            'overall_score': self.overall_score,
            'evaluation_level': self.evaluation_level.value
        }


@dataclass
class EvaluationResult:
    """评价结果"""
    timestamp: float
    objectives_evaluation: Dict[str, PerformanceMetrics]  # 各目标评价
    safety_status: SafetyStatus
    overall_score: float
    overall_level: EvaluationLevel

    # 告警
    warnings: List[str] = field(default_factory=list)
    alarms: List[str] = field(default_factory=list)

    # 建议
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'timestamp': self.timestamp,
            'objectives': {
                name: metrics.to_dict()
                for name, metrics in self.objectives_evaluation.items()
            },
            'safety_status': self.safety_status.value,
            'overall_score': self.overall_score,
            'overall_level': self.overall_level.value,
            'warnings': self.warnings,
            'alarms': self.alarms,
            'recommendations': self.recommendations
        }


class ObjectiveTracker:
    """目标跟踪器"""

    def __init__(self, objective: ControlObjective, window_size: int = 1000):
        self.objective = objective
        self.window_size = window_size

        # 历史数据
        self._error_history = deque(maxlen=window_size)
        self._value_history = deque(maxlen=window_size)
        self._time_history = deque(maxlen=window_size)

        # 积分指标
        self._iae = 0.0
        self._ise = 0.0
        self._itae = 0.0

        # 动态响应分析
        self._settling_time = 0.0
        self._rise_time = 0.0
        self._max_overshoot = 0.0
        self._step_change_time: Optional[float] = None
        self._step_initial_value: Optional[float] = None

    def update(self, actual_value: float, time: float, dt: float) -> PerformanceMetrics:
        """
        更新跟踪

        Args:
            actual_value: 实际值
            time: 当前时间
            dt: 时间步长

        Returns:
            性能指标
        """
        target = self.objective.get_target(time)
        error = actual_value - target

        # 保存历史
        self._error_history.append(error)
        self._value_history.append(actual_value)
        self._time_history.append(time)

        # 更新积分指标
        self._iae += abs(error) * dt
        self._ise += error ** 2 * dt
        self._itae += time * abs(error) * dt

        # 检测阶跃变化
        self._detect_step_response(actual_value, time)

        # 计算统计指标
        errors = np.array(self._error_history)
        rmse = np.sqrt(np.mean(errors ** 2))
        mae = np.mean(np.abs(errors))
        max_error = np.max(np.abs(errors))

        # 相对误差
        if abs(target) > 1e-10:
            relative_error = error / target
        else:
            relative_error = error

        # 构建指标
        metrics = PerformanceMetrics(
            timestamp=time,
            error=error,
            abs_error=abs(error),
            relative_error=relative_error,
            iae=self._iae,
            ise=self._ise,
            itae=self._itae,
            rmse=rmse,
            mae=mae,
            max_error=max_error,
            settling_time=self._settling_time,
            rise_time=self._rise_time,
            overshoot=self._max_overshoot
        )

        metrics.compute_score()

        return metrics

    def _detect_step_response(self, value: float, time: float):
        """检测阶跃响应特性"""
        if len(self._value_history) < 2:
            return

        # 检测阶跃变化（目标值变化）
        target_now = self.objective.get_target(time)
        target_prev = self.objective.get_target(time - 0.1)

        if abs(target_now - target_prev) > 0.01 * abs(target_now):
            # 发生阶跃变化
            self._step_change_time = time
            self._step_initial_value = self._value_history[-2]
            self._max_overshoot = 0.0

        if self._step_change_time is not None:
            target = self.objective.get_target(time)
            step_size = target - (self._step_initial_value or 0)

            if abs(step_size) > 1e-10:
                # 计算超调量
                if step_size > 0 and value > target:
                    overshoot = (value - target) / step_size
                    self._max_overshoot = max(self._max_overshoot, overshoot)
                elif step_size < 0 and value < target:
                    overshoot = (target - value) / abs(step_size)
                    self._max_overshoot = max(self._max_overshoot, overshoot)

                # 计算调节时间（2%容差带）
                tolerance = 0.02 * abs(step_size)
                if abs(value - target) <= tolerance:
                    if self._settling_time == 0:
                        self._settling_time = time - self._step_change_time

                # 计算上升时间（10%-90%）
                progress = (value - self._step_initial_value) / step_size
                if 0.1 <= progress <= 0.9 and self._rise_time == 0:
                    pass  # 需要更复杂的逻辑来准确计算

    def reset(self):
        """重置"""
        self._error_history.clear()
        self._value_history.clear()
        self._time_history.clear()
        self._iae = 0.0
        self._ise = 0.0
        self._itae = 0.0
        self._settling_time = 0.0
        self._rise_time = 0.0
        self._max_overshoot = 0.0
        self._step_change_time = None
        self._step_initial_value = None


class DeviationAnalyzer:
    """偏差分析器"""

    def __init__(self, n_variables: int, names: Optional[List[str]] = None):
        self.n_variables = n_variables
        self.names = names or [f"var_{i}" for i in range(n_variables)]

        # 历史数据
        self._deviation_history: List[np.ndarray] = []
        self._max_history = 10000

        # 统计
        self._running_mean = np.zeros(n_variables)
        self._running_var = np.zeros(n_variables)
        self._count = 0

    def analyze(self, actual: np.ndarray, target: np.ndarray) -> Dict[str, Any]:
        """
        分析偏差

        Args:
            actual: 实际值数组
            target: 目标值数组

        Returns:
            分析结果
        """
        deviation = actual - target

        # 保存历史
        self._deviation_history.append(deviation.copy())
        if len(self._deviation_history) > self._max_history:
            self._deviation_history.pop(0)

        # 更新运行统计
        self._count += 1
        delta = deviation - self._running_mean
        self._running_mean += delta / self._count
        delta2 = deviation - self._running_mean
        self._running_var += delta * delta2

        # 计算统计量
        variance = self._running_var / max(self._count - 1, 1)
        std = np.sqrt(variance)

        # 相对偏差
        relative_deviation = np.zeros_like(deviation)
        for i in range(len(deviation)):
            if abs(target[i]) > 1e-10:
                relative_deviation[i] = deviation[i] / target[i]

        # 趋势分析
        trend = self._analyze_trend()

        # 异常检测
        anomalies = self._detect_anomalies(deviation, std)

        return {
            'deviation': deviation,
            'relative_deviation': relative_deviation,
            'mean_deviation': self._running_mean.copy(),
            'std_deviation': std,
            'max_abs_deviation': np.max(np.abs(deviation)),
            'trend': trend,
            'anomalies': anomalies,
            'variables': {
                self.names[i]: {
                    'deviation': deviation[i],
                    'relative': relative_deviation[i],
                    'mean': self._running_mean[i],
                    'std': std[i]
                }
                for i in range(self.n_variables)
            }
        }

    def _analyze_trend(self) -> Dict[str, Any]:
        """趋势分析"""
        if len(self._deviation_history) < 10:
            return {'direction': 'unknown', 'slope': 0.0}

        recent = np.array(self._deviation_history[-100:])
        mean_recent = np.mean(recent, axis=0)

        # 计算斜率
        x = np.arange(len(recent))
        slopes = np.zeros(self.n_variables)

        for i in range(self.n_variables):
            if np.std(recent[:, i]) > 1e-10:
                slope = np.polyfit(x, recent[:, i], 1)[0]
                slopes[i] = slope

        overall_slope = np.mean(slopes)

        if overall_slope > 0.01:
            direction = 'increasing'
        elif overall_slope < -0.01:
            direction = 'decreasing'
        else:
            direction = 'stable'

        return {
            'direction': direction,
            'slope': overall_slope,
            'slopes_by_variable': slopes.tolist()
        }

    def _detect_anomalies(self, deviation: np.ndarray, std: np.ndarray) -> List[Dict]:
        """异常检测"""
        anomalies = []
        threshold = 3.0  # 3-sigma

        for i in range(self.n_variables):
            if std[i] > 1e-10:
                z_score = abs(deviation[i] - self._running_mean[i]) / std[i]
                if z_score > threshold:
                    anomalies.append({
                        'variable': self.names[i],
                        'z_score': z_score,
                        'deviation': deviation[i],
                        'severity': 'high' if z_score > 5 else 'medium'
                    })

        return anomalies

    def reset(self):
        """重置"""
        self._deviation_history.clear()
        self._running_mean = np.zeros(self.n_variables)
        self._running_var = np.zeros(self.n_variables)
        self._count = 0


class ControlPerformanceEvaluator:
    """控制性能评价器"""

    def __init__(self):
        self.objectives: Dict[str, ControlObjective] = {}
        self.trackers: Dict[str, ObjectiveTracker] = {}

        # 评价历史
        self._evaluation_history: List[EvaluationResult] = []
        self._max_history = 1000

        self._time = 0.0

    def add_objective(self, objective: ControlObjective):
        """添加控制目标"""
        self.objectives[objective.name] = objective
        self.trackers[objective.name] = ObjectiveTracker(objective)

    def remove_objective(self, name: str):
        """移除控制目标"""
        if name in self.objectives:
            del self.objectives[name]
            del self.trackers[name]

    def evaluate(self, actual_values: Dict[str, float], time: float,
                dt: float) -> Dict[str, PerformanceMetrics]:
        """
        评价控制性能

        Args:
            actual_values: 实际值字典
            time: 当前时间
            dt: 时间步长

        Returns:
            各目标的性能指标
        """
        self._time = time
        results = {}

        for name, tracker in self.trackers.items():
            if name in actual_values:
                metrics = tracker.update(actual_values[name], time, dt)
                results[name] = metrics

        return results

    def get_overall_performance(self, metrics: Dict[str, PerformanceMetrics]) -> float:
        """计算整体性能"""
        if not metrics:
            return 1.0

        weighted_sum = 0.0
        total_weight = 0.0

        for name, m in metrics.items():
            if name in self.objectives:
                weight = self.objectives[name].weight
                weighted_sum += m.overall_score * weight
                total_weight += weight

        if total_weight > 0:
            return weighted_sum / total_weight
        return 1.0

    def get_worst_performing(self, metrics: Dict[str, PerformanceMetrics]
                            ) -> Tuple[str, PerformanceMetrics]:
        """获取性能最差的目标"""
        worst_name = None
        worst_metrics = None
        worst_score = float('inf')

        for name, m in metrics.items():
            if m.overall_score < worst_score:
                worst_score = m.overall_score
                worst_name = name
                worst_metrics = m

        return worst_name, worst_metrics

    def reset(self):
        """重置"""
        for tracker in self.trackers.values():
            tracker.reset()
        self._evaluation_history.clear()
        self._time = 0.0


class SafetyEvaluator:
    """安全评价器"""

    def __init__(self):
        # 安全限值
        self.limits: Dict[str, Dict[str, float]] = {}
        self.safety_margins: Dict[str, float] = {}

        # 状态
        self._current_status = SafetyStatus.SAFE
        self._violation_count = 0
        self._last_violation_time = 0.0

    def set_limits(self, variable: str, min_val: float, max_val: float,
                   warning_margin: float = 0.1, alarm_margin: float = 0.05):
        """
        设置安全限值

        Args:
            variable: 变量名
            min_val: 最小值
            max_val: 最大值
            warning_margin: 警告边界（相对于量程）
            alarm_margin: 报警边界
        """
        span = max_val - min_val
        self.limits[variable] = {
            'min': min_val,
            'max': max_val,
            'warning_low': min_val + warning_margin * span,
            'warning_high': max_val - warning_margin * span,
            'alarm_low': min_val + alarm_margin * span,
            'alarm_high': max_val - alarm_margin * span
        }

    def evaluate(self, values: Dict[str, float], time: float) -> Tuple[SafetyStatus, List[str], List[str]]:
        """
        评价安全状态

        Args:
            values: 变量值字典
            time: 当前时间

        Returns:
            (安全状态, 警告列表, 报警列表)
        """
        warnings = []
        alarms = []
        critical = False

        for var, value in values.items():
            if var not in self.limits:
                continue

            limits = self.limits[var]

            # 检查是否超出绝对限值
            if value <= limits['min'] or value >= limits['max']:
                alarms.append(f"{var} exceeded limit: {value:.2f} (limits: {limits['min']:.2f} - {limits['max']:.2f})")
                critical = True
            # 检查报警区
            elif value <= limits['alarm_low'] or value >= limits['alarm_high']:
                alarms.append(f"{var} in alarm zone: {value:.2f}")
            # 检查警告区
            elif value <= limits['warning_low'] or value >= limits['warning_high']:
                warnings.append(f"{var} in warning zone: {value:.2f}")

        # 确定状态
        if critical:
            self._current_status = SafetyStatus.CRITICAL
            self._violation_count += 1
            self._last_violation_time = time
        elif alarms:
            self._current_status = SafetyStatus.ALARM
            self._violation_count += 1
            self._last_violation_time = time
        elif warnings:
            self._current_status = SafetyStatus.WARNING
        else:
            self._current_status = SafetyStatus.SAFE

        return self._current_status, warnings, alarms

    def get_safety_margin(self, variable: str, value: float) -> float:
        """获取安全裕度"""
        if variable not in self.limits:
            return 1.0

        limits = self.limits[variable]
        span = limits['max'] - limits['min']

        # 到最近限值的距离
        dist_to_min = value - limits['min']
        dist_to_max = limits['max'] - value

        margin = min(dist_to_min, dist_to_max) / span
        return max(0, min(1, margin))


class StateEvaluator:
    """系统状态评价器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        # 子评价器
        self.performance_evaluator = ControlPerformanceEvaluator()
        self.safety_evaluator = SafetyEvaluator()
        self.deviation_analyzer: Optional[DeviationAnalyzer] = None

        # 历史
        self._evaluation_history: List[EvaluationResult] = []
        self._max_history = 1000

        # 时间
        self._time = 0.0
        self._lock = threading.Lock()

        # 初始化默认目标和限值
        self._init_defaults()

    def _init_defaults(self):
        """初始化默认配置"""
        # 流量控制目标
        self.performance_evaluator.add_objective(ControlObjective(
            name="flow_rate",
            objective_type=ObjectiveType.SETPOINT,
            target_value=265.0,
            tolerance=0.05,
            weight=1.0,
            priority=1,
            min_value=0.0,
            max_value=400.0
        ))

        # 压力控制目标
        self.performance_evaluator.add_objective(ControlObjective(
            name="pressure",
            objective_type=ObjectiveType.CONSTRAINT,
            target_value=500000.0,
            tolerance=0.1,
            weight=0.8,
            priority=2,
            min_value=-50000.0,
            max_value=1500000.0
        ))

        # 水位控制目标
        self.performance_evaluator.add_objective(ControlObjective(
            name="water_level",
            objective_type=ObjectiveType.TRACKING,
            target_value=6.0,
            tolerance=0.05,
            weight=0.9,
            priority=1
        ))

        # 安全限值
        self.safety_evaluator.set_limits("pressure", -50000, 1500000, 0.15, 0.05)
        self.safety_evaluator.set_limits("flow_rate", 0, 400, 0.1, 0.02)
        self.safety_evaluator.set_limits("water_level", 0, 7, 0.1, 0.02)

        # 偏差分析器
        self.deviation_analyzer = DeviationAnalyzer(
            3, ["flow_rate", "pressure", "water_level"]
        )

    def evaluate(self, system_state: Dict[str, float],
                control_targets: Dict[str, float],
                time: float, dt: float) -> EvaluationResult:
        """
        评价系统状态

        Args:
            system_state: 系统状态
            control_targets: 控制目标
            time: 当前时间
            dt: 时间步长

        Returns:
            评价结果
        """
        self._time = time

        with self._lock:
            # 更新控制目标
            for name, target in control_targets.items():
                if name in self.performance_evaluator.objectives:
                    self.performance_evaluator.objectives[name].target_value = target

            # 控制性能评价
            performance_metrics = self.performance_evaluator.evaluate(
                system_state, time, dt
            )

            # 安全评价
            safety_status, warnings, alarms = self.safety_evaluator.evaluate(
                system_state, time
            )

            # 偏差分析
            if self.deviation_analyzer:
                actual = np.array([
                    system_state.get('flow_rate', 0),
                    system_state.get('pressure', 0),
                    system_state.get('water_level', 0)
                ])
                target = np.array([
                    control_targets.get('flow_rate', 265),
                    control_targets.get('pressure', 500000),
                    control_targets.get('water_level', 6)
                ])
                deviation_analysis = self.deviation_analyzer.analyze(actual, target)

                # 添加偏差相关警告
                for anomaly in deviation_analysis.get('anomalies', []):
                    warnings.append(f"Anomaly in {anomaly['variable']}: z-score={anomaly['z_score']:.2f}")

            # 计算整体得分
            overall_score = self.performance_evaluator.get_overall_performance(
                performance_metrics
            )

            # 安全状态影响得分
            if safety_status == SafetyStatus.CRITICAL:
                overall_score *= 0.0
            elif safety_status == SafetyStatus.ALARM:
                overall_score *= 0.5
            elif safety_status == SafetyStatus.WARNING:
                overall_score *= 0.8

            # 确定整体等级
            if overall_score >= 0.95:
                overall_level = EvaluationLevel.EXCELLENT
            elif overall_score >= 0.80:
                overall_level = EvaluationLevel.GOOD
            elif overall_score >= 0.60:
                overall_level = EvaluationLevel.ACCEPTABLE
            elif overall_score >= 0.40:
                overall_level = EvaluationLevel.MARGINAL
            elif safety_status == SafetyStatus.CRITICAL:
                overall_level = EvaluationLevel.CRITICAL
            else:
                overall_level = EvaluationLevel.POOR

            # 生成建议
            recommendations = self._generate_recommendations(
                performance_metrics, safety_status, system_state, control_targets
            )

            # 构建结果
            result = EvaluationResult(
                timestamp=time,
                objectives_evaluation=performance_metrics,
                safety_status=safety_status,
                overall_score=overall_score,
                overall_level=overall_level,
                warnings=warnings,
                alarms=alarms,
                recommendations=recommendations
            )

            # 保存历史
            self._evaluation_history.append(result)
            if len(self._evaluation_history) > self._max_history:
                self._evaluation_history.pop(0)

            return result

    def _generate_recommendations(self, metrics: Dict[str, PerformanceMetrics],
                                  safety_status: SafetyStatus,
                                  state: Dict[str, float],
                                  targets: Dict[str, float]) -> List[str]:
        """生成建议"""
        recommendations = []

        # 基于性能指标
        for name, m in metrics.items():
            if m.evaluation_level == EvaluationLevel.POOR:
                recommendations.append(f"Consider tuning controller for {name}")

            if m.overshoot > 0.2:
                recommendations.append(f"Reduce controller gain for {name} to decrease overshoot")

            if m.settling_time > 120:
                recommendations.append(f"Increase controller gain for {name} to improve response")

        # 基于安全状态
        if safety_status == SafetyStatus.WARNING:
            recommendations.append("Monitor system closely - approaching safety limits")
        elif safety_status == SafetyStatus.ALARM:
            recommendations.append("Take corrective action - safety limits approached")
        elif safety_status == SafetyStatus.CRITICAL:
            recommendations.append("IMMEDIATE ACTION REQUIRED - safety limits exceeded")

        # 基于偏差
        worst_name, worst_metrics = self.performance_evaluator.get_worst_performing(metrics)
        if worst_metrics and worst_metrics.overall_score < 0.6:
            recommendations.append(f"Priority: Improve {worst_name} control performance")

        return recommendations

    def set_control_objective(self, name: str, objective: ControlObjective):
        """设置控制目标"""
        self.performance_evaluator.add_objective(objective)

    def set_safety_limit(self, variable: str, min_val: float, max_val: float):
        """设置安全限值"""
        self.safety_evaluator.set_limits(variable, min_val, max_val)

    def get_evaluation_history(self, n_samples: Optional[int] = None
                              ) -> List[Dict[str, Any]]:
        """获取评价历史"""
        history = self._evaluation_history[-n_samples:] if n_samples else self._evaluation_history
        return [r.to_dict() for r in history]

    def get_performance_trend(self) -> Dict[str, Any]:
        """获取性能趋势"""
        if not self._evaluation_history:
            return {}

        scores = [r.overall_score for r in self._evaluation_history]
        levels = [r.overall_level.value for r in self._evaluation_history]

        return {
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'min_score': np.min(scores),
            'max_score': np.max(scores),
            'trend': 'improving' if scores[-1] > scores[0] else 'degrading',
            'level_distribution': {
                level: levels.count(level)
                for level in set(levels)
            }
        }

    def reset(self):
        """重置"""
        self._time = 0.0
        self.performance_evaluator.reset()
        if self.deviation_analyzer:
            self.deviation_analyzer.reset()
        self._evaluation_history.clear()
