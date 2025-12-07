"""
Data Analytics Module for CYRP
穿黄工程数据分析模块

实现数据统计分析、趋势分析、相关性分析、异常检测等功能
"""

import asyncio
import math
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import (
    Any, Dict, List, Optional, Callable, Tuple,
    TypeVar, Generic, Union
)
from collections import defaultdict
import logging
import statistics
import json

logger = logging.getLogger(__name__)


# ============================================================
# 枚举定义
# ============================================================

class AggregationType(Enum):
    """聚合类型"""
    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    FIRST = "first"
    LAST = "last"
    MEDIAN = "median"
    STD_DEV = "std_dev"
    VARIANCE = "variance"
    PERCENTILE = "percentile"


class TimeGranularity(Enum):
    """时间粒度"""
    SECOND = "second"
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"


class TrendDirection(Enum):
    """趋势方向"""
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    FLUCTUATING = "fluctuating"


class AnomalyType(Enum):
    """异常类型"""
    SPIKE = "spike"           # 尖峰
    DIP = "dip"               # 低谷
    SHIFT = "shift"           # 漂移
    TREND = "trend"           # 趋势异常
    SEASONAL = "seasonal"     # 季节性异常
    OUTLIER = "outlier"       # 离群点


class AnalysisStatus(Enum):
    """分析状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# ============================================================
# 数据类定义
# ============================================================

@dataclass
class DataPoint:
    """数据点"""
    timestamp: datetime
    value: float
    quality: int = 192  # Good quality
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class TimeSeries:
    """时间序列"""
    series_id: str
    name: str
    unit: str
    data_points: List[DataPoint] = field(default_factory=list)

    def add_point(self, timestamp: datetime, value: float, **kwargs):
        """添加数据点"""
        self.data_points.append(DataPoint(timestamp=timestamp, value=value, **kwargs))

    def get_values(self) -> List[float]:
        """获取值列表"""
        return [dp.value for dp in self.data_points]

    def get_timestamps(self) -> List[datetime]:
        """获取时间戳列表"""
        return [dp.timestamp for dp in self.data_points]

    def slice(self, start: datetime, end: datetime) -> 'TimeSeries':
        """切片"""
        filtered = [dp for dp in self.data_points if start <= dp.timestamp <= end]
        ts = TimeSeries(
            series_id=self.series_id,
            name=self.name,
            unit=self.unit,
            data_points=filtered
        )
        return ts


@dataclass
class StatisticalSummary:
    """统计摘要"""
    count: int
    sum_value: float
    mean: float
    median: float
    mode: Optional[float]
    std_dev: float
    variance: float
    min_value: float
    max_value: float
    range_value: float
    q1: float  # 25th percentile
    q3: float  # 75th percentile
    iqr: float  # Interquartile range
    skewness: float
    kurtosis: float


@dataclass
class TrendAnalysis:
    """趋势分析结果"""
    direction: TrendDirection
    slope: float
    r_squared: float
    confidence: float
    forecast: List[DataPoint] = field(default_factory=list)


@dataclass
class CorrelationResult:
    """相关性分析结果"""
    series_a: str
    series_b: str
    pearson_r: float
    spearman_r: float
    p_value: float
    lag: int = 0
    strength: str = ""  # weak, moderate, strong


@dataclass
class AnomalyDetectionResult:
    """异常检测结果"""
    anomaly_id: str
    series_id: str
    timestamp: datetime
    value: float
    expected_value: float
    anomaly_type: AnomalyType
    severity: float  # 0-1
    confidence: float
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisJob:
    """分析任务"""
    job_id: str
    analysis_type: str
    parameters: Dict[str, Any]
    status: AnalysisStatus = AnalysisStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error_message: Optional[str] = None


# ============================================================
# 统计计算器
# ============================================================

class StatisticsCalculator:
    """统计计算器"""

    @staticmethod
    def calculate_summary(values: List[float]) -> StatisticalSummary:
        """计算统计摘要"""
        if not values:
            raise ValueError("Empty values list")

        n = len(values)
        sorted_values = sorted(values)

        # 基本统计
        sum_value = sum(values)
        mean = sum_value / n
        median = statistics.median(values)

        try:
            mode = statistics.mode(values)
        except statistics.StatisticsError:
            mode = None

        # 标准差和方差
        if n > 1:
            variance = statistics.variance(values)
            std_dev = statistics.stdev(values)
        else:
            variance = 0.0
            std_dev = 0.0

        min_value = min(values)
        max_value = max(values)
        range_value = max_value - min_value

        # 四分位数
        q1 = StatisticsCalculator._percentile(sorted_values, 25)
        q3 = StatisticsCalculator._percentile(sorted_values, 75)
        iqr = q3 - q1

        # 偏度和峰度
        skewness = StatisticsCalculator._skewness(values, mean, std_dev)
        kurtosis = StatisticsCalculator._kurtosis(values, mean, std_dev)

        return StatisticalSummary(
            count=n,
            sum_value=sum_value,
            mean=mean,
            median=median,
            mode=mode,
            std_dev=std_dev,
            variance=variance,
            min_value=min_value,
            max_value=max_value,
            range_value=range_value,
            q1=q1,
            q3=q3,
            iqr=iqr,
            skewness=skewness,
            kurtosis=kurtosis
        )

    @staticmethod
    def _percentile(sorted_values: List[float], p: float) -> float:
        """计算百分位数"""
        n = len(sorted_values)
        k = (n - 1) * p / 100
        f = math.floor(k)
        c = math.ceil(k)

        if f == c:
            return sorted_values[int(k)]

        return sorted_values[int(f)] * (c - k) + sorted_values[int(c)] * (k - f)

    @staticmethod
    def _skewness(values: List[float], mean: float, std_dev: float) -> float:
        """计算偏度"""
        n = len(values)
        if n < 3 or std_dev == 0:
            return 0.0

        m3 = sum((x - mean) ** 3 for x in values) / n
        return m3 / (std_dev ** 3)

    @staticmethod
    def _kurtosis(values: List[float], mean: float, std_dev: float) -> float:
        """计算峰度"""
        n = len(values)
        if n < 4 or std_dev == 0:
            return 0.0

        m4 = sum((x - mean) ** 4 for x in values) / n
        return m4 / (std_dev ** 4) - 3


# ============================================================
# 聚合器
# ============================================================

class TimeSeriesAggregator:
    """时间序列聚合器"""

    def __init__(self):
        self._aggregation_functions = {
            AggregationType.SUM: lambda x: sum(x),
            AggregationType.AVG: lambda x: statistics.mean(x) if x else 0,
            AggregationType.MIN: lambda x: min(x) if x else 0,
            AggregationType.MAX: lambda x: max(x) if x else 0,
            AggregationType.COUNT: lambda x: len(x),
            AggregationType.FIRST: lambda x: x[0] if x else 0,
            AggregationType.LAST: lambda x: x[-1] if x else 0,
            AggregationType.MEDIAN: lambda x: statistics.median(x) if x else 0,
            AggregationType.STD_DEV: lambda x: statistics.stdev(x) if len(x) > 1 else 0,
            AggregationType.VARIANCE: lambda x: statistics.variance(x) if len(x) > 1 else 0,
        }

    def aggregate(
        self,
        series: TimeSeries,
        granularity: TimeGranularity,
        aggregation: AggregationType
    ) -> TimeSeries:
        """聚合时间序列"""
        if not series.data_points:
            return TimeSeries(
                series_id=f"{series.series_id}_agg",
                name=f"{series.name} ({aggregation.value})",
                unit=series.unit
            )

        # 按时间粒度分组
        buckets = self._bucket_data(series.data_points, granularity)

        # 聚合每个桶
        agg_func = self._aggregation_functions.get(aggregation)
        if not agg_func:
            raise ValueError(f"Unsupported aggregation: {aggregation}")

        result = TimeSeries(
            series_id=f"{series.series_id}_agg",
            name=f"{series.name} ({aggregation.value})",
            unit=series.unit
        )

        for bucket_time, points in sorted(buckets.items()):
            values = [p.value for p in points]
            agg_value = agg_func(values)
            result.add_point(bucket_time, agg_value)

        return result

    def _bucket_data(
        self,
        data_points: List[DataPoint],
        granularity: TimeGranularity
    ) -> Dict[datetime, List[DataPoint]]:
        """按时间粒度分桶"""
        buckets = defaultdict(list)

        for point in data_points:
            bucket_time = self._truncate_time(point.timestamp, granularity)
            buckets[bucket_time].append(point)

        return buckets

    def _truncate_time(
        self,
        dt: datetime,
        granularity: TimeGranularity
    ) -> datetime:
        """截断时间到指定粒度"""
        if granularity == TimeGranularity.SECOND:
            return dt.replace(microsecond=0)
        elif granularity == TimeGranularity.MINUTE:
            return dt.replace(second=0, microsecond=0)
        elif granularity == TimeGranularity.HOUR:
            return dt.replace(minute=0, second=0, microsecond=0)
        elif granularity == TimeGranularity.DAY:
            return dt.replace(hour=0, minute=0, second=0, microsecond=0)
        elif granularity == TimeGranularity.WEEK:
            start_of_week = dt - timedelta(days=dt.weekday())
            return start_of_week.replace(hour=0, minute=0, second=0, microsecond=0)
        elif granularity == TimeGranularity.MONTH:
            return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        elif granularity == TimeGranularity.YEAR:
            return dt.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        return dt


# ============================================================
# 趋势分析器
# ============================================================

class TrendAnalyzer:
    """趋势分析器"""

    def analyze(
        self,
        series: TimeSeries,
        forecast_periods: int = 10
    ) -> TrendAnalysis:
        """分析趋势"""
        if len(series.data_points) < 2:
            return TrendAnalysis(
                direction=TrendDirection.STABLE,
                slope=0.0,
                r_squared=0.0,
                confidence=0.0
            )

        values = series.get_values()
        n = len(values)

        # 线性回归
        x = list(range(n))
        slope, intercept, r_squared = self._linear_regression(x, values)

        # 判断趋势方向
        direction = self._determine_direction(slope, values)

        # 计算置信度
        confidence = r_squared

        # 生成预测
        forecast = []
        last_timestamp = series.data_points[-1].timestamp
        interval = self._estimate_interval(series.data_points)

        for i in range(1, forecast_periods + 1):
            pred_x = n + i - 1
            pred_value = slope * pred_x + intercept
            pred_timestamp = last_timestamp + interval * i
            forecast.append(DataPoint(timestamp=pred_timestamp, value=pred_value))

        return TrendAnalysis(
            direction=direction,
            slope=slope,
            r_squared=r_squared,
            confidence=confidence,
            forecast=forecast
        )

    def _linear_regression(
        self,
        x: List[float],
        y: List[float]
    ) -> Tuple[float, float, float]:
        """线性回归"""
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_x2 = sum(xi ** 2 for xi in x)
        sum_y2 = sum(yi ** 2 for yi in y)

        # 斜率
        denominator = n * sum_x2 - sum_x ** 2
        if denominator == 0:
            return 0.0, statistics.mean(y), 0.0

        slope = (n * sum_xy - sum_x * sum_y) / denominator
        intercept = (sum_y - slope * sum_x) / n

        # R-squared
        y_mean = sum_y / n
        ss_tot = sum((yi - y_mean) ** 2 for yi in y)
        ss_res = sum((yi - (slope * xi + intercept)) ** 2 for xi, yi in zip(x, y))

        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        return slope, intercept, r_squared

    def _determine_direction(
        self,
        slope: float,
        values: List[float]
    ) -> TrendDirection:
        """判断趋势方向"""
        if len(values) < 2:
            return TrendDirection.STABLE

        # 计算变异系数
        mean = statistics.mean(values)
        if mean == 0:
            return TrendDirection.STABLE

        std_dev = statistics.stdev(values) if len(values) > 1 else 0
        cv = std_dev / abs(mean)

        # 阈值判断
        slope_threshold = 0.01 * mean if mean != 0 else 0.01

        if abs(slope) < slope_threshold:
            if cv > 0.3:
                return TrendDirection.FLUCTUATING
            return TrendDirection.STABLE
        elif slope > 0:
            return TrendDirection.INCREASING
        else:
            return TrendDirection.DECREASING

    def _estimate_interval(
        self,
        data_points: List[DataPoint]
    ) -> timedelta:
        """估计时间间隔"""
        if len(data_points) < 2:
            return timedelta(hours=1)

        intervals = []
        for i in range(1, len(data_points)):
            diff = data_points[i].timestamp - data_points[i-1].timestamp
            intervals.append(diff)

        # 返回中位数间隔
        sorted_intervals = sorted(intervals)
        mid = len(sorted_intervals) // 2
        return sorted_intervals[mid]


# ============================================================
# 相关性分析器
# ============================================================

class CorrelationAnalyzer:
    """相关性分析器"""

    def analyze(
        self,
        series_a: TimeSeries,
        series_b: TimeSeries,
        max_lag: int = 10
    ) -> CorrelationResult:
        """分析两个序列的相关性"""
        # 对齐时间序列
        aligned_a, aligned_b = self._align_series(series_a, series_b)

        if len(aligned_a) < 3:
            return CorrelationResult(
                series_a=series_a.series_id,
                series_b=series_b.series_id,
                pearson_r=0.0,
                spearman_r=0.0,
                p_value=1.0,
                strength="none"
            )

        # 计算Pearson相关系数
        pearson_r = self._pearson_correlation(aligned_a, aligned_b)

        # 计算Spearman相关系数
        spearman_r = self._spearman_correlation(aligned_a, aligned_b)

        # 计算p值（简化计算）
        n = len(aligned_a)
        t_stat = pearson_r * math.sqrt(n - 2) / math.sqrt(1 - pearson_r ** 2) \
            if abs(pearson_r) < 1 else 0
        p_value = self._approximate_p_value(t_stat, n - 2)

        # 判断相关强度
        strength = self._determine_strength(abs(pearson_r))

        # 计算滞后相关性
        best_lag, best_corr = self._lag_correlation(aligned_a, aligned_b, max_lag)

        return CorrelationResult(
            series_a=series_a.series_id,
            series_b=series_b.series_id,
            pearson_r=pearson_r,
            spearman_r=spearman_r,
            p_value=p_value,
            lag=best_lag,
            strength=strength
        )

    def _align_series(
        self,
        series_a: TimeSeries,
        series_b: TimeSeries
    ) -> Tuple[List[float], List[float]]:
        """对齐时间序列"""
        # 简单实现：假设相同索引对应
        min_len = min(len(series_a.data_points), len(series_b.data_points))
        values_a = [series_a.data_points[i].value for i in range(min_len)]
        values_b = [series_b.data_points[i].value for i in range(min_len)]
        return values_a, values_b

    def _pearson_correlation(
        self,
        x: List[float],
        y: List[float]
    ) -> float:
        """计算Pearson相关系数"""
        n = len(x)
        if n == 0:
            return 0.0

        mean_x = sum(x) / n
        mean_y = sum(y) / n

        numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        denominator_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x))
        denominator_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y))

        denominator = denominator_x * denominator_y
        if denominator == 0:
            return 0.0

        return numerator / denominator

    def _spearman_correlation(
        self,
        x: List[float],
        y: List[float]
    ) -> float:
        """计算Spearman相关系数"""
        # 转换为排名
        rank_x = self._rank(x)
        rank_y = self._rank(y)
        return self._pearson_correlation(rank_x, rank_y)

    def _rank(self, values: List[float]) -> List[float]:
        """计算排名"""
        sorted_indices = sorted(range(len(values)), key=lambda i: values[i])
        ranks = [0.0] * len(values)
        for rank, idx in enumerate(sorted_indices, 1):
            ranks[idx] = float(rank)
        return ranks

    def _approximate_p_value(self, t_stat: float, df: int) -> float:
        """近似p值计算"""
        if df <= 0:
            return 1.0
        # 简化计算，使用正态近似
        return 2.0 * (1.0 - self._norm_cdf(abs(t_stat)))

    def _norm_cdf(self, x: float) -> float:
        """标准正态分布CDF近似"""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    def _determine_strength(self, r: float) -> str:
        """判断相关强度"""
        if r < 0.1:
            return "none"
        elif r < 0.3:
            return "weak"
        elif r < 0.5:
            return "moderate"
        elif r < 0.7:
            return "strong"
        else:
            return "very_strong"

    def _lag_correlation(
        self,
        x: List[float],
        y: List[float],
        max_lag: int
    ) -> Tuple[int, float]:
        """计算滞后相关性"""
        best_lag = 0
        best_corr = self._pearson_correlation(x, y)

        for lag in range(1, min(max_lag + 1, len(x))):
            # 正滞后
            corr_pos = self._pearson_correlation(x[:-lag], y[lag:])
            if abs(corr_pos) > abs(best_corr):
                best_corr = corr_pos
                best_lag = lag

            # 负滞后
            corr_neg = self._pearson_correlation(x[lag:], y[:-lag])
            if abs(corr_neg) > abs(best_corr):
                best_corr = corr_neg
                best_lag = -lag

        return best_lag, best_corr


# ============================================================
# 异常检测器
# ============================================================

class AnomalyDetector:
    """异常检测器"""

    def __init__(
        self,
        z_threshold: float = 3.0,
        iqr_multiplier: float = 1.5
    ):
        self.z_threshold = z_threshold
        self.iqr_multiplier = iqr_multiplier

    def detect(
        self,
        series: TimeSeries,
        method: str = "zscore"
    ) -> List[AnomalyDetectionResult]:
        """检测异常"""
        if len(series.data_points) < 3:
            return []

        if method == "zscore":
            return self._detect_zscore(series)
        elif method == "iqr":
            return self._detect_iqr(series)
        elif method == "moving_avg":
            return self._detect_moving_avg(series)
        elif method == "isolation":
            return self._detect_isolation_forest(series)
        else:
            return self._detect_zscore(series)

    def _detect_zscore(self, series: TimeSeries) -> List[AnomalyDetectionResult]:
        """Z-score异常检测"""
        values = series.get_values()
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 1

        anomalies = []
        for dp in series.data_points:
            if std == 0:
                continue

            z_score = (dp.value - mean) / std

            if abs(z_score) > self.z_threshold:
                anomaly_type = AnomalyType.SPIKE if z_score > 0 else AnomalyType.DIP
                severity = min(abs(z_score) / 10, 1.0)

                anomalies.append(AnomalyDetectionResult(
                    anomaly_id=str(uuid.uuid4()),
                    series_id=series.series_id,
                    timestamp=dp.timestamp,
                    value=dp.value,
                    expected_value=mean,
                    anomaly_type=anomaly_type,
                    severity=severity,
                    confidence=0.95,
                    context={'z_score': z_score, 'threshold': self.z_threshold}
                ))

        return anomalies

    def _detect_iqr(self, series: TimeSeries) -> List[AnomalyDetectionResult]:
        """IQR异常检测"""
        values = series.get_values()
        sorted_values = sorted(values)
        n = len(sorted_values)

        q1 = sorted_values[n // 4]
        q3 = sorted_values[3 * n // 4]
        iqr = q3 - q1

        lower_bound = q1 - self.iqr_multiplier * iqr
        upper_bound = q3 + self.iqr_multiplier * iqr

        anomalies = []
        median = statistics.median(values)

        for dp in series.data_points:
            if dp.value < lower_bound or dp.value > upper_bound:
                anomaly_type = AnomalyType.SPIKE if dp.value > upper_bound else AnomalyType.DIP
                deviation = abs(dp.value - median)
                severity = min(deviation / (iqr + 1e-6), 1.0)

                anomalies.append(AnomalyDetectionResult(
                    anomaly_id=str(uuid.uuid4()),
                    series_id=series.series_id,
                    timestamp=dp.timestamp,
                    value=dp.value,
                    expected_value=median,
                    anomaly_type=anomaly_type,
                    severity=severity,
                    confidence=0.90,
                    context={
                        'lower_bound': lower_bound,
                        'upper_bound': upper_bound,
                        'iqr': iqr
                    }
                ))

        return anomalies

    def _detect_moving_avg(
        self,
        series: TimeSeries,
        window: int = 10,
        threshold: float = 2.0
    ) -> List[AnomalyDetectionResult]:
        """移动平均异常检测"""
        values = series.get_values()
        if len(values) < window:
            return []

        anomalies = []

        for i in range(window, len(values)):
            window_values = values[i-window:i]
            moving_avg = statistics.mean(window_values)
            moving_std = statistics.stdev(window_values) if len(window_values) > 1 else 1

            current = values[i]
            if moving_std == 0:
                continue

            deviation = abs(current - moving_avg) / moving_std

            if deviation > threshold:
                dp = series.data_points[i]
                anomaly_type = AnomalyType.SPIKE if current > moving_avg else AnomalyType.DIP

                anomalies.append(AnomalyDetectionResult(
                    anomaly_id=str(uuid.uuid4()),
                    series_id=series.series_id,
                    timestamp=dp.timestamp,
                    value=dp.value,
                    expected_value=moving_avg,
                    anomaly_type=anomaly_type,
                    severity=min(deviation / 10, 1.0),
                    confidence=0.85,
                    context={
                        'window': window,
                        'moving_avg': moving_avg,
                        'moving_std': moving_std
                    }
                ))

        return anomalies

    def _detect_isolation_forest(
        self,
        series: TimeSeries,
        contamination: float = 0.1
    ) -> List[AnomalyDetectionResult]:
        """简化的Isolation Forest异常检测"""
        values = series.get_values()
        n = len(values)

        # 计算每个点的异常分数
        scores = []
        for i, val in enumerate(values):
            # 简化计算：使用与其他点的平均距离
            distances = [abs(val - v) for j, v in enumerate(values) if j != i]
            avg_dist = sum(distances) / len(distances) if distances else 0
            scores.append(avg_dist)

        # 确定阈值（取最高的contamination比例）
        sorted_scores = sorted(scores, reverse=True)
        threshold_idx = int(n * contamination)
        threshold = sorted_scores[threshold_idx] if threshold_idx < len(sorted_scores) else sorted_scores[-1]

        anomalies = []
        median = statistics.median(values)

        for i, (score, dp) in enumerate(zip(scores, series.data_points)):
            if score >= threshold:
                anomaly_type = AnomalyType.OUTLIER

                anomalies.append(AnomalyDetectionResult(
                    anomaly_id=str(uuid.uuid4()),
                    series_id=series.series_id,
                    timestamp=dp.timestamp,
                    value=dp.value,
                    expected_value=median,
                    anomaly_type=anomaly_type,
                    severity=min(score / (threshold + 1e-6) * 0.5, 1.0),
                    confidence=0.80,
                    context={'isolation_score': score, 'threshold': threshold}
                ))

        return anomalies


# ============================================================
# 数据分析服务
# ============================================================

class DataAnalyticsService:
    """数据分析服务"""

    def __init__(self):
        self.statistics_calculator = StatisticsCalculator()
        self.aggregator = TimeSeriesAggregator()
        self.trend_analyzer = TrendAnalyzer()
        self.correlation_analyzer = CorrelationAnalyzer()
        self.anomaly_detector = AnomalyDetector()

        # 任务管理
        self.jobs: Dict[str, AnalysisJob] = {}
        self._job_queue: asyncio.Queue = asyncio.Queue()
        self._running = False

    async def start(self):
        """启动服务"""
        self._running = True
        asyncio.create_task(self._process_jobs())
        logger.info("Data analytics service started")

    async def stop(self):
        """停止服务"""
        self._running = False
        logger.info("Data analytics service stopped")

    async def calculate_statistics(
        self,
        series: TimeSeries
    ) -> StatisticalSummary:
        """计算统计摘要"""
        values = series.get_values()
        return self.statistics_calculator.calculate_summary(values)

    async def aggregate(
        self,
        series: TimeSeries,
        granularity: TimeGranularity,
        aggregation: AggregationType
    ) -> TimeSeries:
        """聚合时间序列"""
        return self.aggregator.aggregate(series, granularity, aggregation)

    async def analyze_trend(
        self,
        series: TimeSeries,
        forecast_periods: int = 10
    ) -> TrendAnalysis:
        """分析趋势"""
        return self.trend_analyzer.analyze(series, forecast_periods)

    async def analyze_correlation(
        self,
        series_a: TimeSeries,
        series_b: TimeSeries,
        max_lag: int = 10
    ) -> CorrelationResult:
        """分析相关性"""
        return self.correlation_analyzer.analyze(series_a, series_b, max_lag)

    async def detect_anomalies(
        self,
        series: TimeSeries,
        method: str = "zscore"
    ) -> List[AnomalyDetectionResult]:
        """检测异常"""
        return self.anomaly_detector.detect(series, method)

    async def submit_job(
        self,
        analysis_type: str,
        parameters: Dict[str, Any]
    ) -> str:
        """提交分析任务"""
        job = AnalysisJob(
            job_id=str(uuid.uuid4()),
            analysis_type=analysis_type,
            parameters=parameters
        )
        self.jobs[job.job_id] = job
        await self._job_queue.put(job)

        logger.info(f"Analysis job submitted: {job.job_id}")
        return job.job_id

    async def get_job_status(self, job_id: str) -> Optional[AnalysisJob]:
        """获取任务状态"""
        return self.jobs.get(job_id)

    async def _process_jobs(self):
        """处理任务队列"""
        while self._running:
            try:
                job = await asyncio.wait_for(
                    self._job_queue.get(),
                    timeout=1.0
                )

                job.status = AnalysisStatus.RUNNING
                job.started_at = datetime.now()

                try:
                    result = await self._execute_job(job)
                    job.result = result
                    job.status = AnalysisStatus.COMPLETED
                except Exception as e:
                    job.status = AnalysisStatus.FAILED
                    job.error_message = str(e)
                    logger.error(f"Job {job.job_id} failed: {e}")

                job.completed_at = datetime.now()

            except asyncio.TimeoutError:
                continue

    async def _execute_job(self, job: AnalysisJob) -> Any:
        """执行任务"""
        params = job.parameters

        if job.analysis_type == "statistics":
            series = params.get('series')
            return await self.calculate_statistics(series)

        elif job.analysis_type == "aggregation":
            series = params.get('series')
            granularity = params.get('granularity')
            aggregation = params.get('aggregation')
            return await self.aggregate(series, granularity, aggregation)

        elif job.analysis_type == "trend":
            series = params.get('series')
            forecast = params.get('forecast_periods', 10)
            return await self.analyze_trend(series, forecast)

        elif job.analysis_type == "correlation":
            series_a = params.get('series_a')
            series_b = params.get('series_b')
            max_lag = params.get('max_lag', 10)
            return await self.analyze_correlation(series_a, series_b, max_lag)

        elif job.analysis_type == "anomaly":
            series = params.get('series')
            method = params.get('method', 'zscore')
            return await self.detect_anomalies(series, method)

        else:
            raise ValueError(f"Unknown analysis type: {job.analysis_type}")


# ============================================================
# 工厂函数
# ============================================================

def create_cyrp_analytics_service() -> DataAnalyticsService:
    """创建CYRP数据分析服务实例

    Returns:
        DataAnalyticsService: 数据分析服务实例
    """
    return DataAnalyticsService()


# ============================================================
# 示例用法
# ============================================================

async def example_usage():
    """示例用法"""
    # 创建服务
    service = create_cyrp_analytics_service()
    await service.start()

    # 创建测试数据
    import random
    series = TimeSeries(
        series_id="flow_rate",
        name="流量",
        unit="m³/s"
    )

    base_time = datetime.now() - timedelta(hours=24)
    for i in range(1440):  # 24小时，每分钟一个点
        timestamp = base_time + timedelta(minutes=i)
        value = 100 + 20 * math.sin(i * 0.01) + random.gauss(0, 5)
        # 添加一些异常点
        if i in [200, 500, 800]:
            value += 50
        series.add_point(timestamp, value)

    # 统计分析
    stats = await service.calculate_statistics(series)
    print(f"统计摘要:")
    print(f"  数量: {stats.count}")
    print(f"  均值: {stats.mean:.2f}")
    print(f"  标准差: {stats.std_dev:.2f}")
    print(f"  最小值: {stats.min_value:.2f}")
    print(f"  最大值: {stats.max_value:.2f}")

    # 聚合
    hourly = await service.aggregate(
        series,
        TimeGranularity.HOUR,
        AggregationType.AVG
    )
    print(f"\n小时聚合点数: {len(hourly.data_points)}")

    # 趋势分析
    trend = await service.analyze_trend(series)
    print(f"\n趋势分析:")
    print(f"  方向: {trend.direction.value}")
    print(f"  斜率: {trend.slope:.4f}")
    print(f"  R²: {trend.r_squared:.4f}")

    # 异常检测
    anomalies = await service.detect_anomalies(series, method="zscore")
    print(f"\n检测到 {len(anomalies)} 个异常")
    for anomaly in anomalies[:3]:
        print(f"  {anomaly.timestamp}: 值={anomaly.value:.2f}, 类型={anomaly.anomaly_type.value}")

    await service.stop()


if __name__ == "__main__":
    asyncio.run(example_usage())
