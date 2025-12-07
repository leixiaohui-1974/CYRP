"""
Data Visualization and Trending Module for CYRP
穿黄工程数据可视化与趋势分析模块

功能:
- 实时数据趋势显示
- 历史数据回放
- 多变量对比分析
- 统计分析与报表
"""

import asyncio
import json
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import math
import statistics


class TrendType(Enum):
    """趋势图类型"""
    LINE = auto()          # 折线图
    AREA = auto()          # 面积图
    BAR = auto()           # 柱状图
    SCATTER = auto()       # 散点图
    STEP = auto()          # 阶梯图
    CANDLESTICK = auto()   # K线图


class AggregationType(Enum):
    """数据聚合类型"""
    RAW = auto()           # 原始数据
    AVERAGE = auto()       # 平均值
    MIN = auto()           # 最小值
    MAX = auto()           # 最大值
    SUM = auto()           # 求和
    COUNT = auto()         # 计数
    FIRST = auto()         # 第一个值
    LAST = auto()          # 最后一个值
    RANGE = auto()         # 范围(最大-最小)
    STDDEV = auto()        # 标准差


class PlaybackState(Enum):
    """回放状态"""
    STOPPED = auto()
    PLAYING = auto()
    PAUSED = auto()
    FAST_FORWARD = auto()
    REWIND = auto()


@dataclass
class DataPoint:
    """数据点"""
    timestamp: datetime
    value: float
    quality: int = 192  # OPC质量码, 192=Good

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "value": self.value,
            "quality": self.quality
        }


@dataclass
class TrendConfig:
    """趋势配置"""
    tag_name: str
    display_name: str
    unit: str = ""
    color: str = "#1f77b4"
    line_width: int = 2
    trend_type: TrendType = TrendType.LINE
    y_axis_id: int = 0
    visible: bool = True
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    decimals: int = 2


@dataclass
class YAxisConfig:
    """Y轴配置"""
    axis_id: int
    label: str
    position: str = "left"  # left or right
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    auto_scale: bool = True
    color: str = "#333333"
    grid_lines: bool = True


@dataclass
class TrendPanelConfig:
    """趋势面板配置"""
    panel_id: str
    title: str
    trends: List[TrendConfig] = field(default_factory=list)
    y_axes: List[YAxisConfig] = field(default_factory=list)
    time_range_minutes: int = 60
    refresh_interval_ms: int = 1000
    show_legend: bool = True
    show_toolbar: bool = True
    background_color: str = "#ffffff"
    grid_color: str = "#e0e0e0"


class TrendDataBuffer:
    """趋势数据缓冲区"""

    def __init__(self, max_points: int = 10000):
        self.max_points = max_points
        self._buffers: Dict[str, deque] = {}
        self._lock = asyncio.Lock()

    async def add_point(self, tag_name: str, point: DataPoint):
        """添加数据点"""
        async with self._lock:
            if tag_name not in self._buffers:
                self._buffers[tag_name] = deque(maxlen=self.max_points)
            self._buffers[tag_name].append(point)

    async def add_points(self, tag_name: str, points: List[DataPoint]):
        """批量添加数据点"""
        async with self._lock:
            if tag_name not in self._buffers:
                self._buffers[tag_name] = deque(maxlen=self.max_points)
            self._buffers[tag_name].extend(points)

    async def get_points(
        self,
        tag_name: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[DataPoint]:
        """获取数据点"""
        async with self._lock:
            if tag_name not in self._buffers:
                return []

            points = list(self._buffers[tag_name])

            if start_time:
                points = [p for p in points if p.timestamp >= start_time]
            if end_time:
                points = [p for p in points if p.timestamp <= end_time]

            return points

    async def get_latest(self, tag_name: str) -> Optional[DataPoint]:
        """获取最新数据点"""
        async with self._lock:
            if tag_name not in self._buffers or not self._buffers[tag_name]:
                return None
            return self._buffers[tag_name][-1]

    async def clear(self, tag_name: Optional[str] = None):
        """清除缓冲区"""
        async with self._lock:
            if tag_name:
                if tag_name in self._buffers:
                    self._buffers[tag_name].clear()
            else:
                self._buffers.clear()


class DataAggregator:
    """数据聚合器"""

    @staticmethod
    def aggregate(
        points: List[DataPoint],
        interval_seconds: int,
        aggregation_type: AggregationType
    ) -> List[DataPoint]:
        """按时间间隔聚合数据"""
        if not points:
            return []

        if aggregation_type == AggregationType.RAW:
            return points

        # 按时间间隔分组
        buckets: Dict[datetime, List[float]] = {}

        for point in points:
            # 计算桶的起始时间
            bucket_ts = datetime.fromtimestamp(
                (point.timestamp.timestamp() // interval_seconds) * interval_seconds
            )
            if bucket_ts not in buckets:
                buckets[bucket_ts] = []
            buckets[bucket_ts].append(point.value)

        # 计算聚合值
        result = []
        for bucket_ts in sorted(buckets.keys()):
            values = buckets[bucket_ts]
            agg_value = DataAggregator._calculate_aggregation(values, aggregation_type)
            result.append(DataPoint(timestamp=bucket_ts, value=agg_value))

        return result

    @staticmethod
    def _calculate_aggregation(
        values: List[float],
        aggregation_type: AggregationType
    ) -> float:
        """计算聚合值"""
        if not values:
            return 0.0

        if aggregation_type == AggregationType.AVERAGE:
            return statistics.mean(values)
        elif aggregation_type == AggregationType.MIN:
            return min(values)
        elif aggregation_type == AggregationType.MAX:
            return max(values)
        elif aggregation_type == AggregationType.SUM:
            return sum(values)
        elif aggregation_type == AggregationType.COUNT:
            return float(len(values))
        elif aggregation_type == AggregationType.FIRST:
            return values[0]
        elif aggregation_type == AggregationType.LAST:
            return values[-1]
        elif aggregation_type == AggregationType.RANGE:
            return max(values) - min(values)
        elif aggregation_type == AggregationType.STDDEV:
            return statistics.stdev(values) if len(values) > 1 else 0.0
        else:
            return statistics.mean(values)


class StatisticalAnalyzer:
    """统计分析器"""

    @staticmethod
    def calculate_statistics(points: List[DataPoint]) -> Dict[str, float]:
        """计算统计信息"""
        if not points:
            return {}

        values = [p.value for p in points]

        stats = {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "sum": sum(values),
            "range": max(values) - min(values),
        }

        if len(values) > 1:
            stats["stddev"] = statistics.stdev(values)
            stats["variance"] = statistics.variance(values)
            stats["median"] = statistics.median(values)

        # 计算变化率
        if len(points) >= 2:
            time_diff = (points[-1].timestamp - points[0].timestamp).total_seconds()
            if time_diff > 0:
                stats["rate_of_change"] = (values[-1] - values[0]) / time_diff

        return stats

    @staticmethod
    def detect_anomalies(
        points: List[DataPoint],
        threshold_sigma: float = 3.0
    ) -> List[Tuple[DataPoint, str]]:
        """检测异常值"""
        if len(points) < 10:
            return []

        values = [p.value for p in points]
        mean = statistics.mean(values)
        stddev = statistics.stdev(values)

        anomalies = []
        for point in points:
            z_score = abs(point.value - mean) / stddev if stddev > 0 else 0
            if z_score > threshold_sigma:
                reason = f"Z-score={z_score:.2f} > {threshold_sigma}"
                anomalies.append((point, reason))

        return anomalies

    @staticmethod
    def calculate_correlation(
        points1: List[DataPoint],
        points2: List[DataPoint]
    ) -> float:
        """计算两个变量的相关系数"""
        # 对齐时间戳
        timestamps1 = {p.timestamp: p.value for p in points1}
        timestamps2 = {p.timestamp: p.value for p in points2}

        common_ts = set(timestamps1.keys()) & set(timestamps2.keys())
        if len(common_ts) < 2:
            return 0.0

        values1 = [timestamps1[ts] for ts in sorted(common_ts)]
        values2 = [timestamps2[ts] for ts in sorted(common_ts)]

        # 计算皮尔逊相关系数
        n = len(values1)
        mean1 = statistics.mean(values1)
        mean2 = statistics.mean(values2)

        numerator = sum((v1 - mean1) * (v2 - mean2) for v1, v2 in zip(values1, values2))
        denominator1 = math.sqrt(sum((v - mean1) ** 2 for v in values1))
        denominator2 = math.sqrt(sum((v - mean2) ** 2 for v in values2))

        if denominator1 * denominator2 == 0:
            return 0.0

        return numerator / (denominator1 * denominator2)


class HistoricalPlayback:
    """历史数据回放器"""

    def __init__(
        self,
        data_source: Callable[[str, datetime, datetime], List[DataPoint]]
    ):
        self.data_source = data_source
        self.state = PlaybackState.STOPPED
        self.speed = 1.0
        self.current_time: Optional[datetime] = None
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self._playback_task: Optional[asyncio.Task] = None
        self._callbacks: List[Callable[[datetime, Dict[str, DataPoint]], None]] = []
        self._cached_data: Dict[str, List[DataPoint]] = {}
        self._tag_names: List[str] = []

    def add_callback(
        self,
        callback: Callable[[datetime, Dict[str, DataPoint]], None]
    ):
        """添加回放回调"""
        self._callbacks.append(callback)

    async def load_data(
        self,
        tag_names: List[str],
        start_time: datetime,
        end_time: datetime
    ):
        """加载历史数据"""
        self._tag_names = tag_names
        self.start_time = start_time
        self.end_time = end_time
        self.current_time = start_time

        self._cached_data.clear()
        for tag_name in tag_names:
            points = self.data_source(tag_name, start_time, end_time)
            self._cached_data[tag_name] = sorted(points, key=lambda p: p.timestamp)

    async def play(self):
        """开始回放"""
        if self.state == PlaybackState.PLAYING:
            return

        self.state = PlaybackState.PLAYING
        self._playback_task = asyncio.create_task(self._playback_loop())

    async def pause(self):
        """暂停回放"""
        self.state = PlaybackState.PAUSED

    async def stop(self):
        """停止回放"""
        self.state = PlaybackState.STOPPED
        if self._playback_task:
            self._playback_task.cancel()
            try:
                await self._playback_task
            except asyncio.CancelledError:
                pass
        self.current_time = self.start_time

    async def seek(self, target_time: datetime):
        """跳转到指定时间"""
        if self.start_time and self.end_time:
            if self.start_time <= target_time <= self.end_time:
                self.current_time = target_time

    def set_speed(self, speed: float):
        """设置回放速度"""
        self.speed = max(0.1, min(100.0, speed))

    async def _playback_loop(self):
        """回放循环"""
        interval = 0.1  # 100ms更新间隔

        while self.state == PlaybackState.PLAYING:
            if self.current_time and self.end_time:
                if self.current_time >= self.end_time:
                    self.state = PlaybackState.STOPPED
                    break

                # 获取当前时间点的数据
                current_data = {}
                for tag_name, points in self._cached_data.items():
                    # 找到最接近当前时间的数据点
                    for point in points:
                        if point.timestamp <= self.current_time:
                            current_data[tag_name] = point
                        else:
                            break

                # 调用回调
                for callback in self._callbacks:
                    try:
                        callback(self.current_time, current_data)
                    except Exception:
                        pass

                # 更新时间
                time_increment = timedelta(seconds=interval * self.speed)
                self.current_time += time_increment

            await asyncio.sleep(interval)


class TrendRenderer(ABC):
    """趋势渲染器基类"""

    @abstractmethod
    def render(
        self,
        config: TrendPanelConfig,
        data: Dict[str, List[DataPoint]]
    ) -> str:
        """渲染趋势图"""
        pass


class ASCIITrendRenderer(TrendRenderer):
    """ASCII趋势渲染器(用于终端显示)"""

    def __init__(self, width: int = 80, height: int = 20):
        self.width = width
        self.height = height

    def render(
        self,
        config: TrendPanelConfig,
        data: Dict[str, List[DataPoint]]
    ) -> str:
        """渲染ASCII趋势图"""
        lines = []
        lines.append(f"╔{'═' * (self.width - 2)}╗")
        lines.append(f"║ {config.title:^{self.width - 4}} ║")
        lines.append(f"╠{'═' * (self.width - 2)}╣")

        for trend_config in config.trends:
            if trend_config.tag_name in data:
                points = data[trend_config.tag_name]
                if points:
                    chart = self._render_sparkline(points, self.width - 4)
                    label = f"{trend_config.display_name}: "
                    lines.append(f"║ {label}{chart[:self.width - len(label) - 4]} ║")

        lines.append(f"╚{'═' * (self.width - 2)}╝")
        return "\n".join(lines)

    def _render_sparkline(self, points: List[DataPoint], width: int) -> str:
        """渲染迷你趋势线"""
        if not points:
            return ""

        values = [p.value for p in points[-width:]]
        min_val = min(values)
        max_val = max(values)
        range_val = max_val - min_val if max_val != min_val else 1

        chars = "▁▂▃▄▅▆▇█"
        result = ""
        for val in values:
            idx = int((val - min_val) / range_val * (len(chars) - 1))
            result += chars[idx]

        return result


class HTMLTrendRenderer(TrendRenderer):
    """HTML趋势渲染器(用于Web显示)"""

    def render(
        self,
        config: TrendPanelConfig,
        data: Dict[str, List[DataPoint]]
    ) -> str:
        """渲染HTML趋势图(使用Chart.js)"""
        datasets = []
        for trend_config in config.trends:
            if trend_config.tag_name in data:
                points = data[trend_config.tag_name]
                dataset = {
                    "label": trend_config.display_name,
                    "data": [
                        {"x": p.timestamp.isoformat(), "y": p.value}
                        for p in points
                    ],
                    "borderColor": trend_config.color,
                    "borderWidth": trend_config.line_width,
                    "fill": trend_config.trend_type == TrendType.AREA,
                    "stepped": trend_config.trend_type == TrendType.STEP,
                    "yAxisID": f"y{trend_config.y_axis_id}",
                }
                datasets.append(dataset)

        chart_config = {
            "type": "line",
            "data": {"datasets": datasets},
            "options": {
                "responsive": True,
                "plugins": {
                    "title": {"display": True, "text": config.title},
                    "legend": {"display": config.show_legend}
                },
                "scales": {
                    "x": {
                        "type": "time",
                        "time": {"unit": "minute"}
                    }
                }
            }
        }

        # 配置Y轴
        for y_axis in config.y_axes:
            axis_key = f"y{y_axis.axis_id}" if y_axis.axis_id > 0 else "y"
            chart_config["options"]["scales"][axis_key] = {
                "type": "linear",
                "position": y_axis.position,
                "title": {"display": True, "text": y_axis.label},
                "grid": {"display": y_axis.grid_lines}
            }
            if not y_axis.auto_scale:
                if y_axis.min_value is not None:
                    chart_config["options"]["scales"][axis_key]["min"] = y_axis.min_value
                if y_axis.max_value is not None:
                    chart_config["options"]["scales"][axis_key]["max"] = y_axis.max_value

        html = f"""
        <div id="trend-panel-{config.panel_id}" style="background-color: {config.background_color}; padding: 10px;">
            <canvas id="chart-{config.panel_id}"></canvas>
            <script>
                const ctx_{config.panel_id} = document.getElementById('chart-{config.panel_id}');
                new Chart(ctx_{config.panel_id}, {json.dumps(chart_config)});
            </script>
        </div>
        """
        return html


class JSONTrendRenderer(TrendRenderer):
    """JSON趋势渲染器(用于API响应)"""

    def render(
        self,
        config: TrendPanelConfig,
        data: Dict[str, List[DataPoint]]
    ) -> str:
        """渲染JSON格式趋势数据"""
        result = {
            "panel_id": config.panel_id,
            "title": config.title,
            "trends": []
        }

        for trend_config in config.trends:
            trend_data = {
                "tag_name": trend_config.tag_name,
                "display_name": trend_config.display_name,
                "unit": trend_config.unit,
                "color": trend_config.color,
                "points": []
            }

            if trend_config.tag_name in data:
                points = data[trend_config.tag_name]
                trend_data["points"] = [p.to_dict() for p in points]

                # 添加统计信息
                if points:
                    trend_data["statistics"] = StatisticalAnalyzer.calculate_statistics(points)

            result["trends"].append(trend_data)

        return json.dumps(result, indent=2)


class RealTimeTrendManager:
    """实时趋势管理器"""

    def __init__(self):
        self.panels: Dict[str, TrendPanelConfig] = {}
        self.data_buffer = TrendDataBuffer()
        self._update_callbacks: Dict[str, List[Callable]] = {}
        self._running = False
        self._update_task: Optional[asyncio.Task] = None

    def add_panel(self, config: TrendPanelConfig):
        """添加趋势面板"""
        self.panels[config.panel_id] = config
        self._update_callbacks[config.panel_id] = []

    def remove_panel(self, panel_id: str):
        """移除趋势面板"""
        if panel_id in self.panels:
            del self.panels[panel_id]
        if panel_id in self._update_callbacks:
            del self._update_callbacks[panel_id]

    def subscribe(self, panel_id: str, callback: Callable[[str, Dict[str, List[DataPoint]]], None]):
        """订阅面板更新"""
        if panel_id in self._update_callbacks:
            self._update_callbacks[panel_id].append(callback)

    async def update_data(self, tag_name: str, value: float, quality: int = 192):
        """更新数据"""
        point = DataPoint(
            timestamp=datetime.now(),
            value=value,
            quality=quality
        )
        await self.data_buffer.add_point(tag_name, point)

    async def start(self):
        """启动实时更新"""
        self._running = True
        self._update_task = asyncio.create_task(self._update_loop())

    async def stop(self):
        """停止实时更新"""
        self._running = False
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass

    async def _update_loop(self):
        """更新循环"""
        while self._running:
            for panel_id, config in self.panels.items():
                # 获取时间范围内的数据
                end_time = datetime.now()
                start_time = end_time - timedelta(minutes=config.time_range_minutes)

                data = {}
                for trend in config.trends:
                    points = await self.data_buffer.get_points(
                        trend.tag_name, start_time, end_time
                    )
                    data[trend.tag_name] = points

                # 通知订阅者
                for callback in self._update_callbacks.get(panel_id, []):
                    try:
                        callback(panel_id, data)
                    except Exception:
                        pass

            await asyncio.sleep(0.1)  # 100ms更新间隔

    async def get_panel_data(
        self,
        panel_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        aggregation: AggregationType = AggregationType.RAW,
        interval_seconds: int = 60
    ) -> Dict[str, List[DataPoint]]:
        """获取面板数据"""
        if panel_id not in self.panels:
            return {}

        config = self.panels[panel_id]

        if end_time is None:
            end_time = datetime.now()
        if start_time is None:
            start_time = end_time - timedelta(minutes=config.time_range_minutes)

        data = {}
        for trend in config.trends:
            points = await self.data_buffer.get_points(
                trend.tag_name, start_time, end_time
            )

            if aggregation != AggregationType.RAW:
                points = DataAggregator.aggregate(points, interval_seconds, aggregation)

            data[trend.tag_name] = points

        return data


class MultiVariableAnalyzer:
    """多变量分析器"""

    def __init__(self, data_buffer: TrendDataBuffer):
        self.data_buffer = data_buffer

    async def compare_trends(
        self,
        tag_names: List[str],
        start_time: datetime,
        end_time: datetime,
        normalize: bool = True
    ) -> Dict[str, List[DataPoint]]:
        """比较多个趋势"""
        result = {}

        for tag_name in tag_names:
            points = await self.data_buffer.get_points(tag_name, start_time, end_time)

            if normalize and points:
                # 归一化到0-1范围
                values = [p.value for p in points]
                min_val = min(values)
                max_val = max(values)
                range_val = max_val - min_val if max_val != min_val else 1

                normalized_points = [
                    DataPoint(
                        timestamp=p.timestamp,
                        value=(p.value - min_val) / range_val,
                        quality=p.quality
                    )
                    for p in points
                ]
                result[tag_name] = normalized_points
            else:
                result[tag_name] = points

        return result

    async def calculate_correlation_matrix(
        self,
        tag_names: List[str],
        start_time: datetime,
        end_time: datetime
    ) -> Dict[Tuple[str, str], float]:
        """计算相关系数矩阵"""
        # 获取所有数据
        all_data = {}
        for tag_name in tag_names:
            all_data[tag_name] = await self.data_buffer.get_points(
                tag_name, start_time, end_time
            )

        # 计算相关系数
        correlation_matrix = {}
        for i, tag1 in enumerate(tag_names):
            for tag2 in tag_names[i:]:
                if tag1 == tag2:
                    correlation = 1.0
                else:
                    correlation = StatisticalAnalyzer.calculate_correlation(
                        all_data[tag1], all_data[tag2]
                    )
                correlation_matrix[(tag1, tag2)] = correlation
                correlation_matrix[(tag2, tag1)] = correlation

        return correlation_matrix

    async def find_leading_indicators(
        self,
        target_tag: str,
        candidate_tags: List[str],
        start_time: datetime,
        end_time: datetime,
        max_lag_minutes: int = 60
    ) -> List[Tuple[str, int, float]]:
        """寻找领先指标(时滞相关分析)"""
        target_points = await self.data_buffer.get_points(target_tag, start_time, end_time)
        if not target_points:
            return []

        results = []

        for candidate_tag in candidate_tags:
            candidate_points = await self.data_buffer.get_points(
                candidate_tag, start_time, end_time
            )
            if not candidate_points:
                continue

            best_lag = 0
            best_correlation = 0.0

            # 测试不同的时滞
            for lag_minutes in range(-max_lag_minutes, max_lag_minutes + 1, 5):
                # 偏移候选变量的时间
                shifted_points = [
                    DataPoint(
                        timestamp=p.timestamp + timedelta(minutes=lag_minutes),
                        value=p.value,
                        quality=p.quality
                    )
                    for p in candidate_points
                ]

                correlation = StatisticalAnalyzer.calculate_correlation(
                    target_points, shifted_points
                )

                if abs(correlation) > abs(best_correlation):
                    best_correlation = correlation
                    best_lag = lag_minutes

            if abs(best_correlation) > 0.5:  # 只保留显著相关的
                results.append((candidate_tag, best_lag, best_correlation))

        # 按相关性排序
        results.sort(key=lambda x: abs(x[2]), reverse=True)
        return results


def create_cyrp_trend_system() -> RealTimeTrendManager:
    """创建穿黄工程趋势系统"""
    manager = RealTimeTrendManager()

    # 水力监测面板
    hydraulic_panel = TrendPanelConfig(
        panel_id="hydraulic_monitoring",
        title="水力参数实时监测",
        trends=[
            TrendConfig(
                tag_name="inlet_flow",
                display_name="进口流量",
                unit="m³/s",
                color="#1f77b4",
                y_axis_id=0
            ),
            TrendConfig(
                tag_name="outlet_flow",
                display_name="出口流量",
                unit="m³/s",
                color="#ff7f0e",
                y_axis_id=0
            ),
            TrendConfig(
                tag_name="inlet_pressure",
                display_name="进口压力",
                unit="MPa",
                color="#2ca02c",
                y_axis_id=1
            ),
            TrendConfig(
                tag_name="outlet_pressure",
                display_name="出口压力",
                unit="MPa",
                color="#d62728",
                y_axis_id=1
            ),
        ],
        y_axes=[
            YAxisConfig(axis_id=0, label="流量 (m³/s)", position="left"),
            YAxisConfig(axis_id=1, label="压力 (MPa)", position="right"),
        ],
        time_range_minutes=60,
        refresh_interval_ms=1000
    )
    manager.add_panel(hydraulic_panel)

    # 设备状态面板
    equipment_panel = TrendPanelConfig(
        panel_id="equipment_status",
        title="设备运行状态",
        trends=[
            TrendConfig(
                tag_name="pump_speed",
                display_name="泵转速",
                unit="RPM",
                color="#9467bd",
                y_axis_id=0
            ),
            TrendConfig(
                tag_name="pump_power",
                display_name="泵功率",
                unit="kW",
                color="#8c564b",
                y_axis_id=1
            ),
            TrendConfig(
                tag_name="vibration",
                display_name="振动",
                unit="mm/s",
                color="#e377c2",
                y_axis_id=2
            ),
            TrendConfig(
                tag_name="temperature",
                display_name="轴承温度",
                unit="°C",
                color="#7f7f7f",
                y_axis_id=3
            ),
        ],
        y_axes=[
            YAxisConfig(axis_id=0, label="转速 (RPM)", position="left"),
            YAxisConfig(axis_id=1, label="功率 (kW)", position="left"),
            YAxisConfig(axis_id=2, label="振动 (mm/s)", position="right"),
            YAxisConfig(axis_id=3, label="温度 (°C)", position="right"),
        ],
        time_range_minutes=120,
        refresh_interval_ms=2000
    )
    manager.add_panel(equipment_panel)

    # 水质监测面板
    water_quality_panel = TrendPanelConfig(
        panel_id="water_quality",
        title="水质参数监测",
        trends=[
            TrendConfig(
                tag_name="turbidity",
                display_name="浊度",
                unit="NTU",
                color="#bcbd22",
                y_axis_id=0
            ),
            TrendConfig(
                tag_name="ph",
                display_name="pH值",
                unit="",
                color="#17becf",
                y_axis_id=1
            ),
            TrendConfig(
                tag_name="dissolved_oxygen",
                display_name="溶解氧",
                unit="mg/L",
                color="#1f77b4",
                y_axis_id=2
            ),
        ],
        y_axes=[
            YAxisConfig(axis_id=0, label="浊度 (NTU)", position="left"),
            YAxisConfig(axis_id=1, label="pH", position="right", min_value=6.0, max_value=9.0, auto_scale=False),
            YAxisConfig(axis_id=2, label="溶解氧 (mg/L)", position="right"),
        ],
        time_range_minutes=240,
        refresh_interval_ms=5000
    )
    manager.add_panel(water_quality_panel)

    return manager
