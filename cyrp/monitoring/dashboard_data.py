"""
系统监控仪表板数据输出 - Dashboard Data Output Module

提供实时监控数据的结构化输出，支持：
- Grafana/Prometheus格式
- JSON API格式
- 实时流数据
- 历史数据查询

Provides structured output for real-time monitoring data supporting:
- Grafana/Prometheus format
- JSON API format
- Real-time streaming
- Historical data queries
"""

import time
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from collections import deque
from enum import Enum
import threading


class MetricType(Enum):
    """指标类型"""
    GAUGE = "gauge"           # 仪表值（瞬时）
    COUNTER = "counter"       # 计数器（累积）
    HISTOGRAM = "histogram"   # 直方图
    SUMMARY = "summary"       # 摘要


@dataclass
class Metric:
    """指标数据"""
    name: str
    value: float
    timestamp: float
    metric_type: MetricType = MetricType.GAUGE
    labels: Dict[str, str] = field(default_factory=dict)
    unit: str = ""
    description: str = ""

    def to_prometheus(self) -> str:
        """转换为Prometheus格式"""
        labels_str = ""
        if self.labels:
            label_pairs = [f'{k}="{v}"' for k, v in self.labels.items()]
            labels_str = "{" + ",".join(label_pairs) + "}"

        return f"{self.name}{labels_str} {self.value} {int(self.timestamp * 1000)}"

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'name': self.name,
            'value': self.value,
            'timestamp': self.timestamp,
            'type': self.metric_type.value,
            'labels': self.labels,
            'unit': self.unit,
            'description': self.description
        }


@dataclass
class DashboardPanel:
    """仪表板面板"""
    panel_id: str
    title: str
    metrics: List[str]
    panel_type: str = "timeseries"  # timeseries, gauge, stat, table
    thresholds: Dict[str, float] = field(default_factory=dict)
    unit: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SystemStatus:
    """系统状态摘要"""
    timestamp: float
    overall_health: str  # excellent, good, degraded, critical
    health_score: float  # 0-100

    # 子系统状态
    hydraulic_status: str
    structural_status: str
    control_status: str
    sensor_status: str

    # 关键指标
    flow_rate: float
    pressure_avg: float
    pressure_max: float
    water_level: float

    # 预警信息
    active_alarms: int
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DashboardDataProvider:
    """仪表板数据提供器"""

    def __init__(self, history_size: int = 10000):
        """
        初始化数据提供器

        Args:
            history_size: 历史数据保留数量
        """
        self.history_size = history_size
        self._metrics: Dict[str, deque] = {}
        self._current_status: Optional[SystemStatus] = None
        self._lock = threading.Lock()

        # 预定义的面板配置
        self._panels = self._create_default_panels()

        # 指标定义
        self._metric_definitions = self._create_metric_definitions()

    def _create_default_panels(self) -> Dict[str, DashboardPanel]:
        """创建默认面板配置"""
        return {
            'flow_overview': DashboardPanel(
                panel_id='flow_overview',
                title='流量监控',
                metrics=['flow_rate_north', 'flow_rate_south', 'flow_rate_total'],
                panel_type='timeseries',
                thresholds={'warning': 290, 'critical': 305},
                unit='m³/s'
            ),
            'pressure_overview': DashboardPanel(
                panel_id='pressure_overview',
                title='压力监控',
                metrics=['pressure_avg', 'pressure_max', 'pressure_min'],
                panel_type='timeseries',
                thresholds={'warning': 8e5, 'critical': 1e6},
                unit='Pa'
            ),
            'system_health': DashboardPanel(
                panel_id='system_health',
                title='系统健康',
                metrics=['health_score'],
                panel_type='gauge',
                thresholds={'warning': 70, 'critical': 50},
                unit='%'
            ),
            'sensor_status': DashboardPanel(
                panel_id='sensor_status',
                title='传感器状态',
                metrics=['sensor_availability', 'sensor_quality', 'sensor_count'],
                panel_type='stat'
            ),
            'control_performance': DashboardPanel(
                panel_id='control_performance',
                title='控制性能',
                metrics=['setpoint_error', 'control_effort', 'settling_time'],
                panel_type='timeseries'
            ),
            'prediction_accuracy': DashboardPanel(
                panel_id='prediction_accuracy',
                title='预测精度',
                metrics=['prediction_rmse', 'prediction_mape', 'prediction_confidence'],
                panel_type='timeseries'
            ),
        }

    def _create_metric_definitions(self) -> Dict[str, Dict[str, Any]]:
        """创建指标定义"""
        return {
            'flow_rate_north': {
                'type': MetricType.GAUGE,
                'unit': 'm³/s',
                'description': '北线流量'
            },
            'flow_rate_south': {
                'type': MetricType.GAUGE,
                'unit': 'm³/s',
                'description': '南线流量'
            },
            'flow_rate_total': {
                'type': MetricType.GAUGE,
                'unit': 'm³/s',
                'description': '总流量'
            },
            'pressure_avg': {
                'type': MetricType.GAUGE,
                'unit': 'Pa',
                'description': '平均压力'
            },
            'pressure_max': {
                'type': MetricType.GAUGE,
                'unit': 'Pa',
                'description': '最大压力'
            },
            'pressure_min': {
                'type': MetricType.GAUGE,
                'unit': 'Pa',
                'description': '最小压力'
            },
            'health_score': {
                'type': MetricType.GAUGE,
                'unit': '%',
                'description': '系统健康评分'
            },
            'sensor_availability': {
                'type': MetricType.GAUGE,
                'unit': '%',
                'description': '传感器可用率'
            },
            'prediction_rmse': {
                'type': MetricType.GAUGE,
                'unit': '',
                'description': '预测均方根误差'
            },
            'prediction_confidence': {
                'type': MetricType.GAUGE,
                'unit': '%',
                'description': '预测置信度'
            },
        }

    def record_metric(
        self,
        name: str,
        value: float,
        labels: Dict[str, str] = None,
        timestamp: float = None
    ):
        """
        记录指标

        Args:
            name: 指标名称
            value: 指标值
            labels: 标签
            timestamp: 时间戳
        """
        if timestamp is None:
            timestamp = time.time()

        definition = self._metric_definitions.get(name, {})
        metric = Metric(
            name=name,
            value=value,
            timestamp=timestamp,
            metric_type=definition.get('type', MetricType.GAUGE),
            labels=labels or {},
            unit=definition.get('unit', ''),
            description=definition.get('description', '')
        )

        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = deque(maxlen=self.history_size)
            self._metrics[name].append(metric)

    def record_batch(self, metrics: Dict[str, float], timestamp: float = None):
        """
        批量记录指标

        Args:
            metrics: 指标字典 {name: value}
            timestamp: 时间戳
        """
        if timestamp is None:
            timestamp = time.time()

        for name, value in metrics.items():
            self.record_metric(name, value, timestamp=timestamp)

    def update_system_status(self, status: SystemStatus):
        """更新系统状态"""
        with self._lock:
            self._current_status = status

    def get_current_metrics(self) -> Dict[str, Metric]:
        """获取当前指标值"""
        with self._lock:
            result = {}
            for name, history in self._metrics.items():
                if history:
                    result[name] = history[-1]
            return result

    def get_metric_history(
        self,
        name: str,
        start_time: float = None,
        end_time: float = None,
        limit: int = 1000
    ) -> List[Metric]:
        """
        获取指标历史

        Args:
            name: 指标名称
            start_time: 开始时间
            end_time: 结束时间
            limit: 返回数量限制

        Returns:
            指标历史列表
        """
        with self._lock:
            if name not in self._metrics:
                return []

            history = list(self._metrics[name])

            if start_time:
                history = [m for m in history if m.timestamp >= start_time]
            if end_time:
                history = [m for m in history if m.timestamp <= end_time]

            return history[-limit:]

    def get_prometheus_metrics(self) -> str:
        """
        获取Prometheus格式指标

        Returns:
            Prometheus格式字符串
        """
        lines = []
        current = self.get_current_metrics()

        for name, metric in current.items():
            # 添加HELP和TYPE注释
            if metric.description:
                lines.append(f"# HELP {name} {metric.description}")
            lines.append(f"# TYPE {name} {metric.metric_type.value}")
            lines.append(metric.to_prometheus())

        return "\n".join(lines)

    def get_grafana_dashboard(self) -> Dict[str, Any]:
        """
        获取Grafana仪表板配置

        Returns:
            Grafana仪表板JSON配置
        """
        panels = []
        for i, (panel_id, panel) in enumerate(self._panels.items()):
            panel_config = {
                'id': i + 1,
                'title': panel.title,
                'type': panel.panel_type,
                'gridPos': {'x': (i % 2) * 12, 'y': (i // 2) * 8, 'w': 12, 'h': 8},
                'targets': [
                    {'refId': chr(65 + j), 'expr': f'cyrp_{m}'}
                    for j, m in enumerate(panel.metrics)
                ],
                'fieldConfig': {
                    'defaults': {
                        'unit': panel.unit,
                        'thresholds': {
                            'mode': 'absolute',
                            'steps': [
                                {'color': 'green', 'value': None},
                                {'color': 'yellow', 'value': panel.thresholds.get('warning')},
                                {'color': 'red', 'value': panel.thresholds.get('critical')},
                            ]
                        }
                    }
                }
            }
            panels.append(panel_config)

        return {
            'title': 'CYRP 穿黄工程监控仪表板',
            'uid': 'cyrp-main',
            'panels': panels,
            'time': {'from': 'now-1h', 'to': 'now'},
            'refresh': '5s'
        }

    def get_api_response(self) -> Dict[str, Any]:
        """
        获取API响应数据

        Returns:
            API格式的监控数据
        """
        current_metrics = self.get_current_metrics()
        status = self._current_status

        response = {
            'timestamp': time.time(),
            'status': status.to_dict() if status else None,
            'metrics': {
                name: metric.to_dict()
                for name, metric in current_metrics.items()
            },
            'panels': {
                name: panel.to_dict()
                for name, panel in self._panels.items()
            }
        }

        return response

    def get_realtime_snapshot(self) -> Dict[str, Any]:
        """
        获取实时快照数据

        Returns:
            实时监控快照
        """
        current = self.get_current_metrics()
        status = self._current_status

        snapshot = {
            'timestamp': time.time(),
            'system': {
                'health': status.overall_health if status else 'unknown',
                'health_score': status.health_score if status else 0,
            },
            'hydraulic': {
                'flow_rate': current.get('flow_rate_total', Metric('', 0, 0)).value,
                'pressure_avg': current.get('pressure_avg', Metric('', 0, 0)).value,
                'pressure_max': current.get('pressure_max', Metric('', 0, 0)).value,
            },
            'sensors': {
                'availability': current.get('sensor_availability', Metric('', 100, 0)).value,
            },
            'prediction': {
                'confidence': current.get('prediction_confidence', Metric('', 0, 0)).value,
                'rmse': current.get('prediction_rmse', Metric('', 0, 0)).value,
            },
            'alarms': {
                'active': status.active_alarms if status else 0,
                'warnings': status.warnings if status else [],
            }
        }

        return snapshot


class MetricsCollector:
    """指标收集器 - 从各模块收集指标"""

    def __init__(self, dashboard: DashboardDataProvider):
        """
        初始化收集器

        Args:
            dashboard: 仪表板数据提供器
        """
        self.dashboard = dashboard
        self._collectors: Dict[str, callable] = {}

    def register_collector(self, name: str, collector: callable):
        """
        注册指标收集器

        Args:
            name: 收集器名称
            collector: 收集函数，返回Dict[str, float]
        """
        self._collectors[name] = collector

    def collect_all(self):
        """收集所有指标"""
        timestamp = time.time()

        for name, collector in self._collectors.items():
            try:
                metrics = collector()
                if metrics:
                    self.dashboard.record_batch(metrics, timestamp)
            except Exception as e:
                print(f"Error collecting metrics from {name}: {e}")

    def create_system_status(
        self,
        health_score: float,
        flow_rate: float,
        pressure_avg: float,
        pressure_max: float,
        water_level: float,
        active_alarms: int = 0,
        warnings: List[str] = None
    ) -> SystemStatus:
        """
        创建系统状态

        Args:
            health_score: 健康评分 (0-100)
            flow_rate: 流量
            pressure_avg: 平均压力
            pressure_max: 最大压力
            water_level: 水位
            active_alarms: 活动告警数
            warnings: 警告列表

        Returns:
            SystemStatus: 系统状态
        """
        # 确定整体健康状态
        if health_score >= 90:
            overall_health = "excellent"
        elif health_score >= 70:
            overall_health = "good"
        elif health_score >= 50:
            overall_health = "degraded"
        else:
            overall_health = "critical"

        # 子系统状态（简化判断）
        hydraulic_status = "normal" if 200 <= flow_rate <= 300 else "warning"
        structural_status = "normal"
        control_status = "normal" if health_score >= 60 else "degraded"
        sensor_status = "normal" if active_alarms == 0 else "warning"

        return SystemStatus(
            timestamp=time.time(),
            overall_health=overall_health,
            health_score=health_score,
            hydraulic_status=hydraulic_status,
            structural_status=structural_status,
            control_status=control_status,
            sensor_status=sensor_status,
            flow_rate=flow_rate,
            pressure_avg=pressure_avg,
            pressure_max=pressure_max,
            water_level=water_level,
            active_alarms=active_alarms,
            warnings=warnings or []
        )
