"""
数据持久化管理器 - Data Persistence Manager

集成监控系统与存储后端，实现：
- 监控指标自动持久化
- 告警历史记录
- 系统状态快照
- 数据归档与清理

Integrates monitoring system with storage backends:
- Automatic metric persistence
- Alert history recording
- System status snapshots
- Data archiving and cleanup
"""

import time
import threading
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import logging

from cyrp.database.historian import (
    StorageBackend,
    SQLiteBackend,
    TimeSeriesPoint,
    TimeSeriesMetadata,
    DataType,
    AggregationType,
    QueryResult,
)

logger = logging.getLogger(__name__)


class PersistenceLevel(Enum):
    """持久化级别"""
    NONE = "none"              # 不持久化
    MEMORY_ONLY = "memory"     # 仅内存
    IMMEDIATE = "immediate"    # 立即写入
    BUFFERED = "buffered"      # 缓冲写入
    ARCHIVED = "archived"      # 归档存储


@dataclass
class PersistenceConfig:
    """持久化配置"""
    level: PersistenceLevel = PersistenceLevel.BUFFERED
    buffer_size: int = 1000           # 缓冲区大小
    flush_interval: float = 10.0      # 刷新间隔（秒）
    retention_days: int = 365         # 数据保留天数
    downsample_after_days: int = 7    # 降采样起始天数
    archive_after_days: int = 30      # 归档起始天数
    compression: bool = True          # 是否压缩归档


@dataclass
class MetricPersistenceRule:
    """指标持久化规则"""
    metric_name: str
    level: PersistenceLevel = PersistenceLevel.BUFFERED
    retention_days: int = 365
    sample_interval: float = 1.0      # 采样间隔（秒）
    aggregation: AggregationType = AggregationType.MEAN
    tags: Dict[str, str] = field(default_factory=dict)


class MetricBuffer:
    """指标缓冲区"""

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._buffer: Dict[str, deque] = {}
        self._lock = threading.Lock()
        self._last_sample_time: Dict[str, float] = {}

    def add(self, metric_name: str, value: float, timestamp: float = None,
            quality: int = 0, tags: Dict = None, min_interval: float = 0):
        """添加数据点"""
        if timestamp is None:
            timestamp = time.time()

        # 检查采样间隔
        if min_interval > 0:
            last_time = self._last_sample_time.get(metric_name, 0)
            if timestamp - last_time < min_interval:
                return False

        with self._lock:
            if metric_name not in self._buffer:
                self._buffer[metric_name] = deque(maxlen=self.max_size)

            point = TimeSeriesPoint(
                timestamp=datetime.fromtimestamp(timestamp),
                value=value,
                quality=quality,
                tags=tags or {}
            )
            self._buffer[metric_name].append(point)
            self._last_sample_time[metric_name] = timestamp
            return True

    def get_and_clear(self, metric_name: str = None) -> Dict[str, List[TimeSeriesPoint]]:
        """获取并清空缓冲区"""
        with self._lock:
            if metric_name:
                if metric_name in self._buffer:
                    points = list(self._buffer[metric_name])
                    self._buffer[metric_name].clear()
                    return {metric_name: points}
                return {}
            else:
                result = {name: list(points) for name, points in self._buffer.items()}
                for name in self._buffer:
                    self._buffer[name].clear()
                return result

    def get_count(self) -> int:
        """获取总数据点数"""
        with self._lock:
            return sum(len(points) for points in self._buffer.values())

    def get_metrics(self) -> List[str]:
        """获取所有指标名"""
        with self._lock:
            return list(self._buffer.keys())


class AlertPersistence:
    """告警持久化"""

    def __init__(self, backend: StorageBackend):
        self.backend = backend
        self._pending_alerts: List[Dict] = []
        self._lock = threading.Lock()

    def record_alert(
        self,
        alert_id: str,
        rule_id: str,
        name: str,
        severity: str,
        message: str,
        metric_value: float,
        threshold: float,
        timestamp: float,
        tags: Dict = None
    ):
        """记录告警"""
        alert_data = {
            'alert_id': alert_id,
            'rule_id': rule_id,
            'name': name,
            'severity': severity,
            'message': message,
            'metric_value': metric_value,
            'threshold': threshold,
            'timestamp': datetime.fromtimestamp(timestamp),
            'tags': tags or {}
        }

        with self._lock:
            self._pending_alerts.append(alert_data)

    def record_acknowledgement(self, alert_id: str, user_id: str, timestamp: float):
        """记录告警确认"""
        with self._lock:
            # 在实际实现中，这会更新数据库记录
            pass

    def record_resolution(self, alert_id: str, timestamp: float):
        """记录告警解决"""
        with self._lock:
            pass

    def flush(self) -> int:
        """刷新到存储"""
        with self._lock:
            if not self._pending_alerts:
                return 0

            count = len(self._pending_alerts)
            # 转换为时序数据点存储
            for alert in self._pending_alerts:
                try:
                    # 存储告警作为时序数据
                    point = TimeSeriesPoint(
                        timestamp=alert['timestamp'],
                        value=1,  # 告警发生计数
                        quality=0,
                        tags={
                            'alert_id': alert['alert_id'],
                            'severity': alert['severity'],
                            'rule_id': alert['rule_id']
                        }
                    )
                    self.backend.write(f"alert_{alert['severity']}", [point])
                except Exception as e:
                    logger.error(f"Failed to persist alert {alert['alert_id']}: {e}")

            self._pending_alerts.clear()
            return count

    def query_history(
        self,
        start_time: datetime,
        end_time: datetime,
        severity: str = None,
        limit: int = 1000
    ) -> List[Dict]:
        """查询告警历史"""
        series_name = f"alert_{severity}" if severity else "alert_critical"
        try:
            result = self.backend.query(series_name, start_time, end_time, limit=limit)
            return [
                {
                    'timestamp': p.timestamp.isoformat(),
                    'severity': p.tags.get('severity', 'unknown'),
                    'alert_id': p.tags.get('alert_id', ''),
                    'rule_id': p.tags.get('rule_id', '')
                }
                for p in result.points
            ]
        except Exception as e:
            logger.error(f"Failed to query alert history: {e}")
            return []


class StatusSnapshotManager:
    """状态快照管理器"""

    def __init__(self, backend: StorageBackend, snapshot_interval: float = 60.0):
        self.backend = backend
        self.snapshot_interval = snapshot_interval
        self._last_snapshot_time = 0
        self._snapshot_history: deque = deque(maxlen=1000)

    def take_snapshot(self, status: Dict) -> bool:
        """保存状态快照"""
        current_time = time.time()

        # 检查间隔
        if current_time - self._last_snapshot_time < self.snapshot_interval:
            return False

        try:
            timestamp = datetime.fromtimestamp(current_time)

            # 存储各项状态指标
            metrics_to_store = [
                ('health_score', status.get('health_score', 0)),
                ('flow_rate', status.get('flow_rate', 0)),
                ('pressure_avg', status.get('pressure_avg', 0)),
                ('pressure_max', status.get('pressure_max', 0)),
                ('active_alarms', status.get('active_alarms', 0)),
            ]

            for metric_name, value in metrics_to_store:
                point = TimeSeriesPoint(
                    timestamp=timestamp,
                    value=float(value),
                    quality=0,
                    tags={'snapshot': 'true'}
                )
                self.backend.write(f"status_{metric_name}", [point])

            # 保存完整快照到内存
            self._snapshot_history.append({
                'timestamp': current_time,
                'status': status
            })

            self._last_snapshot_time = current_time
            return True

        except Exception as e:
            logger.error(f"Failed to take status snapshot: {e}")
            return False

    def get_recent_snapshots(self, count: int = 10) -> List[Dict]:
        """获取最近的快照"""
        return list(self._snapshot_history)[-count:]


class DataArchiver:
    """数据归档器"""

    def __init__(
        self,
        backend: StorageBackend,
        archive_path: str = "./archive",
        compression: bool = True
    ):
        self.backend = backend
        self.archive_path = archive_path
        self.compression = compression
        self._archive_stats = {
            'total_archived': 0,
            'last_archive_time': None,
            'total_deleted': 0
        }

    def archive_old_data(
        self,
        series_names: List[str],
        older_than_days: int = 30
    ) -> Dict[str, int]:
        """归档旧数据"""
        cutoff_time = datetime.now() - timedelta(days=older_than_days)
        result = {}

        for series_name in series_names:
            try:
                # 查询旧数据
                data = self.backend.query(
                    series_name,
                    datetime.min,
                    cutoff_time,
                    limit=100000
                )

                if data.points:
                    # 归档到文件
                    archived_count = self._archive_to_file(series_name, data.points)
                    result[series_name] = archived_count

                    # 从主存储删除
                    self.backend.delete(series_name, datetime.min, cutoff_time)
                    self._archive_stats['total_archived'] += archived_count

            except Exception as e:
                logger.error(f"Failed to archive {series_name}: {e}")
                result[series_name] = 0

        self._archive_stats['last_archive_time'] = datetime.now()
        return result

    def _archive_to_file(self, series_name: str, points: List[TimeSeriesPoint]) -> int:
        """归档到文件"""
        import os
        os.makedirs(self.archive_path, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{series_name}_{timestamp}.json"
        if self.compression:
            filename += ".gz"

        filepath = os.path.join(self.archive_path, filename)

        data = [p.to_dict() for p in points]

        try:
            if self.compression:
                import gzip
                with gzip.open(filepath, 'wt', encoding='utf-8') as f:
                    json.dump(data, f)
            else:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)

            return len(points)
        except Exception as e:
            logger.error(f"Failed to write archive file {filepath}: {e}")
            return 0

    def cleanup_expired_data(
        self,
        series_names: List[str],
        retention_days: int = 365
    ) -> Dict[str, int]:
        """清理过期数据"""
        cutoff_time = datetime.now() - timedelta(days=retention_days)
        result = {}

        for series_name in series_names:
            try:
                deleted = self.backend.delete(series_name, datetime.min, cutoff_time)
                result[series_name] = deleted
                self._archive_stats['total_deleted'] += deleted
            except Exception as e:
                logger.error(f"Failed to cleanup {series_name}: {e}")
                result[series_name] = 0

        return result

    def get_stats(self) -> Dict[str, Any]:
        """获取归档统计"""
        return self._archive_stats.copy()


class PersistenceManager:
    """持久化管理器"""

    def __init__(
        self,
        backend: StorageBackend = None,
        config: PersistenceConfig = None
    ):
        """
        初始化持久化管理器

        Args:
            backend: 存储后端（默认SQLite）
            config: 持久化配置
        """
        self.backend = backend or SQLiteBackend(":memory:")
        self.config = config or PersistenceConfig()

        # 组件
        self.metric_buffer = MetricBuffer(self.config.buffer_size)
        self.alert_persistence = AlertPersistence(self.backend)
        self.snapshot_manager = StatusSnapshotManager(self.backend)
        self.archiver = DataArchiver(self.backend)

        # 规则
        self._metric_rules: Dict[str, MetricPersistenceRule] = {}

        # 后台任务
        self._running = False
        self._flush_thread = None

        # 统计
        self._stats = {
            'metrics_persisted': 0,
            'alerts_persisted': 0,
            'snapshots_taken': 0,
            'flush_count': 0,
            'last_flush_time': None
        }

        # 初始化
        self._initialize()

    def _initialize(self):
        """初始化"""
        if not self.backend.connect():
            logger.error("Failed to connect to storage backend")
            return

        # 创建默认时序
        default_series = [
            ('status_health_score', DataType.FLOAT, '%', '系统健康评分'),
            ('status_flow_rate', DataType.FLOAT, 'm³/s', '流量'),
            ('status_pressure_avg', DataType.FLOAT, 'Pa', '平均压力'),
            ('status_pressure_max', DataType.FLOAT, 'Pa', '最大压力'),
            ('status_active_alarms', DataType.INTEGER, '', '活动告警数'),
            ('alert_critical', DataType.INTEGER, '', '严重告警'),
            ('alert_warning', DataType.INTEGER, '', '警告告警'),
            ('alert_info', DataType.INTEGER, '', '信息告警'),
        ]

        for name, dtype, unit, desc in default_series:
            metadata = TimeSeriesMetadata(
                name=name,
                data_type=dtype,
                unit=unit,
                description=desc,
                retention_days=self.config.retention_days
            )
            self.backend.create_series(metadata)

    def add_metric_rule(self, rule: MetricPersistenceRule):
        """添加指标持久化规则"""
        self._metric_rules[rule.metric_name] = rule

        # 创建时序
        metadata = TimeSeriesMetadata(
            name=rule.metric_name,
            data_type=DataType.FLOAT,
            tags=rule.tags,
            retention_days=rule.retention_days
        )
        self.backend.create_series(metadata)

    def record_metric(
        self,
        metric_name: str,
        value: float,
        timestamp: float = None,
        quality: int = 0,
        tags: Dict = None
    ):
        """记录指标"""
        rule = self._metric_rules.get(metric_name)
        min_interval = rule.sample_interval if rule else 1.0

        if self.metric_buffer.add(
            metric_name, value, timestamp, quality, tags, min_interval
        ):
            # 立即模式下直接写入
            if self.config.level == PersistenceLevel.IMMEDIATE:
                self._flush_metric(metric_name)

    def record_metrics_batch(self, metrics: Dict[str, float], timestamp: float = None):
        """批量记录指标"""
        for name, value in metrics.items():
            self.record_metric(name, value, timestamp)

    def record_alert(self, alert: Dict):
        """记录告警"""
        self.alert_persistence.record_alert(
            alert_id=alert.get('alert_id', ''),
            rule_id=alert.get('rule_id', ''),
            name=alert.get('name', ''),
            severity=alert.get('severity', 'info'),
            message=alert.get('message', ''),
            metric_value=alert.get('metric_value', 0),
            threshold=alert.get('threshold', 0),
            timestamp=alert.get('timestamp', time.time()),
            tags=alert.get('tags')
        )

    def record_status(self, status: Dict):
        """记录系统状态"""
        if self.snapshot_manager.take_snapshot(status):
            self._stats['snapshots_taken'] += 1

    def _flush_metric(self, metric_name: str):
        """刷新单个指标"""
        data = self.metric_buffer.get_and_clear(metric_name)
        for name, points in data.items():
            if points:
                try:
                    self.backend.write(name, points)
                    self._stats['metrics_persisted'] += len(points)
                except Exception as e:
                    logger.error(f"Failed to flush metric {name}: {e}")

    def flush_all(self) -> Dict[str, int]:
        """刷新所有缓冲数据"""
        result = {'metrics': 0, 'alerts': 0}

        # 刷新指标
        data = self.metric_buffer.get_and_clear()
        for name, points in data.items():
            if points:
                try:
                    self.backend.write(name, points)
                    result['metrics'] += len(points)
                except Exception as e:
                    logger.error(f"Failed to flush metric {name}: {e}")

        # 刷新告警
        result['alerts'] = self.alert_persistence.flush()

        self._stats['metrics_persisted'] += result['metrics']
        self._stats['alerts_persisted'] += result['alerts']
        self._stats['flush_count'] += 1
        self._stats['last_flush_time'] = datetime.now()

        return result

    def start_background_flush(self):
        """启动后台刷新"""
        if self._running:
            return

        self._running = True
        self._flush_thread = threading.Thread(target=self._background_flush_loop)
        self._flush_thread.daemon = True
        self._flush_thread.start()
        logger.info("Background flush started")

    def stop_background_flush(self):
        """停止后台刷新"""
        self._running = False
        if self._flush_thread:
            self._flush_thread.join(timeout=5.0)
        # 最后刷新一次
        self.flush_all()

    def _background_flush_loop(self):
        """后台刷新循环"""
        while self._running:
            try:
                time.sleep(self.config.flush_interval)
                if self._running:
                    self.flush_all()
            except Exception as e:
                logger.error(f"Background flush error: {e}")

    def query_metrics(
        self,
        metric_name: str,
        start_time: datetime,
        end_time: datetime,
        aggregation: AggregationType = None,
        interval: str = None
    ) -> QueryResult:
        """查询指标历史"""
        return self.backend.query(
            metric_name,
            start_time,
            end_time,
            aggregation=aggregation,
            interval=interval
        )

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self._stats,
            'buffer_count': self.metric_buffer.get_count(),
            'buffer_metrics': self.metric_buffer.get_metrics(),
            'archiver_stats': self.archiver.get_stats()
        }

    def close(self):
        """关闭管理器"""
        self.stop_background_flush()
        self.backend.disconnect()


def create_persistence_system(
    db_path: str = "cyrp_history.db",
    config: PersistenceConfig = None
) -> PersistenceManager:
    """
    创建持久化系统

    Args:
        db_path: 数据库路径
        config: 持久化配置

    Returns:
        PersistenceManager: 持久化管理器
    """
    backend = SQLiteBackend(db_path)
    manager = PersistenceManager(backend, config)
    return manager
