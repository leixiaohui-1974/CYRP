"""
System Monitoring and Performance Analysis Module for CYRP
穿黄工程系统监控与性能分析模块

功能:
- 系统资源监控(CPU/内存/磁盘/网络)
- 应用性能监控(APM)
- 性能指标采集与分析
- 健康检查
- 告警阈值监控
"""

import asyncio
import gc
import os
import platform
import sys
import time
import threading
import traceback
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import statistics


class MetricType(Enum):
    """指标类型"""
    COUNTER = auto()      # 计数器(只增不减)
    GAUGE = auto()        # 仪表(可增可减)
    HISTOGRAM = auto()    # 直方图
    SUMMARY = auto()      # 摘要(百分位数)
    TIMER = auto()        # 计时器


class HealthStatus(Enum):
    """健康状态"""
    HEALTHY = auto()
    DEGRADED = auto()
    UNHEALTHY = auto()
    UNKNOWN = auto()


@dataclass
class MetricValue:
    """指标值"""
    name: str
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE


@dataclass
class PerformanceSnapshot:
    """性能快照"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    disk_percent: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_sent_mb: float
    network_recv_mb: float
    active_threads: int
    gc_stats: Dict[str, int]


@dataclass
class HealthCheckResult:
    """健康检查结果"""
    name: str
    status: HealthStatus
    message: str = ""
    duration_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)


class SystemMetricsCollector:
    """系统指标采集器"""

    def __init__(self):
        self._last_cpu_times = None
        self._last_disk_io = None
        self._last_net_io = None
        self._last_collect_time = None

    def collect(self) -> PerformanceSnapshot:
        """采集系统指标"""
        now = datetime.now()

        # CPU使用率(简化实现)
        cpu_percent = self._get_cpu_percent()

        # 内存使用
        memory_info = self._get_memory_info()

        # 磁盘使用
        disk_info = self._get_disk_info()

        # 网络IO
        net_info = self._get_network_info()

        # 线程数
        active_threads = threading.active_count()

        # GC统计
        gc_stats = {
            f"gen{i}": gc.get_count()[i] if i < len(gc.get_count()) else 0
            for i in range(3)
        }

        self._last_collect_time = now

        return PerformanceSnapshot(
            timestamp=now,
            cpu_percent=cpu_percent,
            memory_percent=memory_info["percent"],
            memory_used_mb=memory_info["used_mb"],
            disk_percent=disk_info["percent"],
            disk_io_read_mb=disk_info["read_mb"],
            disk_io_write_mb=disk_info["write_mb"],
            network_sent_mb=net_info["sent_mb"],
            network_recv_mb=net_info["recv_mb"],
            active_threads=active_threads,
            gc_stats=gc_stats
        )

    def _get_cpu_percent(self) -> float:
        """获取CPU使用率"""
        try:
            # 尝试读取/proc/stat(Linux)
            if os.path.exists('/proc/stat'):
                with open('/proc/stat', 'r') as f:
                    line = f.readline()
                    parts = line.split()
                    if parts[0] == 'cpu':
                        user = int(parts[1])
                        nice = int(parts[2])
                        system = int(parts[3])
                        idle = int(parts[4])
                        total = user + nice + system + idle

                        if self._last_cpu_times:
                            total_diff = total - self._last_cpu_times['total']
                            idle_diff = idle - self._last_cpu_times['idle']
                            if total_diff > 0:
                                cpu_percent = 100.0 * (1 - idle_diff / total_diff)
                            else:
                                cpu_percent = 0.0
                        else:
                            cpu_percent = 0.0

                        self._last_cpu_times = {'total': total, 'idle': idle}
                        return cpu_percent
        except Exception:
            pass

        # 回退:返回估计值
        return 0.0

    def _get_memory_info(self) -> Dict[str, float]:
        """获取内存信息"""
        try:
            if os.path.exists('/proc/meminfo'):
                mem_info = {}
                with open('/proc/meminfo', 'r') as f:
                    for line in f:
                        parts = line.split()
                        if len(parts) >= 2:
                            key = parts[0].rstrip(':')
                            value = int(parts[1])  # KB
                            mem_info[key] = value

                total = mem_info.get('MemTotal', 1)
                available = mem_info.get('MemAvailable', mem_info.get('MemFree', 0))
                used = total - available

                return {
                    "percent": 100.0 * used / total if total > 0 else 0.0,
                    "used_mb": used / 1024.0,
                    "total_mb": total / 1024.0
                }
        except Exception:
            pass

        # 使用Python gc作为回退
        gc_stats = gc.get_stats()
        return {
            "percent": 0.0,
            "used_mb": sum(s.get('collected', 0) for s in gc_stats) / 1024.0,
            "total_mb": 0.0
        }

    def _get_disk_info(self) -> Dict[str, float]:
        """获取磁盘信息"""
        try:
            stat = os.statvfs('/')
            total = stat.f_blocks * stat.f_frsize
            free = stat.f_bavail * stat.f_frsize
            used = total - free
            percent = 100.0 * used / total if total > 0 else 0.0

            # 磁盘IO
            read_mb = 0.0
            write_mb = 0.0
            if os.path.exists('/proc/diskstats'):
                with open('/proc/diskstats', 'r') as f:
                    for line in f:
                        parts = line.split()
                        if len(parts) >= 14:
                            read_mb += int(parts[5]) * 512 / (1024 * 1024)
                            write_mb += int(parts[9]) * 512 / (1024 * 1024)

            return {
                "percent": percent,
                "read_mb": read_mb,
                "write_mb": write_mb
            }
        except Exception:
            pass

        return {"percent": 0.0, "read_mb": 0.0, "write_mb": 0.0}

    def _get_network_info(self) -> Dict[str, float]:
        """获取网络信息"""
        try:
            if os.path.exists('/proc/net/dev'):
                sent_bytes = 0
                recv_bytes = 0
                with open('/proc/net/dev', 'r') as f:
                    lines = f.readlines()[2:]  # 跳过头部
                    for line in lines:
                        parts = line.split()
                        if len(parts) >= 10:
                            recv_bytes += int(parts[1])
                            sent_bytes += int(parts[9])

                return {
                    "sent_mb": sent_bytes / (1024 * 1024),
                    "recv_mb": recv_bytes / (1024 * 1024)
                }
        except Exception:
            pass

        return {"sent_mb": 0.0, "recv_mb": 0.0}


class MetricsRegistry:
    """指标注册表"""

    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self._metrics: Dict[str, deque] = {}
        self._lock = threading.Lock()

    def record(self, metric: MetricValue):
        """记录指标"""
        with self._lock:
            key = f"{metric.name}:{','.join(f'{k}={v}' for k, v in sorted(metric.labels.items()))}"
            if key not in self._metrics:
                self._metrics[key] = deque(maxlen=self.max_history)
            self._metrics[key].append(metric)

    def get(self, name: str, labels: Optional[Dict[str, str]] = None) -> List[MetricValue]:
        """获取指标"""
        with self._lock:
            if labels:
                key = f"{name}:{','.join(f'{k}={v}' for k, v in sorted(labels.items()))}"
                return list(self._metrics.get(key, []))
            else:
                result = []
                for key, values in self._metrics.items():
                    if key.startswith(f"{name}:") or key == name:
                        result.extend(values)
                return result

    def get_latest(self, name: str, labels: Optional[Dict[str, str]] = None) -> Optional[MetricValue]:
        """获取最新指标"""
        metrics = self.get(name, labels)
        return metrics[-1] if metrics else None

    def get_statistics(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None,
        window_minutes: int = 5
    ) -> Dict[str, float]:
        """获取指标统计"""
        metrics = self.get(name, labels)
        cutoff = datetime.now() - timedelta(minutes=window_minutes)
        values = [m.value for m in metrics if m.timestamp >= cutoff]

        if not values:
            return {}

        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": statistics.mean(values),
            "median": statistics.median(values),
            "stddev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "p95": self._percentile(values, 95),
            "p99": self._percentile(values, 99),
        }

    def _percentile(self, values: List[float], percentile: float) -> float:
        """计算百分位数"""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]


class Timer:
    """计时器上下文管理器"""

    def __init__(self, registry: MetricsRegistry, name: str, labels: Optional[Dict[str, str]] = None):
        self.registry = registry
        self.name = name
        self.labels = labels or {}
        self.start_time = None
        self.duration = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.duration = (time.perf_counter() - self.start_time) * 1000  # ms
        self.registry.record(MetricValue(
            name=self.name,
            value=self.duration,
            labels=self.labels,
            metric_type=MetricType.TIMER
        ))
        return False


def timed(registry: MetricsRegistry, name: Optional[str] = None):
    """计时装饰器"""
    def decorator(func: Callable) -> Callable:
        metric_name = name or f"{func.__module__}.{func.__name__}"

        @wraps(func)
        def wrapper(*args, **kwargs):
            with Timer(registry, metric_name):
                return func(*args, **kwargs)

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            with Timer(registry, metric_name):
                return await func(*args, **kwargs)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper

    return decorator


class HealthChecker:
    """健康检查器"""

    def __init__(self):
        self._checks: Dict[str, Callable[[], HealthCheckResult]] = {}

    def register(self, name: str, check: Callable[[], HealthCheckResult]):
        """注册健康检查"""
        self._checks[name] = check

    def register_async(self, name: str, check: Callable):
        """注册异步健康检查"""
        self._checks[name] = check

    async def check(self, name: str) -> HealthCheckResult:
        """执行单个健康检查"""
        if name not in self._checks:
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNKNOWN,
                message="Check not found"
            )

        check = self._checks[name]
        start_time = time.perf_counter()

        try:
            if asyncio.iscoroutinefunction(check):
                result = await check()
            else:
                result = check()

            result.duration_ms = (time.perf_counter() - start_time) * 1000
            return result
        except Exception as e:
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=str(e),
                duration_ms=(time.perf_counter() - start_time) * 1000,
                details={"error": traceback.format_exc()}
            )

    async def check_all(self) -> Dict[str, HealthCheckResult]:
        """执行所有健康检查"""
        results = {}
        for name in self._checks:
            results[name] = await self.check(name)
        return results

    def get_overall_status(self, results: Dict[str, HealthCheckResult]) -> HealthStatus:
        """获取总体健康状态"""
        statuses = [r.status for r in results.values()]

        if all(s == HealthStatus.HEALTHY for s in statuses):
            return HealthStatus.HEALTHY
        elif any(s == HealthStatus.UNHEALTHY for s in statuses):
            return HealthStatus.UNHEALTHY
        elif any(s == HealthStatus.DEGRADED for s in statuses):
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.UNKNOWN


class PerformanceMonitor:
    """性能监控器"""

    def __init__(
        self,
        collect_interval: float = 5.0,
        history_size: int = 1000
    ):
        self.collect_interval = collect_interval
        self.history_size = history_size

        self.collector = SystemMetricsCollector()
        self.registry = MetricsRegistry(max_history=history_size)
        self.health_checker = HealthChecker()

        self._snapshots: deque = deque(maxlen=history_size)
        self._running = False
        self._collect_task: Optional[asyncio.Task] = None
        self._alert_callbacks: List[Callable] = []

        # 告警阈值
        self.thresholds = {
            "cpu_percent": 90.0,
            "memory_percent": 85.0,
            "disk_percent": 90.0,
        }

        # 注册默认健康检查
        self._register_default_checks()

    def _register_default_checks(self):
        """注册默认健康检查"""
        # 系统资源检查
        def check_system_resources() -> HealthCheckResult:
            snapshot = self.collector.collect()

            issues = []
            if snapshot.cpu_percent > self.thresholds["cpu_percent"]:
                issues.append(f"CPU使用率过高: {snapshot.cpu_percent:.1f}%")
            if snapshot.memory_percent > self.thresholds["memory_percent"]:
                issues.append(f"内存使用率过高: {snapshot.memory_percent:.1f}%")
            if snapshot.disk_percent > self.thresholds["disk_percent"]:
                issues.append(f"磁盘使用率过高: {snapshot.disk_percent:.1f}%")

            if issues:
                return HealthCheckResult(
                    name="system_resources",
                    status=HealthStatus.DEGRADED if len(issues) < 2 else HealthStatus.UNHEALTHY,
                    message="; ".join(issues),
                    details={
                        "cpu_percent": snapshot.cpu_percent,
                        "memory_percent": snapshot.memory_percent,
                        "disk_percent": snapshot.disk_percent
                    }
                )
            else:
                return HealthCheckResult(
                    name="system_resources",
                    status=HealthStatus.HEALTHY,
                    message="系统资源正常",
                    details={
                        "cpu_percent": snapshot.cpu_percent,
                        "memory_percent": snapshot.memory_percent,
                        "disk_percent": snapshot.disk_percent
                    }
                )

        self.health_checker.register("system_resources", check_system_resources)

        # Python运行时检查
        def check_python_runtime() -> HealthCheckResult:
            gc_stats = gc.get_stats()
            thread_count = threading.active_count()

            details = {
                "python_version": sys.version,
                "platform": platform.platform(),
                "thread_count": thread_count,
                "gc_generations": len(gc_stats)
            }

            return HealthCheckResult(
                name="python_runtime",
                status=HealthStatus.HEALTHY,
                message=f"Python {sys.version_info.major}.{sys.version_info.minor} 运行正常",
                details=details
            )

        self.health_checker.register("python_runtime", check_python_runtime)

    def add_alert_callback(self, callback: Callable[[str, float, float], None]):
        """添加告警回调"""
        self._alert_callbacks.append(callback)

    def set_threshold(self, metric_name: str, value: float):
        """设置告警阈值"""
        self.thresholds[metric_name] = value

    async def start(self):
        """启动监控"""
        self._running = True
        self._collect_task = asyncio.create_task(self._collect_loop())

    async def stop(self):
        """停止监控"""
        self._running = False
        if self._collect_task:
            self._collect_task.cancel()
            try:
                await self._collect_task
            except asyncio.CancelledError:
                pass

    async def _collect_loop(self):
        """采集循环"""
        while self._running:
            try:
                snapshot = self.collector.collect()
                self._snapshots.append(snapshot)

                # 记录指标
                self.registry.record(MetricValue(
                    name="system.cpu.percent",
                    value=snapshot.cpu_percent,
                    metric_type=MetricType.GAUGE
                ))
                self.registry.record(MetricValue(
                    name="system.memory.percent",
                    value=snapshot.memory_percent,
                    metric_type=MetricType.GAUGE
                ))
                self.registry.record(MetricValue(
                    name="system.disk.percent",
                    value=snapshot.disk_percent,
                    metric_type=MetricType.GAUGE
                ))
                self.registry.record(MetricValue(
                    name="system.threads.count",
                    value=snapshot.active_threads,
                    metric_type=MetricType.GAUGE
                ))

                # 检查阈值
                self._check_thresholds(snapshot)

            except Exception as e:
                print(f"采集指标失败: {e}")

            await asyncio.sleep(self.collect_interval)

    def _check_thresholds(self, snapshot: PerformanceSnapshot):
        """检查阈值"""
        checks = [
            ("cpu_percent", snapshot.cpu_percent),
            ("memory_percent", snapshot.memory_percent),
            ("disk_percent", snapshot.disk_percent),
        ]

        for metric_name, value in checks:
            threshold = self.thresholds.get(metric_name)
            if threshold and value > threshold:
                for callback in self._alert_callbacks:
                    try:
                        callback(metric_name, value, threshold)
                    except Exception:
                        pass

    def get_latest_snapshot(self) -> Optional[PerformanceSnapshot]:
        """获取最新快照"""
        return self._snapshots[-1] if self._snapshots else None

    def get_history(
        self,
        duration_minutes: int = 60
    ) -> List[PerformanceSnapshot]:
        """获取历史快照"""
        cutoff = datetime.now() - timedelta(minutes=duration_minutes)
        return [s for s in self._snapshots if s.timestamp >= cutoff]

    def get_statistics(self, window_minutes: int = 5) -> Dict[str, Dict[str, float]]:
        """获取统计信息"""
        return {
            "cpu_percent": self.registry.get_statistics("system.cpu.percent", window_minutes=window_minutes),
            "memory_percent": self.registry.get_statistics("system.memory.percent", window_minutes=window_minutes),
            "disk_percent": self.registry.get_statistics("system.disk.percent", window_minutes=window_minutes),
        }

    def timer(self, name: str, labels: Optional[Dict[str, str]] = None) -> Timer:
        """创建计时器"""
        return Timer(self.registry, name, labels)


class ApplicationProfiler:
    """应用性能分析器"""

    def __init__(self, registry: MetricsRegistry):
        self.registry = registry
        self._request_count = 0
        self._error_count = 0
        self._lock = threading.Lock()

    def record_request(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        duration_ms: float
    ):
        """记录请求"""
        with self._lock:
            self._request_count += 1
            if status_code >= 400:
                self._error_count += 1

        labels = {
            "endpoint": endpoint,
            "method": method,
            "status": str(status_code)
        }

        self.registry.record(MetricValue(
            name="http.request.duration",
            value=duration_ms,
            labels=labels,
            metric_type=MetricType.TIMER
        ))

        self.registry.record(MetricValue(
            name="http.request.count",
            value=1,
            labels=labels,
            metric_type=MetricType.COUNTER
        ))

    def record_database_query(
        self,
        operation: str,
        table: str,
        duration_ms: float,
        rows_affected: int = 0
    ):
        """记录数据库查询"""
        labels = {
            "operation": operation,
            "table": table
        }

        self.registry.record(MetricValue(
            name="db.query.duration",
            value=duration_ms,
            labels=labels,
            metric_type=MetricType.TIMER
        ))

        self.registry.record(MetricValue(
            name="db.query.rows",
            value=rows_affected,
            labels=labels,
            metric_type=MetricType.GAUGE
        ))

    def record_external_call(
        self,
        service: str,
        operation: str,
        duration_ms: float,
        success: bool = True
    ):
        """记录外部调用"""
        labels = {
            "service": service,
            "operation": operation,
            "success": str(success)
        }

        self.registry.record(MetricValue(
            name="external.call.duration",
            value=duration_ms,
            labels=labels,
            metric_type=MetricType.TIMER
        ))

    def get_request_rate(self, window_seconds: int = 60) -> float:
        """获取请求速率"""
        stats = self.registry.get_statistics(
            "http.request.count",
            window_minutes=window_seconds // 60 or 1
        )
        return stats.get("count", 0) / window_seconds

    def get_error_rate(self) -> float:
        """获取错误率"""
        with self._lock:
            if self._request_count == 0:
                return 0.0
            return self._error_count / self._request_count


def create_cyrp_monitoring_system() -> PerformanceMonitor:
    """创建穿黄工程监控系统"""
    monitor = PerformanceMonitor(
        collect_interval=5.0,
        history_size=10000
    )

    # 添加CYRP特定的健康检查
    def check_data_freshness() -> HealthCheckResult:
        # 检查数据新鲜度(模拟)
        return HealthCheckResult(
            name="data_freshness",
            status=HealthStatus.HEALTHY,
            message="数据更新正常",
            details={"last_update": datetime.now().isoformat()}
        )

    monitor.health_checker.register("data_freshness", check_data_freshness)

    def check_communication() -> HealthCheckResult:
        # 检查通信状态(模拟)
        return HealthCheckResult(
            name="communication",
            status=HealthStatus.HEALTHY,
            message="通信连接正常",
            details={"active_connections": 5}
        )

    monitor.health_checker.register("communication", check_communication)

    # 设置CYRP特定阈值
    monitor.set_threshold("cpu_percent", 80.0)
    monitor.set_threshold("memory_percent", 80.0)
    monitor.set_threshold("disk_percent", 85.0)

    return monitor
