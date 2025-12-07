"""
Log Aggregation and Analysis Module for CYRP
穿黄工程日志聚合与分析模块

实现分布式日志收集、存储、搜索、分析等功能
"""

import asyncio
import json
import time
import uuid
import re
import gzip
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import (
    Any, Dict, List, Optional, Callable, Set, Tuple,
    Pattern, Iterator
)
from collections import defaultdict
import logging
import os

logger = logging.getLogger(__name__)


# ============================================================
# 枚举定义
# ============================================================

class LogLevel(Enum):
    """日志级别"""
    TRACE = 0
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50
    FATAL = 60


class LogSource(Enum):
    """日志来源"""
    APPLICATION = "application"
    SYSTEM = "system"
    SECURITY = "security"
    AUDIT = "audit"
    PERFORMANCE = "performance"
    NETWORK = "network"
    DATABASE = "database"


class AlertSeverity(Enum):
    """告警严重程度"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# ============================================================
# 数据类定义
# ============================================================

@dataclass
class LogEntry:
    """日志条目"""
    log_id: str
    timestamp: datetime
    level: LogLevel
    source: LogSource
    service: str
    message: str
    host: str = ""
    process_id: int = 0
    thread_id: int = 0
    logger_name: str = ""
    exception: Optional[str] = None
    stack_trace: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'log_id': self.log_id,
            'timestamp': self.timestamp.isoformat(),
            'level': self.level.name,
            'source': self.source.value,
            'service': self.service,
            'message': self.message,
            'host': self.host,
            'process_id': self.process_id,
            'thread_id': self.thread_id,
            'logger_name': self.logger_name,
            'exception': self.exception,
            'stack_trace': self.stack_trace,
            'context': self.context,
            'tags': self.tags
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LogEntry':
        """从字典创建"""
        return cls(
            log_id=data['log_id'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            level=LogLevel[data['level']],
            source=LogSource(data['source']),
            service=data['service'],
            message=data['message'],
            host=data.get('host', ''),
            process_id=data.get('process_id', 0),
            thread_id=data.get('thread_id', 0),
            logger_name=data.get('logger_name', ''),
            exception=data.get('exception'),
            stack_trace=data.get('stack_trace'),
            context=data.get('context', {}),
            tags=data.get('tags', [])
        )


@dataclass
class LogQuery:
    """日志查询"""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    levels: Optional[List[LogLevel]] = None
    sources: Optional[List[LogSource]] = None
    services: Optional[List[str]] = None
    hosts: Optional[List[str]] = None
    keyword: Optional[str] = None
    regex_pattern: Optional[str] = None
    tags: Optional[List[str]] = None
    context_filters: Optional[Dict[str, Any]] = None
    limit: int = 100
    offset: int = 0
    order_desc: bool = True


@dataclass
class LogStats:
    """日志统计"""
    total_count: int = 0
    by_level: Dict[str, int] = field(default_factory=dict)
    by_source: Dict[str, int] = field(default_factory=dict)
    by_service: Dict[str, int] = field(default_factory=dict)
    by_host: Dict[str, int] = field(default_factory=dict)
    error_rate: float = 0.0
    logs_per_minute: float = 0.0


@dataclass
class LogAlert:
    """日志告警"""
    alert_id: str
    name: str
    condition: str  # 条件表达式
    severity: AlertSeverity
    message_template: str
    cooldown_seconds: int = 300
    enabled: bool = True
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0


@dataclass
class LogPattern:
    """日志模式"""
    pattern_id: str
    name: str
    regex: str
    description: str = ""
    extract_fields: List[str] = field(default_factory=list)
    sample_logs: List[str] = field(default_factory=list)
    occurrence_count: int = 0


# ============================================================
# 日志解析器
# ============================================================

class LogParser(ABC):
    """日志解析器基类"""

    @abstractmethod
    def parse(self, raw_log: str) -> Optional[LogEntry]:
        """解析原始日志"""
        pass


class JSONLogParser(LogParser):
    """JSON日志解析器"""

    def parse(self, raw_log: str) -> Optional[LogEntry]:
        """解析JSON格式日志"""
        try:
            data = json.loads(raw_log)

            # 标准化字段
            level_str = data.get('level', data.get('severity', 'INFO')).upper()
            level = LogLevel[level_str] if level_str in LogLevel.__members__ else LogLevel.INFO

            timestamp_str = data.get('timestamp', data.get('time', data.get('@timestamp')))
            if timestamp_str:
                timestamp = self._parse_timestamp(timestamp_str)
            else:
                timestamp = datetime.now()

            return LogEntry(
                log_id=data.get('log_id', str(uuid.uuid4())),
                timestamp=timestamp,
                level=level,
                source=LogSource(data.get('source', 'application')),
                service=data.get('service', data.get('app', 'unknown')),
                message=data.get('message', data.get('msg', '')),
                host=data.get('host', data.get('hostname', '')),
                process_id=data.get('pid', data.get('process_id', 0)),
                thread_id=data.get('tid', data.get('thread_id', 0)),
                logger_name=data.get('logger', data.get('logger_name', '')),
                exception=data.get('exception', data.get('error')),
                stack_trace=data.get('stack_trace', data.get('stacktrace')),
                context=data.get('context', data.get('extra', {})),
                tags=data.get('tags', [])
            )

        except Exception as e:
            logger.warning(f"Failed to parse JSON log: {e}")
            return None

    def _parse_timestamp(self, ts_str: str) -> datetime:
        """解析时间戳"""
        formats = [
            '%Y-%m-%dT%H:%M:%S.%fZ',
            '%Y-%m-%dT%H:%M:%SZ',
            '%Y-%m-%dT%H:%M:%S.%f',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%d %H:%M:%S.%f',
            '%Y-%m-%d %H:%M:%S',
        ]

        for fmt in formats:
            try:
                return datetime.strptime(ts_str, fmt)
            except ValueError:
                continue

        return datetime.now()


class SyslogParser(LogParser):
    """Syslog格式解析器"""

    # RFC 5424 格式
    SYSLOG_PATTERN = re.compile(
        r'<(\d+)>(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(.*)'
    )

    # 简单格式
    SIMPLE_PATTERN = re.compile(
        r'(\w+\s+\d+\s+\d+:\d+:\d+)\s+(\S+)\s+(\S+?)(?:\[(\d+)\])?:\s*(.*)'
    )

    LEVEL_MAP = {
        0: LogLevel.FATAL,
        1: LogLevel.CRITICAL,
        2: LogLevel.CRITICAL,
        3: LogLevel.ERROR,
        4: LogLevel.WARNING,
        5: LogLevel.INFO,
        6: LogLevel.INFO,
        7: LogLevel.DEBUG,
    }

    def parse(self, raw_log: str) -> Optional[LogEntry]:
        """解析Syslog格式日志"""
        # 尝试RFC 5424格式
        match = self.SYSLOG_PATTERN.match(raw_log)
        if match:
            priority = int(match.group(1))
            severity = priority % 8
            level = self.LEVEL_MAP.get(severity, LogLevel.INFO)

            return LogEntry(
                log_id=str(uuid.uuid4()),
                timestamp=self._parse_timestamp(match.group(3)),
                level=level,
                source=LogSource.SYSTEM,
                service=match.group(5),
                message=match.group(8),
                host=match.group(4),
                process_id=int(match.group(6)) if match.group(6).isdigit() else 0
            )

        # 尝试简单格式
        match = self.SIMPLE_PATTERN.match(raw_log)
        if match:
            return LogEntry(
                log_id=str(uuid.uuid4()),
                timestamp=self._parse_timestamp(match.group(1)),
                level=LogLevel.INFO,
                source=LogSource.SYSTEM,
                service=match.group(3),
                message=match.group(5),
                host=match.group(2),
                process_id=int(match.group(4)) if match.group(4) else 0
            )

        return None

    def _parse_timestamp(self, ts_str: str) -> datetime:
        """解析时间戳"""
        try:
            return datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
        except ValueError:
            pass

        try:
            # Syslog月日时格式
            current_year = datetime.now().year
            return datetime.strptime(f"{current_year} {ts_str}", "%Y %b %d %H:%M:%S")
        except ValueError:
            return datetime.now()


class RegexLogParser(LogParser):
    """正则表达式日志解析器"""

    def __init__(self, pattern: str, field_mapping: Dict[str, str]):
        self.pattern = re.compile(pattern)
        self.field_mapping = field_mapping

    def parse(self, raw_log: str) -> Optional[LogEntry]:
        """解析日志"""
        match = self.pattern.match(raw_log)
        if not match:
            return None

        groups = match.groupdict()

        # 映射字段
        level_str = groups.get(self.field_mapping.get('level', ''), 'INFO').upper()
        level = LogLevel[level_str] if level_str in LogLevel.__members__ else LogLevel.INFO

        return LogEntry(
            log_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            level=level,
            source=LogSource.APPLICATION,
            service=groups.get(self.field_mapping.get('service', ''), 'unknown'),
            message=groups.get(self.field_mapping.get('message', ''), raw_log),
            host=groups.get(self.field_mapping.get('host', ''), ''),
            logger_name=groups.get(self.field_mapping.get('logger', ''), '')
        )


# ============================================================
# 日志存储
# ============================================================

class LogStorage(ABC):
    """日志存储抽象基类"""

    @abstractmethod
    async def write(self, entries: List[LogEntry]) -> int:
        """写入日志"""
        pass

    @abstractmethod
    async def query(self, query: LogQuery) -> List[LogEntry]:
        """查询日志"""
        pass

    @abstractmethod
    async def delete(self, before: datetime) -> int:
        """删除指定时间前的日志"""
        pass

    @abstractmethod
    async def get_stats(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> LogStats:
        """获取统计信息"""
        pass


class InMemoryLogStorage(LogStorage):
    """内存日志存储"""

    def __init__(self, max_entries: int = 100000):
        self.max_entries = max_entries
        self.entries: List[LogEntry] = []
        self._lock = asyncio.Lock()

    async def write(self, entries: List[LogEntry]) -> int:
        """写入日志"""
        async with self._lock:
            self.entries.extend(entries)

            # 清理旧日志
            if len(self.entries) > self.max_entries:
                self.entries = self.entries[-self.max_entries:]

            return len(entries)

    async def query(self, query: LogQuery) -> List[LogEntry]:
        """查询日志"""
        results = []

        for entry in self.entries:
            if not self._matches_query(entry, query):
                continue
            results.append(entry)

        # 排序
        results.sort(key=lambda x: x.timestamp, reverse=query.order_desc)

        # 分页
        start = query.offset
        end = start + query.limit
        return results[start:end]

    def _matches_query(self, entry: LogEntry, query: LogQuery) -> bool:
        """检查日志是否匹配查询条件"""
        if query.start_time and entry.timestamp < query.start_time:
            return False
        if query.end_time and entry.timestamp > query.end_time:
            return False
        if query.levels and entry.level not in query.levels:
            return False
        if query.sources and entry.source not in query.sources:
            return False
        if query.services and entry.service not in query.services:
            return False
        if query.hosts and entry.host not in query.hosts:
            return False
        if query.keyword and query.keyword.lower() not in entry.message.lower():
            return False
        if query.regex_pattern:
            if not re.search(query.regex_pattern, entry.message):
                return False
        if query.tags:
            if not any(tag in entry.tags for tag in query.tags):
                return False
        if query.context_filters:
            for key, value in query.context_filters.items():
                if entry.context.get(key) != value:
                    return False

        return True

    async def delete(self, before: datetime) -> int:
        """删除指定时间前的日志"""
        async with self._lock:
            original_count = len(self.entries)
            self.entries = [e for e in self.entries if e.timestamp >= before]
            return original_count - len(self.entries)

    async def get_stats(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> LogStats:
        """获取统计信息"""
        filtered = [
            e for e in self.entries
            if start_time <= e.timestamp <= end_time
        ]

        stats = LogStats(total_count=len(filtered))

        for entry in filtered:
            stats.by_level[entry.level.name] = \
                stats.by_level.get(entry.level.name, 0) + 1
            stats.by_source[entry.source.value] = \
                stats.by_source.get(entry.source.value, 0) + 1
            stats.by_service[entry.service] = \
                stats.by_service.get(entry.service, 0) + 1
            stats.by_host[entry.host] = \
                stats.by_host.get(entry.host, 0) + 1

        if filtered:
            error_count = sum(
                1 for e in filtered
                if e.level.value >= LogLevel.ERROR.value
            )
            stats.error_rate = error_count / len(filtered)

            time_range = (end_time - start_time).total_seconds() / 60
            if time_range > 0:
                stats.logs_per_minute = len(filtered) / time_range

        return stats


class FileLogStorage(LogStorage):
    """文件日志存储"""

    def __init__(
        self,
        base_path: str,
        rotate_size_mb: int = 100,
        compress: bool = True
    ):
        self.base_path = base_path
        self.rotate_size_mb = rotate_size_mb
        self.compress = compress
        self._current_file = None
        self._current_size = 0
        self._lock = asyncio.Lock()

        os.makedirs(base_path, exist_ok=True)

    async def write(self, entries: List[LogEntry]) -> int:
        """写入日志"""
        async with self._lock:
            for entry in entries:
                line = json.dumps(entry.to_dict()) + '\n'
                await self._write_line(line)
            return len(entries)

    async def _write_line(self, line: str):
        """写入一行"""
        # 检查是否需要轮转
        if self._current_size >= self.rotate_size_mb * 1024 * 1024:
            await self._rotate()

        # 获取当前文件
        if self._current_file is None:
            self._current_file = self._get_current_file_path()

        with open(self._current_file, 'a', encoding='utf-8') as f:
            f.write(line)
            self._current_size += len(line.encode('utf-8'))

    def _get_current_file_path(self) -> str:
        """获取当前文件路径"""
        date_str = datetime.now().strftime('%Y%m%d')
        return os.path.join(self.base_path, f'logs_{date_str}.jsonl')

    async def _rotate(self):
        """轮转日志文件"""
        if self._current_file and os.path.exists(self._current_file):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            rotated = f"{self._current_file}.{timestamp}"
            os.rename(self._current_file, rotated)

            if self.compress:
                with open(rotated, 'rb') as f_in:
                    with gzip.open(f"{rotated}.gz", 'wb') as f_out:
                        f_out.writelines(f_in)
                os.remove(rotated)

        self._current_file = None
        self._current_size = 0

    async def query(self, query: LogQuery) -> List[LogEntry]:
        """查询日志"""
        results = []

        # 确定要搜索的文件
        files = self._get_files_in_range(query.start_time, query.end_time)

        for file_path in files:
            entries = await self._read_file(file_path)
            for entry in entries:
                if self._matches_query(entry, query):
                    results.append(entry)

        # 排序和分页
        results.sort(key=lambda x: x.timestamp, reverse=query.order_desc)
        start = query.offset
        end = start + query.limit
        return results[start:end]

    def _get_files_in_range(
        self,
        start_time: Optional[datetime],
        end_time: Optional[datetime]
    ) -> List[str]:
        """获取时间范围内的文件"""
        files = []
        for filename in os.listdir(self.base_path):
            if filename.startswith('logs_') and (
                filename.endswith('.jsonl') or filename.endswith('.jsonl.gz')
            ):
                files.append(os.path.join(self.base_path, filename))
        return sorted(files)

    async def _read_file(self, file_path: str) -> List[LogEntry]:
        """读取文件"""
        entries = []

        try:
            if file_path.endswith('.gz'):
                with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                    for line in f:
                        entry = LogEntry.from_dict(json.loads(line))
                        entries.append(entry)
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        entry = LogEntry.from_dict(json.loads(line))
                        entries.append(entry)
        except Exception as e:
            logger.error(f"Failed to read log file {file_path}: {e}")

        return entries

    def _matches_query(self, entry: LogEntry, query: LogQuery) -> bool:
        """检查日志是否匹配查询条件"""
        if query.start_time and entry.timestamp < query.start_time:
            return False
        if query.end_time and entry.timestamp > query.end_time:
            return False
        if query.levels and entry.level not in query.levels:
            return False
        if query.keyword and query.keyword.lower() not in entry.message.lower():
            return False
        return True

    async def delete(self, before: datetime) -> int:
        """删除指定时间前的日志"""
        deleted = 0
        for filename in os.listdir(self.base_path):
            file_path = os.path.join(self.base_path, filename)
            # 根据文件名日期判断
            try:
                date_str = filename.split('_')[1].split('.')[0]
                file_date = datetime.strptime(date_str, '%Y%m%d')
                if file_date.date() < before.date():
                    os.remove(file_path)
                    deleted += 1
            except (IndexError, ValueError):
                continue
        return deleted

    async def get_stats(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> LogStats:
        """获取统计信息"""
        query = LogQuery(start_time=start_time, end_time=end_time, limit=100000)
        entries = await self.query(query)

        stats = LogStats(total_count=len(entries))
        for entry in entries:
            stats.by_level[entry.level.name] = \
                stats.by_level.get(entry.level.name, 0) + 1

        return stats


# ============================================================
# 日志分析器
# ============================================================

class LogAnalyzer:
    """日志分析器"""

    def __init__(self, storage: LogStorage):
        self.storage = storage

    async def analyze_errors(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, Any]:
        """分析错误日志"""
        query = LogQuery(
            start_time=start_time,
            end_time=end_time,
            levels=[LogLevel.ERROR, LogLevel.CRITICAL, LogLevel.FATAL],
            limit=10000
        )
        errors = await self.storage.query(query)

        # 按消息模式分组
        patterns = defaultdict(list)
        for entry in errors:
            pattern = self._extract_pattern(entry.message)
            patterns[pattern].append(entry)

        # 排序
        sorted_patterns = sorted(
            patterns.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )

        return {
            'total_errors': len(errors),
            'unique_patterns': len(patterns),
            'top_patterns': [
                {
                    'pattern': p,
                    'count': len(entries),
                    'first_seen': min(e.timestamp for e in entries).isoformat(),
                    'last_seen': max(e.timestamp for e in entries).isoformat(),
                    'sample': entries[0].message
                }
                for p, entries in sorted_patterns[:10]
            ]
        }

    def _extract_pattern(self, message: str) -> str:
        """提取消息模式"""
        # 替换数字
        pattern = re.sub(r'\d+', '<NUM>', message)
        # 替换UUID
        pattern = re.sub(
            r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',
            '<UUID>',
            pattern,
            flags=re.IGNORECASE
        )
        # 替换IP地址
        pattern = re.sub(r'\d+\.\d+\.\d+\.\d+', '<IP>', pattern)
        # 替换路径
        pattern = re.sub(r'/[\w/.-]+', '<PATH>', pattern)

        return pattern[:200]

    async def analyze_trends(
        self,
        start_time: datetime,
        end_time: datetime,
        granularity_minutes: int = 60
    ) -> Dict[str, Any]:
        """分析日志趋势"""
        query = LogQuery(
            start_time=start_time,
            end_time=end_time,
            limit=100000
        )
        entries = await self.storage.query(query)

        # 按时间窗口分组
        buckets = defaultdict(lambda: defaultdict(int))

        for entry in entries:
            bucket_time = entry.timestamp.replace(
                minute=(entry.timestamp.minute // granularity_minutes) * granularity_minutes,
                second=0,
                microsecond=0
            )
            buckets[bucket_time]['total'] += 1
            buckets[bucket_time][entry.level.name] += 1

        # 转换为时间序列
        timeline = []
        for bucket_time in sorted(buckets.keys()):
            data = buckets[bucket_time]
            timeline.append({
                'timestamp': bucket_time.isoformat(),
                'total': data['total'],
                'debug': data.get('DEBUG', 0),
                'info': data.get('INFO', 0),
                'warning': data.get('WARNING', 0),
                'error': data.get('ERROR', 0),
                'critical': data.get('CRITICAL', 0)
            })

        return {
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'granularity_minutes': granularity_minutes,
            'timeline': timeline
        }

    async def detect_anomalies(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> List[Dict[str, Any]]:
        """检测日志异常"""
        trends = await self.analyze_trends(start_time, end_time, 5)
        timeline = trends['timeline']

        if len(timeline) < 10:
            return []

        anomalies = []

        # 计算基线
        total_counts = [t['total'] for t in timeline]
        error_counts = [t['error'] + t['critical'] for t in timeline]

        mean_total = sum(total_counts) / len(total_counts)
        std_total = (sum((x - mean_total) ** 2 for x in total_counts) / len(total_counts)) ** 0.5

        mean_error = sum(error_counts) / len(error_counts) if error_counts else 0
        std_error = (sum((x - mean_error) ** 2 for x in error_counts) / len(error_counts)) ** 0.5 \
            if error_counts else 0

        # 检测异常
        for i, point in enumerate(timeline):
            # 日志量突增
            if std_total > 0:
                z_total = (point['total'] - mean_total) / std_total
                if z_total > 3:
                    anomalies.append({
                        'type': 'volume_spike',
                        'timestamp': point['timestamp'],
                        'value': point['total'],
                        'baseline': mean_total,
                        'z_score': z_total
                    })

            # 错误率突增
            if std_error > 0:
                error_count = point['error'] + point['critical']
                z_error = (error_count - mean_error) / std_error
                if z_error > 3:
                    anomalies.append({
                        'type': 'error_spike',
                        'timestamp': point['timestamp'],
                        'value': error_count,
                        'baseline': mean_error,
                        'z_score': z_error
                    })

            # 日志量骤降（可能表示服务中断）
            if std_total > 0:
                z_total = (point['total'] - mean_total) / std_total
                if z_total < -2 and mean_total > 10:
                    anomalies.append({
                        'type': 'volume_drop',
                        'timestamp': point['timestamp'],
                        'value': point['total'],
                        'baseline': mean_total,
                        'z_score': z_total
                    })

        return anomalies


# ============================================================
# 告警管理器
# ============================================================

class AlertManager:
    """告警管理器"""

    def __init__(self):
        self.alerts: Dict[str, LogAlert] = {}
        self.triggered_alerts: List[Dict[str, Any]] = []
        self._lock = asyncio.Lock()

    async def add_alert(self, alert: LogAlert) -> str:
        """添加告警规则"""
        async with self._lock:
            self.alerts[alert.alert_id] = alert
            return alert.alert_id

    async def remove_alert(self, alert_id: str) -> bool:
        """移除告警规则"""
        async with self._lock:
            if alert_id in self.alerts:
                del self.alerts[alert_id]
                return True
            return False

    async def evaluate(self, entries: List[LogEntry]) -> List[Dict[str, Any]]:
        """评估日志，触发告警"""
        triggered = []

        for alert in self.alerts.values():
            if not alert.enabled:
                continue

            # 检查冷却时间
            if alert.last_triggered:
                elapsed = (datetime.now() - alert.last_triggered).total_seconds()
                if elapsed < alert.cooldown_seconds:
                    continue

            # 评估条件
            matching = self._evaluate_condition(alert.condition, entries)
            if matching:
                alert.last_triggered = datetime.now()
                alert.trigger_count += 1

                triggered_alert = {
                    'alert_id': alert.alert_id,
                    'name': alert.name,
                    'severity': alert.severity.value,
                    'message': self._render_message(alert.message_template, matching),
                    'triggered_at': datetime.now().isoformat(),
                    'matching_entries': len(matching)
                }
                triggered.append(triggered_alert)
                self.triggered_alerts.append(triggered_alert)

        return triggered

    def _evaluate_condition(
        self,
        condition: str,
        entries: List[LogEntry]
    ) -> List[LogEntry]:
        """评估条件"""
        matching = []

        # 简单条件解析
        if 'level:ERROR' in condition:
            matching = [e for e in entries if e.level == LogLevel.ERROR]
        elif 'level:CRITICAL' in condition:
            matching = [e for e in entries if e.level == LogLevel.CRITICAL]
        elif 'message:' in condition:
            keyword = condition.split('message:')[1].split()[0]
            matching = [e for e in entries if keyword.lower() in e.message.lower()]

        # 检查阈值
        threshold_match = re.search(r'count\s*>\s*(\d+)', condition)
        if threshold_match:
            threshold = int(threshold_match.group(1))
            if len(matching) <= threshold:
                return []

        return matching

    def _render_message(
        self,
        template: str,
        entries: List[LogEntry]
    ) -> str:
        """渲染告警消息"""
        message = template
        message = message.replace('${count}', str(len(entries)))
        if entries:
            message = message.replace('${first_message}', entries[0].message[:100])
            message = message.replace('${service}', entries[0].service)
        return message


# ============================================================
# 日志聚合服务
# ============================================================

class LogAggregationService:
    """日志聚合服务"""

    def __init__(
        self,
        storage: Optional[LogStorage] = None,
        buffer_size: int = 1000,
        flush_interval: float = 5.0
    ):
        self.storage = storage or InMemoryLogStorage()
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval

        self.parsers: Dict[str, LogParser] = {
            'json': JSONLogParser(),
            'syslog': SyslogParser(),
        }

        self.analyzer = LogAnalyzer(self.storage)
        self.alert_manager = AlertManager()

        self._buffer: List[LogEntry] = []
        self._running = False
        self._lock = asyncio.Lock()

    async def start(self):
        """启动服务"""
        self._running = True
        asyncio.create_task(self._flush_loop())
        logger.info("Log aggregation service started")

    async def stop(self):
        """停止服务"""
        self._running = False
        await self._flush()
        logger.info("Log aggregation service stopped")

    def register_parser(self, name: str, parser: LogParser):
        """注册解析器"""
        self.parsers[name] = parser

    async def ingest(
        self,
        raw_logs: List[str],
        parser_name: str = 'json',
        default_service: str = 'unknown'
    ) -> int:
        """摄入原始日志"""
        parser = self.parsers.get(parser_name)
        if not parser:
            raise ValueError(f"Unknown parser: {parser_name}")

        entries = []
        for raw in raw_logs:
            entry = parser.parse(raw)
            if entry:
                if not entry.service:
                    entry.service = default_service
                entries.append(entry)

        if entries:
            async with self._lock:
                self._buffer.extend(entries)

                # 检查是否需要刷新
                if len(self._buffer) >= self.buffer_size:
                    await self._flush()

            # 评估告警
            await self.alert_manager.evaluate(entries)

        return len(entries)

    async def log(
        self,
        level: LogLevel,
        message: str,
        service: str,
        source: LogSource = LogSource.APPLICATION,
        **kwargs
    ):
        """直接记录日志"""
        entry = LogEntry(
            log_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            level=level,
            source=source,
            service=service,
            message=message,
            **kwargs
        )

        async with self._lock:
            self._buffer.append(entry)

            if len(self._buffer) >= self.buffer_size:
                await self._flush()

    async def _flush_loop(self):
        """定时刷新循环"""
        while self._running:
            await asyncio.sleep(self.flush_interval)
            await self._flush()

    async def _flush(self):
        """刷新缓冲区"""
        async with self._lock:
            if not self._buffer:
                return

            entries = self._buffer.copy()
            self._buffer.clear()

        await self.storage.write(entries)

    async def search(self, query: LogQuery) -> List[LogEntry]:
        """搜索日志"""
        return await self.storage.query(query)

    async def get_stats(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> LogStats:
        """获取统计信息"""
        if start_time is None:
            start_time = datetime.now() - timedelta(hours=1)
        if end_time is None:
            end_time = datetime.now()

        return await self.storage.get_stats(start_time, end_time)

    async def analyze_errors(
        self,
        hours: int = 24
    ) -> Dict[str, Any]:
        """分析错误"""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        return await self.analyzer.analyze_errors(start_time, end_time)

    async def analyze_trends(
        self,
        hours: int = 24,
        granularity_minutes: int = 60
    ) -> Dict[str, Any]:
        """分析趋势"""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        return await self.analyzer.analyze_trends(
            start_time, end_time, granularity_minutes
        )

    async def detect_anomalies(
        self,
        hours: int = 24
    ) -> List[Dict[str, Any]]:
        """检测异常"""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        return await self.analyzer.detect_anomalies(start_time, end_time)


# ============================================================
# 工厂函数
# ============================================================

def create_cyrp_log_service(
    storage_type: str = 'memory',
    storage_path: str = '/var/log/cyrp',
    **kwargs
) -> LogAggregationService:
    """创建CYRP日志服务实例

    Args:
        storage_type: 存储类型 ('memory' 或 'file')
        storage_path: 文件存储路径
        **kwargs: 其他参数

    Returns:
        LogAggregationService: 日志服务实例
    """
    if storage_type == 'file':
        storage = FileLogStorage(storage_path)
    else:
        storage = InMemoryLogStorage(max_entries=kwargs.get('max_entries', 100000))

    service = LogAggregationService(
        storage=storage,
        buffer_size=kwargs.get('buffer_size', 1000),
        flush_interval=kwargs.get('flush_interval', 5.0)
    )

    # 添加默认告警规则
    default_alerts = [
        LogAlert(
            alert_id='critical_errors',
            name='严重错误告警',
            condition='level:CRITICAL count > 0',
            severity=AlertSeverity.CRITICAL,
            message_template='检测到 ${count} 条严重错误: ${first_message}'
        ),
        LogAlert(
            alert_id='error_spike',
            name='错误激增告警',
            condition='level:ERROR count > 10',
            severity=AlertSeverity.ERROR,
            message_template='服务 ${service} 错误激增，最近检测到 ${count} 条错误'
        ),
    ]

    async def setup_alerts():
        for alert in default_alerts:
            await service.alert_manager.add_alert(alert)

    try:
        loop = asyncio.get_running_loop()
        loop.create_task(setup_alerts())
    except RuntimeError:
        asyncio.run(setup_alerts())

    return service
