"""
Historical Database Module for CYRP
穿黄工程历史数据库模块

支持的存储后端:
- SQLite (内置，适合测试和小规模部署)
- InfluxDB (时序数据库，推荐生产环境)
- TimescaleDB (PostgreSQL扩展，高性能时序)
- 文件存储 (CSV/Parquet，归档用)
"""

import sqlite3
import json
import gzip
import os
import threading
import queue
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Iterator, Union
from datetime import datetime, timedelta
from enum import Enum, auto
import logging
from contextlib import contextmanager
import csv

logger = logging.getLogger(__name__)


# ============================================================================
# 数据模型
# ============================================================================

class DataType(Enum):
    """数据类型"""
    FLOAT = "float"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    STRING = "string"
    JSON = "json"


class AggregationType(Enum):
    """聚合类型"""
    MEAN = "mean"
    MIN = "min"
    MAX = "max"
    SUM = "sum"
    COUNT = "count"
    FIRST = "first"
    LAST = "last"
    STDDEV = "stddev"
    RANGE = "range"


@dataclass
class TimeSeriesPoint:
    """时序数据点"""
    timestamp: datetime
    value: Any
    quality: int = 0  # 0=good, 1=uncertain, 2=bad
    tags: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'value': self.value,
            'quality': self.quality,
            'tags': self.tags
        }


@dataclass
class TimeSeriesMetadata:
    """时序元数据"""
    name: str
    data_type: DataType
    unit: str = ""
    description: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    retention_days: int = 365
    downsample_rules: List[Tuple[str, str, AggregationType]] = field(default_factory=list)


@dataclass
class QueryResult:
    """查询结果"""
    series_name: str
    points: List[TimeSeriesPoint]
    metadata: Optional[TimeSeriesMetadata] = None
    aggregation: Optional[AggregationType] = None
    interval: Optional[str] = None

    def to_dataframe(self):
        """转换为DataFrame (需要pandas)"""
        try:
            import pandas as pd
            data = {
                'timestamp': [p.timestamp for p in self.points],
                'value': [p.value for p in self.points],
                'quality': [p.quality for p in self.points]
            }
            return pd.DataFrame(data).set_index('timestamp')
        except ImportError:
            logger.warning("pandas not installed, returning raw data")
            return self.points


# ============================================================================
# 存储后端抽象
# ============================================================================

class StorageBackend(ABC):
    """存储后端抽象基类"""

    @abstractmethod
    def connect(self) -> bool:
        """连接"""
        pass

    @abstractmethod
    def disconnect(self):
        """断开"""
        pass

    @abstractmethod
    def create_series(self, metadata: TimeSeriesMetadata) -> bool:
        """创建时序"""
        pass

    @abstractmethod
    def write(self, series_name: str, points: List[TimeSeriesPoint]) -> bool:
        """写入数据"""
        pass

    @abstractmethod
    def query(self, series_name: str, start_time: datetime,
              end_time: datetime, **kwargs) -> QueryResult:
        """查询数据"""
        pass

    @abstractmethod
    def delete(self, series_name: str, start_time: datetime,
               end_time: datetime) -> int:
        """删除数据"""
        pass


# ============================================================================
# SQLite 后端
# ============================================================================

class SQLiteBackend(StorageBackend):
    """SQLite存储后端"""

    def __init__(self, db_path: str = "cyrp_history.db"):
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None
        self.lock = threading.Lock()
        self._series_cache: Dict[str, TimeSeriesMetadata] = {}

    def connect(self) -> bool:
        """连接数据库"""
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            self._init_schema()
            self._load_series_cache()
            logger.info(f"SQLite connected: {self.db_path}")
            return True
        except Exception as e:
            logger.error(f"SQLite connection failed: {e}")
            return False

    def disconnect(self):
        """断开连接"""
        if self.conn:
            self.conn.close()
            self.conn = None

    def _init_schema(self):
        """初始化数据库结构"""
        with self.lock:
            cursor = self.conn.cursor()

            # 时序元数据表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS series_metadata (
                    name TEXT PRIMARY KEY,
                    data_type TEXT NOT NULL,
                    unit TEXT,
                    description TEXT,
                    tags TEXT,
                    retention_days INTEGER DEFAULT 365,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # 时序数据表 (分区按月)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS time_series_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    series_name TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    value REAL,
                    value_str TEXT,
                    quality INTEGER DEFAULT 0,
                    tags TEXT,
                    FOREIGN KEY (series_name) REFERENCES series_metadata(name)
                )
            ''')

            # 索引
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_ts_series_time
                ON time_series_data(series_name, timestamp)
            ''')

            # 报警历史表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alarm_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alarm_id TEXT NOT NULL,
                    alarm_type TEXT NOT NULL,
                    severity INTEGER NOT NULL,
                    source TEXT,
                    message TEXT,
                    occurred_at TIMESTAMP NOT NULL,
                    acknowledged_at TIMESTAMP,
                    acknowledged_by TEXT,
                    cleared_at TIMESTAMP,
                    tags TEXT
                )
            ''')

            # 事件日志表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS event_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    source TEXT,
                    message TEXT,
                    details TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # 场景历史表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS scenario_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scenario_id TEXT NOT NULL,
                    scenario_type TEXT,
                    start_time TIMESTAMP NOT NULL,
                    end_time TIMESTAMP,
                    duration_seconds REAL,
                    parameters TEXT,
                    results TEXT
                )
            ''')

            self.conn.commit()

    def _load_series_cache(self):
        """加载时序缓存"""
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute('SELECT * FROM series_metadata')
            for row in cursor.fetchall():
                self._series_cache[row['name']] = TimeSeriesMetadata(
                    name=row['name'],
                    data_type=DataType(row['data_type']),
                    unit=row['unit'] or "",
                    description=row['description'] or "",
                    tags=json.loads(row['tags']) if row['tags'] else {},
                    retention_days=row['retention_days']
                )

    def create_series(self, metadata: TimeSeriesMetadata) -> bool:
        """创建时序"""
        with self.lock:
            try:
                cursor = self.conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO series_metadata
                    (name, data_type, unit, description, tags, retention_days)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    metadata.name,
                    metadata.data_type.value,
                    metadata.unit,
                    metadata.description,
                    json.dumps(metadata.tags),
                    metadata.retention_days
                ))
                self.conn.commit()
                self._series_cache[metadata.name] = metadata
                return True
            except Exception as e:
                logger.error(f"Create series failed: {e}")
                return False

    def write(self, series_name: str, points: List[TimeSeriesPoint]) -> bool:
        """写入数据"""
        with self.lock:
            try:
                cursor = self.conn.cursor()

                # 获取数据类型
                metadata = self._series_cache.get(series_name)
                use_str = metadata and metadata.data_type in [DataType.STRING, DataType.JSON]

                for point in points:
                    if use_str:
                        cursor.execute('''
                            INSERT INTO time_series_data
                            (series_name, timestamp, value_str, quality, tags)
                            VALUES (?, ?, ?, ?, ?)
                        ''', (
                            series_name,
                            point.timestamp.isoformat(),
                            str(point.value) if metadata.data_type == DataType.STRING
                            else json.dumps(point.value),
                            point.quality,
                            json.dumps(point.tags) if point.tags else None
                        ))
                    else:
                        cursor.execute('''
                            INSERT INTO time_series_data
                            (series_name, timestamp, value, quality, tags)
                            VALUES (?, ?, ?, ?, ?)
                        ''', (
                            series_name,
                            point.timestamp.isoformat(),
                            float(point.value) if point.value is not None else None,
                            point.quality,
                            json.dumps(point.tags) if point.tags else None
                        ))

                self.conn.commit()
                return True
            except Exception as e:
                logger.error(f"Write failed: {e}")
                return False

    def query(self, series_name: str, start_time: datetime,
              end_time: datetime, **kwargs) -> QueryResult:
        """查询数据"""
        aggregation = kwargs.get('aggregation')
        interval = kwargs.get('interval')
        limit = kwargs.get('limit', 10000)

        with self.lock:
            cursor = self.conn.cursor()

            if aggregation and interval:
                # 聚合查询
                points = self._aggregate_query(
                    cursor, series_name, start_time, end_time,
                    aggregation, interval
                )
            else:
                # 原始查询
                cursor.execute('''
                    SELECT timestamp, value, value_str, quality, tags
                    FROM time_series_data
                    WHERE series_name = ? AND timestamp >= ? AND timestamp <= ?
                    ORDER BY timestamp
                    LIMIT ?
                ''', (series_name, start_time.isoformat(), end_time.isoformat(), limit))

                points = []
                metadata = self._series_cache.get(series_name)
                use_str = metadata and metadata.data_type in [DataType.STRING, DataType.JSON]

                for row in cursor.fetchall():
                    value = row['value_str'] if use_str else row['value']
                    if metadata and metadata.data_type == DataType.JSON and value:
                        value = json.loads(value)

                    points.append(TimeSeriesPoint(
                        timestamp=datetime.fromisoformat(row['timestamp']),
                        value=value,
                        quality=row['quality'],
                        tags=json.loads(row['tags']) if row['tags'] else {}
                    ))

            return QueryResult(
                series_name=series_name,
                points=points,
                metadata=self._series_cache.get(series_name),
                aggregation=AggregationType(aggregation) if aggregation else None,
                interval=interval
            )

    def _aggregate_query(self, cursor, series_name: str, start_time: datetime,
                        end_time: datetime, aggregation: str, interval: str) -> List[TimeSeriesPoint]:
        """聚合查询"""
        # SQLite的时间聚合
        interval_sql = {
            '1m': "strftime('%Y-%m-%d %H:%M:00', timestamp)",
            '5m': "strftime('%Y-%m-%d %H:', timestamp) || (cast(strftime('%M', timestamp) as integer) / 5 * 5)",
            '15m': "strftime('%Y-%m-%d %H:', timestamp) || (cast(strftime('%M', timestamp) as integer) / 15 * 15)",
            '1h': "strftime('%Y-%m-%d %H:00:00', timestamp)",
            '1d': "strftime('%Y-%m-%d 00:00:00', timestamp)",
        }.get(interval, "strftime('%Y-%m-%d %H:00:00', timestamp)")

        agg_sql = {
            'mean': 'AVG(value)',
            'min': 'MIN(value)',
            'max': 'MAX(value)',
            'sum': 'SUM(value)',
            'count': 'COUNT(*)',
            'first': 'value',  # 需要特殊处理
            'last': 'value',   # 需要特殊处理
        }.get(aggregation, 'AVG(value)')

        cursor.execute(f'''
            SELECT {interval_sql} as time_bucket, {agg_sql} as agg_value
            FROM time_series_data
            WHERE series_name = ? AND timestamp >= ? AND timestamp <= ?
            GROUP BY time_bucket
            ORDER BY time_bucket
        ''', (series_name, start_time.isoformat(), end_time.isoformat()))

        points = []
        for row in cursor.fetchall():
            try:
                ts = datetime.fromisoformat(row['time_bucket'])
                points.append(TimeSeriesPoint(
                    timestamp=ts,
                    value=row['agg_value'],
                    quality=0
                ))
            except:
                pass

        return points

    def delete(self, series_name: str, start_time: datetime,
               end_time: datetime) -> int:
        """删除数据"""
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute('''
                DELETE FROM time_series_data
                WHERE series_name = ? AND timestamp >= ? AND timestamp <= ?
            ''', (series_name, start_time.isoformat(), end_time.isoformat()))
            self.conn.commit()
            return cursor.rowcount

    def cleanup_old_data(self) -> int:
        """清理过期数据"""
        deleted = 0
        with self.lock:
            cursor = self.conn.cursor()
            for name, metadata in self._series_cache.items():
                cutoff = datetime.now() - timedelta(days=metadata.retention_days)
                cursor.execute('''
                    DELETE FROM time_series_data
                    WHERE series_name = ? AND timestamp < ?
                ''', (name, cutoff.isoformat()))
                deleted += cursor.rowcount
            self.conn.commit()
        return deleted


# ============================================================================
# InfluxDB 后端 (模拟实现)
# ============================================================================

@dataclass
class InfluxDBConfig:
    """InfluxDB配置"""
    url: str = "http://localhost:8086"
    token: str = ""
    org: str = "cyrp"
    bucket: str = "cyrp_data"
    timeout: int = 10000


class InfluxDBBackend(StorageBackend):
    """InfluxDB存储后端"""

    def __init__(self, config: InfluxDBConfig):
        self.config = config
        self._connected = False
        self._write_queue: queue.Queue = queue.Queue()
        self._series_cache: Dict[str, TimeSeriesMetadata] = {}

    def connect(self) -> bool:
        """连接InfluxDB"""
        try:
            # 注意：实际需要使用 influxdb-client 库
            logger.info(f"InfluxDB connecting to {self.config.url}")
            self._connected = True
            return True
        except Exception as e:
            logger.error(f"InfluxDB connection failed: {e}")
            return False

    def disconnect(self):
        """断开连接"""
        self._connected = False

    def create_series(self, metadata: TimeSeriesMetadata) -> bool:
        """创建时序（InfluxDB自动创建）"""
        self._series_cache[metadata.name] = metadata
        return True

    def write(self, series_name: str, points: List[TimeSeriesPoint]) -> bool:
        """写入数据"""
        if not self._connected:
            return False

        try:
            # 模拟写入 Line Protocol
            lines = []
            for point in points:
                tags_str = ",".join(f"{k}={v}" for k, v in point.tags.items())
                if tags_str:
                    line = f"{series_name},{tags_str} value={point.value} {int(point.timestamp.timestamp() * 1e9)}"
                else:
                    line = f"{series_name} value={point.value} {int(point.timestamp.timestamp() * 1e9)}"
                lines.append(line)

            logger.debug(f"InfluxDB write: {len(lines)} points to {series_name}")
            return True
        except Exception as e:
            logger.error(f"InfluxDB write failed: {e}")
            return False

    def query(self, series_name: str, start_time: datetime,
              end_time: datetime, **kwargs) -> QueryResult:
        """查询数据"""
        # 模拟查询
        return QueryResult(
            series_name=series_name,
            points=[],
            metadata=self._series_cache.get(series_name)
        )

    def delete(self, series_name: str, start_time: datetime,
               end_time: datetime) -> int:
        """删除数据"""
        return 0


# ============================================================================
# 历史数据管理器
# ============================================================================

class HistorianManager:
    """历史数据管理器"""

    def __init__(self, backend: StorageBackend):
        self.backend = backend
        self._write_buffer: Dict[str, List[TimeSeriesPoint]] = {}
        self._buffer_lock = threading.Lock()
        self._flush_interval = 5.0
        self._max_buffer_size = 1000
        self._running = False
        self._flush_thread: Optional[threading.Thread] = None

    def start(self) -> bool:
        """启动管理器"""
        if not self.backend.connect():
            return False

        self._running = True
        self._flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._flush_thread.start()
        logger.info("Historian manager started")
        return True

    def stop(self):
        """停止管理器"""
        self._running = False
        self._flush_all()
        if self._flush_thread:
            self._flush_thread.join(timeout=5.0)
        self.backend.disconnect()
        logger.info("Historian manager stopped")

    def register_series(self, metadata: TimeSeriesMetadata) -> bool:
        """注册时序"""
        return self.backend.create_series(metadata)

    def write(self, series_name: str, value: Any,
              timestamp: Optional[datetime] = None,
              quality: int = 0,
              tags: Optional[Dict[str, str]] = None):
        """写入单个数据点"""
        point = TimeSeriesPoint(
            timestamp=timestamp or datetime.now(),
            value=value,
            quality=quality,
            tags=tags or {}
        )

        with self._buffer_lock:
            if series_name not in self._write_buffer:
                self._write_buffer[series_name] = []
            self._write_buffer[series_name].append(point)

            # 检查是否需要刷新
            if len(self._write_buffer[series_name]) >= self._max_buffer_size:
                self._flush_series(series_name)

    def write_batch(self, series_name: str, points: List[TimeSeriesPoint]):
        """批量写入"""
        with self._buffer_lock:
            if series_name not in self._write_buffer:
                self._write_buffer[series_name] = []
            self._write_buffer[series_name].extend(points)

            if len(self._write_buffer[series_name]) >= self._max_buffer_size:
                self._flush_series(series_name)

    def query(self, series_name: str,
              start_time: Optional[datetime] = None,
              end_time: Optional[datetime] = None,
              aggregation: Optional[str] = None,
              interval: Optional[str] = None,
              limit: int = 10000) -> QueryResult:
        """查询数据"""
        if start_time is None:
            start_time = datetime.now() - timedelta(hours=1)
        if end_time is None:
            end_time = datetime.now()

        return self.backend.query(
            series_name, start_time, end_time,
            aggregation=aggregation,
            interval=interval,
            limit=limit
        )

    def query_latest(self, series_name: str, count: int = 1) -> List[TimeSeriesPoint]:
        """查询最新数据"""
        result = self.backend.query(
            series_name,
            datetime.now() - timedelta(days=1),
            datetime.now(),
            limit=count
        )
        return result.points[-count:] if result.points else []

    def _flush_loop(self):
        """刷新循环"""
        while self._running:
            time.sleep(self._flush_interval)
            self._flush_all()

    def _flush_all(self):
        """刷新所有缓冲"""
        with self._buffer_lock:
            for series_name in list(self._write_buffer.keys()):
                self._flush_series(series_name)

    def _flush_series(self, series_name: str):
        """刷新指定时序"""
        points = self._write_buffer.get(series_name, [])
        if points:
            if self.backend.write(series_name, points):
                self._write_buffer[series_name] = []
            else:
                logger.error(f"Failed to flush {series_name}")


# ============================================================================
# 数据归档
# ============================================================================

class DataArchiver:
    """数据归档器"""

    def __init__(self, backend: StorageBackend, archive_path: str = "./archive"):
        self.backend = backend
        self.archive_path = archive_path
        os.makedirs(archive_path, exist_ok=True)

    def archive_to_csv(self, series_name: str, start_time: datetime,
                       end_time: datetime, compress: bool = True) -> str:
        """归档到CSV"""
        result = self.backend.query(series_name, start_time, end_time, limit=1000000)

        filename = f"{series_name}_{start_time.strftime('%Y%m%d')}_{end_time.strftime('%Y%m%d')}.csv"
        if compress:
            filename += ".gz"

        filepath = os.path.join(self.archive_path, filename)

        if compress:
            with gzip.open(filepath, 'wt', newline='') as f:
                self._write_csv(f, result.points)
        else:
            with open(filepath, 'w', newline='') as f:
                self._write_csv(f, result.points)

        logger.info(f"Archived {len(result.points)} points to {filepath}")
        return filepath

    def _write_csv(self, f, points: List[TimeSeriesPoint]):
        """写入CSV"""
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'value', 'quality', 'tags'])
        for p in points:
            writer.writerow([
                p.timestamp.isoformat(),
                p.value,
                p.quality,
                json.dumps(p.tags) if p.tags else ''
            ])

    def archive_to_parquet(self, series_name: str, start_time: datetime,
                           end_time: datetime) -> Optional[str]:
        """归档到Parquet（需要pyarrow）"""
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            logger.warning("pyarrow not installed")
            return None

        result = self.backend.query(series_name, start_time, end_time, limit=1000000)

        filename = f"{series_name}_{start_time.strftime('%Y%m%d')}_{end_time.strftime('%Y%m%d')}.parquet"
        filepath = os.path.join(self.archive_path, filename)

        table = pa.table({
            'timestamp': [p.timestamp for p in result.points],
            'value': [p.value for p in result.points],
            'quality': [p.quality for p in result.points],
        })

        pq.write_table(table, filepath, compression='snappy')
        logger.info(f"Archived {len(result.points)} points to {filepath}")
        return filepath


# ============================================================================
# 便捷函数
# ============================================================================

def create_cyrp_historian(db_path: str = "cyrp_history.db") -> HistorianManager:
    """创建穿黄工程历史数据管理器"""
    backend = SQLiteBackend(db_path)
    manager = HistorianManager(backend)

    # 注册常用时序
    series_list = [
        TimeSeriesMetadata("inlet_flow", DataType.FLOAT, "m³/s", "进口流量"),
        TimeSeriesMetadata("outlet_flow", DataType.FLOAT, "m³/s", "出口流量"),
        TimeSeriesMetadata("inlet_pressure", DataType.FLOAT, "kPa", "进口压力"),
        TimeSeriesMetadata("outlet_pressure", DataType.FLOAT, "kPa", "出口压力"),
        TimeSeriesMetadata("water_level_inlet", DataType.FLOAT, "m", "进口水位"),
        TimeSeriesMetadata("water_level_outlet", DataType.FLOAT, "m", "出口水位"),
        TimeSeriesMetadata("inlet_valve_position", DataType.FLOAT, "%", "进口阀门开度"),
        TimeSeriesMetadata("outlet_valve_position", DataType.FLOAT, "%", "出口阀门开度"),
        TimeSeriesMetadata("water_temperature", DataType.FLOAT, "°C", "水温"),
        TimeSeriesMetadata("ambient_temperature", DataType.FLOAT, "°C", "环境温度"),
        TimeSeriesMetadata("tunnel1_flow", DataType.FLOAT, "m³/s", "1号隧洞流量"),
        TimeSeriesMetadata("tunnel2_flow", DataType.FLOAT, "m³/s", "2号隧洞流量"),
        TimeSeriesMetadata("leakage_rate", DataType.FLOAT, "L/s", "渗漏量"),
        TimeSeriesMetadata("vibration_max", DataType.FLOAT, "mm/s", "最大振动"),
        TimeSeriesMetadata("scenario_id", DataType.STRING, "", "当前场景"),
        TimeSeriesMetadata("control_mode", DataType.STRING, "", "控制模式"),
    ]

    if manager.start():
        for series in series_list:
            manager.register_series(series)

    return manager


# 导入时间模块
import time
