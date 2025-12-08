"""
Database Module for CYRP
穿黄工程数据库模块

支持:
- 时序数据存储
- 告警历史记录
- 状态快照管理
- 数据归档与清理
"""

from cyrp.database.historian import (
    DataType,
    AggregationType,
    TimeSeriesPoint,
    TimeSeriesMetadata,
    QueryResult,
    StorageBackend,
    SQLiteBackend,
    InfluxDBConfig,
    InfluxDBBackend,
    HistorianManager,
    DataArchiver as HistorianArchiver,
    create_cyrp_historian,
)

from cyrp.database.persistence_manager import (
    PersistenceLevel,
    PersistenceConfig,
    MetricPersistenceRule,
    MetricBuffer,
    AlertPersistence,
    StatusSnapshotManager,
    DataArchiver,
    PersistenceManager,
    create_persistence_system,
)

__all__ = [
    # 历史数据存储
    "DataType",
    "AggregationType",
    "TimeSeriesPoint",
    "TimeSeriesMetadata",
    "QueryResult",
    "StorageBackend",
    "SQLiteBackend",
    "InfluxDBConfig",
    "InfluxDBBackend",
    "HistorianManager",
    "HistorianArchiver",
    "create_cyrp_historian",
    # 持久化管理
    "PersistenceLevel",
    "PersistenceConfig",
    "MetricPersistenceRule",
    "MetricBuffer",
    "AlertPersistence",
    "StatusSnapshotManager",
    "DataArchiver",
    "PersistenceManager",
    "create_persistence_system",
]
