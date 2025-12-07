"""
Data Backup and Synchronization System for CYRP
穿黄工程数据备份与同步系统

功能:
- 自动定时备份
- 增量/全量备份
- 多目标存储(本地/远程/云)
- 数据同步与复制
- 备份验证与恢复
- 备份策略管理
"""

import asyncio
import gzip
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import sqlite3


class BackupType(Enum):
    """备份类型"""
    FULL = auto()           # 全量备份
    INCREMENTAL = auto()    # 增量备份
    DIFFERENTIAL = auto()   # 差异备份
    SNAPSHOT = auto()       # 快照备份


class BackupStatus(Enum):
    """备份状态"""
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    VERIFIED = auto()
    CORRUPTED = auto()


class StorageType(Enum):
    """存储类型"""
    LOCAL = auto()          # 本地存储
    NFS = auto()            # 网络文件系统
    S3 = auto()             # Amazon S3兼容存储
    FTP = auto()            # FTP服务器
    SFTP = auto()           # SFTP服务器


@dataclass
class BackupMetadata:
    """备份元数据"""
    backup_id: str
    backup_type: BackupType
    created_at: datetime
    completed_at: Optional[datetime] = None
    status: BackupStatus = BackupStatus.PENDING
    size_bytes: int = 0
    file_count: int = 0
    checksum: str = ""
    source_path: str = ""
    destination_path: str = ""
    parent_backup_id: Optional[str] = None  # 增量备份的父备份
    retention_days: int = 30
    compression: bool = True
    encryption: bool = False
    tags: Dict[str, str] = field(default_factory=dict)
    error_message: str = ""


@dataclass
class BackupPolicy:
    """备份策略"""
    policy_id: str
    name: str
    description: str = ""
    enabled: bool = True
    backup_type: BackupType = BackupType.FULL
    schedule_cron: str = "0 2 * * *"  # 每天凌晨2点
    retention_days: int = 30
    retention_count: int = 10  # 保留最近N个备份
    source_paths: List[str] = field(default_factory=list)
    exclude_patterns: List[str] = field(default_factory=list)
    destination: str = ""
    compression: bool = True
    encryption: bool = False
    verify_after_backup: bool = True
    notify_on_failure: bool = True
    notify_on_success: bool = False


@dataclass
class SyncConfig:
    """同步配置"""
    sync_id: str
    name: str
    source: str
    destination: str
    enabled: bool = True
    direction: str = "push"  # push, pull, bidirectional
    interval_seconds: int = 300
    delete_orphans: bool = False  # 删除目标中源不存在的文件
    exclude_patterns: List[str] = field(default_factory=list)
    conflict_resolution: str = "source_wins"  # source_wins, dest_wins, newest, manual


class StorageBackend(ABC):
    """存储后端基类"""

    @abstractmethod
    async def upload(self, local_path: str, remote_path: str) -> bool:
        """上传文件"""
        pass

    @abstractmethod
    async def download(self, remote_path: str, local_path: str) -> bool:
        """下载文件"""
        pass

    @abstractmethod
    async def delete(self, remote_path: str) -> bool:
        """删除文件"""
        pass

    @abstractmethod
    async def list_files(self, path: str) -> List[str]:
        """列出文件"""
        pass

    @abstractmethod
    async def exists(self, path: str) -> bool:
        """检查文件是否存在"""
        pass

    @abstractmethod
    async def get_size(self, path: str) -> int:
        """获取文件大小"""
        pass


class LocalStorageBackend(StorageBackend):
    """本地存储后端"""

    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    async def upload(self, local_path: str, remote_path: str) -> bool:
        """复制文件到目标路径"""
        try:
            dest = self.base_path / remote_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local_path, dest)
            return True
        except Exception:
            return False

    async def download(self, remote_path: str, local_path: str) -> bool:
        """从目标路径复制文件"""
        try:
            src = self.base_path / remote_path
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, local_path)
            return True
        except Exception:
            return False

    async def delete(self, remote_path: str) -> bool:
        """删除文件"""
        try:
            path = self.base_path / remote_path
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)
            return True
        except Exception:
            return False

    async def list_files(self, path: str = "") -> List[str]:
        """列出文件"""
        try:
            target = self.base_path / path if path else self.base_path
            files = []
            for item in target.rglob("*"):
                if item.is_file():
                    files.append(str(item.relative_to(self.base_path)))
            return files
        except Exception:
            return []

    async def exists(self, path: str) -> bool:
        """检查文件是否存在"""
        return (self.base_path / path).exists()

    async def get_size(self, path: str) -> int:
        """获取文件大小"""
        try:
            target = self.base_path / path
            if target.is_file():
                return target.stat().st_size
            elif target.is_dir():
                return sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
            return 0
        except Exception:
            return 0


class S3StorageBackend(StorageBackend):
    """S3兼容存储后端"""

    def __init__(
        self,
        bucket: str,
        endpoint: str = "",
        access_key: str = "",
        secret_key: str = "",
        region: str = "us-east-1"
    ):
        self.bucket = bucket
        self.endpoint = endpoint
        self.access_key = access_key
        self.secret_key = secret_key
        self.region = region
        # 实际实现需要boto3库

    async def upload(self, local_path: str, remote_path: str) -> bool:
        """上传到S3"""
        # 模拟实现
        print(f"[S3] 上传 {local_path} -> s3://{self.bucket}/{remote_path}")
        return True

    async def download(self, remote_path: str, local_path: str) -> bool:
        """从S3下载"""
        print(f"[S3] 下载 s3://{self.bucket}/{remote_path} -> {local_path}")
        return True

    async def delete(self, remote_path: str) -> bool:
        """删除S3对象"""
        print(f"[S3] 删除 s3://{self.bucket}/{remote_path}")
        return True

    async def list_files(self, path: str = "") -> List[str]:
        """列出S3对象"""
        return []

    async def exists(self, path: str) -> bool:
        """检查对象是否存在"""
        return False

    async def get_size(self, path: str) -> int:
        """获取对象大小"""
        return 0


class FileHasher:
    """文件哈希计算器"""

    @staticmethod
    def calculate_file_hash(file_path: str, algorithm: str = "sha256") -> str:
        """计算文件哈希"""
        hash_func = hashlib.new(algorithm)
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hash_func.update(chunk)
        return hash_func.hexdigest()

    @staticmethod
    def calculate_directory_hash(dir_path: str) -> Dict[str, str]:
        """计算目录中所有文件的哈希"""
        hashes = {}
        for root, _, files in os.walk(dir_path):
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, dir_path)
                hashes[rel_path] = FileHasher.calculate_file_hash(file_path)
        return hashes


class BackupEngine:
    """备份引擎"""

    def __init__(self, storage: StorageBackend):
        self.storage = storage
        self._backup_history: Dict[str, BackupMetadata] = {}

    async def create_backup(
        self,
        source_paths: List[str],
        backup_type: BackupType = BackupType.FULL,
        exclude_patterns: Optional[List[str]] = None,
        compression: bool = True,
        parent_backup_id: Optional[str] = None
    ) -> BackupMetadata:
        """创建备份"""
        backup_id = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        exclude_patterns = exclude_patterns or []

        metadata = BackupMetadata(
            backup_id=backup_id,
            backup_type=backup_type,
            created_at=datetime.now(),
            status=BackupStatus.RUNNING,
            source_path=",".join(source_paths),
            compression=compression,
            parent_backup_id=parent_backup_id
        )

        try:
            # 创建临时目录
            with tempfile.TemporaryDirectory() as temp_dir:
                backup_dir = Path(temp_dir) / backup_id

                # 收集要备份的文件
                if backup_type == BackupType.FULL:
                    files_to_backup = await self._collect_files(
                        source_paths, exclude_patterns
                    )
                elif backup_type == BackupType.INCREMENTAL:
                    files_to_backup = await self._collect_incremental_files(
                        source_paths, exclude_patterns, parent_backup_id
                    )
                else:
                    files_to_backup = await self._collect_files(
                        source_paths, exclude_patterns
                    )

                # 复制文件
                file_count = 0
                for src, rel_path in files_to_backup:
                    dest = backup_dir / rel_path
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dest)
                    file_count += 1

                metadata.file_count = file_count

                # 保存元数据
                metadata_file = backup_dir / "metadata.json"
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "backup_id": backup_id,
                        "backup_type": backup_type.name,
                        "created_at": metadata.created_at.isoformat(),
                        "source_paths": source_paths,
                        "file_count": file_count,
                        "parent_backup_id": parent_backup_id,
                    }, f, indent=2, ensure_ascii=False)

                # 创建归档
                if compression:
                    archive_path = f"{temp_dir}/{backup_id}.tar.gz"
                    with tarfile.open(archive_path, "w:gz") as tar:
                        tar.add(backup_dir, arcname=backup_id)
                else:
                    archive_path = f"{temp_dir}/{backup_id}.tar"
                    with tarfile.open(archive_path, "w") as tar:
                        tar.add(backup_dir, arcname=backup_id)

                # 计算校验和
                metadata.checksum = FileHasher.calculate_file_hash(archive_path)
                metadata.size_bytes = os.path.getsize(archive_path)

                # 上传到存储后端
                remote_path = f"backups/{backup_id}/{os.path.basename(archive_path)}"
                success = await self.storage.upload(archive_path, remote_path)

                if success:
                    metadata.destination_path = remote_path
                    metadata.status = BackupStatus.COMPLETED
                    metadata.completed_at = datetime.now()
                else:
                    metadata.status = BackupStatus.FAILED
                    metadata.error_message = "上传失败"

        except Exception as e:
            metadata.status = BackupStatus.FAILED
            metadata.error_message = str(e)

        self._backup_history[backup_id] = metadata
        return metadata

    async def _collect_files(
        self,
        source_paths: List[str],
        exclude_patterns: List[str]
    ) -> List[Tuple[str, str]]:
        """收集要备份的文件"""
        files = []
        for source in source_paths:
            source_path = Path(source)
            if source_path.is_file():
                files.append((str(source_path), source_path.name))
            elif source_path.is_dir():
                for file_path in source_path.rglob("*"):
                    if file_path.is_file():
                        rel_path = file_path.relative_to(source_path.parent)
                        # 检查排除模式
                        excluded = False
                        for pattern in exclude_patterns:
                            if file_path.match(pattern):
                                excluded = True
                                break
                        if not excluded:
                            files.append((str(file_path), str(rel_path)))
        return files

    async def _collect_incremental_files(
        self,
        source_paths: List[str],
        exclude_patterns: List[str],
        parent_backup_id: Optional[str]
    ) -> List[Tuple[str, str]]:
        """收集增量备份文件(只包含变化的文件)"""
        all_files = await self._collect_files(source_paths, exclude_patterns)

        if not parent_backup_id or parent_backup_id not in self._backup_history:
            return all_files

        # 获取父备份的文件哈希
        parent_hashes = await self._get_backup_file_hashes(parent_backup_id)

        # 只保留变化的文件
        changed_files = []
        for src, rel_path in all_files:
            current_hash = FileHasher.calculate_file_hash(src)
            if rel_path not in parent_hashes or parent_hashes[rel_path] != current_hash:
                changed_files.append((src, rel_path))

        return changed_files

    async def _get_backup_file_hashes(self, backup_id: str) -> Dict[str, str]:
        """获取备份文件的哈希值"""
        # 实际实现需要从备份中读取或从数据库获取
        return {}

    async def restore_backup(
        self,
        backup_id: str,
        restore_path: str,
        overwrite: bool = False
    ) -> bool:
        """恢复备份"""
        if backup_id not in self._backup_history:
            return False

        metadata = self._backup_history[backup_id]

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # 下载备份文件
                local_archive = f"{temp_dir}/{os.path.basename(metadata.destination_path)}"
                success = await self.storage.download(
                    metadata.destination_path, local_archive
                )

                if not success:
                    return False

                # 验证校验和
                if metadata.checksum:
                    current_checksum = FileHasher.calculate_file_hash(local_archive)
                    if current_checksum != metadata.checksum:
                        return False

                # 解压
                extract_dir = f"{temp_dir}/extracted"
                with tarfile.open(local_archive, "r:*") as tar:
                    tar.extractall(extract_dir)

                # 恢复文件
                backup_content = Path(extract_dir) / backup_id
                restore_dest = Path(restore_path)

                if restore_dest.exists() and not overwrite:
                    return False

                restore_dest.mkdir(parents=True, exist_ok=True)

                for item in backup_content.rglob("*"):
                    if item.is_file() and item.name != "metadata.json":
                        rel_path = item.relative_to(backup_content)
                        dest = restore_dest / rel_path
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(item, dest)

                return True

        except Exception:
            return False

    async def verify_backup(self, backup_id: str) -> bool:
        """验证备份完整性"""
        if backup_id not in self._backup_history:
            return False

        metadata = self._backup_history[backup_id]

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                local_archive = f"{temp_dir}/verify_{backup_id}"
                success = await self.storage.download(
                    metadata.destination_path, local_archive
                )

                if not success:
                    metadata.status = BackupStatus.CORRUPTED
                    return False

                # 验证校验和
                current_checksum = FileHasher.calculate_file_hash(local_archive)
                if current_checksum != metadata.checksum:
                    metadata.status = BackupStatus.CORRUPTED
                    return False

                # 验证可以解压
                with tarfile.open(local_archive, "r:*") as tar:
                    tar.getmembers()

                metadata.status = BackupStatus.VERIFIED
                return True

        except Exception:
            metadata.status = BackupStatus.CORRUPTED
            return False

    async def delete_backup(self, backup_id: str) -> bool:
        """删除备份"""
        if backup_id not in self._backup_history:
            return False

        metadata = self._backup_history[backup_id]

        try:
            success = await self.storage.delete(
                os.path.dirname(metadata.destination_path)
            )
            if success:
                del self._backup_history[backup_id]
            return success
        except Exception:
            return False

    def list_backups(
        self,
        backup_type: Optional[BackupType] = None,
        status: Optional[BackupStatus] = None,
        since: Optional[datetime] = None
    ) -> List[BackupMetadata]:
        """列出备份"""
        backups = list(self._backup_history.values())

        if backup_type:
            backups = [b for b in backups if b.backup_type == backup_type]
        if status:
            backups = [b for b in backups if b.status == status]
        if since:
            backups = [b for b in backups if b.created_at >= since]

        return sorted(backups, key=lambda b: b.created_at, reverse=True)


class DataSynchronizer:
    """数据同步器"""

    def __init__(self):
        self._sync_configs: Dict[str, SyncConfig] = {}
        self._sync_status: Dict[str, Dict[str, Any]] = {}
        self._running = False
        self._sync_tasks: Dict[str, asyncio.Task] = {}

    def add_sync_config(self, config: SyncConfig):
        """添加同步配置"""
        self._sync_configs[config.sync_id] = config
        self._sync_status[config.sync_id] = {
            "last_sync": None,
            "status": "idle",
            "files_synced": 0,
            "errors": []
        }

    def remove_sync_config(self, sync_id: str):
        """移除同步配置"""
        if sync_id in self._sync_configs:
            del self._sync_configs[sync_id]
        if sync_id in self._sync_status:
            del self._sync_status[sync_id]

    async def start(self):
        """启动同步服务"""
        self._running = True
        for sync_id, config in self._sync_configs.items():
            if config.enabled:
                self._sync_tasks[sync_id] = asyncio.create_task(
                    self._sync_loop(sync_id, config)
                )

    async def stop(self):
        """停止同步服务"""
        self._running = False
        for task in self._sync_tasks.values():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._sync_tasks.clear()

    async def _sync_loop(self, sync_id: str, config: SyncConfig):
        """同步循环"""
        while self._running:
            try:
                await self.sync_now(sync_id)
            except Exception as e:
                self._sync_status[sync_id]["errors"].append({
                    "time": datetime.now().isoformat(),
                    "error": str(e)
                })

            await asyncio.sleep(config.interval_seconds)

    async def sync_now(self, sync_id: str) -> Dict[str, Any]:
        """立即执行同步"""
        if sync_id not in self._sync_configs:
            return {"error": "配置不存在"}

        config = self._sync_configs[sync_id]
        status = self._sync_status[sync_id]
        status["status"] = "syncing"

        try:
            result = await self._perform_sync(config)
            status["last_sync"] = datetime.now()
            status["status"] = "completed"
            status["files_synced"] = result.get("files_synced", 0)
            return result
        except Exception as e:
            status["status"] = "error"
            status["errors"].append({
                "time": datetime.now().isoformat(),
                "error": str(e)
            })
            return {"error": str(e)}

    async def _perform_sync(self, config: SyncConfig) -> Dict[str, Any]:
        """执行同步"""
        source_path = Path(config.source)
        dest_path = Path(config.destination)

        if not source_path.exists():
            return {"error": "源路径不存在"}

        dest_path.mkdir(parents=True, exist_ok=True)

        files_synced = 0
        files_deleted = 0

        # 获取源文件列表
        source_files = {}
        for file_path in source_path.rglob("*"):
            if file_path.is_file():
                rel_path = file_path.relative_to(source_path)
                # 检查排除模式
                excluded = False
                for pattern in config.exclude_patterns:
                    if file_path.match(pattern):
                        excluded = True
                        break
                if not excluded:
                    source_files[str(rel_path)] = {
                        "path": file_path,
                        "mtime": file_path.stat().st_mtime,
                        "size": file_path.stat().st_size
                    }

        # 获取目标文件列表
        dest_files = {}
        for file_path in dest_path.rglob("*"):
            if file_path.is_file():
                rel_path = file_path.relative_to(dest_path)
                dest_files[str(rel_path)] = {
                    "path": file_path,
                    "mtime": file_path.stat().st_mtime,
                    "size": file_path.stat().st_size
                }

        # 同步文件
        for rel_path, src_info in source_files.items():
            dest_file = dest_path / rel_path
            need_copy = False

            if rel_path not in dest_files:
                need_copy = True
            else:
                dest_info = dest_files[rel_path]
                if src_info["mtime"] > dest_info["mtime"]:
                    if config.conflict_resolution == "source_wins":
                        need_copy = True
                    elif config.conflict_resolution == "newest":
                        need_copy = True

            if need_copy:
                dest_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_info["path"], dest_file)
                files_synced += 1

        # 删除孤立文件
        if config.delete_orphans:
            for rel_path in dest_files:
                if rel_path not in source_files:
                    (dest_path / rel_path).unlink()
                    files_deleted += 1

        return {
            "files_synced": files_synced,
            "files_deleted": files_deleted,
            "source_count": len(source_files),
            "dest_count": len(dest_files)
        }

    def get_sync_status(self, sync_id: Optional[str] = None) -> Dict[str, Any]:
        """获取同步状态"""
        if sync_id:
            return self._sync_status.get(sync_id, {})
        return self._sync_status.copy()


class BackupScheduler:
    """备份调度器"""

    def __init__(self, backup_engine: BackupEngine):
        self.backup_engine = backup_engine
        self._policies: Dict[str, BackupPolicy] = {}
        self._running = False
        self._scheduler_task: Optional[asyncio.Task] = None
        self._next_runs: Dict[str, datetime] = {}

    def add_policy(self, policy: BackupPolicy):
        """添加备份策略"""
        self._policies[policy.policy_id] = policy
        self._calculate_next_run(policy.policy_id)

    def remove_policy(self, policy_id: str):
        """移除备份策略"""
        if policy_id in self._policies:
            del self._policies[policy_id]
        if policy_id in self._next_runs:
            del self._next_runs[policy_id]

    def _calculate_next_run(self, policy_id: str):
        """计算下次运行时间"""
        policy = self._policies.get(policy_id)
        if not policy:
            return

        # 简化的cron解析(实际应使用croniter库)
        # 格式: minute hour day month weekday
        parts = policy.schedule_cron.split()
        if len(parts) >= 2:
            minute = int(parts[0]) if parts[0] != "*" else 0
            hour = int(parts[1]) if parts[1] != "*" else 2

            now = datetime.now()
            next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if next_run <= now:
                next_run += timedelta(days=1)

            self._next_runs[policy_id] = next_run

    async def start(self):
        """启动调度器"""
        self._running = True
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())

    async def stop(self):
        """停止调度器"""
        self._running = False
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass

    async def _scheduler_loop(self):
        """调度循环"""
        while self._running:
            now = datetime.now()

            for policy_id, next_run in list(self._next_runs.items()):
                if next_run <= now:
                    policy = self._policies.get(policy_id)
                    if policy and policy.enabled:
                        # 执行备份
                        asyncio.create_task(self._execute_backup(policy))
                    # 计算下次运行
                    self._calculate_next_run(policy_id)

            await asyncio.sleep(60)  # 每分钟检查一次

    async def _execute_backup(self, policy: BackupPolicy):
        """执行备份"""
        try:
            # 获取父备份ID(用于增量备份)
            parent_backup_id = None
            if policy.backup_type == BackupType.INCREMENTAL:
                backups = self.backup_engine.list_backups(
                    status=BackupStatus.COMPLETED
                )
                if backups:
                    parent_backup_id = backups[0].backup_id

            # 创建备份
            metadata = await self.backup_engine.create_backup(
                source_paths=policy.source_paths,
                backup_type=policy.backup_type,
                exclude_patterns=policy.exclude_patterns,
                compression=policy.compression,
                parent_backup_id=parent_backup_id
            )

            # 验证备份
            if policy.verify_after_backup and metadata.status == BackupStatus.COMPLETED:
                await self.backup_engine.verify_backup(metadata.backup_id)

            # 清理旧备份
            await self._cleanup_old_backups(policy)

        except Exception as e:
            print(f"备份失败 [{policy.policy_id}]: {e}")

    async def _cleanup_old_backups(self, policy: BackupPolicy):
        """清理旧备份"""
        backups = self.backup_engine.list_backups(status=BackupStatus.COMPLETED)

        # 按保留数量清理
        if len(backups) > policy.retention_count:
            for backup in backups[policy.retention_count:]:
                await self.backup_engine.delete_backup(backup.backup_id)

        # 按保留天数清理
        cutoff = datetime.now() - timedelta(days=policy.retention_days)
        for backup in backups:
            if backup.created_at < cutoff:
                await self.backup_engine.delete_backup(backup.backup_id)


class BackupManager:
    """备份管理器(统一入口)"""

    def __init__(self, backup_path: str = "./backups"):
        self.backup_path = backup_path
        self.storage = LocalStorageBackend(backup_path)
        self.engine = BackupEngine(self.storage)
        self.scheduler = BackupScheduler(self.engine)
        self.synchronizer = DataSynchronizer()

    async def start(self):
        """启动备份管理器"""
        await self.scheduler.start()
        await self.synchronizer.start()

    async def stop(self):
        """停止备份管理器"""
        await self.scheduler.stop()
        await self.synchronizer.stop()

    async def create_backup(
        self,
        source_paths: List[str],
        backup_type: BackupType = BackupType.FULL,
        **kwargs
    ) -> BackupMetadata:
        """创建备份"""
        return await self.engine.create_backup(
            source_paths, backup_type, **kwargs
        )

    async def restore_backup(
        self,
        backup_id: str,
        restore_path: str,
        **kwargs
    ) -> bool:
        """恢复备份"""
        return await self.engine.restore_backup(
            backup_id, restore_path, **kwargs
        )

    def add_backup_policy(self, policy: BackupPolicy):
        """添加备份策略"""
        self.scheduler.add_policy(policy)

    def add_sync_config(self, config: SyncConfig):
        """添加同步配置"""
        self.synchronizer.add_sync_config(config)


def create_cyrp_backup_system(backup_path: str = "./backups") -> BackupManager:
    """创建穿黄工程备份系统"""
    manager = BackupManager(backup_path)

    # 添加默认备份策略
    # 每日全量备份
    daily_full = BackupPolicy(
        policy_id="daily_full",
        name="每日全量备份",
        description="每天凌晨2点执行全量备份",
        backup_type=BackupType.FULL,
        schedule_cron="0 2 * * *",
        source_paths=["./data", "./config"],
        exclude_patterns=["*.log", "*.tmp", "__pycache__"],
        retention_days=30,
        retention_count=30
    )
    manager.add_backup_policy(daily_full)

    # 每小时增量备份
    hourly_incremental = BackupPolicy(
        policy_id="hourly_incremental",
        name="每小时增量备份",
        description="每小时执行增量备份",
        backup_type=BackupType.INCREMENTAL,
        schedule_cron="0 * * * *",
        source_paths=["./data"],
        exclude_patterns=["*.log"],
        retention_days=7,
        retention_count=168  # 7天 * 24小时
    )
    manager.add_backup_policy(hourly_incremental)

    return manager
