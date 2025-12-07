"""
Unified Configuration Management System for CYRP
穿黄工程统一配置管理系统

功能:
- 多层级配置(系统/模块/设备/用户)
- 多格式支持(YAML/JSON/INI/ENV)
- 配置验证
- 配置版本管理
- 热加载与通知
- 配置加密
"""

import asyncio
import json
import os
import re
import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union
import hashlib
import base64


class ConfigFormat(Enum):
    """配置格式"""
    JSON = "json"
    YAML = "yaml"
    INI = "ini"
    ENV = "env"
    TOML = "toml"


class ConfigScope(Enum):
    """配置作用域"""
    SYSTEM = auto()      # 系统级配置
    MODULE = auto()      # 模块级配置
    DEVICE = auto()      # 设备级配置
    USER = auto()        # 用户级配置
    RUNTIME = auto()     # 运行时配置


class ConfigValueType(Enum):
    """配置值类型"""
    STRING = auto()
    INTEGER = auto()
    FLOAT = auto()
    BOOLEAN = auto()
    LIST = auto()
    DICT = auto()
    SECRET = auto()      # 加密存储


@dataclass
class ConfigSchema:
    """配置模式定义"""
    key: str
    value_type: ConfigValueType
    default: Any = None
    required: bool = False
    description: str = ""
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: Optional[List[Any]] = None
    pattern: Optional[str] = None  # 正则表达式验证
    nested_schema: Optional[Dict[str, 'ConfigSchema']] = None


@dataclass
class ConfigItem:
    """配置项"""
    key: str
    value: Any
    scope: ConfigScope
    value_type: ConfigValueType
    source: str  # 配置来源(文件路径/环境变量等)
    encrypted: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    version: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConfigVersion:
    """配置版本"""
    version_id: str
    config_key: str
    old_value: Any
    new_value: Any
    changed_by: Optional[str]
    changed_at: datetime
    comment: str = ""


class ConfigEncryptor:
    """配置加密器"""

    def __init__(self, secret_key: Optional[str] = None):
        if secret_key:
            self._key = hashlib.sha256(secret_key.encode()).digest()
        else:
            self._key = hashlib.sha256(b"cyrp-default-key").digest()

    def encrypt(self, value: str) -> str:
        """加密值(简单XOR加密,生产环境应使用更强的加密)"""
        value_bytes = value.encode('utf-8')
        encrypted = bytes(
            b ^ self._key[i % len(self._key)]
            for i, b in enumerate(value_bytes)
        )
        return base64.b64encode(encrypted).decode('utf-8')

    def decrypt(self, encrypted_value: str) -> str:
        """解密值"""
        encrypted_bytes = base64.b64decode(encrypted_value.encode('utf-8'))
        decrypted = bytes(
            b ^ self._key[i % len(self._key)]
            for i, b in enumerate(encrypted_bytes)
        )
        return decrypted.decode('utf-8')


class ConfigValidator:
    """配置验证器"""

    def __init__(self, schemas: Dict[str, ConfigSchema]):
        self.schemas = schemas

    def validate(self, key: str, value: Any) -> tuple[bool, List[str]]:
        """验证配置值"""
        if key not in self.schemas:
            return True, []  # 没有定义模式则跳过验证

        schema = self.schemas[key]
        errors = []

        # 类型验证
        type_valid = self._validate_type(value, schema.value_type)
        if not type_valid:
            errors.append(f"类型错误: 期望 {schema.value_type.name}, 实际 {type(value).__name__}")
            return False, errors

        # 范围验证
        if schema.min_value is not None and isinstance(value, (int, float)):
            if value < schema.min_value:
                errors.append(f"值 {value} 小于最小值 {schema.min_value}")

        if schema.max_value is not None and isinstance(value, (int, float)):
            if value > schema.max_value:
                errors.append(f"值 {value} 大于最大值 {schema.max_value}")

        # 允许值验证
        if schema.allowed_values is not None:
            if value not in schema.allowed_values:
                errors.append(f"值 {value} 不在允许列表中: {schema.allowed_values}")

        # 正则验证
        if schema.pattern is not None and isinstance(value, str):
            if not re.match(schema.pattern, value):
                errors.append(f"值 '{value}' 不匹配模式 '{schema.pattern}'")

        return len(errors) == 0, errors

    def _validate_type(self, value: Any, expected_type: ConfigValueType) -> bool:
        """验证类型"""
        type_map = {
            ConfigValueType.STRING: str,
            ConfigValueType.INTEGER: int,
            ConfigValueType.FLOAT: (int, float),
            ConfigValueType.BOOLEAN: bool,
            ConfigValueType.LIST: list,
            ConfigValueType.DICT: dict,
            ConfigValueType.SECRET: str,
        }

        expected = type_map.get(expected_type, object)
        return isinstance(value, expected)

    def validate_required(self, config: Dict[str, Any]) -> tuple[bool, List[str]]:
        """验证必需配置"""
        errors = []

        for key, schema in self.schemas.items():
            if schema.required and key not in config:
                errors.append(f"缺少必需配置项: {key}")

        return len(errors) == 0, errors


class ConfigSource(ABC):
    """配置源基类"""

    @abstractmethod
    async def load(self) -> Dict[str, Any]:
        """加载配置"""
        pass

    @abstractmethod
    async def save(self, config: Dict[str, Any]):
        """保存配置"""
        pass

    @abstractmethod
    def get_source_name(self) -> str:
        """获取源名称"""
        pass


class JSONConfigSource(ConfigSource):
    """JSON配置源"""

    def __init__(self, file_path: str):
        self.file_path = file_path

    async def load(self) -> Dict[str, Any]:
        """加载JSON配置"""
        if not os.path.exists(self.file_path):
            return {}

        with open(self.file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    async def save(self, config: Dict[str, Any]):
        """保存JSON配置"""
        os.makedirs(os.path.dirname(self.file_path) or '.', exist_ok=True)
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    def get_source_name(self) -> str:
        return f"json:{self.file_path}"


class YAMLConfigSource(ConfigSource):
    """YAML配置源"""

    def __init__(self, file_path: str):
        self.file_path = file_path

    async def load(self) -> Dict[str, Any]:
        """加载YAML配置"""
        if not os.path.exists(self.file_path):
            return {}

        # 简单的YAML解析(基础功能)
        config = {}
        with open(self.file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        current_key = None
        current_indent = 0

        for line in content.split('\n'):
            line = line.rstrip()
            if not line or line.strip().startswith('#'):
                continue

            # 计算缩进
            indent = len(line) - len(line.lstrip())

            # 解析键值对
            if ':' in line:
                parts = line.split(':', 1)
                key = parts[0].strip()
                value = parts[1].strip() if len(parts) > 1 else None

                if value:
                    # 类型转换
                    if value.lower() == 'true':
                        value = True
                    elif value.lower() == 'false':
                        value = False
                    elif value.isdigit():
                        value = int(value)
                    elif self._is_float(value):
                        value = float(value)
                    elif value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]

                    config[key] = value

        return config

    def _is_float(self, value: str) -> bool:
        """检查是否为浮点数"""
        try:
            float(value)
            return '.' in value
        except ValueError:
            return False

    async def save(self, config: Dict[str, Any]):
        """保存YAML配置"""
        os.makedirs(os.path.dirname(self.file_path) or '.', exist_ok=True)
        with open(self.file_path, 'w', encoding='utf-8') as f:
            for key, value in config.items():
                if isinstance(value, str):
                    f.write(f'{key}: "{value}"\n')
                elif isinstance(value, bool):
                    f.write(f'{key}: {str(value).lower()}\n')
                else:
                    f.write(f'{key}: {value}\n')

    def get_source_name(self) -> str:
        return f"yaml:{self.file_path}"


class EnvironmentConfigSource(ConfigSource):
    """环境变量配置源"""

    def __init__(self, prefix: str = "CYRP_"):
        self.prefix = prefix

    async def load(self) -> Dict[str, Any]:
        """加载环境变量配置"""
        config = {}

        for key, value in os.environ.items():
            if key.startswith(self.prefix):
                # 移除前缀并转换为小写
                config_key = key[len(self.prefix):].lower()

                # 类型转换
                if value.lower() == 'true':
                    value = True
                elif value.lower() == 'false':
                    value = False
                elif value.isdigit():
                    value = int(value)
                elif self._is_float(value):
                    value = float(value)

                config[config_key] = value

        return config

    def _is_float(self, value: str) -> bool:
        try:
            float(value)
            return '.' in value
        except ValueError:
            return False

    async def save(self, config: Dict[str, Any]):
        """环境变量不支持保存"""
        pass

    def get_source_name(self) -> str:
        return f"env:{self.prefix}*"


class INIConfigSource(ConfigSource):
    """INI配置源"""

    def __init__(self, file_path: str):
        self.file_path = file_path

    async def load(self) -> Dict[str, Any]:
        """加载INI配置"""
        if not os.path.exists(self.file_path):
            return {}

        config = {}
        current_section = "default"

        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or line.startswith(';'):
                    continue

                if line.startswith('[') and line.endswith(']'):
                    current_section = line[1:-1]
                    if current_section not in config:
                        config[current_section] = {}
                elif '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()

                    # 类型转换
                    if value.lower() == 'true':
                        value = True
                    elif value.lower() == 'false':
                        value = False
                    elif value.isdigit():
                        value = int(value)

                    if current_section == "default":
                        config[key] = value
                    else:
                        if current_section not in config:
                            config[current_section] = {}
                        config[current_section][key] = value

        return config

    async def save(self, config: Dict[str, Any]):
        """保存INI配置"""
        os.makedirs(os.path.dirname(self.file_path) or '.', exist_ok=True)
        with open(self.file_path, 'w', encoding='utf-8') as f:
            # 写入非section项
            for key, value in config.items():
                if not isinstance(value, dict):
                    f.write(f'{key} = {value}\n')

            # 写入section
            for section, values in config.items():
                if isinstance(values, dict):
                    f.write(f'\n[{section}]\n')
                    for key, value in values.items():
                        f.write(f'{key} = {value}\n')

    def get_source_name(self) -> str:
        return f"ini:{self.file_path}"


class ConfigChangeEvent:
    """配置变更事件"""

    def __init__(
        self,
        key: str,
        old_value: Any,
        new_value: Any,
        scope: ConfigScope,
        source: str
    ):
        self.key = key
        self.old_value = old_value
        self.new_value = new_value
        self.scope = scope
        self.source = source
        self.timestamp = datetime.now()


class ConfigManager:
    """配置管理器"""

    def __init__(
        self,
        encryptor: Optional[ConfigEncryptor] = None,
        validator: Optional[ConfigValidator] = None
    ):
        self.encryptor = encryptor or ConfigEncryptor()
        self.validator = validator
        self._sources: Dict[ConfigScope, List[ConfigSource]] = {
            scope: [] for scope in ConfigScope
        }
        self._config: Dict[str, ConfigItem] = {}
        self._listeners: List[Callable[[ConfigChangeEvent], None]] = []
        self._version_history: List[ConfigVersion] = []
        self._lock = asyncio.Lock()
        self._watch_task: Optional[asyncio.Task] = None
        self._file_mtimes: Dict[str, float] = {}

    def add_source(self, scope: ConfigScope, source: ConfigSource):
        """添加配置源"""
        self._sources[scope].append(source)

    def add_listener(self, listener: Callable[[ConfigChangeEvent], None]):
        """添加配置变更监听器"""
        self._listeners.append(listener)

    async def load(self):
        """加载所有配置"""
        async with self._lock:
            # 按优先级加载(SYSTEM -> MODULE -> DEVICE -> USER -> RUNTIME)
            for scope in ConfigScope:
                for source in self._sources[scope]:
                    try:
                        config_data = await source.load()
                        await self._merge_config(config_data, scope, source.get_source_name())
                    except Exception as e:
                        print(f"加载配置失败 [{source.get_source_name()}]: {e}")

    async def _merge_config(
        self,
        config_data: Dict[str, Any],
        scope: ConfigScope,
        source: str
    ):
        """合并配置"""
        for key, value in config_data.items():
            # 验证配置
            if self.validator:
                valid, errors = self.validator.validate(key, value)
                if not valid:
                    print(f"配置验证失败 [{key}]: {errors}")
                    continue

            # 确定值类型
            value_type = self._infer_type(value)
            encrypted = False

            # 处理加密值
            if isinstance(value, str) and value.startswith("ENC:"):
                encrypted = True
                value = self.encryptor.decrypt(value[4:])

            # 创建或更新配置项
            if key in self._config:
                old_item = self._config[key]
                # 高优先级覆盖低优先级
                if scope.value >= old_item.scope.value:
                    old_value = old_item.value
                    old_item.value = value
                    old_item.scope = scope
                    old_item.source = source
                    old_item.updated_at = datetime.now()
                    old_item.version += 1

                    # 记录版本历史
                    version = ConfigVersion(
                        version_id=f"v{old_item.version}",
                        config_key=key,
                        old_value=old_value,
                        new_value=value,
                        changed_by=None,
                        changed_at=datetime.now()
                    )
                    self._version_history.append(version)
            else:
                self._config[key] = ConfigItem(
                    key=key,
                    value=value,
                    scope=scope,
                    value_type=value_type,
                    source=source,
                    encrypted=encrypted
                )

    def _infer_type(self, value: Any) -> ConfigValueType:
        """推断值类型"""
        if isinstance(value, bool):
            return ConfigValueType.BOOLEAN
        elif isinstance(value, int):
            return ConfigValueType.INTEGER
        elif isinstance(value, float):
            return ConfigValueType.FLOAT
        elif isinstance(value, list):
            return ConfigValueType.LIST
        elif isinstance(value, dict):
            return ConfigValueType.DICT
        else:
            return ConfigValueType.STRING

    def get(
        self,
        key: str,
        default: Any = None,
        decrypt: bool = True
    ) -> Any:
        """获取配置值"""
        if key not in self._config:
            return default

        item = self._config[key]
        value = item.value

        if item.encrypted and decrypt:
            try:
                value = self.encryptor.decrypt(value)
            except Exception:
                pass

        return value

    def get_int(self, key: str, default: int = 0) -> int:
        """获取整数配置"""
        value = self.get(key, default)
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def get_float(self, key: str, default: float = 0.0) -> float:
        """获取浮点数配置"""
        value = self.get(key, default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def get_bool(self, key: str, default: bool = False) -> bool:
        """获取布尔配置"""
        value = self.get(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ('true', 'yes', '1', 'on')
        return bool(value)

    def get_list(self, key: str, default: Optional[List] = None) -> List:
        """获取列表配置"""
        value = self.get(key, default or [])
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            return [v.strip() for v in value.split(',')]
        return default or []

    def get_dict(self, key: str, default: Optional[Dict] = None) -> Dict:
        """获取字典配置"""
        value = self.get(key, default or {})
        if isinstance(value, dict):
            return value
        return default or {}

    async def set(
        self,
        key: str,
        value: Any,
        scope: ConfigScope = ConfigScope.RUNTIME,
        encrypt: bool = False,
        changed_by: Optional[str] = None
    ):
        """设置配置值"""
        async with self._lock:
            # 验证配置
            if self.validator:
                valid, errors = self.validator.validate(key, value)
                if not valid:
                    raise ValueError(f"配置验证失败: {errors}")

            old_value = None
            if key in self._config:
                old_value = self._config[key].value

            # 加密处理
            stored_value = value
            if encrypt:
                stored_value = self.encryptor.encrypt(str(value))

            # 更新配置
            value_type = self._infer_type(value)
            if key in self._config:
                self._config[key].value = stored_value
                self._config[key].updated_at = datetime.now()
                self._config[key].version += 1
                self._config[key].encrypted = encrypt
            else:
                self._config[key] = ConfigItem(
                    key=key,
                    value=stored_value,
                    scope=scope,
                    value_type=value_type,
                    source="runtime",
                    encrypted=encrypt
                )

            # 记录版本
            version = ConfigVersion(
                version_id=f"v{self._config[key].version}",
                config_key=key,
                old_value=old_value,
                new_value=value,
                changed_by=changed_by,
                changed_at=datetime.now()
            )
            self._version_history.append(version)

            # 通知监听器
            event = ConfigChangeEvent(
                key=key,
                old_value=old_value,
                new_value=value,
                scope=scope,
                source="runtime"
            )
            for listener in self._listeners:
                try:
                    listener(event)
                except Exception:
                    pass

    async def save(self, scope: ConfigScope = ConfigScope.USER):
        """保存配置"""
        async with self._lock:
            # 收集该作用域的配置
            config_data = {}
            for key, item in self._config.items():
                if item.scope == scope:
                    if item.encrypted:
                        config_data[key] = f"ENC:{item.value}"
                    else:
                        config_data[key] = item.value

            # 保存到所有源
            for source in self._sources[scope]:
                try:
                    await source.save(config_data)
                except Exception as e:
                    print(f"保存配置失败 [{source.get_source_name()}]: {e}")

    def get_all(self, scope: Optional[ConfigScope] = None) -> Dict[str, Any]:
        """获取所有配置"""
        result = {}
        for key, item in self._config.items():
            if scope is None or item.scope == scope:
                result[key] = item.value
        return result

    def get_version_history(
        self,
        key: Optional[str] = None,
        limit: int = 100
    ) -> List[ConfigVersion]:
        """获取版本历史"""
        history = self._version_history
        if key:
            history = [v for v in history if v.config_key == key]
        return history[-limit:]

    async def start_watch(self, interval: float = 5.0):
        """开始监视配置文件变化"""
        self._watch_task = asyncio.create_task(self._watch_loop(interval))

    async def stop_watch(self):
        """停止监视"""
        if self._watch_task:
            self._watch_task.cancel()
            try:
                await self._watch_task
            except asyncio.CancelledError:
                pass

    async def _watch_loop(self, interval: float):
        """监视循环"""
        while True:
            await asyncio.sleep(interval)

            for scope in ConfigScope:
                for source in self._sources[scope]:
                    if hasattr(source, 'file_path'):
                        file_path = source.file_path
                        if os.path.exists(file_path):
                            mtime = os.path.getmtime(file_path)
                            if file_path in self._file_mtimes:
                                if mtime > self._file_mtimes[file_path]:
                                    # 文件已修改,重新加载
                                    await self.load()
                            self._file_mtimes[file_path] = mtime

    def export_config(self, format: ConfigFormat = ConfigFormat.JSON) -> str:
        """导出配置"""
        config_data = self.get_all()

        if format == ConfigFormat.JSON:
            return json.dumps(config_data, ensure_ascii=False, indent=2)
        elif format == ConfigFormat.YAML:
            lines = []
            for key, value in config_data.items():
                if isinstance(value, str):
                    lines.append(f'{key}: "{value}"')
                elif isinstance(value, bool):
                    lines.append(f'{key}: {str(value).lower()}')
                else:
                    lines.append(f'{key}: {value}')
            return '\n'.join(lines)
        elif format == ConfigFormat.ENV:
            lines = []
            for key, value in config_data.items():
                lines.append(f'CYRP_{key.upper()}={value}')
            return '\n'.join(lines)
        else:
            return json.dumps(config_data, ensure_ascii=False)


class ConfigBuilder:
    """配置构建器"""

    def __init__(self):
        self._manager = ConfigManager()

    def with_json_source(
        self,
        file_path: str,
        scope: ConfigScope = ConfigScope.SYSTEM
    ) -> 'ConfigBuilder':
        """添加JSON配置源"""
        self._manager.add_source(scope, JSONConfigSource(file_path))
        return self

    def with_yaml_source(
        self,
        file_path: str,
        scope: ConfigScope = ConfigScope.SYSTEM
    ) -> 'ConfigBuilder':
        """添加YAML配置源"""
        self._manager.add_source(scope, YAMLConfigSource(file_path))
        return self

    def with_env_source(
        self,
        prefix: str = "CYRP_",
        scope: ConfigScope = ConfigScope.RUNTIME
    ) -> 'ConfigBuilder':
        """添加环境变量配置源"""
        self._manager.add_source(scope, EnvironmentConfigSource(prefix))
        return self

    def with_ini_source(
        self,
        file_path: str,
        scope: ConfigScope = ConfigScope.SYSTEM
    ) -> 'ConfigBuilder':
        """添加INI配置源"""
        self._manager.add_source(scope, INIConfigSource(file_path))
        return self

    def with_encryption(self, secret_key: str) -> 'ConfigBuilder':
        """配置加密"""
        self._manager.encryptor = ConfigEncryptor(secret_key)
        return self

    def with_validator(self, schemas: Dict[str, ConfigSchema]) -> 'ConfigBuilder':
        """配置验证"""
        self._manager.validator = ConfigValidator(schemas)
        return self

    def with_listener(
        self,
        listener: Callable[[ConfigChangeEvent], None]
    ) -> 'ConfigBuilder':
        """添加监听器"""
        self._manager.add_listener(listener)
        return self

    def build(self) -> ConfigManager:
        """构建配置管理器"""
        return self._manager


def create_cyrp_config_system(config_dir: str = "./config") -> ConfigManager:
    """创建穿黄工程配置系统"""
    # 定义配置模式
    schemas = {
        # 系统配置
        "system.name": ConfigSchema(
            key="system.name",
            value_type=ConfigValueType.STRING,
            default="穿黄工程智能管控系统",
            description="系统名称"
        ),
        "system.version": ConfigSchema(
            key="system.version",
            value_type=ConfigValueType.STRING,
            default="1.0.0",
            description="系统版本"
        ),
        "system.debug": ConfigSchema(
            key="system.debug",
            value_type=ConfigValueType.BOOLEAN,
            default=False,
            description="调试模式"
        ),

        # 数据库配置
        "database.host": ConfigSchema(
            key="database.host",
            value_type=ConfigValueType.STRING,
            default="localhost",
            required=True,
            description="数据库主机"
        ),
        "database.port": ConfigSchema(
            key="database.port",
            value_type=ConfigValueType.INTEGER,
            default=5432,
            min_value=1,
            max_value=65535,
            description="数据库端口"
        ),
        "database.name": ConfigSchema(
            key="database.name",
            value_type=ConfigValueType.STRING,
            default="cyrp",
            description="数据库名称"
        ),

        # 通信配置
        "communication.modbus.port": ConfigSchema(
            key="communication.modbus.port",
            value_type=ConfigValueType.INTEGER,
            default=502,
            min_value=1,
            max_value=65535,
            description="Modbus端口"
        ),
        "communication.opcua.endpoint": ConfigSchema(
            key="communication.opcua.endpoint",
            value_type=ConfigValueType.STRING,
            default="opc.tcp://localhost:4840",
            description="OPC-UA端点"
        ),

        # 报警配置
        "alarm.retention_days": ConfigSchema(
            key="alarm.retention_days",
            value_type=ConfigValueType.INTEGER,
            default=90,
            min_value=1,
            max_value=3650,
            description="报警保留天数"
        ),
        "alarm.max_active": ConfigSchema(
            key="alarm.max_active",
            value_type=ConfigValueType.INTEGER,
            default=1000,
            min_value=100,
            max_value=10000,
            description="最大活动报警数"
        ),

        # 调度配置
        "scheduler.enabled": ConfigSchema(
            key="scheduler.enabled",
            value_type=ConfigValueType.BOOLEAN,
            default=True,
            description="启用调度器"
        ),
        "scheduler.interval_seconds": ConfigSchema(
            key="scheduler.interval_seconds",
            value_type=ConfigValueType.INTEGER,
            default=60,
            min_value=1,
            max_value=3600,
            description="调度间隔(秒)"
        ),

        # 日志配置
        "logging.level": ConfigSchema(
            key="logging.level",
            value_type=ConfigValueType.STRING,
            default="INFO",
            allowed_values=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            description="日志级别"
        ),
        "logging.file": ConfigSchema(
            key="logging.file",
            value_type=ConfigValueType.STRING,
            default="./logs/cyrp.log",
            description="日志文件路径"
        ),

        # Web配置
        "web.host": ConfigSchema(
            key="web.host",
            value_type=ConfigValueType.STRING,
            default="0.0.0.0",
            description="Web服务器主机"
        ),
        "web.port": ConfigSchema(
            key="web.port",
            value_type=ConfigValueType.INTEGER,
            default=8080,
            min_value=1,
            max_value=65535,
            description="Web服务器端口"
        ),
        "web.cors_origins": ConfigSchema(
            key="web.cors_origins",
            value_type=ConfigValueType.LIST,
            default=["*"],
            description="CORS允许的源"
        ),

        # 安全配置
        "security.session_timeout_minutes": ConfigSchema(
            key="security.session_timeout_minutes",
            value_type=ConfigValueType.INTEGER,
            default=30,
            min_value=5,
            max_value=480,
            description="会话超时(分钟)"
        ),
        "security.max_login_attempts": ConfigSchema(
            key="security.max_login_attempts",
            value_type=ConfigValueType.INTEGER,
            default=5,
            min_value=3,
            max_value=10,
            description="最大登录尝试次数"
        ),
    }

    # 构建配置管理器
    manager = (
        ConfigBuilder()
        .with_json_source(f"{config_dir}/system.json", ConfigScope.SYSTEM)
        .with_yaml_source(f"{config_dir}/modules.yaml", ConfigScope.MODULE)
        .with_json_source(f"{config_dir}/devices.json", ConfigScope.DEVICE)
        .with_json_source(f"{config_dir}/user.json", ConfigScope.USER)
        .with_env_source("CYRP_", ConfigScope.RUNTIME)
        .with_validator(schemas)
        .build()
    )

    return manager


# 默认配置模板
DEFAULT_SYSTEM_CONFIG = {
    "system.name": "穿黄工程智能管控系统",
    "system.version": "1.0.0",
    "system.debug": False,

    "database.host": "localhost",
    "database.port": 5432,
    "database.name": "cyrp",
    "database.user": "cyrp_user",

    "communication.modbus.port": 502,
    "communication.modbus.timeout": 5000,
    "communication.opcua.endpoint": "opc.tcp://localhost:4840",
    "communication.opcua.security_mode": "None",

    "alarm.retention_days": 90,
    "alarm.max_active": 1000,
    "alarm.notification.email.enabled": True,
    "alarm.notification.sms.enabled": False,

    "scheduler.enabled": True,
    "scheduler.interval_seconds": 60,

    "logging.level": "INFO",
    "logging.file": "./logs/cyrp.log",
    "logging.max_size_mb": 100,
    "logging.backup_count": 10,

    "web.host": "0.0.0.0",
    "web.port": 8080,
    "web.cors_origins": ["*"],
    "web.static_path": "./static",

    "security.session_timeout_minutes": 30,
    "security.max_login_attempts": 5,
    "security.lockout_duration_minutes": 30,
    "security.password_min_length": 8,

    "hil.enabled": True,
    "hil.real_time": False,
    "hil.time_scale": 1.0,

    "scenario.auto_generate": True,
    "scenario.max_concurrent": 10,

    "maintenance.prediction_enabled": True,
    "maintenance.check_interval_hours": 24,
}
