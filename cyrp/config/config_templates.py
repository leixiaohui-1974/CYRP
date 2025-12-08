"""
CYRP Configuration Templates
穿黄工程配置模板

提供预定义的配置模板和配置生成器
"""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List
from enum import Enum
from pathlib import Path


class EnvironmentType(Enum):
    """环境类型"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class SensorNetworkConfig:
    """传感器网络配置"""
    network_name: str = "tunnel_network"
    total_sensors: int = 31
    pressure_sensors: int = 11
    flow_sensors: int = 10
    level_sensors: int = 10

    # 采样配置
    sampling_rate_hz: float = 10.0
    buffer_size: int = 1000

    # 噪声模型
    noise_std: float = 0.01
    drift_rate: float = 1e-6

    # 故障检测
    failure_detection: bool = True
    stuck_threshold: float = 1e-6
    spike_multiplier: float = 3.0


@dataclass
class PredictorConfig:
    """预测器配置"""
    # 默认预测器
    default_predictor: str = "ensemble"

    # 指数平滑
    es_alpha: float = 0.3
    es_beta: float = 0.1
    es_gamma: float = 0.1

    # ARIMA
    arima_p: int = 1
    arima_d: int = 1
    arima_q: int = 1

    # 预测范围
    default_horizon: int = 10
    max_horizon: int = 100

    # 不确定性
    uncertainty_quantification: bool = True
    confidence_level: float = 0.95


@dataclass
class InterlockConfig:
    """安全联锁配置"""
    # 阈值
    vacuum_threshold_pa: float = -50000.0
    overpressure_threshold_pa: float = 1000000.0
    surge_threshold: float = 0.1
    asymmetric_threshold: float = 0.15

    # 行为
    hysteresis: float = 0.05
    lockout_seconds: float = 30.0

    # 启用状态
    anti_vacuum: bool = True
    anti_overpressure: bool = True
    anti_surge: bool = True
    anti_asymmetric: bool = True


@dataclass
class APIServerConfig:
    """API服务器配置"""
    host: str = "0.0.0.0"
    port: int = 8080
    ws_port: int = 8081

    # 安全
    jwt_secret: str = ""
    jwt_expiry_hours: int = 24
    rate_limit: int = 100

    # CORS
    cors_origins: List[str] = field(default_factory=lambda: ["*"])

    # 超时
    request_timeout: float = 30.0
    ws_heartbeat: float = 30.0


@dataclass
class DatabaseConfig:
    """数据库配置"""
    db_type: str = "sqlite"
    db_path: str = "cyrp_data.db"

    # 连接池
    pool_size: int = 5
    max_overflow: int = 10

    # 缓冲
    buffer_size: int = 1000
    flush_interval: float = 5.0

    # 归档
    archive_enabled: bool = True
    archive_days: int = 30
    cleanup_days: int = 90

    # 压缩
    compression: bool = True
    compression_level: int = 6


@dataclass
class AlertConfig:
    """告警配置"""
    default_cooldown: int = 300
    max_active_alerts: int = 1000

    # 通知
    notification_enabled: bool = True
    email_enabled: bool = False
    webhook_enabled: bool = False

    # 接收者
    email_recipients: List[str] = field(default_factory=list)
    webhook_urls: List[str] = field(default_factory=list)


@dataclass
class LoggingConfig:
    """日志配置"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # 文件日志
    file_enabled: bool = True
    file_path: str = "logs/cyrp.log"
    max_size_mb: int = 100
    backup_count: int = 5

    # JSON格式
    json_format: bool = False


@dataclass
class CYRPConfigTemplate:
    """CYRP完整配置模板"""
    # 元信息
    version: str = "1.0.0"
    environment: str = "development"
    instance_name: str = "cyrp-default"

    # 子配置
    sensor: SensorNetworkConfig = field(default_factory=SensorNetworkConfig)
    predictor: PredictorConfig = field(default_factory=PredictorConfig)
    interlock: InterlockConfig = field(default_factory=InterlockConfig)
    api: APIServerConfig = field(default_factory=APIServerConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    alert: AlertConfig = field(default_factory=AlertConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # 功能开关
    features: Dict[str, bool] = field(default_factory=lambda: {
        "sensor_simulation": True,
        "data_assimilation": True,
        "state_prediction": True,
        "safety_interlocks": True,
        "real_time_push": True,
        "data_persistence": True,
        "monitoring_api": True,
        "cli_management": True,
    })


class ConfigTemplateFactory:
    """配置模板工厂"""

    @staticmethod
    def create_development() -> CYRPConfigTemplate:
        """创建开发环境配置"""
        config = CYRPConfigTemplate()
        config.environment = "development"
        config.logging.level = "DEBUG"
        config.database.db_path = ":memory:"
        config.api.jwt_secret = "dev-secret-key-not-for-production"
        config.alert.notification_enabled = False
        return config

    @staticmethod
    def create_testing() -> CYRPConfigTemplate:
        """创建测试环境配置"""
        config = CYRPConfigTemplate()
        config.environment = "testing"
        config.instance_name = "cyrp-test"
        config.logging.level = "DEBUG"
        config.database.db_path = ":memory:"
        config.database.archive_enabled = False
        config.alert.notification_enabled = False
        config.api.jwt_secret = "test-secret-key"
        return config

    @staticmethod
    def create_staging() -> CYRPConfigTemplate:
        """创建预发布环境配置"""
        config = CYRPConfigTemplate()
        config.environment = "staging"
        config.instance_name = "cyrp-staging"
        config.logging.level = "INFO"
        config.database.db_path = "cyrp_staging.db"
        config.api.rate_limit = 500
        config.alert.notification_enabled = True
        return config

    @staticmethod
    def create_production() -> CYRPConfigTemplate:
        """创建生产环境配置"""
        config = CYRPConfigTemplate()
        config.environment = "production"
        config.instance_name = "cyrp-prod"
        config.logging.level = "WARNING"
        config.logging.json_format = True
        config.database.db_path = "/var/lib/cyrp/data.db"
        config.database.archive_enabled = True
        config.database.compression = True
        config.api.rate_limit = 1000
        config.api.jwt_secret = ""  # Must be set via environment variable
        config.alert.notification_enabled = True
        return config

    @staticmethod
    def create_hil_simulation() -> CYRPConfigTemplate:
        """创建HIL仿真配置"""
        config = CYRPConfigTemplate()
        config.environment = "development"
        config.instance_name = "cyrp-hil"
        config.sensor.sampling_rate_hz = 100.0  # 高频采样
        config.sensor.buffer_size = 10000
        config.predictor.default_horizon = 50
        config.logging.level = "INFO"
        config.features["hil_simulation"] = True
        return config

    @staticmethod
    def create_minimal() -> CYRPConfigTemplate:
        """创建最小配置"""
        config = CYRPConfigTemplate()
        config.environment = "development"
        config.database.db_path = ":memory:"
        config.features = {
            "sensor_simulation": True,
            "safety_interlocks": True,
            "monitoring_api": False,
            "real_time_push": False,
            "data_persistence": False,
        }
        return config


class ConfigGenerator:
    """配置生成器"""

    def __init__(self, template: CYRPConfigTemplate):
        self.template = template

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self.template)

    def to_json(self, indent: int = 2) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def to_yaml(self) -> str:
        """转换为YAML字符串"""
        lines = []
        self._dict_to_yaml(self.to_dict(), lines, 0)
        return "\n".join(lines)

    def _dict_to_yaml(self, d: Dict, lines: List[str], indent: int):
        """递归转换为YAML"""
        prefix = "  " * indent
        for key, value in d.items():
            if isinstance(value, dict):
                lines.append(f"{prefix}{key}:")
                self._dict_to_yaml(value, lines, indent + 1)
            elif isinstance(value, list):
                lines.append(f"{prefix}{key}:")
                for item in value:
                    lines.append(f"{prefix}  - {item}")
            elif isinstance(value, bool):
                lines.append(f"{prefix}{key}: {str(value).lower()}")
            elif isinstance(value, str):
                lines.append(f'{prefix}{key}: "{value}"')
            else:
                lines.append(f"{prefix}{key}: {value}")

    def to_env(self, prefix: str = "CYRP") -> str:
        """转换为环境变量格式"""
        lines = []
        self._dict_to_env(self.to_dict(), lines, prefix)
        return "\n".join(lines)

    def _dict_to_env(self, d: Dict, lines: List[str], prefix: str):
        """递归转换为环境变量"""
        for key, value in d.items():
            env_key = f"{prefix}_{key}".upper()
            if isinstance(value, dict):
                self._dict_to_env(value, lines, env_key)
            elif isinstance(value, list):
                lines.append(f"{env_key}={','.join(str(v) for v in value)}")
            elif isinstance(value, bool):
                lines.append(f"{env_key}={str(value).lower()}")
            else:
                lines.append(f"{env_key}={value}")

    def save_json(self, path: str):
        """保存为JSON文件"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(self.to_json())

    def save_yaml(self, path: str):
        """保存为YAML文件"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(self.to_yaml())

    def save_env(self, path: str, prefix: str = "CYRP"):
        """保存为环境变量文件"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(self.to_env(prefix))


class ConfigValidator:
    """配置验证器"""

    @staticmethod
    def validate(config: CYRPConfigTemplate) -> tuple[bool, List[str]]:
        """验证配置"""
        errors = []

        # 端口验证
        if not (1 <= config.api.port <= 65535):
            errors.append(f"API port must be 1-65535, got {config.api.port}")

        if not (1 <= config.api.ws_port <= 65535):
            errors.append(f"WebSocket port must be 1-65535, got {config.api.ws_port}")

        if config.api.port == config.api.ws_port:
            errors.append("API and WebSocket ports must be different")

        # 采样率验证
        if config.sensor.sampling_rate_hz <= 0:
            errors.append("Sampling rate must be positive")

        # 预测范围验证
        if config.predictor.default_horizon > config.predictor.max_horizon:
            errors.append("Default horizon cannot exceed max horizon")

        # 置信水平验证
        if not (0 < config.predictor.confidence_level < 1):
            errors.append("Confidence level must be between 0 and 1")

        # 生产环境JWT验证
        if config.environment == "production" and not config.api.jwt_secret:
            errors.append("JWT secret is required in production")

        # 日志级别验证
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if config.logging.level.upper() not in valid_levels:
            errors.append(f"Invalid log level: {config.logging.level}")

        return len(errors) == 0, errors


def generate_config_files(output_dir: str = "./config"):
    """生成所有环境的配置文件"""
    environments = [
        ("development", ConfigTemplateFactory.create_development()),
        ("testing", ConfigTemplateFactory.create_testing()),
        ("staging", ConfigTemplateFactory.create_staging()),
        ("production", ConfigTemplateFactory.create_production()),
        ("hil", ConfigTemplateFactory.create_hil_simulation()),
    ]

    for env_name, template in environments:
        generator = ConfigGenerator(template)

        # 验证
        valid, errors = ConfigValidator.validate(template)
        if not valid:
            print(f"Warning: {env_name} config has validation errors: {errors}")

        # 保存
        env_dir = os.path.join(output_dir, env_name)
        generator.save_json(os.path.join(env_dir, "config.json"))
        generator.save_yaml(os.path.join(env_dir, "config.yaml"))
        generator.save_env(os.path.join(env_dir, ".env"))

        print(f"Generated config files for {env_name} in {env_dir}")


def get_config_for_environment(env: str) -> CYRPConfigTemplate:
    """根据环境获取配置"""
    factory_map = {
        "development": ConfigTemplateFactory.create_development,
        "testing": ConfigTemplateFactory.create_testing,
        "staging": ConfigTemplateFactory.create_staging,
        "production": ConfigTemplateFactory.create_production,
        "hil": ConfigTemplateFactory.create_hil_simulation,
    }

    factory = factory_map.get(env.lower(), ConfigTemplateFactory.create_development)
    return factory()
