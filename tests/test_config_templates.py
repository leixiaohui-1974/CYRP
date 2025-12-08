"""
配置模板系统测试
Tests for CYRP Configuration Templates
"""

import pytest
import tempfile
import os
import json


class TestConfigTemplates:
    """测试配置模板"""

    def test_sensor_network_config_defaults(self):
        """测试传感器网络配置默认值"""
        from cyrp.config.config_templates import SensorNetworkConfig

        config = SensorNetworkConfig()
        assert config.network_name == "tunnel_network"
        assert config.total_sensors == 31
        assert config.sampling_rate_hz == 10.0
        assert config.failure_detection is True

    def test_predictor_config_defaults(self):
        """测试预测器配置默认值"""
        from cyrp.config.config_templates import PredictorConfig

        config = PredictorConfig()
        assert config.default_predictor == "ensemble"
        assert config.es_alpha == 0.3
        assert config.default_horizon == 10
        assert config.confidence_level == 0.95

    def test_interlock_config_defaults(self):
        """测试联锁配置默认值"""
        from cyrp.config.config_templates import InterlockConfig

        config = InterlockConfig()
        assert config.vacuum_threshold_pa == -50000.0
        assert config.overpressure_threshold_pa == 1000000.0
        assert config.anti_vacuum is True
        assert config.anti_overpressure is True

    def test_api_server_config_defaults(self):
        """测试API服务器配置默认值"""
        from cyrp.config.config_templates import APIServerConfig

        config = APIServerConfig()
        assert config.host == "0.0.0.0"
        assert config.port == 8080
        assert config.ws_port == 8081
        assert config.rate_limit == 100

    def test_database_config_defaults(self):
        """测试数据库配置默认值"""
        from cyrp.config.config_templates import DatabaseConfig

        config = DatabaseConfig()
        assert config.db_type == "sqlite"
        assert config.buffer_size == 1000
        assert config.archive_enabled is True

    def test_full_config_template(self):
        """测试完整配置模板"""
        from cyrp.config.config_templates import CYRPConfigTemplate

        config = CYRPConfigTemplate()
        assert config.version == "1.0.0"
        assert config.environment == "development"
        assert config.sensor is not None
        assert config.predictor is not None
        assert config.interlock is not None


class TestConfigTemplateFactory:
    """测试配置模板工厂"""

    def test_create_development(self):
        """测试创建开发环境配置"""
        from cyrp.config.config_templates import ConfigTemplateFactory

        config = ConfigTemplateFactory.create_development()
        assert config.environment == "development"
        assert config.logging.level == "DEBUG"
        assert config.database.db_path == ":memory:"
        assert config.alert.notification_enabled is False

    def test_create_testing(self):
        """测试创建测试环境配置"""
        from cyrp.config.config_templates import ConfigTemplateFactory

        config = ConfigTemplateFactory.create_testing()
        assert config.environment == "testing"
        assert config.instance_name == "cyrp-test"
        assert config.database.archive_enabled is False

    def test_create_staging(self):
        """测试创建预发布环境配置"""
        from cyrp.config.config_templates import ConfigTemplateFactory

        config = ConfigTemplateFactory.create_staging()
        assert config.environment == "staging"
        assert config.logging.level == "INFO"
        assert config.api.rate_limit == 500

    def test_create_production(self):
        """测试创建生产环境配置"""
        from cyrp.config.config_templates import ConfigTemplateFactory

        config = ConfigTemplateFactory.create_production()
        assert config.environment == "production"
        assert config.logging.level == "WARNING"
        assert config.logging.json_format is True
        assert config.database.archive_enabled is True

    def test_create_hil_simulation(self):
        """测试创建HIL仿真配置"""
        from cyrp.config.config_templates import ConfigTemplateFactory

        config = ConfigTemplateFactory.create_hil_simulation()
        assert config.sensor.sampling_rate_hz == 100.0
        assert config.sensor.buffer_size == 10000
        assert config.predictor.default_horizon == 50

    def test_create_minimal(self):
        """测试创建最小配置"""
        from cyrp.config.config_templates import ConfigTemplateFactory

        config = ConfigTemplateFactory.create_minimal()
        assert config.features["sensor_simulation"] is True
        assert config.features["monitoring_api"] is False


class TestConfigGenerator:
    """测试配置生成器"""

    def test_to_dict(self):
        """测试转换为字典"""
        from cyrp.config.config_templates import (
            ConfigTemplateFactory, ConfigGenerator
        )

        template = ConfigTemplateFactory.create_development()
        generator = ConfigGenerator(template)
        d = generator.to_dict()

        assert isinstance(d, dict)
        assert d["version"] == "1.0.0"
        assert d["environment"] == "development"
        assert "sensor" in d
        assert "predictor" in d

    def test_to_json(self):
        """测试转换为JSON"""
        from cyrp.config.config_templates import (
            ConfigTemplateFactory, ConfigGenerator
        )

        template = ConfigTemplateFactory.create_testing()
        generator = ConfigGenerator(template)
        json_str = generator.to_json()

        # 验证是有效JSON
        data = json.loads(json_str)
        assert data["environment"] == "testing"

    def test_to_yaml(self):
        """测试转换为YAML"""
        from cyrp.config.config_templates import (
            ConfigTemplateFactory, ConfigGenerator
        )

        template = ConfigTemplateFactory.create_development()
        generator = ConfigGenerator(template)
        yaml_str = generator.to_yaml()

        assert "version:" in yaml_str
        assert "environment:" in yaml_str
        assert "sensor:" in yaml_str

    def test_to_env(self):
        """测试转换为环境变量格式"""
        from cyrp.config.config_templates import (
            ConfigTemplateFactory, ConfigGenerator
        )

        template = ConfigTemplateFactory.create_development()
        generator = ConfigGenerator(template)
        env_str = generator.to_env("CYRP")

        assert "CYRP_VERSION=" in env_str
        assert "CYRP_ENVIRONMENT=" in env_str

    def test_save_json(self):
        """测试保存JSON文件"""
        from cyrp.config.config_templates import (
            ConfigTemplateFactory, ConfigGenerator
        )

        template = ConfigTemplateFactory.create_development()
        generator = ConfigGenerator(template)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "config.json")
            generator.save_json(path)

            assert os.path.exists(path)
            with open(path, 'r') as f:
                data = json.load(f)
            assert data["environment"] == "development"

    def test_save_yaml(self):
        """测试保存YAML文件"""
        from cyrp.config.config_templates import (
            ConfigTemplateFactory, ConfigGenerator
        )

        template = ConfigTemplateFactory.create_testing()
        generator = ConfigGenerator(template)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "config.yaml")
            generator.save_yaml(path)

            assert os.path.exists(path)
            with open(path, 'r') as f:
                content = f.read()
            assert "environment:" in content

    def test_save_env(self):
        """测试保存环境变量文件"""
        from cyrp.config.config_templates import (
            ConfigTemplateFactory, ConfigGenerator
        )

        template = ConfigTemplateFactory.create_staging()
        generator = ConfigGenerator(template)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, ".env")
            generator.save_env(path)

            assert os.path.exists(path)
            with open(path, 'r') as f:
                content = f.read()
            assert "CYRP_" in content


class TestConfigValidator:
    """测试配置验证器"""

    def test_valid_config(self):
        """测试有效配置"""
        from cyrp.config.config_templates import (
            ConfigTemplateFactory, ConfigValidator
        )

        config = ConfigTemplateFactory.create_development()
        valid, errors = ConfigValidator.validate(config)

        assert valid is True
        assert len(errors) == 0

    def test_invalid_port(self):
        """测试无效端口"""
        from cyrp.config.config_templates import (
            CYRPConfigTemplate, ConfigValidator
        )

        config = CYRPConfigTemplate()
        config.api.port = 70000  # 无效端口

        valid, errors = ConfigValidator.validate(config)
        assert valid is False
        assert any("port" in e.lower() for e in errors)

    def test_same_ports(self):
        """测试相同端口"""
        from cyrp.config.config_templates import (
            CYRPConfigTemplate, ConfigValidator
        )

        config = CYRPConfigTemplate()
        config.api.port = 8080
        config.api.ws_port = 8080  # 相同端口

        valid, errors = ConfigValidator.validate(config)
        assert valid is False
        assert any("different" in e.lower() for e in errors)

    def test_invalid_sampling_rate(self):
        """测试无效采样率"""
        from cyrp.config.config_templates import (
            CYRPConfigTemplate, ConfigValidator
        )

        config = CYRPConfigTemplate()
        config.sensor.sampling_rate_hz = -1  # 无效采样率

        valid, errors = ConfigValidator.validate(config)
        assert valid is False

    def test_invalid_horizon(self):
        """测试无效预测范围"""
        from cyrp.config.config_templates import (
            CYRPConfigTemplate, ConfigValidator
        )

        config = CYRPConfigTemplate()
        config.predictor.default_horizon = 200
        config.predictor.max_horizon = 100  # default > max

        valid, errors = ConfigValidator.validate(config)
        assert valid is False

    def test_production_without_jwt(self):
        """测试生产环境缺少JWT"""
        from cyrp.config.config_templates import (
            CYRPConfigTemplate, ConfigValidator
        )

        config = CYRPConfigTemplate()
        config.environment = "production"
        config.api.jwt_secret = ""  # 生产环境缺少JWT

        valid, errors = ConfigValidator.validate(config)
        assert valid is False
        assert any("jwt" in e.lower() for e in errors)


class TestGetConfigForEnvironment:
    """测试环境配置获取"""

    def test_get_development(self):
        """测试获取开发环境配置"""
        from cyrp.config.config_templates import get_config_for_environment

        config = get_config_for_environment("development")
        assert config.environment == "development"

    def test_get_production(self):
        """测试获取生产环境配置"""
        from cyrp.config.config_templates import get_config_for_environment

        config = get_config_for_environment("production")
        assert config.environment == "production"

    def test_get_unknown_defaults_to_development(self):
        """测试未知环境默认返回开发配置"""
        from cyrp.config.config_templates import get_config_for_environment

        config = get_config_for_environment("unknown")
        assert config.environment == "development"

    def test_case_insensitive(self):
        """测试环境名称不区分大小写"""
        from cyrp.config.config_templates import get_config_for_environment

        config1 = get_config_for_environment("PRODUCTION")
        config2 = get_config_for_environment("Production")
        config3 = get_config_for_environment("production")

        assert config1.environment == config2.environment == config3.environment


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
