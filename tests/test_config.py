"""
Tests for Configuration Management Module.
配置管理模块测试
"""

import pytest
import asyncio
import os
import json
import tempfile
from datetime import datetime

from cyrp.config import (
    ConfigFormat,
    ConfigScope,
    ConfigValueType,
    ConfigSchema,
    ConfigItem,
    ConfigEncryptor,
    ConfigValidator,
    JSONConfigSource,
    YAMLConfigSource,
    EnvironmentConfigSource,
    INIConfigSource,
    ConfigChangeEvent,
    ConfigManager,
    ConfigBuilder,
    create_cyrp_config_system,
    DEFAULT_SYSTEM_CONFIG,
)


class TestConfigEncryptor:
    """配置加密器测试类"""

    def setup_method(self):
        """测试前设置"""
        self.encryptor = ConfigEncryptor("test-secret-key")

    def test_encrypt_decrypt(self):
        """测试加密解密"""
        original = "sensitive-password"
        encrypted = self.encryptor.encrypt(original)
        decrypted = self.encryptor.decrypt(encrypted)

        assert encrypted != original
        assert decrypted == original

    def test_different_keys_produce_different_results(self):
        """测试不同密钥产生不同结果"""
        encryptor1 = ConfigEncryptor("key1")
        encryptor2 = ConfigEncryptor("key2")

        value = "test-value"
        encrypted1 = encryptor1.encrypt(value)
        encrypted2 = encryptor2.encrypt(value)

        assert encrypted1 != encrypted2


class TestConfigValidator:
    """配置验证器测试类"""

    def setup_method(self):
        """测试前设置"""
        self.schemas = {
            "port": ConfigSchema(
                key="port",
                value_type=ConfigValueType.INTEGER,
                min_value=1,
                max_value=65535
            ),
            "host": ConfigSchema(
                key="host",
                value_type=ConfigValueType.STRING,
                pattern=r"^[\w.-]+$"
            ),
            "log_level": ConfigSchema(
                key="log_level",
                value_type=ConfigValueType.STRING,
                allowed_values=["DEBUG", "INFO", "WARNING", "ERROR"]
            ),
        }
        self.validator = ConfigValidator(self.schemas)

    def test_validate_integer_in_range(self):
        """测试整数范围验证"""
        valid, errors = self.validator.validate("port", 8080)
        assert valid == True

        valid, errors = self.validator.validate("port", 70000)
        assert valid == False
        assert any("大于最大值" in e for e in errors)

    def test_validate_pattern(self):
        """测试正则验证"""
        valid, errors = self.validator.validate("host", "localhost")
        assert valid == True

        valid, errors = self.validator.validate("host", "invalid host!")
        assert valid == False

    def test_validate_allowed_values(self):
        """测试允许值验证"""
        valid, errors = self.validator.validate("log_level", "INFO")
        assert valid == True

        valid, errors = self.validator.validate("log_level", "TRACE")
        assert valid == False

    def test_validate_required(self):
        """测试必需验证"""
        schemas = {
            "required_key": ConfigSchema(
                key="required_key",
                value_type=ConfigValueType.STRING,
                required=True
            )
        }
        validator = ConfigValidator(schemas)

        valid, errors = validator.validate_required({})
        assert valid == False
        assert any("必需" in e for e in errors)

        valid, errors = validator.validate_required({"required_key": "value"})
        assert valid == True


class TestJSONConfigSource:
    """JSON配置源测试类"""

    def setup_method(self):
        """测试前设置"""
        self.temp_file = tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        )
        self.temp_file.close()
        self.source = JSONConfigSource(self.temp_file.name)

    def teardown_method(self):
        """测试后清理"""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)

    @pytest.mark.asyncio
    async def test_load_empty_file(self):
        """测试加载空文件"""
        os.unlink(self.temp_file.name)
        config = await self.source.load()
        assert config == {}

    @pytest.mark.asyncio
    async def test_save_and_load(self):
        """测试保存和加载"""
        config = {
            "host": "localhost",
            "port": 8080,
            "debug": True
        }

        await self.source.save(config)
        loaded = await self.source.load()

        assert loaded["host"] == "localhost"
        assert loaded["port"] == 8080
        assert loaded["debug"] == True

    def test_get_source_name(self):
        """测试获取源名称"""
        name = self.source.get_source_name()
        assert "json:" in name


class TestYAMLConfigSource:
    """YAML配置源测试类"""

    def setup_method(self):
        """测试前设置"""
        self.temp_file = tempfile.NamedTemporaryFile(
            mode='w', suffix='.yaml', delete=False
        )
        self.temp_file.close()
        self.source = YAMLConfigSource(self.temp_file.name)

    def teardown_method(self):
        """测试后清理"""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)

    @pytest.mark.asyncio
    async def test_save_and_load(self):
        """测试保存和加载"""
        config = {
            "server": "localhost",
            "port": 3000,
            "enabled": True
        }

        await self.source.save(config)
        loaded = await self.source.load()

        assert loaded["server"] == "localhost"
        assert loaded["port"] == 3000
        assert loaded["enabled"] == True


class TestEnvironmentConfigSource:
    """环境变量配置源测试类"""

    def setup_method(self):
        """测试前设置"""
        self.source = EnvironmentConfigSource("TEST_")
        # 设置测试环境变量
        os.environ["TEST_HOST"] = "localhost"
        os.environ["TEST_PORT"] = "8080"
        os.environ["TEST_DEBUG"] = "true"

    def teardown_method(self):
        """测试后清理"""
        for key in ["TEST_HOST", "TEST_PORT", "TEST_DEBUG"]:
            if key in os.environ:
                del os.environ[key]

    @pytest.mark.asyncio
    async def test_load(self):
        """测试加载环境变量"""
        config = await self.source.load()

        assert config["host"] == "localhost"
        assert config["port"] == 8080
        assert config["debug"] == True


class TestINIConfigSource:
    """INI配置源测试类"""

    def setup_method(self):
        """测试前设置"""
        self.temp_file = tempfile.NamedTemporaryFile(
            mode='w', suffix='.ini', delete=False
        )
        self.temp_file.close()
        self.source = INIConfigSource(self.temp_file.name)

    def teardown_method(self):
        """测试后清理"""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)

    @pytest.mark.asyncio
    async def test_save_and_load(self):
        """测试保存和加载"""
        config = {
            "global_key": "value",
            "database": {
                "host": "localhost",
                "port": 5432
            }
        }

        await self.source.save(config)
        loaded = await self.source.load()

        assert "global_key" in loaded or "database" in loaded


class TestConfigManager:
    """配置管理器测试类"""

    def setup_method(self):
        """测试前设置"""
        self.manager = ConfigManager()

    @pytest.mark.asyncio
    async def test_set_and_get(self):
        """测试设置和获取配置"""
        await self.manager.set("test_key", "test_value")

        value = self.manager.get("test_key")
        assert value == "test_value"

    @pytest.mark.asyncio
    async def test_get_with_default(self):
        """测试带默认值获取"""
        value = self.manager.get("nonexistent", "default")
        assert value == "default"

    @pytest.mark.asyncio
    async def test_get_typed_values(self):
        """测试类型化获取"""
        await self.manager.set("int_value", 42)
        await self.manager.set("float_value", 3.14)
        await self.manager.set("bool_value", True)
        await self.manager.set("list_value", ["a", "b", "c"])

        assert self.manager.get_int("int_value") == 42
        assert self.manager.get_float("float_value") == 3.14
        assert self.manager.get_bool("bool_value") == True
        assert self.manager.get_list("list_value") == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_encrypted_value(self):
        """测试加密值"""
        await self.manager.set("secret", "password123", encrypt=True)

        # 直接获取(解密)
        value = self.manager.get("secret")
        assert value == "password123"

    @pytest.mark.asyncio
    async def test_change_listener(self):
        """测试变更监听器"""
        changes = []

        def listener(event: ConfigChangeEvent):
            changes.append(event)

        self.manager.add_listener(listener)

        await self.manager.set("key1", "value1")
        await self.manager.set("key1", "value2")

        assert len(changes) == 2
        assert changes[1].old_value == "value1"
        assert changes[1].new_value == "value2"

    @pytest.mark.asyncio
    async def test_version_history(self):
        """测试版本历史"""
        await self.manager.set("versioned_key", "v1")
        await self.manager.set("versioned_key", "v2")
        await self.manager.set("versioned_key", "v3")

        history = self.manager.get_version_history("versioned_key")

        assert len(history) >= 2

    def test_export_config(self):
        """测试导出配置"""
        asyncio.run(self.manager.set("export_key", "export_value"))

        json_export = self.manager.export_config(ConfigFormat.JSON)
        assert "export_key" in json_export

        yaml_export = self.manager.export_config(ConfigFormat.YAML)
        assert "export_key" in yaml_export

        env_export = self.manager.export_config(ConfigFormat.ENV)
        assert "CYRP_EXPORT_KEY" in env_export


class TestConfigBuilder:
    """配置构建器测试类"""

    def test_build_with_sources(self):
        """测试使用源构建"""
        temp_json = tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        )
        temp_json.write('{"key1": "value1"}')
        temp_json.close()

        try:
            manager = (
                ConfigBuilder()
                .with_json_source(temp_json.name, ConfigScope.SYSTEM)
                .build()
            )

            assert manager is not None
        finally:
            os.unlink(temp_json.name)

    def test_build_with_encryption(self):
        """测试带加密构建"""
        manager = (
            ConfigBuilder()
            .with_encryption("secret-key")
            .build()
        )

        assert manager.encryptor is not None

    def test_build_with_validator(self):
        """测试带验证器构建"""
        schemas = {
            "port": ConfigSchema(
                key="port",
                value_type=ConfigValueType.INTEGER
            )
        }

        manager = (
            ConfigBuilder()
            .with_validator(schemas)
            .build()
        )

        assert manager.validator is not None


class TestDefaultSystemConfig:
    """默认系统配置测试类"""

    def test_default_config_exists(self):
        """测试默认配置存在"""
        assert DEFAULT_SYSTEM_CONFIG is not None
        assert len(DEFAULT_SYSTEM_CONFIG) > 0

    def test_default_config_has_required_keys(self):
        """测试默认配置有必需键"""
        required_keys = [
            "system.name",
            "database.host",
            "web.port",
            "logging.level"
        ]

        for key in required_keys:
            assert key in DEFAULT_SYSTEM_CONFIG, f"Missing key: {key}"


class TestCreateCYRPConfigSystem:
    """测试创建穿黄工程配置系统"""

    def test_create_system(self):
        """测试创建系统"""
        manager = create_cyrp_config_system()

        assert manager is not None
        assert isinstance(manager, ConfigManager)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
