"""
Pytest Configuration and Shared Fixtures
pytest配置和共享测试夹具
"""

import pytest
import asyncio
import os
import sys
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session")
def event_loop():
    """创建事件循环"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_sensor_data():
    """示例传感器数据"""
    return {
        "inlet_flow": 265.0,
        "outlet_flow": 264.5,
        "inlet_pressure": 0.35,
        "outlet_pressure": 0.30,
        "water_level": 8.5,
        "temperature": 18.0,
        "vibration": 2.5,
        "turbidity": 5.0,
        "ph": 7.2,
    }


@pytest.fixture
def sample_alarm_definitions():
    """示例报警定义"""
    from cyrp.alarm import AlarmDefinition, AlarmSeverity

    return [
        AlarmDefinition(
            alarm_id="ALM001",
            name="高水位报警",
            description="水位超过高限",
            severity=AlarmSeverity.HIGH,
            tag_name="water_level",
            high_limit=10.0
        ),
        AlarmDefinition(
            alarm_id="ALM002",
            name="低压报警",
            description="压力低于低限",
            severity=AlarmSeverity.MEDIUM,
            tag_name="inlet_pressure",
            low_limit=0.1
        ),
        AlarmDefinition(
            alarm_id="ALM003",
            name="高温报警",
            description="温度超过高限",
            severity=AlarmSeverity.HIGH,
            tag_name="temperature",
            high_limit=35.0
        ),
    ]


@pytest.fixture
def sample_user():
    """示例用户"""
    from cyrp.security import User, PasswordHasher

    password_hash, salt = PasswordHasher.hash_password("Test@123")

    return User(
        user_id="test-user-001",
        username="testuser",
        display_name="Test User",
        email="test@example.com",
        password_hash=password_hash,
        salt=salt,
        roles=["operator"]
    )


@pytest.fixture
def sample_roles():
    """示例角色"""
    from cyrp.security import Role, Permission

    return [
        Role(
            role_id="viewer",
            name="查看员",
            description="只能查看",
            permissions=Permission.VIEWER
        ),
        Role(
            role_id="operator",
            name="操作员",
            description="可以操作",
            permissions=Permission.OPERATOR
        ),
        Role(
            role_id="admin",
            name="管理员",
            description="完全权限",
            permissions=Permission.ADMIN
        ),
    ]


@pytest.fixture
def sample_config():
    """示例配置"""
    return {
        "system.name": "测试系统",
        "database.host": "localhost",
        "database.port": 5432,
        "web.port": 8080,
        "logging.level": "DEBUG",
    }


@pytest.fixture
def temp_config_dir(tmp_path):
    """临时配置目录"""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    return str(config_dir)


@pytest.fixture
async def security_manager():
    """安全管理器夹具"""
    from cyrp.security import SecurityManager

    manager = SecurityManager()
    await manager.initialize_default_roles()
    await manager.initialize_default_admin("Admin@123")

    return manager


@pytest.fixture
async def alarm_manager(sample_alarm_definitions):
    """报警管理器夹具"""
    from cyrp.alarm import AlarmManager

    manager = AlarmManager()
    for alarm_def in sample_alarm_definitions:
        manager.register_alarm(alarm_def)

    return manager


@pytest.fixture
def mock_data_source():
    """模拟数据源"""
    import random

    class MockDataSource:
        def __init__(self):
            self.base_values = {
                "inlet_flow": 265.0,
                "outlet_flow": 264.5,
                "inlet_pressure": 0.35,
                "outlet_pressure": 0.30,
            }

        def read(self, tag_name: str) -> float:
            base = self.base_values.get(tag_name, 0.0)
            noise = random.gauss(0, base * 0.01)
            return base + noise

        def read_all(self):
            return {tag: self.read(tag) for tag in self.base_values}

    return MockDataSource()


# pytest配置
def pytest_configure(config):
    """pytest配置"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )


def pytest_collection_modifyitems(config, items):
    """修改测试收集"""
    # 如果没有指定-m参数,跳过慢速测试
    if config.getoption("-m"):
        return

    skip_slow = pytest.mark.skip(reason="use -m slow to run slow tests")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
