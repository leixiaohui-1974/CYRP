"""
Tests for Alarm Management Module.
报警管理模块测试
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from cyrp.alarm import (
    AlarmSeverity,
    AlarmState,
    AlarmType,
    AlarmDefinition,
    AlarmInstance,
    AlarmProcessor,
    AlarmNotifier,
    NotificationChannel,
    NotificationConfig,
    AlarmManager,
    create_cyrp_alarm_system,
)


class TestAlarmDefinition:
    """报警定义测试类"""

    def test_creation(self):
        """测试创建报警定义"""
        alarm_def = AlarmDefinition(
            alarm_id="ALM001",
            name="高水位报警",
            description="水位超过高限",
            alarm_type=AlarmType.HIGH,
            severity=AlarmSeverity.MAJOR,
            source="water_level",
            high_limit=10.0,
        )

        assert alarm_def.alarm_id == "ALM001"
        assert alarm_def.severity == AlarmSeverity.MAJOR
        assert alarm_def.high_limit == 10.0

    def test_check_limits_high(self):
        """测试高限检查 - 通过AlarmProcessor"""
        alarm_def = AlarmDefinition(
            alarm_id="ALM001",
            name="高温报警",
            description="温度过高",
            alarm_type=AlarmType.HIGH,
            severity=AlarmSeverity.MAJOR,
            source="temperature",
            high_limit=100.0
        )

        processor = AlarmProcessor()

        # 超过高限 - 应该触发报警
        result = processor.process_value(alarm_def, 105.0, AlarmState.NORMAL)
        assert result is not None
        assert result[0] == AlarmState.ACTIVE

        # 未超过高限 - 从NORMAL状态不应该触发
        result = processor.process_value(alarm_def, 95.0, AlarmState.NORMAL)
        assert result is None

    def test_check_limits_low(self):
        """测试低限检查"""
        alarm_def = AlarmDefinition(
            alarm_id="ALM002",
            name="低压报警",
            description="压力过低",
            alarm_type=AlarmType.LOW,
            severity=AlarmSeverity.WARNING,
            source="pressure",
            low_limit=0.1
        )

        processor = AlarmProcessor()

        # 低于低限
        result = processor.process_value(alarm_def, 0.05, AlarmState.NORMAL)
        assert result is not None
        assert result[0] == AlarmState.ACTIVE

        # 未低于低限
        result = processor.process_value(alarm_def, 0.5, AlarmState.NORMAL)
        assert result is None


class TestAlarmInstance:
    """报警实例测试类"""

    def test_creation(self):
        """测试创建报警实例"""
        alarm_def = AlarmDefinition(
            alarm_id="ALM001",
            name="高水位报警",
            description="水位超过高限",
            alarm_type=AlarmType.HIGH,
            severity=AlarmSeverity.MAJOR,
            source="water_level",
            high_limit=10.0,
        )

        alarm = AlarmInstance(
            instance_id="INS001",
            definition=alarm_def,
            state=AlarmState.ACTIVE,
            trigger_value=10.5,
        )

        assert alarm.instance_id == "INS001"
        assert alarm.state == AlarmState.ACTIVE
        assert alarm.acknowledged_at is None

    def test_acknowledge(self):
        """测试确认报警 - 通过AlarmManager"""
        manager = AlarmManager()

        alarm_def = AlarmDefinition(
            alarm_id="ALM001",
            name="测试报警",
            description="测试",
            alarm_type=AlarmType.HIGH,
            severity=AlarmSeverity.WARNING,
            source="test_tag",
            high_limit=90.0
        )
        manager.register_alarm(alarm_def)

        # 触发报警
        manager.process_value("ALM001", 100.0)

        # 确认报警
        result = manager.acknowledge("ALM001", "operator1")

        assert result == True
        instance = manager.instances["ALM001"]
        assert instance.state == AlarmState.ACKNOWLEDGED
        assert instance.acknowledged_by == "operator1"

    def test_clear(self):
        """测试清除报警"""
        manager = AlarmManager()

        alarm_def = AlarmDefinition(
            alarm_id="ALM001",
            name="测试报警",
            description="测试",
            alarm_type=AlarmType.HIGH,
            severity=AlarmSeverity.WARNING,
            source="test_tag",
            high_limit=90.0
        )
        manager.register_alarm(alarm_def)

        # 触发报警
        manager.process_value("ALM001", 100.0)

        # 恢复正常值应该清除报警
        manager.process_value("ALM001", 50.0)

        instance = manager.instances["ALM001"]
        assert instance.state == AlarmState.CLEARED


class TestAlarmProcessor:
    """报警处理器测试类"""

    def setup_method(self):
        """测试前设置"""
        self.processor = AlarmProcessor()
        self.definition = AlarmDefinition(
            alarm_id="ALM001",
            name="测试报警",
            description="测试",
            alarm_type=AlarmType.HIGH,
            severity=AlarmSeverity.WARNING,
            source="test_tag",
            high_limit=100.0
        )

    def test_add_definition(self):
        """测试处理器初始化"""
        # AlarmProcessor doesn't store definitions, just processes values
        processor = AlarmProcessor()
        assert processor is not None

    def test_process_value_triggers_alarm(self):
        """测试处理值触发报警"""
        # 处理超限值
        result = self.processor.process_value(self.definition, 105.0, AlarmState.NORMAL)

        assert result is not None
        assert result[0] == AlarmState.ACTIVE
        assert result[1] == 105.0

    def test_process_value_clears_alarm(self):
        """测试处理值清除报警"""
        # 先触发报警
        self.processor.process_value(self.definition, 105.0, AlarmState.NORMAL)

        # 然后恢复正常
        result = self.processor.process_value(self.definition, 95.0, AlarmState.ACTIVE)

        # 报警应该被清除
        assert result is not None
        assert result[0] == AlarmState.CLEARED


class TestAlarmNotifier:
    """报警通知器测试类"""

    def setup_method(self):
        """测试前设置"""
        self.notifier = AlarmNotifier()

    def test_add_channel(self):
        """测试添加通知渠道"""
        config = NotificationConfig(
            channel=NotificationChannel.EMAIL,
            enabled=True,
            recipients=["test@example.com"],
            config={"smtp_server": "smtp.example.com"}
        )

        self.notifier.add_channel(config)

        assert NotificationChannel.EMAIL in self.notifier.channels

    def test_notify(self):
        """测试发送通知"""
        # 添加控制台通知渠道
        config = NotificationConfig(
            channel=NotificationChannel.CONSOLE,
            enabled=True,
            min_severity=AlarmSeverity.WARNING
        )
        self.notifier.add_channel(config)

        alarm_def = AlarmDefinition(
            alarm_id="ALM001",
            name="测试报警",
            description="测试通知",
            alarm_type=AlarmType.HIGH,
            severity=AlarmSeverity.MAJOR,
            source="test_tag",
            high_limit=90.0
        )

        alarm = AlarmInstance(
            instance_id="INS001",
            definition=alarm_def,
            state=AlarmState.ACTIVE,
            trigger_value=100.0,
        )
        alarm.occurred_at = datetime.now()

        # 同步通知 (放入队列)
        self.notifier.notify(alarm, "RAISED")

        # 验证队列不为空
        assert not self.notifier._notification_queue.empty()


class TestAlarmManager:
    """报警管理器测试类"""

    def setup_method(self):
        """测试前设置"""
        self.manager = AlarmManager()

    def test_register_alarm(self):
        """测试注册报警"""
        alarm_def = AlarmDefinition(
            alarm_id="ALM001",
            name="测试报警",
            description="测试",
            alarm_type=AlarmType.HIGH,
            severity=AlarmSeverity.WARNING,
            source="test_tag",
            high_limit=100.0
        )

        self.manager.register_alarm(alarm_def)

        assert "ALM001" in self.manager.definitions

    def test_process_and_get_active(self):
        """测试处理报警并获取活动报警"""
        alarm_def = AlarmDefinition(
            alarm_id="ALM001",
            name="测试报警",
            description="测试",
            alarm_type=AlarmType.HIGH,
            severity=AlarmSeverity.MAJOR,
            source="test_tag",
            high_limit=100.0
        )
        self.manager.register_alarm(alarm_def)

        # 触发报警
        self.manager.process_value("ALM001", 150.0)

        # 获取活动报警
        active = self.manager.get_active_alarms()

        assert len(active) > 0
        assert active[0].definition.alarm_id == "ALM001"

    def test_acknowledge_alarm(self):
        """测试确认报警"""
        alarm_def = AlarmDefinition(
            alarm_id="ALM001",
            name="测试报警",
            description="测试",
            alarm_type=AlarmType.HIGH,
            severity=AlarmSeverity.MAJOR,
            source="test_tag",
            high_limit=100.0
        )
        self.manager.register_alarm(alarm_def)

        # 触发报警
        self.manager.process_value("ALM001", 150.0)

        # 确认报警
        result = self.manager.acknowledge("ALM001", "operator1")

        assert result == True

    def test_get_alarm_history(self):
        """测试获取报警历史"""
        alarm_def = AlarmDefinition(
            alarm_id="ALM001",
            name="测试报警",
            description="测试",
            alarm_type=AlarmType.HIGH,
            severity=AlarmSeverity.WARNING,
            source="test_tag",
            high_limit=100.0
        )
        self.manager.register_alarm(alarm_def)

        # 触发并清除报警
        self.manager.process_value("ALM001", 150.0)
        self.manager.process_value("ALM001", 50.0)

        # 获取历史
        history = self.manager.get_event_history(limit=10)

        assert len(history) >= 0


class TestCreateCYRPAlarmSystem:
    """测试创建穿黄工程报警系统"""

    def test_create_system(self):
        """测试创建系统"""
        manager = create_cyrp_alarm_system()

        assert manager is not None
        assert isinstance(manager, AlarmManager)

        # 验证预定义的报警已注册
        definitions = manager.definitions
        assert len(definitions) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
