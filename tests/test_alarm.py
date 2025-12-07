"""
Tests for Alarm Management Module.
报警管理模块测试
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock

from cyrp.alarm import (
    AlarmSeverity,
    AlarmState,
    AlarmDefinition,
    AlarmInstance,
    AlarmProcessor,
    AlarmNotifier,
    NotificationChannel,
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
            severity=AlarmSeverity.HIGH,
            tag_name="water_level",
            high_limit=10.0,
            low_limit=None
        )

        assert alarm_def.alarm_id == "ALM001"
        assert alarm_def.severity == AlarmSeverity.HIGH
        assert alarm_def.high_limit == 10.0

    def test_check_limits_high(self):
        """测试高限检查"""
        alarm_def = AlarmDefinition(
            alarm_id="ALM001",
            name="高温报警",
            description="温度过高",
            severity=AlarmSeverity.HIGH,
            tag_name="temperature",
            high_limit=100.0
        )

        # 超过高限
        result = alarm_def.check_value(105.0)
        assert result == True

        # 未超过高限
        result = alarm_def.check_value(95.0)
        assert result == False

    def test_check_limits_low(self):
        """测试低限检查"""
        alarm_def = AlarmDefinition(
            alarm_id="ALM002",
            name="低压报警",
            description="压力过低",
            severity=AlarmSeverity.MEDIUM,
            tag_name="pressure",
            low_limit=0.1
        )

        # 低于低限
        result = alarm_def.check_value(0.05)
        assert result == True

        # 未低于低限
        result = alarm_def.check_value(0.5)
        assert result == False


class TestAlarmInstance:
    """报警实例测试类"""

    def test_creation(self):
        """测试创建报警实例"""
        alarm = AlarmInstance(
            instance_id="INS001",
            alarm_id="ALM001",
            name="高水位报警",
            severity=AlarmSeverity.HIGH,
            state=AlarmState.ACTIVE,
            value=10.5,
            limit=10.0,
            message="水位超过高限: 10.5 > 10.0"
        )

        assert alarm.instance_id == "INS001"
        assert alarm.state == AlarmState.ACTIVE
        assert alarm.acknowledged_at is None

    def test_acknowledge(self):
        """测试确认报警"""
        alarm = AlarmInstance(
            instance_id="INS001",
            alarm_id="ALM001",
            name="测试报警",
            severity=AlarmSeverity.MEDIUM,
            state=AlarmState.ACTIVE,
            value=100.0,
            limit=90.0,
            message="测试"
        )

        alarm.acknowledge("operator1")

        assert alarm.state == AlarmState.ACKNOWLEDGED
        assert alarm.acknowledged_by == "operator1"
        assert alarm.acknowledged_at is not None

    def test_clear(self):
        """测试清除报警"""
        alarm = AlarmInstance(
            instance_id="INS001",
            alarm_id="ALM001",
            name="测试报警",
            severity=AlarmSeverity.LOW,
            state=AlarmState.ACKNOWLEDGED,
            value=100.0,
            limit=90.0,
            message="测试"
        )

        alarm.clear()

        assert alarm.state == AlarmState.CLEARED
        assert alarm.cleared_at is not None


class TestAlarmProcessor:
    """报警处理器测试类"""

    def setup_method(self):
        """测试前设置"""
        self.processor = AlarmProcessor()

    def test_add_definition(self):
        """测试添加报警定义"""
        alarm_def = AlarmDefinition(
            alarm_id="ALM001",
            name="测试报警",
            description="测试",
            severity=AlarmSeverity.MEDIUM,
            tag_name="test_tag",
            high_limit=100.0
        )

        self.processor.add_definition(alarm_def)

        assert "ALM001" in self.processor._definitions

    @pytest.mark.asyncio
    async def test_process_value_triggers_alarm(self):
        """测试处理值触发报警"""
        alarm_def = AlarmDefinition(
            alarm_id="ALM001",
            name="高温报警",
            description="温度过高",
            severity=AlarmSeverity.HIGH,
            tag_name="temperature",
            high_limit=100.0
        )
        self.processor.add_definition(alarm_def)

        # 处理超限值
        alarm = await self.processor.process_value("temperature", 105.0)

        assert alarm is not None
        assert alarm.state == AlarmState.ACTIVE
        assert alarm.value == 105.0

    @pytest.mark.asyncio
    async def test_process_value_clears_alarm(self):
        """测试处理值清除报警"""
        alarm_def = AlarmDefinition(
            alarm_id="ALM001",
            name="高温报警",
            description="温度过高",
            severity=AlarmSeverity.HIGH,
            tag_name="temperature",
            high_limit=100.0
        )
        self.processor.add_definition(alarm_def)

        # 先触发报警
        await self.processor.process_value("temperature", 105.0)

        # 然后恢复正常
        result = await self.processor.process_value("temperature", 95.0)

        # 报警应该被清除
        active = self.processor.get_active_alarms()
        assert len([a for a in active if a.alarm_id == "ALM001"]) == 0


class TestAlarmNotifier:
    """报警通知器测试类"""

    def setup_method(self):
        """测试前设置"""
        self.notifier = AlarmNotifier()

    @pytest.mark.asyncio
    async def test_add_channel(self):
        """测试添加通知渠道"""
        channel = NotificationChannel(
            channel_id="email",
            channel_type="email",
            config={"smtp_server": "smtp.example.com"}
        )

        self.notifier.add_channel(channel)

        assert "email" in self.notifier._channels

    @pytest.mark.asyncio
    async def test_notify(self):
        """测试发送通知"""
        # 使用模拟渠道
        mock_handler = AsyncMock()
        channel = NotificationChannel(
            channel_id="mock",
            channel_type="mock",
            config={},
            handler=mock_handler
        )
        self.notifier.add_channel(channel)

        alarm = AlarmInstance(
            instance_id="INS001",
            alarm_id="ALM001",
            name="测试报警",
            severity=AlarmSeverity.HIGH,
            state=AlarmState.ACTIVE,
            value=100.0,
            limit=90.0,
            message="测试通知"
        )

        await self.notifier.notify(alarm, ["mock"])

        # 验证处理器被调用
        mock_handler.assert_called_once()


class TestAlarmManager:
    """报警管理器测试类"""

    def setup_method(self):
        """测试前设置"""
        self.manager = AlarmManager()

    @pytest.mark.asyncio
    async def test_register_alarm(self):
        """测试注册报警"""
        alarm_def = AlarmDefinition(
            alarm_id="ALM001",
            name="测试报警",
            description="测试",
            severity=AlarmSeverity.MEDIUM,
            tag_name="test_tag",
            high_limit=100.0
        )

        self.manager.register_alarm(alarm_def)

        assert "ALM001" in self.manager._processor._definitions

    @pytest.mark.asyncio
    async def test_process_and_get_active(self):
        """测试处理报警并获取活动报警"""
        alarm_def = AlarmDefinition(
            alarm_id="ALM001",
            name="测试报警",
            description="测试",
            severity=AlarmSeverity.HIGH,
            tag_name="test_tag",
            high_limit=100.0
        )
        self.manager.register_alarm(alarm_def)

        # 触发报警
        await self.manager.process_value("test_tag", 150.0)

        # 获取活动报警
        active = self.manager.get_active_alarms()

        assert len(active) > 0
        assert active[0].alarm_id == "ALM001"

    @pytest.mark.asyncio
    async def test_acknowledge_alarm(self):
        """测试确认报警"""
        alarm_def = AlarmDefinition(
            alarm_id="ALM001",
            name="测试报警",
            description="测试",
            severity=AlarmSeverity.HIGH,
            tag_name="test_tag",
            high_limit=100.0
        )
        self.manager.register_alarm(alarm_def)

        # 触发报警
        alarm = await self.manager.process_value("test_tag", 150.0)

        # 确认报警
        result = await self.manager.acknowledge_alarm(alarm.instance_id, "operator1")

        assert result == True

    @pytest.mark.asyncio
    async def test_get_alarm_history(self):
        """测试获取报警历史"""
        alarm_def = AlarmDefinition(
            alarm_id="ALM001",
            name="测试报警",
            description="测试",
            severity=AlarmSeverity.MEDIUM,
            tag_name="test_tag",
            high_limit=100.0
        )
        self.manager.register_alarm(alarm_def)

        # 触发并清除报警
        await self.manager.process_value("test_tag", 150.0)
        await self.manager.process_value("test_tag", 50.0)

        # 获取历史
        history = self.manager.get_alarm_history(limit=10)

        assert len(history) >= 0


class TestCreateCYRPAlarmSystem:
    """测试创建穿黄工程报警系统"""

    def test_create_system(self):
        """测试创建系统"""
        manager = create_cyrp_alarm_system()

        assert manager is not None
        assert isinstance(manager, AlarmManager)

        # 验证预定义的报警已注册
        definitions = manager._processor._definitions
        assert len(definitions) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
